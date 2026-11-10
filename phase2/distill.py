"""
CS781 Capstone - Phase 2: Legal Named Entity Recognition (L-NER)
Knowledge distillation

Teacher: an already fine-tuned NER model from train.py (decoder_full, 12 layers).
Student: a smaller decoder made from only the first few layers of the Phase-1
decoder checkpoint (6 of 12 by default), with a new NER head on top.

The student is trained on two things at once: the gold BIO labels (normal
cross entropy) and the teacher's soft predictions (KL divergence with
temperature), mixed with --alpha.

Usage (Colab):
    python distill.py \
        --teacher_ckpt /content/drive/MyDrive/CS781/ckpt_p2/decoder_full/best_checkpoint.pt \
        --p1_ckpt /content/drive/MyDrive/CS781/ckpt_p1/decoder/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p2 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p2/decoder_distill

References:
    - Knowledge distillation: Hinton, Vinyals and Dean, "Distilling the
      Knowledge in a Neural Network", 2015
"""

import argparse
import csv
import json
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from methods import setup_method
from model import ModelConfig, build_ner_model
from train import load_examples, make_batches


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--p1_ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--student_layers", type=int, default=6)

    # alpha is how much of the loss comes from the gold labels, the rest
    # comes from copying the teacher. Temperature softens both sides so the
    # student also learns which wrong labels the teacher thought were close
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    label_vocab = meta["label_vocab"]
    num_labels = len(label_vocab)

    # Loading the teacher, already fine-tuned for NER, and freezing it
    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher = build_ner_model(ModelConfig(**t_ckpt["config"]), num_labels)
    setup_method(teacher, t_ckpt["args"].get("method", "full"))
    teacher.load_state_dict(t_ckpt["model"])
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Building the student: same settings as the Phase-1 decoder but fewer layers
    p1_ckpt = torch.load(args.p1_ckpt, map_location="cpu", weights_only=False)
    student_settings = dict(p1_ckpt["config"])
    student_settings["n_layer"] = args.student_layers
    config = ModelConfig(**student_settings)
    student = build_ner_model(config, num_labels)

    # Copying the Phase-1 weights in. The student only has blocks 0 to
    # student_layers-1, so the weights of the higher blocks are just not used
    missing, unexpected = student.backbone.load_state_dict(p1_ckpt["model"], strict=False)
    print(f"Student backbone loaded, missing keys: {len(missing)}, unused keys: {len(unexpected)}")
    student = student.to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher: {teacher_params / 1e6:.1f}M parameters, "
          f"student: {student_params / 1e6:.1f}M parameters "
          f"({100 * student_params / teacher_params:.0f}% of the teacher)")

    train_examples = load_examples(os.path.join(args.data_dir, "fold1_examples.json"))
    dev_examples = load_examples(os.path.join(args.data_dir, "fold2_examples.json"))
    pad_token_id = config.pad_id
    dev_batches = make_batches(dev_examples, args.batch_size, pad_token_id)

    steps_per_epoch = math.ceil(len(train_examples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch} steps per epoch, {total_steps} steps total")

    decay_params = [p for p in student.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in student.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.learning_rate,
        betas=(0.9, 0.95)
    )

    def learning_rate_at(step):
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / args.warmup_steps
        progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.learning_rate * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine)

    def distill_loss(student_logits, teacher_logits, label_ids, gold_loss):
        # Only comparing real tokens, not padding
        mask = label_ids != -100
        T = args.temperature

        student_log_probs = F.log_softmax(student_logits[mask] / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits[mask] / T, dim=-1)

        # Multiplying by T*T keeps this part on the same scale as the gold
        # loss (from the Hinton et al. paper)
        teacher_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * T * T
        return args.alpha * gold_loss + (1 - args.alpha) * teacher_loss

    @torch.no_grad()
    def evaluate():
        # Plain gold-label loss on fold2, so it can be compared with train.py runs
        student.eval()
        losses = []
        for token_ids, label_ids in dev_batches:
            _, loss = student(token_ids.to(device), label_ids.to(device))
            losses.append(loss.item())
        student.train()
        return sum(losses) / len(losses)

    def save_checkpoint(step, best_val_loss, path):
        # predict.py reads arch and method from here when loading the student
        saved_args = vars(args).copy()
        saved_args["arch"] = "decoder"
        saved_args["method"] = "distill"
        torch.save({
            "model": student.state_dict(),
            "config": config.__dict__,
            "label_vocab": label_vocab,
            "step": step,
            "best_val_loss": best_val_loss,
            "args": saved_args,
        }, path)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "learning_rate", "train_loss", "val_loss"])

    # Training loop
    train_rng = random.Random(args.seed)
    student.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    best_val_loss = float("inf")
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        train_batches = make_batches(train_examples, args.batch_size, pad_token_id, train_rng)

        for token_ids, label_ids in train_batches:
            token_ids, label_ids = token_ids.to(device), label_ids.to(device)

            lr = learning_rate_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.no_grad():
                teacher_logits, _ = teacher(token_ids)

            student_logits, gold_loss = student(token_ids, label_ids)
            loss = distill_loss(student_logits, teacher_logits, label_ids, gold_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            running_count += 1
            step += 1

            if step % args.log_every == 0:
                avg_loss = running_loss / running_count
                elapsed_min = (time.time() - start_time) / 60
                print(f"step {step}/{total_steps}  epoch {epoch + 1}  loss {avg_loss:.4f}  "
                      f"lr {lr:.2e}  elapsed {elapsed_min:.1f} min")
                running_loss, running_count = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                val_loss = evaluate()
                print(f"  validation loss: {val_loss:.4f}")

                log_writer.writerow([step, epoch + 1, f"{lr:.3e}", f"{avg_loss:.4f}", f"{val_loss:.4f}"])
                log_file.flush()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        step,
                        best_val_loss,
                        os.path.join(args.out_dir, "best_checkpoint.pt")
                    )

    log_file.close()

    total_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished: {total_steps} steps, "
          f"best validation loss {best_val_loss:.4f}, {total_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
