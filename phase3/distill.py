"""
CS781 Capstone - Phase 3: Rhetorical Role Prediction (RR)
Knowledge distillation

Trains a smaller 6-layer decoder (the student) to learn from both the gold
labels and the predictions of the trained decoder_full model (the teacher).

Usage (Colab):
    python distill.py \
        --teacher_ckpt /content/drive/MyDrive/CS781/ckpt_p3/decoder_full/best_checkpoint.pt \
        --p1_ckpt /content/drive/MyDrive/CS781/ckpt_p1/decoder/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p3 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p3/decoder_distill

References:
    - Knowledge distillation: Hinton, Vinyals and Dean, "Distilling the
      Knowledge in a Neural Network", 2015
"""

import argparse
import csv
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from methods import setup_method
from model import ModelConfig, build_rr_model
from train import load_json, macro_f1, make_batch, make_class_weights, predict_documents


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--p1_ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--student_layers", type=int, default=6)
    # alpha is the share of the loss from the gold labels, the rest is from the teacher
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    meta = load_json(os.path.join(args.data_dir, "meta.json"))
    id_to_label = meta["id_to_label"]
    label_to_id = {label: i for i, label in enumerate(id_to_label)}
    num_labels = len(id_to_label)

    train_windows = load_json(os.path.join(args.data_dir, "train_windows.json"))
    dev_windows = load_json(os.path.join(args.data_dir, "dev_windows.json"))
    dev_docs = load_json(os.path.join(args.data_dir, "dev_docs.json"))
    dev_num_sentences = {doc["id"]: len(doc["labels"]) for doc in dev_docs}

    # Loading the trained teacher and freezing it
    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher = build_rr_model(ModelConfig(**t_ckpt["config"]), num_labels)
    setup_method(teacher, t_ckpt["args"]["method"])
    teacher.load_state_dict(t_ckpt["model"])
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Making the student from the Phase-1 decoder, but with only the first 6 layers
    p1_ckpt = torch.load(args.p1_ckpt, map_location="cpu", weights_only=False)
    student_settings = dict(p1_ckpt["config"])
    student_settings["n_layer"] = args.student_layers
    config = ModelConfig(**student_settings)

    class_weights = make_class_weights(meta["train_label_counts"])
    student = build_rr_model(config, num_labels, class_weights=class_weights)
    missing, unexpected = student.backbone.load_state_dict(p1_ckpt["model"], strict=False)
    print(f"Student backbone loaded, missing keys: {len(missing)}, unused keys: {len(unexpected)}")
    student = student.to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher: {teacher_params / 1e6:.1f}M parameters, student: {student_params / 1e6:.1f}M "
          f"parameters ({100 * student_params / teacher_params:.0f}% of the teacher)")

    steps_per_epoch = math.ceil(len(train_windows) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch} steps per epoch, {total_steps} steps total")

    # Same optimizer setup as train.py
    backbone_params = [p for n, p in student.named_parameters() if n.startswith("backbone.")]
    head_params = [p for n, p in student.named_parameters() if not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.learning_rate, "base_lr": args.learning_rate},
         {"params": head_params, "lr": args.head_learning_rate, "base_lr": args.head_learning_rate}],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def lr_factor(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine

    # Same distillation loss as in phase2/distill.py
    def distill_loss(student_logits, teacher_logits, label_ids, gold_loss):
        # Comparing the student and teacher only on real sentences, the empty slots with label -100 are skipped
        mask = label_ids != -100
        T = args.temperature
        student_log_probs = F.log_softmax(student_logits[mask].float() / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits[mask].float() / T, dim=-1)

        # Multiplying by T*T so this part is the same size as the gold loss
        teacher_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * T * T
        return args.alpha * gold_loss + (1 - args.alpha) * teacher_loss

    def evaluate():
        predictions = predict_documents(student, dev_windows, dev_num_sentences, num_labels,
                                        args.batch_size, config.pad_id, device)
        return (macro_f1(dev_docs, predictions, label_to_id),
                macro_f1(dev_docs, predictions, label_to_id, "CL"),
                macro_f1(dev_docs, predictions, label_to_id, "IT"))

    def save_checkpoint(step, best_f1, path):
        # Saving the student like a full fine-tuned model, so predict.py can load it
        saved_args = vars(args).copy()
        saved_args["method"] = "full"
        saved_args["lora_rank"] = t_ckpt["args"]["lora_rank"]
        saved_args["lora_alpha"] = t_ckpt["args"]["lora_alpha"]
        torch.save({
            "model": student.state_dict(),
            "config": config.__dict__,
            "id_to_label": id_to_label,
            "step": step,
            "best_dev_f1": best_f1,
            "args": saved_args,
        }, path)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "train_loss", "dev_f1", "dev_f1_cl", "dev_f1_it"])

    # Training loop
    train_rng = random.Random(args.seed)
    student.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    best_f1 = -1.0
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        order = list(range(len(train_windows)))
        train_rng.shuffle(order)

        for start in range(0, len(order), args.batch_size):
            batch_windows = [train_windows[i] for i in order[start:start + args.batch_size]]
            token_ids, sentence_mask, label_ids = make_batch(batch_windows, config.pad_id)
            token_ids = token_ids.to(device)
            sentence_mask = sentence_mask.to(device)
            label_ids = label_ids.to(device)

            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_factor(step)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():
                    teacher_logits, _ = teacher(token_ids, sentence_mask)
                student_logits, gold_loss = student(token_ids, sentence_mask, label_ids)
            loss = distill_loss(student_logits, teacher_logits, label_ids, gold_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.item()
            running_count += 1
            step += 1

            if step % args.log_every == 0:
                avg_loss = running_loss / running_count
                elapsed_min = (time.time() - start_time) / 60
                print(f"step {step}/{total_steps}  epoch {epoch + 1}  loss {avg_loss:.4f}  "
                      f"elapsed {elapsed_min:.1f} min")
                running_loss, running_count = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                dev_f1, dev_f1_cl, dev_f1_it = evaluate()
                print(f"  dev macro-F1: {dev_f1:.4f}  (CL {dev_f1_cl:.4f}, IT {dev_f1_it:.4f})")

                log_writer.writerow([step, epoch + 1, f"{avg_loss:.4f}",
                                     f"{dev_f1:.4f}", f"{dev_f1_cl:.4f}", f"{dev_f1_it:.4f}"])
                log_file.flush()

                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    save_checkpoint(step, best_f1, os.path.join(args.out_dir, "best_checkpoint.pt"))

    log_file.close()

    total_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished: {total_steps} steps, "
          f"best dev macro-F1 {best_f1:.4f}, {total_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
