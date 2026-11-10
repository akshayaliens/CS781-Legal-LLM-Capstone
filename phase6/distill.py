"""
CS781 Capstone - Phase 6: Abstractive Summarization
Knowledge distillation

Trains a smaller encdec student (fewer encoder and decoder layers) to learn
from both the real summaries and the trained decoder_lora model (the
teacher). Both use the same tokenizer, so for every summary token the
student can match the teacher's probabilities over the whole vocabulary.

Usage (Colab):
    python distill.py \
        --teacher_ckpt /content/drive/MyDrive/CS781/ckpt_p6/decoder_lora/best_checkpoint.pt \
        --p1_ckpt /content/drive/MyDrive/CS781/ckpt_p1/encdec/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p6 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p6/encdec_distill

References:
    - Knowledge distillation: Hinton, Vinyals and Dean, "Distilling the
      Knowledge in a Neural Network", 2015
    - Word-level distillation for sequence models: Kim and Rush,
      "Sequence-Level Knowledge Distillation", EMNLP 2016
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
from model import ModelConfig, build_model
from train import load_json, load_phase1_config, make_decoder_batch, make_encdec_batch, val_loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--p1_ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    # The student keeps the first 3 of the 6 encoder layers and the first 3 of the 6 decoder layers
    parser.add_argument("--student_layers", type=int, default=3)
    # kd_alpha is the share of the loss from the real summary, the rest is from the teacher
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--max_len", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--val_eval_docs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=100)
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

    train_docs = load_json(os.path.join(args.data_dir, "train_docs.json"))
    val_docs = load_json(os.path.join(args.data_dir, "val_docs.json"))
    val_eval = val_docs[:args.val_eval_docs]

    # Loading the trained teacher and freezing it
    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    t_config = ModelConfig(**t_ckpt["config"])
    t_args = t_ckpt["args"]
    teacher = build_model(t_config)
    setup_method(teacher, t_args["method"], t_args["rank"], t_args["alpha"], t_args["adapter_size"])
    teacher.load_state_dict(t_ckpt["model"])
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Making the smaller student from the Phase-1 encdec checkpoint
    p1_ckpt = torch.load(args.p1_ckpt, map_location="cpu", weights_only=False)
    config = load_phase1_config(p1_ckpt["config"], args.max_len)
    config.n_enc_layer = args.student_layers
    config.n_dec_layer = args.student_layers

    student = build_model(config)
    # Only the first layers are in the student, the weights of the other layers are not used
    missing, unexpected = student.load_state_dict(p1_ckpt["model"], strict=False)
    print(f"Student ({config.arch}) loaded, missing keys: {len(missing)}, unused keys: {len(unexpected)}")
    setup_method(student, "full")
    student = student.to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher: {teacher_params / 1e6:.1f}M parameters, student: {student_params / 1e6:.1f}M "
          f"parameters ({100 * student_params / teacher_params:.0f}% of the teacher)")

    steps_per_epoch = math.ceil(len(train_docs) / (args.batch_size * args.accum_steps))
    total_steps = steps_per_epoch * args.epochs
    print(f"{args.batch_size * args.accum_steps} documents per update, "
          f"{steps_per_epoch} steps per epoch, {total_steps} steps total")

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
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

    def teacher_summary_logits(batch_docs, student_logits):
        # The teacher reads "document <s> summary", the student only writes "summary"
        # Taking the teacher logits at the summary positions so they line up with the student
        input_ids, teacher_targets = make_decoder_batch(batch_docs, t_config)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                all_logits, _ = teacher(input_ids.to(device), teacher_targets.to(device))

        lined_up = torch.zeros_like(student_logits)
        for i, doc in enumerate(batch_docs):
            doc_len = len(doc["doc_ids"])
            num = len(doc["summary_ids"]) + 1
            lined_up[i, :num] = all_logits[i, doc_len:doc_len + num]
        return lined_up

    def distill_loss(student_logits, teacher_logits, target_ids, gold_loss):
        # For every summary word, the student learns to give the same probabilities as the teacher
        # Padding positions are left out
        T = args.temperature
        mask = target_ids != -100
        student_log_probs = F.log_softmax(student_logits[mask].float() / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits[mask].float() / T, dim=-1)

        # Multiplying by T*T so this part is the same size as the gold loss
        teacher_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * T * T
        return args.kd_alpha * gold_loss + (1 - args.kd_alpha) * teacher_loss

    def save_checkpoint(step, best_loss, path):
        # Saving the student like a full fine-tuned model, so predict.py can load it
        saved_args = vars(args).copy()
        saved_args["method"] = "full"
        saved_args["rank"] = t_args["rank"]
        saved_args["alpha"] = t_args["alpha"]
        saved_args["adapter_size"] = t_args["adapter_size"]
        torch.save({
            "model": student.state_dict(),
            "config": config.__dict__,
            "step": step,
            "best_val_loss": best_loss,
            "args": saved_args,
        }, path)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "train_loss", "val_loss"])

    # Training loop, same as train.py but with the teacher's probabilities added to the loss
    train_rng = random.Random(args.seed)
    student.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        order = list(range(len(train_docs)))
        train_rng.shuffle(order)

        batch_starts = list(range(0, len(order), args.batch_size))
        for batch_num, start in enumerate(batch_starts):
            batch_docs = [train_docs[i] for i in order[start:start + args.batch_size]]
            src_ids, tgt_in, target_ids = make_encdec_batch(batch_docs, config)
            target_ids = target_ids.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                student_logits, gold_loss = student(src_ids.to(device), tgt_in.to(device), target_ids)
            teacher_logits = teacher_summary_logits(batch_docs, student_logits)
            loss = distill_loss(student_logits, teacher_logits, target_ids, gold_loss)

            scaler.scale(loss / args.accum_steps).backward()
            running_loss += loss.item()
            running_count += 1

            if (batch_num + 1) % args.accum_steps != 0 and batch_num + 1 != len(batch_starts):
                continue

            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_factor(step)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_every == 0:
                avg_loss = running_loss / running_count
                elapsed_min = (time.time() - start_time) / 60
                print(f"step {step}/{total_steps}  epoch {epoch + 1}  loss {avg_loss:.4f}  "
                      f"elapsed {elapsed_min:.1f} min", flush=True)
                running_loss, running_count = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                # Validation loss on the real summaries only, so it can be compared with the other runs
                loss_now = val_loss(student, val_eval, args.batch_size, config, device)
                print(f"  validation loss: {loss_now:.4f}", flush=True)

                log_writer.writerow([step, epoch + 1, f"{avg_loss:.4f}", f"{loss_now:.4f}"])
                log_file.flush()

                if loss_now < best_loss:
                    best_loss = loss_now
                    save_checkpoint(step, best_loss, os.path.join(args.out_dir, "best_checkpoint.pt"))

    log_file.close()

    total_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished: {total_steps} steps, "
          f"best validation loss {best_loss:.4f}, {total_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
