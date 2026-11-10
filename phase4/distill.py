"""
CS781 Capstone - Phase 4: Court Judgment Prediction (CJPE)
Knowledge distillation

Trains a smaller 6-layer decoder (the student) to learn from both the gold
labels and the predictions of the trained decoder_full model (the teacher).

Usage (Colab):
    python distill.py \
        --teacher_ckpt /content/drive/MyDrive/CS781/ckpt_p4/decoder_full/best_checkpoint.pt \
        --p1_ckpt /content/drive/MyDrive/CS781/ckpt_p1/decoder/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p4 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p4/decoder_distill

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
from model import ModelConfig, build_cjpe_model
from train import load_json, macro_f1, make_batch, predict_cases


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--p1_ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--student_layers", type=int, default=6)
    # kd_alpha is the share of the loss from the gold labels, the rest is from the teacher
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=2.0)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    # Same as train.py, one update sees batch_size * accum_steps = 16 cases
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_every", type=int, default=125)
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

    train_cases = load_json(os.path.join(args.data_dir, "train_cases.json"))
    dev_cases = load_json(os.path.join(args.data_dir, "dev_cases.json"))

    # Loading the trained teacher and freezing it
    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher = build_cjpe_model(ModelConfig(**t_ckpt["config"]))
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

    student = build_cjpe_model(config)
    missing, unexpected = student.backbone.load_state_dict(p1_ckpt["model"], strict=False)
    print(f"Student backbone loaded, missing keys: {len(missing)}, unused keys: {len(unexpected)}")
    student = student.to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher: {teacher_params / 1e6:.1f}M parameters, student: {student_params / 1e6:.1f}M "
          f"parameters ({100 * student_params / teacher_params:.0f}% of the teacher)")

    steps_per_epoch = math.ceil(len(train_cases) / (args.batch_size * args.accum_steps))
    total_steps = steps_per_epoch * args.epochs
    print(f"{args.batch_size * args.accum_steps} cases per update, "
          f"{steps_per_epoch} steps per epoch, {total_steps} steps total")

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

    # Same distillation loss as in phase3/distill.py
    def distill_loss(student_logits, teacher_logits, gold_loss):
        T = args.temperature
        student_log_probs = F.log_softmax(student_logits.float() / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits.float() / T, dim=-1)

        # Multiplying by T*T so this part is the same size as the gold loss
        teacher_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * T * T
        return args.kd_alpha * gold_loss + (1 - args.kd_alpha) * teacher_loss

    def save_checkpoint(step, best_f1, path):
        # Saving the student like a full fine-tuned model, so predict.py can load it
        saved_args = vars(args).copy()
        saved_args["method"] = "full"
        saved_args["rank"] = t_ckpt["args"]["rank"]
        saved_args["alpha"] = t_ckpt["args"]["alpha"]
        saved_args["train_top_layers"] = t_ckpt["args"]["train_top_layers"]
        torch.save({
            "model": student.state_dict(),
            "config": config.__dict__,
            "step": step,
            "best_dev_f1": best_f1,
            "args": saved_args,
        }, path)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "train_loss", "dev_f1"])

    # Training loop, same as train.py but with the teacher's predictions added to the loss
    train_rng = random.Random(args.seed)
    student.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    best_f1 = -1.0
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        order = list(range(len(train_cases)))
        train_rng.shuffle(order)

        batch_starts = list(range(0, len(order), args.batch_size))
        for batch_num, start in enumerate(batch_starts):
            batch_cases = [train_cases[i] for i in order[start:start + args.batch_size]]
            token_ids, token_mask, labels = make_batch(batch_cases, config.pad_id)
            token_ids = token_ids.to(device)
            token_mask = token_mask.to(device)
            labels = labels.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                with torch.no_grad():
                    teacher_logits, _ = teacher(token_ids, token_mask)
                student_logits, gold_loss = student(token_ids, token_mask, labels)
            loss = distill_loss(student_logits, teacher_logits, gold_loss)

            scaler.scale(loss / args.accum_steps).backward()
            running_loss += loss.item()
            running_count += 1

            # Updating the weights only after every accum_steps batches or at the end of the epoch
            if (batch_num + 1) % args.accum_steps != 0 and batch_num + 1 != len(batch_starts):
                continue

            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_factor(step)

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
                      f"elapsed {elapsed_min:.1f} min")
                running_loss, running_count = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                predictions = predict_cases(student, dev_cases, args.batch_size, config.pad_id, device)
                dev_f1 = macro_f1(dev_cases, predictions)
                print(f"  dev macro-F1: {dev_f1:.4f}  (predicted label 1 for "
                      f"{sum(predictions)} of {len(predictions)} dev cases)")

                log_writer.writerow([step, epoch + 1, f"{avg_loss:.4f}", f"{dev_f1:.4f}"])
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
