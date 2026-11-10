"""
CS781 Capstone - Phase 4: Court Judgment Prediction (CJPE)
Training script

Loads a Phase-1 checkpoint, puts the case head on top and trains it on the
train cases with one of the methods in methods.py. The checkpoint with the
best dev macro-F1 is saved. Dev is only used for picking the checkpoint.

Usage (Colab):
    python train.py --arch decoder --method full \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p1/decoder/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p4 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p4/decoder_full

References:
    - AdamW: Loshchilov and Hutter, "Decoupled Weight Decay Regularization", 2019
    - Cosine LR schedule with warmup: Loshchilov and Hutter, "SGDR", 2017
    - Macro-F1: scikit-learn f1_score
"""

import argparse
import csv
import json
import math
import os
import random
import time

import torch
from sklearn.metrics import f1_score

from methods import count_trainable, setup_method
from model import ModelConfig, build_cjpe_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", required=True, choices=["decoder", "encdec", "moe", "convmix"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--method", default="full", choices=["full", "lorma", "bitfit", "partial"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--train_top_layers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=2)
    # Number of cases in a batch, each case has up to 4 chunks
    parser.add_argument("--batch_size", type=int, default=2)
    # Adding up the gradients of 8 batches before each update, so one update sees 16 cases
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    # Case head is new, so it gets a bigger learning rate
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_every", type=int, default=125)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_batch(cases, pad_id):
    # Padding the cases so they all have the same number of chunks and tokens
    max_chunks = max(len(case["chunks"]) for case in cases)
    max_tokens = max(len(chunk) for case in cases for chunk in case["chunks"])

    token_ids = torch.full((len(cases), max_chunks, max_tokens), pad_id, dtype=torch.long)
    token_mask = torch.zeros(len(cases), max_chunks, max_tokens)

    for i, case in enumerate(cases):
        for j, chunk in enumerate(case["chunks"]):
            token_ids[i, j, :len(chunk)] = torch.tensor(chunk)
            token_mask[i, j, :len(chunk)] = 1

    # Test cases have no label
    labels = None
    if "label" in cases[0]:
        labels = torch.tensor([case["label"] for case in cases])

    return token_ids, token_mask, labels


@torch.no_grad()
def predict_cases(model, cases, batch_size, pad_id, device):
    # Predicting label 0 or 1 for every case
    model.eval()
    predictions = []

    for start in range(0, len(cases), batch_size):
        batch_cases = cases[start:start + batch_size]
        token_ids, token_mask, _ = make_batch(batch_cases, pad_id)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits, _ = model(token_ids.to(device), token_mask.to(device))
        predictions.extend(logits.argmax(dim=-1).tolist())

    model.train()
    return predictions


def macro_f1(cases, predictions):
    gold = [case["label"] for case in cases]
    return f1_score(gold, predictions, average="macro")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Loading what data_prep.py saved
    train_cases = load_json(os.path.join(args.data_dir, "train_cases.json"))
    dev_cases = load_json(os.path.join(args.data_dir, "dev_cases.json"))
    print(f"{len(train_cases)} train cases, {len(dev_cases)} dev cases")

    # One step is one update of the weights, made from batch_size * accum_steps cases
    steps_per_epoch = math.ceil(len(train_cases) / (args.batch_size * args.accum_steps))
    total_steps = steps_per_epoch * args.epochs
    print(f"{args.batch_size * args.accum_steps} cases per update, "
          f"{steps_per_epoch} steps per epoch, {total_steps} steps total")

    # Building the model with the same settings the Phase-1 checkpoint used
    print(f"Loading Phase-1 checkpoint: {args.ckpt}")
    p1_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig(**p1_ckpt["config"])
    model = build_cjpe_model(config)

    missing, unexpected = model.backbone.load_state_dict(p1_ckpt["model"], strict=False)
    print(f"Backbone loaded, missing keys: {len(missing)}, unused keys: {len(unexpected)}")

    # Adding the method after loading the Phase-1 weights
    setup_method(model, args.method, args.rank, args.alpha, args.train_top_layers)
    model = model.to(device)

    trainable, total = count_trainable(model)
    print(f"Method: {args.method}, training {trainable / 1e6:.2f}M of {total / 1e6:.1f}M "
          f"parameters ({100 * trainable / total:.2f}%)")

    # Separate learning rates for the backbone and the case head
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.learning_rate, "base_lr": args.learning_rate},
         {"params": head_params, "lr": args.head_learning_rate, "base_lr": args.head_learning_rate}],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Mixed precision, same as in phase3/train.py
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def lr_factor(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine

    def save_checkpoint(step, best_f1, path):
        torch.save({
            "model": model.state_dict(),
            "config": config.__dict__,
            "step": step,
            "best_dev_f1": best_f1,
            "args": vars(args),
        }, path)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "train_loss", "dev_f1"])

    # Training loop
    train_rng = random.Random(args.seed)
    model.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    best_f1 = -1.0
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        # Shuffling the train cases again at the start of every epoch
        order = list(range(len(train_cases)))
        train_rng.shuffle(order)

        batch_starts = list(range(0, len(order), args.batch_size))
        for batch_num, start in enumerate(batch_starts):
            batch_cases = [train_cases[i] for i in order[start:start + args.batch_size]]
            token_ids, token_mask, labels = make_batch(batch_cases, config.pad_id)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(token_ids.to(device), token_mask.to(device), labels.to(device))

            # Dividing by accum_steps so the added up gradients are an average
            scaler.scale(loss / args.accum_steps).backward()
            running_loss += loss.item()
            running_count += 1

            # Updating the weights only after every accum_steps batches, or at the end of the epoch
            if (batch_num + 1) % args.accum_steps != 0 and batch_num + 1 != len(batch_starts):
                continue

            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_factor(step)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
                predictions = predict_cases(model, dev_cases, args.batch_size, config.pad_id, device)
                dev_f1 = macro_f1(dev_cases, predictions)
                print(f"  dev macro-F1: {dev_f1:.4f}  (predicted label 1 for "
                      f"{sum(predictions)} of {len(predictions)} dev cases)")

                log_writer.writerow([step, epoch + 1, f"{avg_loss:.4f}", f"{dev_f1:.4f}"])
                log_file.flush()

                # Saving only when the dev macro-F1 gets better
                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    save_checkpoint(step, best_f1, os.path.join(args.out_dir, "best_checkpoint.pt"))

    log_file.close()

    total_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished: {total_steps} steps, "
          f"best dev macro-F1 {best_f1:.4f}, {total_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
