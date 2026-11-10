"""
CS781 Capstone - Phase 3: Rhetorical Role Prediction (RR)
Training script

Loads a Phase-1 checkpoint, puts the sentence head on top and trains it on the
CL and IT train windows with one of the methods in methods.py. The checkpoint
with the best dev macro-F1 is saved.

Usage (Colab):
    python train.py --arch decoder --method full \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p1/decoder/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p3 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p3/decoder_full

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
from model import ModelConfig, build_rr_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", required=True, choices=["decoder", "encdec", "moe", "convmix"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--method", default="full",
                        choices=["full", "lora", "lorma", "ia3", "adalora"])
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16)

    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    # Sentence head is new, so it gets a bigger learning rate
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_class_weights(label_counts):
    # Giving rare labels a bigger weight in the loss, but not more than 10
    counts = torch.tensor(label_counts, dtype=torch.float).clamp(min=1)
    weights = (counts.sum() / (len(counts) * counts)).sqrt()
    return weights.clamp(max=10)


def make_batch(windows, pad_id):
    # Making all windows in a batch the same size, the empty sentence slots get label -100
    max_tokens = max(len(w["token_ids"]) for w in windows)
    max_sentences = max(len(w["sentence_spans"]) for w in windows)

    token_ids = torch.full((len(windows), max_tokens), pad_id, dtype=torch.long)
    sentence_mask = torch.zeros(len(windows), max_sentences, max_tokens)
    label_ids = torch.full((len(windows), max_sentences), -100, dtype=torch.long)

    for i, window in enumerate(windows):
        token_ids[i, :len(window["token_ids"])] = torch.tensor(window["token_ids"])
        for j, (start, end) in enumerate(window["sentence_spans"]):
            sentence_mask[i, j, start:end] = 1
        if "label_ids" in window:
            label_ids[i, :len(window["label_ids"])] = torch.tensor(window["label_ids"])

    return token_ids, sentence_mask, label_ids


@torch.no_grad()
def predict_documents(model, windows, num_sentences, num_labels, batch_size, pad_id, device):
    # Adding up each sentence's probabilities from every window it is in, then taking the best label
    model.eval()
    scores = {doc_id: torch.zeros(n, num_labels) for doc_id, n in num_sentences.items()}

    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start:start + batch_size]
        token_ids, sentence_mask, _ = make_batch(batch_windows, pad_id)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits, _ = model(token_ids.to(device), sentence_mask.to(device))
        probs = logits.float().softmax(dim=-1).cpu()

        for i, window in enumerate(batch_windows):
            for j, sentence_id in enumerate(window["sentence_ids"]):
                scores[window["doc_id"]][sentence_id] += probs[i, j]

    model.train()
    return {doc_id: s.argmax(dim=-1).tolist() for doc_id, s in scores.items()}


def macro_f1(docs, predictions, label_to_id, domain=None):
    # Macro-F1 over all dev sentences, or only over CL or IT
    gold, pred = [], []
    for doc in docs:
        if domain is not None and doc["domain"] != domain:
            continue
        gold.extend(label_to_id[label] for label in doc["labels"])
        pred.extend(predictions[doc["id"]])
    return f1_score(gold, pred, average="macro")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Loading what data_prep.py saved
    meta = load_json(os.path.join(args.data_dir, "meta.json"))
    id_to_label = meta["id_to_label"]
    label_to_id = {label: i for i, label in enumerate(id_to_label)}
    num_labels = len(id_to_label)

    train_windows = load_json(os.path.join(args.data_dir, "train_windows.json"))
    dev_windows = load_json(os.path.join(args.data_dir, "dev_windows.json"))
    dev_docs = load_json(os.path.join(args.data_dir, "dev_docs.json"))
    dev_num_sentences = {doc["id"]: len(doc["labels"]) for doc in dev_docs}
    print(f"{len(train_windows)} train windows, {len(dev_windows)} dev windows")

    steps_per_epoch = math.ceil(len(train_windows) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch} steps per epoch, {total_steps} steps total")

    # Class weights from the train label counts
    class_weights = make_class_weights(meta["train_label_counts"])
    print("Class weights:", [round(w, 2) for w in class_weights.tolist()])

    # Building the model with the same settings the Phase-1 checkpoint used
    print(f"Loading Phase-1 checkpoint: {args.ckpt}")
    p1_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig(**p1_ckpt["config"])
    model = build_rr_model(config, num_labels, class_weights=class_weights)

    missing, unexpected = model.backbone.load_state_dict(p1_ckpt["model"], strict=False)
    print(f"Backbone loaded, missing keys: {len(missing)}, unused keys: {len(unexpected)}")

    # Adding the method after loading the Phase-1 weights
    setup_method(model, args.method, args.lora_rank, args.lora_alpha, total_steps)
    model = model.to(device)

    trainable, total = count_trainable(model)
    print(f"Method: {args.method}, training {trainable / 1e6:.2f}M of {total / 1e6:.1f}M "
          f"parameters ({100 * trainable / total:.2f}%)")

    # Separate learning rates for the backbone and the sentence head
    backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("backbone.")]
    head_params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("backbone.")]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.learning_rate, "base_lr": args.learning_rate},
         {"params": head_params, "lr": args.head_learning_rate, "base_lr": args.head_learning_rate}],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    # Mixed precision, same as in phase1/train.py
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def lr_factor(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine

    def evaluate():
        predictions = predict_documents(model, dev_windows, dev_num_sentences, num_labels,
                                        args.batch_size, config.pad_id, device)
        return (macro_f1(dev_docs, predictions, label_to_id),
                macro_f1(dev_docs, predictions, label_to_id, "CL"),
                macro_f1(dev_docs, predictions, label_to_id, "IT"))

    def save_checkpoint(step, best_f1, path):
        torch.save({
            "model": model.state_dict(),
            "config": config.__dict__,
            "id_to_label": id_to_label,
            "step": step,
            "best_dev_f1": best_f1,
            "args": vars(args),
        }, path)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "train_loss", "dev_f1", "dev_f1_cl", "dev_f1_it"])

    # Training loop
    train_rng = random.Random(args.seed)
    model.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    best_f1 = -1.0
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        # Shuffling the train windows again at the start of every epoch
        order = list(range(len(train_windows)))
        train_rng.shuffle(order)

        for start in range(0, len(order), args.batch_size):
            batch_windows = [train_windows[i] for i in order[start:start + args.batch_size]]
            token_ids, sentence_mask, label_ids = make_batch(batch_windows, config.pad_id)

            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_factor(step)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss = model(token_ids.to(device), sentence_mask.to(device), label_ids.to(device))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            # AdaLoRA changes the rank of its layers after each step
            if args.method == "adalora":
                model.backbone.base_model.update_and_allocate(step)

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
