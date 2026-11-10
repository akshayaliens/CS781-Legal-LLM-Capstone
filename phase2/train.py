"""
CS781 Capstone - Phase 2: Legal Named Entity Recognition (L-NER)
Training script

Loads a Phase-1 checkpoint, puts the NERHead from model.py on top, and
trains it on fold1 with one of the methods in methods.py (full, lora,
lorma, bitfit, partial). fold2 is used as the dev set.

Usage (Colab):
    python train.py --arch decoder --method full \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p1/decoder/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p2 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p2/decoder_full

References:
    - AdamW: Loshchilov and Hutter, "Decoupled Weight Decay Regularization", 2019
    - Cosine LR schedule with warmup: Loshchilov and Hutter, "SGDR", 2017
"""

import argparse
import csv
import json
import math
import os
import random
import time

import torch

from methods import count_trainable, setup_method
from model import ModelConfig, build_ner_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", required=True, choices=["decoder", "encdec", "moe", "convmix"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--method", default="full",
                        choices=["full", "lora", "lorma", "bitfit", "partial"])
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--train_top_layers", type=int, default=4)

    # Much smaller learning rate than Phase 1 (6e-4), since the model
    # already knows legal text and only needs to learn the NER labels
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=50)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def load_examples(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pad_batch(examples, pad_token_id):
    # Padding every chunk to the longest one in the batch, since the last
    # chunk of a document is usually shorter than 512.
    # -100 labels are skipped by cross_entropy, so padding is not trained on
    max_len = max(len(ex["token_ids"]) for ex in examples)

    token_ids = torch.full((len(examples), max_len), pad_token_id, dtype=torch.long)
    label_ids = torch.full((len(examples), max_len), -100, dtype=torch.long)

    for i, ex in enumerate(examples):
        length = len(ex["token_ids"])
        token_ids[i, :length] = torch.tensor(ex["token_ids"])
        label_ids[i, :length] = torch.tensor(ex["label_ids"])

    return token_ids, label_ids


def make_batches(examples, batch_size, pad_token_id, rng=None):
    order = list(range(len(examples)))
    if rng is not None:
        rng.shuffle(order)

    batches = []
    for start in range(0, len(order), batch_size):
        batch_examples = [examples[i] for i in order[start:start + batch_size]]
        batches.append(pad_batch(batch_examples, pad_token_id))

    return batches


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Loading the label list that data_prep.py saved
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)
    label_vocab = meta["label_vocab"]
    num_labels = len(label_vocab)
    print(f"Number of labels: {num_labels}")

    # Building the model with the same settings the Phase-1 checkpoint used
    print(f"Loading Phase-1 checkpoint: {args.ckpt}")
    p1_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig(**p1_ckpt["config"])
    model = build_ner_model(config, num_labels)

    # Copying the Phase-1 weights into the backbone. The classifier head is
    # new, so it starts from random weights
    missing, unexpected = model.backbone.load_state_dict(p1_ckpt["model"], strict=False)
    print(f"Backbone loaded, missing keys: {len(missing)}, unused keys: {len(unexpected)}")

    # Freezing / adding weights for the chosen method. This has to come after
    # loading, since lora and lorma rename the layers they wrap
    setup_method(model, args.method, args.lora_rank, args.lora_alpha, args.train_top_layers)
    model = model.to(device)

    trainable, total = count_trainable(model)
    print(f"Method: {args.method}, training {trainable / 1e6:.2f}M of {total / 1e6:.1f}M "
          f"parameters ({100 * trainable / total:.2f}%)")

    # Loading the chunks that data_prep.py made
    train_examples = load_examples(os.path.join(args.data_dir, "fold1_examples.json"))
    dev_examples = load_examples(os.path.join(args.data_dir, "fold2_examples.json"))
    print(f"{len(train_examples)} train chunks, {len(dev_examples)} dev chunks")

    pad_token_id = config.pad_id
    dev_batches = make_batches(dev_examples, args.batch_size, pad_token_id)

    steps_per_epoch = math.ceil(len(train_examples) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch} steps per epoch, {total_steps} steps total")

    # Only giving the optimizer the weights that are being trained.
    # No weight decay on 1D weights (norm scales, biases), same as Phase 1
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    decay_params = [p for p in trainable_params if p.dim() >= 2]
    no_decay_params = [p for p in trainable_params if p.dim() < 2]
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

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = []
        for token_ids, label_ids in dev_batches:
            _, loss = model(token_ids.to(device), label_ids.to(device))
            losses.append(loss.item())
        model.train()
        return sum(losses) / len(losses)

    def save_checkpoint(step, best_val_loss, path):
        torch.save({
            "model": model.state_dict(),
            "config": config.__dict__,
            "label_vocab": label_vocab,
            "step": step,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }, path)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "learning_rate", "train_loss", "val_loss"])

    # Training loop
    train_rng = random.Random(args.seed)
    model.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    best_val_loss = float("inf")
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        # Shuffling the train chunks again at the start of every epoch
        train_batches = make_batches(train_examples, args.batch_size, pad_token_id, train_rng)

        for token_ids, label_ids in train_batches:
            lr = learning_rate_at(step)
            for group in optimizer.param_groups:
                group["lr"] = lr

            _, loss = model(token_ids.to(device), label_ids.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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

                # Saving only when the dev loss gets better
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
