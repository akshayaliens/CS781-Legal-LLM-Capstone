"""
CS781 Capstone - Phase 6: Abstractive Summarization
Training script

Loads a Phase-1 checkpoint, adds one of the methods in methods.py and trains
it to write the summary after the document. The checkpoint with the lowest
validation loss is saved.

Usage (Colab):
    python train.py --arch decoder --method lora \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p1/decoder/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p6 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p6/decoder_lora

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
from model import ModelConfig, build_model, compute_loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", required=True, choices=["decoder", "encdec", "moe", "convmix"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--method", default="lora", choices=["lora", "lorma", "adapter", "full"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--adapter_size", type=int, default=64)

    # Longest sequence is 512 document tokens, the <s> token, 510 summary tokens and the end token = 1024
    parser.add_argument("--max_len", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    # Adding up the gradients of 4 batches before each update, so one update sees 16 documents
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=30)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # Number of validation documents used for the loss during training
    parser.add_argument("--val_eval_docs", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=100)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_decoder_batch(docs, config):
    # One sequence per document: document <s> summary <|endoftext|>
    # The model predicts the next token, so the targets are the sequence moved by one
    sequences = []
    targets = []
    for doc in docs:
        seq = doc["doc_ids"] + [config.bos_id] + doc["summary_ids"] + [config.eot_id]
        target = seq[1:]
        # No loss on the document tokens, only on the summary and the end token
        for j in range(len(doc["doc_ids"])):
            target[j] = -100
        sequences.append(seq[:-1])
        targets.append(target)

    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((len(docs), max_len), config.pad_id, dtype=torch.long)
    target_ids = torch.full((len(docs), max_len), -100, dtype=torch.long)
    for i in range(len(docs)):
        input_ids[i, :len(sequences[i])] = torch.tensor(sequences[i])
        target_ids[i, :len(targets[i])] = torch.tensor(targets[i])

    return input_ids, target_ids


def make_encdec_batch(docs, config):
    # Encoder gets the document, decoder gets <s> summary and predicts summary <|endoftext|>
    max_src = max(len(doc["doc_ids"]) for doc in docs)
    max_tgt = max(len(doc["summary_ids"]) for doc in docs) + 1

    src_ids = torch.full((len(docs), max_src), config.pad_id, dtype=torch.long)
    tgt_in = torch.full((len(docs), max_tgt), config.pad_id, dtype=torch.long)
    target_ids = torch.full((len(docs), max_tgt), -100, dtype=torch.long)

    for i, doc in enumerate(docs):
        src_ids[i, :len(doc["doc_ids"])] = torch.tensor(doc["doc_ids"])
        tgt_in[i, :len(doc["summary_ids"]) + 1] = torch.tensor([config.bos_id] + doc["summary_ids"])
        target_ids[i, :len(doc["summary_ids"]) + 1] = torch.tensor(doc["summary_ids"] + [config.eot_id])

    return src_ids, tgt_in, target_ids


def run_model(model, docs, config, device):
    # Getting the logits and the summary loss for a batch, encdec needs a different batch than the other three
    if config.arch == "encdec":
        src_ids, tgt_in, target_ids = make_encdec_batch(docs, config)
        logits, loss = model(src_ids.to(device), tgt_in.to(device), target_ids.to(device))
    else:
        input_ids, target_ids = make_decoder_batch(docs, config)
        logits, loss = model(input_ids.to(device), target_ids.to(device))
    return logits, loss, target_ids


def load_phase1_config(ckpt_config, max_len):
    # The Phase-1 checkpoints were saved with 512 positions, changing it to 1024 so the longer sequences fit
    settings = dict(ckpt_config)
    settings["block_size"] = max_len
    settings["enc_len"] = max_len // 2
    settings["dec_len"] = max_len // 2
    return ModelConfig(**settings)


@torch.no_grad()
def val_loss(model, docs, batch_size, config, device):
    # Average summary loss on the validation documents, moe's aux loss is not added here
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(docs), batch_size):
        batch_docs = docs[start:start + batch_size]
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            logits, _, target_ids = run_model(model, batch_docs, config, device)
        total += compute_loss(logits, target_ids.to(device)).item()
        count += 1
    model.train()
    return total / count


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Loading the train and validation documents
    train_docs = load_json(os.path.join(args.data_dir, "train_docs.json"))
    val_docs = load_json(os.path.join(args.data_dir, "val_docs.json"))
    val_eval = val_docs[:args.val_eval_docs]
    print(f"{len(train_docs)} train documents, {len(val_eval)} validation documents used for the loss")

    # One step is one update of the weights, made from batch_size * accum_steps documents
    steps_per_epoch = math.ceil(len(train_docs) / (args.batch_size * args.accum_steps))
    total_steps = steps_per_epoch * args.epochs
    print(f"{args.batch_size * args.accum_steps} documents per update, "
          f"{steps_per_epoch} steps per epoch, {total_steps} steps total")

    # Building the model with the same settings the Phase-1 checkpoint used
    print(f"Loading Phase-1 checkpoint: {args.ckpt}")
    p1_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = load_phase1_config(p1_ckpt["config"], args.max_len)
    model = build_model(config)

    missing, unexpected = model.load_state_dict(p1_ckpt["model"], strict=False)
    print(f"Phase-1 weights loaded, missing keys: {len(missing)}, unused keys: {len(unexpected)}")

    # Adding the method after loading the Phase-1 weights
    setup_method(model, args.method, args.rank, args.alpha, args.adapter_size)
    model = model.to(device)

    trainable, total = count_trainable(model)
    print(f"Method: {args.method}, training {trainable / 1e6:.2f}M of {total / 1e6:.1f}M "
          f"parameters ({100 * trainable / total:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
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

    def save_checkpoint(step, best_loss, path):
        torch.save({
            "model": model.state_dict(),
            "config": config.__dict__,
            "step": step,
            "best_val_loss": best_loss,
            "args": vars(args),
        }, path)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "train_loss", "val_loss"])

    # Training loop
    train_rng = random.Random(args.seed)
    model.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    best_loss = float("inf")
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        # Shuffling the train documents again at the start of every epoch
        order = list(range(len(train_docs)))
        train_rng.shuffle(order)

        batch_starts = list(range(0, len(order), args.batch_size))
        for batch_num, start in enumerate(batch_starts):
            batch_docs = [train_docs[i] for i in order[start:start + args.batch_size]]

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _, loss, _ = run_model(model, batch_docs, config, device)

            # Gradients of 4 batches get added up, so dividing the loss by 4 makes it an average
            scaler.scale(loss / args.accum_steps).backward()
            running_loss += loss.item()
            running_count += 1

            # Updating the weights once every 4 batches, and also after the last batch of the epoch
            if (batch_num + 1) % args.accum_steps != 0 and batch_num + 1 != len(batch_starts):
                continue

            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * lr_factor(step)

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
                      f"elapsed {elapsed_min:.1f} min", flush=True)
                running_loss, running_count = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                loss_now = val_loss(model, val_eval, args.batch_size, config, device)
                print(f"  validation loss: {loss_now:.4f}", flush=True)

                log_writer.writerow([step, epoch + 1, f"{avg_loss:.4f}", f"{loss_now:.4f}"])
                log_file.flush()

                # Saving only when the validation loss gets lower
                if loss_now < best_loss:
                    best_loss = loss_now
                    save_checkpoint(step, best_loss, os.path.join(args.out_dir, "best_checkpoint.pt"))

    log_file.close()

    total_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished: {total_steps} steps, "
          f"best validation loss {best_loss:.4f}, {total_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
