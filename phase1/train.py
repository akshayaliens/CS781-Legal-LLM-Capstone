"""
CS781 Capstone - Phase 1: Legal Language Model Pretraining
Training script

Trains one of the four architectures from model.py on the token data that
prepare_data.py produced. Saves a checkpoint after every evaluation, so a
Colab disconnect does not lose the run -- running the same command again
picks up training from the last saved checkpoint.

Usage (Colab):
    python train.py --arch decoder --data_dir /content/drive/MyDrive/CS781/data_p1 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p1/decoder

References:
    - AdamW optimizer: Loshchilov and Hutter, "Decoupled Weight Decay
      Regularization", ICLR 2019
    - Cosine learning rate schedule with warmup: Loshchilov and Hutter,
      "SGDR", ICLR 2017
    - Mixed precision training: Micikevicius et al., "Mixed Precision
      Training", ICLR 2018
"""

import argparse
import csv
import json
import math
import os
import time

import numpy as np
import torch

from model import PRESETS, ModelConfig, build_model, count_parameters


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", required=True, choices=list(PRESETS))
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--total_tokens", type=float, default=4e8,
                         help="total training tokens to process")
    parser.add_argument("--tokens_per_step", type=int, default=131072)
    parser.add_argument("--micro_batch", type=int, default=6)
    parser.add_argument("--block_size", type=int, default=1024)

    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_every", type=int, default=150)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


class TokenData:
    """
    Loads the per-source .bin token files as memory-mapped arrays (so the
    whole file does not need to fit in RAM) and draws random training
    windows, mixing sources in proportion to how many tokens each has.
    """

    def __init__(self, data_dir, sources):
        self.train_arrays = {}
        self.val_arrays = {}

        for source in sources:
            train_path = os.path.join(data_dir, f"train_{source}.bin")
            val_path = os.path.join(data_dir, f"val_{source}.bin")
            self.train_arrays[source] = np.memmap(train_path, dtype=np.uint16, mode="r")
            self.val_arrays[source] = np.memmap(val_path, dtype=np.uint16, mode="r")

        self.sources = np.array(sources)
        sizes = np.array([len(self.train_arrays[s]) for s in sources], dtype=np.float64)
        self.sample_weights = sizes / sizes.sum()

    def get_batch(self, split, batch_size, block_size, rng):
        """Returns a (batch_size, block_size + 1) tensor of token ids."""
        if split == "train":
            chosen_sources = rng.choice(self.sources, size=batch_size, p=self.sample_weights)
            arrays = [self.train_arrays[s] for s in chosen_sources]
        else:
            arrays = [self.val_arrays[split]] * batch_size

        rows = []
        for array in arrays:
            start = rng.integers(0, len(array) - block_size - 1)
            window = array[start:start + block_size + 1].astype(np.int64)
            rows.append(torch.from_numpy(window))

        return torch.stack(rows)


def make_batch(window, arch, config, device):
    """Turns a (batch, block_size + 1) window into model inputs."""
    if arch == "encdec":
        enc_len, dec_len = config.enc_len, config.dec_len
        src = window[:, :enc_len]
        tgt = window[:, enc_len:enc_len + dec_len]
        bos_column = torch.full((window.shape[0], 1), config.bos_id, dtype=torch.long)
        tgt_in = torch.cat([bos_column, tgt[:, :-1]], dim=1)
        return src.to(device), tgt_in.to(device), tgt.to(device)

    idx = window[:, :-1].to(device)
    targets = window[:, 1:].to(device)
    return idx, targets


def run_model(model, batch, arch):
    if arch == "encdec":
        src, tgt_in, targets = batch
        logits, loss = model(src, tgt_in, targets)
    else:
        idx, targets = batch
        logits, loss = model(idx, targets)
    return loss


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    use_amp = device == "cuda"
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16

    # Loading the tokenizer metadata that prepare_data.py saved
    with open(os.path.join(args.data_dir, "meta.json")) as f:
        meta = json.load(f)

    preset = dict(PRESETS[args.arch])
    preset.update(
        vocab_size=meta["vocab_size"],
        eot_id=meta["eot_id"],
        pad_id=meta["pad_id"],
        bos_id=meta["bos_id"],
        block_size=args.block_size,
    )
    config = ModelConfig(**preset)
    window_len = config.enc_len + config.dec_len if args.arch == "encdec" else config.block_size

    model = build_model(config).to(device)
    param_counts = count_parameters(model)
    print(f"Architecture: {args.arch}")
    print(f"Parameters: total {param_counts['total']/1e6:.1f}M, "
          f"active/token {param_counts['active_per_token']/1e6:.1f}M")
    assert 50e6 <= param_counts["total"] <= 300e6, "model size must stay within 50M-300M"

    data = TokenData(args.data_dir, meta["sources"])
    print("Source sampling weights:",
          {s: round(float(w), 3) for s, w in zip(data.sources, data.sample_weights)})

    tokens_per_micro_batch = args.micro_batch * window_len
    grad_accum_steps = max(1, args.tokens_per_step // tokens_per_micro_batch)
    tokens_per_step = grad_accum_steps * tokens_per_micro_batch
    total_steps = max(1, int(args.total_tokens // tokens_per_step))
    print(f"Micro batch {args.micro_batch}, window {window_len}, "
          f"grad accumulation {grad_accum_steps} -> {tokens_per_step} tokens/step, "
          f"{total_steps} steps total")

    decay_params = [p for p in model.parameters() if p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": args.weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=args.learning_rate, betas=(0.9, 0.95)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

    # Resuming from the last checkpoint if this run was interrupted before
    checkpoint_path = os.path.join(args.out_dir, "last_checkpoint.pt")
    start_step = 0
    best_val_loss = float("inf")

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resuming from step {start_step}, best val loss so far {best_val_loss:.4f}")

    def learning_rate_at(step):
        if step < args.warmup_steps:
            return args.learning_rate * (step + 1) / args.warmup_steps
        progress = min(1.0, (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return args.learning_rate * (args.min_lr_ratio + (1 - args.min_lr_ratio) * cosine)

    def save_checkpoint(step, best_val_loss, path):
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.__dict__,
            "step": step,
            "best_val_loss": best_val_loss,
            "args": vars(args),
        }, path + ".tmp")
        os.replace(path + ".tmp", path)

    @torch.no_grad()
    def evaluate():
        model.eval()
        eval_rng = np.random.default_rng(0)  # same validation batches every time
        losses_per_source = {}

        for source in data.sources:
            losses = []
            for _ in range(args.eval_batches):
                window = data.get_batch(source, args.micro_batch, window_len, eval_rng)
                batch = make_batch(window, args.arch, config, device)
                with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    loss = run_model(model, batch, args.arch)
                losses.append(loss.item())
            losses_per_source[source] = float(np.mean(losses))

        losses_per_source["average"] = float(np.mean(list(losses_per_source.values())))
        model.train()
        return losses_per_source

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if os.path.getsize(log_path) == 0:
        log_writer.writerow(["step", "tokens_seen", "learning_rate", "train_loss"]
                             + [f"val_{s}" for s in data.sources] + ["val_average"])

    train_rng = np.random.default_rng(args.seed + start_step)
    model.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    start_time = time.time()

    for step in range(start_step, total_steps):
        lr = learning_rate_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        for _ in range(grad_accum_steps):
            window = data.get_batch("train", args.micro_batch, window_len, train_rng)
            batch = make_batch(window, args.arch, config, device)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                loss = run_model(model, batch, args.arch)

            scaler.scale(loss / grad_accum_steps).backward()
            running_loss += loss.item()
            running_count += 1

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        current_step = step + 1

        if current_step % args.log_every == 0:
            avg_loss = running_loss / running_count
            elapsed_min = (time.time() - start_time) / 60
            print(f"step {current_step}/{total_steps}  loss {avg_loss:.4f}  "
                  f"lr {lr:.2e}  elapsed {elapsed_min:.1f} min")
            running_loss, running_count = 0.0, 0

        if current_step % args.eval_every == 0 or current_step == total_steps:
            val_losses = evaluate()
            print("  validation:", {k: round(v, 4) for k, v in val_losses.items()})

            log_writer.writerow(
                [current_step, current_step * tokens_per_step, f"{lr:.3e}",
                 f"{avg_loss:.4f}"] + [f"{val_losses[s]:.4f}" for s in data.sources]
                + [f"{val_losses['average']:.4f}"]
            )
            log_file.flush()

            save_checkpoint(current_step, best_val_loss, checkpoint_path)
            if val_losses["average"] < best_val_loss:
                best_val_loss = val_losses["average"]
                save_checkpoint(current_step, best_val_loss,
                                 os.path.join(args.out_dir, "best_checkpoint.pt"))

    log_file.close()

    total_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished: {total_steps} steps, "
          f"best validation loss {best_val_loss:.4f}, {total_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
