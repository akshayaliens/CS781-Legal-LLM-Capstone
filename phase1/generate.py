"""
CS781 Capstone - Phase 1: Legal Language Model Pretraining
Generation script

Loads a trained checkpoint and writes the required Codabench submission file:
a prediction.json with a 128-word continuation for every test prompt, zipped
at the root as submission.zip.

Usage (Colab):
    python generate.py --ckpt /content/drive/MyDrive/CS781/ckpt_p1/decoder/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p1 \
        --prompts /content/drive/MyDrive/CS781/data_p1/test.json \
        --out_dir /content/drive/MyDrive/CS781/submissions_p1/decoder

References:
    - Nucleus (top-p) sampling: Holtzman et al., "The Curious Case of Neural
      Text Degeneration", ICLR 2020
    - Repetition penalty: Keskar et al., "CTRL: A Conditional Transformer
      Language Model for Controllable Generation", 2019
"""

import argparse
import json
import os
import zipfile

import torch

from model import ModelConfig, build_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True, help="folder with tokenizer.json")
    parser.add_argument("--prompts", required=True, help="path to test.json")
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--words", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=220)
    parser.add_argument("--batch_size", type=int, default=16)

    # Decoding settings. greedy is deterministic (always the most likely word)
    #  sampling adds controlled randomness so the text does not read the same way every time.
    parser.add_argument("--decoding", default="greedy", choices=["greedy", "sample"])
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)
    parser.add_argument("--seed", type=int, default=1234)

    return parser.parse_args()


def apply_repetition_penalty(logits, generated_ids, penalty):
    # Making already-used tokens less likely, so the model does not loop
    for token_id in set(generated_ids):
        score = logits[token_id]
        logits[token_id] = score / penalty if score > 0 else score * penalty
    return logits


def sample_from_logits(logits, temperature, top_p):
    logits = logits / max(temperature, 1e-5)
    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
    probs = sorted_logits.softmax(dim=-1)
    cumulative = probs.cumsum(dim=-1)

    # Keeping only the smallest set of top words whose probability adds up
    # to top_p, dropping the long unlikely tail (nucleus sampling)
    cutoff = cumulative > top_p
    cutoff[1:] = cutoff[:-1].clone()
    cutoff[0] = False
    sorted_logits[cutoff] = float("-inf")

    filtered_logits = torch.full_like(logits, float("-inf"))
    filtered_logits[sorted_idx] = sorted_logits
    next_id = torch.multinomial(filtered_logits.softmax(dim=-1), 1)
    return next_id.item()


@torch.no_grad()
def generate_one(model, config, prompt_ids, args, device):
    """Generates token ids for a single prompt, stopping once enough words are made."""
    generated = []

    if config.arch == "encdec":
        context = prompt_ids[-config.enc_len:]
        memory = model.encode(torch.tensor([context], device=device))
        seq = [config.bos_id]
        max_new = min(args.max_new_tokens, config.dec_len - 1)
    else:
        seq = list(prompt_ids[-(config.block_size - args.max_new_tokens):])
        max_new = args.max_new_tokens

    for _ in range(max_new):
        if config.arch == "encdec":
            tgt = torch.tensor([seq[-config.dec_len:]], device=device)
            logits, _ = model.decode(tgt, memory, last_only=True), None
        else:
            idx = torch.tensor([seq[-config.block_size:]], device=device)
            logits, _ = model(idx)

        next_logits = logits[0, -1, :].clone()
        next_logits[config.eot_id] = float("-inf")
        next_logits[config.pad_id] = float("-inf")

        if args.repetition_penalty != 1.0:
            next_logits = apply_repetition_penalty(next_logits, generated, args.repetition_penalty)

        if args.decoding == "greedy":
            next_id = int(next_logits.argmax())
        else:
            next_id = sample_from_logits(next_logits, args.temperature, args.top_p)

        seq.append(next_id)
        generated.append(next_id)

        # Stopping a little early once we likely have enough words, so we
        # are not always generating the full max_new_tokens
        if len(generated) >= args.words + 15:
            break

    return generated


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    from tokenizers import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(os.path.join(args.data_dir, "tokenizer.json"))

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["config"])
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded {checkpoint['args']['arch']} checkpoint from step {checkpoint['step']}, "
          f"val loss {checkpoint['best_val_loss']:.4f}")

    with open(args.prompts, encoding="utf-8") as f:
        prompts = json.load(f)

    predictions = {}
    ids = list(prompts.keys())
    for i, example_id in enumerate(ids):
        prompt_text = " ".join(prompts[example_id]["text"].split())
        prompt_ids = tokenizer.encode(prompt_text).ids

        generated_ids = generate_one(model, config, prompt_ids, args, device)
        continuation = tokenizer.decode(generated_ids)

        # Keeping exactly the requested number of words, as the rules
        # state the text field must contain only the continuation
        words = continuation.split()[:args.words]
        predictions[example_id] = {"text": " ".join(words)}

        if (i + 1) % 20 == 0 or (i + 1) == len(ids):
            print(f"generated {i + 1}/{len(ids)} prompts", flush=True)

    pred_path = os.path.join(args.out_dir, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)

    zip_path = os.path.join(args.out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(pred_path, arcname="prediction.json")

    print(f"\nWrote {pred_path}")
    print(f"Wrote {zip_path} (ready to upload to Codabench)")

    # Printing one example so we can sanity-check the output by eye
    sample_id = ids[0]
    print(f"\nSample prompt (last 150 chars): ...{prompts[sample_id]['text'][-150:]}")
    print(f"Sample continuation: {predictions[sample_id]['text']}")


if __name__ == "__main__":
    main()
