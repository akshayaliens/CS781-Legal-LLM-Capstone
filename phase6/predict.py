"""
CS781 Capstone - Phase 6: Abstractive Summarization
Prediction script

Writes summaries with a trained model. With --split val it prints ROUGE-1
and ROUGE-L on some validation documents, with --split test it saves
prediction.json and submission.zip for Codabench.

Usage (Colab):
    python predict.py --split val \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p6/decoder_lora/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p6 \
        --tokenizer_path /content/drive/MyDrive/CS781/data_p1/tokenizer.json \
        --out_dir /content/drive/MyDrive/CS781/submissions_p6/decoder_lora

References:
    - Blocking repeated 3-grams while decoding: Paulus et al., "A Deep
      Reinforced Model for Abstractive Summarization", ICLR 2018
    - ROUGE: Lin, "ROUGE: A Package for Automatic Evaluation of Summaries",
      2004 (rouge-score package)
"""

import argparse
import json
import os
import time
import zipfile

import torch

from methods import setup_method
from model import ModelConfig, build_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--out_dir", required=True)

    # Number of validation documents to write summaries for
    parser.add_argument("--val_docs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--max_new_tokens", type=int, default=510)
    # Not allowing the same 3 tokens in a row twice in a summary (1 = on, 0 = off)
    parser.add_argument("--block_repeats", type=int, default=1)

    return parser.parse_args()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_model(ckpt_path, device):
    # Building the model again with the saved settings and the same method, then loading the trained weights
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = ModelConfig(**ckpt["config"])
    saved_args = ckpt["args"]

    model = build_model(config)
    setup_method(model, saved_args["method"], saved_args["rank"], saved_args["alpha"], saved_args["adapter_size"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    print(f"Loaded {config.arch} ({saved_args['method']}) from step {ckpt['step']}, "
          f"validation loss {ckpt['best_val_loss']:.4f}")
    return model, config


def banned_tokens(summary):
    # Tokens that would repeat 3 tokens in a row that are already in the summary
    banned = []
    if len(summary) < 2:
        return banned
    last_two = summary[-2:]
    for i in range(len(summary) - 2):
        if summary[i:i + 2] == last_two:
            banned.append(summary[i + 2])
    return banned


@torch.no_grad()
def generate_batch(model, docs, config, args, device):
    # Writing the summaries of a batch of documents one token at a time (greedy decoding)
    summaries = []
    for doc in docs:
        summaries.append([])
    finished = [False] * len(docs)

    if config.arch == "encdec":
        # The encoder reads the documents once, padding is put at the end
        max_src = max(len(doc["doc_ids"]) for doc in docs)
        src_ids = torch.full((len(docs), max_src), config.pad_id, dtype=torch.long)
        for i, doc in enumerate(docs):
            src_ids[i, :len(doc["doc_ids"])] = torch.tensor(doc["doc_ids"])
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            memory = model.encode(src_ids.to(device))
    else:
        # document <s>, padding is put at the start so every row ends at the same place
        prompts = []
        for doc in docs:
            prompts.append(doc["doc_ids"] + [config.bos_id])
        max_prompt = max(len(p) for p in prompts)
        prompt_ids = torch.full((len(docs), max_prompt), config.pad_id, dtype=torch.long)
        for i, prompt in enumerate(prompts):
            prompt_ids[i, max_prompt - len(prompt):] = torch.tensor(prompt)

    for step in range(args.max_new_tokens):
        if config.arch == "encdec":
            tgt_rows = []
            for summary in summaries:
                tgt_rows.append([config.bos_id] + summary)
            tgt_ids = torch.tensor(tgt_rows, dtype=torch.long, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits = model.decode(tgt_ids, memory, last_only=True)
        else:
            summary_ids = torch.tensor(summaries, dtype=torch.long)
            input_ids = torch.cat([prompt_ids, summary_ids], dim=1).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                logits, _ = model(input_ids)

        logits = logits[:, -1, :].float()

        for i in range(len(docs)):
            # Finished summaries only get padding, so all rows stay the same length
            if finished[i]:
                summaries[i].append(config.pad_id)
                continue

            if args.block_repeats == 1:
                for token in banned_tokens(summaries[i]):
                    logits[i, token] = float("-inf")
            next_token = int(logits[i].argmax())

            summaries[i].append(next_token)
            if next_token == config.eot_id:
                finished[i] = True

        if all(finished):
            break

    # Keeping the tokens before the end token
    results = []
    for summary in summaries:
        tokens = []
        for token in summary:
            if token == config.eot_id or token == config.pad_id:
                break
            tokens.append(token)
        results.append(tokens)
    return results


def write_summaries(model, docs, config, tokenizer, args, device):
    texts = []
    start_time = time.time()
    for start in range(0, len(docs), args.batch_size):
        batch_docs = docs[start:start + args.batch_size]
        for tokens in generate_batch(model, batch_docs, config, args, device):
            texts.append(tokenizer.decode(tokens).strip())
        elapsed_min = (time.time() - start_time) / 60
        print(f"  {len(texts)}/{len(docs)} documents done, {elapsed_min:.1f} min", flush=True)
    return texts


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    model, config = load_model(args.ckpt, device)

    if args.split == "val":
        docs = load_json(os.path.join(args.data_dir, "val_docs.json"))[:args.val_docs]
    else:
        docs = load_json(os.path.join(args.data_dir, "test_docs.json"))
    print(f"Writing summaries for {len(docs)} {args.split} documents ...")

    texts = write_summaries(model, docs, config, tokenizer, args, device)

    # Average summary length in words
    total_words = 0
    for text in texts:
        total_words += len(text.split())
    print(f"Average generated summary length: {total_words / len(texts):.0f} words")

    if args.split == "val":
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

        rouge1, rougeL, ref_words = 0.0, 0.0, 0
        saved = []
        for doc, text in zip(docs, texts):
            scores = scorer.score(doc["summary"], text)
            rouge1 += scores["rouge1"].fmeasure
            rougeL += scores["rougeL"].fmeasure
            ref_words += len(doc["summary"].split())
            saved.append({"id": doc["id"], "summary": text, "reference": doc["summary"]})

        print(f"Average reference summary length: {ref_words / len(docs):.0f} words")
        print(f"Validation ROUGE-1 F1: {rouge1 / len(docs):.4f}")
        print(f"Validation ROUGE-L F1: {rougeL / len(docs):.4f}")

        with open(os.path.join(args.out_dir, "val_summaries.json"), "w", encoding="utf-8") as f:
            json.dump(saved, f, ensure_ascii=False, indent=1)
    else:
        # One id and summary for every test document
        predictions = []
        for doc, text in zip(docs, texts):
            predictions.append({"id": doc["id"], "summary": text})

        pred_path = os.path.join(args.out_dir, "prediction.json")
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False)

        zip_path = os.path.join(args.out_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pred_path, arcname="prediction.json")

        print(f"{len(predictions)} summaries saved, zip ready: {zip_path}")

    print("\nFirst summary:")
    print(texts[0][:1000])


if __name__ == "__main__":
    main()
