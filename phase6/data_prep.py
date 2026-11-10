"""
CS781 Capstone - Phase 6: Abstractive Summarization
Data preparation

Makes a validation split from train, tokenizes the documents and summaries
and keeps the first 256 and last 256 tokens of every document.

Usage (Colab):
    python data_prep.py --out_dir /content/drive/MyDrive/CS781/data_p6 \
        --tokenizer_path /content/drive/MyDrive/CS781/data_p1/tokenizer.json
"""

import argparse
import json
import os
import random

DATASET_REPO = "Exploration-Lab/CS-781-Capstone"

FILES = {
    "train": "Phase_6_Summ/Train/train.json",
    "test": "Phase_6_Summ/Test/test.json",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--cache_dir", default="/content/hf_cache_p6")
    # Tokens kept from the start and from the end of every document
    parser.add_argument("--head_tokens", type=int, default=256)
    parser.add_argument("--tail_tokens", type=int, default=256)
    parser.add_argument("--summary_tokens", type=int, default=510)
    # Share of train held out for validation
    parser.add_argument("--val_share", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def download_file(rel_path, cache_dir):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        DATASET_REPO,
        rel_path,
        repo_type="dataset",
        local_dir=cache_dir
    )


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def clean_text(text):
    # Changing newlines and extra spaces to a single space, no lowercasing
    return " ".join(text.split())


def cut_document(token_ids, head_tokens, tail_tokens):
    # Keeping the start (facts, the appeal and the Act) and the end (the Court's decision)
    if len(token_ids) <= head_tokens + tail_tokens:
        return token_ids
    return token_ids[:head_tokens] + token_ids[-tail_tokens:]


def build_docs(docs, tokenizer, args, name):
    doc_texts = [" " + clean_text(doc["document"]) for doc in docs]
    doc_encodings = tokenizer.encode_batch(doc_texts)

    built = []
    for i, doc in enumerate(docs):
        doc_ids = doc_encodings[i].ids
        # Giving an empty document one token (0 = <|endoftext|>)
        if len(doc_ids) == 0:
            doc_ids = [0]

        item = {
            "id": doc["id"],
            "doc_ids": cut_document(doc_ids, args.head_tokens, args.tail_tokens),
        }

        # Train and validation documents also have the summary
        if "summary" in doc:
            summary = clean_text(doc["summary"])
            item["summary"] = summary
            summary_ids = tokenizer.encode(" " + summary).ids
            item["summary_ids"] = summary_ids[:args.summary_tokens]
        built.append(item)

    print(f"  {name}: {len(built)} documents")
    return built


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    print("Downloading Phase 6 files ...", flush=True)
    paths = {}
    for name, rel_path in FILES.items():
        paths[name] = download_file(rel_path, args.cache_dir)

    train = load_json(paths["train"])
    test = load_json(paths["test"])

    # Holding out a random part of train for validation
    random.Random(args.seed).shuffle(train)
    num_val = int(len(train) * args.val_share)
    val = train[:num_val]
    train = train[num_val:]

    print("Tokenizing ...", flush=True)
    train_docs = build_docs(train, tokenizer, args, "train")
    val_docs = build_docs(val, tokenizer, args, "val")
    test_docs = build_docs(test, tokenizer, args, "test")

    save_json(train_docs, os.path.join(args.out_dir, "train_docs.json"))
    save_json(val_docs, os.path.join(args.out_dir, "val_docs.json"))
    save_json(test_docs, os.path.join(args.out_dir, "test_docs.json"))

    meta = {
        "num_train": len(train_docs),
        "num_val": len(val_docs),
        "num_test": len(test_docs),
        "head_tokens": args.head_tokens,
        "tail_tokens": args.tail_tokens,
        "summary_tokens": args.summary_tokens,
        "val_share": args.val_share,
        "seed": args.seed,
    }
    save_json(meta, os.path.join(args.out_dir, "meta.json"))

    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
