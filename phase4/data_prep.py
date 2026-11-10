"""
CS781 Capstone - Phase 4: Court Judgment Prediction (CJPE)
Data preparation

Cleans the case texts, tokenizes them and keeps the last 4 chunks of 512
tokens of every case. A balanced random subset of the train cases is used.

Usage (Colab):
    python data_prep.py --out_dir /content/drive/MyDrive/CS781/data_p4 \
        --tokenizer_path /content/drive/MyDrive/CS781/data_p1/tokenizer.json

References:
    - Keeping the end of the judgment and splitting it into 512-token chunks:
      Malik et al., "ILDC for CJPE: Indian Legal Documents Corpus for Court
      Judgment Prediction and Explanation", ACL 2021
"""

import argparse
import json
import os
import random
from collections import Counter

DATASET_REPO = "Exploration-Lab/CS-781-Capstone"

FILES = {
    "train": "Phase_4_CJPE/Train/train.json",
    "dev": "Phase_4_CJPE/Train/dev.json",
    "test": "Phase_4_CJPE/Test/test.json",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--cache_dir", default="/content/hf_cache_p4")
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--chunk_tokens", type=int, default=512)
    # Number of train cases to use, half with label 0 and half with label 1
    parser.add_argument("--train_cases", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


# download_file, load_json and save_json are the same as in phase3/data_prep.py
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
    # Lowercasing (dev and test are lowercase, train is not) and
    # changing newlines and extra spaces to a single space
    return " ".join(text.lower().split())


def make_chunks(token_ids, chunk_tokens, num_chunks):
    # Keeping only the last num_chunks * chunk_tokens tokens of the case
    token_ids = token_ids[-num_chunks * chunk_tokens:]

    # Cutting them into chunks, only the first chunk can be shorter than chunk_tokens
    first_size = len(token_ids) % chunk_tokens
    chunks = []
    if first_size > 0:
        chunks.append(token_ids[:first_size])
    for start in range(first_size, len(token_ids), chunk_tokens):
        chunks.append(token_ids[start:start + chunk_tokens])
    return chunks


def build_cases(cases, tokenizer, args, name):
    texts = [" " + clean_text(case["text"]) for case in cases]
    encodings = tokenizer.encode_batch(texts)

    built = []
    for case, encoding in zip(cases, encodings):
        token_ids = encoding.ids
        # Giving an empty case one token (0 = <|endoftext|>) so it still has a chunk
        if len(token_ids) == 0:
            token_ids = [0]

        item = {
            "id": case["id"],
            "chunks": make_chunks(token_ids, args.chunk_tokens, args.num_chunks),
        }
        # Adding the label for train and dev
        if "label" in case:
            item["label"] = case["label"]
        built.append(item)

    avg_chunks = sum(len(item["chunks"]) for item in built) / len(built)
    print(f"  {name}: {len(built)} cases, {avg_chunks:.2f} chunks per case on average")
    return built


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    print("Downloading Phase 4 files ...", flush=True)
    paths = {name: download_file(rel_path, args.cache_dir) for name, rel_path in FILES.items()}

    train = load_json(paths["train"])
    dev = load_json(paths["dev"])

    # Keeping only id and text for test, test labels are private and never used
    test = [{"id": case["id"], "text": case["text"]} for case in load_json(paths["test"])]

    full_train_counts = Counter(case["label"] for case in train)
    print("Full train label counts:", dict(full_train_counts))

    # Taking a random balanced subset of train, the same number of cases for each label
    rng = random.Random(args.seed)
    per_label = args.train_cases // 2
    train_subset = []
    for label in [0, 1]:
        label_cases = [case for case in train if case["label"] == label]
        train_subset.extend(rng.sample(label_cases, per_label))
    rng.shuffle(train_subset)

    print("Tokenizing and making chunks ...", flush=True)
    train_cases = build_cases(train_subset, tokenizer, args, "train")
    dev_cases = build_cases(dev, tokenizer, args, "dev")
    test_cases = build_cases(test, tokenizer, args, "test")

    save_json(train_cases, os.path.join(args.out_dir, "train_cases.json"))
    save_json(dev_cases, os.path.join(args.out_dir, "dev_cases.json"))
    save_json(test_cases, os.path.join(args.out_dir, "test_cases.json"))

    train_counts = Counter(case["label"] for case in train_cases)
    meta = {
        "num_labels": 2,
        "train_label_counts": [train_counts[0], train_counts[1]],
        "full_train_label_counts": [full_train_counts[0], full_train_counts[1]],
        "num_chunks": args.num_chunks,
        "chunk_tokens": args.chunk_tokens,
        "train_cases": args.train_cases,
        "seed": args.seed,
    }
    save_json(meta, os.path.join(args.out_dir, "meta.json"))

    print(f"\nSaved to {args.out_dir}")
    print("Train subset label counts:", dict(train_counts))


if __name__ == "__main__":
    main()
