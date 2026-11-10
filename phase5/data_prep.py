"""
CS781 Capstone - Phase 5: Legal Statute Identification (LSI)
Data preparation

Cleans the case texts, tokenizes them and keeps the first 4 chunks of 512
tokens of every case. A train subset is used where every statute has at
least a minimum number of cases, the rest is filled at random.

Usage (Colab):
    python data_prep.py --out_dir /content/drive/MyDrive/CS781/data_p5 \
        --tokenizer_path /content/drive/MyDrive/CS781/data_p1/tokenizer.json

References:
    - Splitting long documents into 512-token chunks: Malik et al.,
      "ILDC for CJPE", ACL 2021 (same chunking as phase4/data_prep.py)
    - LSI task and the masked <SECTION>/<ACT> tags: Paul et al., "LeSICiN:
      A Heterogeneous Graph-based Approach for Automatic Legal Statute
      Identification from Indian Legal Documents", AAAI 2022
"""

import argparse
import json
import os
import random
from collections import Counter

DATASET_REPO = "Exploration-Lab/CS-781-Capstone"

FILES = {
    "train": "Phase_5_LSI/Train/train.json",
    "dev": "Phase_5_LSI/Train/dev.json",
    "test": "Phase_5_LSI/Test/test.json",
}

# Statute ids in the data are 0 to 99, so the id is also the output index
NUM_LABELS = 100


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--cache_dir", default="/content/hf_cache_p5")
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--chunk_tokens", type=int, default=512)
    # Number of train cases to use
    parser.add_argument("--train_cases", type=int, default=8000)
    # Every statute gets at least this many cases in the train subset
    parser.add_argument("--min_per_label", type=int, default=150)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


# download_file, load_json and save_json are the same as in phase4/data_prep.py
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
    # Changing newlines and extra spaces to a single space
    # Train, dev and test all keep their capital letters, so the text is not lowercased
    return " ".join(text.split())


def make_chunks(token_ids, chunk_tokens, num_chunks):
    # Keeping only the first num_chunks * chunk_tokens tokens of the case,
    # the start of the case usually has the facts and the charged sections
    token_ids = token_ids[:num_chunks * chunk_tokens]

    # Cutting them into chunks, only the last chunk can be shorter than chunk_tokens
    chunks = []
    for start in range(0, len(token_ids), chunk_tokens):
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
        # Adding the statute ids for train and dev, without duplicates
        if "labels" in case:
            item["labels"] = sorted(set(case["labels"]))
        built.append(item)

    avg_chunks = sum(len(item["chunks"]) for item in built) / len(built)
    print(f"  {name}: {len(built)} cases, {avg_chunks:.2f} chunks per case on average")
    return built


def pick_train_subset(train, args):
    # First making sure every statute has at least min_per_label cases
    rng = random.Random(args.seed)
    full_counts = Counter(label for case in train for label in case["labels"])

    picked = set()
    subset_counts = Counter()

    # Going from the rarest statute to the most common one
    for label, _ in reversed(full_counts.most_common()):
        if subset_counts[label] >= args.min_per_label:
            continue

        label_cases = []
        for i, case in enumerate(train):
            if label in case["labels"] and i not in picked:
                label_cases.append(i)
        rng.shuffle(label_cases)

        num_to_add = args.min_per_label - subset_counts[label]
        for i in label_cases[:num_to_add]:
            picked.add(i)
            subset_counts.update(train[i]["labels"])

    print(f"  {len(picked)} cases picked to cover the rare statutes")

    # Filling the rest of the subset with random cases
    others = [i for i in range(len(train)) if i not in picked]
    rng.shuffle(others)
    num_to_add = args.train_cases - len(picked)
    if num_to_add > 0:
        picked.update(others[:num_to_add])

    subset = [train[i] for i in sorted(picked)]
    rng.shuffle(subset)
    return subset


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    print("Downloading Phase 5 files ...", flush=True)
    paths = {name: download_file(rel_path, args.cache_dir) for name, rel_path in FILES.items()}

    train = load_json(paths["train"])
    dev = load_json(paths["dev"])
    test = [{"id": case["id"], "text": case["text"]} for case in load_json(paths["test"])]

    print("Picking the train subset ...", flush=True)
    train_subset = pick_train_subset(train, args)

    print("Tokenizing and making chunks ...", flush=True)
    train_cases = build_cases(train_subset, tokenizer, args, "train")
    dev_cases = build_cases(dev, tokenizer, args, "dev")
    test_cases = build_cases(test, tokenizer, args, "test")

    save_json(train_cases, os.path.join(args.out_dir, "train_cases.json"))
    save_json(dev_cases, os.path.join(args.out_dir, "dev_cases.json"))
    save_json(test_cases, os.path.join(args.out_dir, "test_cases.json"))

    # Counting how many train cases each statute has, used for the class weights
    train_counts = Counter(label for case in train_cases for label in case["labels"])
    full_train_counts = Counter(label for case in train for label in case["labels"])
    meta = {
        "num_labels": NUM_LABELS,
        "num_train_cases": len(train_cases),
        "train_label_counts": [train_counts[label] for label in range(NUM_LABELS)],
        "full_train_label_counts": [full_train_counts[label] for label in range(NUM_LABELS)],
        "num_chunks": args.num_chunks,
        "chunk_tokens": args.chunk_tokens,
        "train_cases": args.train_cases,
        "min_per_label": args.min_per_label,
        "seed": args.seed,
    }
    save_json(meta, os.path.join(args.out_dir, "meta.json"))

    print(f"\nSaved to {args.out_dir}")
    print("Rarest statutes in the train subset:", train_counts.most_common()[-10:])
    print("Most common statutes in the train subset:", train_counts.most_common(10))


if __name__ == "__main__":
    main()
