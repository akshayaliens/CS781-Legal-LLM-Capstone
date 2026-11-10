"""
CS781 Capstone - Phase 7: Prior Case Retrieval (PCR)
Data preparation

Joins the sentences of every case, tokenizes them and keeps the first 4 chunks
of 512 tokens. Every split keeps its own queries and its own candidate pool.

Usage (Colab):
    python data_prep.py --out_dir /content/drive/MyDrive/CS781/data_p7 \
        --tokenizer_path /content/drive/MyDrive/CS781/data_p1/tokenizer.json
"""

import argparse
import json
import os

DATASET_REPO = "Exploration-Lab/CS-781-Capstone"

SPLITS = {
    "train": "Train",
    "dev": "Dev",
    "test": "Test",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--cache_dir", default="/content/hf_cache_p7")
    parser.add_argument("--num_chunks", type=int, default=4)
    parser.add_argument("--chunk_tokens", type=int, default=512)
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


def clean_text(sentences):
    return " ".join(" ".join(sentences).split())


def make_chunks(token_ids, chunk_tokens, num_chunks):
    # Keeping only the first num_chunks * chunk_tokens tokens of the case
    token_ids = token_ids[:num_chunks * chunk_tokens]

    chunks = []
    for start in range(0, len(token_ids), chunk_tokens):
        chunks.append(token_ids[start:start + chunk_tokens])
    return chunks


def build_cases(cases, tokenizer, args, name):
    built = []
    skipped = 0

    for case_id in cases:
        case = cases[case_id]

        # A train query with no relevant candidates has nothing to learn from
        if name == "train queries" and len(case["relevant_candidates"]) == 0:
            skipped += 1
            continue

        token_ids = tokenizer.encode(" " + clean_text(case["text"])).ids
        # Giving an empty case one token (0 = <|endoftext|>) so it still has a chunk
        if len(token_ids) == 0:
            token_ids = [0]

        item = {
            "id": case_id,
            "chunks": make_chunks(token_ids, args.chunk_tokens, args.num_chunks),
        }
        if "relevant_candidates" in case:
            item["relevant"] = case["relevant_candidates"]
        built.append(item)

    print(f"  {name}: {len(built)} cases kept, {skipped} skipped", flush=True)
    return built


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    for split in SPLITS:
        print(f"Preparing {split} ...", flush=True)
        folder = "Phase_7_PCR/" + SPLITS[split] + "/"

        queries_path = download_file(folder + split + ".json", args.cache_dir)
        candidates_path = download_file(folder + split + "_candidates.json", args.cache_dir)

        queries = build_cases(load_json(queries_path), tokenizer, args, split + " queries")
        candidates = build_cases(load_json(candidates_path), tokenizer, args, split + " candidates")

        save_json(queries, os.path.join(args.out_dir, split + "_queries.json"))
        save_json(candidates, os.path.join(args.out_dir, split + "_candidates.json"))

    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
