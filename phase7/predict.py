"""
CS781 Capstone - Phase 7: Prior Case Retrieval (PCR)
Prediction script

Loads a trained checkpoint and ranks the candidates of every query, using only
the candidate pool of the same split.
With --split dev it prints the dev scores and saves dev_ranked.json.
With --split test it writes prediction.json and submission.zip for Codabench.

Usage (Colab):
    python predict.py --split test \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p7/decoder_full/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p7 \
        --out_dir /content/drive/MyDrive/CS781/submissions_p7/decoder_full
"""

import argparse
import json
import os
import zipfile

import torch

from methods import setup_method
from model import ModelConfig, build_retrieval_model
from train import encode_cases, load_json, micro_scores, rank_candidates


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def count_bad_queries(ranked, candidates):
    # Every query needs exactly 7 different ids from its own candidate pool
    pool_ids = set()
    for candidate in candidates:
        pool_ids.add(candidate["id"])

    bad_queries = 0
    for query_id in ranked:
        ranked_ids = ranked[query_id]
        if len(ranked_ids) != 7 or len(set(ranked_ids)) != 7 or not set(ranked_ids).issubset(pool_ids):
            bad_queries += 1
    return bad_queries


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Building the same model as in training and loading the trained weights
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig(**ckpt["config"])
    train_args = ckpt["args"]

    model = build_retrieval_model(config)
    setup_method(model, train_args["method"], train_args["rank"],
                 train_args["alpha"], train_args["train_top_layers"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"Loaded {args.ckpt} (step {ckpt['step']}, dev Micro-F1@7 {ckpt['best_dev_f1']:.4f})")

    queries = load_json(os.path.join(args.data_dir, args.split + "_queries.json"))
    candidates = load_json(os.path.join(args.data_dir, args.split + "_candidates.json"))

    query_vectors = encode_cases(model, queries, args.batch_size, config.pad_id, device)
    candidate_vectors = encode_cases(model, candidates, args.batch_size, config.pad_id, device)
    ranked = rank_candidates(query_vectors, candidate_vectors, queries, candidates)

    print(f"{len(ranked)} queries, {count_bad_queries(ranked, candidates)} with a bad list")

    if args.split == "dev":
        f1, precision, recall = micro_scores(ranked, queries)
        print(f"dev Micro-F1@7: {f1:.4f}  precision {precision:.4f}  recall {recall:.4f}")

        dev_path = os.path.join(args.out_dir, "dev_ranked.json")
        with open(dev_path, "w", encoding="utf-8") as f:
            json.dump(ranked, f)
        print(f"Saved {dev_path}")
        return

    json_path = os.path.join(args.out_dir, "prediction.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ranked, f)

    # Zipping prediction.json for Codabench
    zip_path = os.path.join(args.out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(json_path, arcname="prediction.json")

    print(f"Saved {zip_path}")


if __name__ == "__main__":
    main()
