"""
CS781 Capstone - Phase 7: Prior Case Retrieval (PCR)
Training script

Loads a Phase-1 checkpoint and trains it so that a query case gets close to its
relevant candidates and far from random candidates of the train pool. The
checkpoint with the best dev Micro-F1@7 is saved.

Usage (Colab):
    python train.py --arch decoder --method full \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p1/decoder/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p7 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p7/decoder_full
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
from model import ModelConfig, build_retrieval_model, contrastive_loss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--arch", required=True, choices=["decoder", "encdec", "moe", "convmix"])
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--method", default="full", choices=["full", "partial", "lorma"])
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--train_top_layers", type=int, default=4)

    parser.add_argument("--epochs", type=int, default=2)
    # Number of queries added up before each update of the weights
    parser.add_argument("--accum_steps", type=int, default=8)
    # Number of random wrong candidates for every query
    parser.add_argument("--num_negatives", type=int, default=8)
    # How many of those are hard negatives, picked from the model's top wrong candidates
    parser.add_argument("--num_hard", type=int, default=0)
    parser.add_argument("--hard_pool", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    # proj layer is new, so it gets a bigger learning rate
    parser.add_argument("--head_learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def make_batch(cases, pad_id):
    # Padding the cases so they all have the same number of chunks and tokens
    max_chunks = 0
    max_tokens = 0
    for case in cases:
        max_chunks = max(max_chunks, len(case["chunks"]))
        for chunk in case["chunks"]:
            max_tokens = max(max_tokens, len(chunk))

    token_ids = torch.full((len(cases), max_chunks, max_tokens), pad_id, dtype=torch.long)
    token_mask = torch.zeros(len(cases), max_chunks, max_tokens)

    for i, case in enumerate(cases):
        for j, chunk in enumerate(case["chunks"]):
            token_ids[i, j, :len(chunk)] = torch.tensor(chunk)
            token_mask[i, j, :len(chunk)] = 1

    return token_ids, token_mask


def pick_candidates(query, candidates, candidate_by_id, num_negatives, rng, hard_ids, num_hard):
    # Picking one relevant candidate and some wrong candidates for the query
    positive = candidate_by_id[rng.choice(query["relevant"])]

    negatives = []
    used_ids = set(query["relevant"])
    used_ids.add(query["id"])

    # Hard negatives first, the rest of the negatives are random
    if num_hard > 0:
        for candidate_id in rng.sample(hard_ids, num_hard):
            negatives.append(candidate_by_id[candidate_id])
            used_ids.add(candidate_id)

    while len(negatives) < num_negatives:
        candidate = rng.choice(candidates)
        if candidate["id"] not in used_ids:
            negatives.append(candidate)
            used_ids.add(candidate["id"])

    return positive, negatives


@torch.no_grad()
def encode_cases(model, cases, batch_size, pad_id, device):
    # Getting the vector of every case
    model.eval()
    all_vectors = []

    for start in range(0, len(cases), batch_size):
        token_ids, token_mask = make_batch(cases[start:start + batch_size], pad_id)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
            vectors = model(token_ids.to(device), token_mask.to(device))
        all_vectors.append(vectors.float().cpu())

    model.train()
    return torch.cat(all_vectors)


def find_hard_negatives(model, queries, candidates, batch_size, pad_id, device, hard_pool):
    # Finding the wrong candidates that the model ranks highest for every query
    query_vectors = encode_cases(model, queries, batch_size, pad_id, device)
    candidate_vectors = encode_cases(model, candidates, batch_size, pad_id, device)
    order = (query_vectors @ candidate_vectors.t()).argsort(dim=1, descending=True).tolist()

    hard_negatives = {}
    for i, query in enumerate(queries):
        wrong_ids = []
        for position in order[i]:
            candidate_id = candidates[position]["id"]
            if candidate_id not in query["relevant"] and candidate_id != query["id"]:
                wrong_ids.append(candidate_id)
            if len(wrong_ids) == hard_pool:
                break
        hard_negatives[query["id"]] = wrong_ids
    return hard_negatives


def rank_candidates(query_vectors, candidate_vectors, queries, candidates, top_k=7):
    # Scoring every candidate for every query and keeping the top 7 ids
    scores = query_vectors @ candidate_vectors.t()

    position_by_id = {}
    for position, candidate in enumerate(candidates):
        position_by_id[candidate["id"]] = position

    # A query should not retrieve itself if it is also in the pool
    for i, query in enumerate(queries):
        if query["id"] in position_by_id:
            scores[i, position_by_id[query["id"]]] = -2.0

    top_positions = scores.topk(top_k, dim=1).indices.tolist()

    ranked = {}
    for i, query in enumerate(queries):
        ranked_ids = []
        for position in top_positions[i]:
            ranked_ids.append(candidates[position]["id"])
        ranked[query["id"]] = ranked_ids
    return ranked


def micro_scores(ranked, queries):
    # Micro: adding up the hits of all queries first, then computing the scores
    hits = 0
    total_predicted = 0
    total_gold = 0
    for query in queries:
        gold_ids = set(query["relevant"])
        total_gold += len(gold_ids)
        total_predicted += len(ranked[query["id"]])
        for candidate_id in ranked[query["id"]]:
            if candidate_id in gold_ids:
                hits += 1

    precision = hits / total_predicted
    recall = hits / total_gold
    if hits == 0:
        return 0.0, 0.0, 0.0
    return 2 * precision * recall / (precision + recall), precision, recall


def evaluate(model, queries, candidates, batch_size, pad_id, device):
    query_vectors = encode_cases(model, queries, batch_size, pad_id, device)
    candidate_vectors = encode_cases(model, candidates, batch_size, pad_id, device)
    ranked = rank_candidates(query_vectors, candidate_vectors, queries, candidates)
    return micro_scores(ranked, queries)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Loading what data_prep.py saved, every split keeps its own candidate pool
    train_queries = load_json(os.path.join(args.data_dir, "train_queries.json"))
    train_candidates = load_json(os.path.join(args.data_dir, "train_candidates.json"))
    dev_queries = load_json(os.path.join(args.data_dir, "dev_queries.json"))
    dev_candidates = load_json(os.path.join(args.data_dir, "dev_candidates.json"))

    candidate_by_id = {}
    for candidate in train_candidates:
        candidate_by_id[candidate["id"]] = candidate

    # One step is one update of the weights, made from accum_steps queries
    steps_per_epoch = math.ceil(len(train_queries) / args.accum_steps)
    total_steps = steps_per_epoch * args.epochs
    print(f"{len(train_queries)} train queries, {args.accum_steps} queries per update, "
          f"{total_steps} steps total")

    # Building the model with the same settings the Phase-1 checkpoint used
    print(f"Loading Phase-1 checkpoint: {args.ckpt}")
    p1_ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig(**p1_ckpt["config"])
    model = build_retrieval_model(config)

    missing, unexpected = model.backbone.load_state_dict(p1_ckpt["model"], strict=False)
    print(f"Backbone loaded, missing keys: {len(missing)}, unused keys: {len(unexpected)}")

    # Adding the method after loading the Phase-1 weights
    setup_method(model, args.method, args.rank, args.alpha, args.train_top_layers)
    model = model.to(device)

    trainable, total = count_trainable(model)
    print(f"Method: {args.method}, training {trainable / 1e6:.2f}M of {total / 1e6:.1f}M "
          f"parameters ({100 * trainable / total:.2f}%)")

    # Separate learning rates for the backbone and the proj layer
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.startswith("backbone."):
                backbone_params.append(param)
            else:
                head_params.append(param)

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.learning_rate, "base_lr": args.learning_rate},
         {"params": head_params, "lr": args.head_learning_rate, "base_lr": args.head_learning_rate}],
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

    def save_checkpoint(step, best_f1, path):
        torch.save({
            "model": model.state_dict(),
            "config": config.__dict__,
            "step": step,
            "best_dev_f1": best_f1,
            "args": vars(args),
        }, path)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "train_loss", "dev_f1", "dev_precision", "dev_recall"])

    # Checking the dev score before any training, to see what Phase 1 gives on its own
    dev_f1, dev_precision, dev_recall = evaluate(
        model, dev_queries, dev_candidates, args.eval_batch_size, config.pad_id, device
    )
    print(f"  before training, dev Micro-F1@7: {dev_f1:.4f}  "
          f"precision {dev_precision:.4f}  recall {dev_recall:.4f}", flush=True)
    log_writer.writerow([0, 0, "", f"{dev_f1:.4f}", f"{dev_precision:.4f}", f"{dev_recall:.4f}"])
    log_file.flush()

    # Training loop
    train_rng = random.Random(args.seed)
    model.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    best_f1 = -1.0
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        # Shuffling the train queries again at the start of every epoch
        order = list(range(len(train_queries)))
        train_rng.shuffle(order)

        # Finding the hard negatives again, because the model changes every epoch
        hard_negatives = {}
        if args.num_hard > 0:
            print("  finding hard negatives ...", flush=True)
            hard_negatives = find_hard_negatives(
                model, train_queries, train_candidates, args.eval_batch_size,
                config.pad_id, device, args.hard_pool
            )

        for query_num, query_index in enumerate(order):
            query = train_queries[query_index]
            positive, negatives = pick_candidates(
                query, train_candidates, candidate_by_id, args.num_negatives, train_rng,
                hard_negatives.get(query["id"]), args.num_hard
            )

            token_ids, token_mask = make_batch([query, positive], config.pad_id)
            neg_token_ids, neg_token_mask = make_batch(negatives, config.pad_id)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                vectors = model(token_ids.to(device), token_mask.to(device))

                # Only 2 cases with gradients fit on the GPU, so the negatives run without gradients
                with torch.no_grad():
                    negative_vectors = model(neg_token_ids.to(device), neg_token_mask.to(device))

                loss = contrastive_loss(vectors[0], vectors[1], negative_vectors, args.temperature)

            # Dividing by accum_steps so the added up gradients are an average
            scaler.scale(loss / args.accum_steps).backward()
            running_loss += loss.item()
            running_count += 1

            # Updating the weights only after every accum_steps queries, or at the end of the epoch
            if (query_num + 1) % args.accum_steps != 0 and query_num + 1 != len(order):
                continue

            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_factor(step)

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
                dev_f1, dev_precision, dev_recall = evaluate(
                    model, dev_queries, dev_candidates, args.eval_batch_size, config.pad_id, device
                )
                print(f"  dev Micro-F1@7: {dev_f1:.4f}  precision {dev_precision:.4f}  "
                      f"recall {dev_recall:.4f}", flush=True)

                log_writer.writerow([step, epoch + 1, f"{avg_loss:.4f}", f"{dev_f1:.4f}",
                                     f"{dev_precision:.4f}", f"{dev_recall:.4f}"])
                log_file.flush()

                # Saving only when the dev Micro-F1@7 gets better
                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    save_checkpoint(step, best_f1, os.path.join(args.out_dir, "best_checkpoint.pt"))

    log_file.close()

    total_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished: {total_steps} steps, "
          f"best dev Micro-F1@7 {best_f1:.4f}, {total_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
