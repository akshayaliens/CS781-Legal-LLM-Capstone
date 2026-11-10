"""
CS781 Capstone - Phase 7: Prior Case Retrieval (PCR)
Knowledge distillation

Trains a smaller student model to learn from both the relevant candidates and
the similarity scores of the trained decoder_full model (the teacher). The
student is the encoder half of the Phase-1 encdec model.

Usage (Colab):
    python distill.py \
        --teacher_ckpt /content/drive/MyDrive/CS781/ckpt_p7/decoder_full/best_checkpoint.pt \
        --p1_ckpt /content/drive/MyDrive/CS781/ckpt_p1/encdec/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p7 \
        --out_dir /content/drive/MyDrive/CS781/ckpt_p7/encdec_distill

References:
    - Knowledge distillation: Hinton, Vinyals and Dean, "Distilling the
      Knowledge in a Neural Network", 2015
"""

import argparse
import csv
import math
import os
import random
import time

import torch
import torch.nn.functional as F

from methods import setup_method
from model import ModelConfig, build_retrieval_model, contrastive_loss
from train import evaluate, load_json, make_batch, pick_candidates


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--teacher_ckpt", required=True)
    parser.add_argument("--p1_ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)

    # kd_alpha is the share of the loss from the relevant candidates, the rest is from the teacher
    parser.add_argument("--kd_alpha", type=float, default=0.5)
    parser.add_argument("--kd_temperature", type=float, default=2.0)

    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--accum_steps", type=int, default=8)
    parser.add_argument("--num_negatives", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
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


def get_scores(vectors, negative_vectors, temperature):
    # Scoring the relevant candidate and the negatives for the query
    # vectors[0] is the query and vectors[1] is the relevant candidate
    positive_score = (vectors[0] * vectors[1]).sum().unsqueeze(0)
    negative_scores = negative_vectors @ vectors[0]
    return torch.cat([positive_score, negative_scores]) / temperature


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    train_queries = load_json(os.path.join(args.data_dir, "train_queries.json"))
    train_candidates = load_json(os.path.join(args.data_dir, "train_candidates.json"))
    dev_queries = load_json(os.path.join(args.data_dir, "dev_queries.json"))
    dev_candidates = load_json(os.path.join(args.data_dir, "dev_candidates.json"))

    candidate_by_id = {}
    for candidate in train_candidates:
        candidate_by_id[candidate["id"]] = candidate

    # Loading the trained teacher and freezing it
    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
    teacher = build_retrieval_model(ModelConfig(**t_ckpt["config"]))
    setup_method(teacher, t_ckpt["args"]["method"])
    teacher.load_state_dict(t_ckpt["model"])
    teacher = teacher.to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    # Making the student from its Phase-1 checkpoint
    p1_ckpt = torch.load(args.p1_ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig(**p1_ckpt["config"])
    student = build_retrieval_model(config)
    missing, unexpected = student.backbone.load_state_dict(p1_ckpt["model"], strict=False)
    print(f"Student ({config.arch}) backbone loaded, missing keys: {len(missing)}, unused keys: {len(unexpected)}")
    student = student.to(device)

    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher: {teacher_params / 1e6:.1f}M parameters, student: {student_params / 1e6:.1f}M "
          f"parameters ({100 * student_params / teacher_params:.0f}% of the teacher)")

    steps_per_epoch = math.ceil(len(train_queries) / args.accum_steps)
    total_steps = steps_per_epoch * args.epochs
    print(f"{len(train_queries)} train queries, {args.accum_steps} queries per update, "
          f"{total_steps} steps total")

    backbone_params = []
    head_params = []
    for name, param in student.named_parameters():
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

    def distill_loss(student_scores, teacher_scores, gold_loss):
        # The student matches the teacher's softened scores over the same candidates
        T = args.kd_temperature
        student_log_probs = F.log_softmax(student_scores / T, dim=-1)
        teacher_probs = F.softmax(teacher_scores / T, dim=-1)
        teacher_loss = F.kl_div(student_log_probs, teacher_probs, reduction="sum")

        # Multiplying by T*T so this part is the same size as the gold loss
        return args.kd_alpha * gold_loss + (1 - args.kd_alpha) * teacher_loss * T * T

    def save_checkpoint(step, best_f1, path):
        # Saving the student like a full fine-tuned model, so predict.py can load it
        saved_args = vars(args).copy()
        saved_args["method"] = "full"
        saved_args["rank"] = t_ckpt["args"]["rank"]
        saved_args["alpha"] = t_ckpt["args"]["alpha"]
        saved_args["train_top_layers"] = t_ckpt["args"]["train_top_layers"]
        torch.save({
            "model": student.state_dict(),
            "config": config.__dict__,
            "step": step,
            "best_dev_f1": best_f1,
            "args": saved_args,
        }, path)

    log_path = os.path.join(args.out_dir, "training_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow(["step", "epoch", "train_loss", "dev_f1", "dev_precision", "dev_recall"])

    dev_f1, dev_precision, dev_recall = evaluate(
        student, dev_queries, dev_candidates, args.eval_batch_size, config.pad_id, device
    )
    print(f"  before training, dev Micro-F1@7: {dev_f1:.4f}  "
          f"precision {dev_precision:.4f}  recall {dev_recall:.4f}", flush=True)
    log_writer.writerow([0, 0, "", f"{dev_f1:.4f}", f"{dev_precision:.4f}", f"{dev_recall:.4f}"])
    log_file.flush()

    # Training loop, the teacher's scores are added to the loss
    train_rng = random.Random(args.seed)
    student.train()
    running_loss, running_count = 0.0, 0
    avg_loss = 0.0
    best_f1 = -1.0
    start_time = time.time()
    step = 0

    for epoch in range(args.epochs):
        order = list(range(len(train_queries)))
        train_rng.shuffle(order)

        for query_num, query_index in enumerate(order):
            query = train_queries[query_index]
            positive, negatives = pick_candidates(
                query, train_candidates, candidate_by_id, args.num_negatives, train_rng, None, 0
            )

            token_ids, token_mask = make_batch([query, positive], config.pad_id)
            neg_token_ids, neg_token_mask = make_batch(negatives, config.pad_id)
            token_ids = token_ids.to(device)
            token_mask = token_mask.to(device)
            neg_token_ids = neg_token_ids.to(device)
            neg_token_mask = neg_token_mask.to(device)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                vectors = student(token_ids, token_mask)

                with torch.no_grad():
                    negative_vectors = student(neg_token_ids, neg_token_mask)
                    teacher_vectors = teacher(token_ids, token_mask)
                    teacher_negatives = teacher(neg_token_ids, neg_token_mask)

                gold_loss = contrastive_loss(vectors[0], vectors[1], negative_vectors, args.temperature)
                student_scores = get_scores(vectors, negative_vectors, args.temperature)
                teacher_scores = get_scores(teacher_vectors, teacher_negatives, args.temperature)
                loss = distill_loss(student_scores, teacher_scores, gold_loss)

            scaler.scale(loss / args.accum_steps).backward()
            running_loss += loss.item()
            running_count += 1

            # Updating the weights only after every accum_steps queries, or at the end of the epoch
            if (query_num + 1) % args.accum_steps != 0 and query_num + 1 != len(order):
                continue

            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * lr_factor(step)

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
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
                    student, dev_queries, dev_candidates, args.eval_batch_size, config.pad_id, device
                )
                print(f"  dev Micro-F1@7: {dev_f1:.4f}  precision {dev_precision:.4f}  "
                      f"recall {dev_recall:.4f}", flush=True)

                log_writer.writerow([step, epoch + 1, f"{avg_loss:.4f}", f"{dev_f1:.4f}",
                                     f"{dev_precision:.4f}", f"{dev_recall:.4f}"])
                log_file.flush()

                if dev_f1 > best_f1:
                    best_f1 = dev_f1
                    save_checkpoint(step, best_f1, os.path.join(args.out_dir, "best_checkpoint.pt"))

    log_file.close()

    total_minutes = (time.time() - start_time) / 60
    print(f"\nTraining finished: {total_steps} steps, "
          f"best dev Micro-F1@7 {best_f1:.4f}, {total_minutes:.1f} minutes")


if __name__ == "__main__":
    main()
