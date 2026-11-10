"""
CS781 Capstone - Phase 5: Legal Statute Identification (LSI)
Prediction script

Loads a trained checkpoint from train.py or distill.py and predicts the
statute ids of every case.
With --split dev it uses the full dev set to pick one threshold for all
statutes and one threshold for each statute, prints the scores of both and
saves them to thresholds.json.
With --split test it uses the saved thresholds and writes prediction.json and
submission.zip for Codabench, so --split dev has to be run first.

Usage (Colab):
    python predict.py --split dev \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p5/decoder_full/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p5 \
        --out_dir /content/drive/MyDrive/CS781/submissions_p5/decoder_full
"""

import argparse
import csv
import json
import os
import zipfile

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from methods import setup_method
from model import ModelConfig, build_lsi_model
from train import best_global_threshold, gold_matrix, load_json, macro_f1, predict_probs, probs_to_preds


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--threshold_mode", default="per_label", choices=["global", "per_label"])
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def per_label_thresholds(probs, gold, global_threshold):
    # Picking the threshold with the best F1 separately for every statute
    thresholds = []
    for label in range(probs.shape[1]):
        best_threshold = global_threshold
        best_f1 = f1_score(gold[:, label], probs[:, label] >= global_threshold, zero_division=0)
        for threshold in np.arange(0.05, 0.96, 0.05):
            f1 = f1_score(gold[:, label], probs[:, label] >= threshold, zero_division=0)
            if f1 > best_f1:
                best_threshold, best_f1 = float(threshold), f1
        thresholds.append(best_threshold)
    return np.array(thresholds)


def print_scores(name, gold, preds):
    micro = f1_score(gold, preds, average="micro", zero_division=0)
    subset_acc = accuracy_score(gold, preds)
    print(f"{name}: macro-F1 {macro_f1(gold, preds):.4f}  micro-F1 {micro:.4f}  "
          f"subset accuracy {subset_acc:.4f}  statutes per case {preds.sum(1).mean():.2f}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Building the same model as in training and loading the trained weights
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig(**ckpt["config"])
    train_args = ckpt["args"]
    num_labels = ckpt["num_labels"]

    model = build_lsi_model(config, num_labels)
    setup_method(model, train_args["method"], train_args["rank"],
                 train_args["alpha"], train_args["train_top_layers"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"Loaded {args.ckpt} (step {ckpt['step']}, dev macro-F1 {ckpt['best_dev_f1']:.4f})")

    cases = load_json(os.path.join(args.data_dir, f"{args.split}_cases.json"))
    probs = predict_probs(model, cases, args.batch_size, config.pad_id, device)
    thresholds_path = os.path.join(args.out_dir, "thresholds.json")

    if args.split == "dev":
        gold = gold_matrix(cases, num_labels)

        # Comparing a fixed 0.5 threshold, the best global threshold and per-statute thresholds
        print_scores("threshold 0.5", gold, probs_to_preds(probs, 0.5))

        global_threshold, _ = best_global_threshold(probs, gold)
        print_scores(f"global threshold {global_threshold:.2f}", gold, probs_to_preds(probs, global_threshold))

        label_thresholds = per_label_thresholds(probs, gold, global_threshold)
        preds = probs_to_preds(probs, label_thresholds)
        print_scores("per-statute thresholds", gold, preds)

        with open(thresholds_path, "w") as f:
            json.dump({"global": global_threshold, "per_label": label_thresholds.tolist()}, f)

        label_f1 = f1_score(gold, preds, average=None, zero_division=0)
        with open(os.path.join(args.out_dir, "dev_label_f1.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["statute", "dev_cases", "f1", "threshold"])
            for label in range(num_labels):
                writer.writerow([label, int(gold[:, label].sum()), f"{label_f1[label]:.4f}",
                                 f"{label_thresholds[label]:.2f}"])

        order = np.argsort(label_f1)
        print("Worst 10 statutes (statute, dev cases, F1):")
        for label in order[:10]:
            print(f"  {label}  {int(gold[:, label].sum())}  {label_f1[label]:.3f}")
        print("Best 10 statutes (statute, dev cases, F1):")
        for label in order[-10:]:
            print(f"  {label}  {int(gold[:, label].sum())}  {label_f1[label]:.3f}")

        print(f"Saved {thresholds_path}")
        return

    with open(thresholds_path) as f:
        saved = json.load(f)
    if args.threshold_mode == "per_label":
        thresholds = np.array(saved["per_label"])
    else:
        thresholds = saved["global"]
    preds = probs_to_preds(probs, thresholds)

    # Saving the case id and the list of predicted statute ids (unique integers) of every test case
    prediction_json = []
    for case, row in zip(cases, preds):
        statute_ids = [int(label) for label in np.nonzero(row)[0]]
        prediction_json.append({"id": case["id"], "pred": statute_ids})

    json_path = os.path.join(args.out_dir, "prediction.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(prediction_json, f)

    # Zipping prediction.json for Codabench
    zip_path = os.path.join(args.out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(json_path, arcname="prediction.json")

    print(f"{len(prediction_json)} test cases, {preds.sum(1).mean():.2f} statutes per case on average "
          f"({args.threshold_mode} thresholds)")
    print(f"Saved {zip_path}")


if __name__ == "__main__":
    main()
