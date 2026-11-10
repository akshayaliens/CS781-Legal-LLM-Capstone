"""
CS781 Capstone - Phase 4: Court Judgment Prediction (CJPE)
Prediction script

Loads a trained checkpoint from train.py and predicts 0 or 1 for every case.
With --split dev it prints the dev macro-F1, with --split test it writes
prediction.json and submission.zip for Codabench.

Usage (Colab):
    python predict.py --split test \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p4/decoder_full/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p4 \
        --out_dir /content/drive/MyDrive/CS781/submissions_p4/decoder_full
"""

import argparse
import json
import os
import zipfile

import torch
from sklearn.metrics import classification_report

from methods import setup_method
from model import ModelConfig, build_cjpe_model
from train import load_json, macro_f1, predict_cases


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--batch_size", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Building the same model as in training and loading the trained weights
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig(**ckpt["config"])
    train_args = ckpt["args"]

    model = build_cjpe_model(config)
    setup_method(model, train_args["method"], train_args["rank"],
                 train_args["alpha"], train_args["train_top_layers"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"Loaded {args.ckpt} (step {ckpt['step']}, dev macro-F1 {ckpt['best_dev_f1']:.4f})")

    cases = load_json(os.path.join(args.data_dir, f"{args.split}_cases.json"))
    predictions = predict_cases(model, cases, args.batch_size, config.pad_id, device)

    if args.split == "dev":
        # Printing the dev macro-F1 and the precision, recall and F1 of each label
        print(f"Dev macro-F1: {macro_f1(cases, predictions):.4f}")
        gold = [case["label"] for case in cases]
        print(classification_report(gold, predictions, digits=4))
        return

    # Saving the case id and the predicted label (0 or 1) of every test case
    prediction_json = [{"id": case["id"], "pred": int(pred)} for case, pred in zip(cases, predictions)]

    json_path = os.path.join(args.out_dir, "prediction.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(prediction_json, f)

    # Zipping prediction.json for Codabench
    zip_path = os.path.join(args.out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(json_path, arcname="prediction.json")

    print(f"{len(prediction_json)} test cases, predicted label 1 for {sum(predictions)} of them")
    print(f"Saved {zip_path}")


if __name__ == "__main__":
    main()
