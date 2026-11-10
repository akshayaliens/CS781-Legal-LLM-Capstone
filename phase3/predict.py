"""
CS781 Capstone - Phase 3: Rhetorical Role Prediction (RR)
Prediction script

Loads a trained checkpoint from train.py and predicts one label for every
sentence. With --split dev it prints the dev macro-F1, with --split test it
writes prediction.json and submission.zip for Codabench.

Usage (Colab):
    python predict.py --split test \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p3/decoder_full/best_checkpoint.pt \
        --data_dir /content/drive/MyDrive/CS781/data_p3 \
        --out_dir /content/drive/MyDrive/CS781/submissions_p3/decoder_full
"""

import argparse
import json
import os
import zipfile
from collections import Counter

import torch
from sklearn.metrics import classification_report

from methods import setup_method
from model import ModelConfig, build_rr_model
from train import load_json, macro_f1, predict_documents


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--split", default="test", choices=["dev", "test"])
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Building the same model as in training and loading the trained weights
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ModelConfig(**ckpt["config"])
    id_to_label = ckpt["id_to_label"]
    train_args = ckpt["args"]

    model = build_rr_model(config, len(id_to_label))
    # total_steps is only needed to build the adalora layers
    setup_method(model, train_args["method"], train_args["lora_rank"],
                 train_args["lora_alpha"], total_steps=ckpt["step"])
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    print(f"Loaded {args.ckpt} (step {ckpt['step']}, dev macro-F1 {ckpt['best_dev_f1']:.4f})")

    windows = load_json(os.path.join(args.data_dir, f"{args.split}_windows.json"))
    docs = load_json(os.path.join(args.data_dir, f"{args.split}_docs.json"))

    if args.split == "dev":
        num_sentences = {doc["id"]: len(doc["labels"]) for doc in docs}
    else:
        num_sentences = {doc["id"]: doc["num_sentences"] for doc in docs}

    predictions = predict_documents(model, windows, num_sentences, len(id_to_label),
                                    args.batch_size, config.pad_id, device)

    if args.split == "dev":
        # Printing the dev macro-F1 for all, CL and IT, and the F1 of each label
        label_to_id = {label: i for i, label in enumerate(id_to_label)}
        print(f"Dev macro-F1: {macro_f1(docs, predictions, label_to_id):.4f}")
        print(f"  CL: {macro_f1(docs, predictions, label_to_id, 'CL'):.4f}")
        print(f"  IT: {macro_f1(docs, predictions, label_to_id, 'IT'):.4f}")

        gold = [label for doc in docs for label in doc["labels"]]
        pred = [id_to_label[i] for doc in docs for i in predictions[doc["id"]]]
        print(classification_report(gold, pred, zero_division=0))
        return

    # Writing one label name per sentence for every test document
    prediction_json = {}
    for doc in docs:
        prediction_json[doc["id"]] = [id_to_label[i] for i in predictions[doc["id"]]]

    json_path = os.path.join(args.out_dir, "prediction.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(prediction_json, f, ensure_ascii=False)

    # Zipping prediction.json for Codabench
    zip_path = os.path.join(args.out_dir, "submission.zip")
    with zipfile.ZipFile(zip_path, "w") as z:
        z.write(json_path, arcname="prediction.json")

    all_labels = Counter(label for labels in prediction_json.values() for label in labels)
    print(f"{len(prediction_json)} test documents, {sum(all_labels.values())} sentences")
    print("Predicted labels:", all_labels.most_common())
    print(f"Saved {zip_path}")


if __name__ == "__main__":
    main()
