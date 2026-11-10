"""
CS781 Capstone - Phase 3: Rhetorical Role Prediction (RR)
Data preparation

Tokenizes the CL and IT documents and packs the sentences of each document
into overlapping windows of up to 512 tokens.

Usage (Colab):
    python data_prep.py --out_dir /content/drive/MyDrive/CS781/data_p3 \
        --tokenizer_path /content/drive/MyDrive/CS781/data_p1/tokenizer.json
"""

import argparse
import json
import os
from collections import Counter

DATASET_REPO = "Exploration-Lab/CS-781-Capstone"

FILES = {
    "CL_train": "Phase_3_RR/Train/CL_train.json",
    "CL_dev": "Phase_3_RR/Train/CL_dev.json",
    "IT_train": "Phase_3_RR/Train/IT_train.json",
    "IT_dev": "Phase_3_RR/Train/IT_dev.json",
    "CL_test": "Phase_3_RR/Test/CL_test.json",
    "IT_test": "Phase_3_RR/Test/IT_test.json",
    "label_vocab": "Phase_3_RR/Train/label_vocab.json",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--cache_dir", default="/content/hf_cache_p3")
    parser.add_argument("--context", default="document", choices=["document", "sentence"])
    parser.add_argument("--window_tokens", type=int, default=512)
    parser.add_argument("--max_sentence_tokens", type=int, default=128)
    return parser.parse_args()


# download_file, load_json and save_json are the same as in phase2/data_prep.py
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


def load_test_documents(path):
    # Changing the test file to the same list format as train and dev
    test_data = load_json(path)
    return [{"id": doc_id, "text": sentences} for doc_id, sentences in test_data.items()]


def tokenize_sentences(sentences, tokenizer, max_sentence_tokens):
    # Tokenizing each sentence and cutting it to max_sentence_tokens
    sentence_tokens = []
    for sentence in sentences:
        token_ids = tokenizer.encode(" " + sentence).ids
        sentence_tokens.append(token_ids[:max_sentence_tokens])
    return sentence_tokens


def make_windows(sentence_tokens, window_tokens, context):
    # Finding the start and end sentence of every window
    num_sentences = len(sentence_tokens)

    if context == "sentence":
        return [(i, i + 1) for i in range(num_sentences)]

    windows = []
    start = 0
    while start < num_sentences:
        # Adding whole sentences until the next one would not fit
        end = start
        total_tokens = 0
        while end < num_sentences and total_tokens + len(sentence_tokens[end]) <= window_tokens:
            total_tokens += len(sentence_tokens[end])
            end += 1
        windows.append((start, end))

        if end == num_sentences:
            break

        # Moving forward by half a window
        start += max(1, (end - start) // 2)

    return windows


def build_windows(documents, domain, tokenizer, label_to_id, args):
    windows = []

    for doc in documents:
        sentence_tokens = tokenize_sentences(doc["text"], tokenizer, args.max_sentence_tokens)

        for start, end in make_windows(sentence_tokens, args.window_tokens, args.context):
            # Joining the sentences and saving where each one starts and ends
            token_ids = []
            sentence_spans = []
            for i in range(start, end):
                sentence_spans.append([len(token_ids), len(token_ids) + len(sentence_tokens[i])])
                token_ids.extend(sentence_tokens[i])

            window = {
                "doc_id": doc["id"],
                "domain": domain,
                "token_ids": token_ids,
                "sentence_spans": sentence_spans,
                "sentence_ids": list(range(start, end)),
            }

            # Adding labels for train and dev
            if "labels" in doc:
                window["label_ids"] = [label_to_id[label] for label in doc["labels"][start:end]]

            windows.append(window)

    print(f"  {domain}: {len(documents)} documents, {len(windows)} windows")
    return windows


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    print("Downloading Phase 3 files ...", flush=True)
    paths = {name: download_file(rel_path, args.cache_dir) for name, rel_path in FILES.items()}

    # Loading the labels
    label_to_id = load_json(paths["label_vocab"])
    id_to_label = sorted(label_to_id, key=label_to_id.get)
    print("Labels:", id_to_label)

    cl_train = load_json(paths["CL_train"])
    it_train = load_json(paths["IT_train"])
    cl_dev = load_json(paths["CL_dev"])
    it_dev = load_json(paths["IT_dev"])
    cl_test = load_test_documents(paths["CL_test"])
    it_test = load_test_documents(paths["IT_test"])

    print(f"Building windows ({args.context} context) ...", flush=True)
    train_windows = (build_windows(cl_train, "CL", tokenizer, label_to_id, args)
                     + build_windows(it_train, "IT", tokenizer, label_to_id, args))
    dev_windows = (build_windows(cl_dev, "CL", tokenizer, label_to_id, args)
                   + build_windows(it_dev, "IT", tokenizer, label_to_id, args))
    test_windows = (build_windows(cl_test, "CL", tokenizer, label_to_id, args)
                    + build_windows(it_test, "IT", tokenizer, label_to_id, args))

    # Saving the dev labels for scoring
    dev_docs = []
    for domain, docs in [("CL", cl_dev), ("IT", it_dev)]:
        for doc in docs:
            dev_docs.append({"id": doc["id"], "domain": domain, "labels": doc["labels"]})

    # Saving the number of sentences in each test document
    test_docs = []
    for domain, docs in [("CL", cl_test), ("IT", it_test)]:
        for doc in docs:
            test_docs.append({"id": doc["id"], "domain": domain, "num_sentences": len(doc["text"])})

    # Counting each label in train
    train_counts = Counter(label for doc in cl_train + it_train for label in doc["labels"])

    save_json(train_windows, os.path.join(args.out_dir, "train_windows.json"))
    save_json(dev_windows, os.path.join(args.out_dir, "dev_windows.json"))
    save_json(test_windows, os.path.join(args.out_dir, "test_windows.json"))
    save_json(dev_docs, os.path.join(args.out_dir, "dev_docs.json"))
    save_json(test_docs, os.path.join(args.out_dir, "test_docs.json"))

    meta = {
        "id_to_label": id_to_label,
        "train_label_counts": [train_counts[label] for label in id_to_label],
        "context": args.context,
        "window_tokens": args.window_tokens,
        "max_sentence_tokens": args.max_sentence_tokens,
    }
    save_json(meta, os.path.join(args.out_dir, "meta.json"))

    print(f"\nSaved to {args.out_dir}")
    print("Train label counts:", dict(zip(id_to_label, meta["train_label_counts"])))


if __name__ == "__main__":
    main()
