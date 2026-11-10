"""
CS781 Capstone - Phase 2: Legal Named Entity Recognition (L-NER)
Data preparation

Fine-tuning the Phase-1 models for NER, so this builds the training data
for that: loads fold1/fold2, runs the Phase-1 tokenizer on each document,
cuts the tokens into overlapping 512-length chunks (docs are way longer
than 512 tokens, median was about 26600 characters when checked in
01_data_exploration.ipynb), and turns the gold spans into a BIO tag per
token.

Usage (Colab):
    python data_prep.py --out_dir /content/drive/MyDrive/CS781/data_p2 \
        --tokenizer_path /content/drive/MyDrive/CS781/data_p1/tokenizer.json

Checked all 8393 gold spans in fold1+fold2 against the raw text
(text[start:end] == entity_text) - 0 mismatches, so the gold offsets are
fine as given. Getting a span back from token offsets needed a fix though:
the tokenizer keeps the space before a word in the same token (" STATE"),
so the raw offset starts 1 char too early - trimming that leading space
fixed 8237 of the 8393. The other 156 stay wrong no matter what, because
the gold boundary sometimes cuts through the middle of one token instead
of landing between two tokens (gold tags "Board", tokenizer only has
"Boards" as a single token) - noting this for the report since it's a
tokenizer limit, not something to fix in code.

References:
    - BIO tagging: Ramshaw and Marcus, "Text Chunking Using
      Transformation-Based Learning", 1995
    - tokenizers library offset mapping: https://github.com/huggingface/tokenizers
"""

import argparse
import json
import os

DATASET_REPO = "Exploration-Lab/CS-781-Capstone"

TRAIN_FILES = {
    "fold1_text": "Phase_2_LNER/Train/fold1-text.json",
    "fold1_gold": "Phase_2_LNER/Train/fold1-gold.json",
    "fold2_text": "Phase_2_LNER/Train/fold2-text.json",
    "fold2_gold": "Phase_2_LNER/Train/fold2-gold.json",
    "labels": "Phase_2_LNER/Train/labels.txt",
}

TEST_FILE = "Phase_2_LNER/Test/fold3-text.json"

# 512 to match the Phase-1 block_size. Overlap so an entity split across
# a chunk boundary still shows up whole in the next chunk.
CHUNK_TOKENS = 512
CHUNK_OVERLAP = 64


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cache_dir", default="/content/hf_cache_p2")
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--chunk_tokens", type=int, default=CHUNK_TOKENS)
    parser.add_argument("--chunk_overlap", type=int, default=CHUNK_OVERLAP)
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


def recover_span(doc_text, offsets, token_indices):
    # Getting the char span back from a run of tokens, then trimming off a
    # leading space - the tokenizer puts the space before a word in the same
    # token (" STATE"), so without this the span starts 1 character too early
    start = offsets[token_indices[0]][0]
    end = offsets[token_indices[-1]][1]

    while start < end and doc_text[start] == " ":
        start += 1

    return start, end


def make_bio_labels(doc_text, offsets, spans):
    # Giving every token a B-LABEL / I-LABEL / O tag based on which gold
    # span (if any) it falls inside. Offset (0, 0) means a special token,
    # those just stay O
    labels = ["O"] * len(offsets)

    for start, end, label, entity_text in spans:
        first_token = True
        for i, (tok_start, tok_end) in enumerate(offsets):
            if tok_start == 0 and tok_end == 0:
                continue
            if tok_start < end and tok_end > start:
                labels[i] = ("B-" if first_token else "I-") + label
                first_token = False

    return labels


def chunk_document(doc_id, doc_text, tokens, offsets, labels, chunk_tokens, chunk_overlap):
    # Cutting one document's tokens into overlapping windows so nothing
    # bigger than chunk_tokens ever has to go through the model at once.
    # Keeping each chunk's own char range too, needed later to stitch
    # predictions back onto the full document
    chunks = []
    step = chunk_tokens - chunk_overlap
    num_tokens = len(tokens)

    start_tok = 0
    while start_tok < num_tokens:
        end_tok = min(start_tok + chunk_tokens, num_tokens)

        chunk_offsets = offsets[start_tok:end_tok]
        chunks.append({
            "doc_id": doc_id,
            "token_ids": tokens[start_tok:end_tok],
            "labels": labels[start_tok:end_tok],
            "chunk_char_start": chunk_offsets[0][0],
            "chunk_char_end": chunk_offsets[-1][1],
        })

        if end_tok == num_tokens:
            break
        start_tok += step

    return chunks


def build_examples(text_data, gold_data, tokenizer, chunk_tokens, chunk_overlap):
    examples = []
    mismatch_count = 0
    total_span_count = 0

    for doc_id, doc_text in text_data.items():
        spans = gold_data.get(doc_id, [])
        encoding = tokenizer.encode(doc_text)
        offsets = list(encoding.offsets)

        # same offset check as in 01_data_exploration.ipynb, just in case
        for start, end, label, entity_text in spans:
            total_span_count += 1
            if doc_text[start:end] != entity_text:
                mismatch_count += 1

        labels = make_bio_labels(doc_text, offsets, spans)
        doc_chunks = chunk_document(
            doc_id, doc_text, encoding.ids, offsets, labels,
            chunk_tokens, chunk_overlap
        )
        examples.extend(doc_chunks)

    print(f"  {len(text_data)} documents -> {len(examples)} chunks "
          f"({total_span_count} gold spans, {mismatch_count} offset mismatches)")

    return examples


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    print("Downloading Phase 2 files ...", flush=True)
    paths = {name: download_file(rel_path, args.cache_dir) for name, rel_path in TRAIN_FILES.items()}
    test_path = download_file(TEST_FILE, args.cache_dir)

    with open(paths["labels"], encoding="utf-8") as f:
        label_names = [line.strip() for line in f if line.strip()]
    print("Labels:", label_names)

    text1 = load_json(paths["fold1_text"])
    gold1 = load_json(paths["fold1_gold"])
    text2 = load_json(paths["fold2_text"])
    gold2 = load_json(paths["fold2_gold"])
    test_text = load_json(test_path)

    print("Building training chunks from fold1 ...", flush=True)
    fold1_examples = build_examples(text1, gold1, tokenizer, args.chunk_tokens, args.chunk_overlap)

    print("Building training chunks from fold2 ...", flush=True)
    fold2_examples = build_examples(text2, gold2, tokenizer, args.chunk_tokens, args.chunk_overlap)

    # "O" is id 0, so anything padded/ignored can just default to it
    label_vocab = ["O"]
    for name in label_names:
        label_vocab.append("B-" + name)
        label_vocab.append("I-" + name)
    label_to_id = {label: i for i, label in enumerate(label_vocab)}

    def encode_labels(examples):
        for ex in examples:
            ex["label_ids"] = [label_to_id[l] for l in ex["labels"]]
            del ex["labels"]
        return examples

    fold1_examples = encode_labels(fold1_examples)
    fold2_examples = encode_labels(fold2_examples)

    # using fold1/fold2 as train/dev against each other - fold3 has no gold
    # labels, it's the real blind test set
    save_json(fold1_examples, os.path.join(args.out_dir, "fold1_examples.json"))
    save_json(fold2_examples, os.path.join(args.out_dir, "fold2_examples.json"))

    meta = {
        "label_vocab": label_vocab,
        "chunk_tokens": args.chunk_tokens,
        "chunk_overlap": args.chunk_overlap,
        "num_fold1_chunks": len(fold1_examples),
        "num_fold2_chunks": len(fold2_examples),
    }
    save_json(meta, os.path.join(args.out_dir, "meta.json"))

    # not chunking the test docs here, predict.py does that at prediction time
    save_json(test_text, os.path.join(args.out_dir, "test_text.json"))

    print(f"\nSaved to {args.out_dir}")
    print(f"Label vocab size: {len(label_vocab)}")


if __name__ == "__main__":
    main()
