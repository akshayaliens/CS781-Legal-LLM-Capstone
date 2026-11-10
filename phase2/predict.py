"""
CS781 Capstone - Phase 2: Legal Named Entity Recognition (L-NER)
Prediction script

Loads a fine-tuned checkpoint from train.py, tags every token in each
document, turns the B-/I- tags back into character spans and writes
prediction.json + submission.zip.

With --gold_file (fold2), it also prints strict macro F1 locally.

Usage (Colab):
    python predict.py \
        --ckpt /content/drive/MyDrive/CS781/ckpt_p2/decoder_full/best_checkpoint.pt \
        --tokenizer_path /content/drive/MyDrive/CS781/data_p1/tokenizer.json \
        --text_file /content/drive/MyDrive/CS781/data_p2/test_text.json \
        --out_dir /content/drive/MyDrive/CS781/submissions_p2/decoder_full
"""

import argparse
import json
import os
import zipfile
from collections import Counter

import torch

from methods import setup_method
from model import ModelConfig, build_ner_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer_path", required=True)
    parser.add_argument("--text_file", required=True)
    parser.add_argument("--gold_file", default=None)
    parser.add_argument("--out_dir", required=True)

    # same chunk size and overlap as data_prep.py
    parser.add_argument("--chunk_tokens", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=64)

    # simple: the first version (used for the first decoder_full submission)
    # merge: stops one word from being split into two entities
    parser.add_argument("--decoding", default="merge", choices=["simple", "merge"])

    return parser.parse_args()


def chunk_starts(num_tokens, chunk_tokens, chunk_overlap):
    starts = []
    start = 0
    while True:
        starts.append(start)
        if start + chunk_tokens >= num_tokens:
            break
        start += chunk_tokens - chunk_overlap
    return starts


@torch.no_grad()
def tag_document(model, token_ids, num_labels, args, device):
    num_tokens = len(token_ids)

    # Some test documents are empty, nothing to tag in them
    if num_tokens == 0:
        return []

    # Tokens in the overlap get scores from 2 chunks, so averaging them
    score_sum = torch.zeros(num_tokens, num_labels)
    seen_count = torch.zeros(num_tokens, 1)

    for start in chunk_starts(num_tokens, args.chunk_tokens, args.chunk_overlap):
        end = min(start + args.chunk_tokens, num_tokens)
        chunk = torch.tensor([token_ids[start:end]], dtype=torch.long, device=device)

        logits, _ = model(chunk)
        score_sum[start:end] += logits[0].float().cpu()
        seen_count[start:end] += 1

    avg_scores = score_sum / seen_count
    return avg_scores.argmax(dim=-1).tolist()


def make_span(token_indices, label, offsets, doc_text):
    start = offsets[token_indices[0]][0]
    end = offsets[token_indices[-1]][1]

    # Trimming spaces at both ends (the leading space fix from data_prep.py)
    while start < end and doc_text[start].isspace():
        start += 1
    while end > start and doc_text[end - 1].isspace():
        end -= 1

    return [start, end, label, doc_text[start:end]]


def finish_span(token_indices, token_labels, offsets, doc_text):
    # Giving the span the label most of its tokens had (a tie goes to the first one)
    label = Counter(token_labels).most_common(1)[0][0]
    return make_span(token_indices, label, offsets, doc_text)


def tags_to_spans(tag_ids, label_vocab, offsets, doc_text, decoding):
    spans = []
    current_tokens = []
    current_labels = []

    for i, tag_id in enumerate(tag_ids):
        tag = label_vocab[tag_id]

        if tag == "O":
            if current_tokens:
                spans.append(finish_span(current_tokens, current_labels, offsets, doc_text))
            current_tokens = []
            current_labels = []
            continue

        prefix, label = tag.split("-", 1)

        # Deciding if this token continues the current span or starts a new one
        if not current_tokens:
            keep_going = False
        elif decoding == "simple":
            keep_going = (prefix == "I" and label == current_labels[-1])
        else:
            # no gap between this token and the last one, so it's the same word
            same_word = (offsets[i][0] == offsets[i - 1][1])
            keep_going = (same_word or prefix == "I")

        if keep_going:
            current_tokens.append(i)
            current_labels.append(label)
        else:
            if current_tokens:
                spans.append(finish_span(current_tokens, current_labels, offsets, doc_text))
            current_tokens = [i]
            current_labels = [label]

    # Last span, if the document ends inside an entity
    if current_tokens:
        spans.append(finish_span(current_tokens, current_labels, offsets, doc_text))

    # Dropping empty spans (when a span was only spaces)
    spans = [span for span in spans if span[0] < span[1]]
    return spans


def strict_scores(predictions, gold, label_names):
    # Span counts only if doc, start, end and label all match the gold
    results = {}
    for label in label_names:
        gold_set = set()
        pred_set = set()
        for doc_id, spans in gold.items():
            for start, end, span_label, _ in spans:
                if span_label == label:
                    gold_set.add((doc_id, start, end))
        for doc_id, spans in predictions.items():
            for start, end, span_label, _ in spans:
                if span_label == label:
                    pred_set.add((doc_id, start, end))

        correct = len(gold_set & pred_set)
        precision = correct / len(pred_set) if pred_set else 0.0
        recall = correct / len(gold_set) if gold_set else 0.0

        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        results[label] = {"precision": precision, "recall": recall, "f1": f1,
                          "gold": len(gold_set), "pred": len(pred_set)}

    macro_f1 = sum(r["f1"] for r in results.values()) / len(label_names)
    macro_p = sum(r["precision"] for r in results.values()) / len(label_names)
    macro_r = sum(r["recall"] for r in results.values()) / len(label_names)
    return results, macro_p, macro_r, macro_f1


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    from tokenizers import Tokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(args.tokenizer_path)

    checkpoint = torch.load(args.ckpt, map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["config"])
    label_vocab = checkpoint["label_vocab"]
    model = build_ner_model(config, len(label_vocab))

    # Adding the same extra weights (lora / lorma / bitfit) that train.py added,
    # otherwise the saved weights don't fit. The first decoder_full checkpoint
    # was saved before --method existed, so it counts as full
    train_args = checkpoint["args"]
    setup_method(
        model,
        train_args.get("method", "full"),
        train_args.get("lora_rank", 8),
        train_args.get("lora_alpha", 16),
        train_args.get("train_top_layers", 4)
    )

    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    print(f"Loaded {train_args['arch']} ({train_args.get('method', 'full')}) NER checkpoint "
          f"from step {checkpoint['step']}, "
          f"val loss {checkpoint['best_val_loss']:.4f}")
    print(f"Decoding: {args.decoding}")

    with open(args.text_file, encoding="utf-8") as f:
        documents = json.load(f)

    # Tagging every document and turning the tags into character spans
    predictions = {}
    doc_ids = list(documents.keys())
    for i, doc_id in enumerate(doc_ids):
        doc_text = documents[doc_id]
        encoding = tokenizer.encode(doc_text)
        offsets = list(encoding.offsets)
        if len(encoding.ids) == 0:
            print(f"document {doc_id} is empty, giving it no spans")

        tag_ids = tag_document(model, encoding.ids, len(label_vocab), args, device)
        predictions[doc_id] = tags_to_spans(tag_ids, label_vocab, offsets, doc_text, args.decoding)

        if (i + 1) % 5 == 0 or (i + 1) == len(doc_ids):
            print(f"tagged {i + 1}/{len(doc_ids)} documents", flush=True)

    pred_path = os.path.join(args.out_dir, "prediction.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False)
    print(f"\nWrote {pred_path}")

    total_spans = sum(len(spans) for spans in predictions.values())
    print(f"{total_spans} predicted spans across {len(predictions)} documents")

    if args.gold_file:
        # Scoring against the fold2 gold spans
        with open(args.gold_file, encoding="utf-8") as f:
            gold = json.load(f)
        # Getting the 12 entity names back from the B- tags
        label_names = [tag[2:] for tag in label_vocab if tag.startswith("B-")]

        results, macro_p, macro_r, macro_f1 = strict_scores(predictions, gold, label_names)
        print(f"\nStrict macro P {macro_p:.4f}  R {macro_r:.4f}  F1 {macro_f1:.4f}\n")
        for label, r in results.items():
            print(f"  {label:10s}  P {r['precision']:.3f}  R {r['recall']:.3f}  F1 {r['f1']:.3f}  "
                  f"(gold {r['gold']}, pred {r['pred']})")
    else:
        zip_path = os.path.join(args.out_dir, "submission.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(pred_path, arcname="prediction.json")
        print(f"Wrote {zip_path} (ready to upload to Codabench)")

    # Printing the first few spans to check them manually
    first_id = doc_ids[0]
    print(f"\nFirst 10 spans in document {first_id}:")
    for span in predictions[first_id][:10]:
        print("  ", span)


if __name__ == "__main__":
    main()
