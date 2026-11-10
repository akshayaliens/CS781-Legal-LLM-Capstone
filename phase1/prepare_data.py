"""
CS781 Capstone - Phase 1: Legal Language Model Pretraining
Data preparation

This script prepares the training data for all four Phase-1 models:
    1. Samples legal text from the four official training files
    2. Keeps some documents aside for validation and a local dev set
    3. Trains a byte-level BPE tokenizer on the legal text
    4. Converts the text into token IDs and saves them as .bin files

Usage (Colab):
    python prepare_data.py --out_dir /content/drive/MyDrive/CS781/data_p1

References:
    - BPE: Sennrich et al., "Neural Machine Translation of Rare Words with
      Subword Units", ACL 2016
    - Byte-level BPE and the <|endoftext|> separator: Radford et al.,
      "Language Models are Unsupervised Multitask Learners" (GPT-2), 2019
    - Hugging Face tokenizers library: https://github.com/huggingface/tokenizers
"""

import argparse
import json
import os
import random
import shutil
import time

import numpy as np

# Turning off the Hugging Face progress bar, since it prints a new line on
# every update in Colab instead of redrawing one line like it does in a
# normal terminal
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

DATASET_REPO = "Exploration-Lab/CS-781-Capstone"

TRAIN_FILES = {
    "acts": "Phase_1_Pretraining/Train/train_acts.json",
    "supreme_court": "Phase_1_Pretraining/Train/train_supreme_court.json",
    "high_courts": "Phase_1_Pretraining/Train/train_high_courts.json",
    "tribunals": "Phase_1_Pretraining/Train/train_tribunals.json",
}

TEST_FILE = "Phase_1_Pretraining/Test/test.json"

# <|endoftext|> separates two documents, <pad> and <s> are needed later
# for padding and for the decoder start token in the encoder-decoder model
SPECIAL_TOKENS = ["<|endoftext|>", "<pad>", "<s>"]

# Length of one official test example: 512-word prompt + 128-word continuation
PROMPT_WORDS = 512
CONTINUATION_WORDS = 128


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--cache_dir", default="/content/hf_cache")

    # Megabytes of text taken from each source.
    # The whole Supreme Court file is used since the test prompts look like
    # Supreme Court judgments. The other three sources add variety.
    parser.add_argument(
        "--budget_mb",
        default="supreme_court=1300,high_courts=400,tribunals=400,acts=200"
    )

    # Documents shorter than this are mostly one-paragraph orders,
    # which do not help with continuing a 512-word prompt
    parser.add_argument("--min_words", type=int, default=200)

    # Every Nth document is kept aside instead of being used for training
    parser.add_argument("--heldout_every", type=int, default=50)
    parser.add_argument("--heldout_per_source", type=int, default=250)
    parser.add_argument("--dev_examples_per_source", type=int, default=50)

    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--tokenizer_sample_mb", type=int, default=300)
    parser.add_argument("--seed", type=int, default=1337)

    return parser.parse_args()


def download_file(rel_path, cache_dir):
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        DATASET_REPO,
        rel_path,
        repo_type="dataset",
        local_dir=cache_dir
    )

    return local_path


def clean_text(text):
    # Replacing newlines, tabs and repeated spaces with a single space,
    # since the test prompts are also plain single-spaced text
    return " ".join(text.split())


def sample_source(name, local_path, budget_mb, args, rng):
    """
    Goes through one training file and keeps around budget_mb of text.

    Returns the paths of the train and held-out .jsonl files, and a small
    dictionary of statistics for the report.
    """
    import ijson

    budget_chars = budget_mb * 1e6
    file_size = os.path.getsize(local_path)

    # Keeping each document with this probability, so the sample is spread
    # over the whole file instead of only coming from the first few records.
    # It is set a little higher than budget/file_size because some documents
    # are later removed by the min_words filter.
    keep_prob = min(1.0, 2.0 * budget_chars / file_size)

    train_path = os.path.join(args.cache_dir, f"{name}_train.jsonl")
    heldout_path = os.path.join(args.cache_dir, f"{name}_heldout.jsonl")

    num_seen = 0
    num_short = 0
    num_train = 0
    num_heldout = 0
    train_chars = 0
    word_counts = []

    start_time = time.time()

    with open(local_path, "rb") as json_file, \
         open(train_path, "w", encoding="utf-8") as train_file, \
         open(heldout_path, "w", encoding="utf-8") as heldout_file:

        # Reading one record at a time, since some files are over 10 GB
        for record in ijson.items(json_file, "item"):
            num_seen += 1

            if num_seen % 100000 == 0:
                print(f"  [{name}] seen {num_seen} records, "
                      f"kept {train_chars / 1e6:.0f} / {budget_mb:.0f} MB", flush=True)

            if rng.random() > keep_prob:
                continue

            text = clean_text(record.get("text", ""))
            num_words = len(text.split())

            if num_words < args.min_words:
                num_short += 1
                continue

            word_counts.append(num_words)
            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"

            # Sending every Nth document to the held-out set
            is_heldout = (len(word_counts) % args.heldout_every == 0)

            if is_heldout and num_heldout < args.heldout_per_source:
                heldout_file.write(line)
                num_heldout += 1
            else:
                train_file.write(line)
                num_train += 1
                train_chars += len(text)

            if train_chars >= budget_chars and num_heldout >= args.heldout_per_source:
                break

    word_counts = np.array(word_counts) if word_counts else np.array([0])

    stats = {
        "file_size_gb": file_size / 1e9,
        "keep_prob": keep_prob,
        "records_seen": num_seen,
        "dropped_too_short": num_short,
        "train_docs": num_train,
        "heldout_docs": num_heldout,
        "train_mb": train_chars / 1e6,
        "words_mean": float(word_counts.mean()),
        "words_median": float(np.median(word_counts)),
        "words_max": int(word_counts.max()),
        "minutes": (time.time() - start_time) / 60,
    }

    print(f"[{name}] {num_train} train docs ({train_chars / 1e6:.0f} MB), "
          f"{num_heldout} held-out docs, median {stats['words_median']:.0f} words", flush=True)

    return train_path, heldout_path, stats


def read_texts(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)["text"]


def train_tokenizer(train_paths, args):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    # Taking an equal amount of text from each source, so the vocabulary
    # is not dominated by the largest source
    chars_per_source = args.tokenizer_sample_mb * 1e6 / len(train_paths)

    def sample_texts():
        for path in train_paths.values():
            used_chars = 0
            for text in read_texts(path):
                yield text
                used_chars += len(text)
                if used_chars >= chars_per_source:
                    break

    # Byte-level BPE, same setup as GPT-2, so every character can be encoded
    # and there is never an unknown token
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True
    )

    tokenizer.train_from_iterator(sample_texts(), trainer=trainer)

    return tokenizer


def write_token_file(tokenizer, jsonl_path, bin_path, eot_id, batch_size=1000):
    """
    Tokenizes every document in jsonl_path and writes all token IDs one after
    another into bin_path, with <|endoftext|> after each document.
    Returns the number of tokens written.
    """
    total_tokens = 0
    batch = []

    def write_batch(texts, out_file):
        token_ids = []
        for encoding in tokenizer.encode_batch(texts):
            token_ids.extend(encoding.ids)
            token_ids.append(eot_id)

        # uint16 is enough because the vocabulary size is below 65,536
        np.array(token_ids, dtype=np.uint16).tofile(out_file)
        return len(token_ids)

    with open(bin_path, "wb") as out_file:
        for text in read_texts(jsonl_path):
            batch.append(text)
            if len(batch) == batch_size:
                total_tokens += write_batch(batch, out_file)
                batch = []

        if batch:
            total_tokens += write_batch(batch, out_file)

    return total_tokens


def build_dev_set(heldout_paths, args, rng):
    """
    Builds a local dev set in the same format as the official test set:
    a 512-word prompt and the true next 128 words as the reference.
    The official references are hidden, so this is used to compare our
    models before submitting to Codabench.
    """
    prompts = {}
    references = {}
    sources = {}

    window = PROMPT_WORDS + CONTINUATION_WORDS

    for name, path in heldout_paths.items():
        count = 0

        for text in read_texts(path):
            words = text.split()
            if len(words) < window:
                continue

            # Picking a random position inside the document
            start = rng.randint(0, len(words) - window)
            prompt_words = words[start:start + PROMPT_WORDS]
            reference_words = words[start + PROMPT_WORDS:start + window]

            example_id = f"{name}_{count}"
            prompts[example_id] = {"text": " ".join(prompt_words)}
            references[example_id] = {"text": " ".join(reference_words)}
            sources[example_id] = name

            count += 1
            if count == args.dev_examples_per_source:
                break

        print(f"[{name}] {count} dev examples", flush=True)

    return prompts, references, sources


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    rng = random.Random(args.seed)

    budgets = {}
    for pair in args.budget_mb.split(","):
        name, mb = pair.split("=")
        budgets[name] = float(mb)

    # Downloading the official test prompts
    test_path = download_file(TEST_FILE, args.cache_dir)
    shutil.copy(test_path, os.path.join(args.cache_dir, "test.json"))

    # Sampling text from each of the four sources
    train_paths = {}
    heldout_paths = {}
    corpus_stats = {}

    for name, rel_path in TRAIN_FILES.items():
        print(f"[{name}] downloading ...", flush=True)
        local_path = download_file(rel_path, args.cache_dir)

        train_path, heldout_path, stats = sample_source(
            name, local_path, budgets[name], args, rng
        )
        train_paths[name] = train_path
        heldout_paths[name] = heldout_path
        corpus_stats[name] = stats

        # Deleting the raw file to free disk space
        os.remove(local_path)

    # Training the tokenizer
    print("Training the BPE tokenizer ...", flush=True)
    tokenizer = train_tokenizer(train_paths, args)
    tokenizer.save(os.path.join(args.cache_dir, "tokenizer.json"))
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}", flush=True)

    # Converting the train and held-out text into token files
    token_counts = {}

    for name in TRAIN_FILES:
        num_train_tokens = write_token_file(
            tokenizer, train_paths[name],
            os.path.join(args.cache_dir, f"train_{name}.bin"), eot_id
        )
        num_val_tokens = write_token_file(
            tokenizer, heldout_paths[name],
            os.path.join(args.cache_dir, f"val_{name}.bin"), eot_id
        )
        token_counts[name] = {"train": num_train_tokens, "val": num_val_tokens}

        print(f"[{name}] {num_train_tokens / 1e6:.1f}M train tokens, "
              f"{num_val_tokens / 1e6:.2f}M val tokens", flush=True)

    # Building the local dev set
    dev_prompts, dev_references, dev_sources = build_dev_set(heldout_paths, args, rng)

    save_json(dev_prompts, os.path.join(args.cache_dir, "dev_prompts.json"))
    save_json(dev_references, os.path.join(args.cache_dir, "dev_references.json"))
    save_json(dev_sources, os.path.join(args.cache_dir, "dev_sources.json"))

    # Saving the information that train.py needs
    total_train_tokens = sum(c["train"] for c in token_counts.values())

    meta = {
        "vocab_size": tokenizer.get_vocab_size(),
        "eot_id": eot_id,
        "pad_id": tokenizer.token_to_id("<pad>"),
        "bos_id": tokenizer.token_to_id("<s>"),
        "sources": list(TRAIN_FILES.keys()),
        "token_counts": token_counts,
        "total_train_tokens": total_train_tokens,
    }
    save_json(meta, os.path.join(args.cache_dir, "meta.json"))
    save_json(corpus_stats, os.path.join(args.cache_dir, "corpus_stats.json"))

    # Copying the final files to Google Drive
    for file_name in os.listdir(args.cache_dir):
        if file_name.endswith((".bin", ".json")):
            shutil.copy(
                os.path.join(args.cache_dir, file_name),
                os.path.join(args.out_dir, file_name)
            )

    print(f"\nTotal training tokens: {total_train_tokens / 1e6:.1f}M")
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
