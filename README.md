# In-House Legal Intelligence System (CS781 Capstone, IIT Kanpur)

Small legal language models trained from scratch on Indian legal text, and then fine-tuned for six legal NLP tasks. The idea behind the project is a law firm that cannot send confidential documents to an outside AI service, so everything has to be trained and run in-house.

All work was done on the free tier of Google Colab with one NVIDIA T4 GPU. Only the official course dataset ([Exploration-Lab/CS-781-Capstone](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone) on Hugging Face) was used, and no pretrained model was used anywhere. Every model in Phases 2-7 starts from a Phase-1 checkpoint.

## Results

Best test score in each phase on the course Codabench leaderboard:

| Phase | Task | Best model | Metric | Test score |
|---|---|---|---|---|
| 1 | Legal language model pretraining | decoder, greedy decoding | BERTScore F1 | 0.8292 |
| 2 | Legal named entity recognition | decoder, full fine-tuning | strict macro-F1 | 0.3507 |
| 3 | Rhetorical role prediction | decoder, full fine-tuning | macro-F1 | 0.4856 |
| 4 | Court judgment prediction | decoder, full fine-tuning | macro-F1 | 0.6561 |
| 5 | Legal statute identification | decoder, full fine-tuning | macro-F1 | 0.2703 |
| 6 | Abstractive summarization | decoder, LoRA | BERTScore F1 | 0.8230 |
| 7 | Prior case retrieval | decoder, full fine-tuning (new layer trained at a lower learning rate) | Micro-F1@7 | 0.1518 |

In total there were 36 submissions. All of them are listed phase by phase below.

## Repository layout

```
phase1/   pretraining: data preparation, tokenizer, 4 model architectures, training, text generation
phase2/   legal NER
phase3/   rhetorical roles
phase4/   judgment prediction
phase5/   statute identification
phase6/   summarization
phase7/   prior case retrieval
```

Phases 2-7 all follow the same pattern:

| File | What it does |
|---|---|
| `data_prep.py` | downloads the phase data, tokenizes it with the Phase-1 tokenizer, cuts or shortens long texts to fit the model, saves json files |
| `model.py` | the four Phase-1 backbones (copied from Phase 1 so each folder works on its own) plus the task head of that phase |
| `methods.py` | the fine-tuning techniques: decides which weights are trained (full, partial, LoRA, LoRMA, BitFit, IA3, AdaLoRA or adapters, depending on the phase) |
| `train.py` | the training loop: loads a Phase-1 checkpoint, trains with the chosen technique, keeps the best dev checkpoint |
| `distill.py` | knowledge distillation: a smaller student learns from a trained teacher |
| `predict.py` | with `--split dev` scores the dev set, with `--split test` writes `prediction.json` and `submission.zip` (`--split val` in Phase 6; Phase 2 uses `--gold_file` for dev scoring instead) |
| `*.ipynb` | the Colab notebook with the commands that were run and their outputs |

The model type and the technique are picked with command-line options, for example `--arch decoder --method full`. One `train.py` covers every run of a phase except the distillation runs, which use `distill.py`.

## Setup

**Requirements:** Python 3 with `torch`, `tokenizers`, `huggingface_hub`, `numpy`, `ijson`, `scikit-learn`, `peft` (Phase 3 only) and `rouge-score` (Phase 6 only). All of these are available in Colab or can be installed with `pip`.

**Data:** the dataset is gated on Hugging Face, so the terms have to be accepted and a Hugging Face login is needed before downloading.

**Folders:** the scripts were run in Colab with Google Drive mounted, using this layout:

```
/content/drive/MyDrive/CS781/
    phase1/ ... phase7/        code from this repository
    data_p1/ ... data_p7/      prepared data (made by the data scripts)
    ckpt_p1/ ... ckpt_p7/      checkpoints and training logs, one folder per run
    submissions_p1/ ...        prediction.json and submission.zip, one folder per run
```

Only the `phase` folders are in this repository. The `data`, `ckpt` and `submissions` folders are created on Google Drive when the scripts run.

The commands below use a shortcut `D` for the project folder in the reader's own Google Drive, mounted in Colab. Set it once and move into that folder before running them:

```bash
D=/content/drive/MyDrive/CS781   # save the folder path under the short name D
cd $D                            # move into that folder
```

After this, `$D/data_p2` means `/content/drive/MyDrive/CS781/data_p2`, and a command like `python phase2/train.py` runs the script from the `phase2` folder.

For example:
```bash
python phase2/train.py --arch decoder --method full \
    --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p2 --out_dir $D/ckpt_p2/decoder_full
```

**Model weights and checkpoints are not included**, following the course instructions and because of their size. The models can be trained again by running the commands under "How to run" for each phase, in order: first data preparation, then training, then prediction.

---

## Phase 1: Legal language model pretraining

**Task.** Given the first 512 words of a legal document, write the next 128 words. Scored with BERTScore F1 (ROUGE-L and BLEU-4 as secondary scores).

**Data.** The training data has four sources: Acts, Supreme Court, High Courts and Tribunals. Most High Court records are short orders (median 95 words in a sample of 200), so documents under 200 words were dropped and the rest were joined with an `<|endoftext|>` token between them. The test prompts read like Supreme Court judgments, so the Supreme Court got the biggest share. The final training data, sampled from more than 4.5 billion available tokens, had 528.9M tokens: 255.1M Supreme Court, 102.2M Tribunals, 91.9M High Courts and 79.6M Acts. While sampling, every 50th document was kept out of training. From these unused documents, 198 examples in the same format as the test were made for checking the model locally: each is a 512-word piece of text and the 128 words that follow it.

**Tokenizer.** A byte-level BPE tokenizer with 32,000 tokens, trained on 300 MB of text with an equal share from each source. Special tokens: `<|endoftext|>` (between documents), `<pad>` (padding) and `<s>` (start token for the encdec decoder).

**Model architecture.** Four models with different designs, all trained from random weights. All of them use RMSNorm, rotary position embeddings (RoPE), SwiGLU feed-forward layers, and the same weight matrix for the input embedding and the output layer.

| Model | Design | Layers | Width | Heads | Feed-forward size | Parameters |
|---|---|---|---|---|---|---|
| decoder | GPT-style decoder-only Transformer | 12 | 768 | 12 | 2048 | 109.5M |
| encdec | encoder reads the prompt (both directions), decoder writes the text | 6 + 6 | 704 | 11 | 1920 | 106.9M |
| moe | decoder with 8 feed-forward experts per layer, each token uses 2 | 10 | 640 | 10 | 1024 per expert | 194.2M (76.3M used per token) |
| convmix | cheap causal convolution (kernel 7) in most layers, full attention in every third layer | 12 | 768 | 12 | 2048 | 104.9M |

In moe, a small layer called the router decides which 2 of the 8 experts handle each token. An extra loss with a small weight (0.01) makes the router use all 8 experts, instead of always picking the same few.

`train.py` checks the parameter count and stops if a model is outside the course's allowed range of 50M-300M.

**Training.** Next-token prediction with AdamW (betas 0.9 and 0.95, weight decay 0.1), learning rate 6e-4 with 150 warmup steps and cosine decay to 10%, gradient clipping 1.0 and mixed precision. The decoder was trained on about 70M tokens over 542 steps (about 8 hours across several Colab sessions, final loss 2.97). The other three were trained for 170 steps each, about 22M tokens and 1-1.5 hours per model, with 512-token blocks. The decoder was trained much longer than the other three because of limited Colab time, so its lower loss is partly due to the extra training. The best checkpoint is saved by validation loss, and a stopped run continues from its last checkpoint when the same command is run again.

**Text generation.** One token at a time, with the end-of-text and padding tokens blocked so the text does not stop early, a repetition penalty of 1.15, and the output cut to exactly 128 words. Two decoding modes: greedy, and nucleus sampling (temperature 0.8, top-p 0.9).

**Submissions.**

| Model | Decoding | Test BERTScore F1 |
|---|---|---|
| **decoder** | **greedy** | **0.8292** |
| decoder | sampling | 0.8278 |
| convmix | greedy | 0.8257 |
| convmix | sampling | 0.8199 |
| moe | greedy | 0.8217 |
| encdec | greedy | 0.8077 |

**How to run.**

```bash
# data and tokenizer
python phase1/prepare_data.py --out_dir $D/data_p1 \
    --budget_mb "supreme_court=1300,high_courts=400,tribunals=400,acts=200"

# training (run again after a disconnect to continue from the last checkpoint)
python phase1/train.py --arch decoder --data_dir $D/data_p1 \
    --out_dir $D/ckpt_p1/decoder --block_size 512 --total_tokens 7e7
python phase1/train.py --arch encdec --data_dir $D/data_p1 \
    --out_dir $D/ckpt_p1/encdec --block_size 512 --total_tokens 2.2e7
# same for --arch moe and --arch convmix

# test predictions with greedy decoding (best result)
python phase1/generate.py --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p1 --prompts $D/data_p1/test.json \
    --out_dir $D/submissions_p1/decoder --decoding greedy

# test predictions with nucleus sampling (saved in a separate folder)
python phase1/generate.py --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p1 --prompts $D/data_p1/test.json \
    --out_dir $D/submissions_p1/decoder_sample --decoding sample
```

**What did not work.** The first decoder run was set for 3,100 steps, which would have needed more than 40 hours on a free T4. Cutting the block size from 1,024 to 512 did not speed it up, because the code keeps the tokens per step fixed. Sampling was slightly worse than greedy decoding for both models it was tried on.

---

## Phase 2: Legal named entity recognition

**Task.** Find every legal entity in a judgment as an exact character span with one of 12 types (APP, RESP, A.COUNSEL, R.COUNSEL, JUDGE, WIT, AUTH, COURT, STAT, PREC, DATE, CASENO). Scored with strict macro-F1, so a span that is off by one character counts as wrong.

**Data.** Two training folds of 35 documents each (8,393 entities) and 35 test documents. fold1 is used for training and fold2 as the dev set. The text is never changed, because the labels are tied to exact character positions.

**Model architecture.** A Phase-1 backbone with a dropout and linear layer on top of every token, which picks one of 25 BIO tags (B and I for each of the 12 types, plus O). For encdec, only the encoder is used.

**Approach.**
- Each token gets a BIO tag when its characters overlap a gold span. The tokenizer keeps the space before a word inside the token, so spans first started one character too early. Skipping leading spaces fixed this. After the fix, 98.1% of the gold spans can be rebuilt exactly from the tokens. The other 1.9% end in the middle of a token and cannot be matched with this tokenizer.
- Documents are cut into 512-token chunks with a 64-token overlap. Where chunks overlap, their scores are averaged.
- **Merge decoding:** the tags are joined into spans by joining the pieces of the same word, keeping a span going on any I tag, and giving the span the type most of its pieces have. With the same model, this raised the fold2 F1 from 0.284 to 0.342 compared with simple B/I decoding, so all submissions use it.

**Setup.** AdamW, weight decay 0.01, 50 warmup steps and cosine decay, batch of 4 chunks, gradient clipping 1.0. The best checkpoint is kept by fold2 loss.

**Submissions.**

| Model | Technique | Trained weights | fold2 F1 | Test F1 |
|---|---|---|---|---|
| **decoder** | **full fine-tuning** | 100% | 0.342 | **0.3507** |
| decoder | LoRA (rank 8) | 0.42% | 0.276 | 0.2772 |
| decoder, 6 layers | knowledge distillation from decoder full | 61% size | 0.306 | 0.3097 |
| encdec | LoRMA (rank 8) | 0.49% | 0.136 | 0.1324 |
| moe | BitFit | 0.13% | 0.253 | 0.2631 |
| convmix | partial (top 4 blocks) | 25.9% | 0.293 | 0.2899 |

**How to run.**

```bash
# data
python phase2/data_prep.py --out_dir $D/data_p2 \
    --tokenizer_path $D/data_p1/tokenizer.json

# training
python phase2/train.py --arch decoder --method full \
    --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p2 --out_dir $D/ckpt_p2/decoder_full
python phase2/train.py --arch decoder --method lora --learning_rate 5e-4 \
    --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p2 --out_dir $D/ckpt_p2/decoder_lora
python phase2/train.py --arch encdec --method lorma --epochs 8 --learning_rate 1e-3 \
    --ckpt $D/ckpt_p1/encdec/best_checkpoint.pt \
    --data_dir $D/data_p2 --out_dir $D/ckpt_p2/encdec_lorma_v2
python phase2/train.py --arch moe --method bitfit --epochs 6 --learning_rate 1e-3 \
    --ckpt $D/ckpt_p1/moe/best_checkpoint.pt \
    --data_dir $D/data_p2 --out_dir $D/ckpt_p2/moe_bitfit
python phase2/train.py --arch convmix --method partial --train_top_layers 4 --learning_rate 1e-4 \
    --ckpt $D/ckpt_p1/convmix/best_checkpoint.pt \
    --data_dir $D/data_p2 --out_dir $D/ckpt_p2/convmix_partial

# knowledge distillation
python phase2/distill.py --teacher_ckpt $D/ckpt_p2/decoder_full/best_checkpoint.pt \
    --p1_ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p2 --out_dir $D/ckpt_p2/decoder_distill

# fold2 score (best run; same for the other runs with their folder name)
python phase2/predict.py --ckpt $D/ckpt_p2/decoder_full/best_checkpoint.pt \
    --tokenizer_path $D/data_p1/tokenizer.json \
    --text_file $D/data_p2/fold2-text.json --gold_file $D/data_p2/fold2-gold.json \
    --out_dir $D/ckpt_p2/decoder_full/dev_check --decoding merge

# test submission (best run; same for the other runs)
python phase2/predict.py --ckpt $D/ckpt_p2/decoder_full/best_checkpoint.pt \
    --tokenizer_path $D/data_p1/tokenizer.json \
    --text_file $D/data_p2/test_text.json \
    --out_dir $D/submissions_p2/decoder_full --decoding merge
```

**What did not work.** The first LoRMA run (3 epochs, learning rate 5e-4) scored only 0.012 on fold2, because it was trained too little; 8 epochs at 1e-3 fixed it. Party names (APP, RESP, WIT) stay weak, because the word that shows a person's role usually comes after the name and three of the four backbones only look at earlier words. LoRA reached almost the same fold2 loss as full fine-tuning but a much lower F1, so validation loss was a poor guide for choosing a model.

---

## Phase 3: Rhetorical role prediction

**Task.** The input is a judgment, already split into sentences. The output is one of 13 roles for each sentence (Fact, Issue, ArgumentPetitioner, ArgumentRespondent, Statute, Dissent, PrecedentReliedUpon, PrecedentNotReliedUpon, PrecedentOverruled, RulingByLowerCourt, RatioOfTheDecision, RulingByPresentCourt, None). The role tells the part the sentence plays in the judgment. The judgments come from two areas of law, called domains: Competition Law (CL) and Income Tax (IT). Scored with macro-F1.

**Data.** 80 training documents (16,845 sentences), 10 dev and 10 test documents. The roles are very unbalanced: 4,301 RatioOfTheDecision sentences, but only 107 Issue and 3 PrecedentOverruled.

**Model architecture.** The backbone reads a window of whole sentences. Each sentence becomes one vector (the average of its token vectors), a BiLSTM runs over the sentence vectors so every sentence sees its neighbours on both sides, and a linear layer picks one of the 13 roles. CL and IT are trained together in one model.

**Approach.** Sentences are cut to at most 128 tokens and packed into windows of up to 512 tokens (about 12 sentences). Each window starts about halfway through the previous one. Mistakes on rare roles are given a bigger weight in the loss (at most 10), so the model does not ignore them. LoRA and LoRMA were coded from scratch in `methods.py`, while IA3 and AdaLoRA come from the Hugging Face `peft` library.

**Setup.** Learning rate 3e-5 (LoRA, LoRMA, AdaLoRA 1e-3; IA3 3e-3; distillation 5e-5), head 1e-3, batch of 4 windows, mixed precision. The best checkpoint is kept by dev macro-F1. Full fine-tuning and distillation ran 3 epochs, the other methods 5 epochs.

**Submissions.**

| Model | Technique | Dev macro-F1 | Test macro-F1 |
|---|---|---|---|
| **decoder** | **full fine-tuning** | 0.510 | **0.4856** |
| decoder | LoRMA | 0.532 | 0.4290 |
| decoder, 6 layers | knowledge distillation | 0.466 | 0.4590 |
| encdec | LoRA | 0.264 | 0.2682 |
| moe | IA3 | 0.453 | 0.3761 |
| convmix | AdaLoRA | 0.513 | 0.4173 |

**How to run.**

```bash
# data
python phase3/data_prep.py --out_dir $D/data_p3 \
    --tokenizer_path $D/data_p1/tokenizer.json

# training
python phase3/train.py --arch decoder --method full \
    --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p3 --out_dir $D/ckpt_p3/decoder_full
python phase3/train.py --arch decoder --method lorma --epochs 5 --learning_rate 1e-3 \
    --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p3 --out_dir $D/ckpt_p3/decoder_lorma
python phase3/train.py --arch encdec --method lora --epochs 5 --learning_rate 1e-3 \
    --ckpt $D/ckpt_p1/encdec/best_checkpoint.pt \
    --data_dir $D/data_p3 --out_dir $D/ckpt_p3/encdec_lora
python phase3/train.py --arch moe --method ia3 --epochs 5 --learning_rate 3e-3 \
    --ckpt $D/ckpt_p1/moe/best_checkpoint.pt \
    --data_dir $D/data_p3 --out_dir $D/ckpt_p3/moe_ia3
python phase3/train.py --arch convmix --method adalora --epochs 5 --learning_rate 1e-3 \
    --ckpt $D/ckpt_p1/convmix/best_checkpoint.pt \
    --data_dir $D/data_p3 --out_dir $D/ckpt_p3/convmix_adalora

# knowledge distillation
python phase3/distill.py --teacher_ckpt $D/ckpt_p3/decoder_full/best_checkpoint.pt \
    --p1_ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p3 --out_dir $D/ckpt_p3/decoder_distill

# dev score (best run; same for the other runs with their folder name)
python phase3/predict.py --split dev \
    --ckpt $D/ckpt_p3/decoder_full/best_checkpoint.pt \
    --data_dir $D/data_p3 --out_dir $D/ckpt_p3/decoder_full

# test submission (best run; same for the other runs)
python phase3/predict.py --split test \
    --ckpt $D/ckpt_p3/decoder_full/best_checkpoint.pt \
    --data_dir $D/data_p3 --out_dir $D/submissions_p3/decoder_full
```

**What did not work.** LoRMA, IA3 and AdaLoRA scored 0.08-0.10 lower on test than on dev, while full fine-tuning lost only 0.02. The dev set has only 10 documents, so the checkpoint that scored best on it was partly lucky and did worse on new documents. The LoRMA model also predicted the most common role far too often on test (RatioOfTheDecision for 832 sentences, against 687 for full fine-tuning). Dissent and PrecedentOverruled never appear in dev, so how well the model predicts them could not be measured before submitting.

---

## Phase 4: Court judgment prediction

**Task.** Predict whether a Supreme Court appeal was accepted (1) or rejected (0), from the judgment with the decision part removed. Scored with macro-F1.

**Data.** 37,387 training cases (59% rejected), 1,988 dev cases (50/50) and 1,517 test cases. Dev and test are lowercase but training is not, so all text is lowercased. Almost every case is longer than 512 tokens.

**Model architecture.** Each case is cut into up to four 512-token chunks taken from its end, where the court's reasoning is. The Phase-1 backbone reads each chunk, and the average of its token vectors gives one vector per chunk. A BiLSTM reads these chunk vectors in order and gives one vector for the whole case. A linear layer turns it into scores for accepted and rejected. This follows the hierarchical models of the ILDC paper (Malik et al., 2021).

**Approach.** A balanced subset of 6,000 training cases (3,000 per label) is used, to match the balanced dev set and to fit the time limit. One batch holds 2 cases (up to 8 chunks). Gradients from 8 batches are added up, so the weights are updated once per 16 cases.

**Setup.** Learning rate 3e-5 (LoRMA 1e-3, distillation 5e-5), head 1e-3, 2 epochs (3 for LoRMA), 30 warmup updates and cosine decay, mixed precision. The best checkpoint is kept by dev macro-F1.

**Submissions.**

| Model | Technique | Dev macro-F1 | Test macro-F1 |
|---|---|---|---|
| **decoder** | **full fine-tuning** | 0.662 | **0.6561** |
| decoder, 6 layers | knowledge distillation | 0.626 | 0.5902 |
| convmix | LoRMA | 0.581 | 0.5794 |
| encdec | partial (top 3 of 6 encoder blocks) | 0.501 | 0.5032 |

**How to run.**

```bash
# data
python phase4/data_prep.py --out_dir $D/data_p4 \
    --tokenizer_path $D/data_p1/tokenizer.json

# training
python phase4/train.py --arch decoder --method full \
    --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p4 --out_dir $D/ckpt_p4/decoder_full
python phase4/train.py --arch convmix --method lorma --epochs 3 --learning_rate 1e-3 \
    --ckpt $D/ckpt_p1/convmix/best_checkpoint.pt \
    --data_dir $D/data_p4 --out_dir $D/ckpt_p4/convmix_lorma
python phase4/train.py --arch encdec --method partial --train_top_layers 3 \
    --ckpt $D/ckpt_p1/encdec/best_checkpoint.pt \
    --data_dir $D/data_p4 --out_dir $D/ckpt_p4/encdec_partial

# knowledge distillation
python phase4/distill.py --teacher_ckpt $D/ckpt_p4/decoder_full/best_checkpoint.pt \
    --p1_ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p4 --out_dir $D/ckpt_p4/decoder_distill

# dev score (best run; same for the other runs with their folder name)
python phase4/predict.py --split dev \
    --ckpt $D/ckpt_p4/decoder_full/best_checkpoint.pt \
    --data_dir $D/data_p4 --out_dir $D/ckpt_p4/decoder_full

# test submission (best run; same for the other runs)
python phase4/predict.py --split test \
    --ckpt $D/ckpt_p4/decoder_full/best_checkpoint.pt \
    --data_dir $D/data_p4 --out_dir $D/submissions_p4/decoder_full
```

**What did not work.** The first run updated the weights after every 2 cases and learned nothing: it predicted label 1 for 1,982 of 1,988 dev cases (macro-F1 0.33). After adding up gradients over 8 batches (16 cases per update), the same model reached 0.662 on dev.

---

## Phase 5: Legal statute identification

**Task.** The input is the text of a court judgment. The output is the list of statutes (sections of Acts) that apply to the case, chosen from 100 possible statutes. A case can have more than one statute (3.71 on average in training), so for each of the 100 statutes the model says yes or no. In the text, statute names are hidden, for example "Section 302 of the Indian Penal Code" appears as "`<SECTION>` of the `<ACT>`", so the answer cannot be copied from the text. Scored with macro-F1 over the 100 statutes (micro-F1 and exact match as secondary scores).

**Data.** 42,750 training, 10,181 dev and 13,019 test cases. The most common statute appears in 11,731 training cases and the rarest in only 272.

**Model architecture.** The same model as Phase 4 (chunks, then a BiLSTM), but it reads the first four chunks of a case, where more statute tags appear, and has 100 outputs, one per statute. Each output is a separate probability between 0 and 1 (a sigmoid), so every statute is its own yes or no.

**Approach.**
- A training subset of 8,000 cases, picked so that every statute has at least 150 cases. This cuts the gap between the most common and the rarest statute from 43 times to 12 times.
- In training, missing a rare statute is penalised more (weight 1.73 for the most common statute, up to 6.91 for the rarest), so rare statutes are not ignored.
- The model scores each statute from 0 to 1 and predicts it if the score is higher than a cut-off (normally 0.5). Each statute has its own cut-off, chosen on dev, since 0.5 is too high for rare ones. The top-scoring statute is always predicted, since every case has at least one.

**Setup.** Batch of 2 cases, with gradients added up over 8 batches (16 cases per update), 2 epochs (1,000 updates), learning rate 3e-5 (LoRMA 1e-3, distillation 5e-5), head 1e-3, mixed precision. The dev check during training uses 2,000 of the 10,181 dev cases.

**Submissions.**

| Model | Technique | Dev macro-F1 | Test macro-F1 |
|---|---|---|---|
| **decoder** | **full fine-tuning** | 0.278 | **0.2703** |
| moe | partial (top 4 blocks) | 0.210 | 0.2036 |
| encdec encoder | knowledge distillation from decoder full | 0.179 | 0.1716 |
| convmix | LoRMA | 0.130 | 0.1253 |

Secondary test scores (decoder full only): micro-F1 0.3643, exact match 0.0098.

**How to run.**

```bash
# data
python phase5/data_prep.py --out_dir $D/data_p5 \
    --tokenizer_path $D/data_p1/tokenizer.json

# training
python phase5/train.py --arch decoder --method full \
    --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p5 --out_dir $D/ckpt_p5/decoder_full
python phase5/train.py --arch convmix --method lorma --learning_rate 1e-3 \
    --ckpt $D/ckpt_p1/convmix/best_checkpoint.pt \
    --data_dir $D/data_p5 --out_dir $D/ckpt_p5/convmix_lorma
python phase5/train.py --arch moe --method partial \
    --ckpt $D/ckpt_p1/moe/best_checkpoint.pt \
    --data_dir $D/data_p5 --out_dir $D/ckpt_p5/moe_partial

# knowledge distillation (encdec student)
python phase5/distill.py --teacher_ckpt $D/ckpt_p5/decoder_full/best_checkpoint.pt \
    --p1_ckpt $D/ckpt_p1/encdec/best_checkpoint.pt \
    --data_dir $D/data_p5 --out_dir $D/ckpt_p5/encdec_distill

# dev: picks one cut-off per statute and saves thresholds.json
# (best run; same for the other runs with their folder name)
python phase5/predict.py --split dev \
    --ckpt $D/ckpt_p5/decoder_full/best_checkpoint.pt \
    --data_dir $D/data_p5 --out_dir $D/submissions_p5/decoder_full

# test submission, using the saved cut-offs (best run; same for the other runs)
python phase5/predict.py --split test \
    --ckpt $D/ckpt_p5/decoder_full/best_checkpoint.pt \
    --data_dir $D/data_p5 --out_dir $D/submissions_p5/decoder_full
```

**What did not work.** Every run was still improving at the last step, so all of them are under-trained. With one cut-off of 0.5 for all statutes, the LoRMA model scored only 0.03 on dev, against 0.13 with one cut-off per statute.

---

## Phase 6: Abstractive summarization

**Task.** The input is a long judgment, and the output is a summary written by the model. Scored with BERTScore F1 (ROUGE-1 and ROUGE-L as secondary scores).

**Data.** 7,030 training documents and 100 test documents. There is no dev set, so 5% of training (351 documents) is kept aside for validation. A typical document has 3,686 tokens and a typical summary 767 tokens.

**Model architecture.** The model writes the summary one token at a time, as in Phase 1, so the Phase-1 output layer is kept and no new head is added. For decoder, moe and convmix, the input is `document <s> summary <|endoftext|>`, and the model is trained only to predict the summary part. For encdec, the encoder reads the document and the decoder writes the summary. The window is 1,024 tokens, twice the 512 used in Phase 1, which works because the position method (RoPE) has no learned weights.

**Approach.** The model can read 1,024 tokens at a time: 512 for the document, 510 for the summary and 2 special tokens. Documents are much longer, so only their first 256 and last 256 tokens are kept. The end shares the most words with the summary, and the start holds the facts. In training, longer summaries are cut to 510 tokens. Summaries are written with greedy decoding, and the same three tokens in a row are not allowed twice.

**Setup.** 1 epoch (418 updates), batch of 4, with gradients added up over 4 batches (16 documents per update), learning rate 1e-3 (distillation 1e-4), mixed precision. The best checkpoint is kept by validation loss. ROUGE was checked on 50 validation documents. BERTScore was only computed by Codabench.

**Submissions.**

| Model | Technique | Val ROUGE-1 / ROUGE-L | Test BERTScore F1 |
|---|---|---|---|
| **decoder** | **LoRA (rank 8)** | 0.381 / 0.205 | **0.8230** |
| convmix | LoRMA | 0.363 / 0.198 | 0.8168 |
| moe | Houlsby adapters (size 64) | 0.373 / 0.204 | 0.8134 |
| encdec, 3+3 layers | knowledge distillation from decoder LoRA | 0.336 / 0.196 | 0.8081 |

Secondary test scores (decoder LoRA only): ROUGE-1 0.3797, ROUGE-L 0.2045. The distilled encdec student writes summaries about 9 times faster than the decoder.

**How to run.**

```bash
# data
python phase6/data_prep.py --out_dir $D/data_p6 \
    --tokenizer_path $D/data_p1/tokenizer.json

# training
python phase6/train.py --arch decoder --method lora \
    --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p6 --out_dir $D/ckpt_p6/decoder_lora
python phase6/train.py --arch convmix --method lorma \
    --ckpt $D/ckpt_p1/convmix/best_checkpoint.pt \
    --data_dir $D/data_p6 --out_dir $D/ckpt_p6/convmix_lorma
python phase6/train.py --arch moe --method adapter \
    --ckpt $D/ckpt_p1/moe/best_checkpoint.pt \
    --data_dir $D/data_p6 --out_dir $D/ckpt_p6/moe_adapter

# knowledge distillation (encdec student)
python phase6/distill.py --teacher_ckpt $D/ckpt_p6/decoder_lora/best_checkpoint.pt \
    --p1_ckpt $D/ckpt_p1/encdec/best_checkpoint.pt \
    --data_dir $D/data_p6 --out_dir $D/ckpt_p6/encdec_distill

# val: summaries for 50 validation documents and their ROUGE
# (best run; same for the other runs with their folder name)
python phase6/predict.py --split val \
    --ckpt $D/ckpt_p6/decoder_lora/best_checkpoint.pt \
    --data_dir $D/data_p6 --tokenizer_path $D/data_p1/tokenizer.json \
    --out_dir $D/submissions_p6/decoder_lora

# test submission (best run; same for the other runs)
python phase6/predict.py --split test \
    --ckpt $D/ckpt_p6/decoder_lora/best_checkpoint.pt \
    --data_dir $D/data_p6 --tokenizer_path $D/data_p1/tokenizer.json \
    --out_dir $D/submissions_p6/decoder_lora
```

**What did not work.** The generated summaries are much shorter than the real ones (278-398 words on average against 806), because summaries were cut to 510 tokens in training. All runs were still improving after the single epoch.

---

## Phase 7: Prior case retrieval

**Task.** The input is a case (the query) and a pool of older cases (the candidates). The output is the 7 older cases from the candidates that the query most likely refers to, best first. Scored with Micro-F1@7.

**Data.** 827 training queries with 4,320 candidates, 118 dev queries with 1,023 candidates, and 237 test queries with 1,727 candidates. Each split has its own pool. In training, each query comes with its correct answers, the candidates it actually cites. The 10 queries with no correct answer were skipped.

**Model architecture.** One model turns the query and every candidate into a vector (a bi-encoder). The Phase-1 backbone reads the first four 512-token chunks of a case, and their averaged token vectors are averaged into one. A new layer (`proj`) turns this into 256 numbers, scaled to length 1. Candidates are ranked by how similar their vector is to the query's (cosine similarity).

**Approach.** For each query, the model sees one correct case and 8 random wrong ones, and is trained to make the query's vector closest to the correct one (contrastive loss, InfoNCE, temperature 0.05). To save GPU memory, gradients are computed only for the query and its correct case, and the 8 wrong ones are run without gradients.

**Setup.** 8 queries per update, 2 epochs (206 updates), backbone learning rate 3e-5 (LoRMA 1e-3, distillation 5e-5), 20 warmup steps and cosine decay, mixed precision. The dev score is checked before training, every 50 steps and at the end.

**Submissions.**

| Model | Technique | Best dev Micro-F1@7 | Test Micro-F1@7 |
|---|---|---|---|
| **decoder** | **full fine-tuning, slower head** (`proj` at 3e-5) | 0.203 | **0.1518** |
| decoder | full fine-tuning (`proj` at 1e-3) | 0.179 | 0.1258 |
| moe | partial, slower head | 0.118 | 0.0981 |
| moe | partial (`proj` at 1e-3) | 0.068 | 0.0555 |
| convmix | LoRMA | 0.090 | 0.0579 |
| encdec encoder | knowledge distillation from decoder full | 0.039 | 0.0165 |

**How to run.**

```bash
# data
python phase7/data_prep.py --out_dir $D/data_p7 \
    --tokenizer_path $D/data_p1/tokenizer.json

# training (first settings: proj at 1e-3)
python phase7/train.py --arch decoder --method full \
    --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p7 --out_dir $D/ckpt_p7/decoder_full
python phase7/train.py --arch moe --method partial \
    --ckpt $D/ckpt_p1/moe/best_checkpoint.pt \
    --data_dir $D/data_p7 --out_dir $D/ckpt_p7/moe_partial
python phase7/train.py --arch convmix --method lorma --learning_rate 1e-3 \
    --ckpt $D/ckpt_p1/convmix/best_checkpoint.pt \
    --data_dir $D/data_p7 --out_dir $D/ckpt_p7/convmix_lorma

# training with the slower head (proj at 3e-5)
python phase7/train.py --arch decoder --method full --head_learning_rate 3e-5 \
    --ckpt $D/ckpt_p1/decoder/best_checkpoint.pt \
    --data_dir $D/data_p7 --out_dir $D/ckpt_p7/decoder_full_lowhead
python phase7/train.py --arch moe --method partial --head_learning_rate 3e-5 \
    --ckpt $D/ckpt_p1/moe/best_checkpoint.pt \
    --data_dir $D/data_p7 --out_dir $D/ckpt_p7/moe_partial_lowhead

# knowledge distillation (encdec student)
python phase7/distill.py --teacher_ckpt $D/ckpt_p7/decoder_full/best_checkpoint.pt \
    --p1_ckpt $D/ckpt_p1/encdec/best_checkpoint.pt \
    --data_dir $D/data_p7 --out_dir $D/ckpt_p7/encdec_distill

# dev score (best run; same for the other runs with their folder name)
python phase7/predict.py --split dev \
    --ckpt $D/ckpt_p7/decoder_full_lowhead/best_checkpoint.pt \
    --data_dir $D/data_p7 --out_dir $D/submissions_p7/decoder_full_lowhead

# test submission (best run; same for the other runs)
python phase7/predict.py --split test \
    --ckpt $D/ckpt_p7/decoder_full_lowhead/best_checkpoint.pt \
    --data_dir $D/data_p7 --out_dir $D/submissions_p7/decoder_full_lowhead
```

**What did not work.** With the first settings, training made the dev score worse (0.209 before training, 0.179 after, for the decoder), mostly because the new `proj` layer learned at 1e-3 while the backbone learned at 3e-5. Training `proj` at 3e-5 raised the test score from 0.126 to 0.152. A simple word-matching method (BM25) scored 0.301 on dev, better than every trained model, but was not submitted, since this phase is about fine-tuning the Phase-1 models.
