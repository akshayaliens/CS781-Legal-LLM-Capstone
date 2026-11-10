"""
CS781 Capstone - Phase 5: Legal Statute Identification (LSI)
Model: Phase-1 backbone + a multi-label case head over chunks

References:
    - Phase-1 backbone references are in phase1/model.py
    - LSTM: Hochreiter and Schmidhuber, "Long Short-Term Memory", 1997
    - BiLSTM over chunks: biRNN from the CS781 lectures (Module 2.1)
    - Chunk vectors + recurrent layer over the chunks (hierarchical model):
      Malik et al., "ILDC for CJPE", ACL 2021
    - One sigmoid output per statute (binary relevance) with BCE loss and
      pos_weight: PyTorch BCEWithLogitsLoss
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


@dataclass
class ModelConfig:
    # Model settings, only default values, the real ones are read from the Phase-1 checkpoint
    arch: str = "decoder"
    vocab_size: int = 32000
    block_size: int = 1024
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    ffn_hidden: int = 2048
    dropout: float = 0.0

    # encdec (encoder-decoder) model
    n_enc_layer: int = 6
    n_dec_layer: int = 6
    enc_len: int = 768
    dec_len: int = 256

    # moe (mixture of experts) model
    n_experts: int = 8
    top_k: int = 2
    aux_loss_weight: float = 0.01

    # convmix (convolution + attention) model
    attn_every: int = 3
    conv_kernel: int = 7

    # Special token ids from the Phase-1 tokenizer
    eot_id: int = 0  # <|endoftext|>, end of a document
    pad_id: int = 1  # <pad>, padding
    bos_id: int = 2  # <s>, beginning of sequence, start token of the encdec decoder


# .....................................................................
# Phase-1 backbone, the same layers as in pretraining so the weights can be loaded
# ..........................................................................
class RMSNorm(nn.Module):
    # Scaling each vector by its root mean square, used instead of LayerNorm
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def build_rope_table(head_dim, max_len, base=10000.0):
    # Making the cos and sin values for every position (RoPE)
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # Rotating the query/key vectors by their position
    seq_len = x.shape[-2]
    half = x.shape[-1] // 2
    cos, sin = cos[:seq_len], sin[:seq_len]
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


class SelfAttention(nn.Module):
    # Multi-head attention, causal for decoder, moe and convmix, not causal for encdec
    # causal: each token only sees the tokens before it
    # not causal: each token sees all the tokens, before and after it
    def __init__(self, config, causal=True):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.causal = causal
        self.dropout = config.dropout

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, rope):
        batch, seq_len, dim = x.shape

        q, k, v = self.qkv_proj(x).split(dim, dim=-1)
        q = q.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        q = apply_rope(q, *rope)
        k = apply_rope(k, *rope)

        out = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(out)


class CausalConvMixer(nn.Module):
    # Convolution used instead of attention in convmix, attention is only in every 3rd layer
    def __init__(self, config):
        super().__init__()
        dim, kernel = config.d_model, config.conv_kernel
        self.in_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, padding=kernel - 1)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope=None):
        seq_len = x.shape[1]
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = self.conv(value.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        return self.out_proj(value * F.silu(gate))


class SwiGLU(nn.Module):
    # Feed-forward layer with a SiLU gate
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_value = nn.Linear(dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_out(F.silu(self.w_gate(x)) * self.w_value(x)))


class MixtureOfExperts(nn.Module):
    # Router sends each token to its top_k experts (SwiGLU layers)
    def __init__(self, config):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k

        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)
        self.experts = nn.ModuleList([
            SwiGLU(config.d_model, config.ffn_hidden, config.dropout)
            for _ in range(config.n_experts)
        ])
        self.aux_loss = torch.zeros(())

    def forward(self, x):
        batch, seq_len, dim = x.shape
        tokens = x.reshape(-1, dim)

        router_logits = self.router(tokens).float()
        router_probs = router_logits.softmax(dim=-1)
        top_probs, top_experts = router_probs.topk(self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(tokens)
        for expert_id, expert in enumerate(self.experts):
            token_rows, slot = (top_experts == expert_id).nonzero(as_tuple=True)
            if token_rows.numel() == 0:
                continue
            expert_out = expert(tokens[token_rows]) * top_probs[token_rows, slot].unsqueeze(-1)
            output.index_add_(0, token_rows, expert_out.to(output.dtype))

        fraction_per_expert = F.one_hot(top_experts, self.n_experts).float().sum(1).mean(0) / self.top_k
        self.aux_loss = self.n_experts * (fraction_per_expert * router_probs.mean(0)).sum()

        return output.view(batch, seq_len, dim)


# One layer of each model: attention or convolution, then feed-forward
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(config, causal=True)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.dropout)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.ffn(self.norm2(x))
        return x


class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(config, causal=True)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = MixtureOfExperts(config)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.ffn(self.norm2(x))
        return x


class ConvMixBlock(nn.Module):
    def __init__(self, config, use_attention):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.mixer = SelfAttention(config, causal=True) if use_attention else CausalConvMixer(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.dropout)

    def forward(self, x, rope):
        x = x + self.mixer(self.norm1(x), rope)
        x = x + self.ffn(self.norm2(x))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(config, causal=False)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.dropout)

    def forward(self, x, rope):
        x = x + self.attn(self.norm1(x), rope)
        x = x + self.ffn(self.norm2(x))
        return x


def init_weights(module, n_layers):
    # Random start weights, replaced by the Phase-1 weights when loading
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    scaled_std = 0.02 / math.sqrt(2 * n_layers)
    for name, param in module.named_parameters():
        if name.endswith(("out_proj.weight", "w_out.weight")):
            nn.init.normal_(param, mean=0.0, std=scaled_std)


class DecoderLM(nn.Module):
    # Backbone for decoder, moe and convmix, gives one vector for every token
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        if config.arch == "moe":
            self.blocks = nn.ModuleList([MoEBlock(config) for _ in range(config.n_layer)])
        elif config.arch == "convmix":
            self.blocks = nn.ModuleList([
                ConvMixBlock(config, use_attention=((i + 1) % config.attn_every == 0))
                for i in range(config.n_layer)
            ])
        else:
            self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.n_layer)])

        self.final_norm = RMSNorm(config.d_model)

        head_dim = config.d_model // config.n_head
        cos, sin = build_rope_table(head_dim, config.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        init_weights(self, config.n_layer)

    def forward(self, idx):
        rope = (self.rope_cos, self.rope_sin)
        x = self.dropout(self.token_emb(idx))

        for block in self.blocks:
            x = block(x, rope)
        x = self.final_norm(x)

        return x  # (batch, seq_len, d_model)


class EncoderDecoderLM(nn.Module):
    # backbone for encdec, only the encoder half is used
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_enc_layer)])
        self.encoder_norm = RMSNorm(config.d_model)

        head_dim = config.d_model // config.n_head
        cos, sin = build_rope_table(head_dim, config.enc_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        init_weights(self, config.n_enc_layer)

    def forward(self, idx):
        rope = (self.rope_cos, self.rope_sin)
        x = self.dropout(self.token_emb(idx))
        for block in self.encoder_blocks:
            x = block(x, rope)
        return self.encoder_norm(x)  # (batch, seq_len, d_model)


def build_backbone(config):
    # Picking the backbone class for the architecture
    if config.arch == "encdec":
        return EncoderDecoderLM(config)
    return DecoderLM(config)


# ------------------------------------------------------------------
# Case head for statute identification, same as phase4/model.py but with
# one score for every statute instead of one label for the case
# -----------------------------------------------------------------------
class CaseHead(nn.Module):
    # Predicting a score for each of the 100 statutes for every case from its chunks
    def __init__(self, backbone, d_model, num_labels=100, pos_weight=None, lstm_hidden=256, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(d_model, lstm_hidden, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(2 * lstm_hidden, num_labels)

        # Weight for the positive examples of each statute, rare statutes get a bigger weight
        # Not saved in the checkpoint, it is only needed for the training loss
        if pos_weight is None:
            pos_weight = torch.ones(num_labels)
        self.register_buffer("pos_weight", pos_weight, persistent=False)

    def forward(self, token_ids, token_mask, labels=None):
        # token_ids are the padded chunks of each case, token_mask is 1 for real tokens and 0 for padding
        batch, num_chunks, num_tokens = token_ids.shape

        # Running the backbone on every chunk on its own
        hidden = self.backbone(token_ids.view(-1, num_tokens))  # (batch * chunks, tokens, d_model)

        # Taking the average of the real token vectors of each chunk
        mask = token_mask.view(-1, num_tokens, 1).to(hidden.dtype)
        chunk_vectors = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        chunk_vectors = chunk_vectors.view(batch, num_chunks, -1)
        chunk_vectors = self.dropout(chunk_vectors)

        # Running the BiLSTM over the chunks of each case, skipping the padding chunks
        chunk_counts = (token_mask.sum(-1) > 0).sum(1).cpu()
        packed = pack_padded_sequence(chunk_vectors, chunk_counts,
                                      batch_first=True, enforce_sorted=False)
        _, (last_hidden, _) = self.lstm(packed)

        # Joining the last forward and backward LSTM states into one vector for the case
        case_vector = torch.cat([last_hidden[0], last_hidden[1]], dim=-1)
        case_vector = self.dropout(case_vector)

        # Getting one score for every statute, each statute is a separate yes/no decision
        logits = self.classifier(case_vector)  # (batch, num_labels)

        if labels is None:
            return logits, None

        # labels are 0/1 vectors with a 1 for every statute of the case
        loss = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=self.pos_weight)
        return logits, loss


def build_lsi_model(config, num_labels=100, pos_weight=None):
    # Putting the case head on top of the Phase-1 backbone
    backbone = build_backbone(config)
    return CaseHead(backbone, config.d_model, num_labels, pos_weight)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())
