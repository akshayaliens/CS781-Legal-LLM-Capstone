"""
CS781 Capstone - Phase 2: Legal Named Entity Recognition (L-NER)
Model: Phase-1 backbone + a token classification head

Backbone classes (RMSNorm, SelfAttention, SwiGLU, DecoderLM, EncoderDecoderLM)
copied as-is from phase1/model.py so this file doesn't depend on phase1/
being around. Only new thing is NERHead - takes the backbone's per-token
hidden states and predicts a BIO label for each one.

References (same as phase1/model.py):
    - Transformer: Vaswani et al., 2017
    - RMSNorm: Zhang and Sennrich, 2019
    - RoPE: Su et al., "RoFormer", 2021
    - SwiGLU: Shazeer, 2020
    - token classification head: Devlin et al., "BERT", 2019
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    arch: str = "decoder"
    vocab_size: int = 32000
    block_size: int = 1024
    d_model: int = 768
    n_layer: int = 12
    n_head: int = 12
    ffn_hidden: int = 2048
    dropout: float = 0.0

    n_enc_layer: int = 6
    n_dec_layer: int = 6
    enc_len: int = 768
    dec_len: int = 256

    n_experts: int = 8
    top_k: int = 2
    aux_loss_weight: float = 0.01

    attn_every: int = 3
    conv_kernel: int = 7

    eot_id: int = 0
    pad_id: int = 1
    bos_id: int = 2


# -------------------------------------------------------------------------------------
# Shared building blocks (copied from phase1/model.py, unchanged and added as it is)...
# ------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def build_rope_table(head_dim, max_len, base=10000.0):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    seq_len = x.shape[-2]
    half = x.shape[-1] // 2
    cos, sin = cos[:seq_len], sin[:seq_len]
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


class SelfAttention(nn.Module):
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


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.head_dim = config.d_model // config.n_head
        self.dropout = config.dropout

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.kv_proj = nn.Linear(config.d_model, 2 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, memory):
        batch, seq_len, dim = x.shape
        mem_len = memory.shape[1]

        q = self.q_proj(x).view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(memory).split(dim, dim=-1)
        k = k.view(batch, mem_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch, mem_len, self.n_head, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, dim)
        return self.out_proj(out)


class CausalConvMixer(nn.Module):
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
    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_value = nn.Linear(dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_out(F.silu(self.w_gate(x)) * self.w_value(x)))


class MixtureOfExperts(nn.Module):
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


# -------------------------------------------------------------------------------
# Transformer blocks (copied from phase1/model.py, unchanged and added the same)
# -------------------------------------------------------------------------------------
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


class DecoderWithCrossBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.self_attn = SelfAttention(config, causal=True)
        self.norm2 = RMSNorm(config.d_model)
        self.cross_attn = CrossAttention(config)
        self.norm3 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.dropout)

    def forward(self, x, rope, memory):
        x = x + self.self_attn(self.norm1(x), rope)
        x = x + self.cross_attn(self.norm2(x), memory)
        x = x + self.ffn(self.norm3(x))
        return x


def init_weights(module, n_layers):
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
    # backbone for decoder/moe/convmix, same as phase1 but forward() just
    # returns the hidden states now, no loss here (NERHead does that)
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

        return x  # (batch, seq_len, d_model) - per-token hidden states


class EncoderDecoderLM(nn.Module):
    # backbone for encdec - only using the encoder half here, nothing to
    # generate for NER so the decoder stack from phase1 isn't needed
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
    if config.arch == "encdec":
        return EncoderDecoderLM(config)
    return DecoderLM(config)


# ------------------------------------------------------------------
#  This is new for Phase 2: the NER head
# ------------------------------------------------------------------
class NERHead(nn.Module):
    # one linear layer on top of the backbone, turning each token's hidden
    # state into a score per BIO label
    def __init__(self, backbone, d_model, num_labels, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, idx, label_ids=None):
        hidden = self.backbone(idx)
        hidden = self.dropout(hidden)
        logits = self.classifier(hidden)  # (batch, seq_len, num_labels)

        if label_ids is None:
            return logits, None

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            label_ids.reshape(-1),
            ignore_index=-100
        )
        return logits, loss


def build_ner_model(config, num_labels):
    backbone = build_backbone(config)
    return NERHead(backbone, config.d_model, num_labels)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    # same shape as the real decoder checkpoint
    config = ModelConfig(arch="decoder", d_model=768, n_layer=12, n_head=12, ffn_hidden=2048)
    num_labels = 25  # matches label_vocab size printed by data_prep.py

    model = build_ner_model(config, num_labels)
    print("NER model parameters:", count_parameters(model) / 1e6, "M")

    dummy_input = torch.randint(0, config.vocab_size, (2, 512))
    dummy_labels = torch.randint(0, num_labels, (2, 512))

    logits, loss = model(dummy_input, dummy_labels)
    print("logits shape:", logits.shape)
    print("loss:", loss.item())
