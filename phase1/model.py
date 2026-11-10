"""
CS781 Capstone - Phase 1: Legal Language Model Pretraining
Model architectures

We need at least four architecturally different models (not just different sizes),
so this file defines four model families:

    decoder    - a standard decoder-only Transformer, similar to GPT-2
    encdec     - an encoder-decoder Transformer: the encoder reads the prompt,
                 the decoder generates the continuation
    moe        - a decoder-only Transformer where the feed-forward layer is
                 replaced by a Mixture-of-Experts layer (only a few experts
                 are used per token, so it is more efficient for its size)
    convmix    - a lightweight hybrid: most layers use a causal convolution
                 instead of attention, with a few attention layers mixed in

All four are trained completely from scratch (random initialisation), as
required by the rules. No pretrained weights are loaded anywhere in this file.

Run this file directly to print the parameter count of every architecture:
    python model.py

References:
    - Transformer: Vaswani et al., "Attention Is All You Need", NeurIPS 2017
    - Pre-norm residual Transformer, GPT-2 style: Radford et al., 2019
    - RMSNorm: Zhang and Sennrich, "Root Mean Square Layer Normalization",
      NeurIPS 2019
    - Rotary position embeddings (RoPE): Su et al., "RoFormer", 2021
    - SwiGLU feed-forward layer: Shazeer, "GLU Variants Improve Transformer",
      2020
    - Mixture-of-Experts routing: Shazeer et al., "Outrageously Large Neural
      Networks", ICLR 2017; Fedus et al., "Switch Transformer", 2021
"""

import math
from dataclasses import dataclass, asdict

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

    # encoder-decoder only
    n_enc_layer: int = 6
    n_dec_layer: int = 6
    enc_len: int = 768
    dec_len: int = 256

    # mixture-of-experts only
    n_experts: int = 8
    top_k: int = 2
    aux_loss_weight: float = 0.01

    # conv-mixer only
    attn_every: int = 3
    conv_kernel: int = 7

    # special token ids, filled in from meta.json before training
    eot_id: int = 0
    pad_id: int = 1
    bos_id: int = 2


# Sizes chosen to land in the 100M-150M range that the rules recommend.
# Exact counts are printed by running this file; see the table in the report.
PRESETS = {
    "decoder": dict(arch="decoder", d_model=768, n_layer=12, n_head=12, ffn_hidden=2048),
    "encdec": dict(arch="encdec", d_model=704, n_head=11, ffn_hidden=1920,
                   n_enc_layer=6, n_dec_layer=6, enc_len=768, dec_len=256),
    "moe": dict(arch="moe", d_model=640, n_layer=10, n_head=10, ffn_hidden=1024,
                n_experts=8, top_k=2),
    "convmix": dict(arch="convmix", d_model=768, n_layer=12, n_head=12, ffn_hidden=2048,
                     attn_every=3, conv_kernel=7),
}


# ------------------------------------------------------------------
# Shared building blocks
# ------------------------------------------------------------------
class RMSNorm(nn.Module):
    """Normalizes by the root-mean-square of the activations (Zhang & Sennrich, 2019)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * norm).type_as(x) * self.weight


def build_rope_table(head_dim, max_len, base=10000.0):
    # Precomputing cos/sin values for rotary position embeddings
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_len).float()
    angles = torch.outer(positions, inv_freq)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    # Rotating pairs of dimensions by the position-dependent angle
    seq_len = x.shape[-2]
    half = x.shape[-1] // 2
    cos, sin = cos[:seq_len], sin[:seq_len]
    x1, x2 = x[..., :half], x[..., half:]
    rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return rotated


class SelfAttention(nn.Module):
    """Multi-head attention with rotary position embeddings."""

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
    """Attention from decoder positions to encoder outputs (no causal mask)."""

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
    """
    Attention-free token mixer used in the convmix model. A causal depthwise
    convolution looks at nearby tokens, gated by a SiLU-activated branch.
    This costs O(seq_len) instead of the O(seq_len^2) of attention.
    """

    def __init__(self, config):
        super().__init__()
        dim, kernel = config.d_model, config.conv_kernel
        self.in_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.conv = nn.Conv1d(dim, dim, kernel, groups=dim, padding=kernel - 1)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x, rope=None):
        seq_len = x.shape[1]
        value, gate = self.in_proj(x).chunk(2, dim=-1)

        # Padding on the left only, so position t only sees positions <= t
        value = self.conv(value.transpose(1, 2))[..., :seq_len].transpose(1, 2)
        return self.out_proj(value * F.silu(gate))


class SwiGLU(nn.Module):
    """Gated feed-forward layer (Shazeer, 2020); used in place of a plain MLP."""

    def __init__(self, dim, hidden, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_value = nn.Linear(dim, hidden, bias=False)
        self.w_out = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w_out(F.silu(self.w_gate(x)) * self.w_value(x)))


class MixtureOfExperts(nn.Module):
    """
    Replaces the single feed-forward layer with several smaller "expert"
    feed-forward layers. A small router network picks the top_k experts for
    each token, so only a fraction of the parameters are used per token.
    The load-balancing loss (Fedus et al., 2021) discourages the router from
    sending every token to the same expert.
    """

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

        # Load-balancing loss: encourages an even share of tokens per expert
        fraction_per_expert = F.one_hot(top_experts, self.n_experts).float().sum(1).mean(0) / self.top_k
        self.aux_loss = self.n_experts * (fraction_per_expert * router_probs.mean(0)).sum()

        return output.view(batch, seq_len, dim)


# ------------------------------------------------------------------
# Transformer blocks (pre-norm, residual connections)
# ------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """One block of the plain decoder-only model."""

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
    """Same as DecoderBlock, but the feed-forward layer is a MoE layer."""

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
    """
    Either a convolution mixer or self-attention, depending on the layer
    index. Every attn_every-th layer uses attention, the rest use the cheap
    convolution mixer.
    """

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
    """Bidirectional (non-causal) block, used in the encoder stack."""

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
    """Decoder block with an extra cross-attention step over the encoder output."""

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


# ------------------------------------------------------------------
# Full models
# ------------------------------------------------------------------
def compute_loss(logits, targets):
    return F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100
    )


def init_weights(module, n_layers):
    # Standard small-std normal initialisation (same idea as GPT-2)
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    # Scaling down the output projection of every residual branch, so the
    # variance of the residual stream does not grow with depth
    scaled_std = 0.02 / math.sqrt(2 * n_layers)
    for name, param in module.named_parameters():
        if name.endswith(("out_proj.weight", "w_out.weight")):
            nn.init.normal_(param, mean=0.0, std=scaled_std)


class DecoderLM(nn.Module):
    """Used for the 'decoder', 'moe' and 'convmix' architectures."""

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
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight  # weight tying

        head_dim = config.d_model // config.n_head
        cos, sin = build_rope_table(head_dim, config.block_size)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        init_weights(self, config.n_layer)

    def forward(self, idx, targets=None):
        rope = (self.rope_cos, self.rope_sin)
        x = self.dropout(self.token_emb(idx))

        for block in self.blocks:
            x = block(x, rope)
        x = self.final_norm(x)

        if targets is None:
            # Only the last position is needed during generation
            return self.lm_head(x[:, -1:, :]), None

        logits = self.lm_head(x)
        loss = compute_loss(logits, targets)

        if self.config.arch == "moe":
            aux_loss = sum(block.ffn.aux_loss for block in self.blocks)
            loss = loss + self.config.aux_loss_weight * aux_loss

        return logits, loss


class EncoderDecoderLM(nn.Module):
    """The 'encdec' architecture: separate encoder and decoder stacks."""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.n_enc_layer)])
        self.encoder_norm = RMSNorm(config.d_model)

        self.decoder_blocks = nn.ModuleList([
            DecoderWithCrossBlock(config) for _ in range(config.n_dec_layer)
        ])
        self.decoder_norm = RMSNorm(config.d_model)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

        head_dim = config.d_model // config.n_head
        max_len = max(config.enc_len, config.dec_len)
        cos, sin = build_rope_table(head_dim, max_len)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        init_weights(self, config.n_enc_layer + config.n_dec_layer)

    def encode(self, src_ids):
        rope = (self.rope_cos, self.rope_sin)
        x = self.dropout(self.token_emb(src_ids))
        for block in self.encoder_blocks:
            x = block(x, rope)
        return self.encoder_norm(x)

    def decode(self, tgt_ids, memory, last_only=False):
        rope = (self.rope_cos, self.rope_sin)
        x = self.dropout(self.token_emb(tgt_ids))
        for block in self.decoder_blocks:
            x = block(x, rope, memory)
        x = self.decoder_norm(x)
        return self.lm_head(x[:, -1:, :] if last_only else x)

    def forward(self, src_ids, tgt_in, targets=None):
        memory = self.encode(src_ids)

        if targets is None:
            return self.decode(tgt_in, memory, last_only=True), None

        logits = self.decode(tgt_in, memory)
        loss = compute_loss(logits, targets)
        return logits, loss


def build_model(config):
    if config.arch == "encdec":
        return EncoderDecoderLM(config)
    return DecoderLM(config)


def count_parameters(model):
    # Tied embedding/output weights are only counted once here
    total = sum(p.numel() for p in model.parameters())

    active = total
    if model.config.arch == "moe":
        # Every token only goes through top_k of the n_experts feed-forward
        # layers, so the "active" parameter count (used per token) is lower
        # than the total stored parameter count
        one_expert = sum(p.numel() for p in model.blocks[0].ffn.experts[0].parameters())
        unused_experts = model.config.n_experts - model.config.top_k
        active = total - len(model.blocks) * one_expert * unused_experts

    return {"total": total, "active_per_token": active}


if __name__ == "__main__":
    for name, preset in PRESETS.items():
        config = ModelConfig(**preset)
        model = build_model(config)
        counts = count_parameters(model)
        within_limit = 50e6 <= counts["total"] <= 300e6

        print(f"{name:10s} total = {counts['total']/1e6:6.1f}M   "
              f"active/token = {counts['active_per_token']/1e6:6.1f}M   "
              f"within 50M-300M: {within_limit}")
