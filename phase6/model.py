"""
CS781 Capstone - Phase 6: Abstractive Summarization
Model: the Phase-1 language models with their LM head

Summarization is done as language modelling: decoder, moe and convmix read
"document <s> summary" as one sequence, encdec reads the document in the
encoder and writes the summary with the decoder. The loss is only on the
summary tokens.

References:
    - Phase-1 model references are in phase1/model.py
    - Adapters: Houlsby et al., "Parameter-Efficient Transfer Learning for
      NLP", ICML 2019
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    eot_id: int = 0  # <|endoftext|>, end of the summary
    pad_id: int = 1  # <pad>, padding
    bos_id: int = 2  # <s>, put between the document and the summary


# ------------------------------------------------------------------
# Phase-1 layers, the same as in pretraining so the weights can be loaded
# ------------------------------------------------------------------
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
    # Multi-head attention, each token sees only the tokens before it, except in the encdec encoder where it sees all
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
    # Attention from the encdec decoder to the encoder outputs
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


class Adapter(nn.Module):
    # Houlsby adapter, a small bottleneck layer added after attention and after the feed-forward layer
    def __init__(self, dim, size):
        super().__init__()
        self.down = nn.Linear(dim, size)
        self.up = nn.Linear(size, dim)
        # The weights of the up layer start at zero, so at the start the adapter changes nothing
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


# Transformer layers of the four models, attention (or convolution in convmix) and then a feed-forward layer
# adapter1 and adapter2 are empty (None) unless the adapter method is used
class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(config, causal=True)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.dropout)
        self.adapter1 = None
        self.adapter2 = None

    def forward(self, x, rope):
        h = self.attn(self.norm1(x), rope)
        if self.adapter1 is not None:
            h = self.adapter1(h)
        x = x + h

        h = self.ffn(self.norm2(x))
        if self.adapter2 is not None:
            h = self.adapter2(h)
        return x + h


class MoEBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn = SelfAttention(config, causal=True)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = MixtureOfExperts(config)
        self.adapter1 = None
        self.adapter2 = None

    def forward(self, x, rope):
        h = self.attn(self.norm1(x), rope)
        if self.adapter1 is not None:
            h = self.adapter1(h)
        x = x + h

        h = self.ffn(self.norm2(x))
        if self.adapter2 is not None:
            h = self.adapter2(h)
        return x + h


class ConvMixBlock(nn.Module):
    def __init__(self, config, use_attention):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.mixer = SelfAttention(config, causal=True) if use_attention else CausalConvMixer(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn = SwiGLU(config.d_model, config.ffn_hidden, config.dropout)
        self.adapter1 = None
        self.adapter2 = None

    def forward(self, x, rope):
        h = self.mixer(self.norm1(x), rope)
        if self.adapter1 is not None:
            h = self.adapter1(h)
        x = x + h

        h = self.ffn(self.norm2(x))
        if self.adapter2 is not None:
            h = self.adapter2(h)
        return x + h


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
    # encdec decoder layer, with cross-attention to the encoder outputs
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


def compute_loss(logits, targets):
    # Positions with target -100 (the document and the padding) are left out of the loss
    return F.cross_entropy(
        logits.float().view(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100
    )


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
    # decoder, moe and convmix: "document <s> summary" as one sequence
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
        self.lm_head.weight = self.token_emb.weight  # weight tying, same as Phase 1

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
    # encdec: the encoder reads the document, the decoder writes the summary
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
        if last_only:
            x = x[:, -1:, :]
        return self.lm_head(x)

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
    return sum(p.numel() for p in model.parameters())
