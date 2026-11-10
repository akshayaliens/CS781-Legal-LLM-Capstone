"""
CS781 Capstone - Phase 6: Abstractive Summarization
Fine-tuning methods: lora, lorma, adapter, full
Knowledge distillation is in distill.py

References:
    - LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models",
      ICLR 2022
    - LoRMA: Bihany, Patel and Modi, "LoRMA: Low-Rank Multiplicative Adaptation
      for LLMs", Findings of ACL 2025
    - Adapters: Houlsby et al., "Parameter-Efficient Transfer Learning for
      NLP", ICML 2019
"""

import torch
import torch.nn as nn

from model import Adapter, CausalConvMixer, SelfAttention


# LoRA layer that keeps the frozen Phase-1 Linear layer and adds two small matrices A and B
class LoRALinear(nn.Module):
    # LoRA: h = W0 x + (alpha / r) * B A x, B starts at zero
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.lora_A.t() @ self.lora_B.t())


# LoRMA layer that keeps the frozen Phase-1 Linear layer and multiplies its output by (I + B A)
class LoRMALinear(nn.Module):
    # LoRMA+: h = (I + (alpha / r) * B A) W0 x, B starts at zero
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lorma_A = nn.Parameter(torch.randn(rank, base.out_features) * 0.02)
        self.lorma_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        h = self.base(x)
        # Applying A and then B on W0 x, so the big B A matrix is not needed
        return h + self.scale * (h @ self.lorma_A.t() @ self.lorma_B.t())


def add_low_rank(model, layer_class, rank, alpha):
    # Putting LoRA or LoRMA on qkv_proj and out_proj of every attention layer
    # convmix has attention only in every 3rd layer, so its convolution layers
    # also get it on in_proj and out_proj
    for module in list(model.modules()):
        if isinstance(module, SelfAttention):
            module.qkv_proj = layer_class(module.qkv_proj, rank, alpha)
            module.out_proj = layer_class(module.out_proj, rank, alpha)
        elif isinstance(module, CausalConvMixer):
            module.in_proj = layer_class(module.in_proj, rank, alpha)
            module.out_proj = layer_class(module.out_proj, rank, alpha)


def add_adapters(model, adapter_size):
    # Putting an adapter after the attention and after the feed-forward layer of every block
    for block in model.blocks:
        block.adapter1 = Adapter(model.config.d_model, adapter_size)
        block.adapter2 = Adapter(model.config.d_model, adapter_size)


def setup_method(model, method, rank=8, alpha=16, adapter_size=64):
    # Adding the new weights for the chosen method
    if method == "lora":
        add_low_rank(model, LoRALinear, rank, alpha)
    elif method == "lorma":
        add_low_rank(model, LoRMALinear, rank, alpha)
    elif method == "adapter":
        add_adapters(model, adapter_size)

    # Choosing which weights get trained, the Phase-1 weights stay frozen except in full fine-tuning
    for name, param in model.named_parameters():
        if method == "full":
            param.requires_grad = True
        elif method == "lora":
            param.requires_grad = ("lora_" in name)
        elif method == "lorma":
            param.requires_grad = ("lorma_" in name)
        elif method == "adapter":
            param.requires_grad = ("adapter" in name)


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
