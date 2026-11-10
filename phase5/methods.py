"""
CS781 Capstone - Phase 5: Legal Statute Identification (LSI)
Fine-tuning methods: full, lorma, bitfit, partial (same as phase4/methods.py)
Knowledge distillation is in distill.py

References:
    - LoRMA: Bihany, Patel and Modi, "LoRMA: Low-Rank Multiplicative Adaptation
      for LLMs", Findings of ACL 2025
    - BitFit: Ben Zaken et al., "BitFit: Simple Parameter-efficient Fine-tuning
      for Transformer-based Masked Language-models", ACL 2022
"""

import torch
import torch.nn as nn

from model import CausalConvMixer, SelfAttention


# LoRMA layer that is put around an existing Linear layer
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


def add_lorma(model, rank, alpha):
    # Putting LoRMA on qkv_proj and out_proj of every attention layer
    # convmix has attention only in every 3rd layer, so its convolution layers
    # also get LoRMA on in_proj and out_proj
    for module in list(model.backbone.modules()):
        if isinstance(module, SelfAttention):
            module.qkv_proj = LoRMALinear(module.qkv_proj, rank, alpha)
            module.out_proj = LoRMALinear(module.out_proj, rank, alpha)
        elif isinstance(module, CausalConvMixer):
            module.in_proj = LoRMALinear(module.in_proj, rank, alpha)
            module.out_proj = LoRMALinear(module.out_proj, rank, alpha)


# Helpers for BitFit and partial fine-tuning
def add_zero_biases(model):
    # The Phase-1 Linear layers were built with bias=False, so there are no
    # biases for BitFit to train. Adding biases that start at zero, so the
    # output doesn't change at first and training only those
    for module in model.backbone.modules():
        if isinstance(module, nn.Linear) and module.bias is None:
            module.bias = nn.Parameter(torch.zeros(module.out_features))


def get_blocks(backbone):
    # Getting the list of layers of the model
    # encdec keeps its layers in encoder_blocks, while decoder, moe and convmix keep them in blocks
    if hasattr(backbone, "encoder_blocks"):
        return backbone.encoder_blocks
    return backbone.blocks


def setup_method(model, method, rank=8, alpha=16, train_top_layers=4):
    # Adding the new weights for the chosen method
    if method == "lorma":
        add_lorma(model, rank, alpha)
    elif method == "bitfit":
        add_zero_biases(model)

    # Choosing which backbone weights get trained
    for name, param in model.backbone.named_parameters():
        if method == "full":
            param.requires_grad = True
        elif method == "lorma":
            param.requires_grad = ("lorma_" in name)
        elif method == "bitfit":
            param.requires_grad = name.endswith(".bias")
        elif method == "partial":
            param.requires_grad = False

    if method == "partial":
        # Training only the last 4 blocks (train_top_layers) and the final norm, the lower blocks stay frozen
        for block in get_blocks(model.backbone)[-train_top_layers:]:
            for param in block.parameters():
                param.requires_grad = True
        for name, param in model.backbone.named_parameters():
            if "norm" in name and "blocks" not in name:
                param.requires_grad = True

    # Case head (BiLSTM and classifier) starts from random weights, so it is trained in every method
    for name, param in model.named_parameters():
        if not name.startswith("backbone."):
            param.requires_grad = True


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
