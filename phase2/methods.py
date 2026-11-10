"""
CS781 Capstone - Phase 2: Legal Named Entity Recognition (L-NER)
Fine-tuning methods

Decides which weights get trained, and adds new small weights for the
methods that need them:
    full     - every weight is trained
    lora     - Phase-1 weights frozen, small B*A added next to each attention layer
    lorma    - same idea, but B*A multiplies the Phase-1 weight instead (LoRMA+)
    bitfit   - only bias terms are trained
    partial  - only the top few blocks are trained

The NER classifier head is always trained, since it starts from random.

References:
    - LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
    - LoRMA: Bihany, Patel and Modi, "LoRMA: Low-Rank Multiplicative Adaptation
      for LLMs", Findings of ACL 2025
    - BitFit: Ben Zaken et al., "BitFit: Simple Parameter-efficient Fine-tuning
      for Transformer-based Masked Language-models", ACL 2022
"""

import torch
import torch.nn as nn

from model import SelfAttention


class LoRALinear(nn.Module):
    # h = W0 x + (alpha / r) * B A x
    # B starts at zero, so at step 0 the layer gives exactly the Phase-1 output
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        # A random (Gaussian, as in the LoRA paper), B zero
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.lora_A.t() @ self.lora_B.t())


class LoRMALinear(nn.Module):
    # LoRMA+ : h = (I + (alpha / r) * B A) W0 x
    # Doing A on (W0 x) first and then B, so the full d x d matrix B A is
    # never built (the reordering trick from the LoRMA paper)
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        # A random, B zero (same as the LoRMA+ paper)
        self.lorma_A = nn.Parameter(torch.randn(rank, base.out_features) * 0.02)
        self.lorma_B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x):
        h = self.base(x)
        return h + self.scale * (h @ self.lorma_A.t() @ self.lorma_B.t())


def wrap_attention_layers(model, wrapper, rank, alpha):
    # Putting the adapter on qkv_proj and out_proj of every attention layer.
    # q, k and v are joined into one qkv_proj in our model, so the adapter
    # covers all three (the LoRMA paper used only q and v).
    # Making a list first since the loop replaces layers while going through them
    for module in list(model.backbone.modules()):
        if isinstance(module, SelfAttention):
            module.qkv_proj = wrapper(module.qkv_proj, rank, alpha)
            module.out_proj = wrapper(module.out_proj, rank, alpha)


def add_zero_biases(model):
    # The Phase-1 Linear layers were built with bias=False, so there are no
    # biases for BitFit to train. Adding biases that start at zero (so the
    # output doesn't change at first) and training only those
    for module in model.backbone.modules():
        if isinstance(module, nn.Linear) and module.bias is None:
            module.bias = nn.Parameter(torch.zeros(module.out_features))


def get_blocks(backbone):
    # encdec only uses its encoder for NER, the other three use .blocks
    if hasattr(backbone, "encoder_blocks"):
        return backbone.encoder_blocks
    return backbone.blocks


def setup_method(model, method, rank=8, alpha=16, train_top_layers=4):
    # Changing the model first (lora, lorma and bitfit add new weights)
    if method == "lora":
        wrap_attention_layers(model, LoRALinear, rank, alpha)
    elif method == "lorma":
        wrap_attention_layers(model, LoRMALinear, rank, alpha)
    elif method == "bitfit":
        add_zero_biases(model)

    # Then choosing which weights get trained
    for name, param in model.named_parameters():
        if method == "full":
            param.requires_grad = True
        elif method == "lora":
            param.requires_grad = ("lora_" in name)
        elif method == "lorma":
            param.requires_grad = ("lorma_" in name)
        elif method == "bitfit":
            param.requires_grad = name.endswith(".bias")
        elif method == "partial":
            param.requires_grad = False

    if method == "partial":
        for block in get_blocks(model.backbone)[-train_top_layers:]:
            for param in block.parameters():
                param.requires_grad = True

        # final_norm / encoder_norm sits after the last block, so training it too
        for name, param in model.backbone.named_parameters():
            if "norm" in name and "blocks" not in name:
                param.requires_grad = True

    # The classifier head is new, so it's always trained
    for param in model.classifier.parameters():
        param.requires_grad = True


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
