"""
CS781 Capstone - Phase 3: Rhetorical Role Prediction (RR)
Fine-tuning methods: full, lora, lorma, ia3, adalora
Knowledge distillation is in distill.py

References:
    - LoRA: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022
    - LoRMA: Bihany, Patel and Modi, "LoRMA: Low-Rank Multiplicative Adaptation
      for LLMs", Findings of ACL 2025
    - IA3: Liu et al., "Few-Shot Parameter-Efficient Fine-Tuning is Better and
      Cheaper than In-Context Learning", NeurIPS 2022
    - AdaLoRA: Zhang et al., "AdaLoRA: Adaptive Budget Allocation for
      Parameter-Efficient Fine-Tuning", ICLR 2023
    - peft library: https://github.com/huggingface/peft
"""

import torch
import torch.nn as nn

from model import SelfAttention


# LoRALinear, LoRMALinear and wrap_attention_layers are copied from phase2/methods.py
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


def wrap_attention_layers(model, wrapper, rank, alpha):
    # Putting the adapter on qkv_proj and out_proj of every attention layer
    for module in list(model.backbone.modules()):
        if isinstance(module, SelfAttention):
            module.qkv_proj = wrapper(module.qkv_proj, rank, alpha)
            module.out_proj = wrapper(module.out_proj, rank, alpha)


def add_peft_adapter(model, method, rank, alpha, total_steps):
    # Adding ia3 or adalora to the backbone with the peft library
    from peft import AdaLoraConfig, IA3Config, get_peft_model

    if method == "ia3":
        # IA3: learned scaling vectors on the attention and feed-forward layers
        peft_config = IA3Config(
            target_modules=["qkv_proj", "w_out"],
            feedforward_modules=["w_out"]
        )
    else:
        # AdaLoRA: LoRA that gives more rank to the more useful layers during training
        peft_config = AdaLoraConfig(
            target_r=rank,
            lora_alpha=alpha,
            target_modules=["qkv_proj", "out_proj", "in_proj"],
            total_step=total_steps
        )

    model.backbone = get_peft_model(model.backbone, peft_config)


def setup_method(model, method, rank=8, alpha=16, total_steps=None):
    # Adding the new weights for the chosen method
    if method == "lora":
        wrap_attention_layers(model, LoRALinear, rank, alpha)
    elif method == "lorma":
        wrap_attention_layers(model, LoRMALinear, rank, alpha)
    elif method in ["ia3", "adalora"]:
        add_peft_adapter(model, method, rank, alpha, total_steps)

    # Choosing which backbone weights get trained, peft does this itself for ia3 and adalora
    for name, param in model.backbone.named_parameters():
        if method == "full":
            param.requires_grad = True
        elif method == "lora":
            param.requires_grad = ("lora_" in name)
        elif method == "lorma":
            param.requires_grad = ("lorma_" in name)

    # Sentence head is always trained
    for name, param in model.named_parameters():
        if not name.startswith("backbone."):
            param.requires_grad = True


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total
