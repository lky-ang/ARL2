"""
LoRA (Low-Rank Adaptation) utilities for fine-tuning.

Provides LoRALinear wrapper and helper to apply LoRA to FFN layers
in specified blocks of CausalWanModel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA wrapper around an existing nn.Linear.
    output = original_linear(x) + scale * B(A(x))
    where A: [in, rank], B: [rank, out], scale = alpha / rank.

    The original linear weights are frozen; only A and B are trainable.
    """

    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.linear = linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        in_features = linear.in_features
        out_features = linear.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with Kaiming, B with zeros (so initial LoRA output = 0)
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.linear(x)
        lora_out = (x @ self.lora_A) @ self.lora_B * self.scale
        return base_out + lora_out

    def merge_weights(self):
        """Merge LoRA weights into the base linear for inference."""
        with torch.no_grad():
            self.linear.weight.data += (
                self.lora_B.t() @ self.lora_A.t() * self.scale
            )
        return self.linear


def apply_lora_to_ffn(
    block: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
) -> list:
    """
    Apply LoRA to the FFN layers (index 0 and 2) in a CausalWanAttentionBlock.

    The FFN is: nn.Sequential(Linear(dim, ffn_dim), GELU, Linear(ffn_dim, dim))
    We wrap both Linear layers with LoRA.

    Returns list of newly created LoRA parameters for the optimizer.
    """
    ffn = block.ffn
    lora_params = []

    # FFN[0]: Linear(dim, ffn_dim)
    if isinstance(ffn[0], nn.Linear):
        lora_0 = LoRALinear(ffn[0], rank=rank, alpha=alpha)
        ffn[0] = lora_0
        lora_params.extend([lora_0.lora_A, lora_0.lora_B])

    # FFN[2]: Linear(ffn_dim, dim)
    if isinstance(ffn[2], nn.Linear):
        lora_2 = LoRALinear(ffn[2], rank=rank, alpha=alpha)
        ffn[2] = lora_2
        lora_params.extend([lora_2.lora_A, lora_2.lora_B])

    return lora_params
