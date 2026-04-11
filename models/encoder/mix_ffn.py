"""
mix_ffn.py

Mix-FFN (Feed-Forward Network with Depthwise Convolution) used in SegFormer's MiT.

Unlike standard Transformer FFN (two linear layers + activation), Mix-FFN inserts
a 3×3 depthwise convolution between the two linear projections. This encodes
local positional information without needing positional embeddings.

The design is intentional: SegFormer removes positional encodings entirely
and relies on Mix-FFN for local context — which also makes the model resolution-agnostic.

Reference: SegFormer paper (Xie et al., 2021), Section 3.1 — "Mix-FFN"
MMSegmentation reference: mmseg/models/backbones/mix_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MixFFN(nn.Module):
    """
    Mix-FFN: two linear layers with a depthwise convolution in between.

    Structure:
        Linear(C → C_hidden) → DWConv(3×3) → GELU → Linear(C_hidden → C) → Dropout

    The depthwise convolution operates on the 2D spatial feature map
    (not the flattened sequence), so the sequence is reshaped before and after.

    Args:
        embed_dim     (int):   Input (and output) channel dimension C.
        mlp_ratio     (float): Hidden expansion ratio. C_hidden = embed_dim * mlp_ratio.
                               SegFormer-B0 uses mlp_ratio=4 (i.e., C_hidden = 4*C).
        dropout       (float): Dropout applied after the final linear projection. Default: 0.0
    """

    def __init__(
        self,
        embed_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        # First linear projection: expand channels
        self.fc1 = nn.Linear(embed_dim, hidden_dim)

        # 3×3 depthwise convolution (groups=hidden_dim → per-channel operation)
        # padding=1 preserves H, W
        self.dw_conv = nn.Conv2d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_dim,  # depthwise: one filter per channel
        )

        # Second linear projection: project back to embed_dim
        self.fc2 = nn.Linear(hidden_dim, embed_dim)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x : (B, N, C)    — flattened sequence, N = H * W
            H : int           — spatial height (needed for DWConv reshape)
            W : int           — spatial width

        Returns:
            out : (B, N, C)   — same shape as input
        """
        B, N, C = x.shape  # N = H * W

        # ── Step 1: Linear expand ──────────────────────────────────────────────
        # (B, N, C) → (B, N, hidden_dim)
        x = self.fc1(x)

        # ── Step 2: Reshape for DWConv ─────────────────────────────────────────
        # (B, N, hidden_dim) → (B, hidden_dim, H, W)
        x = x.permute(0, 2, 1).reshape(B, -1, H, W)

        # ── Step 3: 3×3 Depthwise Conv → preserves H, W ───────────────────────
        # (B, hidden_dim, H, W) → (B, hidden_dim, H, W)
        x = self.dw_conv(x)

        # ── Step 4: Flatten back to sequence ──────────────────────────────────
        # (B, hidden_dim, H, W) → (B, N, hidden_dim)
        x = x.flatten(2).transpose(1, 2)

        # ── Step 5: Activation + Dropout + Linear project back ─────────────────
        x = self.act(x)
        x = self.drop(x)

        # (B, N, hidden_dim) → (B, N, C)
        x = self.fc2(x)
        x = self.drop(x)

        return x  # (B, N, C)