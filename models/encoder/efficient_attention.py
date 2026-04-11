"""
efficient_attention.py

Efficient Self-Attention with Spatial Reduction (SR) used in SegFormer's MiT encoder.

Standard multi-head self-attention has O(N²) complexity where N = H*W.
SegFormer reduces this by applying a spatial reduction (via Conv2d) to K and V
before computing attention, reducing the sequence length from N to N/R² where
R is the sr_ratio.

Reference: SegFormer paper (Xie et al., 2021), Section 3.1 — "Efficient Self-Attention"
MMSegmentation reference: mmseg/models/backbones/mix_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class EfficientSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with Spatial Reduction.

    For each attention layer, the key (K) and value (V) tensors are spatially
    reduced by a factor of `sr_ratio` using a strided convolution, which
    dramatically reduces the sequence length used in the attention computation.

    Query (Q) is kept at full resolution, so the output resolution is unchanged.

    Args:
        embed_dim  (int): Channel dimension (total embedding size).
        num_heads  (int): Number of attention heads. Must evenly divide embed_dim.
        sr_ratio   (int): Spatial reduction ratio for K and V.
                          - sr_ratio=1  → no reduction (standard MHSA)
                          - sr_ratio=8  → reduces H,W by 8× before computing K,V
        attn_drop  (float): Dropout on attention weights. Default: 0.0
        proj_drop  (float): Dropout on output projection. Default: 0.0
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int = 1,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1 / sqrt(d_k)

        # Q, K, V projections (no bias, matching original SegFormer)
        self.q = nn.Linear(embed_dim, embed_dim)
        self.kv = nn.Linear(embed_dim, embed_dim * 2)

        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Spatial reduction: reduce K,V sequence length from N to N/sr_ratio²
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(
                embed_dim,
                embed_dim,
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )
            self.sr_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x : (B, N, C)   — flattened spatial sequence, N = H * W
            H : int          — spatial height (needed to reshape for SR)
            W : int          — spatial width

        Returns:
            out : (B, N, C)  — same shape as input
        """
        B, N, C = x.shape  # N = H * W, C = embed_dim

        # ── Query ──────────────────────────────────────────────────────────────
        # (B, N, C) → (B, N, num_heads, head_dim) → (B, num_heads, N, head_dim)
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # ── Spatial Reduction for Key & Value ──────────────────────────────────
        if self.sr_ratio > 1:
            # Reshape back to spatial map: (B, N, C) → (B, C, H, W)
            x_sr = x.permute(0, 2, 1).reshape(B, C, H, W)

            # Strided conv reduces H,W by sr_ratio:
            # (B, C, H, W) → (B, C, H/R, W/R), N' = (H/R)*(W/R)
            x_sr = self.sr(x_sr)

            # Flatten back to sequence: (B, C, H', W') → (B, N', C)
            x_sr = x_sr.flatten(2).transpose(1, 2)
            x_sr = self.sr_norm(x_sr)
        else:
            x_sr = x  # no reduction; K,V computed from full-resolution sequence

        # ── Key & Value ────────────────────────────────────────────────────────
        # (B, N', C) → (B, N', 2*C) → split into k,v each (B, N', C)
        N_prime = x_sr.shape[1]
        kv = self.kv(x_sr).reshape(B, N_prime, 2, self.num_heads, self.head_dim)
        # kv: (B, N', 2, num_heads, head_dim) → (2, B, num_heads, N', head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)  # each: (B, num_heads, N', head_dim)

        # ── Scaled Dot-Product Attention ───────────────────────────────────────
        # attn: (B, num_heads, N, N')
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # (B, num_heads, N, head_dim) → (B, N, num_heads*head_dim) = (B, N, C)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # Output projection
        out = self.proj(out)
        out = self.proj_drop(out)

        return out  # (B, N, C)