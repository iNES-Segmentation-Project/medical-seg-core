"""
mit_stage.py

A single stage of the Mix Transformer (MiT) encoder used in SegFormer.

Each MiTStage consists of:
  1. OverlapPatchEmbed  — projects input to new embedding dim, reduces spatial resolution
  2. N × TransformerBlock — each block contains:
       a. LayerNorm → EfficientSelfAttention → residual
       b. LayerNorm → MixFFN → residual

The output is reshaped back from (B, N, C) → (B, C, H, W) to serve as
a standard 2D feature map passed to the next stage or decoder.

SegFormer-B0 has 4 stages with the following configurations:
  Stage 1: embed_dim=32,  depth=2, num_heads=1, sr_ratio=8
  Stage 2: embed_dim=64,  depth=2, num_heads=2, sr_ratio=4
  Stage 3: embed_dim=160, depth=2, num_heads=5, sr_ratio=2
  Stage 4: embed_dim=256, depth=2, num_heads=8, sr_ratio=1

Reference: SegFormer paper (Xie et al., 2021)
MMSegmentation reference: mmseg/models/backbones/mix_transformer.py
"""

import torch
import torch.nn as nn
from torch import Tensor

from .overlap_patch_embed import OverlapPatchEmbed
from .efficient_attention import EfficientSelfAttention
from .mix_ffn import MixFFN


class TransformerBlock(nn.Module):
    """
    One Transformer block: pre-norm SA + pre-norm FFN (both with residuals).

    Structure:
        x = x + Attention(LN(x))
        x = x + FFN(LN(x))

    Args:
        embed_dim  (int):   Channel dimension.
        num_heads  (int):   Number of attention heads.
        sr_ratio   (int):   Spatial reduction ratio for efficient attention.
        mlp_ratio  (float): FFN hidden expansion ratio.
        attn_drop  (float): Dropout on attention weights.
        proj_drop  (float): Dropout on projections / FFN.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = EfficientSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            sr_ratio=sr_ratio,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = MixFFN(
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            dropout=proj_drop,
        )

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        """
        Args:
            x : (B, N, C)   — flattened sequence, N = H * W
            H : int
            W : int

        Returns:
            x : (B, N, C)
        """
        # Pre-norm self-attention with residual
        x = x + self.attn(self.norm1(x), H, W)

        # Pre-norm Mix-FFN with residual
        x = x + self.ffn(self.norm2(x), H, W)

        return x  # (B, N, C)


class MiTStage(nn.Module):
    """
    One complete stage of the MiT encoder.

    Performs:
      1. Overlapping patch embedding (downsampling + projection)
      2. `depth` transformer blocks
      3. Final LayerNorm
      4. Reshape back to 2D spatial feature map

    Args:
        in_channels (int):   Input channel count from previous stage (or 3 for stage 1).
        embed_dim   (int):   Output embedding dimension for this stage.
        patch_size  (int):   Patch embedding kernel size (7 for stage 1, 3 for others).
        stride      (int):   Patch embedding stride (4 for stage 1, 2 for others).
        depth       (int):   Number of transformer blocks in this stage.
        num_heads   (int):   Number of attention heads.
        sr_ratio    (int):   Spatial reduction ratio for efficient attention.
        mlp_ratio   (float): FFN hidden expansion ratio (default: 4.0).
        attn_drop   (float): Dropout on attention weights.
        proj_drop   (float): Dropout on projections.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        stride: int,
        depth: int,
        num_heads: int,
        sr_ratio: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        # Step 1: Overlapping patch embedding
        self.patch_embed = OverlapPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            stride=stride,
        )

        # Step 2: Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            for _ in range(depth)
        ])

        # Step 3: Final layer norm (applied before reshape)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor):
        """
        Args:
            x : (B, C_in, H_in, W_in)   — 2D spatial feature map from previous stage

        Returns:
            out : (B, embed_dim, H_out, W_out)   — 2D feature map for next stage / decoder
            H_out, W_out : int                   — reduced spatial dimensions
        """
        # ── Patch Embedding ────────────────────────────────────────────────────
        # (B, C_in, H, W) → (B, N, embed_dim),  N = H_out * W_out
        x, H, W = self.patch_embed(x)

        # ── Transformer Blocks ─────────────────────────────────────────────────
        for block in self.blocks:
            x = block(x, H, W)  # (B, N, embed_dim)

        # ── Final LayerNorm ────────────────────────────────────────────────────
        x = self.norm(x)  # (B, N, embed_dim)

        # ── Reshape to 2D feature map ──────────────────────────────────────────
        # (B, N, embed_dim) → (B, embed_dim, H_out, W_out)
        B, N, C = x.shape
        x = x.permute(0, 2, 1).reshape(B, C, H, W)

        return x  # (B, embed_dim, H_out, W_out)