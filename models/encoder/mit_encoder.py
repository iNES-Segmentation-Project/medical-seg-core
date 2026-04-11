"""
mit_encoder.py

Mix Transformer (MiT) Encoder — SegFormer-B0 variant.

This module assembles 4 MiTStage instances with the exact hyperparameters
defined for SegFormer-B0 in the original paper and MMSegmentation config.

SegFormer-B0 Configuration (from paper Table 1 and mmseg config):
  Input: (B, 3, H, W)   — e.g., H=W=512 for CamVid

  Stage | embed_dim | depth | num_heads | sr_ratio | patch_size | stride | out_resolution*
  ------|-----------|-------|-----------|----------|------------|--------|---------------
    1   |    32     |   2   |     1     |    8     |     7      |   4    |  H/4  × W/4
    2   |    64     |   2   |     2     |    4     |     3      |   2    |  H/8  × W/8
    3   |   160     |   2   |     5     |    2     |     3      |   2    |  H/16 × W/16
    4   |   256     |   2   |     8     |    1     |     3      |   2    |  H/32 × W/32

  * Resolution assumes exact powers-of-2 input size.

Output: [c1, c2, c3, c4]
  c1: (B,  32, H/4,  W/4 )
  c2: (B,  64, H/8,  W/8 )
  c3: (B, 160, H/16, W/16)
  c4: (B, 256, H/32, W/32)

Reference: SegFormer paper (Xie et al., 2021), Table 1
MMSegmentation config: configs/segformer/segformer_mit-b0_*.py
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import List

from .mit_stage import MiTStage


# ─── SegFormer-B0 Stage Configurations ────────────────────────────────────────
# Each dict maps to the constructor args of MiTStage.
# These values are fixed and must NOT be changed per the Architecture Rules.

B0_STAGE_CONFIGS = [
    # Stage 1: large kernel (7×7) + stride 4 for initial large downsampling
    dict(
        embed_dim=32,
        patch_size=7,
        stride=4,
        depth=2,
        num_heads=1,
        sr_ratio=8,
    ),
    # Stage 2: standard 3×3 kernel, stride 2 for ×2 downsampling
    dict(
        embed_dim=64,
        patch_size=3,
        stride=2,
        depth=2,
        num_heads=2,
        sr_ratio=4,
    ),
    # Stage 3: wider embedding to increase capacity at finer spatial scale
    dict(
        embed_dim=160,
        patch_size=3,
        stride=2,
        depth=2,
        num_heads=5,
        sr_ratio=2,
    ),
    # Stage 4: sr_ratio=1 → no spatial reduction (full MHSA at this resolution)
    dict(
        embed_dim=256,
        patch_size=3,
        stride=2,
        depth=2,
        num_heads=8,
        sr_ratio=1,
    ),
]


class MiTEncoder(nn.Module):
    """
    Mix Transformer Encoder (MiT-B0).

    Produces a list of 4 hierarchical feature maps [c1, c2, c3, c4]
    at progressively reduced spatial resolutions.

    This encoder is structurally identical to SegFormer-B0's backbone.
    Decoder and loss modules are fully decoupled from this class.

    Args:
        in_channels (int): Number of input image channels. Default: 3 (RGB).
        drop_path_rate (float): Stochastic depth rate. Set to 0.0 for simplicity
                                (B0 baseline typically uses 0.0).
        attn_drop (float): Dropout on attention weights. Default: 0.0
        proj_drop (float): Dropout on projections / FFN. Default: 0.0
    """

    def __init__(
        self,
        in_channels: int = 3,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()

        self.stages = nn.ModuleList()

        # Build 4 stages; the in_channels of each stage is the embed_dim of the previous
        prev_channels = in_channels
        for cfg in B0_STAGE_CONFIGS:
            stage = MiTStage(
                in_channels=prev_channels,
                embed_dim=cfg["embed_dim"],
                patch_size=cfg["patch_size"],
                stride=cfg["stride"],
                depth=cfg["depth"],
                num_heads=cfg["num_heads"],
                sr_ratio=cfg["sr_ratio"],
                mlp_ratio=4.0,        # fixed for all B0 stages
                attn_drop=attn_drop,
                proj_drop=proj_drop,
            )
            self.stages.append(stage)
            prev_channels = cfg["embed_dim"]

    def forward(self, x: Tensor) -> List[Tensor]:
        """
        Args:
            x : (B, 3, H, W)   — input image batch

        Returns:
            List of 4 feature maps:
              c1 : (B,  32, H/4,  W/4 )
              c2 : (B,  64, H/8,  W/8 )
              c3 : (B, 160, H/16, W/16)
              c4 : (B, 256, H/32, W/32)
        """
        features = []

        for stage in self.stages:
            # Each stage takes a 2D feature map and returns a 2D feature map
            x = stage(x)         # (B, embed_dim_i, H_i, W_i)
            features.append(x)

        # features = [c1, c2, c3, c4]
        return features


# ─── Quick sanity check (run this file directly) ──────────────────────────────
if __name__ == "__main__":
    import torch

    model = MiTEncoder(in_channels=3)
    model.eval()

    # CamVid default resolution: 360×480, but use 512×512 for clean power-of-2 test
    dummy = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        c1, c2, c3, c4 = model(dummy)

    print("SegFormer-B0 Encoder output shapes:")
    print(f"  c1 : {tuple(c1.shape)}  — expected (2,  32, 128, 128)")
    print(f"  c2 : {tuple(c2.shape)}  — expected (2,  64,  64,  64)")
    print(f"  c3 : {tuple(c3.shape)}  — expected (2, 160,  32,  32)")
    print(f"  c4 : {tuple(c4.shape)}  — expected (2, 256,  16,  16)")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {n_params:,}")