"""
overlap_patch_embed.py

Overlapping Patch Embedding used in SegFormer's Mix Transformer (MiT) encoder.

Unlike ViT's non-overlapping patch embedding, this uses a strided convolution
with padding so that patches overlap — preserving local continuity at boundaries.

Reference: SegFormer paper (Xie et al., 2021), Section 3.1
MMSegmentation reference: mmseg/models/backbones/mix_transformer.py
"""

import torch
import torch.nn as nn
from torch import Tensor


class OverlapPatchEmbed(nn.Module):
    """
    Overlapping Patch Embedding via strided convolution.

    Converts an input feature map (or raw image) into a sequence of
    patch embeddings while preserving spatial overlap between patches.

    Used at the beginning of each MiT stage to downsample and project
    the spatial resolution into a new embedding dimension.

    Args:
        in_channels (int):  Number of input channels (3 for raw image, or
                            previous stage's embed_dim for later stages).
        embed_dim   (int):  Output embedding dimension (channel count after projection).
        patch_size  (int):  Kernel size of the convolution (typically 7 for stage 1,
                            3 for stages 2–4 in SegFormer-B0).
        stride      (int):  Convolution stride that controls spatial downsampling
                            (4 for stage 1, 2 for stages 2–4).
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        stride: int,
    ):
        super().__init__()

        # padding is set so spatial size contracts only by the stride factor
        padding = patch_size // 2  # e.g., kernel=7 → pad=3, kernel=3 → pad=1

        # Strided conv: projects in_channels → embed_dim, downsamples H,W by `stride`
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=padding,
        )

        # Layer norm applied over the embed_dim dimension
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor):
        """
        Args:
            x: (B, C_in, H, W)

        Returns:
            x_flat : (B, H'*W', embed_dim)   — flattened sequence for attention
            H'      : int                      — spatial height after patch embed
            W'      : int                      — spatial width  after patch embed
        """
        # (B, C_in, H, W) → (B, embed_dim, H', W')
        x = self.proj(x)

        B, C, H, W = x.shape

        # Flatten spatial dims: (B, C, H', W') → (B, H'*W', C)
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)

        # LayerNorm over the channel (embed_dim) dimension
        x = self.norm(x)

        return x, H, W
