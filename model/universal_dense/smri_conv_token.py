"""Compact 3D CNN tokeniser for T1 structural MRI volumes."""

from __future__ import annotations

import torch
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv3d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Smri3DConvTokenisation(nn.Module):
    """Encode ``(B, 1, D, H, W)`` T1 volumes into learned summary tokens."""

    def __init__(
        self,
        dim: int,
        base_channels: int = 16,
        num_tokens: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if base_channels < 4:
            raise ValueError("smri_base_channels must be >= 4")
        if num_tokens < 1:
            raise ValueError("smri_num_tokens must be >= 1")
        if dim % num_heads != 0:
            raise ValueError(
                f"Model dim={dim} must be divisible by smri_num_heads={num_heads}"
            )

        channels = [
            int(base_channels),
            int(base_channels * 2),
            int(base_channels * 4),
            int(base_channels * 8),
        ]
        self.encoder = nn.Sequential(
            ConvBlock3d(1, channels[0], stride=2),
            ConvBlock3d(channels[0], channels[1], stride=2),
            ConvBlock3d(channels[1], channels[2], stride=2),
            ConvBlock3d(channels[2], channels[3], stride=2),
        )
        self.spatial_projection = nn.Sequential(
            nn.LayerNorm(channels[-1]),
            nn.Linear(channels[-1], dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.queries = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.resampler = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != 1:
            raise ValueError(
                "Smri3DConvTokenisation expects (B, 1, D, H, W), "
                f"got {tuple(x.shape)}"
            )
        features = self.encoder(x)
        spatial_tokens = features.flatten(2).transpose(1, 2)
        spatial_tokens = self.spatial_projection(spatial_tokens)
        queries = self.queries.unsqueeze(0).expand(x.shape[0], -1, -1)
        output, _ = self.resampler(
            queries, spatial_tokens, spatial_tokens, need_weights=False
        )
        return self.output_norm(queries + output)


def build_smri_conv_token(args) -> Smri3DConvTokenisation:
    return Smri3DConvTokenisation(
        dim=args.dim,
        base_channels=args.smri_base_channels,
        num_tokens=args.smri_num_tokens,
        num_heads=args.smri_num_heads,
        dropout=args.smri_dropout_p,
    )
