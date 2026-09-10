import torch
import torch.nn as nn

from .ops.norm import RMSNorm
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ImagePatchEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        image_size: int,
        patch_size: int,
        img_channels: int = 3,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."
        # num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = img_channels * patch_height * patch_width
        self.patch_emb = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                p1=patch_height,
                p2=patch_width,
            ),
            RMSNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            RMSNorm(dim),
        )

    def forward(self, x: torch.Tensor):
        return self.patch_emb(x)


def build_patch_embedding(args):
    return ImagePatchEmbedding(args.dim, args.image_size, args.patch_size)
