import torch
import torch.nn as nn

from .ops.norm import RMSNorm
from einops.layers.torch import Rearrange
from .patch_emb import ImagePatchEmbedding


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MultipleImagesEmbedding(nn.Module):
    def __init__(
        self,
        patch_emb: ImagePatchEmbedding,
        dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.patch_emb = patch_emb
        self.img_splitter = nn.Parameter(torch.randn(1, dim), requires_grad=True)

    def forward(self, x: list[torch.Tensor]):
        patched_imgs = [self.patch_emb(img.unsqueeze(0)).squeeze(0) for img in x]
        splitters = [self.img_splitter for _ in range(len(patched_imgs) - 1)]
        concatenated_imgs = torch.cat(
            [
                val
                for pair in zip(patched_imgs, splitters + [None])
                for val in pair
                if val is not None
            ],
            dim=0,
        )
        return concatenated_imgs


def build_multiple_images_embedding(args):
    return MultipleImagesEmbedding(
        patch_emb=ImagePatchEmbedding(
            args.dim,
            args.image_size,
            args.patch_size,
        ),
        dim=args.dim,
    )
