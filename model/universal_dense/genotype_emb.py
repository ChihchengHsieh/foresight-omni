import torch.nn as nn
import torch
from einops.layers.torch import Rearrange
from typing import List
from model.universal_dense.ops.norm import RMSNorm
from model.pos_emb import SinCosPositionalEncodingProvider
import logging


class GenotypeEmbedding(nn.Module):
    def __init__(
        self,
        intermediate_dims: List[int],
        patch_sizes: List[int],
        out_dim: int,
        # genotype_len: int = 111068,  # glaucoma SNPs
        genotype_len: int = 18000000,  # 17553457,  # 17534013,  # 17549723,  # => ending up with len around 67 thorugh double patching (patch_size=512)
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.genotype_len = genotype_len
        # self.emb = nn.Embedding(5, emb_dim)
        self.pos_provider = SinCosPositionalEncodingProvider(
            d_model=intermediate_dims[0],
            max_len=genotype_len,
        )

        pos = torch.sin(torch.arange(genotype_len)).unsqueeze(0)  # (B, len, 1)
        self.register_parameter("pos", nn.Parameter(pos, requires_grad=False))

        self.genotype_embedding = nn.Embedding(
            num_embeddings=5,
            embedding_dim=intermediate_dims[0],
        )

        compression = nn.ModuleList([])

        for i in range(len(intermediate_dims)):
            in_dim = intermediate_dims[i]
            d = out_dim if i == len(intermediate_dims) - 1 else intermediate_dims[i + 1]
            p_size = patch_sizes[i]
            compression.extend(
                [
                    RMSNorm(in_dim),
                    Rearrange("b l c -> b c l", c=in_dim),
                    nn.Conv1d(
                        kernel_size=p_size,
                        stride=p_size,
                        in_channels=in_dim,
                        out_channels=d,
                    ),  # (B, C, L)
                    Rearrange("b c l -> b l c", c=d),
                ]
            )

        compression.append(
            RMSNorm(out_dim),
        )

        self.compression = nn.Sequential(*compression)


    def forward(self, genotypes):
        emb_out = genotypes
        emb_out = self.genotype_embedding(emb_out.int())  # .transpose(1, 2)
        pos_emb_out = self.pos_provider(emb_out) + emb_out
        patched = self.compression(pos_emb_out)
        return patched


def build_genotype_embedding(args):
    return GenotypeEmbedding(
        out_dim=args.dim,
        patch_sizes=args.genotype_patch_sizes,
        intermediate_dims=args.genotype_emd_dims,  # 8 -> 64 -> 1024?
    )
