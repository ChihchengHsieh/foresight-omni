import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class SinCosPositionalEncodingProvider(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # If d_model is odd, we need to ignore the last dimension when applying cosine
            cosine_values = torch.cos(position * div_term)
            pe[:, 0, 1::2] = cosine_values[:, : pe.size(2) // 2]
        else:
            # Apply cosine to odd indices (1, 3, 5, ...) when d_model is even
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_parameter("pe", nn.Parameter(pe, requires_grad=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        return self.pe[:, : x.size(1), :]


class RMSNormCustom(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class GenotypeEncodingLayer(nn.Module):
    def __init__(self, in_dim, out_dim, patch_size):
        super().__init__()
        self.conv = nn.Conv1d(
            kernel_size=patch_size,
            stride=patch_size,
            in_channels=in_dim,
            out_channels=out_dim,
            padding=0,
        )
        self.norm = RMSNormCustom(in_dim)
        self.after_norm_rearrange = Rearrange("b l c -> b c l", c=in_dim)
        self.output_rearrange = Rearrange("b c l -> b l c", c=out_dim)

    def forward(self, x):
        x = self.norm(x)
        x = self.after_norm_rearrange(x)
        x = self.conv(x)
        x = self.output_rearrange(x)
        return x


class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B, L, D)
        attn_scores = self.attn(x)  # (B, L, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, L, 1)
        weighted_sum = (x * attn_weights).sum(dim=1)  # (B, D)
        return weighted_sum


# class GenotypeNNEncoder(nn.Module):
#     def __init__(self, d_in=6, d_emb=32, mlp_dims=[64, 32], output_dim=1):
#         super().__init__()
#         self.embedding = nn.Linear(d_in, d_emb)     # (B, L, 6) → (B, L, d_emb)
#         self.attn_pool = AttentionPooling(d_emb)    # (B, L, d_emb) → (B, d_emb)
#         self.mlp = nn.Sequential(
#             nn.Linear(d_emb, mlp_dims[0]),
#             nn.ReLU(),
#             nn.Linear(mlp_dims[0], mlp_dims[1]),
#             nn.ReLU(),
#             nn.Linear(mlp_dims[1], output_dim)
#         )

#     def forward(self, x):
#         x = self.embedding(x)       # (B, L, d_emb)
#         x = self.attn_pool(x)       # (B, d_emb)
#         x = self.mlp(x)             # (B, output_dim)
#         return x


class GenotypeNNEncoder(nn.Module):
    def __init__(
        self,
        d_in=6,
        d_emb=32,
        mlp_dims=[2048, 1024],
        output_dim=1,
        genotype_len=110540,
    ):
        super().__init__()
        self.embedding = nn.Linear(d_in, 1)  # (B, L, 6) → (B, L, d_emb)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            # nn.Linear(genotype_len, mlp_dims[0]),
            # nn.ReLU(),
            # nn.Linear(mlp_dims[0], mlp_dims[1]),
            # nn.ReLU(),
            # nn.Linear(mlp_dims[1], output_dim),
            nn.Linear(genotype_len, output_dim),
        )

    def forward(self, x):
        x = self.embedding(x)  # (B, L, 1)
        x = x.squeeze(-1)  # (B, L)
        x = self.mlp(x) # (B, output_dim)

        # unsqueeze 1 and 2 dimensions
        x = x[:, None, None, :] # (B, 1, output_dim)
        return x


class GenotypeDecodingLayer(nn.Module):
    def __init__(self, in_dim, out_dim, patch_size, output_padding=0):
        super().__init__()
        self.conv = nn.ConvTranspose1d(
            kernel_size=patch_size,
            stride=patch_size,
            in_channels=in_dim,
            out_channels=out_dim,
            padding=0,
            output_padding=output_padding,
        )
        self.norm = RMSNormCustom(out_dim)
        self.after_norm_rearrange = Rearrange("b l c -> b c l")
        self.output_rearrange = Rearrange("b c l -> b l c", c=out_dim)

    def forward(self, x):
        x = self.after_norm_rearrange(x)
        x = self.conv(x)
        x = self.output_rearrange(x)
        x = self.norm(x)
        return x


class GenotypeConvEncoder(nn.Module):
    def __init__(self, intermediate_dims, patch_sizes, genotype_len):
        super().__init__()
        self.genotype_len = genotype_len
        self.compression = nn.ModuleList([])
        self.pos_emb = SinCosPositionalEncodingProvider(
            intermediate_dims[0], genotype_len
        )

        for i in range(len(intermediate_dims) - 1):
            in_dim = intermediate_dims[i]
            d = intermediate_dims[i + 1]
            p_size = patch_sizes[i]
            self.compression.append(GenotypeEncodingLayer(in_dim, d, p_size))
            # self.compression.extend(
            #     [
            #         RMSNormCustom(in_dim),
            #         Rearrange("b l c -> b c l", c=in_dim),
            #         nn.Conv1d(
            #             kernel_size=p_size,
            #             stride=p_size,
            #             in_channels=in_dim,
            #             out_channels=d,
            #             padding=0,
            #         ),
            #         Rearrange("b c l -> b l c", c=d),
            #     ]
            # )
        # use an example tensor to check the shape of the output in each layer

        # with torch.no_grad():
        #     example_tensor = torch.zeros((1, genotype_len, intermediate_dims[0]))
        #     n_genotypes_elements = example_tensor.numel()
        #     self.encoded_lens = []
        #     for i, layer in enumerate(self.compression):
        #         example_tensor = layer(example_tensor)
        #         self.encoded_lens.append(example_tensor.shape[1])
        #         # print the compression rate and shape
        #         print(f"Compression rate in {i} layer: {example_tensor.numel() / n_genotypes_elements:.2f}")
        #         print(f"Encoded shape in {i} layer: ", example_tensor.shape)

        self.compression.append(RMSNormCustom(d))
        print("Encoder architecture: ", self.compression)

        self.has_announce = False

    def insert_last_encoding_layer(self, in_dim, out_dim, patch_size):
        # remove the last layer
        self.compression.pop(len(self.compression) - 1)
        # insert last encoding layer before the RMSNorm
        self.compression.extend(
            [
                GenotypeEncodingLayer(in_dim, out_dim, patch_size),
                RMSNormCustom(out_dim),
            ]
        )
        print("Genotype Encoder Architecture after inserting the last layer: ")
        print(self.compression)

    def forward(self, genotypes):
        print("[Encoder] Genotype shape: ", genotypes.shape)
        output = genotypes + self.pos_emb(genotypes)
        for i, layer in enumerate(self.compression):
            output = layer(output)
            if not self.has_announce:
                print(f"Compression shape in {i} layer: ", output.shape)
                print(
                    f"Compression rate in {i} layer: {output.numel()/ genotypes.numel():.2f}"
                )
        self.has_announce = True
        return output


class GenotypeDecoder(nn.Module):
    def __init__(self, intermediate_dims, patch_sizes, genotype_len):
        super().__init__()
        self.genotype_len = genotype_len
        reversed_dims = list(reversed(intermediate_dims))
        reversed_patches = list(reversed(patch_sizes))
        self.decompression = nn.ModuleList([])
        for i in range(len(reversed_dims) - 1):
            in_dim = reversed_dims[i]
            d = reversed_dims[i + 1]
            p_size = reversed_patches[i]
            self.decompression.append(GenotypeDecodingLayer(in_dim, d, p_size + 1))
            # self.decompression.extend(
            #     [
            #         RMSNormCustom(in_dim),
            #         Rearrange("b l c -> b c l"),
            #         build_conv_transpose_for_conv1d(p_size, in_dim, d),
            #         # nn.ConvTranspose1d(
            #         #     in_channels=in_dim,
            #         #     out_channels=d,
            #         #     kernel_size=p_size,
            #         #     stride=p_size,
            #         #     output_padding=output_padding,
            #         # ),
            #         Rearrange("b c l -> b l c"),
            #     ]
            # )
        self.has_announced_shape = False
        print("Decoder architecture: ", self.decompression)

    def forward(self, x):
        for i, layer in enumerate(self.decompression):
            x = layer(x)
            if not self.has_announced_shape:
                print(f"Decompression shape in {i} layer: ", x.shape)
        # Ensure that the output is cut back to the original size
        return x


class GenotypeAutoencoder(nn.Module):
    def __init__(self, intermediate_dims, patch_sizes, genotype_len):
        super().__init__()
        self.genotype_len = genotype_len
        print("Genotype len: ", genotype_len)
        self.encoder = GenotypeConvEncoder(intermediate_dims, patch_sizes, genotype_len)
        self.decoder = GenotypeDecoder(intermediate_dims, patch_sizes, genotype_len)
        self.has_report_compression_rate = False

    def forward(self, genotypes):
        encoded = self.encoder(genotypes)
        if not self.has_report_compression_rate:
            print(f"Original size: {genotypes.shape}")
            print(f"Encoded size: {encoded.shape}")
            print(f"Compression rate: {genotypes.numel() / encoded.numel():.2f}")
            self.has_report_compression_rate = True
        decoded = F.softmax(self.decoder(encoded), dim=2)
        # Trim the decoded output to the original input size
        assert (
            decoded.shape[1] >= self.genotype_len
        ), f"Decoded shape [{decoded.shape}] is smaller than genotype length [{self.genotype_len}]"
        output = decoded[:, : self.genotype_len, :]
        return output
