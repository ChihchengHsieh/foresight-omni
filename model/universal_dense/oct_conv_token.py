import torch
from torch import nn

from model.universal_dense.conv_token import ConvTokenisation


def _to_3tuple(value, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    if len(value) != 3:
        raise ValueError(f"{name} must have three values, got {value}")
    return tuple(int(v) for v in value)


class OCTVolumeConvTokenisation(nn.Module):
    """
    Tokenise an OCT volume as ordered B-scan spatial tokens.

    Input shape is expected to be (B, S, C, H, W), where S is the number of
    B-scans. Each B-scan is encoded by the same 2D CNN tokeniser used for fundus
    images. The resulting slice-spatial tokens can either be resampled directly
    or first compressed per slice and passed through a slice-level transformer.
    """

    def __init__(
        self,
        dim: int,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        img_channels: int = 1,
        max_slices: int = 128,
        num_volume_tokens: int = 128,
        num_heads: int = 4,
        encoder_type: str = "flat_resampler",
        slice_tokens: int = 4,
        slice_transformer_layers: int = 1,
        tubelet_size: tuple[int, int, int] = (8, 32, 32),
        tubelet_transformer_layers: int = 2,
        tubelet_dropout: float = 0.1,
        max_image_size: int = 224,
        bscan_chunk_size: int = 256,
    ) -> None:
        super().__init__()
        if encoder_type not in {
            "flat_resampler",
            "slice_transformer",
            "slice_conv1d",
            "tubelet_vit",
        }:
            raise ValueError(
                f"Unsupported OCT encoder_type: {encoder_type}. "
                "Use 'flat_resampler', 'slice_transformer', 'slice_conv1d', "
                "or 'tubelet_vit'."
            )
        self.max_slices = max_slices
        self.num_volume_tokens = num_volume_tokens
        self.encoder_type = encoder_type
        self.slice_tokens = slice_tokens
        self.bscan_chunk_size = max(int(bscan_chunk_size), 1)

        if encoder_type == "tubelet_vit":
            self.slice_encoder = None
            self.slice_pos_embed = None
            self.tubelet_size = _to_3tuple(tubelet_size, "tubelet_size")
            tubelet_depth, tubelet_height, tubelet_width = self.tubelet_size
            self.tubelet_embed = nn.Conv3d(
                img_channels,
                dim,
                kernel_size=self.tubelet_size,
                stride=self.tubelet_size,
            )
            max_depth_tokens = max(max_slices // tubelet_depth, 1)
            max_height_tokens = max(max_image_size // tubelet_height, 1)
            max_width_tokens = max(max_image_size // tubelet_width, 1)
            self.tubelet_depth_pos = nn.Embedding(max_depth_tokens, dim)
            self.tubelet_height_pos = nn.Embedding(max_height_tokens, dim)
            self.tubelet_width_pos = nn.Embedding(max_width_tokens, dim)
            if tubelet_transformer_layers > 0:
                tubelet_layer = nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout=tubelet_dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.tubelet_transformer = nn.TransformerEncoder(
                    tubelet_layer,
                    num_layers=tubelet_transformer_layers,
                )
            else:
                self.tubelet_transformer = nn.Identity()
            self.tubelet_norm = nn.LayerNorm(dim)
        else:
            self.slice_encoder = ConvTokenisation(
                dim=dim,
                backbone_name=backbone_name,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                img_channels=img_channels,
            )
            self.slice_pos_embed = nn.Embedding(max_slices, dim)
            self.tubelet_embed = None
            self.tubelet_depth_pos = None
            self.tubelet_height_pos = None
            self.tubelet_width_pos = None
            self.tubelet_transformer = None
            self.tubelet_norm = None

        if encoder_type in {"slice_transformer", "slice_conv1d"}:
            self.slice_queries = nn.Parameter(torch.randn(slice_tokens, dim) * 0.02)
            self.slice_resampler = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                batch_first=True,
            )
            self.slice_resampler_norm = nn.LayerNorm(dim)
        else:
            self.slice_queries = None
            self.slice_resampler = None
            self.slice_resampler_norm = None

        if encoder_type == "slice_transformer":
            if slice_transformer_layers > 0:
                slice_layer = nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.slice_transformer = nn.TransformerEncoder(
                    slice_layer,
                    num_layers=slice_transformer_layers,
                )
            else:
                self.slice_transformer = nn.Identity()
            self.slice_conv = None
        elif encoder_type == "slice_conv1d":
            conv_layers = []
            for _ in range(max(slice_transformer_layers, 1)):
                conv_layers.extend(
                    [
                        nn.Conv1d(
                            dim,
                            dim,
                            kernel_size=5,
                            padding=2,
                            groups=dim,
                        ),
                        nn.Conv1d(dim, dim, kernel_size=1),
                        nn.GELU(),
                        nn.BatchNorm1d(dim),
                    ]
                )
            self.slice_conv = nn.Sequential(*conv_layers)
            self.slice_transformer = None
        elif encoder_type != "tubelet_vit":
            self.slice_transformer = None
            self.slice_conv = None
        else:
            self.slice_transformer = None
            self.slice_conv = None

        if num_volume_tokens > 0:
            self.volume_queries = nn.Parameter(torch.randn(num_volume_tokens, dim) * 0.02)
            self.volume_resampler = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_heads,
                batch_first=True,
            )
            self.volume_resampler_norm = nn.LayerNorm(dim)
        else:
            self.volume_queries = None
            self.volume_resampler = None
            self.volume_resampler_norm = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, C, H, W)
        returns: (B, num_volume_tokens, dim) if resampling is enabled,
                 otherwise (B, S * N, dim)
        """
        if x.ndim != 5:
            raise ValueError(
                f"OCTVolumeConvTokenisation expects (B, S, C, H, W), got {tuple(x.shape)}"
            )

        bsz, num_slices, channels, height, width = x.shape
        if num_slices > self.max_slices:
            raise ValueError(
                f"OCT volume has {num_slices} slices, but max_slices={self.max_slices}"
            )

        if self.encoder_type == "tubelet_vit":
            # Dataset layout is (B, S, C, H, W); Conv3d expects (B, C, D, H, W).
            x_3d = x.permute(0, 2, 1, 3, 4).contiguous()
            tubelets = self.tubelet_embed(x_3d)
            _, dim, depth_tokens, height_tokens, width_tokens = tubelets.shape

            if depth_tokens > self.tubelet_depth_pos.num_embeddings:
                raise ValueError(
                    f"OCT tubelet depth tokens={depth_tokens} exceeds max "
                    f"{self.tubelet_depth_pos.num_embeddings}. Increase oct_max_slices "
                    "or use a larger depth tubelet."
                )
            if height_tokens > self.tubelet_height_pos.num_embeddings:
                raise ValueError(
                    f"OCT tubelet height tokens={height_tokens} exceeds max "
                    f"{self.tubelet_height_pos.num_embeddings}. Increase oct_image_size "
                    "or use a larger height tubelet."
                )
            if width_tokens > self.tubelet_width_pos.num_embeddings:
                raise ValueError(
                    f"OCT tubelet width tokens={width_tokens} exceeds max "
                    f"{self.tubelet_width_pos.num_embeddings}. Increase oct_image_size "
                    "or use a larger width tubelet."
                )

            depth_ids = torch.arange(depth_tokens, device=x.device)
            height_ids = torch.arange(height_tokens, device=x.device)
            width_ids = torch.arange(width_tokens, device=x.device)
            pos = (
                self.tubelet_depth_pos(depth_ids)[:, None, None, :]
                + self.tubelet_height_pos(height_ids)[None, :, None, :]
                + self.tubelet_width_pos(width_ids)[None, None, :, :]
            )
            # tubelets: (B, D', H', W', dim), retaining 3D location before flattening.
            volume_tokens = tubelets.permute(0, 2, 3, 4, 1).contiguous()
            volume_tokens = (volume_tokens + pos[None, ...]).reshape(
                bsz,
                depth_tokens * height_tokens * width_tokens,
                dim,
            )
            volume_tokens = self.tubelet_norm(self.tubelet_transformer(volume_tokens))
            if self.volume_resampler is None:
                return volume_tokens

            queries = self.volume_queries.unsqueeze(0).expand(bsz, -1, -1)
            resampled, _ = self.volume_resampler(
                query=queries,
                key=volume_tokens,
                value=volume_tokens,
                need_weights=False,
            )
            return self.volume_resampler_norm(queries + resampled)

        # x: (B, S, C, H, W) -> (B * S, C, H, W), so each B-scan uses
        # the same 2D conv-token encoder as fundus.
        x_bscans = x.reshape(bsz * num_slices, channels, height, width)
        bscan_token_chunks = []
        for start in range(0, x_bscans.shape[0], self.bscan_chunk_size):
            end = start + self.bscan_chunk_size
            bscan_token_chunks.append(self.slice_encoder(x_bscans[start:end]))
        bscan_tokens = torch.cat(bscan_token_chunks, dim=0)
        # bscan_tokens: (B * S, N, dim), where N is spatial CNN tokens per slice.
        _, tokens_per_slice, dim = bscan_tokens.shape
        # bscan_tokens: (B, S, N, dim), restoring the OCT slice axis.
        bscan_tokens = bscan_tokens.reshape(
            bsz, num_slices, tokens_per_slice, dim
        )

        slice_ids = torch.arange(num_slices, device=x.device)
        if self.encoder_type in {"slice_transformer", "slice_conv1d"}:
            # slice_source: (B * S, N, dim), treating each B-scan independently.
            slice_source = bscan_tokens.reshape(
                bsz * num_slices, tokens_per_slice, dim
            )
            # slice_queries: (B * S, slice_tokens, dim), learned per-slice queries.
            slice_queries = self.slice_queries.unsqueeze(0).expand(
                bsz * num_slices, -1, -1
            )
            # slice_tokens: (B * S, slice_tokens, dim), compressed B-scan tokens.
            slice_tokens, _ = self.slice_resampler(
                query=slice_queries,
                key=slice_source,
                value=slice_source,
                need_weights=False,
            )
            slice_tokens = self.slice_resampler_norm(slice_queries + slice_tokens)
            # slice_tokens: (B, S, slice_tokens, dim), restoring slice axis.
            slice_tokens = slice_tokens.reshape(
                bsz, num_slices, self.slice_tokens, dim
            )
            # slice_pos: (1, S, 1, dim), broadcast over B and per-slice tokens.
            slice_pos = self.slice_pos_embed(slice_ids)[None, :, None, :]
            # volume_tokens: (B, S * slice_tokens, dim), slice-aware sequence.
            volume_tokens = (slice_tokens + slice_pos).reshape(
                bsz, num_slices * self.slice_tokens, dim
            )
            if self.encoder_type == "slice_transformer":
                # volume_tokens: (B, S * slice_tokens, dim), after slice transformer.
                volume_tokens = self.slice_transformer(volume_tokens)
            else:
                # Conv1d expects (B, dim, L), where L = S * slice_tokens.
                conv_input = volume_tokens.transpose(1, 2)
                # conv_output: (B, dim, L), local slice-neighbour mixing.
                conv_output = self.slice_conv(conv_input)
                # volume_tokens: (B, L, dim), back to token sequence layout.
                volume_tokens = (conv_input + conv_output).transpose(1, 2)
        else:
            # slice_pos: (1, S, 1, dim), broadcast over B and in-slice tokens.
            slice_pos = self.slice_pos_embed(slice_ids)[None, :, None, :]
            # bscan_tokens: (B, S, N, dim), now with slice-order information.
            bscan_tokens = bscan_tokens + slice_pos

            # volume_tokens: (B, S * N, dim), one long OCT token sequence.
            volume_tokens = bscan_tokens.reshape(
                bsz, num_slices * tokens_per_slice, dim
            )
        if self.volume_resampler is None:
            return volume_tokens

        # queries: (B, num_volume_tokens, dim), learned tokens that attend over
        # the full OCT volume token sequence.
        queries = self.volume_queries.unsqueeze(0).expand(bsz, -1, -1)
        # resampled: (B, num_volume_tokens, dim), compressed OCT representation.
        resampled, _ = self.volume_resampler(
            query=queries,
            key=volume_tokens,
            value=volume_tokens,
            need_weights=False,
        )
        # return: (B, num_volume_tokens, dim), e.g. (B, 128, 256).
        return self.volume_resampler_norm(queries + resampled)


def build_oct_conv_token(args):
    return OCTVolumeConvTokenisation(
        dim=args.dim,
        backbone_name=getattr(args, "oct_backbone_name", "resnet18"),
        pretrained=not getattr(args, "oct_no_pretrained", False),
        freeze_backbone=getattr(args, "oct_freeze_backbone", False),
        img_channels=getattr(args, "oct_img_channels", 1),
        max_slices=getattr(args, "oct_max_slices", 128),
        num_volume_tokens=getattr(args, "oct_num_tokens", 128),
        num_heads=getattr(args, "n_heads", 4),
        encoder_type=getattr(args, "oct_encoder_type", "flat_resampler"),
        slice_tokens=getattr(args, "oct_slice_tokens", 4),
        slice_transformer_layers=getattr(args, "oct_slice_transformer_layers", 1),
        tubelet_size=getattr(args, "oct_tubelet_size", (8, 32, 32)),
        tubelet_transformer_layers=getattr(args, "oct_tubelet_transformer_layers", 2),
        tubelet_dropout=getattr(args, "oct_tubelet_dropout", 0.1),
        max_image_size=getattr(args, "oct_image_size", None)
        or getattr(args, "image_size", 224),
        bscan_chunk_size=getattr(args, "oct_bscan_chunk_size", 256),
    )
