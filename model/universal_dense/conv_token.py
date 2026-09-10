import torch
from torch import nn
from einops.layers.torch import Rearrange
import torchvision.models as models
from .ops.norm import RMSNorm

# assuming you already have RMSNorm defined somewhere
# from your_module import RMSNorm

class ConvTokenisation(nn.Module):
    """
    Convolutional Tokenisation with a pretrained CNN backbone.

    - Uses a pretrained CNN (default: ResNet-18) as a feature extractor.
    - Takes the last conv feature map (C, H, W), flattens spatial dims → (N, C),
      then projects to transformer dim `dim` and returns tokens of shape (B, N, dim).

    This is a drop-in replacement for ImagePatchEmbedding in a multimodal transformer.
    """

    def __init__(
        self,
        dim: int,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        img_channels: int = 3,
        token_grid_size: int | None = None,
        prepend_global_token: bool = False,
        global_token_only: bool = False,
        global_token_dim: int | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # ---- Build backbone ----
        # You can extend this if you want more options (resnet34, resnet50, etc.)
        if backbone_name == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
        elif backbone_name == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
        elif backbone_name == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        if img_channels != 3:
            # Replace first conv to handle different #channels but keep pretrained weights logic simple:
            # if img_channels != 3, we re-init conv1.
            conv1 = nn.Conv2d(
                img_channels,
                backbone.conv1.out_channels,
                kernel_size=backbone.conv1.kernel_size,
                stride=backbone.conv1.stride,
                padding=backbone.conv1.padding,
                bias=False,
            )
            backbone.conv1 = conv1

        # Remove avgpool and fc – keep everything up to the last conv feature map
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        # Optionally freeze pretrained backbone
        if freeze_backbone:
            for p in self.feature_extractor.parameters():
                p.requires_grad = False

        last_ch = backbone.fc.in_features  # C dimension of the final conv feature map

        if token_grid_size is not None and int(token_grid_size) <= 0:
            raise ValueError("token_grid_size must be positive when provided")
        self.token_grid_size = (
            None if token_grid_size is None else int(token_grid_size)
        )
        self.token_pool = (
            nn.Identity()
            if self.token_grid_size is None
            else nn.AdaptiveAvgPool2d(
                (self.token_grid_size, self.token_grid_size)
            )
        )
        self.prepend_global_token = bool(prepend_global_token)
        self.global_token_only = bool(global_token_only)
        if self.prepend_global_token and self.global_token_only:
            raise ValueError(
                "prepend_global_token and global_token_only are mutually exclusive"
            )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.global_token_dim = int(global_token_dim or dim)
        if self.global_token_dim <= 0:
            raise ValueError("global_token_dim must be positive")
        if not self.global_token_only and self.global_token_dim != dim:
            raise ValueError(
                "global_token_dim may differ from dim only in global_token_only mode"
            )
        self.global_proj = (
            nn.Identity()
            if last_ch == self.global_token_dim
            else nn.Linear(last_ch, self.global_token_dim)
        )

        # ---- Project conv channels -> transformer dim and convert to tokens ----
        self.token_proj = nn.Sequential(
            Rearrange("b c h w -> b (h w) c"),  # (B, C, H, W) -> (B, N, C)
            RMSNorm(last_ch),
            nn.Linear(last_ch, dim),
            RMSNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: (B, N, dim)
        """
        feat = self.feature_extractor(x)  # (B, C_out, H_out, W_out)
        if self.global_token_only:
            global_feature = self.global_pool(feat).flatten(1)
            return self.global_proj(global_feature).unsqueeze(1)
        global_token = None
        if self.prepend_global_token:
            global_feature = self.global_pool(feat).flatten(1)
            global_token = self.global_proj(global_feature).unsqueeze(1)
        # Keep the Transformer token count fixed across image resolutions. The
        # CNN can inspect a higher-resolution image without making downstream
        # self-attention grow quadratically with its native feature-map size.
        feat = self.token_pool(feat)
        tokens = self.token_proj(feat)  # (B, N, dim)
        if global_token is not None:
            tokens = torch.cat([global_token, tokens], dim=1)
        return tokens


def build_conv_token(args):
    return ConvTokenisation(
        dim=args.dim,
        backbone_name=getattr(args, "fundus_backbone_name", "resnet18"),
        pretrained=True,
        freeze_backbone=False,
        img_channels=3,
        token_grid_size=getattr(args, "fundus_token_grid_size", None),
        prepend_global_token=getattr(args, "fundus_prepend_global_token", False),
        global_token_only=getattr(args, "fundus_global_token_only", False),
        global_token_dim=getattr(args, "fundus_global_token_dim", None),
    )
