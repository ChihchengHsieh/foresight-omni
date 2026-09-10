import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights

class ImageCNNEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # use resnet18 as the cnn backbone
        self.cnn_emb = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.cnn_emb.fc = nn.Linear(self.cnn_emb.fc.in_features, dim)

    def forward(self, x: torch.Tensor):
        return self.cnn_emb(x).unsqueeze(1)

def build_cnn_embedding(args):
    return ImageCNNEmbedding(args.dim)