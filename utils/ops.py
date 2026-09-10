from typing import Callable
import torch.nn as nn


class FuncModule(nn.Module):
    def __init__(self, func: Callable, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.func = func

    def forward(self, x):
        return self.func(x)
