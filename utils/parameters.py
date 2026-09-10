import pandas as pd
import torch.nn as nn
import logging


def initialize_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.Transformer):
        for p in module.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)




def get_module_parameters(model: nn.Module):
    param_dict = {}
    for n, p in model.named_parameters():
        param_dict.update({n: {"#params": p.nelement()}})
    return pd.DataFrame(param_dict).transpose().sort_values("#params", ascending=False)


def get_param_dict_with_backbone(
    model: nn.Module,
    lr: float,
    backbone_lr_factor: float = 1,
):
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": lr * backbone_lr_factor,
        },
    ]

    return param_dicts


from tabulate import tabulate


def model_size_in_gb(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    size_in_gb = param_size / (1024**3)  # Convert bytes to GB
    return size_in_gb


def print_parameters_count(model):
    n_trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    n_total_parameters = sum(p.numel() for p in model.parameters())

    print_str = f"| Number of trainable parameters: [{n_trainable_parameters}] | Total parameters: [{n_total_parameters}] |"
    print(
        tabulate(
            pd.DataFrame([{n: p.numel() for n, p in model.named_parameters()}])
            .transpose()
            .sort_values(0, ascending=False),
            headers="keys",
            tablefmt="psql",
        )
    )
    print(print_str)
    size_gb = model_size_in_gb(model)
    print(f"Model size: {size_gb:.2f} GB")

    name_trainable_parameters = [
        n for n, p in model.named_parameters() if p.requires_grad
    ]
    name_not_trainable_parameters = [
        n for n, p in model.named_parameters() if not p.requires_grad
    ]
    print("=========================Trainable=========================")
    print(name_trainable_parameters)
    print("=========================Not Trainable=========================")
    print(name_not_trainable_parameters)

    return n_trainable_parameters, n_total_parameters
