import torch
from collections import OrderedDict

def nested_to_device(x, device, non_blocking=False):
    # Tensor
    if isinstance(x, torch.Tensor):
        return x.to(device, non_blocking=non_blocking) if x.device != device else x

    # List
    elif isinstance(x, list):
        return [nested_to_device(v, device, non_blocking=non_blocking) for v in x]

    # Tuple
    elif isinstance(x, tuple):
        return tuple(nested_to_device(v, device, non_blocking=non_blocking) for v in x)

    # Dict
    elif isinstance(x, dict):
        return {
            k: nested_to_device(v, device, non_blocking=non_blocking)
            for k, v in x.items()
        }

    # OrderedDict
    elif isinstance(x, OrderedDict):
        return OrderedDict(
            (k, nested_to_device(v, device, non_blocking=non_blocking))
            for k, v in x.items()
        )

    # Everything else stays untouched
    else:
        return x
