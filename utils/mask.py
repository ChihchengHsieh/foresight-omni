import torch


def get_padding_mask(seq: torch.Tensor, pad_id=0):
    """
    seq: (B, L)
    """
    mask = seq == pad_id
    mask = (
        torch.zeros_like(mask, dtype=torch.float)
        .masked_fill_(mask, float("-inf"))
        .to(seq.device)
    )
    return mask


def get_causal_attn_mask(seq: torch.Tensor):
    """
    seq: (B, L)
    """
    return torch.nn.Transformer.generate_square_subsequent_mask(seq.shape[1])


def merge_mask(attn_mask, padding_mask):
    if (not padding_mask is None) and (not attn_mask is None):
        return padding_mask + attn_mask

    if not padding_mask is None:
        return padding_mask

    if not attn_mask is None:
        return attn_mask

    return None
