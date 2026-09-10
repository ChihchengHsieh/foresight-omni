# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from dataclasses import dataclass
from typing import Optional, Tuple
from torch import nn

import torch
import torch.nn.functional as F
from .ops.norm import RMSNorm
from .ops.attention import Attention, precompute_freqs_cis
from .ops.mlp import FeedForward


@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    return_intermediate: bool = True
    max_batch_size: int = 32
    max_seq_len: int = 4096
    attn_dropout_p: float = 0
    ff_dropout_p: float = 0


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(
            dim=args.dim,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            attn_dropout_p=args.attn_dropout_p,
        )

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
            ff_dropout_p=args.ff_dropout_p,
        )

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        pos: Optional[torch.Tensor],
        need_attn_weights: bool = False,
    ):
        h = self.attention(
            x=self.attention_norm(x),
            freqs_cis=freqs_cis,
            mask=mask,
            pos=pos,
            need_weights=need_attn_weights,
        )

        if need_attn_weights:
            h, attn_weights = h

        h = x + h

        out = h + self.feed_forward(self.ffn_norm(h))

        outputs = {"out": out}

        if need_attn_weights:
            outputs.update({"attn_weights": attn_weights})

        return outputs


class Transformer(nn.Module):

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.return_intermediate = args.return_intermediate
        self.n_layers = args.n_layers
        self.dim = args.dim

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.freqs_cis = precompute_freqs_cis(
            args.dim // args.n_heads,
            args.max_seq_len * 2,
            args.rope_theta,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        need_attn_weights: bool = False,
    ):
        L = x.shape[1]

        self.freqs_cis = self.freqs_cis.to(x.device)
        freqs_cis = self.freqs_cis[:L]

        hidden_state = x

        intermediate = [] if self.return_intermediate else None
        all_attn_weights = [] if need_attn_weights else None

        for layer in self.layers:
            out = layer(
                x=hidden_state,
                freqs_cis=freqs_cis,
                mask=mask,
                pos=pos,
                need_attn_weights=need_attn_weights,
            )

            hidden_state = out["out"]

            if need_attn_weights:
                all_attn_weights.append(out["attn_weights"])

            if self.return_intermediate:
                intermediate.append(hidden_state)

        if self.norm is not None:
            hidden_state = self.norm(hidden_state)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(hidden_state)

        outputs = {}

        if self.return_intermediate:
            outputs.update({"out": torch.stack(intermediate)})
        else:
            outputs.update({"out": hidden_state})

        if need_attn_weights:
            outputs.update({"attn_weights": all_attn_weights})

        return outputs


def build_transformer(args, return_intermediate: bool):
    model_args = ModelArgs(
        dim=args.dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_kv_heads=args.n_kv_heads,
        attn_dropout_p=args.attn_dropout_p,
        ff_dropout_p=args.ff_dropout_p,
        return_intermediate=return_intermediate,
    )

    return Transformer(model_args)
