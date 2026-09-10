from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from .pos_emb import SinCosPositionalEncodingProvider
import time
import logging
import matplotlib.pyplot as plt
from collections import defaultdict
import json, random


def merge_mask(attn_mask: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    """
    attn_mask: (L, L)      : causal or all-zeros (additive mask)
    padding_mask: (B, L)   : 0 for valid, -inf for padding
    returns: (B, 1, L, L)  : additive mask for attention
    """
    # (1, 1, L, L)
    attn = attn_mask.unsqueeze(0).unsqueeze(0)

    # (B, 1, 1, L)
    pad = padding_mask.view(padding_mask.shape[0], 1, 1, -1)

    # Broadcasting:
    # (1, 1, L, L) + (B, 1, 1, L) -> (B, 1, L, L)
    return attn + pad


def get_outputs(activations, outputs):
    """
    Used on the outputs of UniversalPadClassifier, where retuning the logists.
    """
    outputs = [
        {i_k: activations[i_k](outputs[i_k]) for i_k in i.keys()} for i in outputs
    ]
    return outputs


def pad_and_stack(tensor_list, padding_value=0):
    # Find the maximum length among all tensors
    max_length = max(tensor.size(0) for tensor in tensor_list)

    # Pad each tensor to the max_length
    padded_tensors = []
    for tensor in tensor_list:
        pad_size = max_length - tensor.size(0)
        padded_tensor = torch.nn.functional.pad(
            tensor, (0, pad_size), "constant", padding_value
        )
        padded_tensors.append(padded_tensor)

    # Stack the padded tensors
    stacked_tensors = torch.stack(padded_tensors)
    return stacked_tensors


from .universal_dense.transformer import Transformer

class UniversalOneHeadModel(nn.Module):
    """
    Allow different inputs and outputs in a batch.
    """

    def __init__(
        self,
        transformer: Transformer,
        input_to_seq: nn.ModuleDict,
        label_tokens_len: Dict[str, int],
        label_num_classes: Dict[str, int],
        modality_container_type: str = "bracket",  # ['bracket', 'splitter']
        causal=False,
        pos_enc_strategy: str = "sin-input",  # ['rotary', 'sin-input', 'sin-input-label']
        embedding_dropout_p: float = 0.0,
        prediction_head_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.transformer = transformer
        self.dim = self.transformer.dim
        self.input_to_seq = input_to_seq
        self.__init_label_tokens(label_tokens_len)
        self.modality_container_type = modality_container_type
        self.__init_modality_container(
            modality_container_type, list(input_to_seq.keys())
        )
        self.pos_enc_strategy = pos_enc_strategy
        dim = transformer.dim
        self.causal = causal
        self.pos_enc = SinCosPositionalEncodingProvider(d_model=transformer.dim)
        self.padding_token = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        # Drop encoded modality tokens before learned container/output tokens are
        # attached. This regularises the input representation without erasing the
        # learned task tokens themselves.
        self.embedding_dropout = nn.Dropout(p=embedding_dropout_p)
        self.prediction_head_dropout = nn.Dropout(p=prediction_head_dropout_p)
        self.output_layers = nn.ModuleDict(
            {k: nn.Linear(dim, v) for k, v in label_num_classes.items()}
        )

    def __init_label_tokens(self, output_tokens_len: Dict[str, int]):
        self.output_tokens = nn.ParameterDict()
        for k, v in output_tokens_len.items():
            self.output_tokens.update(
                {k: nn.Parameter(torch.randn(v, self.dim), requires_grad=True)}
            )

    def __init_modality_container(
        self, modality_container_type: str, input_modalities: List[str]
    ):
        self.modality_container = nn.ParameterDict()
        if modality_container_type == "bracket":
            container_size = 2
        elif modality_container_type == "splitter":
            container_size = 1
        else:
            return

        for k in input_modalities:
            self.modality_container.update(
                {
                    k: nn.Parameter(
                        torch.randn(container_size, self.dim), requires_grad=True
                    )
                }
            )

    def concat_modalities(self, batch):
        device = next(self.parameters()).device

        concat_input = []
        all_pos = []
        all_input_start_end = []

        for instance in batch:
            chunks = []
            pos_chunks = []
            inst_start_end = OrderedDict()
            cur_len = 0

            for mod in self.input_to_seq.keys():
                if mod in instance:
                    x = instance[mod]  # (L_m, D)
                    L = x.shape[0]

                    chunks.append(x)
                    i_pos = self.pos_enc(x.unsqueeze(0)).squeeze(0)
                    pos_chunks.append(i_pos)

                    inst_start_end[mod] = (cur_len, cur_len + L)
                    cur_len += L

            if chunks:
                instance_seq = torch.cat(chunks, dim=0)
                instance_pos = torch.cat(pos_chunks, dim=0)
            else:
                instance_seq = torch.empty(0, self.dim, device=device)
                instance_pos = torch.empty(0, self.dim, device=device)

            concat_input.append(instance_seq)
            all_pos.append(instance_pos)
            all_input_start_end.append(inst_start_end)

        return concat_input, all_pos, all_input_start_end

    def attach_container(self, batch: List[Dict[str, torch.Tensor]]):
        for instance in batch:
            for k, v in instance.items():
                if self.modality_container_type == "bracket":
                    instance[k] = torch.cat(
                        [
                            self.modality_container[k][[0]],
                            v,
                            self.modality_container[k][[1]],
                        ],
                        dim=0,
                    )
                elif self.modality_container_type == "split":
                    instance[k] = torch.cat(
                        [self.modality_container[k], v],
                        dim=0,
                    )
                else:
                    raise ValueError(
                        f"Unknown modality_container_type: {self.modality_container_type}"
                    )
        return batch

    def add_batch_container(self, input: torch.Tensor, k: str):
        B = input.shape[0]
        if self.modality_container_type == "bracket":
            return torch.concat(
                [
                    self.modality_container[k][[0], :]
                    .unsqueeze(0)
                    .repeat(B, 1, 1),  # (B, 1, D)
                    input,
                    self.modality_container[k][[1], :]
                    .unsqueeze(0)
                    .repeat(B, 1, 1),  # (B, 1, D),
                ],
                dim=1,
            )
        elif self.modality_container_type == "splitter":
            return torch.concat(
                [
                    self.modality_container[k].unsqueeze(0).repeat(B, 1, 1),
                    input,
                ],
                dim=1,
            )
        else:
            return input

    def to_seq(
        self, input: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Encode each modality in batch, instead of per-instance.
        input: list of dicts, each dict maps modality -> raw tensor/obj.
        returns: same structure, but with each modality passed through input_to_seq[k].
        """
        # Copy structure so we can write outputs in-place
        output = [dict(instance) for instance in input]

        # For each modality, gather all instances that have it
        for k, encoder in self.input_to_seq.items():
            raw_list = []
            idx_list = []

            for idx, instance in enumerate(input):
                if k in instance:
                    v = instance[k]
                    if isinstance(v, torch.Tensor):
                        v = v.float()
                    raw_list.append(v)
                    idx_list.append(idx)

            if not raw_list:
                continue  # no instance has this modality in this batch

            # Decide how to batch them – if your encoder expects (B, ...) you can stack
            # If you already have a pad_and_stack(), use that.
            if isinstance(raw_list[0], torch.Tensor):
                batched = torch.stack(raw_list, dim=0)
            else:
                # In case some encoders expect non-tensor types (e.g. strings),
                # just pass the list to the encoder directly.
                batched = raw_list

            encoded = encoder(batched)  # (B_k, ...)

            # Scatter back
            if isinstance(encoded, torch.Tensor):
                for j, idx in enumerate(idx_list):
                    output[idx][k] = encoded[j]
            else:
                for j, idx in enumerate(idx_list):
                    output[idx][k] = encoded[j]

        return output

    def append_output_tokens(
        self, input_seq: List[torch.Tensor], output_labels: List[List[str]]
    ):
        assert len(input_seq) == len(output_labels)

        device = input_seq[0].device
        all_pos = []
        all_label_start_end = []
        appended = []

        for seq, labels in zip(input_seq, output_labels):
            labels = sorted(labels)
            label_chunks = []
            pos_chunks = []
            label_start_end = OrderedDict()

            cur_len = seq.shape[0]

            for l in labels:
                tok = self.output_tokens[l]  # (L_l, D)
                L = tok.shape[0]

                label_chunks.append(tok)
                i_pos = self.pos_enc(tok.unsqueeze(0)).squeeze(0)
                pos_chunks.append(i_pos)

                label_start_end[l] = (cur_len, cur_len + L)
                cur_len += L

            if label_chunks:
                label_block = torch.cat(label_chunks, dim=0)
                pos_block = torch.cat(pos_chunks, dim=0)
                seq_with_labels = torch.cat([seq, label_block], dim=0)
            else:
                pos_block = torch.empty(0, self.dim, device=device)
                seq_with_labels = seq

            appended.append(seq_with_labels)
            all_pos.append(pos_block)
            all_label_start_end.append(label_start_end)

        return appended, all_pos, all_label_start_end

    def extract_label_tokens(
        self, hs, label_start_end: List[Dict[str, Tuple[int, int]]]
    ) -> List[Dict[str, torch.Tensor]]:
        output = []

        for i, i_se in enumerate(label_start_end):
            output.append({k: hs[:, i, v[0] : v[1], :] for k, v in i_se.items()})

        return output

    def apply_padding(self, input_seq):
        device = input_seq[0].device
        B = len(input_seq)
        max_len = max(len(s) for s in input_seq)

        padded = self.padding_token.unsqueeze(0).repeat(B, max_len, 1)
        padding_mask = -torch.inf * torch.ones(B, max_len, device=device)

        for i, seq in enumerate(input_seq):
            L = seq.shape[0]
            padded[i, :L] = seq
            padding_mask[i, :L] = 0.0

        return padded, padding_mask

    def pad_pos(self, input_pos, max_len, label_pos=None):
        """
        input_pos: list[Tensor], each (L_in_i, D)
        label_pos: optional list[Tensor], each (L_lab_i, D)
        returns: (B, max_len, D)
        """
        B = len(input_pos)

        # preserve dtype/device of existing pos embeddings
        pos = input_pos[0].new_zeros((B, max_len, self.dim))

        if label_pos is None:
            for i, p_in in enumerate(input_pos):
                L = p_in.shape[0]
                pos[i, :L] = p_in
        else:
            for i, (p_in, p_lab) in enumerate(zip(input_pos, label_pos)):
                # concat input + label positions along sequence dim
                p = torch.cat([p_in, p_lab], dim=0)
                L = p.shape[0]
                pos[i, :L] = p

        return pos

    def grad_cam_forward(
        self,
        input_fundus_image: torch.Tensor,
        output_label: str,
    ):
        output = self.forward(
            [{"fundus_image": input_fundus_image.squeeze(0)}],
            output_labels=[[output_label]],
        )

        return output["out"][0][output_label]

    def forward(
        self,
        input: List[Dict[str, torch.Tensor]],
        output_labels: List[List[str]] = None,
        need_attn_weights: bool = False,
    ):
        input = self.to_seq(input)
        input = [
            {k: self.embedding_dropout(v) for k, v in instance.items()}
            for instance in input
        ]
        input = self.attach_container(input)
        input, input_pos, input_start_end = self.concat_modalities(input)
        input, label_pos, label_start_end = self.append_output_tokens(
            input, output_labels
        )
        input, padding_mask = self.apply_padding(input)

        L = input.shape[1]
        device = input.device
        dtype = input.dtype

        if self.causal:
            base_attn = torch.nn.Transformer.generate_square_subsequent_mask(L).to(
                device=device, dtype=dtype
            )
        else:
            base_attn = torch.zeros(L, L, device=device, dtype=dtype)

        mask = merge_mask(base_attn, padding_mask)  # (B, 1, L, L)

        max_len = input.shape[1]
        if self.pos_enc_strategy == "sin-input":
            pos = self.pad_pos(input_pos, max_len)
        elif self.pos_enc_strategy == "sin-input-label":
            pos = self.pad_pos(input_pos, max_len, label_pos)
        else:
            pos = None

        tf_out = self.transformer(
            input,
            mask,
            pos,
            need_attn_weights=need_attn_weights,
        )  # tf_out["out"]: (n_layers, B, L, D) or (B, L, D)

        hs = tf_out["out"]

        # Ensure we always have (n_layers, B, L, D)
        if not self.transformer.return_intermediate:
            hs = hs.unsqueeze(0)

        # Use last layer only
        last = hs[-1]  # (B, L, D)

        batch_out = []
        for i, labels in enumerate(output_labels):
            inst_logits = {}
            for l in labels:
                start, end = label_start_end[i][l]  # (start, end) in sequence
                span = last[i, start:end, :]        # (span_len, D)
                # Pool over the label span : can change to last token if you prefer.
                rep = self.prediction_head_dropout(span.mean(dim=0))  # (D,)
                inst_logits[l] = self.output_layers[l](rep)           # (num_classes,)
            batch_out.append(inst_logits)

        outputs = {"out": batch_out}

        if need_attn_weights:
            assert (
                "attn_weights" in tf_out
            ), "The transformer does not return attn_weights when need_attn_weights=True"
            outputs["attn_weights"] = tf_out["attn_weights"]
            outputs["input_start_end"] = input_start_end
            outputs["label_start_end"] = label_start_end
            outputs["seq_len"] = input.shape[1]

        return outputs


class UniversalGradCAMWrapper(nn.Module):
    def __init__(self, model, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = model
        self.output_labels = None

    def set_output_label(self, label):
        self.output_label = label

    def forward(self, input_fundus_image: torch.Tensor):
        return self.model.grad_cam_forward(input_fundus_image, self.output_label)


from .universal_dense.transformer import build_transformer


def build_universal_dense_onehead_vit(
    args,
    input_to_seq,
    label_num_classes,
    label_tokens_len,

):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    transformer = build_transformer(args, return_intermediate=False)
    model = UniversalOneHeadModel(
        input_to_seq=input_to_seq,
        label_num_classes=label_num_classes,
        label_tokens_len=label_tokens_len,
        modality_container_type=args.container,
        pos_enc_strategy=args.pos,
        transformer=transformer,
        causal=args.causal,
        embedding_dropout_p=getattr(args, "embedding_dropout_p", 0.0),
        prediction_head_dropout_p=getattr(args, "prediction_head_dropout_p", 0.0),
    )
    return model
