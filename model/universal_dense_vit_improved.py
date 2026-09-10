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
from copy import copy


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


class MeanPoolFusionStub(nn.Module):
    """Parameter-free shape carrier for the Transformer-free fusion path."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)
        self.return_intermediate = False


class UniversalModel(nn.Module):
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
        pos_enc_strategy: str = "rotary",  # ['rotary', 'sin-input', 'sin-input-label']
        input_dim: int | None = None,
        input_dims: Dict[str, int] | None = None,
        embedding_dropout_p: float = 0.0,
        prediction_head_dropout_p: float = 0.0,
        auxiliary_readout_map: Dict[str, str] | str | None = None,
        prediction_conditioning_map: Dict[str, List[str]] | str | None = None,
        prediction_conditioning_detach: bool = False,
        target_modality_allow_map: Dict[str, List[str]] | str | None = None,
        fusion_mode: str = "transformer",
        fusion_readout_mode: str = "output_token",
        global_summary_modalities: List[str] | None = None,
        global_summary_residual_init: float = 0.0,
    ):
        super().__init__()
        if fusion_mode not in {"transformer", "mean_pool"}:
            raise ValueError(f"Unsupported fusion_mode: {fusion_mode!r}")
        self.fusion_mode = fusion_mode
        if fusion_readout_mode not in {
            "output_token",
            "global_summary",
            "residual_anchor",
        }:
            raise ValueError(
                f"Unsupported fusion_readout_mode: {fusion_readout_mode!r}"
            )
        if fusion_mode != "transformer" and fusion_readout_mode != "output_token":
            raise ValueError(
                "non-output-token fusion readout requires transformer fusion"
            )
        self.fusion_readout_mode = fusion_readout_mode
        self.global_summary_modalities = set(global_summary_modalities or [])
        if fusion_readout_mode == "global_summary":
            self.global_summary_gate = nn.Parameter(
                torch.tensor(float(global_summary_residual_init))
            )
        else:
            self.register_parameter("global_summary_gate", None)
        self.transformer = transformer
        self.dim = self.transformer.dim
        self.input_to_seq = input_to_seq
        input_dim = self.dim if input_dim is None else int(input_dim)
        input_dims = {
            modality: int((input_dims or {}).get(modality, input_dim))
            for modality in input_to_seq.keys()
        }
        self.input_to_fusion = nn.ModuleDict(
            {
                modality: (
                    nn.Identity()
                    if input_dims[modality] == self.dim
                    else nn.Linear(input_dims[modality], self.dim)
                )
                for modality in input_to_seq.keys()
            }
        )
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
        # These modules are parameter-free, so enabling them preserves strict
        # compatibility with checkpoints created before the dropout options
        # were wired into this model implementation.
        self.embedding_dropout = nn.Dropout(p=embedding_dropout_p)
        self.prediction_head_dropout = nn.Dropout(p=prediction_head_dropout_p)
        self.auxiliary_readout_map = self._normalise_auxiliary_readout_map(
            auxiliary_readout_map,
            label_num_classes,
            input_to_seq,
        )
        self.prediction_conditioning_map = self._normalise_prediction_conditioning_map(
            prediction_conditioning_map, label_num_classes
        )
        self.prediction_conditioning_detach = bool(prediction_conditioning_detach)
        self.modality_names = list(input_to_seq.keys())
        self.target_modality_allow_map = self._normalise_target_modality_allow_map(
            target_modality_allow_map,
            label_num_classes,
            input_to_seq,
        )
        self.protected_modalities = {
            modality
            for modalities in self.target_modality_allow_map.values()
            for modality in modalities
        }
        if self.target_modality_allow_map and fusion_mode != "transformer":
            raise ValueError("target_modality_allow_map requires transformer fusion")
        if self.target_modality_allow_map and fusion_readout_mode == "global_summary":
            raise ValueError(
                "target_modality_allow_map is incompatible with global_summary readout"
            )
        if self.fusion_readout_mode == "residual_anchor":
            self.anchor_modality_logits = nn.ParameterDict(
                {
                    label: nn.Parameter(torch.zeros(len(self.modality_names)))
                    for label in label_num_classes
                    if label not in self.auxiliary_readout_map
                }
            )
            self.residual_projections = nn.ModuleDict(
                {
                    label: nn.Linear(dim, dim, bias=False)
                    for label in label_num_classes
                    if label not in self.auxiliary_readout_map
                }
            )
            for projection in self.residual_projections.values():
                nn.init.zeros_(projection.weight)
        else:
            self.anchor_modality_logits = nn.ParameterDict()
            self.residual_projections = nn.ModuleDict()
        self.output_layers = nn.ModuleDict(
            {
                label: nn.Linear(
                    dim
                    + sum(
                        label_num_classes[source]
                        for source in self.prediction_conditioning_map.get(label, [])
                    ),
                    num_classes,
                )
                for label, num_classes in label_num_classes.items()
            }
        )

    def _label_can_access_modality(self, label: str, modality: str) -> bool:
        if modality not in self.protected_modalities:
            return True
        return modality in self.target_modality_allow_map.get(label, set())

    def _residual_anchor_base(self, instance, label):
        summaries = []
        score_indices = []
        for modality_index, modality in enumerate(self.modality_names):
            if not self._label_can_access_modality(label, modality):
                continue
            tokens = instance.get(modality)
            if tokens is None or not tokens.numel():
                continue
            summaries.append(tokens.mean(dim=0))
            score_indices.append(modality_index)
        if not summaries:
            if instance and all(
                modality in self.protected_modalities for modality in instance
            ):
                first_tokens = next(iter(instance.values()))
                return first_tokens.new_zeros(self.dim)
            raise ValueError("residual_anchor received an instance with no modalities")
        logits = self.anchor_modality_logits[label][score_indices]
        weights = torch.softmax(logits, dim=0)
        return torch.sum(torch.stack(summaries) * weights.unsqueeze(-1), dim=0)

    @staticmethod
    def _normalise_target_modality_allow_map(
        spec, label_num_classes, input_to_seq
    ):
        """Parse target-specific access to protected input modalities."""
        if not spec:
            return {}
        if isinstance(spec, str):
            parsed = {}
            for raw_item in spec.split(","):
                item = raw_item.strip()
                if not item:
                    continue
                if ":" not in item:
                    raise ValueError(
                        "Invalid target_modality_allow_map item "
                        f"{item!r}; expected 'target:modality+modality'."
                    )
                target, raw_modalities = item.split(":", 1)
                parsed[target.strip()] = [
                    modality.strip()
                    for modality in raw_modalities.split("+")
                    if modality.strip()
                ]
            spec = parsed
        normalised = {
            str(target): set(modalities)
            for target, modalities in dict(spec).items()
        }
        known_targets = set(label_num_classes)
        known_modalities = set(input_to_seq)
        unknown_targets = set(normalised) - known_targets
        if unknown_targets:
            raise ValueError(
                "Unknown target_modality_allow_map target(s): "
                f"{sorted(unknown_targets)}"
            )
        for target, modalities in normalised.items():
            if not modalities:
                raise ValueError(
                    f"Target {target!r} has no allowed protected modalities"
                )
            unknown_modalities = modalities - known_modalities
            if unknown_modalities:
                raise ValueError(
                    f"Unknown protected modalities for {target!r}: "
                    f"{sorted(unknown_modalities)}"
                )
        return normalised

    def _apply_target_modality_attention_policy(
        self,
        mask: torch.Tensor,
        input_start_end: List[Dict[str, Tuple[int, int]]],
        label_start_end: List[Dict[str, Tuple[int, int]]],
    ) -> torch.Tensor:
        """Isolate protected modalities from every target not explicitly allowed."""
        if not self.protected_modalities:
            return mask

        policy_mask = mask.clone()
        for batch_index, (input_spans, label_spans) in enumerate(
            zip(input_start_end, label_start_end)
        ):
            protected_key_indices = []
            input_indices = []
            label_indices = []
            for modality, (start, end) in input_spans.items():
                indices = list(range(start, end))
                input_indices.extend(indices)
                if modality in self.protected_modalities:
                    protected_key_indices.extend(indices)
            for start, end in label_spans.values():
                label_indices.extend(range(start, end))

            if input_indices and protected_key_indices:
                policy_mask[
                    batch_index,
                    0,
                    torch.tensor(input_indices, device=mask.device)[:, None],
                    torch.tensor(protected_key_indices, device=mask.device),
                ] = -torch.inf
            if input_indices and label_indices:
                policy_mask[
                    batch_index,
                    0,
                    torch.tensor(input_indices, device=mask.device)[:, None],
                    torch.tensor(label_indices, device=mask.device),
                ] = -torch.inf

            for label, (start, end) in label_spans.items():
                query_indices = torch.arange(start, end, device=mask.device)
                other_label_indices = [
                    index
                    for index in label_indices
                    if not (start <= index < end)
                ]
                if other_label_indices:
                    policy_mask[
                        batch_index,
                        0,
                        query_indices[:, None],
                        torch.tensor(other_label_indices, device=mask.device),
                    ] = -torch.inf

                for modality in self.protected_modalities:
                    span = input_spans.get(modality)
                    if span is None:
                        continue
                    modality_indices = torch.arange(
                        span[0], span[1], device=mask.device
                    )
                    value = (
                        0.0
                        if self._label_can_access_modality(label, modality)
                        else -torch.inf
                    )
                    policy_mask[
                        batch_index, 0, query_indices[:, None], modality_indices
                    ] = value
        return policy_mask

    @staticmethod
    def _normalise_auxiliary_readout_map(spec, label_num_classes, input_to_seq):
        if not spec:
            return {}
        if isinstance(spec, str):
            parsed = {}
            for raw_item in spec.split(","):
                item = raw_item.strip()
                if not item:
                    continue
                if ":" not in item:
                    raise ValueError(
                        "Invalid auxiliary_readout_map item "
                        f"{item!r}; expected 'target:modality'."
                    )
                target, modality = item.split(":", 1)
                parsed[target.strip()] = modality.strip()
            spec = parsed
        normalised = {str(k): str(v) for k, v in dict(spec).items()}
        unknown_targets = set(normalised) - set(label_num_classes)
        if unknown_targets:
            raise ValueError(
                "Unknown auxiliary readout target(s): "
                f"{sorted(unknown_targets)}"
            )
        unknown_modalities = set(normalised.values()) - set(input_to_seq)
        if unknown_modalities:
            raise ValueError(
                "Unknown auxiliary readout modality/modalities: "
                f"{sorted(unknown_modalities)}"
            )
        return normalised

    @staticmethod
    def _normalise_prediction_conditioning_map(spec, label_num_classes):
        if not spec:
            return {}
        if isinstance(spec, str):
            parsed = {}
            for raw_item in spec.split(","):
                item = raw_item.strip()
                if not item:
                    continue
                if ":" not in item:
                    raise ValueError(
                        "Invalid prediction_conditioning_map item "
                        f"{item!r}; expected 'target:source+source'."
                    )
                target, raw_sources = item.split(":", 1)
                parsed[target.strip()] = [
                    source.strip()
                    for source in raw_sources.split("+")
                    if source.strip()
                ]
            spec = parsed
        normalised = {str(k): list(v) for k, v in dict(spec).items()}
        known = set(label_num_classes)
        for target, sources in normalised.items():
            if target not in known:
                raise ValueError(f"Unknown conditioned target {target!r}")
            if not sources:
                raise ValueError(f"Conditioned target {target!r} has no sources")
            unknown = set(sources) - known
            if unknown:
                raise ValueError(
                    f"Unknown conditioning sources for {target!r}: {sorted(unknown)}"
                )
            if target in sources:
                raise ValueError(f"Conditioned target {target!r} cannot condition on itself")
            nested = set(sources) & set(normalised)
            if nested:
                raise ValueError(
                    "Nested prediction conditioning is not supported; "
                    f"{target!r} depends on conditioned target(s) {sorted(nested)}"
                )
        return normalised

    def _expand_conditioning_labels(self, labels: List[str]) -> List[str]:
        expanded = list(labels)
        for label in labels:
            for source in self.prediction_conditioning_map.get(label, []):
                if source not in expanded:
                    expanded.append(source)
        return expanded

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

            encoded = encoder(batched)  # (B_k, ..., input_dim)
            encoded = self.input_to_fusion[k](encoded)

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

    def encode_modalities(
        self, input: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:
        """Encode raw modalities without running the fusion transformer.

        Keeping this boundary explicit lets inference tools cache expensive image
        encoder outputs and perform controlled fusion-token perturbations without
        changing the ordinary training/inference path.
        """
        return self.to_seq(input)

    def forward_encoded_modalities(
        self,
        encoded_input: List[Dict[str, torch.Tensor]],
        output_labels: List[List[str]],
        need_attn_weights: bool = False,
    ):
        """Run fusion and prediction from already encoded modality tokens."""
        requested_output_labels = [list(labels) for labels in output_labels]
        expanded_output_labels = [
            self._expand_conditioning_labels(labels)
            for labels in requested_output_labels
        ]

        if self.fusion_mode == "mean_pool":
            dropped_input = [
                {
                    modality: self.embedding_dropout(tokens)
                    for modality, tokens in instance.items()
                }
                for instance in encoded_input
            ]
            return self._forward_mean_pooled(
                dropped_input,
                requested_output_labels,
                expanded_output_labels,
                need_attn_weights=need_attn_weights,
            )

        base_summaries = None
        if self.fusion_readout_mode == "global_summary":
            dropped_input = []
            base_summaries = []
            for instance in encoded_input:
                modality_summaries = []
                body_instance = {}
                for modality, tokens in instance.items():
                    if tokens.ndim != 2 or tokens.shape[0] < 1:
                        raise ValueError(
                            "global_summary requires non-empty 2D modality tokens; "
                            f"got {modality}={tuple(tokens.shape)}"
                        )
                    if modality in self.global_summary_modalities:
                        modality_summaries.append(tokens[0])
                        body_tokens = tokens[1:]
                    else:
                        modality_summaries.append(tokens.mean(dim=0))
                        body_tokens = tokens
                    body_instance[modality] = self.embedding_dropout(body_tokens)
                if not modality_summaries:
                    raise ValueError(
                        "global_summary received an instance with no modalities"
                    )
                base_summaries.append(torch.stack(modality_summaries).mean(dim=0))
                dropped_input.append(body_instance)
        else:
            # Regularise encoded modality tokens only. Learned containers and
            # output tokens are attached afterwards and remain intact.
            dropped_input = [
                {
                    modality: self.embedding_dropout(tokens)
                    for modality, tokens in instance.items()
                }
                for instance in encoded_input
            ]

        input = self.attach_container(dropped_input)
        input, input_pos, input_start_end = self.concat_modalities(input)
        if self.fusion_readout_mode == "global_summary":
            input = [
                torch.cat([summary.unsqueeze(0), seq], dim=0)
                for summary, seq in zip(base_summaries, input)
            ]
            input_pos = [
                torch.cat([pos.new_zeros((1, self.dim)), pos], dim=0)
                for pos in input_pos
            ]
            input_start_end = [
                OrderedDict(
                    [("global_summary", (0, 1))]
                    + [
                        (name, (start + 1, end + 1))
                        for name, (start, end) in spans.items()
                    ]
                )
                for spans in input_start_end
            ]
            label_pos = None
            label_start_end = [OrderedDict() for _ in input]
        else:
            fusion_output_labels = [
                [
                    label
                    for label in labels
                    if label not in self.auxiliary_readout_map
                ]
                for labels in expanded_output_labels
            ]
            input, label_pos, label_start_end = self.append_output_tokens(
                input, fusion_output_labels
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

        mask = merge_mask(base_attn, padding_mask)
        mask = self._apply_target_modality_attention_policy(
            mask,
            input_start_end,
            label_start_end,
        )

        max_len = input.shape[1]
        if self.pos_enc_strategy == "sin-input":
            pos = self.pad_pos(input_pos, max_len)
        elif self.pos_enc_strategy == "sin-input-label":
            pos = self.pad_pos(input_pos, max_len, label_pos)
        else:
            pos = None

        assert torch.isfinite(input).all(), "NaN leaked into transformer!"

        tf_out = self.transformer(
            input,
            mask,
            pos,
            need_attn_weights=need_attn_weights,
        )

        hs = tf_out["out"]
        if not self.transformer.return_intermediate:
            hs = hs.unsqueeze(0)

        last = hs[-1]
        batch_out = []
        for i, labels in enumerate(requested_output_labels):
            inst_logits = {}
            base_logits = {}
            if self.fusion_readout_mode == "global_summary":
                transformed_summary = last[i, 0, :]
                shared_representation = base_summaries[i] + self.global_summary_gate * (
                    transformed_summary - base_summaries[i]
                )
                shared_representation = self.prediction_head_dropout(
                    shared_representation
                )
                representations = {
                    label: shared_representation
                    for label in expanded_output_labels[i]
                }
            else:
                representations = {}
                for label in expanded_output_labels[i]:
                    auxiliary_modality = self.auxiliary_readout_map.get(label)
                    if auxiliary_modality is not None:
                        modality_tokens = encoded_input[i].get(auxiliary_modality)
                        if modality_tokens is None or not modality_tokens.numel():
                            continue
                        representations[label] = self.prediction_head_dropout(
                            modality_tokens.mean(dim=0)
                        )
                    else:
                        start, end = label_start_end[i][label]
                        transformed = last[i, start:end, :].mean(dim=0)
                        if self.fusion_readout_mode == "residual_anchor":
                            base = self._residual_anchor_base(
                                encoded_input[i], label
                            )
                            transformed = base + self.residual_projections[label](
                                transformed - base
                            )
                        representations[label] = self.prediction_head_dropout(
                            transformed
                        )
            for label in expanded_output_labels[i]:
                if (
                    label in representations
                    and label not in self.prediction_conditioning_map
                ):
                    base_logits[label] = self.output_layers[label](
                        representations[label]
                    )
            for label in labels:
                if label not in representations:
                    continue
                sources = self.prediction_conditioning_map.get(label, [])
                if sources:
                    if any(source not in base_logits for source in sources):
                        continue
                    source_logits = [base_logits[source] for source in sources]
                    if self.prediction_conditioning_detach:
                        source_logits = [value.detach() for value in source_logits]
                    conditioned_rep = torch.cat(
                        [representations[label], *source_logits], dim=-1
                    )
                    inst_logits[label] = self.output_layers[label](conditioned_rep)
                else:
                    inst_logits[label] = base_logits[label]
            batch_out.append(inst_logits)

        outputs = {"out": batch_out}
        if need_attn_weights:
            assert "attn_weights" in tf_out, (
                "The transformer does not return attn_weights when "
                "need_attn_weights=True"
            )
            outputs["attn_weights"] = tf_out["attn_weights"]
            outputs["input_start_end"] = input_start_end
            outputs["label_start_end"] = label_start_end
            outputs["seq_len"] = input.shape[1]
        return outputs

    def _forward_mean_pooled(
        self,
        encoded_input: List[Dict[str, torch.Tensor]],
        requested_output_labels: List[List[str]],
        expanded_output_labels: List[List[str]],
        need_attn_weights: bool = False,
    ):
        """Predict directly from mean-pooled encoder tokens.

        This is a true Transformer-free path for single-image CNN literature
        baselines.  It also defines a simple equal-token multimodal lower bound,
        although reproduction configs intentionally use fundus only.
        """

        if need_attn_weights:
            raise ValueError("Attention weights are unavailable in mean_pool fusion mode")
        batch_out = []
        for instance, labels, expanded_labels in zip(
            encoded_input, requested_output_labels, expanded_output_labels
        ):
            token_blocks = [tokens for tokens in instance.values() if tokens.numel()]
            if not token_blocks:
                raise ValueError("mean_pool fusion received an instance with no tokens")
            representation = torch.cat(token_blocks, dim=0).mean(dim=0)
            representation = self.prediction_head_dropout(representation)
            base_logits = {
                label: self.output_layers[label](representation)
                for label in expanded_labels
                if label not in self.prediction_conditioning_map
            }
            inst_logits = {}
            for label in labels:
                sources = self.prediction_conditioning_map.get(label, [])
                if sources:
                    source_logits = [base_logits[source] for source in sources]
                    if self.prediction_conditioning_detach:
                        source_logits = [value.detach() for value in source_logits]
                    conditioned_rep = torch.cat(
                        [representation, *source_logits], dim=-1
                    )
                    inst_logits[label] = self.output_layers[label](conditioned_rep)
                else:
                    inst_logits[label] = base_logits[label]
            batch_out.append(inst_logits)
        return {"out": batch_out}

    def forward(
        self,
        input: List[Dict[str, torch.Tensor]],
        output_labels: List[List[str]] = None,
        need_attn_weights: bool = False,
    ):
        encoded_input = self.encode_modalities(input)
        return self.forward_encoded_modalities(
            encoded_input,
            output_labels=output_labels,
            need_attn_weights=need_attn_weights,
        )

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


def build_universal_dense_vit(
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
    input_dim = int(args.dim)
    fusion_dim = int(getattr(args, "fusion_dim", None) or input_dim)
    if fusion_dim <= 0:
        raise ValueError(f"fusion_dim must be positive, got {fusion_dim}")
    if fusion_dim % int(args.n_heads) != 0:
        raise ValueError(
            f"fusion_dim={fusion_dim} must be divisible by n_heads={args.n_heads}"
        )
    fusion_mode = getattr(args, "fusion_mode", "transformer")
    transformer_args = copy(args)
    transformer_args.dim = fusion_dim
    transformer = (
        MeanPoolFusionStub(fusion_dim)
        if fusion_mode == "mean_pool"
        else build_transformer(transformer_args, return_intermediate=False)
    )
    model = UniversalModel(
        input_to_seq=input_to_seq,
        label_num_classes=label_num_classes,
        label_tokens_len=label_tokens_len,
        modality_container_type=args.container,
        pos_enc_strategy=args.pos,
        transformer=transformer,
        causal=args.causal,
        input_dim=input_dim,
        input_dims={
            modality: (
                int(getattr(args, "fundus_global_token_dim", None) or input_dim)
                if (
                    modality == "fundus_image"
                    and getattr(args, "fundus_global_token_only", False)
                )
                else input_dim
            )
            for modality in input_to_seq.keys()
        },
        embedding_dropout_p=getattr(args, "embedding_dropout_p", 0.0),
        prediction_head_dropout_p=getattr(args, "prediction_head_dropout_p", 0.0),
        auxiliary_readout_map=getattr(args, "auxiliary_readout_map", ""),
        prediction_conditioning_map=getattr(args, "prediction_conditioning_map", ""),
        prediction_conditioning_detach=getattr(
            args, "prediction_conditioning_detach", False
        ),
        target_modality_allow_map=getattr(
            args, "target_modality_allow_map", ""
        ),
        fusion_mode=getattr(args, "fusion_mode", "transformer"),
        fusion_readout_mode=getattr(args, "fusion_readout_mode", "output_token"),
        global_summary_modalities=(
            ["fundus_image"]
            if getattr(args, "fundus_prepend_global_token", False)
            else []
        ),
        global_summary_residual_init=getattr(
            args, "global_summary_residual_init", 0.0
        ),
    )
    return model
