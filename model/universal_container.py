from collections import OrderedDict
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
from .pos_emb import SinCosPositionalEncodingProvider
import time
import logging


def merge_mask(attn_mask, padding_mask):
    """
    attn_mask: (L, L)
    padding_mask: (B, L)
    """
    B = padding_mask.shape[0]
    attn_mask = attn_mask.unsqueeze(0).repeat(B, 1, 1)  # (B, L, L)
    padding_mask = padding_mask.unsqueeze(1)  # (B, 1, L)
    return (attn_mask + padding_mask).unsqueeze(1)  # (B, 1, L, L)


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


class UniversalContainerModel(nn.Module):
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

    def concat_modalities(self, input):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        concat_input = []
        all_pos = []
        all_input_start_end = []

        for instance in input:
            instance_seq = torch.zeros(0, self.dim, device=device)
            instance_pos = torch.zeros(0, self.dim, device=device)
            instance_input_start_end = OrderedDict({})
            for i in self.input_to_seq.keys():
                if i in instance.keys():
                    start_idx = instance_seq.shape[0]
                    instance_seq = torch.concat([instance_seq, instance[i]], dim=0)
                    end_idx = instance_seq.shape[0]
                    i_pos = self.pos_enc(instance[i].unsqueeze(0)).squeeze(0)
                    instance_pos = torch.concat([instance_pos, i_pos], dim=0)
                    instance_input_start_end.update({i: (start_idx, end_idx)})
            all_input_start_end.append(instance_input_start_end)
            concat_input.append(instance_seq)
            all_pos.append(instance_pos)
        return concat_input, all_pos, all_input_start_end

    def attach_container(self, input: List[Dict[str, torch.Tensor]]):
        contained = []
        for instance in input:
            for k, v in instance.items():
                if self.modality_container_type == "bracket":

                    instance[k] = torch.concat(
                        [
                            self.modality_container[k][[0], :],
                            v,
                            self.modality_container[k][[1], :],
                        ],
                        dim=0,
                    )
                elif self.modality_container_type == "split":
                    instance[k] = torch.concat([self.modality_container[k], v], dim=1)
                else:
                    pass
            contained.append(instance)
        return contained

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
        elif self.modality_container_type == "split":
            return torch.concat(
                [
                    self.modality_container[k].unsqueeze(0).repeact(B, 1, 1),
                    input,
                ],
                dim=1,
            )
        else:
            return input

    def batch_to_seq_contained(
        self, input: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:

        # deal each modality one by one.
        for k in self.input_to_seq.keys():
            k_instances = []
            contain_k_batch_idx = []
            for idx, instance in enumerate(input):
                if k in instance:
                    k_instances.append(instance[k])
                    contain_k_batch_idx.append(idx)

            if len(k_instances) > 0:
                k_instances = self.input_to_seq[k](
                    pad_and_stack(k_instances)
                )  # (B_k, L, D)

                # add container or splitter
                k_instances = self.add_batch_container(k_instances, k)

                for i, (batch_idx) in enumerate(contain_k_batch_idx):
                    input[batch_idx][k] = k_instances[i]

        return input

    def to_seq(
        self, input: List[Dict[str, torch.Tensor]]
    ) -> List[Dict[str, torch.Tensor]]:

        return [
            {
                k: self.input_to_seq[k](v.float() if isinstance(v, torch.Tensor) else v)
                for k, v in instance.items()
            }
            for instance in input
        ]

    def append_output_tokens(
        self, input_seq: List[torch.Tensor], output_labels: List[List[str]]
    ):
        assert len(input_seq) == len(output_labels)

        device = input_seq[0].device

        all_pos = []
        all_label_start_end = []
        appended = []
        for i_s, i_l in zip(input_seq, output_labels):
            # sort i_l alphabetically
            i_l = sorted(i_l)
            instance_pos = torch.zeros(0, self.dim, device=device)
            instance_label_start_end = OrderedDict({})
            for l in i_l:
                # print("label: " + l)
                start_idx = i_s.shape[0]  # (L, D)
                # print("mean of i_s before concat: " + str(i_s.mean()))
                i_s = torch.concat([i_s, self.output_tokens[l]], dim=0)
                # print("mean of i_s after concat: " + str(i_s.mean()))
                end_idx = i_s.shape[0]  # (L, D)
                i_pos = self.pos_enc(self.output_tokens[l].unsqueeze(0)).squeeze(0)
                instance_pos = torch.concat([instance_pos, i_pos], dim=0)
                instance_label_start_end.update({l: (start_idx, end_idx)})
            # raise StopIteration()
            all_label_start_end.append(instance_label_start_end)
            all_pos.append(instance_pos)
            appended.append(i_s)
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
        max_len = max([len(s) for s in input_seq])
        padded = []
        padding_mask = []
        i = 0
        all_current_len = []
        for seq in input_seq:
            i += 1
            current_len = len(seq)
            all_current_len.append(current_len)
            padding_len = max_len - current_len
            i_padded = torch.concat(
                [seq, self.padding_token.repeat(padding_len, 1)], dim=0
            )
            # print(f"[{i}] Input Mean after padding {i}: {i_padded.mean()}")
            padded.append(i_padded)
            padding_mask.append(
                torch.tensor(
                    [0] * current_len + [-torch.inf] * padding_len,
                    device=device,
                )
            )

        # padded = torch.stack(padded, dim=0)
        # print("All Current Lengths: " + str(all_current_len))
        # print("First Element Mean: " + str(padded[0].mean()))
        # print("Padded Tensor Mean: " + str(padded.mean()))
        # raise StopIteration()
        return torch.stack(padded, dim=0), torch.stack(padding_mask, dim=0)

    def apply_padding_assign(self, input_seq):
        B = len(input_seq)
        device = input_seq[0].device
        max_len = max([len(s) for s in input_seq])
        padded = self.padding_token.unsqueeze(0).repeat(B, max_len, 1)
        padding_mask = -torch.ones(B, max_len, device=device) * torch.inf
        return padded, padding_mask

    def pad_pos(self, input_pos, max_len, label_pos=None):

        device = input_pos[0].device

        if label_pos:
            combined_pos = [
                torch.concat([i, l], dim=0) for i, l in zip(input_pos, label_pos)
            ]
        else:
            combined_pos = input_pos

        B = len(input_pos)

        pos = torch.zeros(B, max_len, self.dim, device=device)
        # assign
        for i, p in enumerate(combined_pos):
            pos[i, : len(p)] = p

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
        input = self.to_seq(input)  # List[Dict[str, torch.Tensor]]
        out = [{"has_glaucoma_in_13_years": i["genotype"]} for i in input]
        outputs = {"out": out}
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
    transformer = build_transformer(args, return_intermediate=False)
    model = UniversalContainerModel(
        input_to_seq=input_to_seq,
        label_num_classes=label_num_classes,
        label_tokens_len=label_tokens_len,
        modality_container_type=args.container,
        pos_enc_strategy=args.pos,
        transformer=transformer,
        causal=args.causal,
    )
    return model
