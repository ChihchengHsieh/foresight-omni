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

def get_pattern_key(sample):
    """
    Get the pattern key from the sample.
    """
    # input_features = "+".join(sorted(sample['input'].keys()))
    # label_features = "+".join(sorted(sample['label'].keys()))
    # pattern_key = f"input[{input_features}]_label[{label_features}]"
    pattern_key = json.dumps(sample, sort_keys=True)
    return pattern_key


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


def grab_default_colour(pattern_color_dict):
    """
    Generate a unique random RGB color not already used.

    Returns:
        A (R, G, B) tuple with values in [0.2, 0.85], rounded to 3 decimals.
    """
    used_colors = set(pattern_color_dict.values())
    max_attempts = 1000

    for _ in range(max_attempts):
        color = (
            round(random.uniform(0.2, 0.85), 3),
            round(random.uniform(0.2, 0.85), 3),
            round(random.uniform(0.2, 0.85), 3),
        )
        if color not in used_colors:
            return color

    raise ValueError("Failed to generate a unique color after 1000 attempts.")


from .universal_dense.transformer import Transformer


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
        pattern_color_dict: Dict[str, str] = None,
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

        self.passed_patterns = []
        self.pattern_counts = defaultdict(int)
        # self.pattern_counts = {}
        self.pattern_color_dict = pattern_color_dict

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
                    instance[k] = torch.concat([self.modality_container[k], v], dim=0)
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
            padded.append(i_padded)
            padding_mask.append(
                torch.tensor(
                    [0] * current_len + [-torch.inf] * padding_len,
                    device=device,
                )
            )

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

        input = self.to_seq(input)

        input = self.attach_container(input)

        input, input_pos, input_start_end = self.concat_modalities(input)

        input, label_pos, label_start_end = self.append_output_tokens(
            input, output_labels
        )

        # loop through the batch
        for i_p, l_p in zip(input_start_end, label_start_end):
            # check if the pattern already exists in the list
            pattern_to_add = {"input": i_p, "label": l_p}

            if pattern_to_add not in self.passed_patterns:
                self.passed_patterns.append(pattern_to_add)

            pattern_key = get_pattern_key(pattern_to_add)
            self.pattern_counts[pattern_key] += 1

        input, padding_mask = self.apply_padding(input)

        mask = (
            torch.nn.Transformer.generate_square_subsequent_mask(input.shape[1]).to(
                input.device
            )
            if self.causal
            else torch.zeros(input.shape[1], input.shape[1], device=input.device)
        )

        # combine mask
        # logging.info("merge mask")
        mask = merge_mask(mask, padding_mask)
        # mask += padding_mask

        max_len = input.shape[1]
        if self.pos_enc_strategy == "sin-input":
            pos = self.pad_pos(input_pos, max_len)
        elif "sin-input-label":
            pos = self.pad_pos(input_pos, max_len, label_pos)
        else:
            pos = None

        tf_out = self.transformer(
            input,
            mask,
            pos,
            need_attn_weights=need_attn_weights,
        )  # (n_layers, B, L, D)
        hs = tf_out["out"]

        if not self.transformer.return_intermediate:
            hs = hs.unsqueeze(0)

        # out = self.extract_label_tokens(hs, label_start_end)

        # Comment this to get all labels?
        out = [
            {l: self.output_layers[l](out[idx][l]) for l in ls}
            for idx, ls in enumerate(output_labels)
        ]

        outputs = {"out": out}

        if need_attn_weights:
            assert (
                "attn_weights" in tf_out
            ), "The transformer does not return attn_weights when need_attn_weights=True "
            outputs["attn_weights"] = tf_out["attn_weights"]
            outputs["input_start_end"] = (
                input_start_end  # list[OrderedDict[str, (start, end)]], per instance
            )
            outputs["label_start_end"] = (
                label_start_end  # list[OrderedDict[str, (start, end)]], per instance
            )
            outputs["seq_len"] = input.shape[1]  # a

        return outputs

    def plot_passed_patterns(self, save_path=None):
        """
        Visualize a list of feature dictionaries with separate input/label sections.

        Parameters:
            dict_list (list of dict): Each dict has "input" and "label" dicts with feature names to (start, end) tuples.
            color_dict (dict): Maps feature names to fill colors.
            default_color (str): Used if a feature name is missing from color_dict.
            count_dict (dict): Maps pattern keys (joined input feature names) to count.
        """
        if self.pattern_color_dict is None:
            self.pattern_color_dict = {}

        fig, ax = plt.subplots(figsize=(400, 1 * len(self.passed_patterns)))
        yticklabels = []

        for row_idx, sample in enumerate(self.passed_patterns):

            input_dict = sample.get("input", {})
            label_dict = sample.get("label", {})

            # Sort input by start index
            sorted_input = sorted(input_dict.items(), key=lambda x: x[1][0])
            sorted_label = sorted(label_dict.items(), key=lambda x: x[1][0])

            # Create a key and get count if available

            pattern_key = get_pattern_key(sample)
            count = self.pattern_counts.get(pattern_key, None)

            input_keys = ",".join(sorted(input_dict.keys()))
            label_keys = ",".join(sorted(label_dict.keys()))
            ytick_name = f"I[{input_keys}] L[{label_keys}]"

            yticklabel = f"N={count}" if count is not None else None
            yticklabels.append(yticklabel)
            # Plot input features
            for feature, (start, end) in sorted_input:

                color = self.pattern_color_dict.get(feature, None)
                if color is None:
                    color = grab_default_colour(self.pattern_color_dict)
                    self.pattern_color_dict[feature] = color
                ax.broken_barh(
                    [(start, end - start)],
                    (row_idx - 0.4, 0.8),
                    facecolors=color,
                    edgecolors="black",
                    linewidth=1.0,
                )
                ax.text(
                    (start + end) / 2,
                    row_idx,
                    feature,
                    ha="center",
                    va="center",
                    fontsize=6,
                    # rotation=45,
                )

            label_idx = 0
            # Plot label features (with different edge color)
            for feature, (start, end) in sorted_label:
                label_idx += 1
                # logging.info(f"Feature: {feature}, Start: {start}, End: {end}, Label Index: {label_idx}/{len(sorted_label)}")
                color = self.pattern_color_dict.get(feature, None)

                if color is None:
                    color = grab_default_colour(self.pattern_color_dict)
                    self.pattern_color_dict[feature] = color

                ax.broken_barh(
                    [(start, end - start)],
                    (row_idx - 0.4, 0.8),
                    facecolors=color,
                    edgecolors="darkred",
                    linewidth=1.0,
                )
                ax.text(
                    (start + end) / 2,
                    row_idx,
                    feature,
                    ha="center",
                    va="center",
                    fontsize=6,
                    # rotation=45,
                    color="darkred",
                )

        ax.set_yticks(range(len(self.passed_patterns)))
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel("Index Range")
        ax.set_title("Feature + Label Layout Across Samples")
        ax.set_ylim(-1, len(self.passed_patterns))
        ax.grid(True, axis="x", linestyle="--", alpha=0.3)

        # plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight")
            # logging.info(f"Passed Pattern Plot saved to {save_path}")

        plt.cla()
        plt.clf()
        plt.close()

    def save_passed_patterns(self, save_path):
        if save_path is None:
            return
        with open(save_path, "w") as f:
            json.dump(self.pattern_counts, f, indent=4)
        # logging.info(f"Passed Pattern Counts saved to {save_path}")


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
    model = UniversalModel(
        input_to_seq=input_to_seq,
        label_num_classes=label_num_classes,
        label_tokens_len=label_tokens_len,
        modality_container_type=args.container,
        pos_enc_strategy=args.pos,
        transformer=transformer,
        causal=args.causal,
    )
    return model
