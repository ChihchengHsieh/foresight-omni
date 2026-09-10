import torch, sys, math
import torch.nn as nn
from collections import defaultdict
from typing import Callable, Dict, Iterable, List, Set, Optional
from dataset.cols import *
from evaluators.classification import ClassificationEvaluator
from evaluators.mse import MSEEvaluator
from utils.tensor import nested_to_device
from utils.misc import (
    MetricLogger,
    SmoothedValue,
    all_gather,
    get_total_grad_norm,
)
from random import sample, randint
from model.loss import AUCMLoss, WeightedSumLosses, SigmoidWrapper
import logging
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor
from model.universal_dense_vit import UniversalGradCAMWrapper
from torch.nn.parallel import DistributedDataParallel as DDP
import cv2
import re
import logging
import random

import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR


PROFILE_TIMING_KEYS = (
    "getitem_total_s",
    "oct_total_s",
    "oct_zip_open_s",
    "oct_list_s",
    "oct_select_s",
    "oct_read_s",
    "oct_decode_resize_s",
    "oct_stack_s",
    "oct_slices",
    "oct_bytes",
    "smri_load_preprocess_s",
)


def _model_forward(model, samples, output_labels, device, use_bfloat16=False):
    enabled = bool(use_bfloat16 and device.type == "cuda")
    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
        enabled=enabled,
    ):
        output_dict = model(samples, output_labels=output_labels)

    # Metrics and probability-space losses remain float32. The cast preserves
    # gradients while keeping the expensive encoder/transformer forward in BF16.
    output_dict["out"] = [
        {
            key: value.float()
            if torch.is_tensor(value) and value.is_floating_point()
            else value
            for key, value in output.items()
        }
        for output in output_dict["out"]
    ]
    return output_dict


def _cuda_sync_for_timing(device, enabled: bool):
    if (
        enabled
        and torch.cuda.is_available()
        and isinstance(device, torch.device)
        and device.type == "cuda"
    ):
        torch.cuda.synchronize(device)


def _collect_batch_profile(data):
    batch_profile = {k: 0.0 for k in PROFILE_TIMING_KEYS}
    profiled_samples = 0
    for item in data:
        profile = item.get("__profile__") if isinstance(item, dict) else None
        if not profile:
            continue
        profiled_samples += 1
        for key in PROFILE_TIMING_KEYS:
            value = profile.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                batch_profile[key] += float(value)

    if profiled_samples == 0:
        return {}

    batch_profile["profiled_samples"] = float(profiled_samples)
    for key in PROFILE_TIMING_KEYS:
        if key.endswith("_s"):
            batch_profile[f"{key}_per_sample"] = batch_profile[key] / profiled_samples
    return batch_profile


def _add_profile_meters(metric_logger, batch_profile=None):
    meter_names = [
        "profile_sample_select_s",
        "profile_to_device_s",
        "profile_forward_s",
        "profile_loss_s",
        "profile_backward_step_s",
        "profile_evaluator_s",
    ]
    if batch_profile:
        for key in batch_profile:
            meter_names.append(key if key.startswith("profile_") else f"profile_{key}")

    for name in meter_names:
        if name not in metric_logger.meters:
            metric_logger.add_meter(name, SmoothedValue(fmt="{avg:.4f}"))


def _update_profile_meters(metric_logger, values):
    cleaned = {
        f"profile_{key}" if not key.startswith("profile_") else key: value
        for key, value in values.items()
        if isinstance(value, (int, float, np.integer, np.floating))
    }
    if cleaned:
        _add_profile_meters(metric_logger, cleaned)
        metric_logger.update(**cleaned)


def random_sample_input_and_labels(
    instance: Dict,
    possible_inputs: List[str],
    possible_labels: List[str],
    sample_prob: Dict[str, float] = None,
    default_prob: float = 1.0,
):
    """
    Randomly sample input and label modalities from the given instance.

    Args:
        instance: Dict containing all modalities and their data.
        possible_inputs: List of possible input modality names.
        possible_labels: List of possible label modality names.
        sample_prob: Optional dict mapping modality -> relative probability of being chosen.
                     Higher values mean higher chance to be sampled.
        default_prob: Default weight to use for modalities not listed in sample_prob.
    """
    instance_keys = set(instance.keys())
    input_intersection = set(possible_inputs).intersection(instance_keys)
    label_intersection = set(possible_labels).intersection(instance_keys)

    # No labels available -> nothing to do
    if not label_intersection:
        return {}, {}

    # Weights (no need to normalise for random.choices)
    if sample_prob is None:
        sample_prob = {}
    weights = {k: sample_prob.get(k, default_prob) for k in instance_keys}

    # --- Sample labels ---
    labels_sorted = list(label_intersection)
    label_weights = [weights[l] for l in labels_sorted]

    num_labels = random.randint(1, len(labels_sorted))
    sampled_labels = random.choices(labels_sorted, weights=label_weights, k=num_labels)
    sampled_labels = list(set(sampled_labels))  # unique

    # Make sure we still have at least one label
    if not sampled_labels:
        # Fallback: pick a single label, ignoring weights
        sampled_labels = [random.choice(labels_sorted)]

    # --- Sample inputs (disjoint from labels if possible) ---
    input_options = input_intersection.difference(set(sampled_labels))

    if input_options:
        inputs_sorted = list(input_options)
        input_weights = [weights[i] for i in inputs_sorted]
        num_inputs = random.randint(1, len(inputs_sorted))
        sampled_inputs = random.choices(
            inputs_sorted, weights=input_weights, k=num_inputs
        )
        sampled_inputs = list(set(sampled_inputs))
    else:
        # No disjoint input available
        # Option A: allow overlap and keep labels as-is (but that risks leakage)
        # Option B: follow your original logic but ensure at least one label survives
        if input_intersection:
            sampled_inputs = random.sample(list(input_intersection), 1)
            # only drop this label if more than one label is available
            if sampled_inputs[0] in sampled_labels and len(sampled_labels) > 1:
                sampled_labels.remove(sampled_inputs[0])
        else:
            sampled_inputs = []

    return (
        {i: instance[i] for i in set(sampled_inputs)},
        {l: instance[l] for l in set(sampled_labels)},
    )


def get_fixed_inputs_and_labels(
    instance: Dict,
    possible_inputs,
    possible_labels,
):
    # directly remove the labels from possible_inputs to prevent data leakage.
    keys = set(instance.keys())
    input_intersection = (
        set(possible_inputs).difference(set(possible_labels)).intersection(keys)
    )
    label_intersection = set(possible_labels).intersection(keys)

    return (
        {i: instance[i] for i in input_intersection},
        {l: instance[l] for l in label_intersection},
    )


def stack_apply_activation(activations, outputs_stacked):
    return {k: activations[k](outputs_stacked[k]) for k in outputs_stacked.keys()}


from typing import Collection, Dict, List, Tuple
from collections import defaultdict


# def same_key_stack(
#     keys: List[str],
#     outputs: List[Dict[str, torch.Tensor]],
#     targets: List[Dict[str, torch.Tensor]],
# ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

#     outputs_stacked: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
#     targets_stacked: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}

#     key_set = set(keys)

#     for o, t in zip(outputs, targets):
#         # Only consider tasks that exist in both t and keys
#         for k, t_val in t.items():
#             if k not in key_set:
#                 # ignore labels we don't have a criterion for
#                 continue
#             if k not in o:
#                 raise KeyError(f"Output missing key '{k}' present in targets.")
#             outputs_stacked[k].append(o[k])
#             targets_stacked[k].append(t_val)

#     outputs_, targets_ = {}, {}
#     for k in keys:
#         if outputs_stacked[k]:  # at least one sample for this task
#             outputs_[k] = torch.stack(outputs_stacked[k], dim=0)
#             targets_[k] = torch.stack(targets_stacked[k], dim=0)

#     return outputs_, targets_


def same_key_stack(
    keys: List[str],
    outputs: List[Dict[str, torch.Tensor]],
    targets: List[Dict[str, torch.Tensor]],
    allow_missing_output_keys: Collection[str] = (),
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:

    outputs_stacked: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}
    targets_stacked: Dict[str, List[torch.Tensor]] = {k: [] for k in keys}

    key_set = set(keys)
    allowed_missing_outputs = set(allow_missing_output_keys)

    def is_target_only_key(k: str) -> bool:
        # Survival-related target-only keys
        return k.endswith("_time_bin") or k.endswith("_event")

    def is_output_only_key(k: str) -> bool:
        # Survival-related output-only keys
        return k.endswith("_survival")

    for o, t in zip(outputs, targets):
        # For each sample, look at all keys we care about
        for k in keys:
            if k not in key_set:
                continue

            o_has = k in o
            t_has = k in t

            if o_has and t_has:
                # Normal case: both output and target exist
                outputs_stacked[k].append(o[k])
                targets_stacked[k].append(t[k])

            elif o_has and not t_has:
                # Output-only key (e.g. "glaucoma_survival")
                if not is_output_only_key(k):
                    # If it's not marked as output-only, this is suspicious
                    continue
                    # not appeding anything
                    # raise KeyError(f"Missing target for key '{k}' (has output only).")
                outputs_stacked[k].append(o[k])
                # no target stacked for this key

            elif (not o_has) and t_has:
                # Target-only key (e.g. "glaucoma_time_bin", "glaucoma_event")
                if k in allowed_missing_outputs:
                    # A source-specific auxiliary head is intentionally omitted
                    # when its source modality is unavailable or was removed by
                    # modality dropout. Its target must be omitted from this
                    # sample's loss as well.
                    continue
                if not is_target_only_key(k):
                    raise KeyError(f"Output missing key '{k}' present in targets.")
                targets_stacked[k].append(t[k])
                # no output stacked for this key

            else:
                # Neither output nor target present for this key in this sample -> skip
                continue

    outputs_, targets_ = {}, {}
    for k in keys:
        if outputs_stacked[k]:
            outputs_[k] = torch.stack(outputs_stacked[k], dim=0)
        if targets_stacked[k]:
            targets_[k] = torch.stack(targets_stacked[k], dim=0)

    return outputs_, targets_


def get_instance_loss(output, target, criterions):
    return {o_k: criterions[o_k](output[o_k], target[o_k]) for o_k in output.keys()}


def get_all_instances_losses(criterions, outputs, targets):
    losses = []
    for o, t in zip(outputs, targets):
        losses.append(get_instance_loss(o, t, criterions))
    return losses


def sum_instances_losses(losses, criterions):
    losses = {k: 0 for k in criterions.keys()}

    for i in losses:
        for c_k in i.keys():
            losses[c_k] += i[c_k]
    return {k: torch.mean(v) for k, v in losses.items()}


def get_losses(criterions, outputs, targets):
    losses = {k: 0 for k in criterions.keys()}

    for o, t in zip(outputs, targets):
        for o_k in o.keys():
            losses[o_k] += criterions[o_k](o[o_k], t[o_k])

    return {f"loss_{k}": torch.mean(v) for k, v in losses.items()}


def stack_get_losses(
    criterions,
    outputs_stacked,
    targets_stacked,
):
    """
    outputs_stacked: dict[label_name -> Tensor]
        e.g. "has_glaucoma_in_0_years" -> [B, 1] logits
             "glaucoma_survival"       -> [B, T] logits
    targets_stacked: dict[label_name -> Tensor]
        normal labels: same key as outputs
        survival: "{d}_time_bin", "{d}_event", OPTIONAL "{d}_obs_time_years"

    survival_label_num_bins: dict[str, int]
        keys like "glaucoma_survival", "ad_survival", ...
    """
    losses = {}

    for k, out in outputs_stacked.items():
        if k not in targets_stacked:
            continue

        tgt = targets_stacked[k].float()
        if tgt.dim() == 1:
            tgt = tgt.unsqueeze(-1)
        if out.dim() == 1:
            out = out.unsqueeze(-1)

        losses[k] = criterions[k](out, tgt)

    return losses


def instance_update_evaluators(evaluators, output, target):
    for k in output.keys():
        evaluators[k].update(output[k], target[k])


def all_instances_update_evaluators(evaluators: Dict, outputs, targets):
    for o, t in zip(outputs, targets):
        instance_update_evaluators(evaluators, o, t)


from evaluators.deephit import DeepHitCIndexEvaluator


def stack_update_evaluators(
    evaluators: Dict,
    outputs_stacked: Dict[str, torch.Tensor],
    targets_stacked: Dict[str, torch.Tensor],
):
    """
    Updates evaluators with stacked outputs/targets.

    For survival:
      - outputs_stacked["{d}_survival"] : [B, T]
      - targets_stacked["{d}_time_bin"] : [B] or [B,1]
      - targets_stacked["{d}_event"]    : [B] or [B,1]
      - OPTIONAL targets_stacked["{d}_obs_time_years"] : [B] or [B,1]
    """

    # 1) Normal evaluators: classification / regression
    for k, out_k in outputs_stacked.items():
        tgt_k = targets_stacked[k]
        evaluators[k].update(out_k, tgt_k)
    return evaluators


def save_preds_tgts(
    evaluators: Dict,
    diseases,
    saving_dir,
    sample_ids=None,
    direct_labels=None,
):
    frames = []
    row_frames = []
    for d in diseases:
        disease_df = evaluators[d].raw_dataframe()
        if (
            "patient_eid" not in disease_df.columns
            and sample_ids is not None
            and len(sample_ids) == len(disease_df)
        ):
            disease_df.insert(1, "sample_id", [item.get("idx") for item in sample_ids])
            disease_df.insert(2, "patient_eid", [item.get("patient_eid") for item in sample_ids])
            disease_df.insert(3, "dataset", [item.get("dataset") for item in sample_ids])
        disease_df.insert(1, "disease", d)
        frames.append(disease_df)

        if getattr(evaluators[d], "aggregate_by_group", False):
            row_df = evaluators[d].raw_row_dataframe()
            if sample_ids is not None and len(sample_ids) == len(row_df):
                if "patient_eid" not in row_df.columns:
                    row_df.insert(2, "patient_eid", [item.get("patient_eid") for item in sample_ids])
                row_df.insert(1, "sample_id", [item.get("idx") for item in sample_ids])
                row_df.insert(3, "dataset", [item.get("dataset") for item in sample_ids])
            row_df.insert(1, "disease", d)
            row_frames.append(row_df)

    for label in direct_labels or []:
        evaluator = evaluators.get(label)
        if evaluator is None or not hasattr(evaluator, "raw_dataframe"):
            continue
        label_df = evaluator.raw_dataframe()
        label_df.insert(1, "task", label)
        if sample_ids is not None and len(sample_ids) == len(label_df):
            label_df.insert(2, "sample_id", [item.get("idx") for item in sample_ids])
            label_df.insert(
                3,
                "patient_eid",
                [item.get("patient_eid") for item in sample_ids],
            )
            label_df.insert(4, "dataset", [item.get("dataset") for item in sample_ids])
        frames.append(label_df)

    output_path = os.path.abspath(saving_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if not frames:
        raise ValueError("No evaluator predictions were available to save")
    pd.concat(frames, ignore_index=True).to_csv(output_path, index=False)
    if row_frames:
        root, extension = os.path.splitext(output_path)
        pd.concat(row_frames, ignore_index=True).to_csv(
            f"{root}_rows{extension or '.csv'}",
            index=False,
        )

    return evaluators


def configure_temporal_auroc_ci(
    evaluators: Dict,
    diseases,
    *,
    n_bootstrap: int = 0,
    seed: int = 42,
    low_positive_threshold: int = 50,
):
    for d in diseases:
        evaluator = evaluators.get(d)
        if evaluator is not None and hasattr(evaluator, "set_auroc_ci"):
            evaluator.set_auroc_ci(
                n_bootstrap=n_bootstrap,
                seed=seed,
                low_positive_threshold=low_positive_threshold,
            )
    return evaluators


def get_scaled_unscaled_loss_dict(loss_dict, loss_weight_dict):
    reduced_list_dict = all_gather(loss_dict)
    loss_dict_reduced = defaultdict(
        list
    )  # {f"loss_{k}": [] for k in list(criterions.keys())}
    for ld in reduced_list_dict:
        for k, v in ld.items():
            loss_dict_reduced[k].append(v)

    loss_dict_reduced = {
        k: torch.tensor(v).mean() for k, v in loss_dict_reduced.items()
    }

    loss_dict_reduced_unscaled = {
        f"{k}_loss_unscaled": v for k, v in loss_dict_reduced.items()
    }
    loss_dict_reduced_scaled = {
        f"{k}_loss_scaled": v * loss_weight_dict[k]
        for k, v in loss_dict_reduced.items()
    }

    return loss_dict_reduced_scaled, loss_dict_reduced_unscaled


def append_nan_to_lost_dict(all_keys, loss_dict, to_key: Callable, device):
    out_dict = {}
    for k in all_keys:
        append_k = to_key(k)
        if append_k in loss_dict.keys():
            out_dict[append_k] = loss_dict[append_k]
        else:
            out_dict[append_k] = torch.tensor(torch.nan, device=device)
    return out_dict


def random_only_fundus_image_sample(data, possible_inputs, possible_labels, fundus_only_prob=0.5):
    if random.random() < fundus_only_prob:
        samples, targets = zip(
            *(
                get_fixed_inputs_and_labels(
                    d,
                    possible_inputs=["fundus_image"],
                    possible_labels=possible_labels,
                )
                for d in data
            )
        )
    else:
        samples, targets = zip(
            *(
                random_sample_input_and_labels(
                    d,
                    possible_inputs=possible_inputs,
                    possible_labels=possible_labels,
                )
                for d in data
            )
        )
    return samples, targets


def train_sample_inputs_labels(
    data,
    random_sample,
    possible_inputs,
    possible_labels,
    sample_prob=None,
    only_fundus_image_sample= False,
    only_fundus_image_prob=0.5,
    modality_dropout_p=0.0,
):
    if random_sample:
        if only_fundus_image_sample:
            samples, targets = random_only_fundus_image_sample(
                data,
                possible_inputs,
                possible_labels,
                fundus_only_prob=only_fundus_image_prob,
            )
        else:
            samples, targets = zip(
                *(
                    random_sample_input_and_labels(
                        d,
                        possible_inputs=possible_inputs,
                        possible_labels=possible_labels,
                        sample_prob=sample_prob,
                    )
                    for d in data
                )
            )
    else:
        # remove the labels from input to prevent data leakage.
        samples, targets = zip(
            *(
                get_fixed_inputs_and_labels(
                    d,
                    possible_inputs=possible_inputs,
                    possible_labels=possible_labels,
                )
                for d in data
            )
        )
        if modality_dropout_p > 0:
            dropped_samples = []
            for sample in samples:
                available = list(sample.keys())
                kept = {
                    key: value
                    for key, value in sample.items()
                    if random.random() >= modality_dropout_p
                }
                if not kept and available:
                    key = random.choice(available)
                    kept[key] = sample[key]
                dropped_samples.append(kept)
            samples = tuple(dropped_samples)
    return samples, targets


from evaluators.deephit import DeepHitCIndexEvaluator


def get_log_values(all_keys, evaluators, metric_logger, diseases, evaluator_workers=1):
    # Start with any values already tracked by metric_logger (e.g. losses)
    log_values = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    computed = {}
    compute_keys = [key for key in all_keys if key in evaluators]
    if int(evaluator_workers) > 1 and len(compute_keys) > 1:
        with ThreadPoolExecutor(
            max_workers=min(int(evaluator_workers), len(compute_keys))
        ) as executor:
            computed = dict(
                zip(
                    compute_keys,
                    executor.map(lambda key: evaluators[key].compute(), compute_keys),
                )
            )

    for k in all_keys:
        if k not in evaluators:
            continue

        ev = evaluators[k]
        res = computed.get(k)
        if k not in computed:
            res = ev.compute()

        if res is None:
            continue
        if not isinstance(res, dict):
            # Just in case some evaluator returns a single tensor/scalar
            if hasattr(res, "detach"):
                val = res.detach().cpu().item()
            else:
                val = float(res)
            log_values[f"{k}"] = val
            continue

        # Normal path: dict of metric_name -> value
        for m_name, m_val in res.items():
            if hasattr(m_val, "detach"):
                # Tensor-like
                try:
                    m_val = m_val.detach().cpu().item()
                except Exception:
                    # In case it's not a scalar tensor, skip or handle differently
                    continue
            else:
                m_val = float(m_val)

            log_values[f"{k}_{m_name}"] = m_val

    return log_values


def group_evaluators_by_disease_and_year(
    evaluators,
):
    disease_to_evaluators = defaultdict(dict)

    # Regex pattern: matches strings like "has_glaucoma_in_2_years"
    pattern = re.compile(r"^has_([a-z]+)_in_(\d+)_years$")

    for label_name, evaluator in evaluators.items():
        match = pattern.match(label_name)
        if match:
            disease = match.group(1)
            year = int(match.group(2))
            disease_to_evaluators[disease][year] = evaluator

    return disease_to_evaluators


def expand_survival_labels(label_list):
    new_list = []
    for label in label_list:
        if label.endswith("_survival"):
            d = label.replace("_survival", "")
            new_list.append(f"{d}_time_bin")
            new_list.append(f"{d}_event")
        else:
            new_list.append(label)
    return new_list


def seperate_onehead_outputs(
    out: List[Dict[str, torch.Tensor]],
    dieseases,
    progression_label_years,
    direct_labels=None,
):
    new_out = []
    for o in out:
        o_new = {}
        for i, y in enumerate(progression_label_years):
            for d in dieseases:
                key = f"has_{d}_in_{y}_years"
                o_new[key] = o[d][i]
        for label in direct_labels or []:
            if label in o:
                o_new[label] = o[label]
        new_out.append(o_new)
    return new_out


def train_one_epoch(
    model: nn.Module,
    possible_inputs: List[str],
    possible_labels: List[str],
    criterions: Dict[str, nn.Module],
    optimizer: torch.optim.Optimizer,
    activations: Dict[str, nn.Module],
    evaluators: Dict[str, nn.ModuleDict],
    dataloader: Iterable,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.0,
    print_freq: int = 10,
    loss_weight_dict: Dict[str, float] = None,
    random_sample: bool = True,
    diseases: list = None,
    direct_labels: list = None,
    sample_prob: Dict[str, float] = None,
    lr_scheduler=None,
    progression_label_years: list[int] = None,
    only_fundus_image_sample: bool = False,
    only_fundus_image_prob: float = 0.5,
    modality_dropout_p: float = 0.0,
    profile_timing: bool = False,
    profile_timing_sync_cuda: bool = False,
    use_bfloat16: bool = False,
    gradient_accumulation_steps: int = 1,
):
    model.train()
    for v in criterions.values():
        v.train()

    if loss_weight_dict is None:
        # Either assert or default to 1.0 weights
        loss_weight_dict = {k: 1.0 for k in criterions.keys()}

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "grad_norm", SmoothedValue(window_size=1, fmt="{value:.2f}")
    )
    if profile_timing:
        _add_profile_meters(metric_logger)

    all_loss_keys = list(criterions.keys())
    for k in all_loss_keys:
        metric_logger.add_meter(f"{k}_loss_scaled", SmoothedValue())
        metric_logger.add_meter(f"{k}_loss_unscaled", SmoothedValue())

    header = f"Epoch: [{epoch}]"

    all_label_cols = expand_survival_labels(possible_labels)

    gradient_accumulation_steps = int(gradient_accumulation_steps)
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    optimizer.zero_grad(set_to_none=True)

    for micro_step, data in enumerate(
        metric_logger.log_every(dataloader, print_freq, header)
    ):
        profile_values = {}
        if profile_timing:
            profile_values.update(_collect_batch_profile(data))

        section_start = time.perf_counter()
        samples, targets = train_sample_inputs_labels(
            data,
            random_sample,
            possible_inputs,
            all_label_cols,
            sample_prob=sample_prob,
            only_fundus_image_sample=only_fundus_image_sample,
            only_fundus_image_prob=only_fundus_image_prob,
            modality_dropout_p=modality_dropout_p,
        )
        if profile_timing:
            profile_values["sample_select_s"] = time.perf_counter() - section_start

        section_start = time.perf_counter()
        targets = nested_to_device(targets, device, non_blocking=True)
        samples = nested_to_device(samples, device, non_blocking=True)
        _cuda_sync_for_timing(device, profile_timing_sync_cuda)
        if profile_timing:
            profile_values["to_device_s"] = time.perf_counter() - section_start

        # Make all of them the same to avoid leakage
        requested_outputs = list(diseases or []) + list(direct_labels or [])
        output_labels = [requested_outputs for _ in targets]
        # Alternative (use actual target keys):
        # output_labels = [list(t.keys()) for t in targets]

        section_start = time.perf_counter()
        output_dict = _model_forward(
            model,
            samples,
            output_labels,
            device,
            use_bfloat16=use_bfloat16,
        )
        _cuda_sync_for_timing(device, profile_timing_sync_cuda)
        if profile_timing:
            profile_values["forward_s"] = time.perf_counter() - section_start

        section_start = time.perf_counter()
        out = output_dict["out"]

        # have to seperate the onehead here.

        out = seperate_onehead_outputs(
            out, diseases, progression_label_years, direct_labels=direct_labels
        )
        keys_for_stacking = list(criterions.keys())
        out_stacked, targets_stacked = same_key_stack(
            keys_for_stacking,
            out,
            targets,
            allow_missing_output_keys=direct_labels or (),
        )

        out_stacked = stack_apply_activation(activations, out_stacked)
        loss_dict = stack_get_losses(criterions, out_stacked, targets_stacked)

        if len(loss_dict) == 0:
            logging.warning(
                "No per-task losses computed for this batch. "
                f"criterion keys={list(criterions.keys())}, "
                f"out keys={list(out_stacked.keys())}, "
                f"target keys={list(targets_stacked.keys())}"
            )
            continue

        # Weighted sum of task losses
        losses = sum(loss_dict[k] * loss_weight_dict[k] for k in loss_dict.keys())

        loss_dict_reduced_scaled, loss_dict_reduced_unscaled = (
            get_scaled_unscaled_loss_dict(
                loss_dict,
                loss_weight_dict,
            )
        )

        loss_value = sum(loss_dict_reduced_scaled.values())
        if isinstance(loss_value, torch.Tensor):
            loss_value = loss_value.item()

        if not math.isfinite(loss_value):
            msg = f"Loss became non-finite ({loss_value}). Stopping training to avoid NaNs."
            logging.error(msg)
            raise RuntimeError(msg)

        params = [p for p in model.parameters() if p.requires_grad]

        if not torch.is_tensor(losses):
            msg = f"Expected tensor loss, got {type(losses)} with value {losses}"
            logging.error(msg)
            raise RuntimeError(msg)
        _cuda_sync_for_timing(device, profile_timing_sync_cuda)
        if profile_timing:
            profile_values["loss_s"] = time.perf_counter() - section_start

        section_start = time.perf_counter()
        group_start = (
            micro_step // gradient_accumulation_steps
        ) * gradient_accumulation_steps
        group_size = min(
            gradient_accumulation_steps,
            len(dataloader) - group_start,
        )
        (losses / group_size).backward()

        should_step = (
            (micro_step + 1) % gradient_accumulation_steps == 0
            or micro_step + 1 == len(dataloader)
        )
        grad_total_norm = None
        if should_step:
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(params, max_norm)
            else:
                grad_total_norm = get_total_grad_norm(params)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if lr_scheduler is not None:
                lr_scheduler.step()
        _cuda_sync_for_timing(device, profile_timing_sync_cuda)
        if profile_timing:
            profile_values["backward_step_s"] = time.perf_counter() - section_start

        section_start = time.perf_counter()
        # stack_update_evaluators(
        #     evaluators=evaluators,
        #     outputs_stacked=out_stacked,
        #     targets_stacked=targets_stacked,
        # )
        for d in diseases:
            evaluators[d].update(
                outputs=out,
                targets=targets,
                group_ids=[item.get("patient_eid") for item in data],
            )
        for label in direct_labels or []:
            if label in evaluators and label in out_stacked and label in targets_stacked:
                evaluators[label].update(out_stacked[label], targets_stacked[label])
        if profile_timing:
            profile_values["evaluator_s"] = time.perf_counter() - section_start

        # Logging
        all_loss_keys = list(criterions.keys())

        scaled_to_append = append_nan_to_lost_dict(
            all_loss_keys,
            loss_dict_reduced_scaled,
            lambda x: f"{x}_loss_scaled",
            device,
        )

        unscaled_to_append = append_nan_to_lost_dict(
            all_loss_keys,
            loss_dict_reduced_unscaled,
            lambda x: f"{x}_loss_unscaled",
            device,
        )

        metric_logger.update(
            loss=loss_value,
            **scaled_to_append,
            **unscaled_to_append,
        )
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if grad_total_norm is not None:
            metric_logger.update(grad_norm=grad_total_norm)
        if profile_timing:
            _update_profile_meters(metric_logger, profile_values)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    evaluation_keys = list(diseases or []) + list(direct_labels or [])
    return get_log_values(evaluation_keys, evaluators, metric_logger, diseases)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    eval_inputs: List[str],
    eval_labels: List[str],
    criterions: Dict[str, nn.Module],
    activations: Dict[str, nn.Module],
    evaluators: Dict[str, nn.ModuleDict],
    dataloader: Iterable,
    device: torch.device,
    print_freq: int = 10,
    loss_weight_dict: Dict[str, float] = None,
    header="Test:",
    epoch=None,
    diseases: list = [],
    direct_labels: list = None,
    save_pred_path=None,
    print_performance_on_fly=False,
    progression_label_years: list[int] = None,
    profile_timing: bool = False,
    profile_timing_sync_cuda: bool = False,
    auroc_ci_bootstrap: int = 0,
    auroc_ci_seed: int = 42,
    low_positive_threshold: int = 50,
    fixed_thresholds: dict | None = None,
    auroc_ci_workers: int = 1,
    use_bfloat16: bool = False,
):
    model.to(device)
    model.eval()
    for v in criterions.values():
        v.eval()
    if fixed_thresholds:
        for disease in diseases:
            evaluator = evaluators.get(disease)
            if evaluator is not None and hasattr(evaluator, "set_fixed_thresholds"):
                evaluator.set_fixed_thresholds(fixed_thresholds.get(disease, {}))
        for label in direct_labels or []:
            evaluator = evaluators.get(label)
            if evaluator is not None and hasattr(evaluator, "set_fixed_threshold"):
                evaluator.set_fixed_threshold(fixed_thresholds.get(label))

    if not epoch is None:
        header = f"{header} Epoch: [{epoch}]"

    metric_logger = MetricLogger(delimiter="  ")
    if profile_timing:
        _add_profile_meters(metric_logger)

    all_loss_keys = list(criterions.keys())

    for k in all_loss_keys:
        metric_logger.add_meter(f"{k}_loss_scaled", SmoothedValue())
        metric_logger.add_meter(f"{k}_loss_unscaled", SmoothedValue())

    all_label_cols = expand_survival_labels(eval_labels)
    evaluated_sample_ids = []

    for data in metric_logger.log_every(dataloader, print_freq, header):
        evaluated_sample_ids.extend(
            {
                "idx": item.get("idx"),
                "patient_eid": item.get("patient_eid"),
                "dataset": item.get("dataset"),
            }
            for item in data
        )
        profile_values = {}
        if profile_timing:
            profile_values.update(_collect_batch_profile(data))

        section_start = time.perf_counter()
        samples, targets = zip(
            *(
                get_fixed_inputs_and_labels(
                    d,
                    possible_inputs=eval_inputs,
                    possible_labels=all_label_cols,
                )
                for d in data
            )
        )
        if profile_timing:
            profile_values["sample_select_s"] = time.perf_counter() - section_start

        section_start = time.perf_counter()
        targets = nested_to_device(targets, device, non_blocking=True)
        samples = nested_to_device(samples, device, non_blocking=True)
        _cuda_sync_for_timing(device, profile_timing_sync_cuda)
        if profile_timing:
            profile_values["to_device_s"] = time.perf_counter() - section_start

        # output_labels = [list(t.keys()) for t in targets]
        requested_outputs = list(diseases or []) + list(direct_labels or [])
        output_labels = [requested_outputs for _ in targets]
        # output_labels = [eval_labels for _ in targets]

        section_start = time.perf_counter()
        output_dict = _model_forward(
            model,
            samples,
            output_labels,
            device,
            use_bfloat16=use_bfloat16,
        )
        _cuda_sync_for_timing(device, profile_timing_sync_cuda)
        if profile_timing:
            profile_values["forward_s"] = time.perf_counter() - section_start

        section_start = time.perf_counter()
        out = output_dict["out"]
        out = seperate_onehead_outputs(
            out, diseases, progression_label_years, direct_labels=direct_labels
        )
        keys_for_stacking = list(criterions.keys())

        out_stacked, targets_stacked = same_key_stack(
            keys_for_stacking,
            out,
            targets,
            allow_missing_output_keys=direct_labels or (),
        )

        out_stacked = stack_apply_activation(activations, out_stacked)
        loss_dict = stack_get_losses(criterions, out_stacked, targets_stacked)

        loss_dict_reduced_scaled, loss_dict_reduced_unscaled = (
            get_scaled_unscaled_loss_dict(
                loss_dict,
                loss_weight_dict,
            )
        )
        _cuda_sync_for_timing(device, profile_timing_sync_cuda)
        if profile_timing:
            profile_values["loss_s"] = time.perf_counter() - section_start

        section_start = time.perf_counter()
        # stack_update_evaluators(
        #     evaluators=evaluators,
        #     outputs_stacked=out_stacked,
        #     targets_stacked=targets_stacked,
        # )
        for d in diseases:
            evaluators[d].update(
                outputs=out,
                targets=targets,
                group_ids=[item.get("patient_eid") for item in data],
            )
        for label in direct_labels or []:
            if label in evaluators and label in out_stacked and label in targets_stacked:
                evaluators[label].update(out_stacked[label], targets_stacked[label])
        if profile_timing:
            profile_values["evaluator_s"] = time.perf_counter() - section_start

        scaled_to_append = append_nan_to_lost_dict(
            all_loss_keys,
            loss_dict_reduced_scaled,
            lambda x: f"{x}_loss_scaled",
            device,
        )

        unscaled_to_append = append_nan_to_lost_dict(
            all_loss_keys,
            loss_dict_reduced_unscaled,
            lambda x: f"{x}_loss_unscaled",
            device,
        )

        metric_logger.update(
            loss=sum(loss_dict_reduced_scaled.values()),
            **scaled_to_append,
            **unscaled_to_append,
        )
        if profile_timing:
            _update_profile_meters(metric_logger, profile_values)

        if print_performance_on_fly:
            log_values = {}
            for k in criterions.keys():
                for e_k, e_v in evaluators[k].compute().items():
                    log_values.update({f"{k}_{e_k}": e_v.cpu().item()})
            print(log_values)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    configure_temporal_auroc_ci(
        evaluators,
        diseases,
        n_bootstrap=auroc_ci_bootstrap,
        seed=auroc_ci_seed,
        low_positive_threshold=low_positive_threshold,
    )
    evaluation_keys = list(diseases or []) + list(direct_labels or [])
    log_values = get_log_values(
        evaluation_keys,
        evaluators,
        metric_logger,
        diseases,
        evaluator_workers=(auroc_ci_workers if auroc_ci_bootstrap > 0 else 1),
    )

    if save_pred_path:
        save_preds_tgts(
            evaluators,
            diseases,
            save_pred_path,
            sample_ids=evaluated_sample_ids,
            direct_labels=direct_labels,
        )

    return log_values
