"""Canonical entry point for direct one-head multimodal training.

Historical staged experiments remain in ``train_onehead_staged_eval.py``.
Experiment settings may come from YAML, ordinary CLI flags, or repeatable
``--set KEY=VALUE`` overrides.
"""

import json
import math
import os, logging, warnings, torch, argparse, datetime, random, time, gc, shutil
import pandas as pd
import numpy as np
from evaluators.temporal import TemporalDiseaseEvaluator, TemporalEvalConfig
import utils.misc as utils
from model.universal_dense_vit_improved import build_universal_dense_vit
from utils.sampler import ImbalancedDatasetSampler, MultiLabelImbalancedDatasetSampler
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
from model.builder import setup_distributed_model
from utils.checkpoint import (
    save_dist_with_time,
    load_checkpoint_from_path,
)
from dataset.universal_public_dataset import build_combined_datasets
from dataset.universal_image import build_image_level_universal_datasets
from dataset.ukb_smri import build_smri_datasets
from dataset.builder import build_joint_task_loaders, build_loader, build_validation_loaders
from utils.checkpoint import load_continue_training, get_trained_epoch_from_name
from engine.universal_dense_onehead import (
    train_one_epoch,
    evaluate,
)
from utils.logger import GeneralTrainingLogger, log_mapping
import torch.distributed as dist
from utils.parameters import print_parameters_count
from model.universal_dense.input_output_projs import (
    build_universal_input_output_projs_batch,
)
from copy import deepcopy
from torch.distributed.elastic.multiprocessing.errors import record
from model.universal_dense.patch_emb import build_patch_embedding
from utils.gpu import log_gpu_info
from collections import OrderedDict
import torch.nn as nn
from evaluators.classification import ClassificationEvaluator
from evaluators.mse import MSEEvaluator
from utils.args import list_of_ints, list_of_str, list_of_floats
from utils.warning import supress_warnings
from utils.yaml_config import add_config_arguments, parse_configured_args
from training.checkpoint_loading import (
    inflate_checkpoint_tensor,
    load_pretrained_components,
    parse_component_pretrained_spec,
)
from training.model_selection import (
    full_validation_due,
    get_model_selection_value,
    metric_improved,
    selection_metric_improved,
    test_ci_bootstrap,
    update_patience_counter,
    validation_ci_bootstrap_for_epoch,
)
from training.stage2 import (
    build_stage2_optimizer,
    build_stage2_scheduler,
    set_backbone_requires_grad,
    stage2_epoch_range,
)
from transformers import get_cosine_schedule_with_warmup
from dataset.aug import get_default_aug
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from dataset.clsa_gender import CLSAGenderDataset

supress_warnings()


def get_participant_pos_weight(dataset, label: str, scale: float = 1.0) -> float:
    """Compute standard neg/pos BCE weight from unique participants."""
    if not hasattr(dataset, "df"):
        raise ValueError("Participant-level class weights require dataset.df.")
    required = {"patient_eid", label}
    missing = required.difference(dataset.df.columns)
    if missing:
        raise ValueError(
            f"Participant-level class weight is missing columns: {sorted(missing)}"
        )

    labels = dataset.df[["patient_eid", label]].dropna(subset=[label]).copy()
    labels["patient_eid"] = labels["patient_eid"].astype(str)
    binary = labels[label].astype(float) >= 0.5
    labels = labels.assign(_binary_label=binary)
    conflicts = labels.groupby("patient_eid")["_binary_label"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        examples = conflicts.index[:5].tolist()
        raise ValueError(
            f"Conflicting participant targets for {label}: {examples}"
        )

    participant_labels = labels.groupby("patient_eid")["_binary_label"].first()
    positive_count = int(participant_labels.sum())
    negative_count = int((~participant_labels).sum())
    if positive_count == 0 or negative_count == 0:
        raise ValueError(
            f"Cannot compute participant pos_weight for {label}: "
            f"positive={positive_count}, negative={negative_count}."
        )
    pos_weight = float(negative_count / positive_count) * float(scale)
    logging.info(
        "Participant class weight %s: participants=%d, positive=%d, "
        "negative=%d, pos_weight=%.6f",
        label,
        len(participant_labels),
        positive_count,
        negative_count,
        pos_weight,
    )
    return pos_weight


def sampler_summary_log_values(summary: dict) -> dict:
    """Convert any sampler summary into scalar logger fields."""
    return {
        f"sampler_{key}": value
        for key, value in summary.items()
        if isinstance(value, (bool, int, float, np.integer, np.floating))
    }


def parse_task_loss_weights(spec: str, possible_labels: list[str]) -> dict[str, float]:
    """Parse optional ``label:weight`` overrides for multi-task training."""
    weights = {label: 1.0 for label in possible_labels}
    if not spec:
        return weights

    known_labels = set(possible_labels)
    for raw_item in spec.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(
                "Invalid --task_loss_weights entry. Expected 'label:weight', "
                f"got {item!r}."
            )
        label, raw_weight = [part.strip() for part in item.split(":", 1)]
        if label not in known_labels:
            raise ValueError(
                f"Unknown task-loss label {label!r}; expected one of "
                f"{sorted(known_labels)}."
            )
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(
                f"Task-loss weight for {label!r} must be finite and non-negative, "
                f"got {raw_weight!r}."
            )
        weights[label] = weight
    return weights


def parse_joint_task_sampling(spec: str, diseases: list[str]) -> dict[str, str]:
    """Parse ``disease:participant|row`` task-loader modes."""
    if not spec:
        return {}
    modes = {}
    for raw_item in spec.split(","):
        if ":" not in raw_item:
            raise ValueError(
                "Invalid --joint_task_sampling entry; expected disease:participant|row"
            )
        disease, mode = [part.strip() for part in raw_item.split(":", 1)]
        if disease not in diseases:
            raise ValueError(
                f"Joint sampling task {disease!r} is not in diseases={diseases!r}"
            )
        if mode not in {"participant", "row"}:
            raise ValueError(f"Unsupported joint sampling mode {mode!r}")
        modes[disease] = mode
    if set(modes) != set(diseases):
        raise ValueError(
            "--joint_task_sampling must define every disease exactly once; "
            f"defined={sorted(modes)}, diseases={sorted(diseases)}"
        )
    return modes


# fmt: off
def get_args_parser():
    """Define all training arguments shared by YAML and direct CLI execution."""
    parser = argparse.ArgumentParser("Detection Training Script", add_help=False)

    data_args = parser.add_argument_group("Data")
    data_args.add_argument("--image_size", default=None, type=int, help="size of the images used in training, validation and testing.")
    data_args.add_argument(
        "--ukb_processed_df_path",
        default="data/ukb/foresight_omni_manifest.parquet",
        type=str,
        help=(
            "Processed UKB dataframe/manifest used by non-sMRI datasets. "
            "Legacy runs keep the fundus-derived default; expanded modality-only "
            "runs must pass a registry-controlled dataframe explicitly."
        ),
    )
    data_args.add_argument(
        "--public_data_root",
        default=os.environ.get("FORESIGHT_OMNI_PUBLIC_DATA_ROOT", "data/public"),
        type=str,
        help="Root containing Glaucoma_fundus and PAPILA ImageFolder directories.",
    )
    data_args.add_argument(
        "--require_any_requested_input",
        action="store_true",
        help=(
            "Drop rows that have none of the requested modality inputs. Use for "
            "expanded modality-only manifests; legacy behavior remains unchanged."
        ),
    )
    data_args.add_argument(
        "--allow_legacy_iop_manifest",
        action="store_true",
        help=(
            "Allow the historical non-eye-specific IOP representation. Use only "
            "for an explicitly documented legacy reproduction or correction run."
        ),
    )

    output_args = parser.add_argument_group("Experiment output")
    output_args.add_argument("--name", default="test", type=str, help="Name of the model")
    output_args.add_argument("--output_dir", default="", help="path where to save, empty for no saving")

    validation_args = parser.add_argument_group("Validation and early stopping")
    training_args = parser.add_argument_group("Training")
    validation_args.add_argument("--val_freq", default=1, type=int, help="Frequency of epochs to run evaluation, set it to 1 for running after every epoch.")
    validation_args.add_argument("--test_ci_bootstrap", default=2000, type=int, help="Bootstrap resamples for final/test AUROC 95%% CIs. Set 0 to disable.")
    validation_args.add_argument("--no_test_ci", dest="test_ci", action="store_false", help="Disable final/test AUROC confidence intervals.")
    parser.set_defaults(test_ci=True)
    validation_args.add_argument("--val_ci_bootstrap", default=0, type=int, help="Bootstrap resamples for optional validation AUROC 95%% CIs. Default 0 keeps training validation fast.")
    validation_args.add_argument("--val_ci_freq", default=0, type=int, help="Run validation AUROC CIs every N epochs when --val_ci_bootstrap > 0. Default 0 disables.")
    validation_args.add_argument("--auroc_ci_seed", default=42, type=int, help="Random seed for AUROC CI bootstrap.")
    validation_args.add_argument("--auroc_ci_workers", default=1, type=int, help="CPU workers used to evaluate independent disease bootstrap groups.")
    validation_args.add_argument("--low_positive_threshold", default=50, type=int, help="Flag AUROC tasks with fewer positive samples than this threshold.")
    validation_args.add_argument("--plot_freq", default=10, type=int, help="Regenerate compact monitoring figures every N epochs. Set 0 to render only at stage completion.")
    validation_args.add_argument("--legacy_detailed_plots", action="store_true", help="Also render the legacy large per-metric dashboards.")
    validation_args.add_argument("--save_val_predictions_freq", default=0, type=int, help="Save one combined validation-prediction CSV every N epochs. Set 0 to disable periodic saves.")
    validation_args.add_argument("--no_save_best_val_predictions", dest="save_best_val_predictions", action="store_false", help="Disable saving validation predictions for the best mean-AUROC epoch.")
    validation_args.add_argument("--no_save_last_val_predictions", dest="save_last_val_predictions", action="store_false", help="Disable saving validation predictions for the last evaluated epoch.")
    validation_args.add_argument("--participant_level_metrics", action="store_true", help="Average predictions across rows from the same patient before validation/test metrics and checkpoint selection.")
    validation_args.add_argument("--participant_level_metric_diseases", default=[], type=list_of_str, help="Diseases whose validation/test predictions are participant-aggregated while other diseases retain row metrics.")
    validation_args.add_argument(
        "--no_participant_level_metrics",
        dest="participant_level_metrics",
        action="store_false",
        help="Keep row/image-level metrics when a base configuration enables participant aggregation.",
    )
    parser.set_defaults(participant_level_metrics=False)
    validation_args.add_argument(
        "--classification_threshold_mode",
        default="validation_youden",
        choices=["fixed_0p5", "validation_youden"],
        help=(
            "Operating threshold for binary accuracy/confusion metrics. "
            "fixed_0p5 never fits a threshold; validation_youden fits on validation "
            "and locks it for final test evaluation."
        ),
    )
    validation_args.add_argument(
        "--evaluate_clsa_sex",
        action="store_true",
        help="After UKB testing, evaluate a patient_gender model on both QC and full CLSA without fine-tuning.",
    )
    validation_args.add_argument(
        "--clsa_sex_csv_path",
        default="data/clsa/clsa.csv",
        type=str,
    )
    validation_args.add_argument("--skip_final_test", action="store_true", help="Stop after validation-selected training without evaluating the reusable final test set.")
    parser.set_defaults(save_best_val_predictions=True, save_last_val_predictions=True)
    training_args.add_argument("--verbose", action="store_true", help="Print the info and debug information.")
    training_args.add_argument("--debug", action="store_true")
    training_args.add_argument("--imbalanced", action="store_true", help="if true, the dataset provide is imbalanced, and imbalanced sampler will be applied.")
    training_args.add_argument("--num_samples", default=None, type=int, help="Instance to pass through an imbalanced dataset for an epoch.")
    training_args.add_argument("--participant_level_sampling", action="store_true", help="Select exactly one randomly chosen row per participant per epoch, without replacement.")
    training_args.add_argument("--joint_task_sampling", default="", type=str, help="Alternate endpoint-specific loaders, e.g. 'mace:participant,t2d:row'.")
    training_args.add_argument("--participant_class_weight_labels", default=[], type=list_of_str, help="Binary labels whose pos_weight is computed from unique participants rather than rows.")
    training_args.add_argument("--smoke_test_max_rows_per_split", default=None, type=int, help="Deterministically cap each split for an end-to-end pipeline smoke test only.")
    training_args.add_argument("--sharding", action="store_true", help="Shard the model.")
    training_args.add_argument("--seed", default=42, type=int)
    training_args.add_argument("--device", default="cuda", help="device to use for training / testing")
    training_args.add_argument("--inspecting", action="store_true")
    training_args.add_argument("--lr", default=1e-3, type=float)
    training_args.add_argument("--lr_scheduler", action="store_true", help="Whether to use learning rate scheduler.")
    training_args.add_argument("--lr_drop", default=None, type=int)
    training_args.add_argument("--weight_decay", default=1e-4, type=float)
    training_args.add_argument("--epochs", default=300, type=int)
    training_args.add_argument("--batch_size", default=2, type=int)
    training_args.add_argument("--train_batch_size", default=None, type=int, help="Training batch size; falls back to --batch_size.")
    training_args.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Accumulate this many training micro-batches before each optimizer/scheduler step.")
    training_args.add_argument("--val_batch_size", default=None, type=int, help="Validation inference batch size; falls back to --batch_size.")
    training_args.add_argument("--test_batch_size", default=None, type=int, help="Test inference batch size; falls back to --batch_size.")
    training_args.add_argument("--hybrid_sampler", action="store_true", help="Use 75/25 coverage and rare-positive sampling instead of weighted replacement sampling.")
    training_args.add_argument("--hybrid_coverage_per_batch", default=48, type=int, help="Coverage rows per logical hybrid-sampler batch.")
    training_args.add_argument("--num_workers", default=0, type=int, help="Number of DataLoader worker processes.")
    training_args.add_argument("--eval_num_workers", default=None, type=int, help="Validation/test DataLoader workers; defaults to num_workers.")
    training_args.add_argument("--pin_memory", action="store_true", help="Use pinned host memory in DataLoaders.")
    training_args.add_argument("--bfloat16", action="store_true", help="Run model forward passes with CUDA bfloat16 autocast while keeping losses and metrics in float32.")
    training_args.add_argument("--persistent_workers", action="store_true", help="Keep DataLoader workers alive between epochs. Requires --num_workers > 0.")
    training_args.add_argument("--prefetch_factor", default=2, type=int, help="Number of batches prefetched by each DataLoader worker when --num_workers > 0.")
    training_args.add_argument(
        "--multiprocessing_sharing_strategy",
        default="file_descriptor",
        choices=["file_descriptor", "file_system"],
        help="PyTorch CPU tensor sharing strategy used by DataLoader workers.",
    )
    training_args.add_argument("--clip_max_norm", default=1, type=float, help="gradient clipping max norm")
    training_args.add_argument("--random_sample_start_epoch", default=None, type=int, help="The epoch to start random sampling ")
    training_args.add_argument("--modality_dropout_p", default=0.0, type=float, help="Independently drop available input modalities during training; never drops all modalities.")
    training_args.add_argument(
        "--modality_dropout_schedule",
        default="constant",
        choices=["constant", "linear"],
        help="Use a fixed rate or linearly ramp modality dropout by epoch.",
    )
    training_args.add_argument("--modality_dropout_start_p", default=0.0, type=float)
    training_args.add_argument("--modality_dropout_start_epoch", default=1, type=int)
    training_args.add_argument("--modality_dropout_end_epoch", default=1, type=int)
    training_args.add_argument("--sample_val_ratio", default=None, type=float, help="Ratio of the training dataset set used for validation. (The instances are sampled from validation dataset.)")
    training_args.add_argument("--elastic", action="store_true", help="Use L1 regularization with coef as weight_decay")
    training_args.add_argument("--class_weight", action="store_true", help="Weight for positive instances in the binary cross entropy loss.")
    training_args.add_argument("--pos_weight_scale", default=1, type=float, help="Scale the positive weight in the binary cross entropy loss.")
    training_args.add_argument("--binary_loss_type", default="bce", type=str, help="Loss for binary classification tasks.")
    training_args.add_argument(
        "--task_loss_weights",
        default="",
        type=str,
        help=(
            "Optional comma-separated per-target loss weights, for example "
            "'has_mace_in_0_years:1,patient_gender:0.1'. Unspecified targets "
            "retain weight 1."
        ),
    )
    validation_args.add_argument("--early_stop", action="store_true", help="Whether to use early stopping.")
    training_args.add_argument("--glaucoma_side_check", action="store_true")
    training_args.add_argument("--see_no_side_as_both", action="store_true")
    training_args.add_argument("--no_quality_control", action="store_true")
    training_args.add_argument(
        "--quality_control_splits",
        default=[],
        type=list_of_str,
        help=(
            "Optional comma-separated split override for image-level QC "
            "(train,val,test). When supplied, QC is enabled only for the "
            "listed splits, regardless of --no_quality_control. This permits "
            "training on the larger no-QC cohort while selecting on a fixed "
            "QC validation cohort."
        ),
    )
    training_args.add_argument("--include_self_report_label", action="store_true")
    training_args.add_argument("--include_eye_problem_label", action="store_true")
    training_args.add_argument("--survival_analysis_label", action="store_true", help="Whether to use survival analysis for the progression label.")
    training_args.add_argument("--no_aug", action="store_true", help="Not using augmentation during training.")
    training_args.add_argument("--enhanced_aug", action="store_true", help="Whether to use enhanced augmentation.")
    training_args.add_argument(
        "--fundus_aug_profile",
        default="kim_enhanced",
        choices=["default", "kim", "kim_enhanced"],
        help=(
            "Fundus augmentation recipe. 'kim' uses the matched crop/flip/rotation "
            "profile; 'kim_enhanced' adds mild colour jitter and is the default."
        ),
    )
    training_args.add_argument("--run_xai", action="store_true", help="Whether to run gradcam during validation")
    training_args.add_argument("--run_xai_fix", action="store_true", help="Whether to run gradcam for certain indexes during validation")
    training_args.add_argument("--ukb_downsampling", action="store_true", help="Whether to down sample the UKB dataset.")
    training_args.add_argument("--progression_label_ignorant_label_years", default=1, type=float, help="The years to ignore the progression label.")

    validation_args.add_argument("--xai_top_n", default=3, type=int, help="Top n classes to run gradcam on.")
    validation_args.add_argument("--best_model_buffer_size", default=5, type=int, help="Number of best checkpoints to keep per criterion (loss, auroc) before evicting the oldest.")
    validation_args.add_argument("--patience", "--stage2_patience", dest="patience", default=20, type=int, help="Early-stopping patience in authoritative validation checks.")
    validation_args.add_argument("--fast_val_size", default=8000, type=int, help="Fixed patient-level validation panel size used each epoch.")
    validation_args.add_argument("--two_tier_validation", action="store_true", help="Use a fixed fast panel each epoch plus authoritative full natural validation.")
    validation_args.add_argument("--full_val_freq", default=10, type=int, help="Frequency of authoritative full natural-validation checks.")
    validation_args.add_argument("--full_val_candidate_delta", default=0.001, type=float, help="Run full validation when fast-panel mean AUROC improves by this amount.")
    validation_args.add_argument("--model_selection_min_positives", default=10, type=int, help="Minimum validation positives required for an endpoint to enter checkpoint selection.")
    validation_args.add_argument(
        "--model_selection_labels",
        default=[],
        type=list_of_str,
        help=(
            "Optional comma-separated primary labels used for checkpoint selection, "
            "for example 'mace'. Auxiliary-label metrics remain logged but are excluded."
        ),
    )
    validation_args.add_argument("--early_stop_min_epoch", default=100, type=int, help="Do not early stop before this epoch.")

    checkpoint_args = parser.add_argument_group("Checkpoint loading")
    checkpoint_args.add_argument("--continue_training", type=str, default=None,  help="continue to train a model with given path, these weights will replace the pretrained one.")
    checkpoint_args.add_argument(
        "--reset_epoch_on_continue_training",
        action="store_true",
        help=(
            "Load --continue_training weights as a warm start but begin a new "
            "training schedule at epoch 1. Optimizer state is never restored."
        ),
    )
    checkpoint_args.add_argument(
        "--eval_only_checkpoint",
        action="store_true",
        help=(
            "Evaluate --continue_training directly without running additional "
            "epochs. Set scheduler_type=none and epochs to the checkpoint epoch."
        ),
    )
    checkpoint_args.add_argument(
        "--eval_only_validation_only",
        action="store_true",
        help=(
            "With --eval_only_checkpoint, evaluate and save the full validation "
            "results without evaluating the test set."
        ),
    )
    checkpoint_args.add_argument(
        "--leave_one_modality_out",
        action="store_true",
        help=(
            "During final-test evaluation, also evaluate the selected checkpoint "
            "with each configured input modality withheld in turn and save one "
            "prediction file per withheld modality."
        ),
    )
    checkpoint_args.add_argument("--inflate_mismatched_weights", action="store_true", help="When continuing from a smaller checkpoint, copy mismatched tensors into the overlapping prefix/block of the new larger tensors.")
    checkpoint_args.add_argument("--pretrained_path", type=str, default=None,  help="path to pretrained weights.")
    checkpoint_args.add_argument("--component_pretrained", default=[], type=list_of_str, help="Comma-separated encoder initializers using 'modality[+modality]=/checkpoint/path'. Only input_to_seq branches are loaded; fusion/head weights remain newly initialized.")
    checkpoint_args.add_argument(
        "--frozen_encoder_modalities",
        default=[],
        type=list_of_str,
        help=(
            "Comma-separated component-pretrained input_to_seq modalities to keep "
            "permanently frozen. Unlisted/newly initialized modality tokenizers "
            "remain trainable; the fusion transformer and prediction head are "
            "always outside this freeze set."
        ),
    )

    transformer_args = parser.add_argument_group("Shared transformer")
    transformer_args.add_argument("--causal", action="store_true", help="Use casual mask for attention if true")
    transformer_args.add_argument("--attn_dropout_p", default=0.0, type=float, help="Dropout rate for the attention score.",)
    transformer_args.add_argument("--ff_dropout_p", default=0.0, type=float, help="Dropout rate for the attention score.",)
    transformer_args.add_argument("--embedding_dropout_p", default=0.0, type=float, help="Dropout applied to encoded modality tokens before modality containers and output tokens are attached.",)
    transformer_args.add_argument("--prediction_head_dropout_p", default=0.0, type=float, help="Dropout applied to each pooled output-token representation immediately before its prediction head.",)
    transformer_args.add_argument(
        "--auxiliary_readout_map",
        default="",
        type=str,
        help=(
            "Optional leakage-safe auxiliary readouts as "
            "'target:modality,target:modality'. Mapped targets are predicted "
            "directly from the named modality encoder representation rather "
            "than the fused multimodal output."
        ),
    )
    transformer_args.add_argument("--prediction_conditioning_map", default="", type=str, help="Optional predicted-risk-factor bottleneck as 'target:source+source'. Conditioning sources are predicted from the same inputs and do not need to be observed for every target sample.")
    transformer_args.add_argument("--prediction_conditioning_detach", action="store_true", help="Stop target-loss gradients through conditioning prediction heads, approximating a two-stage risk-factor bottleneck while training jointly.")
    transformer_args.add_argument(
        "--target_modality_allow_map",
        default="",
        type=str,
        help=(
            "Target-specific access to protected modalities as "
            "'target:modality+modality,target:modality'. Any named modality is "
            "blocked from all unlisted targets and isolated from shared token "
            "updates to prevent indirect leakage."
        ),
    )
    transformer_args.add_argument(
        "--fusion_mode",
        default="transformer",
        choices=["transformer", "mean_pool"],
        help=(
            "Fusion architecture. 'mean_pool' bypasses learned output tokens and "
            "the fusion Transformer, applying prediction heads directly to the "
            "mean-pooled encoder representation; useful for literature-matched "
            "single-image CNN baselines."
        ),
    )
    transformer_args.add_argument(
        "--fusion_readout_mode",
        default="output_token",
        choices=["output_token", "global_summary", "residual_anchor"],
        help=(
            "Prediction readout for Transformer fusion. 'global_summary' "
            "initializes a data-dependent summary token from available modality "
            "representations and predicts only from its gated Transformer update. "
            "'residual_anchor' builds a disease-specific available-modality "
            "anchor and adds a zero-initialized Transformer correction."
        ),
    )
    transformer_args.add_argument(
        "--global_summary_residual_init",
        default=0.0,
        type=float,
        help=(
            "Initial interpolation gate between the encoder summary and its "
            "Transformer-updated state when fusion_readout_mode=global_summary."
        ),
    )
    transformer_args.add_argument("--patch_size", default=16, type=int, help="Size of patches used in input projector.",)
    transformer_args.add_argument("--n_layers", default=6, type=int, help="Number of decoding layers in the transformer")
    transformer_args.add_argument("--dim", default=512, type=int, help="Size of the embeddings (dimension of the transformer)")
    transformer_args.add_argument("--fusion_dim", default=None, type=int, help="Optional fusion Transformer width. Input tokenizers retain --dim for pretrained-checkpoint compatibility and use learned projections into this width.")
    transformer_args.add_argument("--n_heads", default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    transformer_args.add_argument("--n_kv_heads", default=None, type=int, help="Number of n_kv_head for group attention.")
    transformer_args.add_argument("--container", default="bracket", type=str, help="Method to contain modalities.")
    transformer_args.add_argument("--pos", default="sin-input", type=str, help="Positional Encoding strategy.")

    genotype_args = parser.add_argument_group("Genotype")
    genotype_args.add_argument("--genotype_autoencoder_path", default=None, type=str, help="Path to the pretrained Genotype Autoencoder")
    genotype_args.add_argument("--genotype_autoencoder_intermediate_dims", default="32,2048", type=list_of_ints, help="Intermediate dimensions for the genotype autoencoder.")
    genotype_args.add_argument("--genotype_autoencoder_patch_sizes", default="64,64", type=list_of_ints, help="Patch sizes for the genotype autoencoder.")
    genotype_args.add_argument("--genotype_last_patch_size", default="64", type=list_of_ints, help="Intermediate dimensions for the genotype embedding.")
    genotype_args.add_argument("--genotype_length", default=15889070, type=int, help="Length of the genotype.")
    genotype_args.add_argument("--genotype_lifetime_prediction", action="store_true", help="Whether to use genotype for predicting glaucoma progression.")
    multimodal_args = parser.add_argument_group("Omics and clinical history")
    multimodal_args.add_argument("--input_modalities", default=["fundus_image"], type=list_of_str, help="The input modalities to use. (fundus_image, oct_image, smri_image, prs, genotype, anthropometrics, family_history, principal_components, lifestyle, mental_health, socioeconomic, vitals, medications)")
    multimodal_args.add_argument("--omics_num_tokens", default=8, type=int, help="Number of tokens per omics modality (metabolomics/proteomics).")
    multimodal_args.add_argument("--omics_tokenizer", default="grouped", choices=["grouped", "linear"], help="Omics tokenizer: grouped is missingness-aware; linear preserves the legacy projection.")
    multimodal_args.add_argument("--omics_dropout_p", default=0.0, type=float, help="Dropout used inside the grouped omics tokenizer.")
    multimodal_args.add_argument("--no_omics_qc", dest="omics_qc", action="store_false", help="Disable train-fitted omics QC.")
    parser.set_defaults(omics_qc=True)
    multimodal_args.add_argument("--omics_missing_rate_threshold", default=0.40, type=float, help="Mask omics features with train missing rate above this threshold.")
    multimodal_args.add_argument("--omics_min_std", default=1e-6, type=float, help="Mask omics features with train standard deviation below this value.")
    multimodal_args.add_argument("--omics_min_unique_values", default=10, type=int, help="Mask omics features with fewer train unique non-missing values.")
    multimodal_args.add_argument("--omics_winsor_lower_quantile", default=0.005, type=float, help="Train quantile used for lower omics winsorization.")
    multimodal_args.add_argument("--omics_winsor_upper_quantile", default=0.995, type=float, help="Train quantile used for upper omics winsorization.")
    multimodal_args.add_argument("--clinical_history_min_count", default=10, type=int, help="Minimum train ICD-10 count retained in clinical-history vocabulary.")
    multimodal_args.add_argument("--clinical_history_max_len", default=128, type=int, help="Maximum pre-index ICD-10 events used for clinical-history encoding.")
    multimodal_args.add_argument("--clinical_history_num_tokens", default=1, type=int, help="Number of clinical-history summary tokens.")
    multimodal_args.add_argument("--clinical_history_encoder_type", default="pooled", choices=["pooled", "attention"], help="Clinical-history encoder: pooled preserves the legacy one-token mean pool; attention uses learned query tokens over ICD-10 events.")
    multimodal_args.add_argument("--no_clinical_history_mask_targets", dest="clinical_history_mask_targets", action="store_false", help="Disable target/proxy ICD-10 masking for clinical history.")
    parser.set_defaults(clinical_history_mask_targets=True)
    multimodal_args.add_argument("--clinical_history_mask_map_path", default="reports/icd10/parent_descendants_complications_icd10_map.json", type=str, help="Path to target/proxy ICD-10 masking JSON.")
    multimodal_args.add_argument("--clinical_history_dropout_p", default=0.0, type=float, help="Dropout used inside the clinical-history tokenizer.")
    multimodal_args.add_argument("--clinical_history_vocab_size", default=10000, type=int, help="Fallback ICD-10 vocab size; overwritten from train dataset when clinical_history is used.")
    multimodal_args.add_argument("--questionnaire_max_choices", default=16, type=int, help="Maximum selected responses retained for one multi-select questionnaire field.")
    multimodal_args.add_argument("--questionnaire_dropout_p", default=0.1, type=float, help="Dropout inside the type-aware questionnaire tokenizer.")
    multimodal_args.add_argument("--questionnaire_field_dropout_p", default=0.1, type=float, help="Probability of replacing an observed questionnaire field with a learned field-dropout state during training.")
    multimodal_args.add_argument("--questionnaire_use_subgroup_embeddings", action="store_true", help="Add a learned clinical subgroup embedding to each questionnaire field token.")
    multimodal_args.add_argument("--questionnaire_subgroup_scheme", default="six", choices=["six", "seven"], help="Internal questionnaire grouping: six preserves the initial proposal; seven uses Puya's approved mental-health/general-health split.")
    multimodal_args.add_argument("--progression_label_years", default=[0, 2, 5, 10, 15], type=list_of_ints, help="Years to use for the progression label.")
    target_args = parser.add_argument_group("Targets and clinical features")
    target_args.add_argument("--diseases", default=[], type=list_of_str, help="Diseases to use for the progression label. Available diseases: glaucoma, ad, pd, hd, ms, cvd,t2d")
    target_args.add_argument("--external_binary_phenotype_path", default=None, type=str, help="Optional tabular case/control phenotype merged by UKB participant ID.")
    target_args.add_argument("--external_binary_phenotype_id_col", default="IID", type=str, help="Participant ID column in --external_binary_phenotype_path.")
    target_args.add_argument("--external_binary_phenotype_label_col", default="phenotype", type=str, help="Binary label column in --external_binary_phenotype_path.")
    target_args.add_argument("--external_binary_target_disease", default=None, type=str, help="Disease name receiving the external binary label; requires a single year-0 target.")
    target_args.add_argument("--external_binary_phenotype_date_col", default=None, type=str, help="Optional phenotype index-date column for participant-and-date matching.")
    target_args.add_argument("--external_binary_instance_date_col", default=None, type=str, help="Manifest visit-date column paired with --external_binary_phenotype_date_col.")
    target_args.add_argument("--no_external_binary_restrict_cohort", dest="external_binary_restrict_cohort", action="store_false", help="Merge the external target but retain rows without that target for masked joint-task training.")
    parser.set_defaults(external_binary_restrict_cohort=True)
    target_args.add_argument("--external_prs_path", default=None, type=str, help="Optional external PRS score table merged by UKB participant ID.")
    target_args.add_argument("--external_prs_id_col", default="IID", type=str, help="Participant ID column in --external_prs_path.")
    target_args.add_argument("--external_prs_score_col", default="SCORE1_AVG", type=str, help="Score column in --external_prs_path.")
    target_args.add_argument("--require_external_prs", action="store_true", help="Restrict the cohort to participants with a nonmissing external PRS.")
    target_args.add_argument("--disease_label_tokens", default=1, type=int, help="Default number of output label tokens per disease.")
    target_args.add_argument("--disease_label_tokens_map", default="", type=str, help="Optional per-disease token overrides as 'glaucoma:2,ad:3'.")
    target_args.add_argument("--clinical_cat_modalities", default=["instance_comparative_body_size_at_age_10"], type=list_of_str, help="Clinical categorical modalities to use as input.")
    target_args.add_argument("--clinical_num_modalities", default=["instance_age_at_time", "instance_iop", "instance_systolic_bp", "instance_diastolic_bp", "instance_pulse_rate", "instance_height_cm", "instance_weight_kg", "instance_body_mass_index_bmi", "instance_waist_circumference_cm", "instance_hip_circumference_cm", ], type=list_of_str, help="Clinical numerical modalities to use as input.")
    target_args.add_argument("--cat_labels", default=None, type=list_of_str, help="Categorical labels to use. If None, use all available categorical labels.")
    target_args.add_argument("--numerical_labels", default=None, type=list_of_str, help="Numerical labels to use. If None, use all available numerical labels.")
    target_args.add_argument("--fundus_sample_prob", default=1.0, type=float, help="Sample probability for fundus images during training.")
    target_args.add_argument("--finetune_epochs", default=50, type=int, help="Number of epochs to finetune on CLSA dataset.")

    image_args = parser.add_argument_group("Fundus, OCT, and sMRI encoders")
    image_args.add_argument("--normalise_fundus_image", action="store_true", help="Whether to normalise fundus images.")
    image_args.add_argument("--fundus_cache_path", default=None, type=str, help="Optional uint8 fundus memmap cache staged on the compute node.")
    image_args.add_argument("--fundus_cache_index_path", default=None, type=str, help="Parquet mapping from image_path to cache_index.")
    image_args.add_argument("--fundus_cache_metadata_path", default=None, type=str, help="JSON metadata describing the fundus memmap shape and preprocessing.")
    image_args.add_argument("--fundus_require_cache", action="store_true", help="Fail during dataset construction if any requested fundus image is absent from the cache.")
    image_args.add_argument(
        "--fundus_cache_allow_resize",
        action="store_true",
        help=(
            "Allow a centre-crop uint8 cache at a different source resolution; "
            "the ordinary dataset transform then resizes it to --image_size."
        ),
    )
    image_args.add_argument("--fundus_token_grid_size", default=None, type=int, help="Optional fixed H=W grid for adaptive pooling of final fundus CNN features before Transformer token projection.")
    image_args.add_argument(
        "--fundus_prepend_global_token",
        action="store_true",
        help=(
            "Prepend the native global-average-pooled CNN representation to "
            "fundus spatial tokens. When dim matches the CNN width, this token "
            "has no learned projection."
        ),
    )
    image_args.add_argument(
        "--fundus_global_token_only",
        action="store_true",
        help=(
            "Return only the native global-average-pooled CNN representation "
            "as the fundus token. This bypasses spatial token projection but "
            "retains the ordinary Transformer output-token prediction path."
        ),
    )
    image_args.add_argument(
        "--fundus_global_token_dim",
        default=None,
        type=int,
        help=(
            "Optional native GAP output width for --fundus_global_token_only. "
            "This allows a 512d ResNet anchor while other pretrained modality "
            "tokenizers retain a smaller --dim."
        ),
    )
    image_args.add_argument("--use_combined_dataset", action="store_true", help="Whether to use the combined CLSA dataset.")
    image_args.add_argument("--image_encoder_type", default="patch_emb", type=str, help="Type of fundus image encoder to use. Options are 'patch_emb', 'cnn', or 'conv_token'.")
    image_args.add_argument(
        "--fundus_backbone_name",
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="ImageNet-pretrained CNN backbone used by the fundus conv-token encoder.",
    )
    image_args.add_argument("--oct_backbone_name", default="resnet18", type=str, help="2D CNN backbone used for OCT B-scan tokenisation.")
    image_args.add_argument("--oct_img_channels", default=1, type=int, help="Number of channels per OCT B-scan.")
    image_args.add_argument("--oct_max_slices", default=128, type=int, help="Maximum number of OCT B-scans in one volume.")
    image_args.add_argument("--oct_num_slices", default=128, type=int, help="Number of OCT B-scans loaded from each zip volume.")
    image_args.add_argument("--oct_image_size", default=None, type=int, help="Spatial size for OCT B-scan resizing. If unset, uses image_size.")
    image_args.add_argument(
        "--oct_aug_profile",
        default="oct_clinical_v1",
        choices=["none", "oct_clinical_v1"],
        help=(
            "OCT volume augmentation recipe. The default applies coherent small "
            "geometry, intensity/gamma jitter, noise, slice dropout and slice-grid jitter."
        ),
    )
    image_args.add_argument("--oct_num_tokens", default=128, type=int, help="Number of learned OCT volume tokens after resampling. Set <=0 to keep all slice-spatial tokens.")
    image_args.add_argument("--oct_encoder_type", default="flat_resampler", choices=["flat_resampler", "slice_transformer", "slice_conv1d", "tubelet_vit"], help="OCT tokenisation strategy.")
    image_args.add_argument("--oct_slice_tokens", default=4, type=int, help="Number of tokens used to summarise each B-scan before the slice transformer.")
    image_args.add_argument("--oct_slice_transformer_layers", default=1, type=int, help="Number of slice-level transformer layers for OCT.")
    image_args.add_argument("--oct_bscan_chunk_size", default=256, type=int, help="Number of OCT B-scans encoded at once by the 2D slice encoder.")
    image_args.add_argument("--oct_tubelet_size", default=[8, 32, 32], type=list_of_ints, help="Depth,height,width Conv3D tubelet size for --oct_encoder_type tubelet_vit.")
    image_args.add_argument("--oct_tubelet_transformer_layers", default=2, type=int, help="Number of Transformer encoder layers used after OCT tubelet embedding.")
    image_args.add_argument("--oct_tubelet_dropout", default=0.1, type=float, help="Dropout used inside the OCT tubelet Transformer.")
    image_args.add_argument("--oct_freeze_backbone", action="store_true", help="Freeze the OCT B-scan CNN backbone.")
    image_args.add_argument("--oct_no_pretrained", action="store_true", help="Do not use ImageNet-pretrained weights for the OCT B-scan CNN backbone.")
    image_args.add_argument("--smri_manifest_path", default="data/ukb/smri_manifest.parquet", type=str, help="Scan-level T1 sMRI manifest created from the full UKB participant tab.")
    image_args.add_argument("--smri_cache_root", default="data/ukb/smri_cache_96", type=str, help="Root containing preprocessed T1 volumes grouped by participant-ID suffix.")
    image_args.add_argument("--smri_require_cache", action="store_true", help="Fail instead of inflating the source ZIP when a preprocessed T1 cache file is missing.")
    image_args.add_argument("--smri_volume_size", default=96, type=int, help="Isotropic voxel size after foreground crop and trilinear resizing.")
    image_args.add_argument("--smri_num_tokens", default=4, type=int, help="Number of learned T1 sMRI summary tokens emitted to multimodal fusion.")
    image_args.add_argument("--smri_base_channels", default=16, type=int, help="Base channel width of the T1 3D CNN.")
    image_args.add_argument("--smri_num_heads", default=4, type=int, help="Attention heads used to resample T1 spatial features into tokens.")
    image_args.add_argument("--smri_dropout_p", default=0.1, type=float, help="Dropout in the T1 sMRI token resampler.")
    image_args.add_argument("--smri_no_aug", action="store_true", help="Disable conservative random-axis flips in sMRI training.")
    image_args.add_argument("--profile_timing", action="store_true", help="Log detailed timing breakdowns for data loading, OCT decoding, device transfer, model forward, loss, and optimizer steps.")
    image_args.add_argument("--profile_timing_sync_cuda", action="store_true", help="Synchronize CUDA around profiled sections for more accurate GPU timing. Slower; intended for debugging.")

    qc_args = parser.add_argument_group("Quality control and sampling")
    qc_args.add_argument("--algo_qc", action="store_true", help="Whether to use algorithmic quality control for fundus images.")
    qc_args.add_argument(
        "--algo_qc_path",
        default=os.environ.get(
            "FORESIGHT_OMNI_UKB_ALGO_QC", "data/ukb/algorithmic_qc.csv"
        ),
        type=str,
        help="Per-image algorithmic QC CSV aligned to the prepared UKB manifest.",
    )

    qc_args.add_argument("--weight_sample_validation", action="store_true", help="Whether to use weighted sampling for validation dataset.")

    qc_args.add_argument("--finetune_portions", default=[0.4], type=list_of_floats, help="Comma-separated CLSA finetuning data fractions, e.g. 0.4 or 0.2,0.4,0.6.")
    qc_args.add_argument("--finetune_train_frac", default=0.75, type=float, help="Fraction of each CLSA finetune subset used for finetune train split (rest is validation).")

    stage2_args = parser.add_argument_group("Multimodal optimization")
    stage2_args.add_argument("--backbone_lr", "--stage2_backbone_lr", dest="backbone_lr", default=1e-4, type=float, help="Learning rate for pretrained modality encoders.")
    stage2_args.add_argument("--head_lr", "--stage2_head_lr", dest="head_lr", default=1e-3, type=float, help="Learning rate for fusion and prediction parameters.")
    stage2_args.add_argument("--optimizer_type", default="adamw", choices=["adamw", "adam"], help="Optimizer for direct multimodal training.")
    stage2_args.add_argument("--freeze_backbone_epochs", "--stage2_freeze_backbone_epochs", dest="freeze_backbone_epochs", default=3, type=int, help="Number of initial epochs to freeze pretrained modality encoders.")
    stage2_args.add_argument(
        "--backbone_freeze_patterns",
        default=[],
        type=list_of_str,
        help=(
            "Optional comma-separated parameter-name substrings identifying "
            "the pretrained parameters to freeze initially and assign to the "
            "backbone learning-rate group. Empty preserves legacy behavior and "
            "treats every input_to_seq parameter as backbone."
        ),
    )
    stage2_args.add_argument(
        "--backbone_unfreeze_patterns",
        default=[],
        type=list_of_str,
        help=(
            "Optional comma-separated parameter-name substrings to unfreeze "
            "after freeze_backbone_epochs. Empty means unfreeze every eligible "
            "encoder parameter. For the current ResNet fundus encoder, a "
            "conservative last-block adaptation can select feature_extractor.7 "
            "and token_proj."
        ),
    )
    stage2_args.add_argument(
        "--scheduler_type", "--stage2_scheduler_type",
        dest="scheduler_type",
        default="cosine",
        choices=["none", "cosine"],
        type=str,
        help="Scheduler type for stage 2.",
    )
    stage2_args.add_argument("--warmup_ratio", "--stage2_warmup_ratio", dest="warmup_ratio", default=0.0, type=float, help="Warmup ratio for the multimodal scheduler.")
    stage2_args.add_argument(
        "--delayed_backbone_scheduler",
        action="store_true",
        help=(
            "Keep the pretrained-backbone LR at zero while frozen, then start "
            "a separate warmup and cosine schedule when it is unfrozen."
        ),
    )
    stage2_args.add_argument(
        "--backbone_warmup_epochs",
        default=1,
        type=int,
        help="Warmup epochs for the delayed pretrained-backbone schedule.",
    )
    stage2_args.add_argument(
        "--model_selection_metric",
        default="val_loss",
        choices=["val_loss", "mean_auroc", "mean_accuracy"],
        type=str,
        help="Metric used by early-stopping/best-checkpoint selection.",
    )

    return parser
# fmt: on


@record
def main(args):
    """Run direct multimodal training, checkpoint selection, and final testing."""
    # --- Runtime initialisation -------------------------------------------------

    if args.debug:
        args.verbose = True

    logging.basicConfig(
        filename=None if args.debug else (args.output_dir + f"{args.name}.log"),
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    log_mapping("Resolved training configuration", vars(args))

    logging.info("Initializing runtime")
    utils.init_distributed_mode(args)
    output_dir = Path(args.output_dir)

    torch.multiprocessing.set_sharing_strategy(args.multiprocessing_sharing_strategy)
    logging.info(
        "PyTorch multiprocessing sharing strategy: %s",
        torch.multiprocessing.get_sharing_strategy(),
    )

    if torch.cuda.is_available():
        if args.bfloat16 and not torch.cuda.is_bf16_supported():
            raise RuntimeError("--bfloat16 requested but the selected CUDA device does not support BF16")
        torch.cuda.empty_cache()
        gc.collect()
        logging.info(torch.cuda.memory_summary())

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # --- Modalities and targets -------------------------------------------------

    diseases = args.diseases
    joint_task_sampling = parse_joint_task_sampling(args.joint_task_sampling, diseases)
    restricted_inputs = args.input_modalities
    progression_label_years = args.progression_label_years

    if args.input_modalities is None or len(args.input_modalities) == 0:
        raise ValueError(
            "No input modalities provided, please provide at least one input modality."
        )
    cat_labels = []
    if args.cat_labels is not None and len(args.cat_labels) > 0:
        cat_labels.extend(args.cat_labels)

    if args.diseases is not None and len(args.diseases) > 0:
        disease_cat_labels = [
            f"has_{disease}_in_{y}_years"
            for disease in diseases
            for y in progression_label_years
        ]
        cat_labels.extend(disease_cat_labels)

    if args.numerical_labels is not None and len(args.numerical_labels) > 0:
        numerical_labels = args.numerical_labels
    else:
        numerical_labels = []

    possible_labels = cat_labels + numerical_labels
    disease_cat_label_set = set(disease_cat_labels) if diseases else set()
    direct_cat_labels = [label for label in cat_labels if label not in disease_cat_label_set]
    direct_labels = direct_cat_labels + numerical_labels
    balance_label_cols = possible_labels

    label_num_classes = {d: len(args.progression_label_years) for d in diseases}
    label_num_classes.update({label: 1 for label in direct_labels})
    if args.disease_label_tokens < 1:
        raise ValueError(
            f"disease_label_tokens must be >= 1, got {args.disease_label_tokens}"
        )
    label_tokens_len = {d: int(args.disease_label_tokens) for d in diseases}
    label_tokens_len.update({label: 1 for label in direct_labels})

    if args.disease_label_tokens_map:
        for raw_item in args.disease_label_tokens_map.split(","):
            item = raw_item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(
                    "Invalid --disease_label_tokens_map format. "
                    f"Expected 'disease:tokens', got '{item}'."
                )
            disease_name, token_str = [part.strip() for part in item.split(":", 1)]
            if disease_name not in label_tokens_len:
                raise ValueError(
                    f"Unknown disease [{disease_name}] in --disease_label_tokens_map. "
                    f"Known diseases: {sorted(label_tokens_len.keys())}"
                )
            token_count = int(token_str)
            if token_count < 1:
                raise ValueError(
                    f"Token count for disease [{disease_name}] must be >= 1, got {token_count}"
                )
            label_tokens_len[disease_name] = token_count

    eval_inputs = restricted_inputs
    eval_labels = possible_labels

    logging.info("Input modalities: %s", restricted_inputs)
    logging.info("Categorical labels: %s", cat_labels)
    logging.info("Numerical labels: %s", numerical_labels)
    logging.info("All labels: %s", possible_labels)

    # --- Data loaders -----------------------------------------------------------
    logging.info("Building datasets and data loaders")

    if "smri_image" in restricted_inputs:
        if restricted_inputs != ["smri_image"]:
            raise NotImplementedError(
                "The first implementation supports modality-specific sMRI pretraining only. "
                "Build a temporally re-indexed ocular+sMRI fusion manifest before mixing "
                "smri_image with other inputs."
            )
        if args.use_combined_dataset:
            raise ValueError("--use_combined_dataset is not supported for sMRI pretraining")
        dataset_builder = build_smri_datasets
    else:
        dataset_builder = (
            build_combined_datasets
            if args.use_combined_dataset
            else build_image_level_universal_datasets
        )

    train_dataset, val_dataset, test_dataset = dataset_builder(
        args,
        possible_inputs=restricted_inputs,
        possible_labels=possible_labels,
        balance_label_cols=balance_label_cols,
        quality_control=not args.no_quality_control,
        progression_label_years=progression_label_years,
        clinical_numerical_features=args.clinical_num_modalities,
        clinical_categorical_features=args.clinical_cat_modalities,
        enhanced_aug=args.enhanced_aug,
        no_aug=args.no_aug,
        numerical_label_cols=numerical_labels,
        progression_label_ignorant_label_years=args.progression_label_ignorant_label_years,
        normalise_fundus_image=args.normalise_fundus_image,
        fundus_cache_path=args.fundus_cache_path,
        fundus_cache_index_path=args.fundus_cache_index_path,
        fundus_cache_metadata_path=args.fundus_cache_metadata_path,
        fundus_require_cache=args.fundus_require_cache,
        algo_qc=args.algo_qc,
        algo_qc_path=args.algo_qc_path,
        external_binary_phenotype_path=args.external_binary_phenotype_path,
        external_binary_phenotype_id_col=args.external_binary_phenotype_id_col,
        external_binary_phenotype_label_col=args.external_binary_phenotype_label_col,
        external_binary_target_disease=args.external_binary_target_disease,
        external_binary_restrict_cohort=args.external_binary_restrict_cohort,
        external_binary_phenotype_date_col=args.external_binary_phenotype_date_col,
        external_binary_instance_date_col=args.external_binary_instance_date_col,
        external_prs_path=args.external_prs_path,
        external_prs_id_col=args.external_prs_id_col,
        external_prs_score_col=args.external_prs_score_col,
        require_external_prs=args.require_external_prs,
        fundus_aug_profile=args.fundus_aug_profile,
    )
    if args.smoke_test_max_rows_per_split is not None:
        cap = int(args.smoke_test_max_rows_per_split)
        if cap <= 0:
            raise ValueError("smoke_test_max_rows_per_split must be positive")
        for split_offset, dataset in enumerate(
            (train_dataset, val_dataset, test_dataset)
        ):
            if not hasattr(dataset, "df"):
                logging.info(
                    "Top-level smoke cap skipped for %s; component datasets "
                    "apply their own supported caps.",
                    type(dataset).__name__,
                )
                continue
            if len(dataset.df) > cap:
                dataset.df = dataset.df.sample(
                    n=cap,
                    random_state=args.seed + split_offset,
                ).reset_index(drop=True)
        logging.warning(
            "SMOKE TEST DATA CAP ACTIVE: train=%d val=%d test=%d",
            len(train_dataset),
            len(val_dataset),
            len(test_dataset),
        )
    if hasattr(train_dataset, "omics_qc_report") and args.output_dir:
        omics_qc_report_path = output_dir / "omics_qc_report.json"
        with open(omics_qc_report_path, "w") as f:
            json.dump(train_dataset.omics_qc_report, f, indent=2)
        logging.info("Saved omics QC report: %s", omics_qc_report_path)
    if "clinical_history" in restricted_inputs and hasattr(train_dataset, "icd10_code_to_id"):
        args.clinical_history_vocab_size = len(train_dataset.icd10_code_to_id)
        logging.info(
            "Clinical history vocabulary size: %d",
            args.clinical_history_vocab_size,
        )
        if args.output_dir:
            clinical_history_report_path = output_dir / "clinical_history_vocab_report.json"
            with open(clinical_history_report_path, "w") as f:
                json.dump(train_dataset.clinical_history_vocab_report, f, indent=2)
            logging.info(
                f"Saved clinical history vocab report to [{clinical_history_report_path}]"
            )
    if "questionnaire" in restricted_inputs:
        args.questionnaire_num_fields = train_dataset.questionnaire_num_fields
        args.questionnaire_category_vocab_size = (
            train_dataset.questionnaire_category_vocab_size
        )
        if args.questionnaire_num_fields <= 0:
            raise ValueError("questionnaire requested, but no questionnaire fields were fitted")
        logging.info(
            "Questionnaire schema: fields=%d, category_vocab=%d, max_choices=%d",
            args.questionnaire_num_fields,
            args.questionnaire_category_vocab_size,
            args.questionnaire_max_choices,
        )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": args.pin_memory,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    eval_num_workers = args.num_workers if args.eval_num_workers is None else int(args.eval_num_workers)
    if eval_num_workers < 0:
        raise ValueError("eval_num_workers must be >= 0")
    eval_loader_kwargs = {
        "num_workers": eval_num_workers,
        "pin_memory": args.pin_memory,
    }
    if eval_num_workers > 0:
        eval_loader_kwargs["persistent_workers"] = False
        eval_loader_kwargs["prefetch_factor"] = args.prefetch_factor
    logging.info(
        "DataLoader workers: train=%d persistent=%s; eval=%d persistent=False",
        args.num_workers,
        args.persistent_workers and args.num_workers > 0,
        eval_num_workers,
    )

    if joint_task_sampling:
        if args.two_tier_validation:
            raise ValueError("Joint task sampling does not yet support two-tier validation")
        if args.participant_level_sampling or args.imbalanced or args.num_samples:
            raise ValueError(
                "Joint task sampling owns task-specific sampling and cannot be combined "
                "with global participant/imbalanced/num_samples settings"
            )
        primary_labels = [
            f"has_{disease}_in_{year}_years"
            for disease in diseases
            for year in progression_label_years
        ]
        train_d, val_d, test_d = build_joint_task_loaders(
            args,
            train_dataset,
            val_dataset,
            test_dataset,
            task_sampling=joint_task_sampling,
            primary_labels=primary_labels,
            **loader_kwargs,
        )
    else:
        train_d, val_d, test_d = build_loader(
            args,
            train_dataset,
            val_dataset,
            test_dataset,
            collate_fn=lambda x: list(x),
            weight_sample_validation=args.weight_sample_validation,
            eval_loader_kwargs=eval_loader_kwargs,
            **loader_kwargs,
        )
    if getattr(args, "two_tier_validation", False):
        val_d, full_val_d, fast_val_manifest_path = build_validation_loaders(
            args,
            val_dataset,
            collate_fn=lambda x: list(x),
            **eval_loader_kwargs,
        )
        logging.info("Fixed fast-validation panel: %s", fast_val_manifest_path)
    else:
        full_val_d = val_d
    sampler_state_path = output_dir / "training_sampler_state.pt"
    if args.continue_training and sampler_state_path.exists() and hasattr(
        train_d.sampler, "load_state_dict"
    ):
        train_d.sampler.load_state_dict(
            torch.load(sampler_state_path, map_location="cpu")
        )
        logging.info("Restored training sampler state: %s", sampler_state_path)

    # --- Model construction and checkpoint loading -----------------------------
    logging.info("Building model")
    log_gpu_info()

    device = torch.device(args.device, args.local_rank if args.distributed else 0)
    input_to_seq, activations = build_universal_input_output_projs_batch(
        args,
        device,
        possible_input_modalities=restricted_inputs,
        binary_label_cols=possible_labels,
        numerical_label_cols=numerical_labels,
        omics_num_tokens=args.omics_num_tokens,
        genotype_autoencoder_intermediate_dims=args.genotype_autoencoder_intermediate_dims,
        genotype_autoencoder_patch_sizes=args.genotype_autoencoder_patch_sizes,
        genotype_last_patch_size=args.genotype_last_patch_size,
        genotype_length=args.genotype_length,
        image_encoder_type=args.image_encoder_type,
    )
    model = build_universal_dense_vit(
        args,
        input_to_seq=input_to_seq,
        label_num_classes=label_num_classes,
        label_tokens_len=label_tokens_len,
    )
    logging.info("Model architecture:\n%s", model)
    log_gpu_info()

    n_trainable_parameters, n_total_parameters = print_parameters_count(model)
    model.to(device)

    if args.pretrained_path:
        cp = load_checkpoint_from_path(args.pretrained_path, device)
        # Older checkpoints used "image" where current modules use "fundus_image".
        remapped_cp_model_dict = OrderedDict()
        for key, value in cp["model"].items():
            # Rename only a complete legacy module-path component.  A global
            # string replacement corrupts already-current names such as
            # ``fundus_image`` into ``fundus_fundus_image``.
            remapped_key = ".".join(
                "fundus_image" if part in {"image", "fundus-image"} else part
                for part in key.split(".")
            )
            remapped_cp_model_dict[remapped_key] = value
        # Pretraining targets are not transferable task features.  In particular,
        # a disease checkpoint may have a different number of horizon logits from
        # the downstream task.  Load the encoders/fusion transformer but always
        # initialise downstream output tokens and heads afresh, matching the sex
        # benchmark protocol and avoiding accidental target-head reuse.
        model_state = model.state_dict()
        cp_model_dict = OrderedDict()
        skipped_pretrained_keys = []
        for key, value in remapped_cp_model_dict.items():
            if key.startswith(("output_tokens.", "output_layers.")):
                skipped_pretrained_keys.append((key, "downstream target parameter"))
                continue
            if key not in model_state:
                skipped_pretrained_keys.append((key, "not present downstream"))
                continue
            if model_state[key].shape != value.shape:
                skipped_pretrained_keys.append(
                    (
                        key,
                        f"shape {tuple(value.shape)} -> {tuple(model_state[key].shape)}",
                    )
                )
                continue
            cp_model_dict[key] = value
        model.load_state_dict(cp_model_dict, strict=False)
        logging.info(
            "Loaded %d compatible pretrained encoder/fusion tensors from %s; "
            "skipped %d target-specific, absent, or shape-incompatible tensors",
            len(cp_model_dict),
            args.pretrained_path,
            len(skipped_pretrained_keys),
        )
        if skipped_pretrained_keys:
            logging.info("Skipped pretrained tensors: %s", skipped_pretrained_keys)
        del cp
        del remapped_cp_model_dict
        del cp_model_dict
        del model_state
        torch.cuda.empty_cache()
        gc.collect()

    if args.component_pretrained:
        if args.pretrained_path or args.continue_training:
            raise ValueError(
                "--component_pretrained cannot be combined with --pretrained_path "
                "or --continue_training"
            )
        component_summary = load_pretrained_components(
            model,
            args.component_pretrained,
            device=device,
            checkpoint_loader=load_checkpoint_from_path,
            logger=logging,
        )
        logging.info("Component checkpoint summary: %s", component_summary)
        torch.cuda.empty_cache()
        gc.collect()

    loaded_component_modalities = set()
    for spec in args.component_pretrained:
        modalities, _ = parse_component_pretrained_spec(spec)
        loaded_component_modalities.update(modalities)

    frozen_encoder_modalities = set(args.frozen_encoder_modalities)
    if frozen_encoder_modalities:
        not_component_pretrained = (
            frozen_encoder_modalities - loaded_component_modalities
        )
        if not_component_pretrained:
            raise ValueError(
                "--frozen_encoder_modalities may contain only modalities loaded "
                "through --component_pretrained; missing component checkpoint(s): "
                f"{sorted(not_component_pretrained)}"
            )
    newly_initialized_encoder_modalities = (
        set(args.input_modalities) - loaded_component_modalities
        if frozen_encoder_modalities
        else set()
    )

    current_epoch = 1
    loaded_epoch = None

    if args.continue_training:
        cp = load_checkpoint_from_path(args.continue_training, device)

        fixed_state_dict = {}
        for k, v in cp["model"].items():
            new_k = k.replace("fundus-image", "fundus_image")
            fixed_state_dict[new_k] = v


        model_sd = model.state_dict()
        loaded = 0
        inflated = 0
        legacy_mapped = 0
        skipped = 0

        for k, v in fixed_state_dict.items():
            if k not in model_sd:
                continue

            if model_sd[k].shape != v.shape:
                if args.inflate_mismatched_weights:
                    inflated_v = inflate_checkpoint_tensor(model_sd[k], v)
                    if inflated_v is not None:
                        model_sd[k].copy_(inflated_v)
                        inflated += 1
                        logging.info(
                            "Inflated checkpoint tensor: "
                            f"{k} {tuple(v.shape)} -> {tuple(model_sd[k].shape)}"
                        )
                        continue
                skipped += 1
                logging.info(
                    "Skipped checkpoint key with shape mismatch: "
                    f"{k} {tuple(v.shape)} -> {tuple(model_sd[k].shape)}"
                )
                continue

            model_sd[k].copy_(v)
            loaded += 1

        # Map legacy Sequential Linear keys to the current CatEmbedder projection.

        import re

        pat = re.compile(r"^input_to_seq\.([^.]+)\.(\d+)\.(weight|bias)$")

        for old_k, old_v in fixed_state_dict.items():
            m = pat.match(old_k)
            if not m:
                continue

            modality = m.group(1)
            wb = m.group(3)
            new_k = f"input_to_seq.{modality}.proj.{wb}"

            if new_k not in model_sd:
                skipped += 1
                logging.info("Skipped legacy mapping: %s -> %s", old_k, new_k)
                continue

            if model_sd[new_k].shape != old_v.shape:
                if args.inflate_mismatched_weights:
                    inflated_v = inflate_checkpoint_tensor(model_sd[new_k], old_v)
                    if inflated_v is not None:
                        model_sd[new_k].copy_(inflated_v)
                        inflated += 1
                        legacy_mapped += 1
                        logging.info(
                            "Inflated legacy checkpoint tensor: "
                            f"{old_k} -> {new_k} "
                            f"{tuple(old_v.shape)} -> {tuple(model_sd[new_k].shape)}"
                        )
                        continue
                skipped += 1
                logging.info(
                    f"Skipped: {new_k}, {old_k} "
                    f"{tuple(old_v.shape)} -> {tuple(model_sd[new_k].shape)}"
                )
                continue

            model_sd[new_k].copy_(old_v)
            legacy_mapped += 1

        load_result = model.load_state_dict(model_sd, strict=True)

        if load_result.missing_keys:
            logging.info("Missing checkpoint keys: %s", load_result.missing_keys)
        if load_result.unexpected_keys:
            logging.info("Unexpected checkpoint keys: %s", load_result.unexpected_keys)

        checkpoint_epoch = get_trained_epoch_from_name(
            os.path.basename(args.continue_training)
        )
        if args.reset_epoch_on_continue_training:
            loaded_epoch = None
            current_epoch = 1
            logging.info(
                "Checkpoint epoch %d is used only as a warm start; resetting "
                "the direct multimodal schedule to epoch 1.",
                checkpoint_epoch,
            )
        else:
            loaded_epoch = checkpoint_epoch
            current_epoch = loaded_epoch + 1

        del cp
        del fixed_state_dict
        logging.info(
            f"Continue training weights are loaded from [{args.continue_training}] "
            f"(loaded {loaded} tensors; inflated {inflated} tensors; "
            f"mapped {legacy_mapped} legacy input_to_seq "
            f"Linear params to CatEmbedder.proj; skipped {skipped})"
        )
        torch.cuda.empty_cache()
        gc.collect()

    multimodal_total_epochs = args.epochs
    assert multimodal_total_epochs > 0, "epochs must be > 0"
    logging.info("Direct multimodal epoch budget: %d", multimodal_total_epochs)

    model = setup_distributed_model(args, model)

    # --- Losses and evaluators --------------------------------------------------

    logging.info("Building losses and evaluators")
    loss_weight_dict = parse_task_loss_weights(
        args.task_loss_weights,
        possible_labels,
    )
    logging.info("Task loss weights: %s", loss_weight_dict)

    criterions = {}

    participant_weight_labels = set(args.participant_class_weight_labels)
    unknown_participant_weight_labels = participant_weight_labels - set(cat_labels)
    if unknown_participant_weight_labels:
        raise ValueError(
            "Unknown --participant_class_weight_labels: "
            f"{sorted(unknown_participant_weight_labels)}"
        )

    for l in cat_labels:
        pos_weight = None
        if args.class_weight:
            if args.participant_level_sampling or l in participant_weight_labels:
                weight_value = get_participant_pos_weight(
                    train_dataset,
                    l,
                    scale=args.pos_weight_scale,
                )
            else:
                weight_value = float(train_dataset.get_pos_weights(l)) * float(
                    args.pos_weight_scale
                )
            pos_weight = torch.tensor(weight_value, dtype=torch.float, device=device)
        criterions[l] = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)

    for l in numerical_labels:
        criterions[l] = nn.MSELoss().to(device)

    from deprecated.losses import DeepHitLoss

    criterions_stage2 = deepcopy(criterions)

    evaluators = {}

    participant_metric_diseases = set(args.participant_level_metric_diseases)
    unknown_participant_metric_diseases = participant_metric_diseases - set(diseases)
    if unknown_participant_metric_diseases:
        raise ValueError(
            "Unknown --participant_level_metric_diseases: "
            f"{sorted(unknown_participant_metric_diseases)}"
        )

    for d in diseases:
        evaluators[d] = TemporalDiseaseEvaluator(
            disease=d,
            progression_years=progression_label_years,
            config=TemporalEvalConfig(
                threshold_mode=(
                    "fixed_0p5"
                    if args.classification_threshold_mode == "fixed_0p5"
                    else "youden_j"
                )
            ),
            aggregate_by_group=(
                args.participant_level_metrics or d in participant_metric_diseases
            ),
        )

    for label in direct_cat_labels:
        evaluators[label] = ClassificationEvaluator(
            num_classes=2,
            task="binary",
            threshold_mode=(
                "fixed_0p5"
                if args.classification_threshold_mode == "fixed_0p5"
                else "youden_j"
            ),
        )

    for l in numerical_labels:
        evaluators[l] = MSEEvaluator(std=train_dataset.mean_std_map[l]["std"])

    # --- Direct multimodal training ---------------------------------------------

    previous_saved_path = None
    start_time = time.time()
    logger = GeneralTrainingLogger(inspecting=args.inspecting)
    test_result = {}

    # Keep checkpoint-selection state defined even when early stopping is disabled.
    # Final evaluation can then fail with the intended, actionable message rather
    # than an AttributeError if a short diagnostic run creates no best checkpoint.
    main.best_loss = None
    main.best_loss_buffer = []
    main.best_auroc = None
    main.best_auroc_buffer = []
    main.best_selection_score = None
    main.best_selection_buffer = []
    selection_metric = args.model_selection_metric
    selection_label = {
        "val_loss": "loss",
        "mean_accuracy": "accuracy",
        "mean_auroc": "auroc",
    }[selection_metric]

    logging.info("Starting direct multimodal training")

    current_epoch = 1
    if loaded_epoch is not None:
        current_epoch = loaded_epoch + 1
        logging.info(
            "Direct multimodal start epoch inferred from checkpoint: epoch=%d. "
            "Optimizer/scheduler states are not restored in this script.",
            current_epoch,
        )
    backbone_freeze_patterns = tuple(args.backbone_freeze_patterns)
    backbone_unfreeze_patterns = tuple(args.backbone_unfreeze_patterns)
    if args.freeze_backbone_epochs > 0 and current_epoch <= args.freeze_backbone_epochs:
        matched, changed = set_backbone_requires_grad(
            model,
            requires_grad=False,
            name_patterns=backbone_freeze_patterns,
        )
        if backbone_freeze_patterns and matched == 0:
            raise ValueError(
                "--backbone_freeze_patterns matched no eligible encoder "
                f"parameters: {list(backbone_freeze_patterns)}"
            )
        logging.info(
            "Multimodal backbone frozen for warm-start: matched=%d, changed=%d, "
            "freeze_epochs=%d, patterns=%s",
            matched,
            changed,
            args.freeze_backbone_epochs,
            list(backbone_freeze_patterns) or "ALL",
        )
        backbone_frozen = True
    else:
        if backbone_unfreeze_patterns:
            set_backbone_requires_grad(model, requires_grad=False)
        matched, changed = set_backbone_requires_grad(
            model,
            requires_grad=True,
            exclude_modalities=frozen_encoder_modalities,
            name_patterns=backbone_unfreeze_patterns,
        )
        if backbone_unfreeze_patterns and matched == 0:
            raise ValueError(
                "--backbone_unfreeze_patterns matched no eligible encoder "
                f"parameters: {list(backbone_unfreeze_patterns)}"
            )
        logging.info(
            "Multimodal backbone trainable from start: matched=%d, changed=%d, "
            "patterns=%s",
            matched,
            changed,
            list(backbone_unfreeze_patterns) or "ALL",
        )
        backbone_frozen = False

    permanently_matched, permanently_changed = set_backbone_requires_grad(
        model,
        requires_grad=False,
        modalities=frozen_encoder_modalities,
    )
    if frozen_encoder_modalities:
        logging.info(
            "Permanently frozen component-pretrained encoders: modalities=%s, "
            "matched=%d, changed=%d",
            sorted(frozen_encoder_modalities),
            permanently_matched,
            permanently_changed,
        )

    optimizer_stage2, backbone_names, head_names = build_stage2_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        frozen_backbone_modalities=frozen_encoder_modalities,
        head_backbone_modalities=newly_initialized_encoder_modalities,
        backbone_param_patterns=backbone_freeze_patterns,
        optimizer_type=args.optimizer_type,
    )

    logging.info(
        "Multimodal optimizer groups built: "
        f"backbone_params={len(backbone_names)}, head_params={len(head_names)}, "
        f"backbone_lr={args.backbone_lr}, head_lr={args.head_lr}, "
        f"optimizer={args.optimizer_type}"
    )
    if newly_initialized_encoder_modalities:
        logging.info(
            "Newly initialized input adapters assigned to head LR: %s",
            sorted(newly_initialized_encoder_modalities),
        )
    trainable_numel = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_numel = sum(
        parameter.numel() for parameter in model.parameters() if not parameter.requires_grad
    )
    fusion_transformer_numel = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "transformer." in name
    )
    # Persist the effective post-freeze counts in test_result.csv. The initial
    # counts above are useful while constructing the model, but they precede
    # permanent component freezing and therefore overstate trainable capacity.
    n_trainable_parameters = trainable_numel
    n_total_parameters = trainable_numel + frozen_numel
    logging.info(
        "Effective parameter counts after freezing: trainable=%d, frozen=%d, "
        "fusion_transformer=%d",
        trainable_numel,
        frozen_numel,
        fusion_transformer_numel,
    )

    if args.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    micro_batches_per_epoch = (
        len(train_d)
        if args.num_samples is None
        else math.ceil(
            args.num_samples / int(args.train_batch_size or args.batch_size)
        )
    )
    steps_per_epoch_stage2 = math.ceil(
        micro_batches_per_epoch / args.gradient_accumulation_steps
    )
    lr_scheduler_stage2 = build_stage2_scheduler(
        scheduler_type=args.scheduler_type,
        optimizer=optimizer_stage2,
        steps_per_epoch=steps_per_epoch_stage2,
        start_epoch=current_epoch,
        total_epochs=multimodal_total_epochs,
        warmup_ratio=args.warmup_ratio,
        delayed_backbone_schedule=args.delayed_backbone_scheduler,
        backbone_freeze_epochs=args.freeze_backbone_epochs,
        backbone_warmup_epochs=args.backbone_warmup_epochs,
    )
    if lr_scheduler_stage2 is None:
        logging.info("Multimodal scheduler: none")
    else:
        logging.info(
            "Multimodal scheduler: cosine "
            f"(warmup_ratio={args.warmup_ratio}, start_epoch={current_epoch}, "
            f"delayed_backbone={args.delayed_backbone_scheduler}, "
            f"backbone_warmup_epochs={args.backbone_warmup_epochs})"
        )

    stage2_bad_loss_epochs = 0
    stage2_bad_selection_epochs = 0
    best_fast_panel_auroc = None
    validation_predictions_dir = output_dir / "validation_predictions"
    current_val_predictions_path = validation_predictions_dir / ".current.csv"

    if args.eval_only_checkpoint:
        logging.info(
            "Evaluation-only checkpoint mode: skipping the Stage 2 training loop "
            "regardless of checkpoint epoch metadata."
        )
    for epoch in stage2_epoch_range(
        current_epoch,
        multimodal_total_epochs,
        eval_only_checkpoint=args.eval_only_checkpoint,
    ):
        if backbone_frozen and epoch > args.freeze_backbone_epochs:
            matched, changed = set_backbone_requires_grad(
                model,
                requires_grad=True,
                exclude_modalities=frozen_encoder_modalities,
                name_patterns=backbone_unfreeze_patterns,
            )
            if backbone_unfreeze_patterns and matched == 0:
                raise ValueError(
                    "--backbone_unfreeze_patterns matched no eligible encoder "
                    f"parameters: {list(backbone_unfreeze_patterns)}"
                )
            logging.info(
                "Multimodal backbone unfrozen at epoch=%d: matched=%d, changed=%d, "
                "patterns=%s",
                epoch,
                matched,
                changed,
                list(backbone_unfreeze_patterns) or "ALL",
            )
            backbone_frozen = False

        if hasattr(train_d.sampler, "set_epoch"):
            train_d.sampler.set_epoch(epoch)

        if args.distributed and (
            isinstance(val_d.sampler, MultiLabelImbalancedDatasetSampler)
            or isinstance(val_d.sampler, ImbalancedDatasetSampler)
            or isinstance(val_d.sampler, DistributedSampler)
        ):
            val_d.sampler.set_epoch(epoch)

        logging.info("Stage 2 epoch %d starting", epoch)

        modality_dropout_p = args.modality_dropout_p
        if args.modality_dropout_schedule == "linear":
            if args.modality_dropout_end_epoch <= args.modality_dropout_start_epoch:
                raise ValueError(
                    "Linear modality dropout requires end_epoch > start_epoch"
                )
            progress = (
                (epoch - args.modality_dropout_start_epoch)
                / (
                    args.modality_dropout_end_epoch
                    - args.modality_dropout_start_epoch
                )
            )
            progress = min(1.0, max(0.0, progress))
            modality_dropout_p = (
                args.modality_dropout_start_p
                + progress
                * (
                    args.modality_dropout_p
                    - args.modality_dropout_start_p
                )
            )
        logging.info(
            "Stage 2 epoch %d modality dropout: %.6f (%s)",
            epoch,
            modality_dropout_p,
            args.modality_dropout_schedule,
        )

        train_log = train_one_epoch(
            model=model,
            criterions=criterions_stage2,
            evaluators=deepcopy(evaluators),
            optimizer=optimizer_stage2,
            activations=activations,
            dataloader=train_d,
            device=device,
            epoch=epoch,
            max_norm=args.clip_max_norm,
            loss_weight_dict=loss_weight_dict,
            possible_inputs=train_dataset.possible_inputs,
            possible_labels=train_dataset.possible_labels,
            random_sample=(args.random_sample_start_epoch is not None)
            and args.random_sample_start_epoch <= epoch,
            diseases=diseases,
            direct_labels=direct_labels,
            sample_prob={"fundus_image": args.fundus_sample_prob},
            lr_scheduler=lr_scheduler_stage2,
            progression_label_years=progression_label_years,
            only_fundus_image_prob=0.5,
            only_fundus_image_sample=False,
            modality_dropout_p=modality_dropout_p,
            profile_timing=args.profile_timing,
            profile_timing_sync_cuda=args.profile_timing_sync_cuda,
            use_bfloat16=args.bfloat16,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
        )

        fast_val_out = {}
        val_out = {}
        if args.inspecting and (epoch % args.val_freq == 0):
            fast_val_out = evaluate(
                model=model,
                criterions=criterions,
                activations=activations,
                evaluators=deepcopy(evaluators),
                loss_weight_dict=loss_weight_dict,
                dataloader=val_d,
                device=device,
                header="Validation:",
                eval_inputs=eval_inputs,
                eval_labels=eval_labels,
                epoch=epoch,
                diseases=diseases,
                direct_labels=direct_labels,
                progression_label_years=progression_label_years,
                profile_timing=args.profile_timing,
                profile_timing_sync_cuda=args.profile_timing_sync_cuda,
                use_bfloat16=args.bfloat16,
                auroc_ci_bootstrap=0,
                auroc_ci_seed=args.auroc_ci_seed,
                low_positive_threshold=args.low_positive_threshold,
                save_pred_path=(
                    current_val_predictions_path
                    if (
                        not args.two_tier_validation
                        and (not args.distributed or dist.get_rank() == 0)
                    )
                    else None
                ),
            )

            fast_panel_auroc = get_model_selection_value(
                fast_val_out,
                logger,
                selection_metric,
                min_positive_samples=args.model_selection_min_positives,
                selected_labels=args.model_selection_labels,
            )
            full_due, scheduled_due, candidate_due = full_validation_due(
                epoch,
                args.full_val_freq,
                fast_panel_auroc,
                best_fast_panel_auroc,
                args.full_val_candidate_delta,
                metric=selection_metric,
            )
            if candidate_due:
                best_fast_panel_auroc = fast_panel_auroc
            if args.two_tier_validation:
                run_full_validation = full_due
            else:
                # In the single-tier path, val_d is already the authoritative
                # natural validation loader. Reuse its result instead of
                # evaluating the same records a second time.
                val_out = fast_val_out
                run_full_validation = False
            if run_full_validation:
                val_out = evaluate(
                    model=model,
                    criterions=criterions,
                    activations=activations,
                    evaluators=deepcopy(evaluators),
                    loss_weight_dict=loss_weight_dict,
                    dataloader=full_val_d,
                    device=device,
                    header="Full natural validation:",
                    eval_inputs=eval_inputs,
                    eval_labels=eval_labels,
                    epoch=epoch,
                    diseases=diseases,
                    direct_labels=direct_labels,
                    progression_label_years=progression_label_years,
                    profile_timing=args.profile_timing,
                    profile_timing_sync_cuda=args.profile_timing_sync_cuda,
                    auroc_ci_bootstrap=validation_ci_bootstrap_for_epoch(args, epoch),
                    auroc_ci_seed=args.auroc_ci_seed,
                    low_positive_threshold=args.low_positive_threshold,
                    auroc_ci_workers=args.auroc_ci_workers,
                    use_bfloat16=args.bfloat16,
                    save_pred_path=(
                        current_val_predictions_path
                        if (not args.distributed or dist.get_rank() == 0)
                        else None
                    ),
                )
                logging.info(
                    "Authoritative full validation completed: epoch=%d scheduled=%s candidate=%s",
                    epoch,
                    scheduled_due,
                    candidate_due,
                )

            if val_out and (not args.distributed or dist.get_rank() == 0) and (
                args.save_val_predictions_freq > 0
                and epoch % args.save_val_predictions_freq == 0
            ):
                validation_predictions_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(
                    current_val_predictions_path,
                    validation_predictions_dir / f"epoch_{epoch:04d}.csv",
                )

        train_log["epoch"] = epoch
        train_log["modality_dropout_p"] = modality_dropout_p
        train_log["examples_this_epoch"] = len(train_d.sampler)
        train_log["batches_this_epoch"] = len(train_d)
        optimizer_steps_this_epoch = math.ceil(
            len(train_d) / args.gradient_accumulation_steps
        )
        train_log["optimizer_steps_this_epoch"] = optimizer_steps_this_epoch
        train_log["optimizer_steps_cumulative"] = epoch * optimizer_steps_this_epoch
        if hasattr(train_d.sampler, "summary"):
            sampler_summary = train_d.sampler.summary()
            train_log.update(sampler_summary_log_values(sampler_summary))
        displayed_val_out = val_out or fast_val_out
        if displayed_val_out:
            displayed_val_out["epoch"] = epoch
        logger.update(train_log, displayed_val_out)

        logging.info("Saving Stage 2 epoch %d artifacts", epoch)

        if (not args.distributed) or dist.get_rank() == 0:
            all_plotting_names = list(diseases) + list(direct_labels)
            render_plots = args.plot_freq > 0 and epoch % args.plot_freq == 0
            logger.save_optimized_artifacts(
                output_dir,
                diseases=all_plotting_names,
                render_plots=render_plots,
            )
            if args.legacy_detailed_plots and render_plots:
                logger.save_to_one_figure(
                    output_dir, label_names=all_plotting_names
                )

        saved_path = save_dist_with_time(
            args=args,
            model=model,
            optimizer=optimizer_stage2,
            scheduler=lr_scheduler_stage2,
            epoch=epoch,
        )
        if (not args.distributed or dist.get_rank() == 0) and hasattr(
            train_d.sampler, "state_dict"
        ):
            sampler_state = train_d.sampler.state_dict()
            torch.save(sampler_state, sampler_state_path)
            sampler_summary_path = output_dir / "training_sampler_summary.json"
            sampler_summary_path.write_text(
                json.dumps(train_d.sampler.summary(), indent=2) + "\n"
            )

        if previous_saved_path is not None:
            assert os.path.exists(
                previous_saved_path
            ), "Previous path provided, but not found on the path."
            os.remove(previous_saved_path)

        # Best-checkpoint tracking is independent of whether early stopping is
        # enabled. A fixed-budget run still needs its best validation model.
        if "loss" in val_out:
            current_loss = val_out.get("loss")
            current_val_selection = get_model_selection_value(
                val_out,
                logger,
                selection_metric,
                min_positive_samples=args.model_selection_min_positives,
                selected_labels=args.model_selection_labels,
            )
            stage2_loss_improved = False
            stage2_selection_improved = False

            if current_loss is not None and (
                main.best_loss is None or current_loss <= main.best_loss
            ):
                main.best_loss = current_loss
                stage2_loss_improved = True
                saved_best_loss = save_dist_with_time(
                    args=args,
                    model=model,
                    optimizer=optimizer_stage2,
                    scheduler=lr_scheduler_stage2,
                    epoch=epoch,
                    affix="stage2_best_loss",
                )
                logging.info(
                    f"Best loss model saved: loss={main.best_loss:.6f} -> {saved_best_loss}"
                )
                main.best_loss_buffer.append(saved_best_loss)
                if len(main.best_loss_buffer) > args.best_model_buffer_size:
                    oldest = main.best_loss_buffer.pop(0)
                    if os.path.exists(oldest):
                        os.remove(oldest)

            if selection_metric_improved(
                current_val_selection,
                main.best_selection_score,
                selection_metric,
                args.full_val_candidate_delta,
            ):
                main.best_selection_score = current_val_selection
                main.best_auroc = current_val_selection
                stage2_selection_improved = True
                saved_best_selection = save_dist_with_time(
                    args=args,
                    model=model,
                    optimizer=optimizer_stage2,
                    scheduler=lr_scheduler_stage2,
                    epoch=epoch,
                    affix=f"stage2_best_{selection_label}",
                )
                logging.info(
                    f"Best selection model saved: metric={selection_metric} "
                    f"score={main.best_selection_score:.6f} -> {saved_best_selection}"
                )
                if args.save_best_val_predictions and (
                    not args.distributed or dist.get_rank() == 0
                ):
                    validation_predictions_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(
                        current_val_predictions_path,
                        validation_predictions_dir / f"best_{selection_metric}.csv",
                    )
                main.best_selection_buffer.append(saved_best_selection)
                main.best_auroc_buffer = main.best_selection_buffer
                if len(main.best_selection_buffer) > args.best_model_buffer_size:
                    oldest = main.best_selection_buffer.pop(0)
                    if os.path.exists(oldest):
                        os.remove(oldest)

        previous_saved_path = saved_path

        if args.early_stop and "loss" in val_out:
            stage2_bad_loss_epochs = update_patience_counter(
                current_loss is not None,
                stage2_loss_improved,
                stage2_bad_loss_epochs,
            )
            stage2_bad_selection_epochs = update_patience_counter(
                current_val_selection is not None and np.isfinite(current_val_selection),
                stage2_selection_improved,
                stage2_bad_selection_epochs,
            )

            if current_loss is None:
                logging.info(
                    "Stage 2 loss unavailable for early stopping; loss patience counter unchanged."
                )
            elif stage2_loss_improved:
                logging.info(
                    "Stage 2 loss improved; loss patience reset."
                )
            else:
                logging.info(
                    f"Stage 2 loss did not improve; loss patience={stage2_bad_loss_epochs}/{args.patience}."
                )

            if current_val_selection is None or not np.isfinite(current_val_selection):
                logging.info(
                    f"Stage 2 {selection_metric} unavailable for early stopping; "
                    "selection patience counter unchanged."
                )
            elif stage2_selection_improved:
                logging.info(
                    f"Stage 2 {selection_metric} improved; selection patience reset."
                )
            else:
                logging.info(
                    f"Stage 2 {selection_metric} did not improve; "
                    f"selection patience={stage2_bad_selection_epochs}/{args.patience}."
                )

            if (
                epoch >= args.early_stop_min_epoch
                and args.patience >= 0
                and stage2_bad_selection_epochs >= args.patience
            ):
                logging.info(
                    f"Early stopping triggered because authoritative validation {selection_metric} "
                    f"failed to improve by {args.full_val_candidate_delta:.6f} for "
                    f"{args.patience} full-validation checks after epoch "
                    f"{args.early_stop_min_epoch}."
                )
                break

    if (not args.distributed or dist.get_rank() == 0) and current_val_predictions_path.exists():
        if args.save_last_val_predictions:
            validation_predictions_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(
                current_val_predictions_path,
                validation_predictions_dir / "last.csv",
            )
        current_val_predictions_path.unlink()

    if (not args.distributed) or dist.get_rank() == 0:
        logger.save_optimized_artifacts(
            output_dir,
            diseases=list(diseases) + list(direct_labels),
            render_plots=True,
        )

    if args.skip_final_test:
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        validation_result = {
            "final_test_skipped": True,
            "n_trainable_parameters": n_trainable_parameters,
            "n_total_parameters": n_total_parameters,
            "training_time": total_time_str,
            "model_selection_metric": selection_metric,
            "best_validation_selection_score": main.best_selection_score,
        }
        pd.DataFrame([validation_result]).to_csv(
            os.path.join(output_dir, "validation_run_result.csv"),
            index=False,
        )
        logging.info(
            "Final test skipped by configuration; saved validation-only run result."
        )
        return

    # --- Final test evaluation --------------------------------------------------
    logging.info("Starting final test evaluation")

    test_eval_d = test_d
    if args.leave_one_modality_out:
        if args.eval_only_validation_only:
            raise ValueError(
                "--leave_one_modality_out requires final-test evaluation and cannot "
                "be combined with --eval_only_validation_only."
            )
        logging.info(
            "Caching deterministic test batches once for repeated modality-ablation evaluation"
        )
        cached_test_batches = []
        for batch_index, batch in enumerate(test_d):
            cached_test_batches.append(batch)
            if batch_index % 25 == 0:
                logging.info(
                    "Cached test batch %d of %d",
                    batch_index + 1,
                    len(test_d),
                )
        test_eval_d = cached_test_batches
        logging.info("Cached %d test batches", len(test_eval_d))

    def evaluate_checkpoint_variant(variant_name, checkpoint_path=None):
        if checkpoint_path is None and variant_name != "last":
            logging.info(
                f"No {variant_name} checkpoint found; skipping final evaluation for that variant."
            )
            return

        if checkpoint_path is not None:
            cp = load_checkpoint_from_path(checkpoint_path, device)
            model.load_state_dict(cp["model"])
            logging.info(
                f"Loaded {variant_name} checkpoint for final evaluation: [{checkpoint_path}]"
            )
            del cp
            torch.cuda.empty_cache()
            gc.collect()
        else:
            logging.info("Evaluating the last model currently in memory.")

        validation_for_thresholds = evaluate(
            model=model,
            criterions=criterions,
            activations=activations,
            evaluators=deepcopy(evaluators),
            loss_weight_dict=loss_weight_dict,
            dataloader=full_val_d,
            device=device,
            header="Validation threshold fitting:",
            eval_inputs=eval_inputs,
            eval_labels=eval_labels,
            diseases=diseases,
            direct_labels=direct_labels,
            progression_label_years=progression_label_years,
            auroc_ci_bootstrap=0,
            use_bfloat16=args.bfloat16,
            save_pred_path=(
                output_dir / f"{variant_name}_combined_validation_predictions.csv"
                if args.eval_only_validation_only
                and (not args.distributed or dist.get_rank() == 0)
                else None
            ),
        )
        if args.eval_only_validation_only:
            test_result.update(
                {
                    f"{variant_name}_combined_validation_{key}": value
                    for key, value in validation_for_thresholds.items()
                }
            )
            log_mapping(
                f"Combined validation results ({variant_name})",
                validation_for_thresholds,
            )
        validation_thresholds = {disease: {} for disease in diseases}
        for disease in diseases:
            disease_prefix = f"{disease}_"
            for key, value in validation_for_thresholds.items():
                if key.startswith(disease_prefix) and key.endswith("_threshold"):
                    validation_thresholds[disease][key[len(disease_prefix):]] = float(value)
            for year in progression_label_years:
                validation_thresholds[disease].setdefault(f"{year}_threshold", 0.5)
            for start, end in evaluators[disease].windows:
                validation_thresholds[disease].setdefault(
                    f"{start}_{end}_threshold",
                    0.5,
                )
        for label in direct_cat_labels:
            validation_thresholds[label] = float(
                validation_for_thresholds.get(f"{label}_threshold", 0.5)
            )
        if not args.distributed or dist.get_rank() == 0:
            threshold_path = output_dir / "validation_fitted_thresholds.json"
            threshold_path.write_text(json.dumps(validation_thresholds, indent=2) + "\n")
            logging.info("Saved validation-fitted thresholds: %s", threshold_path)

        if args.eval_only_validation_only:
            return

        variant_test_out = evaluate(
            model=model,
            criterions=criterions,
            activations=activations,
            evaluators=deepcopy(evaluators),
            loss_weight_dict=loss_weight_dict,
            dataloader=test_eval_d,
            device=device,
            eval_inputs=eval_inputs,
            eval_labels=eval_labels,
            diseases=diseases,
            direct_labels=direct_labels,
            progression_label_years=progression_label_years,
            profile_timing=args.profile_timing,
            profile_timing_sync_cuda=args.profile_timing_sync_cuda,
            auroc_ci_bootstrap=test_ci_bootstrap(args),
            auroc_ci_seed=args.auroc_ci_seed,
            low_positive_threshold=args.low_positive_threshold,
            use_bfloat16=args.bfloat16,
            fixed_thresholds=validation_thresholds,
            auroc_ci_workers=args.auroc_ci_workers,
            save_pred_path=(
                output_dir / f"{variant_name}_ukb_stage2_test_predictions.csv"
                if (not args.distributed or dist.get_rank() == 0)
                else None
            ),
        )
        test_result.update(
            {f"{variant_name}_ukb_stage2_{k}": v for k, v in variant_test_out.items()}
        )

        log_mapping(f"Multimodal UKB test results ({variant_name})", variant_test_out)

        if args.leave_one_modality_out:
            logging.info(
                "Starting leave-one-modality-out evaluation for checkpoint variant [%s]",
                variant_name,
            )
            for withheld_modality in eval_inputs:
                retained_modalities = [
                    modality
                    for modality in eval_inputs
                    if modality != withheld_modality
                ]
                if not retained_modalities:
                    raise ValueError(
                        "Leave-one-modality-out evaluation cannot remove the only input modality."
                    )
                logging.info(
                    "Withholding modality [%s]; retained modalities: %s",
                    withheld_modality,
                    retained_modalities,
                )
                withheld_out = evaluate(
                    model=model,
                    criterions=criterions,
                    activations=activations,
                    evaluators=deepcopy(evaluators),
                    loss_weight_dict=loss_weight_dict,
                    dataloader=test_eval_d,
                    device=device,
                    header=f"Without {withheld_modality}:",
                    eval_inputs=retained_modalities,
                    eval_labels=eval_labels,
                    diseases=diseases,
                    direct_labels=direct_labels,
                    progression_label_years=progression_label_years,
                    profile_timing=args.profile_timing,
                    profile_timing_sync_cuda=args.profile_timing_sync_cuda,
                    auroc_ci_bootstrap=0,
                    use_bfloat16=args.bfloat16,
                    fixed_thresholds=validation_thresholds,
                    save_pred_path=(
                        output_dir
                        / f"{variant_name}_without_{withheld_modality}_ukb_stage2_test_predictions.csv"
                        if (not args.distributed or dist.get_rank() == 0)
                        else None
                    ),
                )
                test_result.update(
                    {
                        f"{variant_name}_without_{withheld_modality}_{key}": value
                        for key, value in withheld_out.items()
                    }
                )
            logging.info(
                "Completed leave-one-modality-out evaluation for checkpoint variant [%s]",
                variant_name,
            )

        if args.evaluate_clsa_sex:
            if direct_cat_labels != ["patient_gender"] or diseases or numerical_labels:
                raise ValueError(
                    "--evaluate_clsa_sex currently requires patient_gender as the "
                    "only target and no disease/numerical targets."
                )
            for clsa_name, clsa_qc in (("clsa_qc", True), ("clsa_full", False)):
                clsa_dataset = CLSAGenderDataset(
                    baseline_followup_combined_path=args.clsa_sex_csv_path,
                    image_size=args.image_size,
                    split=None,
                    qc=clsa_qc,
                    enhanced=True,
                    normalise_fundus_image=args.normalise_fundus_image,
                )
                clsa_loader = DataLoader(
                    clsa_dataset,
                    batch_size=args.test_batch_size or args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=args.pin_memory,
                    persistent_workers=args.persistent_workers and args.num_workers > 0,
                    prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
                    collate_fn=list,
                )
                clsa_out = evaluate(
                    model=model,
                    criterions=criterions,
                    activations=activations,
                    evaluators=deepcopy(evaluators),
                    loss_weight_dict=loss_weight_dict,
                    dataloader=clsa_loader,
                    device=device,
                    header=f"External {clsa_name}:",
                    eval_inputs=eval_inputs,
                    eval_labels=eval_labels,
                    diseases=diseases,
                    direct_labels=direct_labels,
                    progression_label_years=progression_label_years,
                    fixed_thresholds=validation_thresholds,
                    use_bfloat16=args.bfloat16,
                    save_pred_path=(
                        output_dir / f"{variant_name}_{clsa_name}_predictions.csv"
                        if (not args.distributed or dist.get_rank() == 0)
                        else None
                    ),
                )
                test_result.update(
                    {
                        f"{variant_name}_{clsa_name}_{key}": value
                        for key, value in clsa_out.items()
                    }
                )
                log_mapping(
                    f"External CLSA results ({variant_name}, {clsa_name})",
                    clsa_out,
                )

    best_selection_model_path = None
    if args.eval_only_checkpoint:
        if not args.continue_training:
            raise ValueError(
                "--eval_only_checkpoint requires --continue_training"
            )
        best_selection_model_path = args.continue_training
    elif main.best_selection_buffer:
        best_selection_model_path = main.best_selection_buffer[-1]
    if best_selection_model_path is None:
        raise RuntimeError(
            "No authoritative validation-selected checkpoint was created; refusing final test evaluation."
        )
    evaluate_checkpoint_variant(f"best_{selection_label}", best_selection_model_path)

    if args.eval_only_validation_only:
        total_time = time.time() - start_time
        test_result["n_trainable_parameters"] = n_trainable_parameters
        test_result["n_total_parameters"] = n_total_parameters
        test_result["evaluation_time"] = str(
            datetime.timedelta(seconds=int(total_time))
        )
        validation_result_path = output_dir / "combined_validation_result.csv"
        pd.DataFrame([test_result]).to_csv(validation_result_path, index=False)
        logging.info("Saved combined validation results: %s", validation_result_path)
        return

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    model.train()
    test_result["n_trainable_parameters"] = n_trainable_parameters
    test_result["n_total_parameters"] = n_total_parameters
    test_result["training_time"] = total_time_str
    training_result_df = pd.DataFrame([test_result])
    training_result_df.to_csv(os.path.join(output_dir, "test_result.csv"))
    logging.info("Saved final results: %s", args.output_dir)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Detection Training Script", parents=[get_args_parser()]
    )
    add_config_arguments(parser)
    args = parse_configured_args(parser)
    main(args)
