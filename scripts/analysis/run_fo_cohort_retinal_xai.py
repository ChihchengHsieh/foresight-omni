#!/usr/bin/env python3
"""Cohort-level fundus localisation with laterality and faithfulness checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.universal_dense_onehead import get_fixed_inputs_and_labels
from scripts.analysis.run_real_validation_xai_demo import (
    build_datasets,
    build_model,
    disease_labels,
)
from train import get_args_parser
from utils.tensor import nested_to_device
from utils.yaml_config import add_config_arguments, parse_configured_args
from xai.faithfulness import attach_deletion_logits
from xai.image_attribution import grad_cam_for_module
from xai.provenance import fundus_spatial_manifest
from xai.raw_attribution import conditional_modality_logits


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
DISEASE_LABELS = {
    "glaucoma": "Glaucoma",
    "ad": "Alzheimer disease",
    "pd": "Parkinson disease",
    "cvd": "Cardiovascular disease",
    "t2d": "Type 2 diabetes",
}


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_int_csv(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def eye_side_from_row(row: pd.Series) -> str | None:
    for column in ("eye_side", "laterality", "eye", "side"):
        if column not in row or pd.isna(row[column]):
            continue
        text = str(row[column]).strip().lower()
        if text in {"0", "left", "l", "21015"}:
            return "left"
        if text in {"1", "right", "r", "21016"}:
            return "right"
    path = str(row.get("image_path", ""))
    tokens = re.split(r"[^a-zA-Z0-9]+", Path(path).name.lower())
    if "21015" in tokens or "left" in tokens:
        return "left"
    if "21016" in tokens or "right" in tokens:
        return "right"
    return None


def pseudonym(value: object) -> str:
    return hashlib.sha256(f"fo-retinal-xai:{value}".encode()).hexdigest()[:12]


def display_image(image: torch.Tensor) -> np.ndarray:
    value = image.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return value.clamp(0, 1).permute(1, 2, 0).numpy()


def orient_array(array: np.ndarray, side: str) -> np.ndarray:
    # Common right-eye view: optic disc should be on the left of the image.
    if side != "left":
        return array.copy()
    axis = 1 if array.ndim == 3 and array.shape[-1] in (1, 3, 4) else -1
    return np.flip(array, axis=axis).copy()


def grid_from_rows(rows: list[dict], grid_size: int, key: str) -> np.ndarray:
    grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    for row in rows:
        grid[int(row["row"]), int(row["column"])] = float(row[key])
    return grid


def region_coverage(mask: np.ndarray, grid_size: int) -> np.ndarray:
    values = np.zeros((grid_size, grid_size), dtype=np.float32)
    height, width = mask.shape
    for row in range(grid_size):
        for column in range(grid_size):
            y0 = round(row * height / grid_size)
            y1 = round((row + 1) * height / grid_size)
            x0 = round(column * width / grid_size)
            x1 = round((column + 1) * width / grid_size)
            values[row, column] = float(mask[y0:y1, x0:x1].mean())
    return values


def bootstrap_grid(values: np.ndarray, replicates: int, seed: int):
    rng = np.random.default_rng(seed)
    estimates = np.empty((replicates,) + values.shape[1:], dtype=np.float32)
    for index in range(replicates):
        sampled = rng.integers(0, values.shape[0], size=values.shape[0])
        estimates[index] = np.nanmean(values[sampled], axis=0)
    return (
        np.nanpercentile(estimates, 2.5, axis=0),
        np.nanpercentile(estimates, 97.5, axis=0),
    )


def bootstrap_scalar(values: np.ndarray, replicates: int, seed: int):
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        estimates[index] = rng.choice(clean, size=clean.size, replace=True).mean()
    return (
        float(clean.mean()),
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def spatial_similarity(
    left: np.ndarray, right: np.ndarray, valid_mask: np.ndarray | None = None
) -> tuple[float, float]:
    if valid_mask is None:
        valid_mask = np.ones_like(left, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    left_flat = left[valid_mask].ravel()
    right_flat = right[valid_mask].ravel()
    if left_flat.size < 3:
        return np.nan, np.nan
    if np.std(left_flat) == 0 or np.std(right_flat) == 0:
        correlation = np.nan
    else:
        correlation = float(np.corrcoef(left_flat, right_flat)[0, 1])
    count = max(1, int(np.ceil(left_flat.size * 0.2)))
    left_top = set(np.argpartition(left_flat, -count)[-count:].tolist())
    right_top = set(np.argpartition(right_flat, -count)[-count:].tolist())
    jaccard = len(left_top & right_top) / len(left_top | right_top)
    return correlation, float(jaccard)


def erode_retinal_mask(mask: np.ndarray, pixels: int) -> torch.Tensor:
    value = torch.from_numpy(mask.astype(np.float32))[None, None]
    if pixels <= 0:
        return value[0, 0].bool()
    kernel = pixels * 2 + 1
    eroded = 1.0 - F.max_pool2d(
        1.0 - value, kernel_size=kernel, stride=1, padding=pixels
    )
    return eroded[0, 0] > 0.5


def retinal_deletion_plan(rows: list[dict], counts, seed: int) -> list[dict]:
    eligible = [
        row
        for row in rows
        if float(row["eroded_retinal_coverage"]) > 0.05
    ]
    absolute_order = [
        row["token_index"]
        for row in sorted(
            eligible,
            key=lambda row: abs(float(row["signed_logit_contribution"])),
            reverse=True,
        )
    ]
    positive_order = [
        row["token_index"]
        for row in sorted(
            eligible,
            key=lambda row: float(row["signed_logit_contribution"]),
            reverse=True,
        )
        if float(row["signed_logit_contribution"]) > 0
    ]
    low_order = list(reversed(absolute_order))
    random_order = list(absolute_order)
    random.Random(seed).shuffle(random_order)
    plan = []
    for strategy, order in (
        ("positive_top", positive_order),
        ("absolute_top", absolute_order),
        ("random", random_order),
        ("low", low_order),
    ):
        for count in sorted({int(value) for value in counts if int(value) > 0}):
            if len(order) < count:
                continue
            plan.append(
                {
                    "strategy": strategy,
                    "count": count,
                    "identities": order[:count],
                }
            )
    return plan


def select_positions(
    dataset, disease: str, horizon: int, max_participants: int, seed: int
):
    label = f"has_{disease}_in_{horizon}_years"
    frame = dataset.df.copy()
    frame["_position"] = np.arange(len(frame))
    frame["_eye_side"] = frame.apply(eye_side_from_row, axis=1)
    frame["_label"] = pd.to_numeric(frame[label], errors="coerce")
    frame = frame[(frame["_label"] == 1) & frame["_eye_side"].notna()].copy()
    participant_column = "patient_eid" if "patient_eid" in frame else None
    if participant_column is None:
        frame["_participant"] = frame["_position"].astype(str)
    else:
        frame["_participant"] = frame[participant_column].astype(str)
    frame = frame.sort_values(["_participant", "_eye_side", "image_path"])
    frame = frame.drop_duplicates(["_participant", "_eye_side"], keep="first")
    participants = np.asarray(sorted(frame["_participant"].unique()), dtype=object)
    if len(participants) > max_participants:
        rng = np.random.default_rng(seed)
        participants = np.sort(
            rng.choice(participants, size=max_participants, replace=False)
        )
    return frame[frame["_participant"].isin(participants)].copy(), int(
        dataset.df[label].notna().sum()
    )


def explain_image(model, dataset, position, disease, horizon, args, device, case_seed):
    item = dataset[int(position)]
    labels = disease_labels(args)
    sample, _ = get_fixed_inputs_and_labels(item, args.input_modalities, labels)
    sample = nested_to_device(sample, device)
    image = sample["fundus_image"]
    horizon_index = args.progression_label_years.index(horizon)
    with torch.no_grad():
        encoded = model.encode_modalities([sample])[0]
        baseline_logit = float(
            model.forward_encoded_modalities([encoded], output_labels=[[disease]])[
                "out"
            ][0][disease][horizon_index]
            .detach()
            .cpu()
        )
    reference = F.avg_pool2d(
        image.unsqueeze(0), kernel_size=31, stride=1, padding=15
    )[0]
    original = display_image(image)
    retinal_mask = original.mean(axis=-1) > args.xai_retinal_mask_threshold
    eroded_mask = erode_retinal_mask(
        retinal_mask, args.xai_retinal_mask_erosion_pixels
    ).to(image.device)
    manifest = fundus_spatial_manifest(
        args.xai_fundus_grid_size * args.xai_fundus_grid_size,
        int(image.shape[-2]),
        int(image.shape[-1]),
    )
    variants = []
    for region in manifest:
        variant = image.clone()
        y0, y1 = int(region["y0"]), int(region["y1"])
        x0, x1 = int(region["x0"]), int(region["x1"])
        local_mask = eroded_mask[y0:y1, x0:x1][None]
        variant[:, y0:y1, x0:x1] = torch.where(
            local_mask,
            reference[:, y0:y1, x0:x1],
            image[:, y0:y1, x0:x1],
        )
        variants.append(variant)
    logits = conditional_modality_logits(
        model,
        encoded,
        "fundus_image",
        variants,
        disease,
        horizon_index,
        batch_size=args.xai_encoder_batch_size,
    )
    rows = []
    for region, logit in zip(manifest, logits):
        record = dict(region)
        record["masked_logit"] = float(logit)
        record["signed_logit_contribution"] = baseline_logit - float(logit)
        y0, y1 = int(region["y0"]), int(region["y1"])
        x0, x1 = int(region["x0"]), int(region["x1"])
        record["retinal_coverage"] = float(
            retinal_mask[y0:y1, x0:x1].mean()
        )
        record["eroded_retinal_coverage"] = float(
            eroded_mask[y0:y1, x0:x1].float().mean().detach().cpu()
        )
        rows.append(record)

    plan = retinal_deletion_plan(rows, args.xai_deletion_counts, case_seed)
    lookup = {int(row["token_index"]): row for row in manifest}
    deletion_variants = []
    for deletion in plan:
        variant = image.clone()
        for token_index in deletion["identities"]:
            region = lookup[int(token_index)]
            y0, y1 = int(region["y0"]), int(region["y1"])
            x0, x1 = int(region["x0"]), int(region["x1"])
            local_mask = eroded_mask[y0:y1, x0:x1][None]
            variant[:, y0:y1, x0:x1] = torch.where(
                local_mask,
                reference[:, y0:y1, x0:x1],
                image[:, y0:y1, x0:x1],
            )
        deletion_variants.append(variant)
    deletion_logits = conditional_modality_logits(
        model,
        encoded,
        "fundus_image",
        deletion_variants,
        disease,
        horizon_index,
        batch_size=args.xai_encoder_batch_size,
    )
    deletion_rows = attach_deletion_logits(plan, deletion_logits, baseline_logit)

    encoder = model.input_to_seq["fundus_image"]
    encoder = encoder[0] if isinstance(encoder, torch.nn.Sequential) else encoder
    grad_cam = grad_cam_for_module(
        model,
        sample,
        disease,
        horizon_index,
        encoder.feature_extractor,
        output_size=image.shape[-2:],
    )
    return {
        "baseline_logit": baseline_logit,
        "original": original.astype(np.float32),
        "gradcam": grad_cam.detach().cpu().numpy().astype(np.float32),
        "signed_grid": grid_from_rows(
            rows, args.xai_fundus_grid_size, "signed_logit_contribution"
        ),
        "coverage_grid": grid_from_rows(
            rows, args.xai_fundus_grid_size, "retinal_coverage"
        ),
        "eroded_coverage_grid": grid_from_rows(
            rows, args.xai_fundus_grid_size, "eroded_retinal_coverage"
        ),
        "deletion": deletion_rows,
    }


def aggregate_participants(records: list[dict], key: str):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["participant"]].append(record[key])
    participants = sorted(grouped)
    return participants, np.stack(
        [np.mean(grouped[participant], axis=0) for participant in participants]
    )


def add_image(axis, image, title, cmap="magma", vmin=None, vmax=None):
    axis.imshow(image, cmap=cmap, interpolation="bilinear", vmin=vmin, vmax=vmax)
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def add_overlay(axis, base, heat, coverage, title, cmap="magma", vmax=None):
    axis.imshow(np.clip(base, 0, 1), interpolation="bilinear")
    if vmax is None:
        vmax = max(float(np.nanmax(heat)), 1e-8)
    if coverage.shape != heat.shape:
        coverage = F.interpolate(
            torch.from_numpy(coverage)[None, None].float(),
            size=heat.shape,
            mode="nearest",
        )[0, 0].numpy()
    relative = np.clip(heat / vmax, 0, 1)
    alpha = np.where(coverage > 0.05, 0.12 + 0.70 * relative, 0.0)
    axis.imshow(
        heat,
        cmap=cmap,
        interpolation="nearest",
        vmin=0,
        vmax=vmax,
        alpha=alpha,
    )
    axis.set_title(title, fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])


def render_atlas(aggregates: dict, diseases, endpoint_horizons, output_path: Path):
    columns = max(len(endpoint_horizons[disease]) for disease in diseases) * 2
    fig, axes = plt.subplots(len(diseases), columns, figsize=(13, 15))
    for disease_index, disease in enumerate(diseases):
        for horizon_index, horizon in enumerate(endpoint_horizons[disease]):
            item = aggregates[(disease, horizon)]
            if item is None:
                for column in (horizon_index * 2, horizon_index * 2 + 1):
                    axes[disease_index, column].text(
                        0.5,
                        0.5,
                        f"{horizon}-year\nnot estimable",
                        ha="center",
                        va="center",
                    )
                    axes[disease_index, column].set_axis_off()
                continue
            add_overlay(
                axes[disease_index, horizon_index * 2],
                item["mean_image"],
                item["mean_gradcam"],
                item["mean_eroded_coverage"],
                f"{horizon}-year Grad-CAM, n={item['n_participants']}",
            )
            add_overlay(
                axes[disease_index, horizon_index * 2 + 1],
                item["mean_image"],
                item["mean_abs_occlusion"],
                item["mean_eroded_coverage"],
                f"{horizon}-year occlusion, max |change|={np.nanmax(item['mean_abs_occlusion']):.3f}",
                cmap="viridis",
            )
        axes[disease_index, 0].set_ylabel(
            DISEASE_LABELS.get(disease, disease), fontsize=10, weight="semibold"
        )
    fig.suptitle(
        "Cohort-level retinal localisation after left/right orientation harmonisation",
        fontsize=14,
        weight="semibold",
    )
    fig.text(
        0.5,
        0.01,
        "Observed-positive UKB validation participants. Maps are overlaid on the cohort-average fundus. Grad-CAM is normalised per image; occlusion retains target-logit magnitude.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.025, 1, 0.975))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def render_laterality(aggregates: dict, diseases, endpoint_horizons, output_dir: Path):
    for disease in diseases:
        horizons = endpoint_horizons[disease]
        fig, axes = plt.subplots(len(horizons), 3, figsize=(10, 6.5))
        if len(horizons) == 1:
            axes = np.asarray([axes])
        for row_index, horizon in enumerate(horizons):
            item = aggregates[(disease, horizon)]
            if item is None:
                for axis in axes[row_index]:
                    axis.text(
                        0.5,
                        0.5,
                        f"{horizon}-year not estimable",
                        ha="center",
                        va="center",
                    )
                    axis.set_axis_off()
                continue
            left = item.get("left_mean_abs_occlusion")
            right = item.get("right_mean_abs_occlusion")
            if left is None or right is None:
                for axis in axes[row_index]:
                    axis.text(0.5, 0.5, "Insufficient laterality data", ha="center")
                    axis.set_axis_off()
                continue
            vmax = max(float(np.nanmax(left)), float(np.nanmax(right)), 1e-8)
            add_overlay(
                axes[row_index, 0],
                item["left_mean_image"],
                left,
                item["left_mean_eroded_coverage"],
                f"{horizon}-year left eye, n={item['n_left']}",
                cmap="viridis",
                vmax=vmax,
            )
            add_overlay(
                axes[row_index, 1],
                item["right_mean_image"],
                right,
                item["right_mean_eroded_coverage"],
                f"{horizon}-year right eye, n={item['n_right']}",
                cmap="viridis",
                vmax=vmax,
            )
            add_image(
                axes[row_index, 2],
                np.abs(left - right),
                f"Absolute difference, r={item['lr_correlation']:.2f}",
                cmap="cividis",
                vmin=0,
            )
        fig.suptitle(
            f"{DISEASE_LABELS.get(disease, disease)}: laterality sensitivity",
            fontsize=13,
            weight="semibold",
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(
            output_dir / f"laterality_{disease}.png", dpi=220, bbox_inches="tight"
        )
        plt.close(fig)


def render_faithfulness(summary: pd.DataFrame, diseases, endpoint_horizons, output_path: Path):
    columns = max(len(endpoint_horizons[disease]) for disease in diseases)
    fig, axes = plt.subplots(len(diseases), columns, figsize=(11, 14), sharex=True)
    if columns == 1:
        axes = axes[:, None]
    for disease_index, disease in enumerate(diseases):
        for horizon_index, horizon in enumerate(endpoint_horizons[disease]):
            axis = axes[disease_index, horizon_index]
            part = summary[
                (summary["disease"] == disease) & (summary["horizon"] == horizon)
            ]
            for strategy, label, color in (
                ("positive_top", "positive-ranked", "#b24c3d"),
                ("random", "random", "#666666"),
                ("low", "low-absolute", "#3f7c99"),
            ):
                rows = part[part["strategy"] == strategy].sort_values("count")
                axis.plot(rows["count"], rows["mean_signed_logit_change"], marker="o", color=color, label=label)
                axis.fill_between(rows["count"], rows["signed_ci_low"], rows["signed_ci_high"], color=color, alpha=0.14)
            axis.axhline(0, color="#999999", linewidth=0.7)
            axis.set_title(f"{DISEASE_LABELS.get(disease, disease)}, {horizon} years", fontsize=9)
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.supxlabel("Number of 7 by 7 retinal regions masked")
    fig.supylabel("Target-logit decrease after masking")
    fig.suptitle("Retinal attribution faithfulness: positive-ranked versus control deletion", fontsize=14, weight="semibold")
    fig.tight_layout(rect=(0.035, 0.035, 1, 0.975))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(args):
    output_dir = Path(args.output_dir)
    # The configured-argument helper writes resolved_config.yaml before main runs.
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    diseases = parse_csv(args.xai_diseases)
    horizons = parse_int_csv(args.xai_horizons)
    endpoint_horizons = {
        disease: tuple(
            args.xai_ad_early_horizon if disease == "ad" and horizon == 0 else horizon
            for horizon in horizons
        )
        for disease in diseases
    }

    labels = disease_labels(args)
    train_dataset, val_dataset, _ = build_datasets(args, labels)
    model = build_model(args, train_dataset, labels, device)

    manifest_rows = []
    region_rows = []
    deletion_rows = []
    aggregate_results = {}
    similarity_rows = []
    saved_arrays = {}

    for disease_index, disease in enumerate(diseases):
        for horizon_index, horizon in enumerate(endpoint_horizons[disease]):
            selected, valid_label_rows = select_positions(
                val_dataset,
                disease,
                horizon,
                args.xai_max_participants,
                args.seed + disease_index * 100 + horizon_index,
            )
            records = []
            for case_index, (_, row) in enumerate(selected.iterrows()):
                position = int(row["_position"])
                side = str(row["_eye_side"])
                participant = str(row["_participant"])
                result = explain_image(
                    model,
                    val_dataset,
                    position,
                    disease,
                    horizon,
                    args,
                    device,
                    args.seed + disease_index * 100000 + horizon_index * 10000 + case_index,
                )
                oriented_gradcam = orient_array(result["gradcam"], side)
                oriented_signed = orient_array(result["signed_grid"], side)
                oriented_coverage = orient_array(result["coverage_grid"], side)
                oriented_eroded_coverage = orient_array(
                    result["eroded_coverage_grid"], side
                )
                oriented_original = orient_array(result["original"], side)
                record = {
                    "participant": participant,
                    "participant_key": pseudonym(participant),
                    "side": side,
                    "gradcam": oriented_gradcam,
                    "original": oriented_original,
                    "signed_grid": oriented_signed,
                    "abs_grid": np.abs(oriented_signed),
                    "coverage_grid": oriented_coverage,
                    "eroded_coverage_grid": oriented_eroded_coverage,
                }
                records.append(record)
                manifest_rows.append(
                    {
                        "disease": disease,
                        "horizon": horizon,
                        "participant_key": record["participant_key"],
                        "eye_side": side,
                        "source_position": position,
                        "baseline_logit": result["baseline_logit"],
                    }
                )
                for grid_row in range(args.xai_fundus_grid_size):
                    for grid_column in range(args.xai_fundus_grid_size):
                        region_rows.append(
                            {
                                "disease": disease,
                                "horizon": horizon,
                                "participant_key": record["participant_key"],
                                "eye_side": side,
                                "row": grid_row,
                                "column": grid_column,
                                "signed_logit_contribution": float(oriented_signed[grid_row, grid_column]),
                                "absolute_logit_contribution": float(abs(oriented_signed[grid_row, grid_column])),
                                "retinal_coverage": float(oriented_coverage[grid_row, grid_column]),
                                "eroded_retinal_coverage": float(oriented_eroded_coverage[grid_row, grid_column]),
                                "retinal_zone": (
                                    "background"
                                    if oriented_coverage[grid_row, grid_column] < 0.1
                                    else "boundary"
                                    if oriented_eroded_coverage[grid_row, grid_column] < 0.5
                                    else "interior"
                                ),
                            }
                        )
                for deletion in result["deletion"]:
                    deletion_rows.append(
                        {
                            "disease": disease,
                            "horizon": horizon,
                            "participant_key": record["participant_key"],
                            "eye_side": side,
                            "strategy": deletion["strategy"],
                            "count": int(deletion["count"]),
                            "signed_logit_change": float(deletion["logit_drop"]),
                            "absolute_logit_change": float(abs(deletion["logit_drop"])),
                        }
                    )

            if not records:
                aggregate_results[(disease, horizon)] = None
                similarity_rows.append(
                    {
                        "disease": disease,
                        "horizon": horizon,
                        "participants": 0,
                        "images": 0,
                        "left_participants": 0,
                        "right_participants": 0,
                        "left_right_spatial_correlation": np.nan,
                        "left_right_top20_jaccard": np.nan,
                        "gradcam_occlusion_spatial_correlation": np.nan,
                        "gradcam_occlusion_top20_jaccard": np.nan,
                        "status": "not estimable: no eye-resolved observed-positive cases",
                    }
                )
                continue
            participants, participant_gradcam = aggregate_participants(records, "gradcam")
            _, participant_image = aggregate_participants(records, "original")
            _, participant_abs = aggregate_participants(records, "abs_grid")
            _, participant_signed = aggregate_participants(records, "signed_grid")
            _, participant_coverage = aggregate_participants(records, "coverage_grid")
            _, participant_eroded_coverage = aggregate_participants(
                records, "eroded_coverage_grid"
            )
            ci_low, ci_high = bootstrap_grid(
                participant_abs,
                args.xai_bootstrap_replicates,
                args.seed + disease_index * 100 + horizon_index,
            )
            item = {
                "n_participants": len(participants),
                "n_images": len(records),
                "valid_label_rows": valid_label_rows,
                "mean_image": participant_image.mean(axis=0),
                "mean_gradcam": participant_gradcam.mean(axis=0),
                "mean_abs_occlusion": participant_abs.mean(axis=0),
                "mean_signed_occlusion": participant_signed.mean(axis=0),
                "mean_coverage": participant_coverage.mean(axis=0),
                "mean_eroded_coverage": participant_eroded_coverage.mean(axis=0),
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
            for side in ("left", "right"):
                side_records = [record for record in records if record["side"] == side]
                item[f"n_{side}"] = len({record["participant"] for record in side_records})
                if side_records:
                    _, side_values = aggregate_participants(side_records, "abs_grid")
                    item[f"{side}_mean_abs_occlusion"] = side_values.mean(axis=0)
                    _, side_images = aggregate_participants(side_records, "original")
                    item[f"{side}_mean_image"] = side_images.mean(axis=0)
                    _, side_coverage = aggregate_participants(
                        side_records, "eroded_coverage_grid"
                    )
                    item[f"{side}_mean_eroded_coverage"] = side_coverage.mean(axis=0)
            if "left_mean_abs_occlusion" in item and "right_mean_abs_occlusion" in item:
                lr_valid = (
                    (item["left_mean_eroded_coverage"] > 0.05)
                    & (item["right_mean_eroded_coverage"] > 0.05)
                )
                correlation, jaccard = spatial_similarity(
                    item["left_mean_abs_occlusion"],
                    item["right_mean_abs_occlusion"],
                    lr_valid,
                )
            else:
                correlation, jaccard = np.nan, np.nan
            item["lr_correlation"] = correlation
            item["lr_top20_jaccard"] = jaccard
            gradcam_grid = F.adaptive_avg_pool2d(
                torch.from_numpy(item["mean_gradcam"])[None, None],
                (args.xai_fundus_grid_size, args.xai_fundus_grid_size),
            )[0, 0].numpy()
            gradcam_occ_correlation, gradcam_occ_jaccard = spatial_similarity(
                gradcam_grid,
                item["mean_abs_occlusion"],
                item["mean_eroded_coverage"] > 0.05,
            )
            item["gradcam_occlusion_correlation"] = gradcam_occ_correlation
            item["gradcam_occlusion_top20_jaccard"] = gradcam_occ_jaccard
            aggregate_results[(disease, horizon)] = item
            similarity_rows.append(
                {
                    "disease": disease,
                    "horizon": horizon,
                    "participants": len(participants),
                    "images": len(records),
                    "left_participants": item["n_left"],
                    "right_participants": item["n_right"],
                    "left_right_spatial_correlation": correlation,
                    "left_right_top20_jaccard": jaccard,
                    "gradcam_occlusion_spatial_correlation": gradcam_occ_correlation,
                    "gradcam_occlusion_top20_jaccard": gradcam_occ_jaccard,
                    "status": "estimated",
                }
            )
            prefix = f"{disease}_{horizon}y"
            for name in (
                "mean_image",
                "mean_gradcam",
                "mean_abs_occlusion",
                "mean_signed_occlusion",
                "mean_coverage",
                "mean_eroded_coverage",
                "ci_low",
                "ci_high",
            ):
                saved_arrays[f"{prefix}_{name}"] = item[name]

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(output_dir / "case_manifest.csv", index=False)
    region_frame = pd.DataFrame(region_rows)
    region_frame.to_csv(
        output_dir / "participant_region_values.csv", index=False
    )
    if not region_frame.empty:
        (
            region_frame.groupby(
                ["disease", "horizon", "retinal_zone"], as_index=False
            )[["signed_logit_contribution", "absolute_logit_contribution"]]
            .mean()
            .to_csv(output_dir / "retinal_zone_summary.csv", index=False)
        )
    pd.DataFrame(similarity_rows).to_csv(
        output_dir / "laterality_and_method_similarity.csv", index=False
    )
    np.savez_compressed(output_dir / "cohort_retinal_maps.npz", **saved_arrays)

    deletion_frame = pd.DataFrame(deletion_rows)
    participant_deletion = (
        deletion_frame.groupby(
            ["disease", "horizon", "participant_key", "strategy", "count"],
            as_index=False,
        )[["signed_logit_change", "absolute_logit_change"]]
        .mean()
    )
    participant_deletion.to_csv(
        output_dir / "faithfulness_participant_values.csv", index=False
    )
    summary_rows = []
    for keys, group in participant_deletion.groupby(
        ["disease", "horizon", "strategy", "count"], sort=True
    ):
        mean, low, high = bootstrap_scalar(
            group["absolute_logit_change"].to_numpy(float),
            args.xai_bootstrap_replicates,
            args.seed + len(summary_rows),
        )
        signed_mean, signed_low, signed_high = bootstrap_scalar(
            group["signed_logit_change"].to_numpy(float),
            args.xai_bootstrap_replicates,
            args.seed + 10000 + len(summary_rows),
        )
        summary_rows.append(
            {
                "disease": keys[0],
                "horizon": keys[1],
                "strategy": keys[2],
                "count": keys[3],
                "participants": group["participant_key"].nunique(),
                "mean_absolute_logit_change": mean,
                "ci_low": low,
                "ci_high": high,
                "mean_signed_logit_change": signed_mean,
                "signed_ci_low": signed_low,
                "signed_ci_high": signed_high,
            }
        )
    faithfulness_summary = pd.DataFrame(summary_rows)
    faithfulness_summary.to_csv(output_dir / "faithfulness_summary.csv", index=False)

    render_atlas(
        aggregate_results,
        diseases,
        endpoint_horizons,
        figure_dir / "cohort_retinal_localisation_atlas.png",
    )
    render_laterality(aggregate_results, diseases, endpoint_horizons, figure_dir)
    render_faithfulness(
        faithfulness_summary,
        diseases,
        endpoint_horizons,
        figure_dir / "retinal_attribution_faithfulness.png",
    )

    run_summary = {
        "checkpoint": args.xai_checkpoint,
        "population": "Observed-positive UKB validation participants",
        "diseases": list(diseases),
        "horizons": list(horizons),
        "endpoint_horizons": {
            disease: list(values) for disease, values in endpoint_horizons.items()
        },
        "max_participants_per_endpoint": args.xai_max_participants,
        "orientation": "Left-eye images and maps flipped horizontally to a common right-eye view",
        "participant_aggregation": "Eye-level values averaged within participant before cohort averaging",
        "bootstrap_replicates": args.xai_bootstrap_replicates,
        "interpretation": "Conditional model sensitivity, not causality or anatomical validation",
    }
    (output_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2) + "\n")
    print(json.dumps(run_summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Foresight Omni cohort retinal XAI",
        parents=[get_args_parser()],
    )
    add_config_arguments(parser)
    parser.add_argument("--xai_checkpoint", required=True)
    parser.add_argument("--xai_diseases", default="glaucoma,ad,pd,cvd,t2d")
    parser.add_argument("--xai_horizons", default="0,10")
    parser.add_argument("--xai_ad_early_horizon", type=int, default=5)
    parser.add_argument("--xai_max_participants", type=int, default=64)
    parser.add_argument("--xai_encoder_batch_size", type=int, default=64)
    parser.add_argument("--xai_fundus_grid_size", type=int, default=7)
    parser.add_argument("--xai_retinal_mask_threshold", type=float, default=0.035)
    parser.add_argument("--xai_retinal_mask_erosion_pixels", type=int, default=6)
    parser.add_argument(
        "--xai_deletion_counts",
        type=lambda value: [int(item) for item in value.split(",")],
        default=[1, 3, 5, 10],
    )
    parser.add_argument("--xai_bootstrap_replicates", type=int, default=1000)
    main(parse_configured_args(parser))
