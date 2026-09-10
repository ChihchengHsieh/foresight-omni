#!/usr/bin/env python3
"""Render disease pages for paired 0-year and 10-year multimodal XAI."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xai.token_attribution import canonical_feature_display_name


DISPLAY_NAMES = {
    "ad": "Alzheimer's disease",
    "pd": "Parkinson's disease",
    "glaucoma": "Glaucoma",
    "cvd": "Cardiovascular disease",
    "t2d": "Type 2 diabetes",
}

TITLE_NAMES = {
    "ad": "AD",
    "pd": "PD",
    "glaucoma": "Glaucoma",
    "cvd": "CVD",
    "t2d": "T2D",
}


def shorten(value: str, width: int = 32) -> str:
    return textwrap.shorten(str(value).replace(" token 1", ""), width=width, placeholder="...")


def crop_fundus_panels(rendered):
    height, width = rendered.shape[:2]
    ranges = [(0.005, 0.230), (0.260, 0.485), (0.765, 0.965)]
    y_starts = (0.080, 0.080, 0.110)
    return [
        rendered[
            int(y_start * height) : int(0.965 * height),
            int(left * width) : int(right * width),
        ]
        for y_start, (left, right) in zip(y_starts, ranges)
    ]


def draw_absolute_fundus_overlay(axis, original_panel, region_csv: Path):
    rows = pd.read_csv(region_csv)
    side = int(max(rows["row"].max(), rows["column"].max())) + 1
    magnitude = np.zeros((side, side), dtype=np.float32)
    for row in rows.itertuples(index=False):
        magnitude[int(row.row), int(row.column)] = abs(
            float(row.signed_logit_contribution)
        )
    height, width = original_panel.shape[:2]
    axis.imshow(original_panel)
    overlay = axis.imshow(
        magnitude,
        cmap="magma",
        alpha=0.52,
        interpolation="nearest",
        extent=(0, width, height, 0),
        vmin=0,
        vmax=max(float(magnitude.max()), 1e-8),
    )
    axis.figure.colorbar(overlay, ax=axis, fraction=0.046, pad=0.02)


def format_probability(value: float) -> str:
    return f"{value:.4f}" if value < 0.01 else f"{value:.3f}"


def classification_label(observed: int, probability: float, threshold: float | None):
    if threshold is None:
        return "threshold not estimable"
    predicted = int(probability >= threshold)
    return {(1, 1): "TP", (1, 0): "FN", (0, 1): "FP", (0, 0): "TN"}[
        (observed, predicted)
    ]


def combined_sensitivity_chart(
    axis,
    modality_labels,
    modality_values,
    feature_labels,
    feature_values,
    fundus_rank,
    modality_count,
    contribution_mode,
):
    labels = [f"Modality: {label}" for label in modality_labels]
    labels += [f"Feature: {label}" for label in feature_labels]
    signed_values = list(modality_values) + list(feature_values)
    values = (
        [abs(value) for value in signed_values]
        if contribution_mode == "absolute"
        else signed_values
    )
    kinds = ["modality"] * len(modality_labels) + ["feature"] * len(feature_labels)
    colors = (
        ["#3E6F8E" if kind == "modality" else "#D29B45" for kind in kinds]
        if contribution_mode == "absolute"
        else ["#C73E3A" if value > 0 else "#3973AC" for value in values]
    )
    bars = axis.barh(range(len(labels)), values, color=colors, alpha=0.88)
    for bar, label, kind in zip(bars, labels, kinds):
        if kind == "feature":
            bar.set_edgecolor("#333333")
            bar.set_linewidth(1.0)
            bar.set_hatch("..")
        if label == "Modality: Fundus image":
            bar.set_edgecolor("#111111")
            bar.set_linewidth(1.8)
            bar.set_hatch("//")
    axis.axvline(0, color="#333333", linewidth=0.8)
    axis.axhline(len(modality_labels) - 0.5, color="#777777", linewidth=0.8)
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
    axis.set_axisbelow(True)
    mode_label = "absolute sensitivities" if contribution_mode == "absolute" else "signed sensitivities"
    axis.set_title(
        "Whole-modality and leading structured-feature "
        f"{mode_label} (fundus modality absolute rank {fundus_rank}/{modality_count})",
        fontsize=12,
    )
    axis.set_xlabel(
        "Absolute target-logit change after withholding or masking"
        if contribution_mode == "absolute"
        else "Change in target logit after withholding a modality or masking one encoded feature token",
        fontsize=10,
    )
    axis.set_yticks(range(len(labels)), labels)
    axis.tick_params(axis="x", labelsize=9)
    axis.tick_params(axis="y", labelsize=8)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.bar_label(
        bars,
        labels=[
            f"{value:.2f}" if contribution_mode == "absolute" else f"{value:+.2f}"
            for value in values
        ],
        padding=2,
        fontsize=9,
    )
    maximum = max(max(abs(value) for value in values), 0.1)
    axis.set_xlim(0, 1.22 * maximum) if contribution_mode == "absolute" else axis.set_xlim(
        -1.30 * maximum, 1.30 * maximum
    )


def render_disease(
    disease: str,
    summaries: list[dict],
    raw_input_dir: Path,
    token_frame: pd.DataFrame,
    modality_frame: pd.DataFrame,
    thresholds: dict[tuple[str, int], float],
    output_path: Path,
    contribution_mode: str,
):
    disease_summaries = sorted(
        [item for item in summaries if item["disease"] == disease],
        key=lambda item: int(item["horizon_years"]),
    )
    if [int(item["horizon_years"]) for item in disease_summaries] != [0, 10]:
        raise RuntimeError(f"Expected paired 0-year and 10-year cases for {disease}")

    fig = plt.figure(figsize=(22, 14), constrained_layout=True)
    outer = fig.add_gridspec(2, 1, hspace=0.08)
    for row_index, summary in enumerate(disease_summaries):
        case = summary["case"]
        horizon = int(summary["horizon_years"])
        observed = int(summary["observed_label"])
        probability = float(summary["baseline_probability"])
        threshold = thresholds.get((disease, horizon))
        classification = classification_label(observed, probability, threshold)
        threshold_text = "NE" if threshold is None else format_probability(threshold)

        section = outer[row_index, 0].subgridspec(
            2, 1, height_ratios=[1.15, 1.35], hspace=0.06
        )
        slug = case.lower().replace(" ", "_")
        rendered = mpimg.imread(raw_input_dir / slug / "fundus_spatial_attribution.png")
        image_grid = section[0, 0].subgridspec(1, 3, wspace=0.04)
        image_titles = (
            "Fundus image",
            "Grad-CAM",
            "Occlusion magnitude"
            if contribution_mode == "absolute"
            else "Signed occlusion",
        )
        image_axes = []
        cropped_panels = crop_fundus_panels(rendered)
        for column, (panel, panel_title) in enumerate(zip(cropped_panels, image_titles)):
            axis = fig.add_subplot(image_grid[0, column])
            if contribution_mode == "absolute" and column == 2:
                draw_absolute_fundus_overlay(
                    axis,
                    cropped_panels[0],
                    raw_input_dir / slug / "fundus_regions.csv",
                )
            else:
                axis.imshow(panel)
            axis.axis("off")
            axis.set_title(panel_title, fontsize=11)
            image_axes.append(axis)
        image_axes[1].text(
            0.5,
            1.13,
            f"{horizon}-year head: observed {observed}, p = {format_probability(probability)}, "
            f"threshold = {threshold_text}, {classification}",
            ha="center",
            transform=image_axes[1].transAxes,
            fontsize=14,
        )

        sensitivity_axis = fig.add_subplot(section[1, 0])
        modalities = modality_frame.loc[modality_frame["case"].eq(case)].copy()
        modalities["absolute"] = modalities["signed_logit_contribution"].abs()
        fundus_rank = int(
            modalities["absolute"].rank(method="min", ascending=False).loc[
                modalities["modality"].eq("fundus_image")
            ].iloc[0]
        )
        modalities = modalities.sort_values(
            "absolute" if contribution_mode == "absolute" else "signed_logit_contribution"
        )
        modality_labels = [
            "Fundus image" if value == "fundus_image" else value.replace("_", " ").title()
            for value in modalities["modality"]
        ]
        features = token_frame.loc[
            token_frame["case"].eq(case)
            & ~token_frame["modality"].eq("fundus_image")
        ].copy()
        features["absolute"] = features["signed_logit_contribution"].abs()
        features = features.nlargest(6, "absolute").sort_values(
            "absolute" if contribution_mode == "absolute" else "signed_logit_contribution"
        )
        feature_labels = [
            f"{shorten(canonical_feature_display_name(item.feature))} "
            f"[raw: {item.raw_value_display}]"
            for item in features.itertuples(index=False)
        ]
        combined_sensitivity_chart(
            sensitivity_axis,
            modality_labels,
            modalities["signed_logit_contribution"].tolist(),
            feature_labels,
            features["signed_logit_contribution"].tolist(),
            fundus_rank,
            len(modalities),
            contribution_mode,
        )

    fig.suptitle(
        f"{TITLE_NAMES[disease]}: same-participant 0-year and 10-year multimodal XAI"
        + (" (absolute contribution)" if contribution_mode == "absolute" else ""),
        fontsize=18,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_input_dir", required=True)
    parser.add_argument("--token_attributions", required=True)
    parser.add_argument("--modality_attributions", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--contribution_mode",
        choices=("signed", "absolute"),
        default="signed",
    )
    args = parser.parse_args()

    raw_input_dir = Path(args.raw_input_dir)
    summaries = json.loads((raw_input_dir / "summaries.json").read_text())
    token_frame = pd.read_csv(args.token_attributions)
    modality_frame = pd.read_csv(args.modality_attributions)
    threshold_frame = pd.read_csv(args.thresholds)
    thresholds = {
        (row.disease, int(row.horizon_years)): float(row.youden_threshold)
        for row in threshold_frame.itertuples(index=False)
    }
    output_dir = Path(args.output_dir)
    for disease in ("ad", "pd", "glaucoma", "cvd", "t2d"):
        render_disease(
            disease,
            summaries,
            raw_input_dir,
            token_frame,
            modality_frame,
            thresholds,
            output_dir / f"supplementary_paired_multimodal_xai_{disease}.png",
            args.contribution_mode,
        )


if __name__ == "__main__":
    main()
