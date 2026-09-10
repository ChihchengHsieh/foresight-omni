#!/usr/bin/env python3
"""Combine fundus maps and structured-feature sensitivity for the same cases."""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xai.token_attribution import canonical_feature_display_name


DISPLAY_NAMES = {
    "glaucoma": "Glaucoma",
    "cvd": "Cardiovascular disease",
    "t2d": "Type 2 diabetes",
}


def shorten(value: str, width: int = 28) -> str:
    return textwrap.shorten(value.replace(" token 1", ""), width=width, placeholder="...")


def render(
    case_ids: list[int],
    summaries: dict[int, dict],
    thresholds: dict[tuple[str, int], float],
    token_frame: pd.DataFrame,
    panel_dir: Path,
    output_path: Path,
    title: str,
) -> None:
    ncols = 2
    nrows = (len(case_ids) + ncols - 1) // ncols
    fig = plt.figure(figsize=(20, 7.0 * nrows), constrained_layout=True)
    outer = fig.add_gridspec(nrows, ncols, hspace=0.12, wspace=0.08)

    for position, case_id in enumerate(case_ids):
        summary = summaries[case_id]
        disease_key = summary["disease"]
        horizon = int(summary["horizon_years"])
        probability = float(summary["baseline_probability"])
        threshold = thresholds[(disease_key, horizon)]
        classification = "TP" if probability >= threshold else "FN"
        disease = DISPLAY_NAMES.get(disease_key, disease_key)

        inner = outer[position // ncols, position % ncols].subgridspec(
            2, 1, height_ratios=[1.15, 1.0], hspace=0.04
        )
        image_axis = fig.add_subplot(inner[0, 0])
        panel = mpimg.imread(
            panel_dir / f"case_{case_id}" / "fundus_spatial_attribution.png"
        )
        image_axis.imshow(panel)
        image_axis.axis("off")
        image_axis.set_title(
            f"{disease}, {horizon}-year horizon, observed positive\n"
            f"p = {probability:.3f}, threshold = {threshold:.3f}, {classification}",
            fontsize=12,
            pad=5,
        )

        feature_axis = fig.add_subplot(inner[1, 0])
        case_rows = token_frame.loc[
            token_frame["case"].eq(f"Case {case_id}")
            & ~token_frame["modality"].eq("fundus_image")
        ].copy()
        case_rows["absolute_contribution"] = case_rows[
            "signed_logit_contribution"
        ].abs()
        case_rows = case_rows.nlargest(6, "absolute_contribution").sort_values(
            "signed_logit_contribution"
        )
        labels = [
            f"{shorten(canonical_feature_display_name(row.feature))} ({row.modality})"
            for row in case_rows.itertuples(index=False)
        ]
        values = case_rows["signed_logit_contribution"].to_numpy()
        colors = ["#C73E3A" if value > 0 else "#3973AC" for value in values]
        bars = feature_axis.barh(labels, values, color=colors, alpha=0.9)
        feature_axis.axvline(0, color="#333333", linewidth=0.8)
        feature_axis.grid(axis="x", color="#D9D9D9", linewidth=0.6, alpha=0.7)
        feature_axis.set_axisbelow(True)
        feature_axis.set_xlabel("Signed target-logit contribution after token masking")
        feature_axis.set_title("Leading non-image feature sensitivities", fontsize=11)
        feature_axis.spines[["top", "right", "left"]].set_visible(False)
        feature_axis.tick_params(axis="y", labelsize=8)
        feature_axis.bar_label(
            bars,
            labels=[f"{value:+.2f}" for value in values],
            padding=3,
            fontsize=8,
        )
        maximum = max(abs(values).max(), 0.1)
        feature_axis.set_xlim(-1.25 * maximum, 1.25 * maximum)

    fig.suptitle(
        f"{title}\nRed supports the target prediction; blue suppresses it",
        fontsize=17,
    )
    fig.text(
        0.5,
        0.002,
        "Structured-feature bars show conditional token-masking effects, not raw-feature dose responses or causality.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_input_dir", required=True)
    parser.add_argument("--panel_dir", required=True)
    parser.add_argument("--token_attributions", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    raw_input_dir = Path(args.raw_input_dir)
    summaries_raw = json.loads((raw_input_dir / "summaries.json").read_text())
    summaries = {
        int(item["case"].replace("Case ", "")): item for item in summaries_raw
    }
    threshold_frame = pd.read_csv(args.thresholds)
    thresholds = {
        (row.disease, int(row.horizon_years)): float(row.youden_threshold)
        for row in threshold_frame.itertuples(index=False)
    }
    token_frame = pd.read_csv(args.token_attributions)
    output_dir = Path(args.output_dir)

    render(
        [25, 27, 38, 45],
        summaries,
        thresholds,
        token_frame,
        Path(args.panel_dir),
        output_dir / "main_multimodal_xai_true_positive_cases.png",
        "Illustrative true-positive cases: fundus and structured-feature sensitivity",
    )
    render(
        [25, 26, 27, 38, 42, 45],
        summaries,
        thresholds,
        token_frame,
        Path(args.panel_dir),
        output_dir / "supplementary_multimodal_xai_selected_cases.png",
        "Selected cases: fundus and structured-feature sensitivity",
    )


if __name__ == "__main__":
    main()
