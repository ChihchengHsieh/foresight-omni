#!/usr/bin/env python3
"""Render publication-oriented cohort structured-feature sensitivity figures."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xai.token_attribution import canonical_feature_display_name


def wrap_label(value: str, width: int = 25) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width))


def draw_endpoint(axis, selected: pd.DataFrame, title: str, top_n: int) -> None:
    selected = selected.nlargest(top_n, "mean_absolute_logit_change")
    selected = selected.sort_values("mean_absolute_logit_change")
    lower = selected["mean_absolute_logit_change"] - selected["mean_absolute_ci_low"]
    upper = selected["mean_absolute_ci_high"] - selected["mean_absolute_logit_change"]
    axis.errorbar(
        selected["mean_absolute_logit_change"],
        [wrap_label(value) for value in selected["feature"]],
        xerr=np.vstack([lower, upper]),
        fmt="o",
        color="#1f4e79",
        ecolor="#8ca8bf",
        capsize=2,
        markersize=4,
    )
    axis.set_title(title, fontsize=11, pad=7)
    axis.grid(axis="x", alpha=0.2)
    axis.tick_params(axis="both", labelsize=8)


def render_compact(positive: pd.DataFrame, output: Path) -> None:
    diseases = ["glaucoma", "ad", "pd", "cvd", "t2d"]
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.9), constrained_layout=True)
    for axis, disease in zip(axes, diseases):
        selected = positive.loc[
            positive["disease"].eq(disease)
            & positive["horizon_years"].eq(10)
        ]
        draw_endpoint(axis, selected, f"{disease.upper()}, 10 years", top_n=5)
        axis.set_xlabel("Mean absolute target-logit\nchange after masking", fontsize=8)
    fig.suptitle(
        "Cohort structured-feature sensitivity among positive validation cases",
        fontsize=14,
    )
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def render_full(positive: pd.DataFrame, output: Path) -> None:
    endpoints = list(
        positive[["disease", "horizon_years"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    fig, axes = plt.subplots(5, 2, figsize=(16, 22), constrained_layout=True)
    for axis, (disease, year) in zip(axes.reshape(-1), endpoints):
        selected = positive.loc[
            positive["disease"].eq(disease)
            & positive["horizon_years"].eq(year)
        ]
        draw_endpoint(axis, selected, f"{disease.upper()}, {year} years", top_n=10)
        axis.set_xlabel("Mean absolute change in target logit after token masking", fontsize=8)
    fig.suptitle(
        "Cohort structured-feature sensitivity among positive validation cases",
        fontsize=15,
    )
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.input_csv)
    summary["feature"] = summary["feature"].map(canonical_feature_display_name)
    positive = summary.loc[summary["observed_label"].eq(1)].copy()
    render_compact(positive, output_dir / "cohort_structured_feature_sensitivity_compact.png")
    render_full(positive, output_dir / "cohort_structured_feature_sensitivity_full.png")


if __name__ == "__main__":
    main()
