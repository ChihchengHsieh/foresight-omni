#!/usr/bin/env python3
"""Summarise and plot cohort structured-feature directionality."""

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


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values.astype(float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_bootstrap: int = 1000):
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[draws].mean(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarise(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = rows.copy()
    rows["baseline_probability"] = sigmoid(rows["baseline_logit"].to_numpy())
    rows["masked_probability"] = sigmoid(rows["masked_logit"].to_numpy())
    rows["signed_probability_change"] = (
        rows["baseline_probability"] - rows["masked_probability"]
    )
    rows["absolute_probability_change"] = rows["signed_probability_change"].abs()

    group_cols = ["disease", "horizon_years", "observed_label", "modality", "feature"]
    output = []
    for group_index, (keys, group) in enumerate(rows.groupby(group_cols, sort=True)):
        signed = group["signed_probability_change"].to_numpy(dtype=float)
        absolute = np.abs(signed)
        signed_low, signed_high = bootstrap_mean_ci(signed, seed + group_index)
        absolute_low, absolute_high = bootstrap_mean_ci(
            absolute, seed + 100000 + group_index
        )
        output.append(
            {
                **dict(zip(group_cols, keys)),
                "n_participants": int(len(group)),
                "mean_signed_probability_change": float(signed.mean()),
                "mean_signed_probability_change_ci_low": float(signed_low),
                "mean_signed_probability_change_ci_high": float(signed_high),
                "mean_absolute_probability_change": float(absolute.mean()),
                "mean_absolute_probability_change_ci_low": float(absolute_low),
                "mean_absolute_probability_change_ci_high": float(absolute_high),
                "proportion_increasing_probability": float((signed > 0).mean()),
                "proportion_decreasing_probability": float((signed < 0).mean()),
                "top10_frequency": float(group["is_top10_structured"].mean()),
            }
        )
    return pd.DataFrame(output)


def wrap_label(value: str, width: int = 26) -> str:
    replacements = {
        "Iop token 1": "IOP",
        "Age token 1": "Age",
        "Gender token 1": "Sex",
        "Selfrated Health": "Self-rated health",
        "Current Employment Paid Or Self Employed": "Paid or self-employed",
        "Bipolar And Major Depression Status": "Bipolar or major depression status",
    }
    value = replacements.get(str(value), str(value))
    value = value.replace("Pc ", "Genetic PC ")
    value = value.replace("Prs ", "PRS ")
    value = value.replace("Portionsday", "portions/day")
    value = value.replace("Unitsweek", "units/week")
    return "\n".join(textwrap.wrap(str(value), width=width))


def selected_panels(summary: pd.DataFrame, endpoints, top_n: int) -> list[pd.DataFrame]:
    positive = summary.loc[summary["observed_label"].eq(1)]
    panels = []
    for disease, year in endpoints:
        current = positive.loc[
            positive["disease"].eq(disease)
            & positive["horizon_years"].eq(year)
        ].nlargest(top_n, ["top10_frequency", "mean_absolute_probability_change"])
        panels.append(current.sort_values("mean_signed_probability_change"))
    return panels


def plot_directional(summary: pd.DataFrame, endpoints, top_n: int, output: Path) -> None:
    panels = selected_panels(summary, endpoints, top_n)
    ncols = 2 if len(endpoints) > 5 else len(endpoints)
    nrows = int(np.ceil(len(endpoints) / ncols))
    figsize = (16, 4.4 * nrows) if ncols == 2 else (18, 5.2)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    axes = np.asarray(axes).reshape(-1)

    limit = max(
        max(abs(frame["mean_signed_probability_change_ci_low"]).max(),
            abs(frame["mean_signed_probability_change_ci_high"]).max())
        for frame in panels
    )
    limit = max(limit * 1.12, 0.005)

    for axis, (disease, year), selected in zip(axes, endpoints, panels):
        values = 100.0 * selected["mean_signed_probability_change"]
        lower = 100.0 * (
            selected["mean_signed_probability_change"]
            - selected["mean_signed_probability_change_ci_low"]
        )
        upper = 100.0 * (
            selected["mean_signed_probability_change_ci_high"]
            - selected["mean_signed_probability_change"]
        )
        colors = ["#b65d1e" if value >= 0 else "#24557a" for value in values]
        for y, value, low, high, color in zip(
            range(len(selected)), values, lower, upper, colors
        ):
            axis.errorbar(
                value,
                y,
                xerr=np.array([[low], [high]]),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=2,
                markersize=4,
            )
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_yticks(range(len(selected)))
        axis.set_yticklabels([wrap_label(value) for value in selected["feature"]])
        axis.set_xlim(-100.0 * limit, 100.0 * limit)
        axis.set_title(f"{disease.upper()}, {year} years", fontsize=11)
        axis.set_xlabel("Mean change in predicted probability\n(percentage points)", fontsize=8)
        axis.grid(axis="x", alpha=0.2)
        axis.tick_params(axis="both", labelsize=8)
    for axis in axes[len(endpoints) :]:
        axis.axis("off")
    fig.suptitle(
        "Direction of structured-feature sensitivity among positive validation cases",
        fontsize=14,
    )
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_note(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Cohort structured-feature directionality",
        "",
        "For each participant, signed probability change is the prediction with all inputs minus the prediction after setting one encoded feature token to zero.",
        "",
        "- Positive values mean the observed feature token increased the model's predicted probability relative to token zeroing.",
        "- Negative values mean the observed feature token decreased the model's predicted probability relative to token zeroing.",
        "- Values do not estimate the effect of increasing the raw feature value and do not imply causality.",
        "- Direction may differ between participants because raw values and interactions with other modalities differ.",
        "- Features are selected by how often they rank among a participant's ten largest structured-feature sensitivities, with mean absolute probability change used to break ties.",
        "- Selected features are displayed using their signed mean and participant-bootstrap 95% interval.",
        "",
        "The complete machine-readable table is `cohort_feature_direction_summary.csv`.",
    ]
    (output_dir / "cohort_feature_directionality.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.participant_csv)
    rows["feature"] = rows["feature"].map(canonical_feature_display_name)
    summary = summarise(rows, args.seed)
    summary.to_csv(output_dir / "cohort_feature_direction_summary.csv", index=False)

    compact_endpoints = [(disease, 10) for disease in ("glaucoma", "ad", "pd", "cvd", "t2d")]
    full_endpoints = [
        ("glaucoma", 0), ("glaucoma", 10),
        ("ad", 5), ("ad", 10),
        ("pd", 0), ("pd", 10),
        ("cvd", 0), ("cvd", 10),
        ("t2d", 0), ("t2d", 10),
    ]
    plot_directional(
        summary,
        compact_endpoints,
        top_n=5,
        output=output_dir / "cohort_feature_directionality_compact.png",
    )
    plot_directional(
        summary,
        full_endpoints,
        top_n=8,
        output=output_dir / "cohort_feature_directionality_full.png",
    )
    write_note(summary, output_dir)


if __name__ == "__main__":
    main()
