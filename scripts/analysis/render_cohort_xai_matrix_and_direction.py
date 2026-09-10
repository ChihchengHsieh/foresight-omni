#!/usr/bin/env python3
"""Render compact cohort-level XAI magnitude and direction figures."""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xai.token_attribution import canonical_feature_display_name


ENDPOINTS = [
    ("glaucoma", 0), ("glaucoma", 10),
    ("ad", 5), ("ad", 10),
    ("pd", 0), ("pd", 10),
    ("cvd", 0), ("cvd", 10),
    ("t2d", 0), ("t2d", 10),
]

DISEASE_LABELS = {
    "glaucoma": "Glaucoma",
    "ad": "Alzheimer disease",
    "pd": "Parkinson disease",
    "cvd": "Cardiovascular disease",
    "t2d": "Type 2 diabetes",
}

PRS_LABELS = {
    "enhanced prs for glaucoma": "Glaucoma PRS",
    "enhanced prs for primary open angle glaucoma (poag)": "Glaucoma PRS",
    "enhanced prs for alzheimers disease (ad)": "Alzheimer disease PRS",
    "enhanced prs for alzheimer's disease (ad)": "Alzheimer disease PRS",
    "enhanced prs for alzheimer disease (ad)": "Alzheimer disease PRS",
    "enhanced prs for parkinsons disease (pd)": "Parkinson disease PRS",
    "enhanced prs for parkinson's disease (pd)": "Parkinson disease PRS",
    "enhanced prs for parkinson disease (pd)": "Parkinson disease PRS",
    "enhanced prs for cardiovascular disease (cvd)": "CVD PRS",
    "enhanced prs for type 2 diabetes (t2d)": "Type 2 diabetes PRS",
}


def display_name(value: str) -> str:
    value = canonical_feature_display_name(str(value))
    lowered = value.lower().strip()
    if lowered in PRS_LABELS:
        return PRS_LABELS[lowered]
    replacements = {
        "Iop token 1": "IOP",
        "Age token 1": "Age",
        "Gender token 1": "Sex",
        "Selfrated Health": "Self-rated health",
        "Current Employment Paid Or Self Employed": "Paid or self-employed",
        "Current Employment Retired": "Retired",
        "Bipolar And Major Depression Status": "Bipolar or major depression",
    }
    value = replacements.get(value, value)
    value = value.replace("Pc ", "Genetic PC ")
    value = value.replace("Prs ", "PRS ")
    return value


def endpoint_label(disease: str, year: int) -> str:
    short = {"glaucoma": "Glaucoma", "ad": "AD", "pd": "PD", "cvd": "CVD", "t2d": "T2D"}[disease]
    return f"{short}\n{year} y"


def select_recurrent_features(positive: pd.DataFrame, n_recurrent: int) -> list[str]:
    ranked = positive.copy()
    ranked["endpoint_rank"] = ranked.groupby(["disease", "horizon_years"])[
        "mean_absolute_logit_change"
    ].rank(method="first", ascending=False)
    top = ranked.loc[ranked["endpoint_rank"].le(10)]
    recurrence = (
        top.groupby("feature", as_index=False)
        .agg(
            endpoint_count=("endpoint_rank", "size"),
            mean_rank=("endpoint_rank", "mean"),
            mean_magnitude=("mean_absolute_logit_change", "mean"),
        )
        .sort_values(
            ["endpoint_count", "mean_rank", "mean_magnitude"],
            ascending=[False, True, False],
        )
    )
    recurrent = recurrence.head(n_recurrent)["feature"].tolist()

    # Keep disease-specific PRS visible when present, even when sensitivity is small.
    prs_features = [
        feature for feature in positive["feature"].drop_duplicates()
        if display_name(feature) in set(PRS_LABELS.values())
    ]
    return recurrent + [feature for feature in prs_features if feature not in recurrent]


def plot_magnitude_heatmap(summary: pd.DataFrame, output: Path, n_recurrent: int) -> pd.DataFrame:
    positive = summary.loc[summary["observed_label"].eq(1)].copy()
    positive = positive.merge(
        pd.DataFrame(ENDPOINTS, columns=["disease", "horizon_years"]),
        on=["disease", "horizon_years"],
        how="inner",
    )
    features = select_recurrent_features(positive, n_recurrent=n_recurrent)

    raw = positive.pivot_table(
        index="feature",
        columns=["disease", "horizon_years"],
        values="mean_absolute_logit_change",
        aggfunc="first",
    ).reindex(index=features, columns=pd.MultiIndex.from_tuples(ENDPOINTS))
    relative = raw.divide(raw.max(axis=0), axis=1) * 100.0

    # Sort recurrent rows by cross-endpoint prominence, then retain PRS as a labelled block.
    prs_mask = [display_name(feature).endswith("PRS") for feature in raw.index]
    recurrent_features = [f for f, is_prs in zip(raw.index, prs_mask) if not is_prs]
    prs_features = [f for f, is_prs in zip(raw.index, prs_mask) if is_prs]
    recurrent_features = sorted(
        recurrent_features,
        key=lambda feature: (-float(relative.loc[feature].mean()), display_name(feature)),
    )
    features = recurrent_features + prs_features
    raw = raw.reindex(features)
    relative = relative.reindex(features)

    fig_height = max(7.6, 0.39 * len(features) + 2.8)
    fig, axis = plt.subplots(figsize=(14.8, fig_height))
    image = axis.imshow(relative.to_numpy(), cmap="Blues", vmin=0, vmax=100, aspect="auto")

    axis.set_xticks(range(len(ENDPOINTS)))
    axis.set_xticklabels([endpoint_label(*endpoint) for endpoint in ENDPOINTS], fontsize=10)
    axis.set_yticks(range(len(features)))
    axis.set_yticklabels([display_name(feature) for feature in features], fontsize=9.5)
    axis.tick_params(length=0)
    axis.set_title(
        "Structured-feature sensitivity among observed-positive validation participants",
        fontsize=15,
        weight="semibold",
        pad=28,
        loc="left",
    )
    axis.text(
        0,
        1.025,
        "Colour is relative to the most sensitive structured feature within each endpoint. "
        "Cell labels are mean absolute target-logit changes.",
        transform=axis.transAxes,
        fontsize=10,
        color="#5d6b78",
        va="bottom",
    )

    for col in range(2, len(ENDPOINTS), 2):
        axis.axvline(col - 0.5, color="white", linewidth=3.0)
    if prs_features and recurrent_features:
        axis.axhline(len(recurrent_features) - 0.5, color="#8f9aa4", linewidth=1.2)
        for tick_label, feature in zip(axis.get_yticklabels(), features):
            if feature in prs_features:
                tick_label.set_color("#73510f")
                tick_label.set_weight("semibold")

    for row in range(raw.shape[0]):
        for col in range(raw.shape[1]):
            value = raw.iat[row, col]
            if not np.isfinite(value):
                continue
            text_color = "white" if relative.iat[row, col] >= 58 else "#263746"
            label = "<0.01" if value < 0.01 else f"{value:.2f}"
            axis.text(col, row, label, ha="center", va="center", fontsize=8.3, color=text_color)

    colorbar = fig.colorbar(image, ax=axis, fraction=0.026, pad=0.018)
    colorbar.set_label("Relative absolute sensitivity within endpoint (%)", fontsize=9)
    colorbar.ax.tick_params(labelsize=8)
    axis.set_xlabel("Disease and cumulative prediction horizon", fontsize=10, labelpad=12)
    for spine in axis.spines.values():
        spine.set_visible(False)
    fig.subplots_adjust(left=0.25, right=0.92, top=0.86, bottom=0.11)
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    exported = raw.copy()
    exported.index = [display_name(feature) for feature in exported.index]
    exported.columns = [f"{disease}_{year}y" for disease, year in ENDPOINTS]
    return exported


def choose_direction_features(frame: pd.DataFrame, top_n: int) -> list[str]:
    score = (
        frame.groupby("feature", as_index=False)
        .agg(
            recurrence=("top10_frequency", "mean"),
            magnitude=("mean_absolute_probability_change", "mean"),
        )
        .sort_values(["recurrence", "magnitude"], ascending=False)
    )
    return score.head(top_n)["feature"].tolist()


def plot_direction(summary: pd.DataFrame, output: Path, top_n: int) -> None:
    positive = summary.loc[summary["observed_label"].eq(1)].copy()
    diseases = ["glaucoma", "ad", "pd", "cvd", "t2d"]
    horizon_map = {"glaucoma": (0, 10), "ad": (5, 10), "pd": (0, 10), "cvd": (0, 10), "t2d": (0, 10)}
    marker_map = {0: "o", 5: "o", 10: "D"}

    global_limit = 0.0
    panels: dict[str, tuple[list[str], pd.DataFrame]] = {}
    for disease in diseases:
        years = horizon_map[disease]
        current = positive.loc[
            positive["disease"].eq(disease)
            & positive["horizon_years"].isin(years)
        ].copy()
        features = choose_direction_features(current, top_n=top_n)
        current = current.loc[current["feature"].isin(features)]
        panels[disease] = (features, current)
        if not current.empty:
            global_limit = max(
                global_limit,
                float(current["mean_signed_probability_change_ci_low"].abs().max()),
                float(current["mean_signed_probability_change_ci_high"].abs().max()),
            )
    global_limit = max(12.0, np.ceil(global_limit * 100.0 / 2.0) * 2.0 + 1.0)

    fig, axes = plt.subplots(3, 2, figsize=(15.2, 11.0), sharex=True)
    axes = axes.reshape(-1)
    for axis, disease in zip(axes, diseases):
        features, current = panels[disease]
        years = horizon_map[disease]
        order = sorted(
            features,
            key=lambda feature: current.loc[current["feature"].eq(feature), "mean_absolute_probability_change"].mean(),
        )
        y_positions = {feature: idx for idx, feature in enumerate(order)}
        offsets = {years[0]: -0.13, years[1]: 0.13}
        for _, row in current.iterrows():
            estimate = 100.0 * float(row["mean_signed_probability_change"])
            low_ci = 100.0 * float(row["mean_signed_probability_change_ci_low"])
            high_ci = 100.0 * float(row["mean_signed_probability_change_ci_high"])
            stable = low_ci > 0 or high_ci < 0
            color = "#b76122" if low_ci > 0 else "#285b7f" if high_ci < 0 else "#8a939b"
            y = y_positions[row["feature"]] + offsets[int(row["horizon_years"])]
            axis.errorbar(
                estimate,
                y,
                xerr=np.array([[estimate - low_ci], [high_ci - estimate]]),
                fmt=marker_map[int(row["horizon_years"])],
                markersize=5.2,
                markerfacecolor=color if stable else "white",
                markeredgecolor=color,
                markeredgewidth=1.1,
                color=color,
                ecolor=color,
                elinewidth=1.25,
                capsize=2.2,
                zorder=3,
            )
        axis.axvline(0, color="#36424c", linewidth=1.0, zorder=1)
        axis.grid(axis="x", color="#dfe5ea", linewidth=0.8, zorder=0)
        axis.set_xlim(-global_limit, global_limit)
        axis.set_yticks(range(len(order)))
        axis.set_yticklabels(["\n".join(textwrap.wrap(display_name(feature), 27)) for feature in order], fontsize=9)
        axis.set_title(DISEASE_LABELS[disease], fontsize=12, weight="semibold", loc="left", pad=7)
        axis.tick_params(axis="x", labelsize=8.5, labelbottom=True)
        axis.tick_params(axis="y", length=0)
        for spine in ("top", "right", "left"):
            axis.spines[spine].set_visible(False)
        axis.spines["bottom"].set_color("#9ba7b0")

    axes[-1].axis("off")
    for axis in axes[:5]:
        axis.set_xlabel("Mean prediction change (percentage points)", fontsize=9)

    fig.suptitle(
        "Direction of recurrent structured-feature sensitivity",
        fontsize=16,
        weight="semibold",
        x=0.08,
        ha="left",
        y=0.982,
    )
    fig.text(
        0.08,
        0.948,
        "Observed-positive validation participants. Estimates compare retaining the observed encoded token with masking it; "
        "they are not raw-feature dose responses.",
        fontsize=10,
        color="#5d6b78",
        ha="left",
    )
    horizon_legend = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor="#36424c", markerfacecolor="white", label="Earlier horizon"),
        Line2D([0], [0], marker="D", color="none", markeredgecolor="#36424c", markerfacecolor="white", label="10-year horizon"),
    ]
    direction_legend = [
        Line2D([0], [0], marker="o", color="#b76122", markerfacecolor="#b76122", label="CI above zero"),
        Line2D([0], [0], marker="o", color="#285b7f", markerfacecolor="#285b7f", label="CI below zero"),
        Line2D([0], [0], marker="o", color="#8a939b", markerfacecolor="white", label="CI includes zero"),
    ]
    axes[-1].text(
        0.0,
        0.88,
        "HORIZON",
        transform=axes[-1].transAxes,
        fontsize=9,
        weight="bold",
        color="#73510f",
    )
    first_legend = axes[-1].legend(
        handles=horizon_legend,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.84),
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        labelspacing=0.9,
        borderaxespad=0,
    )
    axes[-1].add_artist(first_legend)
    axes[-1].text(
        0.0,
        0.58,
        "DIRECTIONAL CERTAINTY",
        transform=axes[-1].transAxes,
        fontsize=9,
        weight="bold",
        color="#73510f",
    )
    axes[-1].legend(
        handles=direction_legend,
        loc="upper left",
        bbox_to_anchor=(0.0, 0.54),
        frameon=False,
        fontsize=10,
        handlelength=1.8,
        labelspacing=0.9,
        borderaxespad=0,
    )
    axes[-1].text(
        0.0,
        0.19,
        "Points show mean signed prediction change.\n"
        "Whiskers are participant-cluster bootstrap 95% CIs.\n"
        "Open grey points indicate uncertain direction.",
        transform=axes[-1].transAxes,
        fontsize=9.5,
        color="#4b5964",
        va="top",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.19, right=0.98, top=0.90, bottom=0.07, hspace=0.42, wspace=0.30)
    fig.savefig(output, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort_summary_csv", required=True)
    parser.add_argument("--direction_summary_csv", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_recurrent", type=int, default=12)
    parser.add_argument("--direction_top_n", type=int, default=6)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort_summary_csv)
    direction = pd.read_csv(args.direction_summary_csv)

    magnitude_table = plot_magnitude_heatmap(
        cohort,
        output_dir / "supplementary_figure_2_cohort_feature_magnitude.png",
        n_recurrent=args.n_recurrent,
    )
    magnitude_table.to_csv(output_dir / "supplementary_figure_2_values.csv")
    plot_direction(
        direction,
        output_dir / "supplementary_figure_3_cohort_feature_direction.png",
        top_n=args.direction_top_n,
    )


if __name__ == "__main__":
    main()
