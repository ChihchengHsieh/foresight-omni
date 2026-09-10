#!/usr/bin/env python3
"""Cohort-level structured-token sensitivity for prespecified validation strata."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.universal_dense_onehead import get_fixed_inputs_and_labels
from scripts.analysis.run_real_validation_xai_demo import (
    build_datasets,
    build_model,
    disease_labels,
    logits_for_variants,
)
from train import get_args_parser
from utils.tensor import nested_to_device
from utils.yaml_config import add_config_arguments, parse_configured_args
from xai.token_attribution import encoded_token_manifest, masked_token_variants


def parse_endpoint_specs(value: str) -> tuple[tuple[str, int], ...]:
    parsed = []
    for raw_item in value.split(","):
        disease, year = (part.strip() for part in raw_item.split(":"))
        parsed.append((disease, int(year)))
    return tuple(parsed)


def participant_column(frame: pd.DataFrame) -> str | None:
    for name in ("patient_eid", "entity_id", "eid", "participant_id"):
        if name in frame.columns:
            return name
    return None


def select_positions(
    frame: pd.DataFrame,
    label: str,
    desired_label: int,
    maximum: int,
    rng: np.random.Generator,
) -> list[int]:
    eligible = frame.loc[
        frame[label].notna()
        & pd.to_numeric(frame[label], errors="coerce").eq(desired_label)
    ].copy()
    participant_col = participant_column(eligible)
    if participant_col is not None:
        eligible = eligible.drop_duplicates(participant_col, keep="first")
    positions = eligible.index.to_numpy(dtype=int)
    if len(positions) > maximum:
        positions = rng.choice(positions, size=maximum, replace=False)
    return sorted(int(position) for position in positions)


def bootstrap_mean_ci(values: np.ndarray, seed: int, draws: int = 1000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def summarise(rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    output = []
    group_cols = ["disease", "horizon_years", "observed_label", "modality", "feature"]
    for group_index, (keys, group) in enumerate(rows.groupby(group_cols, sort=True)):
        values = group["signed_logit_contribution"].to_numpy(dtype=float)
        absolute = np.abs(values)
        ci_low, ci_high = bootstrap_mean_ci(
            absolute,
            seed=seed + group_index,
        )
        signed_ci_low, signed_ci_high = bootstrap_mean_ci(
            values,
            seed=seed + 100_000 + group_index,
        )
        output.append(
            {
                **dict(zip(group_cols, keys)),
                "n_participants": int(len(group)),
                "mean_signed_logit_change": float(values.mean()),
                "mean_absolute_logit_change": float(absolute.mean()),
                "median_absolute_logit_change": float(np.median(absolute)),
                "mean_absolute_ci_low": ci_low,
                "mean_absolute_ci_high": ci_high,
                "mean_signed_ci_low": signed_ci_low,
                "mean_signed_ci_high": signed_ci_high,
                "positive_direction_fraction": float((values > 0).mean()),
                "negative_direction_fraction": float((values < 0).mean()),
                "top10_frequency": float(group["is_top10_structured"].mean()),
            }
        )
    return pd.DataFrame(output)


def plot_positive_summary(summary: pd.DataFrame, output_path: Path):
    positive = summary.loc[summary["observed_label"].eq(1)].copy()
    endpoints = list(
        positive[["disease", "horizon_years"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    ncols = 2
    nrows = int(np.ceil(len(endpoints) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.2 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for axis, (disease, year) in zip(axes, endpoints):
        selected = positive.loc[
            positive["disease"].eq(disease)
            & positive["horizon_years"].eq(year)
        ].nlargest(10, "mean_absolute_logit_change")
        selected = selected.sort_values("mean_absolute_logit_change")
        lower = selected["mean_absolute_logit_change"] - selected["mean_absolute_ci_low"]
        upper = selected["mean_absolute_ci_high"] - selected["mean_absolute_logit_change"]
        axis.errorbar(
            selected["mean_absolute_logit_change"],
            selected["feature"],
            xerr=np.vstack([lower, upper]),
            fmt="o",
            color="#1f4e79",
            ecolor="#8ca8bf",
            capsize=2,
        )
        axis.set_title(f"{disease.upper()} at {year} years")
        axis.set_xlabel("Mean absolute change in target logit after token masking")
        axis.grid(axis="x", alpha=0.2)
    for axis in axes[len(endpoints) :]:
        axis.axis("off")
    fig.suptitle(
        "Validation-cohort structured-feature sensitivity among positive cases",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_positive_direction_summary(summary: pd.DataFrame, output_path: Path):
    positive = summary.loc[summary["observed_label"].eq(1)].copy()
    endpoints = list(
        positive[["disease", "horizon_years"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    ncols = 2
    nrows = int(np.ceil(len(endpoints) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes = np.asarray(axes).reshape(-1)
    for axis, (disease, year) in zip(axes, endpoints):
        endpoint = positive.loc[
            positive["disease"].eq(disease)
            & positive["horizon_years"].eq(year)
        ]
        selected = endpoint.nlargest(10, ["top10_frequency", "mean_absolute_logit_change"])
        selected = selected.sort_values("mean_signed_logit_change")
        values = selected["mean_signed_logit_change"].to_numpy()
        lower = values - selected["mean_signed_ci_low"].to_numpy()
        upper = selected["mean_signed_ci_high"].to_numpy() - values
        colours = np.where(values >= 0, "#c84e4b", "#4f86b8")
        axis.barh(selected["feature"], values, color=colours, alpha=0.9)
        axis.errorbar(
            values,
            selected["feature"],
            xerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#333333",
            capsize=2,
            linewidth=0.8,
        )
        axis.axvline(0, color="#333333", linewidth=0.8)
        axis.set_title(f"{disease.upper()} at {year} years")
        axis.set_xlabel("Mean signed change in target logit after token masking")
        axis.grid(axis="x", alpha=0.2)
    for axis in axes[len(endpoints) :]:
        axis.axis("off")
    fig.suptitle(
        "Directions of recurrent structured-feature sensitivities among positive cases",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: pd.DataFrame, counts: list[dict], output_dir: Path, args):
    lines = [
        "# Cohort-level structured-token sensitivity",
        "",
        "Validation participants were sampled within prespecified disease, horizon and outcome strata without using predictions or explanation appearance.",
        "",
        f"- Checkpoint: `{args.xai_checkpoint}`",
        f"- Maximum participants per outcome stratum: {args.xai_max_participants_per_stratum}",
        "- Perturbation: mask one encoded structured-feature token while holding all other available inputs fixed",
        "- Summary: participant-level mean absolute and signed target-logit change with 1,000 bootstrap confidence intervals",
        "- Interpretation: conditional model sensitivity, not causality or standalone feature importance",
        "",
        "![Positive-case structured-feature sensitivity](positive_structured_feature_sensitivity.png)",
        "",
        "![Directions of recurrent positive-case sensitivities](positive_structured_feature_directions.png)",
        "",
        "## Sample counts",
        "",
        "| Disease | Horizon | Label | Participants |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in counts:
        lines.append(
            f"| {row['disease']} | {row['horizon_years']} | {row['observed_label']} | {row['n_participants']} |"
        )
    lines.extend(["", "## Leading positive-case sensitivities", ""])
    positive = summary.loc[summary["observed_label"].eq(1)]
    for (disease, year), group in positive.groupby(["disease", "horizon_years"]):
        lines.extend(
            [
                f"### {disease.upper()}, {year} years",
                "",
                "| Feature | Modality | N | Mean absolute logit change (95% CI) | Mean signed change (95% CI) | Positive direction | Negative direction | Top-10 frequency |",
                "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for _, row in group.nlargest(10, "mean_absolute_logit_change").iterrows():
            lines.append(
                f"| {row['feature']} | {row['modality']} | {int(row['n_participants'])} | "
                f"{row['mean_absolute_logit_change']:.4f} ({row['mean_absolute_ci_low']:.4f} to {row['mean_absolute_ci_high']:.4f}) | "
                f"{row['mean_signed_logit_change']:+.4f} ({row['mean_signed_ci_low']:+.4f} to {row['mean_signed_ci_high']:+.4f}) | "
                f"{row['positive_direction_fraction']:.1%} | {row['negative_direction_fraction']:.1%} | "
                f"{row['top10_frequency']:.1%} |"
            )
        lines.append("")
    (output_dir / "cohort_structured_token_xai.md").write_text("\n".join(lines))


def main(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    labels = disease_labels(args)
    train_dataset, val_dataset, _ = build_datasets(args, labels)
    model = build_model(args, train_dataset, labels, device)
    frame = val_dataset.df.reset_index(drop=True)

    rows = []
    counts = []
    case_number = 0
    for disease, year in parse_endpoint_specs(args.xai_endpoint_specs):
        label_name = f"has_{disease}_in_{year}_years"
        horizon_index = args.progression_label_years.index(year)
        for observed_label in (0, 1):
            positions = select_positions(
                frame,
                label_name,
                observed_label,
                args.xai_max_participants_per_stratum,
                rng,
            )
            counts.append(
                {
                    "disease": disease,
                    "horizon_years": year,
                    "observed_label": observed_label,
                    "n_participants": len(positions),
                }
            )
            for position in positions:
                case_number += 1
                item = val_dataset[position]
                sample, _ = get_fixed_inputs_and_labels(item, args.input_modalities, labels)
                sample = nested_to_device(sample, device)
                encoded = model.encode_modalities([sample])[0]
                manifest = [
                    entry
                    for entry in encoded_token_manifest(encoded, dataset=val_dataset)
                    if entry["modality"] != "fundus_image"
                ]
                variants = masked_token_variants(encoded, manifest)
                baseline_logit = logits_for_variants(
                    model,
                    [encoded],
                    disease,
                    horizon_index,
                    args.xai_mask_batch_size,
                )[0]
                masked_logits = logits_for_variants(
                    model,
                    variants,
                    disease,
                    horizon_index,
                    args.xai_mask_batch_size,
                )
                case_rows = []
                for entry, masked_logit in zip(manifest, masked_logits):
                    case_rows.append(
                        {
                            "case_number": case_number,
                            "disease": disease,
                            "horizon_years": year,
                            "observed_label": observed_label,
                            "modality": entry["modality"],
                            "feature": entry["feature"],
                            "baseline_logit": baseline_logit,
                            "masked_logit": masked_logit,
                            "signed_logit_contribution": baseline_logit - masked_logit,
                        }
                    )
                ranked = sorted(
                    range(len(case_rows)),
                    key=lambda index: abs(case_rows[index]["signed_logit_contribution"]),
                    reverse=True,
                )
                top = set(ranked[:10])
                for index, row in enumerate(case_rows):
                    row["is_top10_structured"] = index in top
                rows.extend(case_rows)

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("No validation strata were available for structured XAI")
    summary = summarise(detail, args.seed)
    detail.to_csv(output_dir / "participant_feature_sensitivity.csv", index=False)
    summary.to_csv(output_dir / "cohort_feature_summary.csv", index=False)
    (output_dir / "stratum_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    plot_positive_summary(summary, output_dir / "positive_structured_feature_sensitivity.png")
    plot_positive_direction_summary(
        summary,
        output_dir / "positive_structured_feature_directions.png",
    )
    write_report(summary, counts, output_dir, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Cohort structured-token XAI",
        parents=[get_args_parser()],
    )
    add_config_arguments(parser)
    parser.add_argument("--xai_checkpoint", required=True)
    parser.add_argument("--xai_mask_batch_size", type=int, default=128)
    parser.add_argument("--xai_max_participants_per_stratum", type=int, default=24)
    parser.add_argument(
        "--xai_endpoint_specs",
        default="glaucoma:0,glaucoma:10,ad:5,ad:10,pd:0,pd:10,cvd:0,cvd:10,t2d:0,t2d:10",
    )
    main(parse_configured_args(parser))
