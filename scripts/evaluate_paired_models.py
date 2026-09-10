#!/usr/bin/env python3
"""Paired participant-cluster bootstrap comparison of two prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


DEFAULT_DISEASES = ("glaucoma", "ad", "pd", "cvd", "t2d")
DEFAULT_HORIZONS = (0, 2, 5, 10)


def canonical(value: object) -> str:
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else str(value)


def prepare(
    path: Path,
    score_name: str,
    diseases: tuple[str, ...],
    horizons: tuple[int, ...],
    dataset: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["dataset"] = frame["dataset"].astype(str).str.strip().str.lower()
    frame = frame.loc[
        frame["dataset"].eq(dataset.lower()) & frame["disease"].isin(diseases)
    ].copy()
    frame["index"] = pd.to_numeric(frame["index"], errors="raise")
    frame["row_key"] = frame["disease"].astype(str) + ":" + frame["index"].map(
        canonical
    )
    if frame["row_key"].duplicated().any():
        raise ValueError(f"Duplicate row keys in {path}")
    frame["participant"] = frame["patient_eid"].map(canonical)
    pred_columns = {f"pred_{h}y": f"{score_name}_{h}y" for h in horizons}
    keep = [
        "row_key",
        "disease",
        "participant",
        *[f"tgt_{h}y" for h in horizons],
        *[f"{score_name}_{h}y" for h in horizons],
    ]
    return frame.rename(columns=pred_columns)[keep].set_index("row_key").sort_index()


def auc_for_weights(
    target: np.ndarray,
    score: np.ndarray,
    cluster_codes: np.ndarray,
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(score, kind="mergesort")
    target = target[order]
    cluster_codes = cluster_codes[order]
    score = score[order]
    starts = np.r_[0, np.flatnonzero(np.diff(score) != 0) + 1]
    lengths = np.diff(np.r_[starts, len(score)])
    weights = counts[:, cluster_codes].astype(np.float64, copy=False)
    positive_weight = weights * target
    negative_weight = weights * (1 - target)
    negative_by_group = np.add.reduceat(negative_weight, starts, axis=1)
    negative_before = np.cumsum(negative_by_group, axis=1) - negative_by_group
    comparison_weight = np.repeat(
        negative_before + 0.5 * negative_by_group, lengths, axis=1
    )
    numerator = np.sum(positive_weight * comparison_weight, axis=1)
    denominator = positive_weight.sum(axis=1) * negative_weight.sum(axis=1)
    valid = denominator > 0
    values = np.full(len(denominator), np.nan, dtype=float)
    values[valid] = numerator[valid] / denominator[valid]
    return values, valid


def paired_bootstrap(
    target: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    cluster_codes: np.ndarray,
    n_clusters: int,
    n_bootstrap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    draws_a: list[float] = []
    draws_b: list[float] = []
    probabilities = np.full(n_clusters, 1.0 / n_clusters)
    attempts = 0
    while len(draws_a) < n_bootstrap and attempts < n_bootstrap * 50:
        current = min(32, n_bootstrap * 50 - attempts)
        counts = rng.multinomial(n_clusters, probabilities, size=current).astype(
            np.int16, copy=False
        )
        auc_a, valid_a = auc_for_weights(target, score_a, cluster_codes, counts)
        auc_b, valid_b = auc_for_weights(target, score_b, cluster_codes, counts)
        valid = valid_a & valid_b
        draws_a.extend(auc_a[valid].tolist())
        draws_b.extend(auc_b[valid].tolist())
        attempts += current
    return np.asarray(draws_a[:n_bootstrap]), np.asarray(draws_b[:n_bootstrap])


def interval(draws: np.ndarray) -> tuple[float, float]:
    if not len(draws):
        return np.nan, np.nan
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-a", type=Path, required=True)
    parser.add_argument("--predictions-b", type=Path, required=True)
    parser.add_argument("--name-a", default="model_a")
    parser.add_argument("--name-b", default="model_b")
    parser.add_argument("--dataset", default="ukb")
    parser.add_argument("--diseases", default=",".join(DEFAULT_DISEASES))
    parser.add_argument("--horizons", default="0,2,5,10")
    parser.add_argument("--n-bootstrap", type=int, default=2000)
    parser.add_argument("--min-positives", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    diseases = tuple(item.strip() for item in args.diseases.split(",") if item.strip())
    horizons = tuple(int(item) for item in args.horizons.split(","))
    first = prepare(args.predictions_a, "a", diseases, horizons, args.dataset)
    second = prepare(args.predictions_b, "b", diseases, horizons, args.dataset)
    if not first.index.equals(second.index):
        raise ValueError("Prediction files do not contain identical endpoint rows")
    merged = first.join(second[[f"b_{h}y" for h in horizons]], how="inner")

    rows: list[dict[str, object]] = []
    for endpoint_index, (disease, horizon) in enumerate(
        (d, h) for d in diseases for h in horizons
    ):
        target_col = f"tgt_{horizon}y"
        work = merged.loc[
            merged["disease"].eq(disease),
            ["participant", target_col, f"a_{horizon}y", f"b_{horizon}y"],
        ].replace([np.inf, -np.inf], np.nan).dropna()
        target = (work[target_col].to_numpy(float) >= 0.5).astype(np.int8)
        score_a = work[f"a_{horizon}y"].to_numpy(float)
        score_b = work[f"b_{horizon}y"].to_numpy(float)
        cluster_codes, participants = pd.factorize(work["participant"], sort=True)
        positives = int(target.sum())
        negatives = int(len(target) - positives)
        if positives and negatives:
            point_a = float(roc_auc_score(target, score_a))
            point_b = float(roc_auc_score(target, score_b))
            draws_a, draws_b = paired_bootstrap(
                target,
                score_a,
                score_b,
                cluster_codes,
                len(participants),
                args.n_bootstrap,
                args.seed + endpoint_index,
            )
        else:
            point_a = point_b = np.nan
            draws_a = draws_b = np.asarray([], dtype=float)
        low_a, high_a = interval(draws_a)
        low_b, high_b = interval(draws_b)
        low_delta, high_delta = interval(draws_b - draws_a)
        rows.append(
            {
                "disease": disease,
                "horizon_years": horizon,
                "participants": len(participants),
                "positive_samples": positives,
                "negative_samples": negatives,
                "supported": positives >= args.min_positives and negatives > 0,
                f"{args.name_a}_auroc": point_a,
                f"{args.name_a}_ci_low": low_a,
                f"{args.name_a}_ci_high": high_a,
                f"{args.name_b}_auroc": point_b,
                f"{args.name_b}_ci_low": low_b,
                f"{args.name_b}_ci_high": high_b,
                f"{args.name_b}_minus_{args.name_a}": point_b - point_a,
                "delta_ci_low": low_delta,
                "delta_ci_high": high_delta,
                "bootstrap_replicates": min(len(draws_a), len(draws_b)),
            }
        )

    result = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
