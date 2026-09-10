#!/usr/bin/env python3
"""Calculate endpoint-specific Youden thresholds from frozen validation predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions)
    frame = frame.loc[frame["dataset"].eq("ukb")].copy()
    rows: list[dict] = []
    for disease, disease_frame in frame.groupby("disease", sort=True):
        for horizon in (0, 2, 5, 10):
            prediction_column = f"pred_{horizon}y"
            target_column = f"tgt_{horizon}y"
            valid = disease_frame[[prediction_column, target_column]].dropna()
            target = valid[target_column].astype(int).to_numpy()
            prediction = valid[prediction_column].astype(float).to_numpy()
            if np.unique(target).size < 2:
                continue
            false_positive_rate, true_positive_rate, thresholds = roc_curve(
                target, prediction
            )
            finite = np.isfinite(thresholds)
            youden = true_positive_rate - false_positive_rate
            eligible = np.flatnonzero(finite)
            selected = eligible[np.argmax(youden[eligible])]
            rows.append(
                {
                    "disease": disease,
                    "horizon_years": horizon,
                    "youden_threshold": float(thresholds[selected]),
                    "sensitivity": float(true_positive_rate[selected]),
                    "specificity": float(1.0 - false_positive_rate[selected]),
                    "n": int(len(valid)),
                    "positive_n": int(target.sum()),
                }
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "validation_youden_thresholds.csv", index=False)
    lines = [
        "# Validation-fitted Youden thresholds",
        "",
        "Thresholds were fitted separately for each UKB validation disease and cumulative horizon using the clean `fo_imgfrz_c4` best-checkpoint predictions. They are intended for descriptive case classification and must not be refitted on locked test data.",
        "",
        "| Disease | Horizon | Threshold | Sensitivity | Specificity | N | Positives |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['disease'].upper()} | {row['horizon_years']} years | "
            f"{row['youden_threshold']:.4f} | {row['sensitivity']:.3f} | "
            f"{row['specificity']:.3f} | {row['n']} | {row['positive_n']} |"
        )
    (output_dir / "validation_youden_thresholds.md").write_text(
        "\n".join(lines) + "\n"
    )


if __name__ == "__main__":
    main()
