from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import roc_auc_score, roc_curve

from torchmetrics.functional import (
    accuracy,
    auroc,
    average_precision,
    recall,
    f1_score,
    precision,
    confusion_matrix,
)

from utils.misc import all_gather


def generate_adjacent_windows(years: List[int]) -> List[Tuple[int, int]]:
    """
    Generate adjacent windows plus baseline-to-horizon windows.

    Example:
        Input: [0, 2, 5, 10]
        Output: [(0, 2), (2, 5), (5, 10), (0, 5), (0, 10)]
    """
    years = sorted(years)
    if len(years) < 2:
        return []
    baseline_year = years[0]
    adjacent_windows = [(years[i], years[i + 1]) for i in range(len(years) - 1)]
    baseline_windows = [(baseline_year, year) for year in years[2:]]
    return adjacent_windows + baseline_windows


@dataclass
class TemporalEvalConfig:
    """
    Config for temporal evaluation.

    - threshold_mode:
        "fixed_0p5"  : always use 0.5 for accuracy/CM-based metrics
        "youden_j"   : choose threshold via Youden's J (on the evaluated subset)

    - drop_prevalent_in_window_metrics:
        For adjacent windows (start->end), evaluate incident-in-window among those
        not already positive at 'start'. This is your existing behavior.

    - interval_score_mode:
        "end_horizon"      : use P(event by end), preserving existing behavior
        "conditional_risk" : use max(P_end - P_start, 0) / max(1 - P_start, eps)
    """
    threshold_mode: Literal["fixed_0p5", "youden_j"] = "youden_j"
    drop_prevalent_in_window_metrics: bool = True
    interval_score_mode: Literal["end_horizon", "conditional_risk"] = "end_horizon"
    eps: float = 1e-8
    auroc_ci_bootstrap: int = 0
    auroc_ci_seed: int = 42
    low_positive_threshold: int = 50


class TemporalDiseaseEvaluator:
    """
    Evaluator for temporal disease prediction across multiple horizons:
      1) Per-year single-horizon metrics for each y:
         - {disease}_{y}_accuracy/auroc/recall/precision/f1_score
         - {disease}_{y}_tp/tn/fp/fn
         - {disease}_{y}_samples / _positive_samples / _negative_samples
         - {disease}_{y}_threshold

      2) Adjacent-window (start->end) incident metrics:
         - {disease}_{start}_{end}_accuracy/auroc/recall/precision/f1_score
         - {disease}_{start}_{end}_tp/tn/fp/fn
         - {disease}_{start}_{end}_samples / _positive_samples / _negative_samples
         - {disease}_{start}_{end}_threshold

      3) Dataset balance diagnostics per horizon label:
         - {disease}_timebin_{y}_counts
         - {disease}_timebin_{y}_event_counts

    Storage:
      - preds: probability via sigmoid(logit)
      - tgts : float in [0,1] or NaN for missing
    """

    def __init__(
        self,
        disease: str,
        progression_years: List[int],
        *,
        config: TemporalEvalConfig | None = None,
        aggregate_by_group: bool = False,
    ):
        self.disease = disease
        self.progression_years = sorted(progression_years)
        self.year_idx = {y: i for i, y in enumerate(self.progression_years)}
        self.windows = generate_adjacent_windows(self.progression_years)
        self.config = config if config is not None else TemporalEvalConfig()
        self.aggregate_by_group = bool(aggregate_by_group)

        self.preds: List[torch.Tensor] = []  # each: [T] with NaNs
        self.tgts: List[torch.Tensor] = []   # each: [T] with NaNs
        self.group_ids: List[str] = []

    def set_auroc_ci(
        self,
        *,
        n_bootstrap: int = 0,
        seed: int | None = None,
        low_positive_threshold: int | None = None,
    ):
        self.config.auroc_ci_bootstrap = max(0, int(n_bootstrap or 0))
        if seed is not None:
            self.config.auroc_ci_seed = int(seed)
        if low_positive_threshold is not None:
            self.config.low_positive_threshold = int(low_positive_threshold)

    def set_fixed_thresholds(self, thresholds: Dict[str, float] | None):
        """Use thresholds fitted on validation for subsequent computation."""
        self.fixed_thresholds = dict(thresholds or {})
        self.require_fixed_thresholds = thresholds is not None

    def _bootstrap_auroc_ci(
        self,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        *,
        seed_offset: int = 0,
    ) -> Tuple[float, float, float, int]:
        n_bootstrap = int(self.config.auroc_ci_bootstrap)
        if n_bootstrap <= 0:
            return float("nan"), float("nan"), float("nan"), 0

        y_true_np = y_true.detach().cpu().numpy().astype(int).reshape(-1)
        y_pred_np = y_pred.detach().cpu().numpy().astype(float).reshape(-1)
        pos_idx = np.flatnonzero(y_true_np == 1)
        neg_idx = np.flatnonzero(y_true_np == 0)
        if len(pos_idx) == 0 or len(neg_idx) == 0:
            return float("nan"), float("nan"), float("nan"), 0

        rng = np.random.default_rng(int(self.config.auroc_ci_seed) + int(seed_offset))
        aucs = []
        for _ in range(n_bootstrap):
            sample_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
            sample_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
            sample_idx = np.concatenate([sample_pos, sample_neg])
            try:
                auc_val = float(roc_auc_score(y_true_np[sample_idx], y_pred_np[sample_idx]))
            except Exception:
                continue
            if np.isfinite(auc_val):
                aucs.append(auc_val)

        if len(aucs) == 0:
            return float("nan"), float("nan"), float("nan"), 0

        ci_low, ci_high = np.percentile(np.asarray(aucs, dtype=float), [2.5, 97.5])
        ci_width = float(ci_high - ci_low)
        return float(ci_low), float(ci_high), ci_width, len(aucs)

    def _add_auroc_ci_metrics(
        self,
        results: Dict[str, float],
        prefix: str,
        y_true: torch.Tensor,
        y_pred: torch.Tensor,
        *,
        seed_offset: int = 0,
    ) -> None:
        pos = int((y_true == 1).sum().item())
        neg = int((y_true == 0).sum().item())
        ci_low, ci_high, ci_width, bootstrap_used = self._bootstrap_auroc_ci(
            y_true,
            y_pred,
            seed_offset=seed_offset,
        )
        results[f"{prefix}_auroc_ci_low"] = ci_low
        results[f"{prefix}_auroc_ci_high"] = ci_high
        results[f"{prefix}_auroc_ci_width"] = ci_width
        results[f"{prefix}_auroc_ci_bootstrap_samples"] = int(bootstrap_used)
        results[f"{prefix}_low_positive_flag"] = int(
            pos < int(self.config.low_positive_threshold)
        )
        results[f"{prefix}_ci_crosses_0p5_flag"] = int(
            np.isfinite(ci_low) and np.isfinite(ci_high) and ci_low <= 0.5 <= ci_high
        )
        results[f"{prefix}_negative_samples_for_ci"] = int(neg)

    @torch.no_grad()
    def update(
        self,
        outputs: List[Dict[str, torch.Tensor]],
        targets: List[Dict[str, torch.Tensor]],
        group_ids: List[str] | None = None,
    ):
        """
        outputs/targets are per-sample dicts:
          key: "has_{disease}_in_{y}_years"
          output: logit tensor (scalar or [1])
          target: probability tensor (scalar or [1]) or missing/NaN
        """
        T = len(self.progression_years)
        if group_ids is not None and len(group_ids) != len(outputs):
            raise ValueError("group_ids must have one value per output row")
        if self.aggregate_by_group and group_ids is None:
            raise ValueError(
                "aggregate_by_group=True requires patient group IDs on every update."
            )

        for row_index, (o, t) in enumerate(zip(outputs, targets)):
            pred_row = torch.full((T,), float("nan"))
            tgt_row = torch.full((T,), float("nan"))

            for y in self.progression_years:
                key = f"has_{self.disease}_in_{y}_years"
                j = self.year_idx[y]

                # prediction: store probability
                if key in o and o[key] is not None:
                    logit = o[key]
                    if isinstance(logit, torch.Tensor):
                        logit = logit.view(-1)[0]
                        prob = torch.sigmoid(logit).float()
                        if torch.isfinite(prob):
                            pred_row[j] = prob
                    else:
                        v = float(logit)
                        if np.isfinite(v):
                            pred_row[j] = v

                # target: store probability or NaN
                if key in t and t[key] is not None:
                    val = t[key]
                    if isinstance(val, torch.Tensor):
                        val = val.view(-1)[0].float()
                        if torch.isfinite(val):
                            tgt_row[j] = val
                    else:
                        v = float(val)
                        if np.isfinite(v):
                            tgt_row[j] = v

            self.preds.append(pred_row)
            self.tgts.append(tgt_row)
            # Row-level metrics do not need participant identifiers. Combined
            # datasets may legitimately contain rows without patient_eid, so
            # only validate/store group IDs when participant aggregation was
            # explicitly requested for this evaluator.
            if self.aggregate_by_group:
                group_id = group_ids[row_index]
                if group_id is None:
                    raise ValueError("Cannot aggregate predictions with a missing group ID.")
                self.group_ids.append(str(group_id))

    def _evaluation_tensors(self):
        preds = torch.stack(self.preds, dim=0)
        tgts = torch.stack(self.tgts, dim=0)
        if not self.aggregate_by_group:
            return preds, tgts, None
        if len(self.group_ids) != len(self.preds):
            raise ValueError("Stored group IDs do not match stored prediction rows.")

        positions = {}
        for row_index, group_id in enumerate(self.group_ids):
            positions.setdefault(group_id, []).append(row_index)

        grouped_preds = []
        grouped_tgts = []
        grouped_ids = []
        for group_id, row_indices in positions.items():
            row_preds = preds[row_indices]
            row_tgts = tgts[row_indices]
            pred_row = torch.full((len(self.progression_years),), float("nan"))
            tgt_row = torch.full((len(self.progression_years),), float("nan"))
            for horizon_index in range(len(self.progression_years)):
                valid_preds = row_preds[:, horizon_index]
                valid_preds = valid_preds[torch.isfinite(valid_preds)]
                if valid_preds.numel():
                    pred_row[horizon_index] = valid_preds.mean()

                valid_tgts = row_tgts[:, horizon_index]
                valid_tgts = valid_tgts[torch.isfinite(valid_tgts)]
                if valid_tgts.numel():
                    binary_tgts = valid_tgts >= 0.5
                    if torch.unique(binary_tgts).numel() > 1:
                        year = self.progression_years[horizon_index]
                        raise ValueError(
                            f"Conflicting {self.disease} {year}y targets for patient "
                            f"{group_id}."
                        )
                    tgt_row[horizon_index] = valid_tgts[0]
            grouped_preds.append(pred_row)
            grouped_tgts.append(tgt_row)
            grouped_ids.append(group_id)

        return (
            torch.stack(grouped_preds, dim=0),
            torch.stack(grouped_tgts, dim=0),
            grouped_ids,
        )

    def _choose_threshold(
        self,
        y_true_bin: torch.Tensor,
        y_pred: torch.Tensor,
        key: str | None = None,
    ) -> float:
        """
        Choose threshold for discrete metrics (accuracy/CM).
        """
        if key is not None and key in getattr(self, "fixed_thresholds", {}):
            return float(self.fixed_thresholds[key])
        if key is not None and getattr(self, "require_fixed_thresholds", False):
            raise KeyError(
                f"Missing validation-fitted threshold {key!r} for {self.disease}."
            )
        if self.config.threshold_mode == "fixed_0p5":
            return 0.5

        y_true_np = y_true_bin.detach().cpu().numpy().astype(int).flatten()
        y_pred_np = y_pred.detach().cpu().numpy().astype(float).flatten()

        # If only one class present, ROC is undefined -> fallback
        if len(np.unique(y_true_np)) < 2:
            return 0.5

        fpr, tpr, thresholds = roc_curve(y_true_np, y_pred_np)
        j_scores = tpr - fpr
        best_threshold = float(thresholds[int(np.argmax(j_scores))])

        # sometimes sklearn returns inf thresholds; clamp
        if not (0.0 <= best_threshold <= 1.0):
            best_threshold = 0.5
        return best_threshold

    @torch.no_grad()
    def compute(self, *, distributed_gather: bool = False) -> Dict[str, float]:
        """
        Returns a FLAT dict of metrics (plot-friendly keys).
        """
        if not self.preds:
            return {}

        preds, tgts, _ = self._evaluation_tensors()

        # DDP gather (your util gathers python objects)
        if distributed_gather:
            preds_list = preds.detach().cpu().tolist()
            tgts_list  = tgts.detach().cpu().tolist()

            preds_list = all_gather(preds_list)
            tgts_list  = all_gather(tgts_list)

            # FLATTEN across ranks
            if len(preds_list) > 0 and isinstance(preds_list[0], list):
                preds_list = [row for rank_rows in preds_list for row in rank_rows]
            if len(tgts_list) > 0 and isinstance(tgts_list[0], list):
                tgts_list = [row for rank_rows in tgts_list for row in rank_rows]

            preds = torch.tensor(preds_list, dtype=torch.float32)  # [N_total, T]
            tgts  = torch.tensor(tgts_list, dtype=torch.float32)   # [N_total, T]

        results: Dict[str, float] = {}

        # ------------------------------------------------------------
        # (A) Per-year label counts (balance diagnostics)
        # ------------------------------------------------------------
        for y in self.progression_years:
            j = self.year_idx[y]
            t_j = tgts[:, j]
            valid = torch.isfinite(t_j)

            counts = int(valid.sum().item())
            event_counts = int((t_j[valid] >= 0.5).sum().item()) if counts > 0 else 0

            # results[f"{self.disease}_timebin_{y}_counts"] = counts
            # results[f"{self.disease}_timebin_{y}_event_counts"] = event_counts

            results[f"timebin_{y}_counts"] = counts
            results[f"timebin_{y}_event_counts"] = event_counts

        # ------------------------------------------------------------
        # (B) Per-year metrics (ClassificationEvaluator-style)
        # ------------------------------------------------------------
        for y in self.progression_years:
            j = self.year_idx[y]
            p = preds[:, j]
            t = tgts[:, j]

            valid = torch.isfinite(p) & torch.isfinite(t)
            if valid.sum() == 0:
                continue

            y_pred = p[valid].float()
            y_true = (t[valid] >= 0.5).int()
            n_samples = int(y_true.numel())
            if n_samples == 0:
                continue

            thr = self._choose_threshold(y_true, y_pred, f"{y}_threshold")

            cm = confusion_matrix(
                y_pred,
                y_true,
                task="binary",
                num_classes=2,
                threshold=thr,
            ).cpu().numpy()
            tn, fp, fn, tp = cm.ravel()

            # prefix = f"{self.disease}_{y}"
            prefix = f"{y}"


            # AUROC undefined if only one class present
            if len(torch.unique(y_true)) < 2:
                auroc_val = float("nan")
            else:
                auroc_val = float(auroc(y_pred, y_true, task="binary").item())

            results[f"{prefix}_accuracy"] = float(
                accuracy(y_pred, y_true, task="binary", num_classes=2, threshold=thr).item()
            )
            results[f"{prefix}_auroc"] = auroc_val
            results[f"{prefix}_auprc"] = (
                float(average_precision(y_pred, y_true, task="binary").item())
                if int((y_true == 1).sum().item()) > 0
                else float("nan")
            )
            results[f"{prefix}_recall"] = float(
                recall(y_pred, y_true, task="binary", num_classes=2, threshold=thr).item()
            )
            results[f"{prefix}_f1_score"] = float(
                f1_score(y_pred, y_true, task="binary", num_classes=2, threshold=thr).item()
            )
            results[f"{prefix}_precision"] = float(
                precision(y_pred, y_true, task="binary", num_classes=2, threshold=thr).item()
            )

            results[f"{prefix}_tp"] = int(tp)
            results[f"{prefix}_tn"] = int(tn)
            results[f"{prefix}_fp"] = int(fp)
            results[f"{prefix}_fn"] = int(fn)

            results[f"{prefix}_positive_samples"] = int(tp + fn)
            results[f"{prefix}_negative_samples"] = int(tn + fp)
            results[f"{prefix}_samples"] = int(n_samples)
            results[f"{prefix}_threshold"] = float(thr)
            if self.config.auroc_ci_bootstrap > 0:
                self._add_auroc_ci_metrics(
                    results,
                    prefix,
                    y_true,
                    y_pred,
                    seed_offset=(j + 1) * 1009,
                )

        # ------------------------------------------------------------
        # (C) Adjacent-window metrics (incident in window)
        # ------------------------------------------------------------
        for start, end in self.windows:
            s = self.year_idx[start]
            e = self.year_idx[end]

            p_s = preds[:, s]
            p_e = preds[:, e]
            t_s = tgts[:, s]
            t_e = tgts[:, e]

            valid = (
                torch.isfinite(p_s)
                & torch.isfinite(p_e)
                & torch.isfinite(t_s)
                & torch.isfinite(t_e)
            )
            if valid.sum() == 0:
                continue

            p_s = p_s[valid].float()
            p_e = p_e[valid].float()
            t_s = t_s[valid]
            t_e = t_e[valid]

            if self.config.interval_score_mode == "conditional_risk":
                p_delta = torch.clamp(p_e - p_s, min=0.0)
                interval_pred = p_delta / torch.clamp(
                    1.0 - p_s, min=self.config.eps
                )
            else:
                interval_pred = p_e

            onset_before = (t_s >= 0.5)
            onset_in_window = (t_e >= 0.5)

            if self.config.drop_prevalent_in_window_metrics:
                keep = ~onset_before
                if keep.sum() == 0:
                    continue
                y_true = onset_in_window[keep].int()
                y_pred = interval_pred[keep]
            else:
                y_true = onset_in_window.int()
                y_pred = interval_pred

            n_samples = int(y_true.numel())
            if n_samples == 0:
                continue

            thr = self._choose_threshold(
                y_true,
                y_pred,
                f"{start}_{end}_threshold",
            )

            cm = confusion_matrix(
                y_pred,
                y_true,
                task="binary",
                num_classes=2,
                threshold=thr,
            ).cpu().numpy()
            tn, fp, fn, tp = cm.ravel()

            # prefix = f"{self.disease}_{start}_{end}"
            prefix = f"{start}_{end}"


            if len(torch.unique(y_true)) < 2:
                auroc_val = float("nan")
            else:
                auroc_val = float(auroc(y_pred, y_true, task="binary").item())

            results[f"{prefix}_accuracy"] = float(
                accuracy(y_pred, y_true, task="binary", num_classes=2, threshold=thr).item()
            )
            results[f"{prefix}_auroc"] = auroc_val
            results[f"{prefix}_auprc"] = (
                float(average_precision(y_pred, y_true, task="binary").item())
                if int((y_true == 1).sum().item()) > 0
                else float("nan")
            )
            results[f"{prefix}_recall"] = float(
                recall(y_pred, y_true, task="binary", num_classes=2, threshold=thr).item()
            )
            results[f"{prefix}_f1_score"] = float(
                f1_score(y_pred, y_true, task="binary", num_classes=2, threshold=thr).item()
            )
            results[f"{prefix}_precision"] = float(
                precision(y_pred, y_true, task="binary", num_classes=2, threshold=thr).item()
            )

            results[f"{prefix}_tp"] = int(tp)
            results[f"{prefix}_tn"] = int(tn)
            results[f"{prefix}_fp"] = int(fp)
            results[f"{prefix}_fn"] = int(fn)

            results[f"{prefix}_positive_samples"] = int(tp + fn)
            results[f"{prefix}_negative_samples"] = int(tn + fp)
            results[f"{prefix}_samples"] = int(n_samples)
            results[f"{prefix}_threshold"] = float(thr)
            if self.config.auroc_ci_bootstrap > 0:
                self._add_auroc_ci_metrics(
                    results,
                    prefix,
                    y_true,
                    y_pred,
                    seed_offset=(s + 1) * 1009 + (e + 1) * 9176,
                )

        return results

    def raw_dataframe(self) -> pd.DataFrame:
        """Return metric-unit predictions, grouped by participant when enabled."""
        if not self.preds:
            raise ValueError("No stored predictions. Call update() before save_raw_csv().")

        preds, tgts, group_ids = self._evaluation_tensors()
        preds = preds.detach().cpu().numpy()
        tgts = tgts.detach().cpu().numpy()

        df = pd.DataFrame({"index": np.arange(preds.shape[0])})
        if group_ids is not None:
            df["patient_eid"] = group_ids
        for j, y in enumerate(self.progression_years):
            df[f"pred_{y}y"] = preds[:, j]
            df[f"tgt_{y}y"] = tgts[:, j]
        return df

    def raw_row_dataframe(self) -> pd.DataFrame:
        """Return unaggregated rows for auditing participant aggregation."""
        if not self.preds:
            raise ValueError("No stored predictions. Call update() before save_raw_csv().")
        preds = torch.stack(self.preds, dim=0).detach().cpu().numpy()
        tgts = torch.stack(self.tgts, dim=0).detach().cpu().numpy()
        df = pd.DataFrame({"index": np.arange(preds.shape[0])})
        if self.group_ids:
            df["patient_eid"] = self.group_ids
        for j, y in enumerate(self.progression_years):
            df[f"pred_{y}y"] = preds[:, j]
            df[f"tgt_{y}y"] = tgts[:, j]
        return df

    def save_raw_csv(self, filepath: str) -> str:
        """Save raw predictions and targets to CSV."""
        df = self.raw_dataframe()

        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        df.to_csv(abs_path, index=False)
        return abs_path
