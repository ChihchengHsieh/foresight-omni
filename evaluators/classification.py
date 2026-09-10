import torch
import torch.nn.functional as F
import os
from typing import Literal, Tuple, List
from utils.misc import all_gather
from itertools import chain
from torchmetrics.functional import (
    accuracy,
    auroc,
    recall,
    f1_score,
    precision,
    confusion_matrix,
)
from collections import defaultdict
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve


class ClassificationEvaluatorDeprecated:
    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"] = "binary",
        num_classes: int = None,
        num_labels: int = None,
    ) -> None:
        self.preds = []
        self.tgts = []
        self.task = task
        self.num_classes = num_classes
        self.num_labels = num_labels

    def top_n_index(self, n: int, mode: str) -> list[int]:
        if mode == "max":
            return torch.topk(
                torch.tensor(self.preds), n, largest=True
            ).indices.tolist()
        elif mode == "min":
            return torch.topk(
                torch.tensor(self.preds), n, largest=False
            ).indices.tolist()
        elif mode == "mid":
            mid_values = torch.abs(torch.tensor(self.preds) - 0.5)
            return torch.topk(mid_values, n, largest=False).indices.tolist()
        else:
            raise ValueError(f"mode must be max, min or mid, got {mode}")

    def get_best_worst_n(
        self,
        n: int,
        tgt_value: int,
        mode: Literal["best", "worst", "hesitate"],
    ) -> list[tuple]:
        if mode == "hesitate":
            indices = [i for i, tgt in enumerate(self.tgts) if tgt == tgt_value]
            sorted_indices = sorted(
                indices,
                key=lambda i: abs(self.preds[i] - 0.5),
                reverse=False,
            )
            return sorted_indices[:n]
            # return [(self.preds[i], self.tgts[i]) for i in sorted_indices[:n]]
        else:
            indices = [i for i, tgt in enumerate(self.tgts) if tgt == tgt_value]
            sorted_indices = sorted(
                indices,
                key=lambda i: abs(self.preds[i] - self.tgts[i]),
                reverse=True if mode == "worst" else False,
            )
            return sorted_indices[:n]
            # return [(self.preds[i], self.tgts[i]) for i in sorted_indices[:n]]

    def get_false_positive_indexes(
        self,
    ) -> list[tuple]:
        indices = [
            i
            for i, (pred, tgt) in enumerate(zip(self.preds, self.tgts))
            if pred > 0.5 and tgt == 0
        ]
        return indices

    def get_false_negative_indexes(
        self,
    ) -> list[tuple]:
        indices = [
            i
            for i, (pred, tgt) in enumerate(zip(self.preds, self.tgts))
            if pred < 0.5 and tgt == 1
        ]
        return indices

    def get_true_positive_indexes(
        self,
    ) -> list[tuple]:
        indices = [
            i
            for i, (pred, tgt) in enumerate(zip(self.preds, self.tgts))
            if pred > 0.5 and tgt == 1
        ]
        return indices

    def get_true_negative_indexes(
        self,
    ) -> list[tuple]:
        indices = [
            i
            for i, (pred, tgt) in enumerate(zip(self.preds, self.tgts))
            if pred < 0.5 and tgt == 0
        ]
        return indices

    def get_score_and_target_from_idx(self, idx: int) -> tuple:
        return self.preds[idx], self.tgts[idx]

    def save_raw_csv(self, filepath: str) -> str:
        """
        Save raw self.preds and self.tgts to CSV in the order they were collected.
        - Binary/multilabel (your current impl): columns = [index, pred, tgt]
        - Multiclass (per-sample probability vector): columns = [index, pred_0..pred_{C-1}, tgt]
        Returns absolute path to the written CSV.
        """
        # Normalize to plain Python types
        def to_list(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        preds = [to_list(p) for p in self.preds]
        tgts  = [to_list(t) for t in self.tgts]

        if len(preds) != len(tgts):
            raise ValueError(f"Length mismatch: preds={len(preds)} vs tgts={len(tgts)}")

        # Decide whether preds are scalar or vector per row
        is_vector = isinstance(preds[0], (list, tuple, np.ndarray))

        if not is_vector:
            # Simple two-column case
            df = pd.DataFrame({
                "index": np.arange(len(preds)),
                "pred": preds,
                "tgt": tgts,
            })
        else:
            # Expand vectors to columns pred_0..pred_{k-1}
            pred_mat = pd.DataFrame(preds)
            pred_mat.columns = [f"pred_{j}" for j in range(pred_mat.shape[1])]
            df = pred_mat
            df.insert(0, "index", np.arange(len(preds)))
            df["tgt"] = tgts

        # Ensure directory exists
        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        df.to_csv(abs_path, index=False)
        return abs_path

    @torch.no_grad
    def update(self, outputs, targets):
        """
        Args:
            outputs: model outputs (logits)
            targets: ground truth labels
        """
        # expect logits from outputs
        if self.task == "binary" or self.task == "multilabel":
            self.preds.extend(
                F.sigmoid(outputs).detach().cpu().numpy().flatten()
            )  # then sigmoid is applied here
            # self.preds.append(F.sigmoid(outputs).detach().cpu())
        elif self.task == "multiclass":
            self.preds.extend(
                F.softmax(outputs, dim=-1)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, self.num_classes)
            )
            # self.preds.append(F.softmax(outputs, dim=-1).detach().cpu())
        else:
            raise NotImplementedError(f"{self.task} is not implemented.")

        # self.tgts.append(targets.detach().cpu())
        self.tgts.extend(targets.detach().cpu().numpy().flatten())

    @torch.no_grad
    def compute(
        self,
    ):
        all_preds = all_gather(self.preds)
        all_tgts = all_gather(self.tgts)

        # all_preds = list(chain.from_iterable(all_preds))
        # all_tgts = list(chain.from_iterable(all_tgts))

        all_preds, all_tgts = torch.tensor(all_preds), torch.tensor(all_tgts)

        if len(all_preds) == 0 or all_preds.nelement() == 0:
            return {
                "accuracy": torch.tensor(torch.nan),
                "auroc": torch.tensor(torch.nan),
                "recall": torch.tensor(torch.nan),
                "f1_score": torch.tensor(torch.nan),
                "precision": torch.tensor(torch.nan),
            }

        # all_preds, all_tgts = torch.concat(all_preds, dim=0), torch.concat(
        #     all_tgts, dim=0
        # )

        cm = (
            confusion_matrix(
                all_preds,
                all_tgts,
                task=self.task,
                num_classes=self.num_classes,
                num_labels=self.num_labels,
            )
            .cpu()
            .numpy()
        )

        # Flatten the confusion matrix to store it in a dictionary
        cm_flat = cm.flatten()
        if self.task == "binary":
            tn, fp, fn, tp = cm.ravel()
            cm_dict = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        else:
            cm_dict = {f"cm_{i}": cm_flat[i] for i in range(len(cm_flat))}

        # Convert confusion matrix values to tensors
        cm_dict = {key: torch.tensor(value) for key, value in cm_dict.items()}

        # Get the portion of sampled instances
        positive_samples = cm_dict.get("tp", 0) + cm_dict.get("fn", 0)
        negative_samples = cm_dict.get("tn", 0) + cm_dict.get("fp", 0)

        try:
            performance = {
                "accuracy": accuracy(
                    all_preds,
                    all_tgts.int(),
                    task=self.task,
                    num_classes=self.num_classes,
                ),
                "auroc": auroc(
                    all_preds,
                    all_tgts.int(),
                    task=self.task,
                    num_classes=self.num_classes,
                ),
                "recall": recall(
                    all_preds,
                    all_tgts.int(),
                    task=self.task,
                    num_classes=self.num_classes,
                ),
                "f1_score": f1_score(
                    all_preds,
                    all_tgts.int(),
                    task=self.task,
                    num_classes=self.num_classes,
                ),
                "precision": precision(
                    all_preds,
                    all_tgts.int(),
                    task=self.task,
                    num_classes=self.num_classes,
                ),
                "positive_samples": positive_samples,  # plot them seperately
                "negative_samples": negative_samples,
                **cm_dict,
            }
        except Exception as e:
            print(e)
            print("All preds")
            print(all_preds)
            print("All targets")
            print(all_tgts)
            raise StopIteration()

        return performance

class ClassificationEvaluator:
    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel"] = "binary",
        num_classes: int = None,
        num_labels: int = None,
        use_soft_labels: bool = False,
        threshold_mode: Literal["fixed_0p5", "youden_j"] = "youden_j",
    ) -> None:
        self.preds = []
        self.tgts = []
        self.task = task
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.use_soft_labels = (
            use_soft_labels  # Whether to use soft labels for evaluation
        )
        self.threshold_mode = threshold_mode
        self.fixed_threshold = None

    def set_fixed_threshold(self, threshold: float | None):
        """Lock the operating threshold, normally using validation only."""
        self.fixed_threshold = None if threshold is None else float(threshold)

    def top_n_index(self, n: int, mode: str) -> list[int]:
        if mode == "max":
            return torch.topk(
                torch.tensor(self.preds), n, largest=True
            ).indices.tolist()
        elif mode == "min":
            return torch.topk(
                torch.tensor(self.preds), n, largest=False
            ).indices.tolist()
        elif mode == "mid":
            mid_values = torch.abs(torch.tensor(self.preds) - 0.5)
            return torch.topk(mid_values, n, largest=False).indices.tolist()
        else:
            raise ValueError(f"mode must be max, min or mid, got {mode}")

    def get_best_worst_n(
        self,
        n: int,
        tgt_value: int,
        mode: Literal["best", "worst", "hesitate"],
    ) -> list[tuple]:
        if mode == "hesitate":
            indices = [i for i, tgt in enumerate(self.tgts) if tgt == tgt_value]
            sorted_indices = sorted(
                indices,
                key=lambda i: abs(self.preds[i] - 0.5),
                reverse=False,
            )
            return sorted_indices[:n]
            # return [(self.preds[i], self.tgts[i]) for i in sorted_indices[:n]]
        else:
            indices = [i for i, tgt in enumerate(self.tgts) if tgt == tgt_value]
            sorted_indices = sorted(
                indices,
                key=lambda i: abs(self.preds[i] - self.tgts[i]),
                reverse=True if mode == "worst" else False,
            )
            return sorted_indices[:n]
            # return [(self.preds[i], self.tgts[i]) for i in sorted_indices[:n]]

    def get_false_positive_indexes(
        self,
    ) -> list[tuple]:
        indices = [
            i
            for i, (pred, tgt) in enumerate(zip(self.preds, self.tgts))
            if pred > 0.5 and tgt == 0
        ]
        return indices

    def get_false_negative_indexes(
        self,
    ) -> list[tuple]:
        indices = [
            i
            for i, (pred, tgt) in enumerate(zip(self.preds, self.tgts))
            if pred < 0.5 and tgt == 1
        ]
        return indices

    def get_true_positive_indexes(
        self,
    ) -> list[tuple]:
        indices = [
            i
            for i, (pred, tgt) in enumerate(zip(self.preds, self.tgts))
            if pred > 0.5 and tgt == 1
        ]
        return indices

    def get_true_negative_indexes(
        self,
    ) -> list[tuple]:
        indices = [
            i
            for i, (pred, tgt) in enumerate(zip(self.preds, self.tgts))
            if pred < 0.5 and tgt == 0
        ]
        return indices

    def get_score_and_target_from_idx(self, idx: int) -> tuple:
        return self.preds[idx], self.tgts[idx]

    @torch.no_grad
    def update(self, outputs, targets):
        """
        Args:
            outputs: model outputs (logits)
            targets: ground truth labels
        """
        # expect logits from outputs
        if self.task == "binary" or self.task == "multilabel":
            self.preds.extend(
                F.sigmoid(outputs).detach().cpu().numpy().flatten()
            )  # then sigmoid is applied here
            # self.preds.append(F.sigmoid(outputs).detach().cpu())
        elif self.task == "multiclass":
            self.preds.extend(
                F.softmax(outputs, dim=-1)
                .detach()
                .cpu()
                .numpy()
                .reshape(-1, self.num_classes)
            )
            # self.preds.append(F.softmax(outputs, dim=-1).detach().cpu())
        else:
            raise NotImplementedError(f"{self.task} is not implemented.")

        targets_np = targets.detach().cpu().numpy().flatten()

        if self.use_soft_labels:
            # Convert soft labels to hard labels for evaluation
            targets_np = (targets_np >= 0.5).astype(int)

        # self.tgts.append(targets.detach().cpu())
        self.tgts.extend(targets_np)


    @torch.no_grad
    def compute(
        self,
    ):
        all_preds = all_gather(self.preds)
        all_tgts = all_gather(self.tgts)

        # all_preds = list(chain.from_iterable(all_preds))
        # all_tgts = list(chain.from_iterable(all_tgts))

        all_preds, all_tgts = torch.tensor(all_preds), torch.tensor(all_tgts)

        if len(all_preds) == 0 or all_preds.nelement() == 0:
            return {
                "accuracy": torch.tensor(torch.nan),
                "auroc": torch.tensor(torch.nan),
                "recall": torch.tensor(torch.nan),
                "f1_score": torch.tensor(torch.nan),
                "precision": torch.tensor(torch.nan),
            }
        # all_preds, all_tgts = torch.concat(all_preds, dim=0), torch.concat(
        #     all_tgts, dim=0
        # )
        # print all the set values in all_preds and all_tgts
        # print("All preds unique values:", torch.unique(all_preds))
        # print("All targets unique values:", torch.unique(all_tgts))
        # # also the shape
        # print("All preds shape:", all_preds.shape)
        # print("All targets shape:", all_tgts.shape)

        if self.fixed_threshold is not None:
            best_threshold = self.fixed_threshold
        elif self.threshold_mode == "fixed_0p5":
            best_threshold = 0.5
        else:
            fpr, tpr, thresholds = roc_curve(
                all_tgts.flatten().numpy(), all_preds.flatten().numpy()
            )
            j_scores = tpr - fpr
            best_threshold = thresholds[j_scores.argmax()].item()
            if not (0 <= best_threshold <= 1):
                best_threshold = 0.5

        # print(f"Best threshold is")
        # print(type(best_threshold))
        # print(best_threshold)

        cm = (
            confusion_matrix(
                all_preds,
                all_tgts,
                task=self.task,
                num_classes=self.num_classes,
                num_labels=self.num_labels,
                threshold=best_threshold,
            )
            .cpu()
            .numpy()
        )

        # Flatten the confusion matrix to store it in a dictionary
        cm_flat = cm.flatten()
        if self.task == "binary":
            tn, fp, fn, tp = cm.ravel()
            cm_dict = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
        else:
            cm_dict = {f"cm_{i}": cm_flat[i] for i in range(len(cm_flat))}

        # Convert confusion matrix values to tensors
        cm_dict = {key: torch.tensor(value) for key, value in cm_dict.items()}

        # Get the portion of sampled instances
        positive_samples = cm_dict.get("tp", 0) + cm_dict.get("fn", 0)
        negative_samples = cm_dict.get("tn", 0) + cm_dict.get("fp", 0)

        try:
            performance = {
                "accuracy": accuracy(
                    all_preds,
                    all_tgts.int(),
                    task=self.task,
                    num_classes=self.num_classes,
                    threshold=best_threshold,
                ),
                "auroc": auroc(
                    all_preds,
                    all_tgts.int(),
                    task=self.task,
                    num_classes=self.num_classes,
                ),
                "recall": recall(
                    all_preds,
                    all_tgts.int(),
                    task=self.task,
                    num_classes=self.num_classes,
                    threshold=best_threshold,
                ),
                "f1_score": f1_score(
                    all_preds,
                    all_tgts.int(),
                    task=self.task,
                    num_classes=self.num_classes,
                    threshold=best_threshold,
                ),
                "precision": precision(
                    all_preds,
                    all_tgts.int(),
                    task=self.task,
                    num_classes=self.num_classes,
                    threshold=best_threshold,
                ),
                "positive_samples": positive_samples,  # plot them seperately
                "negative_samples": negative_samples,
                "threshold": torch.tensor(best_threshold),
                **cm_dict,
            }
        except Exception as e:
            print(e)
            print("All preds")
            print(all_preds)
            print("All targets")
            print(all_tgts)
            raise StopIteration()

        return performance

    def raw_dataframe(self) -> pd.DataFrame:
        """Return one row per evaluated sample for reproducible re-analysis."""
        return pd.DataFrame(
            {
                "row": np.arange(len(self.preds)),
                "probability": np.asarray(self.preds, dtype=float),
                "target": np.asarray(self.tgts),
            }
        )

    def save_raw_csv(self, filepath: str) -> str:
        """
        Save raw self.preds and self.tgts to CSV in the order they were collected.
        - Binary/multilabel (your current impl): columns = [index, pred, tgt]
        - Multiclass (per-sample probability vector): columns = [index, pred_0..pred_{C-1}, tgt]
        Returns absolute path to the written CSV.
        """
        # Normalize to plain Python types
        def to_list(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().tolist()
            if isinstance(x, np.ndarray):
                return x.tolist()
            return x

        preds = [to_list(p) for p in self.preds]
        tgts  = [to_list(t) for t in self.tgts]

        if len(preds) != len(tgts):
            raise ValueError(f"Length mismatch: preds={len(preds)} vs tgts={len(tgts)}")

        # Decide whether preds are scalar or vector per row
        is_vector = isinstance(preds[0], (list, tuple, np.ndarray))

        if not is_vector:
            # Simple two-column case
            df = pd.DataFrame({
                "index": np.arange(len(preds)),
                "pred": preds,
                "tgt": tgts,
            })
        else:
            # Expand vectors to columns pred_0..pred_{k-1}
            pred_mat = pd.DataFrame(preds)
            pred_mat.columns = [f"pred_{j}" for j in range(pred_mat.shape[1])]
            df = pred_mat
            df.insert(0, "index", np.arange(len(preds)))
            df["tgt"] = tgts

        # Ensure directory exists
        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        df.to_csv(abs_path, index=False)
        return abs_path

@torch.no_grad()
def evaluate_disease_by_time_window(
    evaluators_by_year: dict[int, ClassificationEvaluator],
    window: Tuple[int, int],  # e.g., (2, 5) or (0, 0)
):
    year_keys = sorted(evaluators_by_year.keys())  # [0, 2, 5, 10]
    preds = {y: evaluators_by_year[y].preds for y in year_keys}
    tgts = {y: evaluators_by_year[y].tgts for y in year_keys}

    # Sanity check: all years must have same patient count
    n_patients = len(tgts[year_keys[0]])
    assert all(len(v) == n_patients for v in tgts.values()), "Mismatch in patient count"

    start_year, end_year = window
    eval_years = [y for y in year_keys if start_year <= y <= end_year]

    selected_preds, selected_tgts = [], []

    for i in range(n_patients):
        label_vec = [tgts[y][i] for y in year_keys]
        pred_vec = [preds[y][i] for y in year_keys]

        if start_year == end_year:
            # Point evaluation
            y = start_year
            selected_preds.append(pred_vec[year_keys.index(y)])
            selected_tgts.append(label_vec[year_keys.index(y)])
        else:
            # Windowed evaluation: only include if no prior onset, and onset within window
            onset_before_window = any(
                label_vec[year_keys.index(y)] == 1 for y in year_keys if y < start_year
            )
            onset_in_window = any(
                label_vec[year_keys.index(y)] == 1 for y in eval_years
            )
            if not onset_before_window and onset_in_window:
                selected_preds.append(
                    max(pred_vec[year_keys.index(y)] for y in eval_years)
                )
                selected_tgts.append(1)

    if not selected_tgts:
        return {
            "accuracy": torch.tensor(float("nan")),
            "auroc": torch.tensor(float("nan")),
            "recall": torch.tensor(float("nan")),
            "f1_score": torch.tensor(float("nan")),
            "precision": torch.tensor(float("nan")),
            "positive_samples": torch.tensor(0),
            "negative_samples": torch.tensor(0),
            "tn": torch.tensor(0),
            "fp": torch.tensor(0),
            "fn": torch.tensor(0),
            "tp": torch.tensor(0),
        }

    # Compute metrics
    all_preds = torch.tensor(selected_preds)
    all_tgts = torch.tensor(selected_tgts).int()
    cm = (
        confusion_matrix(all_preds > 0.5, all_tgts, task="binary", num_classes=2)
        .cpu()
        .numpy()
    )

    tn, fp, fn, tp = cm.ravel()
    cm_dict = {
        "tn": torch.tensor(tn),
        "fp": torch.tensor(fp),
        "fn": torch.tensor(fn),
        "tp": torch.tensor(tp),
    }
    pos_samples = cm_dict["tp"] + cm_dict["fn"]
    neg_samples = cm_dict["tn"] + cm_dict["fp"]

    return {
        "accuracy": accuracy(all_preds > 0.5, all_tgts, task="binary", num_classes=2),
        "auroc": auroc(all_preds, all_tgts, task="binary"),
        "recall": recall(all_preds > 0.5, all_tgts, task="binary", num_classes=2),
        "f1_score": f1_score(all_preds > 0.5, all_tgts, task="binary", num_classes=2),
        "precision": precision(all_preds > 0.5, all_tgts, task="binary", num_classes=2),
        "positive_samples": pos_samples,
        "negative_samples": neg_samples,
        **cm_dict,
    }


def group_evaluators_by_disease_and_year(evaluators, progression_label_years):
    disease_to_evaluators = defaultdict(dict)

    # Regex pattern: matches strings like "has_glaucoma_in_2_years"
    pattern = re.compile(r"^has_([a-z]+)_in_(\d+)_years$")

    for label_name, evaluator in evaluators.items():
        match = pattern.match(label_name)
        if match:
            disease = match.group(1)
            year = int(match.group(2))

            if year in progression_label_years:
                disease_to_evaluators[disease][year] = evaluator

    return disease_to_evaluators


def generate_eval_windows(years: List[int]) -> List[Tuple[int, int]]:
    """
    Generate only adjacent time windows as (start_year, end_year) tuples.

    Example:
        Input: [0, 2, 5, 10]
        Output: [(0, 2), (2, 5), (5, 10)]
    """
    years = sorted(years)
    return [(years[i], years[i + 1]) for i in range(len(years) - 1)]



class DeepHitHorizonEvaluator:
    """
    Wraps one ClassificationEvaluator per horizon.
    Expects:
      - probs in [0,1] for each horizon (not logits)
      - binary labels 0/1 for each horizon
    """

    def __init__(self, progression_years: list[int]) -> None:
        self.progression_years = progression_years
        self.evals = {
            h: ClassificationEvaluator(task="binary", num_classes=2)
            for h in progression_years
        }
        self.eps = 1e-7

    @torch.no_grad
    def update(
        self,
        risks_by_horizon: dict[int, torch.Tensor],  # h -> [B] prob
        labels_by_horizon: dict[int, torch.Tensor], # h -> [B] 0/1
    ):
        """
        risks_by_horizon[h]: predicted P(event by h years), shape [B]
        labels_by_horizon[h]: ground-truth 0/1, shape [B]
        """
        for h in self.progression_years:
            if h not in risks_by_horizon:
                continue
            prob = risks_by_horizon[h].detach()

            # ClassificationEvaluator expects logits, so convert prob -> logit
            prob = prob.clamp(self.eps, 1.0 - self.eps)
            logit = torch.log(prob / (1.0 - prob))  # [B]

            tgt = labels_by_horizon[h].detach().float()

            self.evals[h].update(logit, tgt)

    @torch.no_grad
    def compute(self):
        """
        Returns a dict: {h: metrics_dict_for_that_horizon}
        """
        results = {}
        for h in self.progression_years:
            results[h] = self.evals[h].compute()
        return results
