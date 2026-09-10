import logging
import os
from typing import List
from utils.plot import plot_perf, plot_perf_one_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import re


def log_mapping(title, values):
    """Write a consistently formatted mapping to the experiment log."""
    logging.info("%s", title)
    for key, value in sorted(values.items()):
        logging.info("  %s: %s", key, value)

def clean_saving_str(input):
    input = input.replace(" ", "").replace("<", "LessThan")
    if len(input) == 0:
        return "performance"
    return input


def plot_trend_curve(ax, values, color):
    x = np.arange(len(values))
    z = np.polyfit(x, values, 1)  # slope, intercept
    p = np.poly1d(z)
    ax.plot(x, p(x), "r--", color=color)


def is_ci_helper_metric(metric_name):
    return metric_name.endswith(
        (
            "_auroc_ci_low",
            "_auroc_ci_high",
            "_auroc_ci_width",
            "_auroc_ci_bootstrap_samples",
            "_low_positive_flag",
            "_ci_crosses_0p5_flag",
            "_negative_samples_for_ci",
        )
    )


def finite_or_none(value):
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


class GeneralTrainingLogger(object):
    def __init__(self, inspecting: bool, exceptions: List[str] = ["classes"]) -> None:
        self.train_logs = []
        self.exceptions = exceptions
        self.inspecting = inspecting
        if self.inspecting:
            self.val_logs = []

    def update(self, train_log, val_log={}):
        self.train_logs.append(train_log)
        if self.inspecting:
            # assert not val_log is None, "Inspecting mode need val_out."
            self.val_logs.append(val_log)

    @staticmethod
    def _primary_endpoint_rows(logs, split, diseases):
        rows = []
        disease_pattern = "|".join(re.escape(d) for d in diseases)
        endpoint_re = re.compile(rf"^({disease_pattern})_(\d+)_auroc$")
        for log_index, log in enumerate(logs):
            if not log:
                continue
            epoch = int(log.get("epoch", log_index))
            for metric, value in log.items():
                match = endpoint_re.match(metric)
                auroc_value = finite_or_none(value)
                if match is None or auroc_value is None:
                    continue
                disease, horizon_text = match.groups()
                horizon = int(horizon_text)
                prefix = f"{disease}_{horizon}"
                loss_key = f"has_{disease}_in_{horizon}_years_loss_scaled"
                rows.append(
                    {
                        "epoch": epoch,
                        "split": split,
                        "disease": disease,
                        "horizon": horizon,
                        "loss": finite_or_none(log.get(loss_key)),
                        "auroc": auroc_value,
                        "positive_samples": finite_or_none(log.get(f"{prefix}_positive_samples")),
                        "negative_samples": finite_or_none(log.get(f"{prefix}_negative_samples")),
                        "ci_low": finite_or_none(log.get(f"{prefix}_auroc_ci_low")),
                        "ci_high": finite_or_none(log.get(f"{prefix}_auroc_ci_high")),
                        "low_support": finite_or_none(log.get(f"{prefix}_low_positive_flag")),
                    }
                )
        return rows

    def endpoint_metrics_dataframe(self, diseases):
        rows = self._primary_endpoint_rows(self.train_logs, "train", diseases)
        if self.inspecting:
            rows.extend(self._primary_endpoint_rows(self.val_logs, "validation", diseases))
        return pd.DataFrame(
            rows,
            columns=[
                "epoch",
                "split",
                "disease",
                "horizon",
                "loss",
                "auroc",
                "positive_samples",
                "negative_samples",
                "ci_low",
                "ci_high",
                "low_support",
            ],
        )

    def epoch_summary_dataframe(self, diseases):
        endpoint_df = self.endpoint_metrics_dataframe(diseases)
        rows = []
        n_epochs = len(self.train_logs)
        for log_index in range(n_epochs):
            train_log = self.train_logs[log_index] or {}
            val_log = self.val_logs[log_index] if self.inspecting else {}
            val_log = val_log or {}
            epoch = int(train_log.get("epoch", log_index))
            epoch_endpoints = endpoint_df[endpoint_df["epoch"] == epoch]
            train_auc = epoch_endpoints.loc[
                epoch_endpoints["split"] == "train", "auroc"
            ].mean()
            val_auc = epoch_endpoints.loc[
                epoch_endpoints["split"] == "validation", "auroc"
            ].mean()
            row = {
                "epoch": epoch,
                "train_loss": finite_or_none(train_log.get("loss")),
                "val_loss": finite_or_none(val_log.get("loss")),
                "mean_train_auroc": finite_or_none(train_auc),
                "mean_val_auroc": finite_or_none(val_auc),
                "train_val_auroc_gap": (
                    float(train_auc - val_auc)
                    if np.isfinite(train_auc) and np.isfinite(val_auc)
                    else None
                ),
                "learning_rate": finite_or_none(train_log.get("lr")),
            }
            for disease in diseases:
                disease_rows = epoch_endpoints[epoch_endpoints["disease"] == disease]
                row[f"train_{disease}_mean_auroc"] = finite_or_none(
                    disease_rows.loc[disease_rows["split"] == "train", "auroc"].mean()
                )
                row[f"val_{disease}_mean_auroc"] = finite_or_none(
                    disease_rows.loc[
                        disease_rows["split"] == "validation", "auroc"
                    ].mean()
                )
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _metric_points(logs, metric):
        """Return finite ``(epoch, value)`` points for a logged metric."""
        points = []
        for log_index, log in enumerate(logs):
            if not log:
                continue
            value = finite_or_none(log.get(metric))
            if value is None:
                continue
            points.append((int(log.get("epoch", log_index)), value))
        return points

    def monitoring_panel_specs(self, labels, endpoint_df):
        """Describe the appropriate monitoring panel for each requested target.

        Disease/horizon targets use AUROC from ``endpoint_df``. Direct
        categorical targets (for example sex) use their logged AUROC, whereas
        numerical regression targets use mean squared error. Targets without a
        supported finite metric are omitted instead of producing an empty,
        misleading AUROC panel.
        """
        specs = []
        for label in labels:
            label_endpoints = endpoint_df[endpoint_df["disease"] == label]
            if not label_endpoints.empty:
                specs.append(
                    {
                        "label": label,
                        "kind": "endpoint_auroc",
                        "metric": None,
                        "ylabel": "AUROC",
                    }
                )
                continue

            direct_metrics = (
                (f"{label}_auroc", "direct_auroc", "AUROC"),
                (
                    f"{label}_mean_squared_error",
                    "direct_mse",
                    "Mean squared error",
                ),
            )
            all_logs = self.train_logs + (self.val_logs if self.inspecting else [])
            for metric, kind, ylabel in direct_metrics:
                if self._metric_points(all_logs, metric):
                    specs.append(
                        {
                            "label": label,
                            "kind": kind,
                            "metric": metric,
                            "ylabel": ylabel,
                        }
                    )
                    break
        return specs

    def _save_monitoring_figures(self, figures_dir, diseases, endpoint_df, summary_df):
        os.makedirs(figures_dir, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0, 0].plot(summary_df["epoch"], summary_df["train_loss"], label="Train")
        axes[0, 0].plot(summary_df["epoch"], summary_df["val_loss"], label="Validation")
        axes[0, 0].set_title("Loss")
        axes[0, 1].plot(summary_df["epoch"], summary_df["mean_train_auroc"], label="Train")
        axes[0, 1].plot(summary_df["epoch"], summary_df["mean_val_auroc"], label="Validation")
        axes[0, 1].set_title("Mean AUROC across diseases and horizons")
        axes[1, 0].plot(summary_df["epoch"], summary_df["train_val_auroc_gap"], color="purple")
        axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
        axes[1, 0].set_title("Train − validation AUROC gap")
        axes[1, 1].plot(summary_df["epoch"], summary_df["learning_rate"], color="green")
        axes[1, 1].set_title("Learning rate")
        for ax in axes.flat:
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "training_overview.png"), dpi=140)
        plt.close(fig)

        panel_specs = self.monitoring_panel_specs(diseases, endpoint_df)
        if not panel_specs:
            return

        fig, axes = plt.subplots(
            len(panel_specs),
            1,
            figsize=(12, 3 * len(panel_specs)),
            sharex=True,
        )
        axes = np.asarray(axes, dtype=object).reshape(-1)
        colors = plt.cm.tab10.colors
        for ax, spec in zip(axes, panel_specs):
            label = spec["label"]
            if spec["kind"] == "endpoint_auroc":
                disease_df = endpoint_df[endpoint_df["disease"] == label]
                horizons = sorted(disease_df["horizon"].dropna().unique())
                for color_index, horizon in enumerate(horizons):
                    color = colors[color_index % len(colors)]
                    horizon_df = disease_df[disease_df["horizon"] == horizon]
                    train_df = horizon_df[horizon_df["split"] == "train"]
                    val_df = horizon_df[horizon_df["split"] == "validation"]
                    ax.plot(
                        train_df["epoch"],
                        train_df["auroc"],
                        linestyle="--",
                        marker="o",
                        markersize=3,
                        color=color,
                        alpha=0.55,
                    )
                    ax.plot(
                        val_df["epoch"],
                        val_df["auroc"],
                        marker="o",
                        markersize=4,
                        color=color,
                        label=f"{horizon}y val",
                    )
                    ci_df = val_df.dropna(subset=["ci_low", "ci_high"])
                    if not ci_df.empty:
                        ax.errorbar(
                            ci_df["epoch"],
                            ci_df["auroc"],
                            yerr=[
                                ci_df["auroc"] - ci_df["ci_low"],
                                ci_df["ci_high"] - ci_df["auroc"],
                            ],
                            fmt="none",
                            color=color,
                            capsize=3,
                            alpha=0.8,
                        )
                legend_columns = min(4, max(1, len(horizons)))
            else:
                metric = spec["metric"]
                train_points = self._metric_points(self.train_logs, metric)
                val_points = (
                    self._metric_points(self.val_logs, metric)
                    if self.inspecting
                    else []
                )
                if train_points:
                    train_epochs, train_values = zip(*train_points)
                    ax.plot(
                        train_epochs,
                        train_values,
                        linestyle="--",
                        marker="o",
                        markersize=3,
                        color=colors[0],
                        alpha=0.65,
                        label="train",
                    )
                if val_points:
                    val_epochs, val_values = zip(*val_points)
                    ax.plot(
                        val_epochs,
                        val_values,
                        marker="o",
                        markersize=4,
                        color=colors[0],
                        label="validation",
                    )
                legend_columns = 2

            metric_title = "AUROC" if spec["ylabel"] == "AUROC" else "MSE"
            ax.set_title(
                f"{label.upper()} {metric_title} : solid validation, dashed train"
            )
            ax.set_ylabel(spec["ylabel"])
            ax.grid(alpha=0.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(ncol=legend_columns, fontsize=8)
        axes[-1].set_xlabel("Epoch")
        fig.tight_layout()
        fig.savefig(os.path.join(figures_dir, "disease_horizon_auroc.png"), dpi=140)
        plt.close(fig)

    def save_optimized_artifacts(self, directory, diseases, render_plots=True):
        """Write compact metrics every epoch and lightweight monitoring plots on demand."""
        metrics_dir = os.path.join(directory, "metrics")
        figures_dir = os.path.join(directory, "figures")
        os.makedirs(metrics_dir, exist_ok=True)
        endpoint_df = self.endpoint_metrics_dataframe(diseases)
        summary_df = self.epoch_summary_dataframe(diseases)
        summary_df.to_csv(os.path.join(metrics_dir, "epoch_summary.csv"), index=False)
        endpoint_df.to_csv(os.path.join(metrics_dir, "endpoint_metrics.csv"), index=False)
        # Preserve the established files for existing analysis scripts.
        pd.DataFrame(self.train_logs).to_csv(
            os.path.join(directory, "train_logs.csv"), index=False
        )
        if self.inspecting:
            pd.DataFrame(self.val_logs).to_csv(
                os.path.join(directory, "val_logs.csv"), index=False
            )
        try:
            endpoint_df.to_parquet(
                os.path.join(metrics_dir, "endpoint_metrics.parquet"), index=False
            )
        except (ImportError, ModuleNotFoundError, ValueError) as error:
            logging.warning("Could not write endpoint_metrics.parquet: %s", error)
        if render_plots and not summary_df.empty:
            self._save_monitoring_figures(figures_dir, diseases, endpoint_df, summary_df)

    def plot(
        self,
    ):
        for m in self.train_logs[0].keys():
            plot_perf(
                [t_log[m] if m in t_log else None for t_log in self.train_logs],
                (
                    [v_log[m] if m in v_log else None for v_log in self.val_logs]
                    if self.inspecting
                    else None
                ),
                m,
            )
        plt.close()

    def save_to_one_figure(self, dir, label_names: list[str], plot_trend=False):
        # saving loss figure first
        fig, axes = plt.subplots(1, 1, figsize=(15, 5))
        axes.set_title("Loss", fontsize=12, fontweight="bold")
        axes.set_xlabel("Epochs")
        train_values = [p["loss"] if "loss" in p else None for p in self.train_logs]
        axes.plot(train_values, label="Training", color="steelblue")
        if self.inspecting:
            val_values = [
                p["loss"] if ((p is not None) and "loss" in p) else None
                for p in self.val_logs
            ]
            axes.plot(val_values, label="Validation", color="orange")
            if plot_trend:
                plot_trend_curve(axes, val_values, "orange")
        axes.legend()
        axes.grid(True)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(dir, "loss.png"))
        plt.cla()
        plt.clf()
        plt.close()

        # Then lr figure
        fig, axes = plt.subplots(1, 1, figsize=(15, 5))
        axes.set_title("Learning Rate", fontsize=12, fontweight="bold")
        axes.set_xlabel("Epochs")
        train_values = [p["lr"] if "lr" in p else None for p in self.train_logs]
        axes.plot(train_values, label="Training", color="steelblue")
        axes.grid(True)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(dir, "lr.png"))
        plt.cla()
        plt.clf()
        plt.close()

        # Then other metrics figures
        for label in label_names:
            metrics = sorted(
                [
                    m
                    for m in self.train_logs[0].keys()
                    if m not in self.exceptions and m.startswith(label)
                    and not is_ci_helper_metric(m)
                ]
            )
            train_val_sep_metrics = [
                m
                for m in metrics
                if m.endswith(
                    (
                        # "_tp",
                        # "_fp",
                        # "_tn",
                        # "_fn",
                        # "_samples",
                        # "_counts",
                        # "_mean_squared_error",
                        # "_c_index"
                    )
                )
            ]
            loss_metrics = [m for m in metrics if "loss" in m]
            normal_metrics = [
                m
                for m in metrics
                if m not in train_val_sep_metrics and m not in loss_metrics
            ]  # First

            # Order: Normal metrics -> Loss metrics -> CM metrics
            ordered_metrics = normal_metrics + loss_metrics + train_val_sep_metrics

            num_metrics = len(metrics) + len(train_val_sep_metrics)
            num_cols = 3  # Adjust as needed
            num_rows = (
                num_metrics + num_cols - 1
            ) // num_cols  # Calculate rows dynamically

            if num_rows == 0:
                num_rows = 1

            fig, axes = plt.subplots(
                num_rows, num_cols, figsize=(30, 5 * num_rows)
            )  # Large figure

            axes = np.asarray(axes, dtype=object).reshape(-1)

            subplot_idx = 0  # Track subplot index

            for m in ordered_metrics:
                ax = axes[subplot_idx]

                # check if m is either end with _fp, _fn, _tp, _tn
                if m in train_val_sep_metrics:
                    train_values = [p[m] if m in p else None for p in self.train_logs]
                    ax.set_title(f"Train - {m}", fontsize=12, fontweight="bold")
                    ax.set_xlabel("Epochs")
                    ax.plot(
                        train_values, label="Training", color="steelblue", marker="o"
                    )
                    if plot_trend:
                        plot_trend_curve(ax, train_values, "steelblue")

                    ax.legend()
                    ax.grid(True)
                    subplot_idx += 1  # Move to the next subplot for validation

                    if self.inspecting:
                        ax = axes[subplot_idx]
                        ax.set_title(
                            f"Validation - {m}", fontsize=12, fontweight="bold"
                        )
                        ax.set_xlabel("Epochs")
                        val_values = [
                            p[m] if (p is not None) and m in p else None
                            for p in self.val_logs
                        ]
                        ax.plot(
                            val_values, label="Validation", color="orange", marker="o"
                        )
                        if plot_trend:
                            plot_trend_curve(ax, val_values, "orange")
                        ax.legend()
                        ax.grid(True)
                        subplot_idx += 1  # Move to the next available subplot

                else:
                    ax.set_title(m, fontsize=12, fontweight="bold")
                    ax.set_xlabel("Epochs")
                    train_values = [p[m] if m in p else None for p in self.train_logs]
                    val_values = (
                        [
                            p[m] if (p is not None) and m in p else None
                            for p in self.val_logs
                        ]
                        if self.inspecting and (m in self.train_logs[0].keys())
                        else None
                    )
                    ax.plot(
                        train_values, label="Training", color="steelblue", marker="o"
                    )
                    if plot_trend:
                        plot_trend_curve(ax, train_values, "steelblue")
                    if val_values is not None:
                        ax.plot(
                            val_values,
                            label="Validation AUROC" if m.endswith("_auroc") else "Validation",
                            color="orange",
                            marker="o",
                        )
                        ci_low_key = f"{m}_ci_low"
                        ci_high_key = f"{m}_ci_high"
                        if m.endswith("_auroc") and any(
                            (p is not None)
                            and (ci_low_key in p)
                            and (ci_high_key in p)
                            for p in self.val_logs
                        ):
                            ci_epochs = []
                            ci_values = []
                            yerr_low = []
                            yerr_high = []
                            for epoch, (p, y) in enumerate(zip(self.val_logs, val_values)):
                                y = finite_or_none(y)
                                lo = (
                                    finite_or_none(p.get(ci_low_key))
                                    if p is not None
                                    else None
                                )
                                hi = (
                                    finite_or_none(p.get(ci_high_key))
                                    if p is not None
                                    else None
                                )
                                if y is None or lo is None or hi is None:
                                    continue
                                ci_epochs.append(epoch)
                                ci_values.append(y)
                                yerr_low.append(max(0.0, y - lo))
                                yerr_high.append(max(0.0, hi - y))
                            ax.errorbar(
                                ci_epochs,
                                ci_values,
                                yerr=[yerr_low, yerr_high],
                                label="Validation 95% CI",
                                color="darkorange",
                                fmt="none",
                                capsize=4,
                                elinewidth=1.5,
                                capthick=1.5,
                                zorder=3,
                            )
                        if plot_trend:
                            plot_trend_curve(ax, val_values, "orange")

                    ax.legend()
                    ax.grid(True)
                    subplot_idx += 1  # Move to the next available subplot

            for unused_ax in axes[subplot_idx:]:
                unused_ax.set_visible(False)

            fig.suptitle(
                f"Performance Metrics for {label}", fontsize=16, fontweight="bold"
            )
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(os.path.join(dir, f"{clean_saving_str(label)}.png"))
            plt.cla()
            plt.clf()
            plt.close()

        pd.DataFrame(self.train_logs).to_csv(os.path.join(dir, "train_logs.csv"))
        if self.inspecting:
            pd.DataFrame(self.val_logs).to_csv(os.path.join(dir, "val_logs.csv"))

    def save(self, dir):
        for m in self.train_logs[0].keys():
            if m in self.exceptions:
                continue
            # check if m is either end with _fp, _fn, _tp, _tn
            if (
                m.endswith("_fp")
                or m.endswith("_fn")
                or m.endswith("_tp")
                or m.endswith("_tn")
            ):
                # then plot the trianing and valdiation seperatrely.
                fig = plot_perf_one_curve(
                    [p[m] if m in p else None for p in self.train_logs],
                    m,
                    save=True,
                    colour="steelblue",
                )
                fig.savefig(os.path.join(dir, f"{clean_saving_str(m)}_training.png"))
                plt.cla()
                plt.clf()
                plt.close()

                if self.inspecting:
                    fig = plot_perf_one_curve(
                        (
                            [
                                p[m] if (p is not None) and m in p else None
                                for p in self.val_logs
                            ]
                            if self.inspecting and (m in self.train_logs[0].keys())
                            else None
                        ),
                        m,
                        save=True,
                        colour="orange",
                    )
                    fig.savefig(
                        os.path.join(dir, f"{clean_saving_str(m)}_validation.png")
                    )
                    plt.cla()
                    plt.clf()
                    plt.close()

            else:
                fig = plot_perf(
                    [p[m] if m in p else None for p in self.train_logs],
                    (
                        [
                            p[m] if (p is not None) and m in p else None
                            for p in self.val_logs
                        ]
                        if self.inspecting and (m in self.train_logs[0].keys())
                        else None
                    ),
                    m,
                    save=True,
                )
                fig.savefig(os.path.join(dir, f"{clean_saving_str(m)}.png"))
                plt.cla()
                plt.clf()
                plt.close()

        pd.DataFrame(self.train_logs).to_csv(os.path.join(dir, "train_logs.csv"))
        if self.inspecting:
            pd.DataFrame(self.val_logs).to_csv(os.path.join(dir, "val_logs.csv"))

    def get_mean_auroc(
        self,
        log_dict: dict,
        include_train: bool = False,
        return_details: bool = False,
    ):
        """
        Compute the mean AUROC from a single log dictionary.

        Parameters
        ----------
        log_dict : dict
            One epoch log, e.g. self.val_logs[-1]
        include_train : bool
            If False, ignore keys ending with '_training' if you ever use such naming.
        return_details : bool
            If True, also return the AUROC keys and values used in the mean.

        Returns
        -------
        float or tuple
            Mean AUROC, or (mean_auroc, used_metrics_dict) if return_details=True
        """
        used = {}

        for k, v in log_dict.items():
            if not k.endswith("_auroc"):
                continue

            if (not include_train) and ("train" in k.lower()):
                continue

            if v is None:
                continue

            # handle tensor / numpy / list-like scalar cases
            if isinstance(v, (list, tuple)):
                if len(v) == 0:
                    continue
                v = v[0]

            try:
                v = float(v)
            except (TypeError, ValueError):
                continue

            if math.isnan(v):
                continue

            used[k] = v

        mean_auroc = float("nan") if len(used) == 0 else sum(used.values()) / len(used)

        if return_details:
            return mean_auroc, used
        return mean_auroc

    def get_latest_val_mean_auroc(self, return_details: bool = False):
        """
        Mean AUROC from the most recent validation log.
        Use this for early stopping.
        """
        assert self.inspecting, "Validation logs are only available when inspecting=True."
        assert len(self.val_logs) > 0, "No validation logs recorded yet."
        return self.get_mean_auroc(
            self.val_logs[-1],
            include_train=False,
            return_details=return_details,
        )


class TrainingLogger(object):
    def __init__(
        self,
        inspecting: bool,
    ) -> None:
        self.train_losses = []
        self.train_sublosses = []
        self.inspecting = inspecting
        if self.inspecting:
            self.train_perfs = []
            self.val_losses = []
            self.val_sublosses = []
            self.val_perfs = []

    def update(self, train_out, val_out=None):
        self.train_losses.append(train_out["loss"])
        self.train_sublosses.append(train_out["sublosses"])

        if self.inspecting:
            assert not val_out is None, "Inspecting mode need val_out."
            self.train_perfs.append(train_out["perf_dict"])
            self.val_losses.append(val_out["loss"])
            self.val_sublosses.append(val_out["sublosses"])
            self.val_perfs.append(val_out["perf_dict"])

    def plot(
        self,
    ):
        plot_perf(
            self.train_losses,
            self.val_losses if self.inspecting else None,
            title="Loss",
        )

        for m in self.train_sublosses[0].keys():
            plot_perf(
                [p[m] for p in self.train_sublosses],
                [p[m] for p in self.val_sublosses] if self.inspecting else None,
                m,
            )

        if self.inspecting:
            for m in self.train_perfs[0].keys():
                plot_perf(
                    [p[m] for p in self.train_perfs],
                    [p[m] for p in self.val_perfs],
                    m,
                )

    def save(self, dir):
        plot_perf(
            self.train_losses,
            self.val_losses if self.inspecting else None,
            title="Loss",
            save=True,
        ).savefig(os.path.join(dir, "loss.png"))

        for m in self.train_sublosses[0].keys():
            plot_perf(
                [p[m] for p in self.train_sublosses],
                [p[m] for p in self.val_sublosses] if self.inspecting else None,
                m,
                save=True,
            ).savefig(os.path.join(dir, f"{m}.png"))

        if self.inspecting:
            for m in self.train_perfs[0].keys():
                plot_perf(
                    [p[m] for p in self.train_perfs],
                    [p[m] for p in self.val_perfs],
                    m,
                    save=True,
                ).savefig(os.path.join(dir, f"{m}.png"))
