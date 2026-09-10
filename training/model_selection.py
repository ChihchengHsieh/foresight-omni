"""Validation metric selection, patience tracking, and CI scheduling."""

import numpy as np


def _label_selected(metric_prefix, selected_labels):
    if not selected_labels:
        return True
    for label in selected_labels:
        label = str(label)
        if label.startswith("="):
            if metric_prefix == label[1:]:
                return True
        elif metric_prefix == label or metric_prefix.startswith(f"{label}_"):
            return True
    return False


def collect_auroc_values(metrics, min_positive_samples=0, selected_labels=None):
    """Collect endpoint AUROCs with both classes and adequate support.

    Confidence-interval fields, bootstrap diagnostics, and aggregate values are
    deliberately excluded from the authoritative checkpoint score.
    """
    values = []
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, (dict, list, tuple)):
                values.extend(
                    collect_auroc_values(value, min_positive_samples, selected_labels)
                )
            elif (
                isinstance(value, (int, float, np.floating))
                and str(key).lower().endswith("_auroc")
                and "_ci_" not in str(key).lower()
            ):
                prefix = str(key)[:-6]
                positives = metrics.get(f"{prefix}_positive_samples")
                negatives = metrics.get(f"{prefix}_negative_samples")
                if (
                    _label_selected(prefix, selected_labels)
                    and positives is not None
                    and negatives is not None
                    and int(positives) >= int(min_positive_samples)
                    and int(negatives) > 0
                ):
                    values.append(float(value))
    elif isinstance(metrics, (list, tuple)):
        for value in metrics:
            values.extend(
                collect_auroc_values(value, min_positive_samples, selected_labels)
            )
    return values


def collect_accuracy_values(metrics, min_positive_samples=0, selected_labels=None):
    """Collect supported endpoint accuracies at their configured thresholds."""
    values = []
    if isinstance(metrics, dict):
        for key, value in metrics.items():
            if isinstance(value, (dict, list, tuple)):
                values.extend(
                    collect_accuracy_values(value, min_positive_samples, selected_labels)
                )
            elif (
                isinstance(value, (int, float, np.floating))
                and str(key).lower().endswith("_accuracy")
            ):
                prefix = str(key)[:-9]
                positives = metrics.get(f"{prefix}_positive_samples")
                negatives = metrics.get(f"{prefix}_negative_samples")
                if (
                    _label_selected(prefix, selected_labels)
                    and positives is not None
                    and negatives is not None
                    and int(positives) >= int(min_positive_samples)
                    and int(negatives) > 0
                ):
                    values.append(float(value))
    elif isinstance(metrics, (list, tuple)):
        for value in metrics:
            values.extend(
                collect_accuracy_values(value, min_positive_samples, selected_labels)
            )
    return values


def get_model_selection_value(
    val_out, logger, metric, min_positive_samples=10, selected_labels=None
):
    """Return a finite validation loss or mean AUROC for model selection."""
    if metric == "val_loss":
        return val_out.get("loss")

    if metric == "mean_accuracy":
        finite_accuracies = [
            value
            for value in collect_accuracy_values(
                val_out, min_positive_samples, selected_labels
            )
            if np.isfinite(value)
        ]
        return float(np.mean(finite_accuracies)) if finite_accuracies else None

    finite_aurocs = [
        value
        for value in collect_auroc_values(
            val_out, min_positive_samples, selected_labels
        )
        if np.isfinite(value)
    ]
    if finite_aurocs:
        return float(np.mean(finite_aurocs))

    # A requested primary-label filter must never silently fall back to a
    # logger aggregate that can include auxiliary tasks.
    if selected_labels:
        return None

    # During the first validation pass the current metrics may legitimately
    # contain no supported two-class endpoint (for example in a small smoke
    # panel), and the logger has not been updated yet. Treat that as an
    # unavailable selection metric instead of asserting on an empty history.
    val_logs = getattr(logger, "val_logs", None)
    if not val_logs:
        return None
    logger_mean_auroc = logger.get_latest_val_mean_auroc()
    if logger_mean_auroc is not None and np.isfinite(logger_mean_auroc):
        return float(logger_mean_auroc)
    return None


def update_patience_counter(metric_available, improved, bad_epochs):
    """Update consecutive non-improving validation checks."""
    if not metric_available:
        return bad_epochs
    return 0 if improved else bad_epochs + 1


def metric_improved(current, best, min_delta=0.001):
    """Return whether an authoritative metric cleared the declared margin."""
    return (
        current is not None
        and np.isfinite(current)
        and (best is None or current >= best + float(min_delta))
    )


def selection_metric_improved(current, best, metric, min_delta=0.001):
    """Return whether a checkpoint-selection metric improved in its direction.

    AUROC and accuracy are maximised, whereas validation loss is minimised.
    Keeping the direction here prevents a ``val_loss`` run from silently using
    the higher-is-better comparison intended for ranking metrics.
    """
    if current is None or not np.isfinite(current):
        return False
    if best is None:
        return True
    if metric == "val_loss":
        return current <= best - float(min_delta)
    return current >= best + float(min_delta)


def full_validation_due(
    epoch,
    frequency,
    fast_value,
    best_fast,
    min_delta=0.001,
    metric="mean_auroc",
):
    """Return scheduled/candidate full-validation decisions.

    This function only schedules an authoritative check; it never updates
    checkpoint or patience state itself.
    """
    scheduled = int(frequency) > 0 and int(epoch) % int(frequency) == 0
    candidate = selection_metric_improved(
        fast_value, best_fast, metric, min_delta
    )
    return scheduled or candidate, scheduled, candidate


def validation_ci_bootstrap_for_epoch(args, epoch):
    """Return validation bootstrap count when CI evaluation is due."""
    if args.val_ci_bootstrap <= 0 or args.val_ci_freq <= 0:
        return 0
    return args.val_ci_bootstrap if epoch % args.val_ci_freq == 0 else 0


def test_ci_bootstrap(args):
    """Return the enabled final-test bootstrap count."""
    if not args.test_ci:
        return 0
    return max(0, int(args.test_ci_bootstrap))
