from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch


def remove_clinical_history_event(
    history: torch.Tensor,
    event_index: int,
) -> torch.Tensor:
    """Remove one encoded ICD event and recompute visible-history burden.

    The exact dataset-backed XAI path should regenerate the tensor from source
    events. This tensor-only helper is deterministic and useful for regression
    tests or cached inputs where the raw event list is unavailable.
    """
    if history.ndim != 2 or history.shape[1] != 6:
        raise ValueError(f"Expected clinical history shape (L, 6), got {history.shape}")
    valid_indices = torch.nonzero(history[:, 0] > 0, as_tuple=False).flatten().tolist()
    if event_index not in valid_indices:
        raise IndexError(f"Clinical event index {event_index} is not valid")

    kept = [index for index in valid_indices if index != event_index]
    result = torch.zeros_like(history)
    if not kept:
        return result
    kept_rows = history[kept].clone()
    event_count = len(kept)
    unique_count = len(set(int(value.item()) for value in kept_rows[:, 0]))
    kept_rows[:, 4] = math.log1p(event_count)
    kept_rows[:, 5] = math.log1p(unique_count)
    result[:event_count] = kept_rows
    return result


def clinical_history_event_variants(history: torch.Tensor) -> list[tuple[int, torch.Tensor]]:
    indices = torch.nonzero(history[:, 0] > 0, as_tuple=False).flatten().tolist()
    return [
        (index, remove_clinical_history_event(history, index))
        for index in indices
    ]


def omics_reference_variants(
    values: torch.Tensor,
    feature_indices: Iterable[int] | None = None,
    reference_value: float = 0.0,
) -> list[tuple[int, str, torch.Tensor]]:
    """Create value and missingness perturbations for an omics vector.

    Observed features receive an observed-to-reference value perturbation and an
    observed-to-missing perturbation. Missing features receive a
    missing-to-reference perturbation. These must be reported separately.
    """
    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional omics vector, got {values.shape}")
    indices = range(values.numel()) if feature_indices is None else feature_indices
    variants = []
    for index in indices:
        index = int(index)
        observed = bool(torch.isfinite(values[index]).item())
        reference = values.clone()
        reference[index] = reference_value
        if observed:
            variants.append((index, "observed_to_reference", reference))
            missing = values.clone()
            missing[index] = float("nan")
            variants.append((index, "observed_to_missing", missing))
        else:
            variants.append((index, "missing_to_reference", reference))
    return variants
