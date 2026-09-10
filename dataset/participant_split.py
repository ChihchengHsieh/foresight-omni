"""Participant-disjoint split helpers for cohort fine-tuning."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd


def assign_participant_finetune_split(
    df: pd.DataFrame,
    *,
    fine_tune_portion: float,
    train_fraction: float,
    seed: int,
    participant_col: str = "entity_id",
    split_col: str = "finetune_split",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Assign each participant to exactly one train, validation, or test split.

    ``fine_tune_portion`` is the combined training and validation fraction.
    ``train_fraction`` is the training fraction within that adaptation subset.
    Assignment is deterministic for a given participant set and seed, and is
    independent of dataframe row order.
    """

    if participant_col not in df.columns:
        raise ValueError(
            f"Participant-level fine-tuning requires column {participant_col!r}."
        )
    if df.empty:
        raise ValueError("Cannot split an empty CLSA dataframe.")
    if not 0.0 < float(fine_tune_portion) < 1.0:
        raise ValueError("fine_tune_portion must be strictly between 0 and 1.")
    if not 0.0 < float(train_fraction) < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1.")

    participant_keys = df[participant_col].astype("string")
    if participant_keys.isna().any() or participant_keys.str.strip().eq("").any():
        raise ValueError(
            f"Column {participant_col!r} contains missing participant identifiers."
        )

    participants = np.asarray(sorted(participant_keys.unique().tolist()), dtype=object)
    n_participants = int(len(participants))
    n_adaptation = int(round(n_participants * float(fine_tune_portion)))
    n_train = int(round(n_adaptation * float(train_fraction)))
    n_val = n_adaptation - n_train
    n_test = n_participants - n_adaptation
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            "Participant-level split would leave an empty train, validation, or test set."
        )

    shuffled = np.random.default_rng(int(seed)).permutation(participants)
    split_values = np.empty(n_participants, dtype=object)
    split_values[:n_train] = "train"
    split_values[n_train:n_adaptation] = "val"
    split_values[n_adaptation:] = "test"

    manifest = pd.DataFrame(
        {
            participant_col: shuffled.astype(str),
            split_col: split_values,
        }
    ).sort_values(participant_col, kind="stable", ignore_index=True)
    if manifest[participant_col].duplicated().any():
        raise AssertionError("Participant split manifest contains duplicate identifiers.")

    mapping = manifest.set_index(participant_col)[split_col]
    assigned = df.copy()
    assigned[split_col] = participant_keys.map(mapping)
    if assigned[split_col].isna().any():
        raise AssertionError("Some CLSA rows were not assigned to a fine-tuning split.")

    participant_split_counts = (
        assigned.assign(_participant_key=participant_keys)
        .groupby("_participant_key", dropna=False)[split_col]
        .nunique()
    )
    if int(participant_split_counts.max()) != 1:
        raise AssertionError("A CLSA participant occurs in more than one split.")

    summary = {
        "participants_total": n_participants,
        "participants_train": n_train,
        "participants_val": n_val,
        "participants_test": n_test,
        "rows_total": int(len(assigned)),
        "rows_train": int((assigned[split_col] == "train").sum()),
        "rows_val": int((assigned[split_col] == "val").sum()),
        "rows_test": int((assigned[split_col] == "test").sum()),
        "cross_split_participants": 0,
    }
    return assigned, manifest, summary
