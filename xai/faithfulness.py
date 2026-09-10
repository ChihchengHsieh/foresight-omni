from __future__ import annotations

import random
from typing import Hashable, Sequence


def ranked_deletion_plan(
    rows: Sequence[dict],
    identity_key: str,
    counts: Sequence[int] = (1, 3, 5, 10),
    seed: int = 42,
) -> list[dict]:
    """Build matched top, low-ranked, and random deletion sets."""
    if not rows:
        return []
    identities = [row[identity_key] for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError(f"Deletion identities are not unique for key [{identity_key}]")

    top = [
        row[identity_key]
        for row in sorted(
            rows,
            key=lambda row: abs(float(row["signed_logit_contribution"])),
            reverse=True,
        )
    ]
    low = list(reversed(top))
    random_order = list(identities)
    random.Random(seed).shuffle(random_order)

    valid_counts = sorted({min(int(count), len(rows)) for count in counts if count > 0})
    plan = []
    for strategy, order in (("top", top), ("low", low), ("random", random_order)):
        for count in valid_counts:
            plan.append(
                {
                    "strategy": strategy,
                    "count": count,
                    "identities": order[:count],
                }
            )
    return plan


def attach_deletion_logits(
    plan: Sequence[dict],
    masked_logits: Sequence[float],
    baseline_logit: float,
) -> list[dict]:
    if len(plan) != len(masked_logits):
        raise ValueError("Deletion plan/logit length mismatch")
    rows = []
    for item, masked_logit in zip(plan, masked_logits):
        row = dict(item)
        row["identities"] = ";".join(str(value) for value in item["identities"])
        row["masked_logit"] = float(masked_logit)
        row["logit_drop"] = float(baseline_logit - masked_logit)
        rows.append(row)
    return rows
