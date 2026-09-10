from __future__ import annotations

import math
import re
from typing import Mapping, Sequence

import torch

from xai.token_attribution import clean_feature_name


def fundus_spatial_manifest(
    token_count: int,
    image_height: int,
    image_width: int,
) -> list[dict]:
    """Map square CNN token grids to approximate image-space cells."""
    side = int(math.isqrt(token_count))
    if side * side != token_count:
        raise ValueError(
            f"Fundus spatial manifest requires a square token grid, got {token_count}"
        )
    manifest = []
    for index in range(token_count):
        row, column = divmod(index, side)
        manifest.append(
            {
                "token_index": index,
                "row": row,
                "column": column,
                "y0": round(row * image_height / side),
                "y1": round((row + 1) * image_height / side),
                "x0": round(column * image_width / side),
                "x1": round((column + 1) * image_width / side),
                "feature": f"Fundus CNN spatial cell r{row + 1}c{column + 1}",
            }
        )
    return manifest


def clinical_history_manifest(
    history: torch.Tensor,
    id_to_code: Mapping[int, str] | None = None,
) -> list[dict]:
    """Create a traceable manifest from a clinical-history model tensor."""
    if history.ndim != 2 or history.shape[1] != 6:
        raise ValueError(f"Expected clinical history shape (L, 6), got {history.shape}")
    id_to_code = id_to_code or {}
    rows = []
    for row_index, row in enumerate(history.detach().cpu()):
        code_id = int(row[0].item())
        if code_id <= 0:
            continue
        days_ago = max(math.expm1(float(row[1].item())), 0.0)
        rows.append(
            {
                "event_index": row_index,
                "code_id": code_id,
                "icd10_code": id_to_code.get(code_id, f"ID_{code_id}"),
                "days_before_index": days_ago,
                "years_before_index": days_ago / 365.25,
                "recent_1yr": bool(row[2].item() > 0.5),
                "recent_5yr": bool(row[3].item() > 0.5),
            }
        )
    return rows


def omics_feature_manifest(
    modality: str,
    feature_names: Sequence[str],
    values: torch.Tensor,
    mean_std_map: Mapping[str, Mapping[str, float]] | None = None,
    feature_name_map: Mapping[str, str] | None = None,
    protein_name_map: Mapping[str, str] | None = None,
) -> list[dict]:
    """Map omics vector positions back to source columns and display values."""
    if values.ndim != 1:
        raise ValueError(f"Expected one-dimensional omics vector, got {values.shape}")
    if len(feature_names) != values.numel():
        raise ValueError(
            f"Feature/value length mismatch: {len(feature_names)} != {values.numel()}"
        )
    mean_std_map = mean_std_map or {}
    feature_name_map = feature_name_map or {}
    protein_name_map = protein_name_map or {}
    cpu_values = values.detach().cpu()
    rows = []
    for index, (name, value_tensor) in enumerate(zip(feature_names, cpu_values)):
        model_value = float(value_tensor)
        observed = math.isfinite(model_value)
        stats = mean_std_map.get(name, {})
        mean = stats.get("mean")
        std = stats.get("std")
        original_value = None
        if observed:
            if mean is not None and std is not None:
                original_value = model_value * float(std) + float(mean)
            else:
                original_value = model_value
        feature_id = None
        if modality == "metabolomics":
            match = re.match(r"^met_(\d+)_", name)
            if match:
                feature_id = match.group(1)
                display_name = feature_name_map.get(
                    feature_id, f"UKB metabolomics field {feature_id}"
                )
            else:
                display_name = clean_feature_name(name)
        elif modality == "proteomics":
            match = re.match(r"^prot_\d+_(\d+)$", name)
            if match:
                feature_id = match.group(1)
                display_name = protein_name_map.get(
                    feature_id, f"Olink protein ID {feature_id}"
                )
            else:
                display_name = clean_feature_name(name)
        else:
            display_name = clean_feature_name(name)
        rows.append(
            {
                "modality": modality,
                "feature_index": index,
                "feature_id": feature_id,
                "source_column": name,
                "feature": display_name,
                "observed": observed,
                "model_value": model_value if observed else None,
                "original_value": original_value,
                "training_mean": float(mean) if mean is not None else None,
                "training_std": float(std) if std is not None else None,
            }
        )
    return rows
