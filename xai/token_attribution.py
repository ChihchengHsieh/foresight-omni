from __future__ import annotations

import math
from typing import Iterable

import torch

from dataset.questionnaire import QUESTIONNAIRE_FIELD_IDS, QUESTIONNAIRE_FIELD_NAMES
from dataset.universal_image import MODALITIES_TO_COLS


DISPLAY_NAMES = {
    "instance_age_at_time": "Age",
    "patient_gender": "Sex",
    "patient_ancestry": "Ethnic background",
    "instance_height_cm": "Height",
    "instance_weight_kg": "Weight",
    "instance_body_mass_index_bmi": "Body mass index",
    "instance_waist_circumference_cm": "Waist circumference",
    "instance_hip_circumference_cm": "Hip circumference",
    "instance_systolic_bp": "Systolic blood pressure",
    "instance_diastolic_bp": "Diastolic blood pressure",
    "instance_pulse_rate": "Pulse rate",
    # These legacy column names do not match the UKB fields that populate them.
    # Keep the model-facing names unchanged for checkpoint compatibility, but
    # always expose the source-field meaning in tables and XAI figures.
    "instance_alcohol_intake_frequency": "Alcohol drinker status",
    "instance_alcohol_consumption_unitsweek": "Alcohol intake frequency",
    "instance_fruit_intake_portionsday": "Fresh fruit intake",
    "instance_vegetable_intake_portionsday": "Dried fruit intake",
    "instance_poultry_intake": "Beef intake",
}


LEGACY_DISPLAY_NAME_CORRECTIONS = {
    "Alcohol Intake Frequency": "Alcohol drinker status",
    "Alcohol Consumption Unitsweek": "Alcohol intake frequency",
    "Fruit Intake Portionsday": "Fresh fruit intake",
    "Vegetable Intake Portionsday": "Dried fruit intake",
    "Poultry Intake": "Beef intake",
    "Physical Activity Met Minswk": "Moderate physical activity frequency",
}


def canonical_feature_display_name(name: str) -> str:
    """Return the source-field meaning for model and legacy XAI labels."""
    value = str(name)
    if value in DISPLAY_NAMES:
        return DISPLAY_NAMES[value]
    return LEGACY_DISPLAY_NAME_CORRECTIONS.get(value, value)


def clean_feature_name(name: str) -> str:
    corrected = canonical_feature_display_name(name)
    if corrected != name:
        return corrected
    for prefix in ("instance_", "patient_"):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.replace("_", " ").strip().title()


def token_names(modality: str, count: int, dataset=None) -> list[str]:
    """Return stable human-facing names for encoder content tokens."""
    if modality == "fundus_image":
        side = int(math.isqrt(count))
        if side * side == count:
            return [
                f"Fundus CNN spatial cell row {i // side + 1}, column {i % side + 1}"
                for i in range(count)
            ]
    if modality == "oct_image":
        return [f"OCT latent token {i + 1}" for i in range(count)]
    if modality == "questionnaire" and count == len(QUESTIONNAIRE_FIELD_IDS):
        return [
            f"{QUESTIONNAIRE_FIELD_NAMES[field_id]} (UKB {field_id})"
            for field_id in QUESTIONNAIRE_FIELD_IDS
        ]
    if modality == "demographics" and count == 3:
        return ["Age", "Sex", "Ethnic background"]
    if modality in MODALITIES_TO_COLS and count == len(MODALITIES_TO_COLS[modality]):
        return [clean_feature_name(name) for name in MODALITIES_TO_COLS[modality]]
    if modality == "prs" and dataset is not None:
        columns = list(getattr(dataset, "prs_cols", []))
        if len(columns) == count:
            return [clean_feature_name(name) for name in columns]
    if modality == "metabolomics" and count == 1:
        return ["Grouped metabolomics representation"]
    if modality == "proteomics" and count == 1:
        return ["Grouped proteomics representation"]
    if modality == "clinical_history" and count == 1:
        return ["ICD-10 history attention summary"]
    return [f"{clean_feature_name(modality)} token {i + 1}" for i in range(count)]


def encoded_token_manifest(encoded: dict[str, torch.Tensor], dataset=None) -> list[dict]:
    manifest = []
    for modality, tokens in encoded.items():
        names = token_names(modality, int(tokens.shape[0]), dataset=dataset)
        for token_index, name in enumerate(names):
            manifest.append(
                {
                    "modality": modality,
                    "token_index": token_index,
                    "feature": name,
                }
            )
    return manifest


def masked_token_variants(
    encoded: dict[str, torch.Tensor], manifest: Iterable[dict]
) -> list[dict[str, torch.Tensor]]:
    variants = []
    for item in manifest:
        modality = item["modality"]
        token_index = int(item["token_index"])
        variant = dict(encoded)
        changed = encoded[modality].clone()
        changed[token_index].zero_()
        variant[modality] = changed
        variants.append(variant)
    return variants


def drop_modalities(
    encoded: dict[str, torch.Tensor]
) -> list[tuple[str, dict[str, torch.Tensor]]]:
    return [
        (modality, {key: value for key, value in encoded.items() if key != modality})
        for modality in encoded
    ]
