from __future__ import annotations
import torch
import torch.nn as nn
import pandas as pd
import math
from typing import List, Optional, Tuple, Callable
from torch.utils.data import Dataset, DataLoader
from .paths import *
import logging
from .aug import get_default_aug, get_oct_volume_aug, select_oct_slice_indices
import numpy as np
import json
import io
import zipfile
import time
from datetime import datetime
from utils.data import get_split_list
import os
from dataset.fid import PRS_FID_TO_NAME
from dateutil.relativedelta import relativedelta


def canonicalize_fundus_cache_key(path: object) -> str:
    """Return a stable cache key across optionally aliased data mounts."""
    normalized = os.path.normpath(str(path))
    alias_spec = os.environ.get("FORESIGHT_OMNI_PATH_ALIAS", "")
    if "=" in alias_spec:
        source_prefix, canonical_prefix = alias_spec.split("=", 1)
        source_prefix = os.path.normpath(source_prefix)
        canonical_prefix = os.path.normpath(canonical_prefix)
        if normalized == source_prefix or normalized.startswith(source_prefix + os.sep):
            suffix = normalized[len(source_prefix) :].lstrip(os.sep)
            return os.path.join(canonical_prefix, suffix)
    return normalized
from PIL import Image, ImageDraw
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import re
from typing import Dict, List, Tuple
from datetime import timedelta
from dataset.questionnaire import (
    QUESTIONNAIRE_FIELD_IDS,
    QuestionnaireTensorizer,
    parse_questionnaire_record,
)


_HORIZON_RE = re.compile(r"^has_(?P<disease>.+?)_in_(?P<years>\d+)_years$")
_INCIDENT_BY_RE = re.compile(
    r"^incident_(?P<disease>.+?)_by_(?P<years>\d+)_years$"
)


def add_years_fraction(dt, years: float):
    if years is None or (isinstance(years, float) and math.isnan(years)):
        return None

    # handle numpy scalars etc.
    years = float(years)

    yrs_int = int(math.floor(years))
    yrs_frac = years - yrs_int

    out = dt + relativedelta(years=yrs_int)

    # convert remaining fraction to days (choose your convention)
    if yrs_frac:
        out = out + timedelta(days=yrs_frac * 365.25)
    return out


def build_year_to_importance_linear(
    *,
    min_year: int = 0,
    max_year: int = 15,
    base_importance: float = 0.3,  # importance at year=0
    max_importance: float = 3.0,  # cap for long horizons
) -> Dict[int, float]:
    """
    Build a monotonic importance mapping from year -> importance.

    Example (0–15):
        0  -> 0.3
        5  -> ~1.2
        10 -> ~2.2
        15 -> 3.0
    """
    slope = (max_importance - base_importance) / max(max_year - min_year, 1)

    year_to_importance = {}
    for y in range(min_year, max_year + 1):
        w = base_importance + slope * (y - min_year)
        w = min(max_importance, max(base_importance, w))
        year_to_importance[y] = float(w)

    return year_to_importance


def build_label_importance_from_balance_cols(
    balance_label_cols: List[str],
    *,
    # default mapping: down-weight baseline prevalence, up-weight longer horizons
    year_to_importance: Dict[int, float] | None = None,
    # if a column doesn't match the pattern, either ignore it or give it a default weight
    default_importance_for_unmatched: float | None = None,
) -> Dict[str, float]:
    """
    Build label_importance dict from columns like:
        has_{disease}_in_{n}_years

    Returns:
        label_importance: dict[col_name -> weight]
    """
    if year_to_importance is None:
        year_to_importance = {
            0: 0.3,
            2: 1.0,
            5: 2.0,
            10: 3.0,
            13: 3.0,
        }

    label_importance: Dict[str, float] = {}

    for col in balance_label_cols:
        m = _HORIZON_RE.match(col)
        if not m:
            if default_importance_for_unmatched is not None:
                label_importance[col] = float(default_importance_for_unmatched)
            # else ignore unmatched columns
            continue

        y = int(m.group("years"))

        # If the horizon isn't in the mapping, you can:
        # 1) use nearest lower defined horizon, or
        # 2) default to 1.0
        if y in year_to_importance:
            w = year_to_importance[y]
        else:
            # nearest lower horizon fallback
            lower = [k for k in year_to_importance.keys() if k <= y]
            w = year_to_importance[max(lower)] if lower else 1.0

        label_importance[col] = float(w)

    return label_importance


@dataclass
class MultiLabelBalanceConfig:
    """
    Configuration for multi-label balanced sampling.

    This controls how per-sample sampling weights are computed when
    balancing multiple binary labels (e.g. has_{disease}_in_{n}_years)
    simultaneously.
    """

    # How to aggregate contributions from multiple labels:
    # - "sum":    sum inverse-frequency weights across labels
    #             (more aggressive, closer to per-label 1:1 balancing)
    # - "mean":   average weights across valid labels
    #             (more stable, recommended for multi-task training)
    aggregation: str = "mean"

    # Laplace smoothing term added to class counts when computing
    # inverse-frequency weights:
    #   w_pos = 1 / (n_pos + smooth)
    #   w_neg = 1 / (n_neg + smooth)
    #
    # - smooth > 0 prevents extremely large weights when a class is rare
    # - smooth = 0 gives pure inverse-frequency weighting (not recommended)
    smooth: float = 1.0

    # Clamp the final per-sample sampling weights into this range.
    # This is critical to prevent a few rare samples from dominating
    # the sampler.
    min_weight: float = 1e-3
    max_weight: float = 50.0

    # If a label has fewer than this many positive or negative samples,
    # it will be ignored entirely when computing sampling weights.
    # This avoids unstable weights caused by extremely rare labels.
    min_count_per_class: int = 10

    # After all weights are computed, normalize them so that
    # the mean weight equals 1.0.
    # This keeps the overall sampling scale stable and makes debugging easier.
    normalize_mean_to_one: bool = True

    # If a sample has no valid labels (all labels are None/NaN),
    # assign this default weight.
    #
    # - 0.0   => sample is effectively excluded from sampling
    # - small >0 value can be used if you want to keep such samples
    default_weight_if_all_missing: float = 0.0


def build_multi_label_balanced_weights(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    *,
    label_importance: Optional[Dict[str, float]] = None,
    cfg: MultiLabelBalanceConfig = MultiLabelBalanceConfig(),
) -> torch.DoubleTensor:
    """
    Multi-label balanced sampling weights by summing inverse-frequency weights.

    - Only uses rows where label is definitely True/False for that label.
    - None/NaN contributes 0 for that label.
    - Each label contributes alpha[label] * inv_count(class).
    - Final weights are clamped & normalized.

    Returns:
        weights: torch.double shape [N]
    """
    if cfg.aggregation not in ("sum", "mean"):
        raise ValueError("cfg.aggregation must be 'sum' or 'mean'")

    N = len(df)
    w = np.zeros(N, dtype=np.float64)
    cnt_valid = np.zeros(
        N, dtype=np.float64
    )  # how many labels contributed for each sample

    label_importance = label_importance or {}

    for col in label_cols:
        if col not in df.columns:
            continue

        alpha = float(label_importance.get(col, 1.0))

        s = df[col]
        valid = s.notna()
        if valid.sum() == 0:
            continue

        # only definite True/False
        y = s[valid].astype(bool).to_numpy()

        n_pos = int(y.sum())
        n_neg = int((~y).sum())

        # skip degenerate labels
        if n_pos < cfg.min_count_per_class or n_neg < cfg.min_count_per_class:
            continue

        # inverse freq with smoothing
        # (bigger counts -> smaller weight)
        w_pos = alpha / (n_pos + cfg.smooth)
        w_neg = alpha / (n_neg + cfg.smooth)

        add = np.where(y, w_pos, w_neg).astype(np.float64)

        idx = np.flatnonzero(valid.to_numpy())
        w[idx] += add
        cnt_valid[idx] += 1.0

    # handle samples with no valid labels
    no_valid = cnt_valid == 0
    w[no_valid] = float(cfg.default_weight_if_all_missing)

    # optional: mean aggregation (more stable than sum)
    if cfg.aggregation == "mean":
        denom = np.maximum(cnt_valid, 1.0)
        w = w / denom

    # clamp to avoid extreme weights
    w = np.clip(w, cfg.min_weight, cfg.max_weight)

    # normalize (mean=1) for nicer scale
    if cfg.normalize_mean_to_one:
        m = w.mean()
        if m > 0:
            w = w / m

    return torch.as_tensor(w, dtype=torch.double)


PRS_COLS = [
    "patient_Enhanced PRS for primary open angle glaucoma (POAG)",
    "patient_Enhanced PRS for alzheimer's disease (AD)",
    "patient_Enhanced PRS for parkinson's disease (PD)",
    "patient_Enhanced PRS for multiple sclerosis (MS)",
    "patient_Enhanced PRS for type 2 diabetes (T2D)",
    "patient_Enhanced PRS for cardiovascular disease (CVD)",
]

ANTHROPOMETRICS_COLS = [
    # "instance_height_cm", # Most peoploe don't have this
    "instance_weight_kg",
    "instance_body_mass_index_bmi",
    "instance_waist_circumference_cm",
    "instance_hip_circumference_cm",
    "instance_comparative_body_size_at_age_10",
]

# Grouped clinical representation used by the expanded 102-field model. Keep
# this separate from the legacy anthropometrics modality so older checkpoints
# retain their established five-token input shape and field ordering.
DEMOGRAPHICS_COLS = [
    "instance_age_at_time",
    "patient_gender",
    "patient_ancestry",
]

ANTHROPOMETRICS_CORE_COLS = [
    "instance_height_cm",
    "instance_weight_kg",
    "instance_body_mass_index_bmi",
    "instance_waist_circumference_cm",
    "instance_hip_circumference_cm",
]

FAMILY_HISTORY_COLS = [
    # Father
    "instance_father_heart_disease",
    "instance_father_stroke",
    "instance_father_lung_cancer",
    "instance_father_bowel_cancer",
    "instance_father_breast_cancer",
    "instance_father_chronic_bronchitis_emphysema",
    "instance_father_high_blood_pressure",
    "instance_father_diabetes",
    "instance_father_alzheimers_dementia",
    "instance_father_parkinsons_disease",
    "instance_father_severe_depression",
    "instance_father_prostate_cancer",
    "instance_father_hip_fracture",
    "instance_father_prefer_not_to_answer",
    # Mother
    "instance_maternal_heart_disease",
    "instance_maternal_stroke",
    "instance_maternal_lung_cancer",
    "instance_maternal_bowel_cancer",
    "instance_maternal_breast_cancer",
    "instance_maternal_chronic_bronchitis_emphysema",
    "instance_maternal_high_blood_pressure",
    "instance_maternal_diabetes",
    "instance_maternal_alzheimers_dementia",
    "instance_maternal_parkinsons_disease",
    "instance_maternal_severe_depression",
    "instance_maternal_prostate_cancer",
    "instance_maternal_hip_fracture",
    "instance_maternal_prefer_not_to_answer",
    # Siblings
    "instance_sibling_heart_disease",
    "instance_sibling_stroke",
    "instance_sibling_lung_cancer",
    "instance_sibling_bowel_cancer",
    "instance_sibling_breast_cancer",
    "instance_sibling_chronic_bronchitis_emphysema",
    "instance_sibling_high_blood_pressure",
    "instance_sibling_diabetes",
    "instance_sibling_alzheimers_dementia",
    "instance_sibling_parkinsons_disease",
    "instance_sibling_severe_depression",
    "instance_sibling_prostate_cancer",
    "instance_sibling_hip_fracture",
    "instance_sibling_prefer_not_to_answer",
]

PRINCIPAL_COMPONENT_COLS = [f"patient_pc_{i}" for i in range(1, 11)]

ANCESTRY_COLS = ["patient_ancestry"]

# Harmonized ancestry codes:
#   0 = White / Caucasian
#   1 = Asian (incl. Chinese)
#   2 = Black / African
#   3 = Mixed / Hispanic
#   4 = Other
#   NaN = Unknown / prefer not to answer
UKB_ANCESTRY_MAP = {
    # White
    1: 0, 1001: 0, 1002: 0, 1003: 0,
    # Asian or Asian British (incl. Chinese UKB=5)
    3: 1, 3001: 1, 3002: 1, 3003: 1, 3004: 1, 5: 1,
    # Black or Black British
    4: 2, 4001: 2, 4002: 2, 4003: 2,
    # Mixed / Hispanic
    2: 3, 2001: 3, 2002: 3, 2003: 3, 2004: 3,
    # Other
    6: 4,
    # Unknown → NaN (not included; unmapped keys return NaN)
}

CLSA_ANCESTRY_MAP = {
    1: 0,  # Caucasian → White
    2: 1,  # Asian
    3: 2,  # African → Black
    4: 3,  # Hispanic → Mixed
    5: 4,  # Other
    # Negative sentinel values → NaN (not included; unmapped keys return NaN)
}

LIFESTYLE_COLS = [
    "instance_smoking_status",
    "instance_packyears_of_smoking",
    "instance_alcohol_intake_frequency",
    "instance_alcohol_consumption_unitsweek",
    "instance_physical_activity_met_minswk",
    "instance_coffee_intake_cupsday",
    "instance_tea_intake_cupsday",
    "instance_salt_added_to_food_yesno",
    "instance_fruit_intake_portionsday",
    "instance_vegetable_intake_portionsday",
    "instance_selfrated_health",
    "instance_sleep_duration_hoursnight",
    "instance_type_of_milk_usually_consumed",
    "instance_cheese_intake",
    "instance_processed_meat_intake",
    "instance_poultry_intake",
    # "instance_time_outdoors_in_winter",
    # "instance_time_outdoors_in_summer",
]

MENTAL_HEALTH_COLS = [
    "instance_seen_doctor_nerves_anxiety_tension_or_depression",
    "instance_ever_depressed_for_a_whole_week",
    "instance_bipolar_and_major_depression_status",
]

SOCIOECONOMIC_COLS = [
    "instance_qualifications_college_or_university_degree",
    "instance_qualifications_a_levels_as_levels_or_equivalent",
    "instance_qualifications_o_levels_gcse_or_equivalent",
    "instance_qualifications_nvq_or_hnd_or_hnc_or_equivalent",
    "instance_qualifications_other_professional_qualifications",
    "instance_qualifications_prefer_not_to_answer",
    "instance_current_employment_paid_or_self_employed",
    "instance_current_employment_retired",
    "instance_current_employment_looking_after_home_and_or_family",
    "instance_current_employment_unable_to_work_sickness_or_disability",
    "instance_current_employment_unemployed",
    "instance_current_employment_unpaid_or_voluntary_work",
    "instance_current_employment_full_or_part_time_student",
    "instance_current_employment_prefer_not_to_answer",
]

VITALS_COLS = [
    "instance_systolic_bp",
    "instance_diastolic_bp",
    "instance_pulse_rate",
]

MEDICATIONS_COLS = [
    "instance_med_cholesterol",
    "instance_med_blood_pressure",
    "instance_med_diabetes",
    "instance_med_hormone_replacement",
    "instance_med_oral_contraceptive",
    "instance_med_prefer_not_to_answer",
]

INCIDENT_CVD_MEDICATIONS_COLS = [
    "instance_cvd_med",
    "instance_t2d_med",
]

MODALITIES_TO_COLS = {
    "prs": PRS_COLS,
    "anthropometrics": ANTHROPOMETRICS_COLS,
    "anthropometrics_core": ANTHROPOMETRICS_CORE_COLS,
    "demographics": DEMOGRAPHICS_COLS,
    "family_history": FAMILY_HISTORY_COLS,
    "principal_components": PRINCIPAL_COMPONENT_COLS,
    "lifestyle": LIFESTYLE_COLS,
    "mental_health": MENTAL_HEALTH_COLS,
    "socioeconomic": SOCIOECONOMIC_COLS,
    "vitals": VITALS_COLS,
    "medications": MEDICATIONS_COLS,
    "incident_cvd_medications": INCIDENT_CVD_MEDICATIONS_COLS,
    "ancestry": ANCESTRY_COLS,
}

MODALITIES_TO_LEN = {
    "prs": len(PRS_COLS),
    "anthropometrics": len(ANTHROPOMETRICS_COLS),
    "anthropometrics_core": len(ANTHROPOMETRICS_CORE_COLS),
    "demographics": len(DEMOGRAPHICS_COLS),
    "family_history": len(FAMILY_HISTORY_COLS),
    "principal_components": len(PRINCIPAL_COMPONENT_COLS),
    "lifestyle": len(LIFESTYLE_COLS),
    "mental_health": len(MENTAL_HEALTH_COLS),
    "socioeconomic": len(SOCIOECONOMIC_COLS),
    "vitals": len(VITALS_COLS),
    "medications": len(MEDICATIONS_COLS),
    "incident_cvd_medications": len(INCIDENT_CVD_MEDICATIONS_COLS),
    "ancestry": len(ANCESTRY_COLS),
}

ANTHROPOMETRICS_MODALITIES = [
    "height_cm",
    "weight_kg",
    "body_mass_index_bmi",
    "waist_circumference_cm",
    "hip_circumference_cm",
    "comparative_body_size_at_age_10",
]

FAMILY_HISTORY_MODALITIES = [
    # Father
    "father_heart_disease",
    "father_stroke",
    "father_lung_cancer",
    "father_bowel_cancer",
    "father_breast_cancer",
    "father_chronic_bronchitis_emphysema",
    "father_high_blood_pressure",
    "father_diabetes",
    "father_alzheimers_dementia",
    "father_parkinsons_disease",
    "father_severe_depression",
    "father_prostate_cancer",
    "father_hip_fracture",
    "father_prefer_not_to_answer",
    # Mother
    "maternal_heart_disease",
    "maternal_stroke",
    "maternal_lung_cancer",
    "maternal_bowel_cancer",
    "maternal_breast_cancer",
    "maternal_chronic_bronchitis_emphysema",
    "maternal_high_blood_pressure",
    "maternal_diabetes",
    "maternal_alzheimers_dementia",
    "maternal_parkinsons_disease",
    "maternal_severe_depression",
    "maternal_prostate_cancer",
    "maternal_hip_fracture",
    "maternal_prefer_not_to_answer",
    # Siblings
    "sibling_heart_disease",
    "sibling_stroke",
    "sibling_lung_cancer",
    "sibling_bowel_cancer",
    "sibling_breast_cancer",
    "sibling_chronic_bronchitis_emphysema",
    "sibling_high_blood_pressure",
    "sibling_diabetes",
    "sibling_alzheimers_dementia",
    "sibling_parkinsons_disease",
    "sibling_severe_depression",
    "sibling_prostate_cancer",
    "sibling_hip_fracture",
    "sibling_prefer_not_to_answer",
]

PRINCIPAL_COMPONENT_MODALITIES = [f"patient_pc_{i}" for i in range(1, 11)]

LIFESTYLE_MODALITIES = [
    "smoking_status",
    "packyears_of_smoking",
    "alcohol_intake_frequency", # previous, never, current
    "alcohol_consumption_unitsweek", # categorical
    "physical_activity_met_minswk",
    "coffee_intake_cupsday",
    "tea_intake_cupsday",
    "salt_added_to_food_yesno",
    "fruit_intake_portionsday",
    "vegetable_intake_portionsday",
    "selfrated_health",
    "sleep_duration_hoursnight",
    "type_of_milk_usually_consumed",
    "cheese_intake",
    "processed_meat_intake",
    "poultry_intake",
]

MENTAL_HEALTH_MODALITIES = [
    "seen_doctor_nerves_anxiety_tension_or_depression",
    "ever_depressed_for_a_whole_week",
    "bipolar_and_major_depression_status",
]

SOCIOECONOMIC_MODALITIES = [
    "qualifications_college_or_university_degree",
    "qualifications_a_levels_as_levels_or_equivalent",
    "qualifications_o_levels_gcse_or_equivalent",
    "qualifications_nvq_or_hnd_or_hnc_or_equivalent",
    "qualifications_other_professional_qualifications",
    "qualifications_prefer_not_to_answer",
    "current_employment_paid_or_self_employed",
    "current_employment_retired",
    "current_employment_looking_after_home_and_or_family",
    "current_employment_unable_to_work_sickness_or_disability",
    "current_employment_unemployed",
    "current_employment_unpaid_or_voluntary_work",
    "current_employment_full_or_part_time_student",
    "current_employment_prefer_not_to_answer",
]

VITALS_MODALITIES = [
    "systolic_bp",
    "diastolic_bp",
    "pulse_rate",
]

import re
import cv2


def _clahe_rgb_rgbin(arr_uint8, clip=2.0, tiles=(8, 8)):
    """CLAHE on L channel; assumes input is RGB, returns RGB."""
    lab = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    L2 = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tiles).apply(L)
    lab2 = cv2.merge((L2, A, B))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def _gray_world(arr_uint8, eps=1e-6):
    """Simple gray-world white balance; RGB in → RGB out."""
    x = arr_uint8.astype(np.float32)
    m = x.reshape(-1, 3).mean(axis=0) + eps
    x *= (m.mean() / m)[None, None, :]
    return np.clip(x, 0, 255).astype(np.uint8)


def _lab_ab_neutralize_rgbin(arr_uint8, target_a=0.0, target_b=0.0, strength=0.75):
    """Shift LAB a*,b* toward targets; RGB in → RGB out."""
    lab = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    A -= strength * (A.mean() - target_a)
    B -= strength * (B.mean() - target_b)
    lab2 = cv2.merge(
        (np.clip(L, 0, 255), np.clip(A, 0, 255), np.clip(B, 0, 255))
    ).astype(np.uint8)
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


# ---------- helpers ----------


def _gray_world_white_balance(arr_uint8, eps=1e-6):
    """Gray-world white balance: equalize per-channel means."""
    x = arr_uint8.astype(np.float32)
    means = x.reshape(-1, 3).mean(axis=0) + eps
    scale = means.mean() / means
    x *= scale[None, None, :]
    return np.clip(x, 0, 255).astype(np.uint8)


def _clahe_L(arr_uint8, clip=2.0, tiles=(8, 8)):
    """CLAHE on L channel in LAB space (illumination + local contrast)."""
    lab = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=tiles)
    L2 = clahe.apply(L)
    lab2 = cv2.merge((L2, A, B))
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def _lab_neutralize_chroma_aware(
    arr_uint8,
    target_a=0.0,
    target_b=0.0,
    strength=0.5,
    chroma_floor=10.0,
    chroma_soft=8.0,
):
    """
    Shift LAB a*, b* toward targets but protect high-chroma pixels (keeps vessels/disc color).
    strength in [0,1]. Larger chroma_floor/soft → more protection (less desaturation).
    """
    lab = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)

    a_mean, b_mean = float(A.mean()), float(B.mean())
    dA = a_mean - target_a  # global cast direction
    dB = b_mean - target_b

    C = np.sqrt(A**2 + B**2)  # chroma magnitude
    # mask ∈ (0,1]: 1 = full correction (low chroma), →0 for high chroma (protect color)
    mask = 1.0 / (1.0 + np.exp((C - chroma_floor) / max(chroma_soft, 1e-6)))

    A2 = A - strength * mask * dA
    B2 = B - strength * mask * dB

    lab2 = cv2.merge(
        (np.clip(L, 0, 255), np.clip(A2, 0, 255), np.clip(B2, 0, 255))
    ).astype(np.uint8)
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def _reinhard_lab_transfer(arr_uint8, ref_means, ref_stds, eps=1e-6):
    """Match LAB mean/std to a reference (e.g., UKB) : gentle, dataset-aligned color normalisation."""
    lab = cv2.cvtColor(arr_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    L, A, B = cv2.split(lab)
    out = []
    for ch, mu_ref, sd_ref in zip((L, A, B), ref_means, ref_stds):
        mu = float(ch.mean())
        sd = float(ch.std() + eps)
        ch2 = (ch - mu) / sd * sd_ref + mu_ref
        out.append(np.clip(ch2, 0, 255))
    lab2 = cv2.merge([o.astype(np.uint8) for o in out])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def has_icd10_eye_disease_except_poag(codes: list[str]) -> bool:
    """
    Returns True if any code is in H00–H59 (inclusive), except H401.
    """
    pattern = re.compile(r"^H([0-4]\d|5[0-9])", re.IGNORECASE)  # H00–H59
    for code in codes:
        if pattern.match(code):
            if not code.upper().startswith("H401"):
                return True
    return False


def column_normalisation(x, mean, std):
    if x is None:
        return None
    return (x - mean) / std


def normalise_dataframe_columns(df, columns, *, chunk_size=64):
    """Normalise columns from train-split statistics in bounded vectorized chunks."""
    columns = sorted(set(columns))
    mean_std_map = defaultdict(dict)
    train_mask = df["split"] == "train"
    for start in range(0, len(columns), int(chunk_size)):
        chunk = columns[start : start + int(chunk_size)]
        train_frame = df.loc[train_mask, chunk]
        means = train_frame.mean(axis=0)
        stds = train_frame.std(axis=0)
        for column in chunk:
            mean_std_map[column]["mean"] = means[column]
            mean_std_map[column]["std"] = stds[column]
        df.loc[:, chunk] = (df.loc[:, chunk] - means) / stds
    return mean_std_map


def add_fundus_auxiliary_target_columns(df):
    """Derive well-formed smoking targets while preserving missing labels."""
    if "instance_smoking_status" in df.columns:
        status = pd.to_numeric(df["instance_smoking_status"], errors="coerce")
        # UKB 20116: 0=never, 1=previous, 2=current. Negative response codes
        # are missing rather than negative examples.
        df["instance_current_smoker"] = status.map({0.0: 0.0, 1.0: 0.0, 2.0: 1.0})

    if "instance_packyears_of_smoking" in df.columns:
        packyears = pd.to_numeric(
            df["instance_packyears_of_smoking"], errors="coerce"
        ).where(lambda values: values >= 0)
        # The raw distribution is strongly right-skewed (observed max 182), so
        # use a stable regression target while retaining zero exposure.
        df["instance_log1p_packyears_of_smoking"] = np.log1p(packyears)

    return df


def get_datetime(date_str):
    if isinstance(date_str, datetime):
        return date_str
    if hasattr(pd, "Timestamp") and isinstance(date_str, pd.Timestamp):
        if pd.isna(date_str):
            return None
        return date_str.to_pydatetime()
    if isinstance(
        date_str, str
    ):  # and (not date_str is np.nan) and math.isnan(date_str):
        return datetime(*[int(x) for x in date_str.split("-")])
    return None


UKB_PROCERESS_DF_SAVED_PATH = str(DATA_ROOT / "ukb" / "foresight_omni_manifest.parquet")
UKB_METABOLOMICS_PARQUET_PATH = str(DATA_ROOT / "ukb" / "ukb_metabolomics.parquet")
UKB_PROTEOMICS_PARQUET_PATH = str(DATA_ROOT / "ukb" / "ukb_proteomics.parquet")
DEFAULT_ICD10_MASK_MAP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "reports",
    "icd10",
    "parent_descendants_complications_icd10_map.json",
)


def validate_eye_specific_iop_manifest(dataframe: pd.DataFrame) -> None:
    """Fail closed when an IOP-consuming task could pair the wrong eye."""
    required = {
        "instance_iop",
        "instance_iop_left",
        "instance_iop_right",
        "instance_iop_goldmann",
        "instance_iop_goldmann_left",
        "instance_iop_goldmann_right",
        "image_side",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(
            "IOP input requires an eye-specific v2 manifest; missing columns: "
            f"{missing}. Rebuild or use an *_eye_iop_v2.parquet manifest."
        )

    def mismatch_count(observed: pd.Series, expected: pd.Series) -> int:
        observed_num = pd.to_numeric(observed, errors="coerce")
        expected_num = pd.to_numeric(expected, errors="coerce")
        equal = (observed_num.isna() & expected_num.isna()) | np.isclose(
            observed_num.fillna(0.0),
            expected_num.fillna(0.0),
            rtol=0.0,
            atol=1e-7,
        )
        return int((~equal).sum())

    ocular = dataframe["image_side"].isin(["left", "right"])
    if ocular.any():
        side = dataframe.loc[ocular, "image_side"]
        expected_cc = dataframe.loc[ocular, "instance_iop_left"].where(
            side.eq("left"), dataframe.loc[ocular, "instance_iop_right"]
        )
        expected_goldmann = dataframe.loc[
            ocular, "instance_iop_goldmann_left"
        ].where(side.eq("left"), dataframe.loc[ocular, "instance_iop_goldmann_right"])
        cc_mismatches = mismatch_count(dataframe.loc[ocular, "instance_iop"], expected_cc)
        goldmann_mismatches = mismatch_count(
            dataframe.loc[ocular, "instance_iop_goldmann"], expected_goldmann
        )
        if cc_mismatches or goldmann_mismatches:
            raise ValueError(
                "Eye-specific IOP validation failed: "
                f"corneal_compensated_mismatches={cc_mismatches}, "
                f"goldmann_mismatches={goldmann_mismatches}."
            )

    assessment = dataframe["image_side"].isna()
    if assessment.any():
        expected_cc = dataframe.loc[
            assessment, ["instance_iop_left", "instance_iop_right"]
        ].max(axis=1, skipna=True)
        expected_goldmann = dataframe.loc[
            assessment,
            ["instance_iop_goldmann_left", "instance_iop_goldmann_right"],
        ].max(axis=1, skipna=True)
        cc_mismatches = mismatch_count(
            dataframe.loc[assessment, "instance_iop"], expected_cc
        )
        goldmann_mismatches = mismatch_count(
            dataframe.loc[assessment, "instance_iop_goldmann"], expected_goldmann
        )
        if cc_mismatches or goldmann_mismatches:
            raise ValueError(
                "Assessment IOP aggregation validation failed: "
                f"corneal_compensated_mismatches={cc_mismatches}, "
                f"goldmann_mismatches={goldmann_mismatches}."
            )


def validate_eye_specific_vcdr_manifest(dataframe: pd.DataFrame) -> None:
    """Fail closed when a VCDR-consuming task could pair the wrong eye."""
    required = {
        "instance_vcdr",
        "instance_vcdr_left",
        "instance_vcdr_right",
        "instance_vcdr_transformed",
        "instance_vcdr_transformed_left",
        "instance_vcdr_transformed_right",
        "image_side",
    }
    missing = sorted(required - set(dataframe.columns))
    if missing:
        raise ValueError(
            "VCDR input/label requires an eye-specific v3 manifest; missing columns: "
            f"{missing}. Rebuild or use an *_eye_iop_vcdr_v3.parquet manifest."
        )

    def mismatch_count(observed: pd.Series, expected: pd.Series) -> int:
        observed_num = pd.to_numeric(observed, errors="coerce")
        expected_num = pd.to_numeric(expected, errors="coerce")
        equal = (observed_num.isna() & expected_num.isna()) | np.isclose(
            observed_num.fillna(0.0),
            expected_num.fillna(0.0),
            rtol=0.0,
            atol=1e-7,
        )
        return int((~equal).sum())

    raw_values = pd.concat(
        [
            pd.to_numeric(dataframe["instance_vcdr_left"], errors="coerce"),
            pd.to_numeric(dataframe["instance_vcdr_right"], errors="coerce"),
        ],
        ignore_index=True,
    ).dropna()
    if not raw_values.between(0.0, 1.0).all():
        raise ValueError("Raw VCDR values must lie in [0, 1].")

    ocular = dataframe["image_side"].isin(["left", "right"])
    if ocular.any():
        side = dataframe.loc[ocular, "image_side"]
        expected_raw = dataframe.loc[ocular, "instance_vcdr_left"].where(
            side.eq("left"), dataframe.loc[ocular, "instance_vcdr_right"]
        )
        expected_transformed = dataframe.loc[
            ocular, "instance_vcdr_transformed_left"
        ].where(side.eq("left"), dataframe.loc[ocular, "instance_vcdr_transformed_right"])
        raw_mismatches = mismatch_count(
            dataframe.loc[ocular, "instance_vcdr"], expected_raw
        )
        transformed_mismatches = mismatch_count(
            dataframe.loc[ocular, "instance_vcdr_transformed"], expected_transformed
        )
        if raw_mismatches or transformed_mismatches:
            raise ValueError(
                "Eye-specific VCDR validation failed: "
                f"raw_mismatches={raw_mismatches}, "
                f"transformed_mismatches={transformed_mismatches}."
            )

    assessment = dataframe["image_side"].isna()
    if assessment.any():
        expected_raw = dataframe.loc[
            assessment, ["instance_vcdr_left", "instance_vcdr_right"]
        ].max(axis=1, skipna=True)
        expected_transformed = dataframe.loc[
            assessment,
            ["instance_vcdr_transformed_left", "instance_vcdr_transformed_right"],
        ].max(axis=1, skipna=True)
        raw_mismatches = mismatch_count(
            dataframe.loc[assessment, "instance_vcdr"], expected_raw
        )
        transformed_mismatches = mismatch_count(
            dataframe.loc[assessment, "instance_vcdr_transformed"],
            expected_transformed,
        )
        if raw_mismatches or transformed_mismatches:
            raise ValueError(
                "Assessment VCDR aggregation validation failed: "
                f"raw_mismatches={raw_mismatches}, "
                f"transformed_mismatches={transformed_mismatches}."
            )


class ImageLevelUKBUniversalDataset(Dataset):
    def __init__(
        self,
        image_size: int,
        split: str,
        possible_inputs: List[str] = [
            "fundus_image",
            "glaucoma_prs",
            "ad_prs",
            "pd_prs",
            "ms_prs",
            "stroke_prs",
            "diabetes_prs",
            "clinical-cat",
            "clinical-num",
            "genotype",
        ],
        possible_labels: List[str] = ["glaucoma"],
        label_cols: List[str] = ["instance_glaucoma_label"],
        transform: Optional[nn.Module] = None,
        balance_label_cols: List[str] = ["instance_glaucoma_label"],
        glaucoma_side_check: bool = False,  # See both as positive if no side is provided.
        see_no_side_as_both=True,  # If no side is provided, see it as both.
        quality_control: bool = True,  # Remove the cases that without great quality.
        include_self_report_label: bool = True,
        include_eye_problem_label: bool = True,
        progression_label_years: list = [0, 2, 5, 10, 13],
        use_survival_analysis_for_progression: bool = True,
        enhanced_aug: bool = True,
        no_aug: bool = False,
        fundus_aug_profile: str = "kim_enhanced",
        nucleotide_map={"A": 0, "T": 1, "C": 2, "G": 3, "0": 4, "N": 5},
        instances_with_genotype_only: bool = False,
        genotype_lifetime_prediction: bool = False,
        df_conditions: Optional[List[Callable]] = None,
        clinical_numerical_features: List[str] = [
            "instance_age_at_time",
            "instance_iop",
            "instance_systolic_bp",
            "instance_diastolic_bp",
            "instance_pulse_rate",
            "instance_height_cm",
            "instance_weight_kg",
            "instance_body_mass_index_bmi",
            "instance_waist_circumference_cm",
            "instance_hip_circumference_cm",
        ],
        clinical_categorical_features: List[str] = [
            "instance_comparative_body_size_at_age_10"
        ],
        progression_label_ignorant_label_years: int = 1,  # removing cases who can get the disease in n + y years. (set to None)
        positive_buffer_days: int = 90,
        label_date_name_to_disease: dict = {
            "patient_ad_date": "ad",
            "patient_pd_date": "pd",
            "patient_hd_date": "hd",
            "patient_ms_date": "ms",
            "patient_t2d_date": "t2d",
            "patient_cvd_date": "cvd",
            "patient_glaucoma_date": "glaucoma",
        },
        numerical_label_cols: list[str] = [],
        df_path: str = IMAGE_LEVEL_UNIVERSAL_SPREADSHEET_PATH,
        processed_df_path: str = UKB_PROCERESS_DF_SAVED_PATH,
        image_dir: str = FUNDUS_DIR,
        genotype_path: str = GENOTYPE_PATH,
        metabolomics_path: str = UKB_METABOLOMICS_PARQUET_PATH,
        proteomics_path: str = UKB_PROTEOMICS_PARQUET_PATH,
        omics_qc: bool = True,
        omics_missing_rate_threshold: float = 0.40,
        omics_min_std: float = 1e-6,
        omics_min_unique_values: int = 10,
        omics_winsor_lower_quantile: float = 0.005,
        omics_winsor_upper_quantile: float = 0.995,
        clinical_history_min_count: int = 10,
        clinical_history_max_len: int = 128,
        clinical_history_mask_targets: bool = True,
        clinical_history_mask_map_path: str = DEFAULT_ICD10_MASK_MAP_PATH,
        clinical_history_target_diseases: Optional[List[str]] = None,
        questionnaire_max_choices: int = 16,
        last_update: datetime = datetime(2024, 3, 1),
        # ------- Glaucoma Cleaning -------
        poag_only: bool = False,
        remove_self_report_eye_problem_in_negative: bool = True,
        remove_eye_icd10_not_poag: bool = False,
        remove_eye_icd10_not_poag_only_on_fales: bool = False,
        quality_control_only_on_false: bool = False,
        normalise_fundus_image: bool = False,
        fundus_cache_path: Optional[str] = None,
        fundus_cache_index_path: Optional[str] = None,
        fundus_cache_metadata_path: Optional[str] = None,
        fundus_require_cache: bool = False,
        fundus_cache_allow_resize: bool = False,
        algo_qc: bool = False,
        algo_qc_path: str = "data/ukb/algorithmic_qc.csv",
        oct_num_slices: int = 128,
        oct_image_size: Optional[int] = None,
        oct_aug_profile: str = "oct_clinical_v1",
        profile_timing: bool = False,
        smoke_test_max_rows_per_split: Optional[int] = None,
        smoke_test_seed: int = 42,
        external_binary_phenotype_path: Optional[str] = None,
        external_binary_phenotype_id_col: str = "IID",
        external_binary_phenotype_label_col: str = "phenotype",
        external_binary_target_disease: Optional[str] = None,
        external_binary_restrict_cohort: bool = True,
        external_binary_phenotype_date_col: Optional[str] = None,
        external_binary_instance_date_col: Optional[str] = None,
        external_prs_path: Optional[str] = None,
        external_prs_id_col: str = "IID",
        external_prs_score_col: str = "SCORE1_AVG",
        require_external_prs: bool = False,
        require_any_requested_input: bool = False,
        allow_legacy_iop_manifest: bool = False,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.image_size = image_size
        self.df_path = df_path
        self.processed_df_path = processed_df_path
        self.possible_inputs = possible_inputs
        self.possible_labels = possible_labels
        self.all_modalities = list(set(possible_inputs + possible_labels))
        self.label_cols = label_cols
        self.balance_label_cols = balance_label_cols
        self.split = split
        self.quality_control = quality_control
        self.glaucoma_side_check = glaucoma_side_check
        self.see_no_side_as_both = see_no_side_as_both
        self.include_self_report_label = include_self_report_label
        self.include_eye_problem_label = include_eye_problem_label
        self.progression_label_years = progression_label_years
        self.genotype_lifetime_prediction = genotype_lifetime_prediction
        self.label_date_to_disease = label_date_name_to_disease
        self.use_survival_analysis_for_progression = (
            use_survival_analysis_for_progression
        )
        self.progression_label_ignorant_label_years = (
            progression_label_ignorant_label_years
        )
        self.genotype_path = genotype_path
        self.metabolomics_path = metabolomics_path
        self.proteomics_path = proteomics_path
        self.omics_qc = omics_qc
        self.omics_missing_rate_threshold = omics_missing_rate_threshold
        self.omics_min_std = omics_min_std
        self.omics_min_unique_values = omics_min_unique_values
        self.omics_winsor_lower_quantile = omics_winsor_lower_quantile
        self.omics_winsor_upper_quantile = omics_winsor_upper_quantile
        self.clinical_history_min_count = clinical_history_min_count
        self.clinical_history_max_len = clinical_history_max_len
        self.clinical_history_mask_targets = clinical_history_mask_targets
        self.clinical_history_mask_map_path = clinical_history_mask_map_path
        self.clinical_history_target_diseases = (
            clinical_history_target_diseases
            if clinical_history_target_diseases is not None
            else self.__infer_target_diseases(possible_labels)
        )
        self.questionnaire_max_choices = questionnaire_max_choices
        self.instances_with_genotype_only = instances_with_genotype_only
        self.nucleotide_map = nucleotide_map
        self.clinical_numerical_features = clinical_numerical_features
        self.clinical_categorical_features = clinical_categorical_features
        self.numerical_label_cols = numerical_label_cols
        self.last_update = last_update
        self.quality_control_only_on_false = quality_control_only_on_false
        self.normalise_fundus_image = normalise_fundus_image
        self.fundus_cache_path = fundus_cache_path
        self.fundus_cache_index_path = fundus_cache_index_path
        self.fundus_cache_metadata_path = fundus_cache_metadata_path
        self.fundus_require_cache = bool(fundus_require_cache)
        self.fundus_cache_allow_resize = bool(fundus_cache_allow_resize)
        self._fundus_cache = None
        self._fundus_cache_index = None
        self._fundus_cache_shape = None
        self.__init_fundus_cache()
        self.oct_num_slices = oct_num_slices
        self.oct_image_size = oct_image_size or image_size
        self.oct_aug_profile = oct_aug_profile
        self.oct_transform = get_oct_volume_aug(
            split=split,
            profile=oct_aug_profile,
            no_aug=no_aug,
        )
        self.profile_timing = profile_timing
        self.smoke_test_max_rows_per_split = smoke_test_max_rows_per_split
        self.smoke_test_seed = int(smoke_test_seed)
        self.external_binary_phenotype_path = external_binary_phenotype_path
        self.external_binary_phenotype_id_col = external_binary_phenotype_id_col
        self.external_binary_phenotype_label_col = external_binary_phenotype_label_col
        self.external_binary_target_disease = external_binary_target_disease
        self.external_binary_restrict_cohort = bool(external_binary_restrict_cohort)
        self.external_binary_phenotype_date_col = external_binary_phenotype_date_col
        self.external_binary_instance_date_col = external_binary_instance_date_col
        self.external_prs_path = external_prs_path
        self.external_prs_id_col = external_prs_id_col
        self.external_prs_score_col = external_prs_score_col
        self.require_external_prs = bool(require_external_prs)
        self.require_any_requested_input = bool(require_any_requested_input)
        self.external_numeric_cols = []
        self._last_oct_profile = None
        self.transform = (
            get_default_aug(
                image_size,
                split,
                enhanced_aug,
                no_aug,
                profile=fundus_aug_profile,
            )
            if transform is None
            else transform
        )

        # ------- Glaucoma Cleaning -------
        self.poag_only = poag_only
        self.remove_self_report_eye_problem_in_negative = (
            remove_self_report_eye_problem_in_negative
        )
        self.remove_eye_icd10_not_poag = remove_eye_icd10_not_poag
        self.remove_eye_icd10_not_poag_only_on_fales = (
            remove_eye_icd10_not_poag_only_on_fales
        )
        self.algo_qc = algo_qc
        self.algo_qc_path = algo_qc_path
        self.positive_buffer_days = positive_buffer_days

        if not os.path.exists(self.processed_df_path):
            if self.processed_df_path != UKB_PROCERESS_DF_SAVED_PATH:
                raise FileNotFoundError(
                    "Requested registry-controlled processed dataframe does not "
                    f"exist: [{self.processed_df_path}]"
                )
            self.df = pd.read_csv(df_path, low_memory=False)
            self.mask_glaucoma_label()
            self.__init_survival_progression_labels()

            # save it
            logging.info("Saving processed UKB dataframe to saved path...")
            self.df.to_parquet(self.processed_df_path)

        else:
            logging.info(
                "Loading processed UKB dataframe from [%s]...",
                self.processed_df_path,
            )
            self.df = pd.read_parquet(self.processed_df_path)

        self.__init_requested_incident_labels()

        requires_iop = "iop" in self.possible_inputs or (
            "clinical-num" in self.possible_inputs
            and "instance_iop" in self.clinical_numerical_features
        )
        if requires_iop and not allow_legacy_iop_manifest:
            validate_eye_specific_iop_manifest(self.df)
        elif requires_iop:
            logging.warning(
                "Using a legacy non-eye-specific IOP manifest for an explicit historical reproduction run."
            )
        requires_vcdr = "vcdr" in self.possible_inputs or any(
            column.startswith("instance_vcdr")
            for column in self.clinical_numerical_features
            + self.numerical_label_cols
        )
        if requires_vcdr:
            validate_eye_specific_vcdr_manifest(self.df)

        self.df = add_fundus_auxiliary_target_columns(self.df)
        self.__merge_external_binary_target_and_prs_if_needed()
        self.__json_loads_cols()
        self.__hydrate_cached_clinical_history_if_needed()
        self.__init_questionnaire_if_needed()

        # self.df = self.df[:500]
        if "image_modality" in self.df.columns:
            if (
                "fundus_image" in self.all_modalities
                and "oct_image" not in self.all_modalities
            ):
                self.df = self.df[
                    self.df["image_modality"].fillna("fundus") == "fundus"
                ]
            elif (
                "oct_image" in self.all_modalities
                and "fundus_image" not in self.all_modalities
            ):
                oct_rows = self.df["image_modality"] == "oct"
                if oct_rows.any():
                    self.df = self.df[oct_rows]
                elif "oct_image_path" in self.df.columns:
                    logging.warning(
                        "No rows with image_modality == 'oct' found; using rows with "
                        "non-null oct_image_path for OCT-only dataset."
                    )
                    self.df = self.df[self.df["oct_image_path"].notna()]
                elif "has_oct_image" in self.df.columns:
                    logging.warning(
                        "No rows with image_modality == 'oct' or oct_image_path found; "
                        "using has_oct_image for OCT-only dataset."
                    )
                    self.df = self.df[self.df["has_oct_image"].fillna(False)]
                else:
                    self.df = self.df[oct_rows]

        self.metabolomics_cols = []
        self.proteomics_cols = []
        self.metabolomics_instance_cols = {}
        self.proteomics_instance_cols = {}
        self.__merge_omics_if_needed()
        self.__init_clinical_history_if_needed()

        # intialise prs cols
        self.prs_cols = []
        # map the dictionary in the df['patient_prs'] to the actually columns
        for v in PRS_FID_TO_NAME.values():
            self.df[f"patient_{v}"] = self.df["patient_prs"].apply(
                lambda x: x.get(v, None)
            )

            self.prs_cols.append(f"patient_{v}")

        # Map raw UKB ancestry codes to harmonized values
        if "patient_ancestry" in self.df.columns:
            self.df["patient_ancestry"] = pd.to_numeric(
                self.df["patient_ancestry"], errors="coerce"
            ).apply(
                lambda x: float(UKB_ANCESTRY_MAP[int(x)])
                if (not pd.isna(x) and int(x) in UKB_ANCESTRY_MAP)
                else float("nan")
            )

        if self.require_any_requested_input:
            self.__filter_rows_without_requested_inputs()
        self.__apply_omics_qc_if_needed()

        # End-to-end smoke tests need the real schemas and train-fitted vocabularies,
        # but do not need full-cohort numerical normalization three times. Retain a
        # deterministic eligible sample from every split before selecting this
        # dataset object's split.
        if self.smoke_test_max_rows_per_split is not None:
            cap = int(self.smoke_test_max_rows_per_split)
            if cap <= 0:
                raise ValueError("smoke_test_max_rows_per_split must be positive")
            sampled = []
            for split_offset, split_name in enumerate(("train", "val", "test")):
                split_df = self.df[self.df["split"] == split_name]
                if len(split_df) > cap:
                    split_df = split_df.sample(
                        n=cap,
                        random_state=self.smoke_test_seed + split_offset,
                    )
                sampled.append(split_df)
            self.df = pd.concat(sampled, axis=0).sort_index()
            logging.warning(
                "SMOKE TEST PREPROCESSING CAP ACTIVE: retained rows by split=%s",
                self.df["split"].value_counts().to_dict(),
            )

        # ---- Common Filtering ----
        # filter out the data that doesn't have genotype.
        if self.instances_with_genotype_only:
            self.df = self.df[self.df["patient_has_genotype"]]
        self.__set_categorical_clinical_features()
        if glaucoma_side_check:
            logging.info("Glaucoma Side Check")
            self.relabel_with_side_check(see_no_side_as_both)

        logging.info("Normalising numerical values...")
        direct_numeric_input_cols = [
            column
            for modality, column in {
                "age": "instance_age_at_time",
                "iop": "instance_iop",
                "vcdr": "instance_vcdr",
            }.items()
            if modality in self.possible_inputs
        ]
        normalisation_cols = sorted(set(
            self.clinical_numerical_features
            + direct_numeric_input_cols
            + self.prs_cols
            + self.numerical_label_cols
            + self.metabolomics_cols
            + self.proteomics_cols
            + self.external_numeric_cols
        ))
        self.mean_std_map = normalise_dataframe_columns(
            self.df,
            normalisation_cols,
            chunk_size=64,
        )

        logging.info("Dataframe preprocessing...")
        # # quality control before mean and std calculated.
        if quality_control:
            # load algo quality df
            if self.algo_qc:
                algo_qc_df = pd.read_csv(self.algo_qc_path)
                if len(algo_qc_df) != len(self.df):
                    raise ValueError(
                        "Algorithmic QC rows must align one-to-one with the UKB manifest: "
                        f"received {len(algo_qc_df)} QC rows for {len(self.df)} images."
                    )
                self.df["algo_qc"] = ~algo_qc_df["is_bad"]
                self.df = self.df[(self.df["image_quality"] == 1) & self.df["algo_qc"]]
            else:
                # merge into self.df
                self.df = self.df[(self.df["image_quality"] == 1)]
            logging.info(f"Quality Control | After [{len(self.df)}]")

        ### Splitting
        self.__dataset_split()  # split after normalisation to avoid data leakage

        logging.info("Dataset Condition Applying...")
        if df_conditions is not None:
            for condition in df_conditions:
                original_len = len(self.df)
                self.df = self.df[condition(self.df)]
                removed_length = original_len - len(self.df)
                # print original and after length
                logging.info(
                    f"[UKB] | [{self.split}] | [{condition.__name__}] | Original instances: {original_len} | After instances: {len(self.df)} | Removed instances: {removed_length}"
                )

        logging.info("Dataset statistics calculating...")
        self.__validate_split_fundus_cache()
        logging.info(f"UKB Dataset [{self.split}] | Total: {len(self.df)}")

    def __init_fundus_cache(self) -> None:
        supplied = [
            self.fundus_cache_path,
            self.fundus_cache_index_path,
            self.fundus_cache_metadata_path,
        ]
        if not any(supplied):
            if self.fundus_require_cache:
                raise ValueError(
                    "--fundus_require_cache needs cache, index, and metadata paths"
                )
            return
        if not all(supplied):
            raise ValueError(
                "Fundus cache requires --fundus_cache_path, "
                "--fundus_cache_index_path, and --fundus_cache_metadata_path"
            )
        if self.normalise_fundus_image:
            raise ValueError(
                "The v1 fundus cache represents standard centre-crop preprocessing "
                "and cannot be combined with --normalise_fundus_image"
            )
        for path in supplied:
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Required fundus cache artifact is missing: {path}")

        with open(self.fundus_cache_metadata_path) as handle:
            metadata = json.load(handle)
        cache_image_size = int(metadata["image_size"])
        if cache_image_size != int(self.image_size) and not self.fundus_cache_allow_resize:
            raise ValueError(
                "Fundus cache image-size mismatch: "
                f"cache={cache_image_size} requested={self.image_size}; "
                "set --fundus_cache_allow_resize for an explicit resize"
            )
        if metadata.get("dtype") != "uint8" or int(metadata.get("channels", 0)) != 3:
            raise ValueError(f"Unsupported fundus cache metadata: {metadata}")
        rows = int(metadata["rows"])
        expected_bytes = rows * cache_image_size * cache_image_size * 3
        actual_bytes = os.path.getsize(self.fundus_cache_path)
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"Fundus cache size mismatch: expected={expected_bytes}, actual={actual_bytes}"
            )

        index_frame = pd.read_parquet(
            self.fundus_cache_index_path, columns=["image_path", "cache_index"]
        )
        if len(index_frame) != rows:
            raise ValueError(
                f"Fundus cache index rows={len(index_frame)} but metadata rows={rows}"
            )
        if index_frame["image_path"].duplicated().any():
            raise ValueError("Fundus cache index contains duplicate image paths")
        cache_indices = index_frame["cache_index"].astype(np.int64)
        if set(cache_indices.tolist()) != set(range(rows)):
            raise ValueError("Fundus cache indices are not a complete 0..rows-1 mapping")
        canonical_paths = index_frame["image_path"].map(canonicalize_fundus_cache_key)
        if canonical_paths.duplicated().any():
            raise ValueError(
                "Fundus cache index contains duplicate paths after mount-prefix "
                "canonicalization"
            )
        self._fundus_cache_index = dict(zip(canonical_paths, cache_indices))
        self._fundus_cache_shape = (rows, cache_image_size, cache_image_size, 3)
        logging.info(
            "Fundus uint8 cache enabled: rows=%d source_size=%d requested_size=%d path=%s",
            rows,
            cache_image_size,
            self.image_size,
            self.fundus_cache_path,
        )

    def __validate_split_fundus_cache(self) -> None:
        if self._fundus_cache_index is None or "fundus_image" not in self.possible_inputs:
            return
        paths = self.df["image_path"].dropna().map(canonicalize_fundus_cache_key)
        missing = [path for path in paths.unique() if path not in self._fundus_cache_index]
        if missing and self.fundus_require_cache:
            raise ValueError(
                f"Fundus cache is missing {len(missing)} paths in split={self.split}; "
                f"examples={missing[:5]}"
            )
        if missing:
            logging.warning(
                "Fundus cache missing %d paths in split=%s; these rows will use source files",
                len(missing),
                self.split,
            )

    def _get_fundus_cache(self):
        if self._fundus_cache is None:
            self._fundus_cache = np.memmap(
                self.fundus_cache_path,
                dtype=np.uint8,
                mode="r",
                shape=self._fundus_cache_shape,
            )
        return self._fundus_cache

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fundus_cache"] = None
        return state

    def __infer_target_diseases(self, possible_labels: List[str]) -> List[str]:
        diseases = set()
        for label in possible_labels:
            m = _HORIZON_RE.match(label)
            if m:
                diseases.add(m.group("disease"))
        return sorted(diseases)

    def __filter_rows_without_requested_inputs(self) -> None:
        """Retain rows with at least one real requested modality input."""
        mask = pd.Series(False, index=self.df.index)

        def add_columns(columns):
            nonlocal mask
            available = [column for column in columns if column in self.df.columns]
            if available:
                mask |= self.df[available].notna().any(axis=1)

        for modality in self.possible_inputs:
            if modality == "fundus_image":
                add_columns(["image_path"])
            elif modality == "oct_image":
                add_columns(["oct_image_path"])
            elif modality == "prs":
                add_columns(self.prs_cols)
            elif modality == "metabolomics":
                for instance, columns in self.metabolomics_instance_cols.items():
                    instance_mask = (
                        pd.to_numeric(self.df.get("instance_idx"), errors="coerce")
                        == int(instance)
                    )
                    mask |= instance_mask & self.df[columns].notna().any(axis=1)
            elif modality == "proteomics":
                for instance, columns in self.proteomics_instance_cols.items():
                    instance_mask = (
                        pd.to_numeric(self.df.get("instance_idx"), errors="coerce")
                        == int(instance)
                    )
                    mask |= instance_mask & self.df[columns].notna().any(axis=1)
            elif modality == "questionnaire":
                add_columns(["instance_questionnaire"])
            elif modality == "clinical_history":
                # An empty pre-index history is a valid observed absence; the
                # assessment date is required to define that representation.
                add_columns(["instance_assessment_centre_visit_date"])
            else:
                grouped_columns = {
                    "demographics": DEMOGRAPHICS_COLS,
                    "anthropometrics_core": ANTHROPOMETRICS_CORE_COLS,
                    "anthropometrics": ANTHROPOMETRICS_COLS,
                    "family_history": FAMILY_HISTORY_COLS,
                    "principal_components": PRINCIPAL_COMPONENT_COLS,
                    "lifestyle": LIFESTYLE_COLS,
                    "mental_health": MENTAL_HEALTH_COLS,
                    "socioeconomic": SOCIOECONOMIC_COLS,
                    "vitals": VITALS_COLS,
                    "medications": MEDICATIONS_COLS,
                    "incident_cvd_medications": INCIDENT_CVD_MEDICATIONS_COLS,
                    "ancestry": ANCESTRY_COLS,
                }.get(modality)
                if grouped_columns is not None:
                    add_columns(grouped_columns)
                elif modality in self.df.columns:
                    add_columns([modality])
                elif f"instance_{modality}" in self.df.columns:
                    add_columns([f"instance_{modality}"])

        removed = int((~mask).sum())
        self.df = self.df.loc[mask].copy()
        logging.info(
            "Requested-input eligibility filter retained %d rows and removed %d",
            len(self.df),
            removed,
        )
        if self.df.empty:
            raise ValueError(
                "No rows contain any requested modality input after eligibility filtering"
            )

    @staticmethod
    def __canonical_participant_ids(values: pd.Series) -> pd.Series:
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.astype("Int64").astype("string")

    def __merge_external_binary_target_and_prs_if_needed(self) -> None:
        if not self.external_binary_phenotype_path and not self.external_prs_path:
            return
        if "patient_eid" not in self.df.columns:
            raise KeyError("External phenotype/PRS merge requires patient_eid")

        participant_ids = self.__canonical_participant_ids(self.df["patient_eid"])

        if self.external_binary_phenotype_path:
            disease = self.external_binary_target_disease
            if not disease:
                raise ValueError(
                    "--external_binary_target_disease is required with an external phenotype"
                )
            if self.progression_label_years != [0]:
                raise ValueError(
                    "External binary phenotype supports only --progression_label_years 0"
                )
            label_name = f"has_{disease}_in_0_years"
            if label_name not in self.possible_labels:
                raise ValueError(
                    f"External target {label_name!r} is not present in possible labels "
                    f"{self.possible_labels!r}"
                )
            if str(self.external_binary_phenotype_path).lower().endswith(
                (".parquet", ".pq")
            ):
                phenotype_columns = [
                    self.external_binary_phenotype_id_col,
                    self.external_binary_phenotype_label_col,
                ]
                if self.external_binary_phenotype_date_col:
                    phenotype_columns.append(self.external_binary_phenotype_date_col)
                phenotype = pd.read_parquet(
                    self.external_binary_phenotype_path,
                    columns=phenotype_columns,
                )
            else:
                phenotype = pd.read_csv(
                    self.external_binary_phenotype_path,
                    sep="\t",
                    dtype={self.external_binary_phenotype_id_col: "string"},
                )
            required = {
                self.external_binary_phenotype_id_col,
                self.external_binary_phenotype_label_col,
            }
            missing = required.difference(phenotype.columns)
            if missing:
                raise KeyError(f"External phenotype is missing columns: {sorted(missing)}")
            phenotype_ids = self.__canonical_participant_ids(
                phenotype[self.external_binary_phenotype_id_col]
            )
            labels = pd.to_numeric(
                phenotype[self.external_binary_phenotype_label_col], errors="coerce"
            )
            if phenotype_ids.duplicated().any():
                raise ValueError("External phenotype contains duplicate participant IDs")
            observed_labels = set(labels.dropna().unique().tolist())
            if not observed_labels.issubset({0, 1, 0.0, 1.0}):
                raise ValueError(
                    f"External phenotype must be binary 0/1, found {sorted(observed_labels)}"
                )
            if bool(self.external_binary_phenotype_date_col) != bool(
                self.external_binary_instance_date_col
            ):
                raise ValueError(
                    "External binary date matching requires both phenotype and "
                    "instance date columns"
                )
            if self.external_binary_phenotype_date_col:
                instance_date_col = self.external_binary_instance_date_col
                if instance_date_col not in self.df.columns:
                    raise KeyError(
                        f"External binary instance date column missing: {instance_date_col}"
                    )
                phenotype_dates = pd.to_datetime(
                    phenotype[self.external_binary_phenotype_date_col], errors="coerce"
                ).dt.normalize()
                instance_dates = pd.to_datetime(
                    self.df[instance_date_col], errors="coerce"
                ).dt.normalize()
                phenotype_keys = pd.MultiIndex.from_arrays(
                    [phenotype_ids, phenotype_dates]
                )
                if phenotype_keys.duplicated().any():
                    raise ValueError(
                        "External phenotype contains duplicate participant/date keys"
                    )
                label_map = pd.Series(labels.to_numpy(), index=phenotype_keys)
                instance_keys = pd.MultiIndex.from_arrays(
                    [participant_ids, instance_dates]
                )
                self.df[label_name] = label_map.reindex(instance_keys).to_numpy()
            else:
                label_map = pd.Series(labels.to_numpy(), index=phenotype_ids).to_dict()
                self.df[label_name] = participant_ids.map(label_map)
            if self.external_binary_restrict_cohort:
                before = len(self.df)
                self.df = self.df[self.df[label_name].notna()].copy()
                participant_ids = participant_ids.loc[self.df.index]
                logging.info(
                    "External binary target %s retained %d/%d image rows",
                    label_name,
                    len(self.df),
                    before,
                )
            else:
                logging.info(
                    "External binary target %s merged without restricting the shared cohort: "
                    "eligible_rows=%d total_rows=%d",
                    label_name,
                    int(self.df[label_name].notna().sum()),
                    len(self.df),
                )

        if self.external_prs_path:
            prs = pd.read_csv(
                self.external_prs_path,
                sep=r"\s+",
                dtype={self.external_prs_id_col: "string"},
            )
            required = {self.external_prs_id_col, self.external_prs_score_col}
            missing = required.difference(prs.columns)
            if missing:
                raise KeyError(f"External PRS is missing columns: {sorted(missing)}")
            prs_ids = self.__canonical_participant_ids(prs[self.external_prs_id_col])
            if prs_ids.duplicated().any():
                raise ValueError("External PRS contains duplicate participant IDs")
            scores = pd.to_numeric(prs[self.external_prs_score_col], errors="coerce")
            score_map = pd.Series(scores.to_numpy(), index=prs_ids).to_dict()
            column = "patient_dr_t2d_prs"
            self.df[column] = participant_ids.map(score_map)
            if self.require_external_prs:
                before = len(self.df)
                self.df = self.df[self.df[column].notna()].copy()
                logging.info(
                    "Required external PRS retained %d/%d image rows",
                    len(self.df),
                    before,
                )
            self.external_numeric_cols.append(column)

    def __normalise_icd10_code(self, code) -> str:
        if code is None or (isinstance(code, float) and math.isnan(code)):
            return ""
        return str(code).upper().replace(".", "").strip()

    def __load_clinical_history_mask_prefixes(self) -> set[str]:
        mask_path = self.clinical_history_mask_map_path
        if mask_path and not os.path.isabs(mask_path):
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            mask_path = os.path.join(repo_root, mask_path)
        if (
            not self.clinical_history_mask_targets
            or not mask_path
            or not os.path.exists(mask_path)
        ):
            return set()

        with open(mask_path, "r") as f:
            mask_map = json.load(f)

        disease_map = mask_map.get("disease_to_icd10_map", {})
        prefixes = set()
        for disease in self.clinical_history_target_diseases:
            disease_cfg = disease_map.get(disease, {})
            for group in ("parent_codes", "descendants_or_complications", "high_risk_proxies"):
                for item in disease_cfg.get(group, []):
                    code = self.__normalise_icd10_code(item.get("code"))
                    if code:
                        prefixes.add(code)
        return prefixes

    def __parse_clinical_history(self, value) -> list[dict]:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return []
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
        else:
            parsed = value
        return parsed if isinstance(parsed, list) else []

    def __hydrate_cached_clinical_history_if_needed(self):
        if "clinical_history" not in self.all_modalities:
            return
        if "patient_icd10_history" in self.df.columns:
            return
        if "patient_eid" not in self.df.columns:
            raise KeyError(
                "clinical_history requested, but cached dataframe is missing "
                "patient_icd10_history and patient_eid for hydration"
            )
        if not os.path.exists(self.df_path):
            raise FileNotFoundError(
                "clinical_history requested, but cached dataframe is missing "
                f"patient_icd10_history and source df_path does not exist: [{self.df_path}]"
            )

        logging.info(
            "Cached processed dataframe is missing patient_icd10_history; "
            f"hydrating it from source dataframe [{self.df_path}]"
        )
        source_cols = pd.read_csv(self.df_path, nrows=0).columns
        needed_cols = ["patient_eid", "patient_icd10_history"]
        missing_source_cols = [col for col in needed_cols if col not in source_cols]
        if missing_source_cols:
            raise KeyError(
                "clinical_history requested, but source dataframe is missing "
                f"columns: {missing_source_cols}"
            )

        history_df = pd.read_csv(
            self.df_path,
            usecols=needed_cols,
            low_memory=False,
        )
        history_df["patient_eid"] = pd.to_numeric(
            history_df["patient_eid"], errors="coerce"
        ).astype("Int64")
        history_df = history_df.dropna(subset=["patient_eid"])
        history_df = history_df.drop_duplicates(subset=["patient_eid"])

        self.df["patient_eid"] = pd.to_numeric(
            self.df["patient_eid"], errors="coerce"
        ).astype("Int64")
        original_len = len(self.df)
        self.df = self.df.merge(history_df, on="patient_eid", how="left")
        logging.info(
            "Hydrated patient_icd10_history into cached dataframe "
            f"with rows preserved [{original_len} -> {len(self.df)}]"
        )

    def __history_event_is_allowed(self, event: dict, index_date) -> bool:
        code = self.__normalise_icd10_code(event.get("code"))
        if not code:
            return False

        diagnosis_date = get_datetime(event.get("diagnosis_date"))
        if diagnosis_date is None or index_date is None:
            return False
        if diagnosis_date >= index_date:
            return False
        if any(code.startswith(prefix) for prefix in self.clinical_history_mask_prefixes):
            return False
        return True

    def __iter_preindex_clinical_history_events(self, row):
        index_date = get_datetime(row.get("instance_assessment_centre_visit_date"))
        for event in self.__parse_clinical_history(row.get("patient_icd10_history")):
            if self.__history_event_is_allowed(event, index_date):
                yield event

    def __init_clinical_history_if_needed(self):
        self.clinical_history_prefixes = set()
        self.icd10_code_to_id = {"<PAD>": 0, "<UNK>": 1}
        self.clinical_history_vocab_report = {}
        if "clinical_history" not in self.all_modalities:
            return
        if "patient_icd10_history" not in self.df.columns:
            raise KeyError("clinical_history requested, but patient_icd10_history is missing")

        self.clinical_history_prefixes = self.__load_clinical_history_mask_prefixes()
        self.clinical_history_mask_prefixes = self.clinical_history_prefixes

        train_mask = self.df["split"] == "train"
        code_counts = defaultdict(int)
        for _, row in self.df.loc[train_mask].iterrows():
            for event in self.__iter_preindex_clinical_history_events(row):
                code_counts[self.__normalise_icd10_code(event.get("code"))] += 1

        kept_codes = sorted(
            code for code, count in code_counts.items()
            if count >= self.clinical_history_min_count
        )
        self.icd10_code_to_id.update(
            {code: idx + 2 for idx, code in enumerate(kept_codes)}
        )
        self.clinical_history_vocab_report = {
            "min_count": int(self.clinical_history_min_count),
            "max_len": int(self.clinical_history_max_len),
            "n_codes_seen_train": int(len(code_counts)),
            "n_codes_kept": int(len(kept_codes)),
            "n_mask_prefixes": int(len(self.clinical_history_mask_prefixes)),
            "target_diseases": list(self.clinical_history_target_diseases),
        }
        logging.info(f"Clinical history vocab report: {self.clinical_history_vocab_report}")

    def __apply_omics_qc_if_needed(self):
        self.omics_qc_report = {}
        if not self.omics_qc:
            logging.info("Omics QC disabled.")
            return

        self.__apply_omics_qc_for_modality("metabolomics", self.metabolomics_cols)
        self.__apply_omics_qc_for_modality("proteomics", self.proteomics_cols)

    def __apply_omics_qc_for_modality(self, modality_name: str, cols: List[str]):
        if len(cols) == 0:
            return

        train_mask = self.df["split"] == "train"
        if train_mask.sum() == 0:
            raise ValueError("Cannot fit omics QC without train split rows.")

        report = {
            "n_features_before": len(cols),
            "n_rows_total": int(len(self.df)),
            "n_rows_train": int(train_mask.sum()),
            "missing_rate_threshold": float(self.omics_missing_rate_threshold),
            "min_std": float(self.omics_min_std),
            "min_unique_values": int(self.omics_min_unique_values),
            "winsor_lower_quantile": float(self.omics_winsor_lower_quantile),
            "winsor_upper_quantile": float(self.omics_winsor_upper_quantile),
            "masked_features": [],
            "kept_features": [],
        }

        modality_available = self.df[cols].notna().any(axis=1)
        train_available = self.df.loc[train_mask, cols].notna().any(axis=1)
        report["n_rows_with_any_value_total"] = int(modality_available.sum())
        report["n_rows_with_any_value_train"] = int(train_available.sum())

        for col in cols:
            train_s = pd.to_numeric(self.df.loc[train_mask, col], errors="coerce")
            missing_rate = float(train_s.isna().mean())
            non_missing = train_s.dropna()
            std = float(non_missing.std()) if len(non_missing) > 1 else 0.0
            n_unique = int(non_missing.nunique())

            reasons = []
            if missing_rate > self.omics_missing_rate_threshold:
                reasons.append("high_missing_rate")
            if not np.isfinite(std) or std < self.omics_min_std:
                reasons.append("low_variance")
            if n_unique < self.omics_min_unique_values:
                reasons.append("low_unique_values")

            if reasons:
                self.df[col] = np.nan
                report["masked_features"].append(
                    {
                        "column": col,
                        "missing_rate": missing_rate,
                        "std": std,
                        "n_unique": n_unique,
                        "reasons": reasons,
                    }
                )
                continue

            lower = float(non_missing.quantile(self.omics_winsor_lower_quantile))
            upper = float(non_missing.quantile(self.omics_winsor_upper_quantile))
            if np.isfinite(lower) and np.isfinite(upper) and lower <= upper:
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)

            report["kept_features"].append(
                {
                    "column": col,
                    "missing_rate": missing_rate,
                    "std": std,
                    "n_unique": n_unique,
                    "winsor_lower": lower,
                    "winsor_upper": upper,
                }
            )

        report["n_features_kept"] = len(report["kept_features"])
        report["n_features_masked"] = len(report["masked_features"])
        self.omics_qc_report[modality_name] = report
        logging.info(
            f"Omics QC [{modality_name}] kept [{report['n_features_kept']}] "
            f"and masked [{report['n_features_masked']}] of [{len(cols)}] features."
        )

    def __merge_omics_if_needed(self):
        if "metabolomics" in self.all_modalities:
            self.metabolomics_cols = self.__merge_omics_parquet(
                parquet_path=self.metabolomics_path,
                prefix="met_",
                modality_name="metabolomics",
            )
            self.metabolomics_instance_cols = self.__group_omics_cols_by_instance(
                self.metabolomics_cols, "metabolomics"
            )

        if "proteomics" in self.all_modalities:
            self.proteomics_cols = self.__merge_omics_parquet(
                parquet_path=self.proteomics_path,
                prefix="prot_",
                modality_name="proteomics",
            )
            self.proteomics_instance_cols = self.__group_omics_cols_by_instance(
                self.proteomics_cols, "proteomics"
            )

    def __merge_omics_parquet(
        self,
        *,
        parquet_path: str,
        prefix: str,
        modality_name: str,
    ) -> List[str]:
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(
                f"{modality_name} parquet not found at [{parquet_path}]"
            )

        logging.info(f"Loading {modality_name} parquet from [{parquet_path}]")
        omics_df = pd.read_parquet(parquet_path)

        if "eid" not in omics_df.columns:
            raise ValueError(f"{modality_name} parquet must contain an 'eid' column")

        omics_cols = [col for col in omics_df.columns if col.startswith(prefix)]
        if len(omics_cols) == 0:
            raise ValueError(
                f"No columns with prefix [{prefix}] found in {modality_name} parquet"
            )

        omics_df = omics_df[["eid"] + omics_cols].copy()
        omics_df["eid"] = pd.to_numeric(omics_df["eid"], errors="coerce").astype("Int64")
        omics_df = omics_df.dropna(subset=["eid"])
        omics_df = omics_df.rename(columns={"eid": "patient_eid"})

        self.df["patient_eid"] = pd.to_numeric(
            self.df["patient_eid"], errors="coerce"
        ).astype("Int64")

        original_len = len(self.df)
        self.df = self.df.merge(omics_df, on="patient_eid", how="left")

        for col in omics_cols:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        logging.info(
            f"Merged {modality_name} features [{len(omics_cols)} cols] into dataset "
            f"with rows preserved [{original_len} -> {len(self.df)}]"
        )
        return omics_cols

    def __group_omics_cols_by_instance(
        self, omics_cols: List[str], modality_name: str
    ) -> Dict[str, List[str]]:
        cols_by_instance = defaultdict(list)

        for col in omics_cols:
            parts = col.split("_")
            if modality_name == "metabolomics":
                if len(parts) < 4:
                    raise ValueError(
                        f"Unexpected metabolomics column format: [{col}]"
                    )
                instance = parts[2]
            elif modality_name == "proteomics":
                if len(parts) < 3:
                    raise ValueError(f"Unexpected proteomics column format: [{col}]")
                instance = parts[1]
            else:
                raise ValueError(f"Unsupported modality_name: [{modality_name}]")

            cols_by_instance[instance].append(col)

        grouped = {k: sorted(v) for k, v in cols_by_instance.items()}
        logging.info(
            f"{modality_name} columns grouped by instance: "
            f"{ {k: len(v) for k, v in grouped.items()} }"
        )
        return grouped

    def __get_image_instance(self, data) -> Optional[str]:
        img_path = data.get("image_path", None)
        if img_path is None or (isinstance(img_path, float) and math.isnan(img_path)):
            instance_idx = data.get("instance_idx", None)
            if instance_idx is None or pd.isna(instance_idx):
                return None
            return str(int(instance_idx))

        file_stem = os.path.splitext(os.path.basename(img_path))[0]
        parts = file_stem.split("_")
        if len(parts) < 4:
            logging.warning(f"Unexpected image filename format: [{img_path}]")
            return None

        return parts[-2]

    def get_instance_omics_cols(self, data, cols_by_instance: Dict[str, List[str]]):
        image_instance = self.__get_image_instance(data)
        if image_instance is None:
            return None

        label_cols = cols_by_instance.get(image_instance)
        if label_cols is None:
            return None

        return self.get_label_cols(data, label_cols)

    def get_clinical_history_with_provenance(
        self,
        data,
        excluded_event_indices=None,
    ):
        """
        Return the fixed model tensor plus a source-event provenance manifest.

        Fixed tensor format: (max_len, 6)
        columns: code_id, log1p(days_ago), recent_1yr, recent_5yr,
                 log1p(num_events), log1p(num_unique_codes)

        ``excluded_event_indices`` refers to stable indices assigned before
        sorting/truncation. It lets XAI rebuild a true leave-one-event-out
        counterfactual, including updated event and unique-code burdens.
        """
        excluded_event_indices = set(excluded_event_indices or [])
        max_len = self.clinical_history_max_len
        out = torch.zeros((max_len, 6), dtype=torch.float32)
        index_date = get_datetime(data.get("instance_assessment_centre_visit_date"))
        if index_date is None:
            return out, []

        events = []
        for source_event_index, event in enumerate(
            self.__iter_preindex_clinical_history_events(data)
        ):
            if source_event_index in excluded_event_indices:
                continue
            code = self.__normalise_icd10_code(event.get("code"))
            diagnosis_date = get_datetime(event.get("diagnosis_date"))
            if diagnosis_date is None:
                continue
            days_ago = max((index_date - diagnosis_date).days, 0)
            code_id = self.icd10_code_to_id.get(code, self.icd10_code_to_id["<UNK>"])
            events.append(
                {
                    "source_event_index": source_event_index,
                    "diagnosis_date": diagnosis_date,
                    "code_id": code_id,
                    "days_ago": days_ago,
                    "icd10_code": code,
                }
            )

        events.sort(key=lambda item: item["diagnosis_date"], reverse=True)
        retained_events = events[:max_len]
        burden_events = math.log1p(len(retained_events))
        burden_unique = math.log1p(len({item["icd10_code"] for item in events}))

        provenance = []
        for i, event in enumerate(retained_events):
            code_id = event["code_id"]
            days_ago = event["days_ago"]
            out[i, 0] = float(code_id)
            out[i, 1] = math.log1p(days_ago)
            out[i, 2] = float(days_ago <= 365)
            out[i, 3] = float(days_ago <= 365 * 5)
            out[i, 4] = burden_events
            out[i, 5] = burden_unique

            provenance.append(
                {
                    "event_index": i,
                    "source_event_index": event["source_event_index"],
                    "code_id": int(code_id),
                    "icd10_code": event["icd10_code"],
                    "days_before_index": int(days_ago),
                    "years_before_index": float(days_ago / 365.25),
                    "recent_1yr": bool(days_ago <= 365),
                    "recent_5yr": bool(days_ago <= 365 * 5),
                }
            )

        return out, provenance

    def get_clinical_history(self, data):
        history, _ = self.get_clinical_history_with_provenance(data)
        return history

    def get_omics_feature_manifest(self, data, modality_name):
        """Return the exact source-column order used for an omics model input."""
        if modality_name == "metabolomics":
            columns_by_instance = self.metabolomics_instance_cols
        elif modality_name == "proteomics":
            columns_by_instance = self.proteomics_instance_cols
        else:
            raise ValueError(f"Unsupported omics modality: [{modality_name}]")

        image_instance = self.__get_image_instance(data)
        columns = list(columns_by_instance.get(image_instance, []))
        values = self.get_label_cols(data, columns)
        return [
            {
                "feature_index": index,
                "source_column": column,
                "observed": bool(torch.isfinite(values[index]).item()),
            }
            for index, column in enumerate(columns)
        ]

    def mask_glaucoma_label(
        self,
    ):
        self.df["has_glaucoma_lifetime"] = self.df["patient_glaucoma_date"].notna()
        self.df["is_poag"] = self.df["patient_icd10_codes"].apply(
            lambda x: "H401" in json.loads(x)
        )

        # initialize flag column
        self.df["glaucoma_should_be_none"] = False

        # ---- poag filtering ----
        if self.poag_only:
            mask = self.df["has_glaucoma_lifetime"] & (~self.df["is_poag"])
            self.df.loc[mask, "glaucoma_should_be_none"] = True

        # ---- self-report filtering ----
        if self.remove_self_report_eye_problem_in_negative:
            # mark self-report glaucoma but no lifetime glaucoma
            mask = self.df["instance_self_report_glaucoma"] & (
                ~self.df["has_glaucoma_lifetime"]
            )
            self.df.loc[mask, "glaucoma_should_be_none"] = True

            # check for eye problems
            self.df["has_eye_problems"] = self.df["instance_eye_problems"].apply(
                lambda x: any([i > 0 for i in json.loads(x)])
            )
            mask = self.df["has_eye_problems"] & (~self.df["has_glaucoma_lifetime"])
            self.df.loc[mask, "glaucoma_should_be_none"] = True

        # ---- ICD10 filtering ----
        if self.remove_eye_icd10_not_poag:
            self.df["has_icd10_eye_not_poag"] = self.df["patient_icd10_codes"].apply(
                lambda x: has_icd10_eye_disease_except_poag(json.loads(x))
            )

            if self.remove_eye_icd10_not_poag_only_on_fales:
                mask = (~self.df["has_glaucoma_lifetime"]) & self.df[
                    "has_icd10_eye_not_poag"
                ]
            else:
                mask = self.df["has_icd10_eye_not_poag"]

            self.df.loc[mask, "glaucoma_should_be_none"] = True

    def get_onset_date(self, x):
        on_set_dates = []
        on_set_dates.append(x["patient_glaucoma_date"])

        if self.include_self_report_label:
            on_set_dates.append(x["patient_self_report_glaucoma_onset_date"])

        if self.include_eye_problem_label:
            on_set_dates.append(x["patient_eye_problem_glaucoma_onset_date"])

        # Remove None values from list
        on_set_dates = [x for x in on_set_dates if x is not None]

        if len(on_set_dates) == 0:
            return None

        return min(on_set_dates)

    def __set_categorical_clinical_features(
        self,
    ):
        for col in self.clinical_categorical_features:
            feature_dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
            self.clinical_categorical_cols = feature_dummies.columns

    def get_glaucoma_onset_after_days(self, x):
        onset_date = x["glaucoma_onset_date"]
        image_taken_time = x["instance_assessment_centre_visit_date"]

        if onset_date is None:
            return None

        return (onset_date - image_taken_time).days

    def get_genotype(self, data):
        # get line number
        eid = data["patient_eid"]
        has_it = data["patient_has_genotype"]
        if not data["patient_has_genotype"]:
            return None

        file_name = os.path.join(self.genotype_path, f"{eid}.ped")
        with open(file_name, "r") as file:
            data = file.read().strip().split()
            genotype_sequence = data[6:]
            encoded_genotype = [
                self.nucleotide_map.get(nuc, 5) for nuc in genotype_sequence
            ]  # Default to 'N' if not found
            genotype_tensor = torch.tensor(encoded_genotype, dtype=torch.long)

        genotype_tensor = torch.nn.functional.one_hot(
            genotype_tensor.long(), num_classes=len(self.nucleotide_map)
        )
        logging.info("[Dataset] Genotype tensor shape: ", genotype_tensor.shape)
        return genotype_tensor

    def __init_survival_progression_labels(self):
        for col in [
            "patient_glaucoma_date",
            "patient_final_assessment_centre_visit_date",
            "instance_assessment_centre_visit_date",
            "patient_ad_date",
            "patient_pd_date",
            "patient_hd_date",
            "patient_ms_date",
            "patient_t2d_date",
            "patient_cvd_date",
            "patient_dod",
        ]:
            self.df[col] = self.df[col].apply(get_datetime)

        logging.info(
            f"Using progression_label_ignorant_label_years: [{self.progression_label_ignorant_label_years}]"
        )

        for label_date_name in [
            "patient_glaucoma_date",
            "patient_ad_date",
            "patient_pd_date",
            "patient_hd_date",
            "patient_ms_date",
            "patient_t2d_date",
            "patient_cvd_date",
        ]:
            disease_name = self.label_date_to_disease[label_date_name]
            for year in self.progression_label_years:
                self.df[f"has_{disease_name}_in_{year}_years"] = self.df.apply(
                    lambda x: self.has_disease_by_n_years(
                        x,
                        year,
                        label_date_name,
                        positive_buffer_days=self.positive_buffer_days,
                        ignorant_buffer_years=self.progression_label_ignorant_label_years,
                    ),
                    axis=1,
                )

    def __init_requested_incident_labels(self) -> None:
        """Create disease-free risk-set labels requested as direct targets."""
        for label in self.possible_labels:
            match = _INCIDENT_BY_RE.match(label)
            if not match:
                continue
            disease = match.group("disease")
            year = int(match.group("years"))
            if year <= 0:
                raise ValueError(f"Incident target must have year > 0: {label}")
            baseline_col = f"has_{disease}_in_0_years"
            horizon_col = f"has_{disease}_in_{year}_years"
            missing = [
                column
                for column in (baseline_col, horizon_col)
                if column not in self.df.columns
            ]
            if missing:
                raise KeyError(
                    f"Cannot derive {label}; missing cumulative labels {missing}"
                )
            baseline = self.df[baseline_col]
            horizon = self.df[horizon_col]
            eligible = baseline.eq(False) & horizon.notna()
            values = pd.Series(float("nan"), index=self.df.index, dtype="float32")
            values.loc[eligible] = horizon.loc[eligible].astype("float32")
            self.df[label] = values
            logging.info(
                "Derived incident target %s: eligible=%d positive=%d excluded_prevalent=%d",
                label,
                int(eligible.sum()),
                int(values.eq(1).sum()),
                int(baseline.eq(True).sum()),
            )

    def has_disease_by_n_years(
        self,
        row,
        n: int,
        label_date_name: str,
        *,
        index_date_name: str = "instance_assessment_centre_visit_date",
        dod_name: str = "patient_dod",
        positive_buffer_days: int = 90,  # small grace for positives
        ignorant_buffer_years: int = 3,  # if event in (n, n+y], return None
    ):
        """
        Cumulative label for irreversible diseases:
        True  -> disease present at/before n years after index (prevalent or incident)
        False -> no disease by min(death,last_update) and follow-up spans >= n+y years
        None  -> uncertain (censored before n years, or event in (n, n+y])
        """
        idx = row.get(index_date_name)
        evt = row.get(label_date_name)  # first diagnosis date
        dod = row.get(dod_name)

        if ("glaucoma" in label_date_name) and row["glaucoma_should_be_none"]:
            return None

        # Normalise NaT/None
        def _nz(x):
            return None if (x is None or (hasattr(pd, "isna") and pd.isna(x))) else x

        idx, evt, dod = map(_nz, (idx, evt, dod))

        if idx is None:
            return None  # cannot define horizon

        # Calendar-accurate horizons
        n_cutoff = idx + relativedelta(years=n)
        n_buffer_cut = n_cutoff + pd.Timedelta(days=positive_buffer_days)
        cutoff_yrs = n + ignorant_buffer_years
        ny_cutoff = add_years_fraction(idx, cutoff_yrs)
        # ny_cutoff = idx + relativedelta(years=n + ignorant_buffer_years)

        # Censoring time
        censor = min(
            [d for d in (dod, self.last_update) if d is not None], default=None
        )

        # --- Event logic (irreversible disease) ---
        if evt is not None:
            # Prevalent at baseline -> already has disease by any horizon
            if evt <= idx:
                return True
            # Incident after baseline
            if evt <= n_buffer_cut:
                return True
            if evt <= ny_cutoff:
                return None
            # Event after n+y: if follow-up spans past n+y we can safely call False for n-year label,
            # but if censored earlier, we’re uncertain.
            return False if (censor is not None and censor >= ny_cutoff) else None

        # --- No recorded event: rely on follow-up ---
        if censor is None:
            raise ValueError(
                "No censoring date, please provide last_update_date at least."
            )
        if censor >= ny_cutoff:
            return False  # no event through at least n+y years
        if censor >= n_cutoff:
            return None  # between n and n+y -> uncertain by design
        return None  # censored before n

    @staticmethod
    def __json_loads_value(value):
        for _ in range(3):
            if isinstance(value, dict):
                return value
            if value is None:
                return {}
            try:
                if pd.isna(value):
                    return {}
            except (TypeError, ValueError):
                pass
            if not isinstance(value, str):
                return value
            value = value.strip()
            if value.lower() in {"nan", "none", "null"}:
                return {}
            if not value:
                return {}
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                return {}
        return value

    def __json_loads_cols(self):
        for col in [
            "patient_prs",
            "instance_questionnaire",
        ]:
            if col in self.df.columns:
                self.df[col] = self.df[col].apply(self.__json_loads_value)

    def __init_questionnaire_if_needed(self):
        self.questionnaire_tensorizer = None
        self.questionnaire_num_fields = 0
        self.questionnaire_category_vocab_size = 0
        if "questionnaire" not in self.all_modalities:
            return
        if "instance_questionnaire" not in self.df.columns:
            raise KeyError(
                "questionnaire requested, but processed data is missing "
                "instance_questionnaire. Rebuild preprocessing with fields_to_extract_v3.json."
            )
        train_records = self.df.loc[
            self.df["split"] == "train", "instance_questionnaire"
        ].tolist()
        self.questionnaire_tensorizer = QuestionnaireTensorizer.fit(
            train_records,
            max_choices=self.questionnaire_max_choices,
        )
        self.questionnaire_num_fields = len(QUESTIONNAIRE_FIELD_IDS)
        self.questionnaire_category_vocab_size = (
            self.questionnaire_tensorizer.category_vocab_size
        )
        logging.info(
            "Questionnaire tensorizer fitted: fields=%d, category_vocab=%d, "
            "max_choices=%d",
            self.questionnaire_num_fields,
            self.questionnaire_category_vocab_size,
            self.questionnaire_max_choices,
        )

    def __dataset_split(self):
        self.df = self.df[self.df["split"] == self.split]

    def __len__(self):
        return len(self.df)

    ###################################
    # Dataset Balancing
    ###################################

    def get_multi_class_labels_for_balance(self):
        values = self.df[self.balance_label_cols].apply(
            pd.to_numeric,
            errors="coerce",
        )
        return torch.from_numpy(
            values.fillna(0.0).to_numpy(dtype=np.float32) >= 0.5
        ).long()

    def get_sampling_weights(self):
        cfg = MultiLabelBalanceConfig(
            aggregation="mean",  # mean is more stable.
            smooth=1.0,
            min_weight=1e-3,
            max_weight=50.0,
            min_count_per_class=20,  # at least 20 samples for ukb
            normalize_mean_to_one=True,
            default_weight_if_all_missing=0.0,
        )

        label_importance = build_label_importance_from_balance_cols(
            self.balance_label_cols,
            year_to_importance=build_year_to_importance_linear(
                min_year=0,
                max_year=15,
                base_importance=0.3,
                max_importance=3.0,
            ),
            default_importance_for_unmatched=None,  # ignore non-matching cols
        )
        return build_multi_label_balanced_weights(
            self.df,
            self.balance_label_cols,
            label_importance=label_importance,
            cfg=cfg,
        )

    # def get_sampling_weights(self):
    #     labels = self.get_multi_class_labels_for_balance()
    #     indices = list(range(len(self)))
    #     df = pd.DataFrame(labels, index=indices)
    #     label_to_count = df.apply(pd.Series.value_counts)
    #     weights = [1 / (label_to_count[col][df[col]]) for col in df.columns]
    #     weights = torch.DoubleTensor(np.array(weights).sum(axis=0))
    #     return weights

    def get_pos_weights(self, label):
        pos_count = (self.df[label] == True).sum()

        if pos_count == 0:
            return 1

        return len(self.df) / pos_count

    ##################################
    # Modalities Grabbers
    ##################################

    def get_fundus_image_normalised(
        self,
        data,
        *,
        pad=16,
        threshold=40,  # apply on L-channel (0..255)
        size=None,  # final square size; set None to keep native
        debug=False,
        # Illumination controls
        illum_norm="post",  # 'none' | 'pre' | 'post'
        clahe_clip=2.0,
        clahe_tiles=(8, 8),
        white_balance=True,  # gray-world WB (pre) to tame strong casts
        # Color normalisation
        color_norm="none",  # 'none' | 'neutralize' | 'reinhard'
        target_a=0.0,
        target_b=0.0,
        neutralize_strength=0.5,  # reduce if you see desaturation
        chroma_floor=10.0,
        chroma_soft=8.0,
        ref_lab_means=None,  # for 'reinhard' e.g., (Lμ,aμ,bμ) from UKB train
        ref_lab_stds=None,  # for 'reinhard' e.g., (Lσ,aσ,bσ) from UKB train
    ):
        """
        Load a fundus image, detect vertical retina bounds via luminance threshold,
        crop a centered square using detected height (h_d), then apply illumination/color normalisation.

        Steps:
        1) (optional) WB + CLAHE (pre)  [keep detection stable → usually prefer post]
        2) Foreground detect on LAB L-channel > threshold
        3) Crop vertically [top:bottom]; square side = h_d; horizontally centered at W//2
        4) (optional) WB + CLAHE (post)  [recommended]
        5) (optional) Color normalise ('neutralize' chroma-aware or 'reinhard')
        6) (optional) Resize to (size,size)
        """
        img_path = data["image_path"]
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img)  # uint8 HxWx3
        H, W, _ = arr.shape

        if debug:
            print(f"[DEBUG] {img_path}  W={W} H={H}")
            print(
                f"[DEBUG] pad={pad} threshold(L)={threshold} illum_norm='{illum_norm}' WB={white_balance} color_norm='{color_norm}'"
            )

        # 1) Optional PRE normalisation (usually keep detection on raw-ish luminance)
        if white_balance:
            arr = _gray_world_white_balance(arr)
            if debug:
                print("[DEBUG] Applied gray-world WB (pre).")
        if illum_norm == "pre":
            arr = _clahe_L(arr, clip=clahe_clip, tiles=clahe_tiles)
            if debug:
                print("[DEBUG] Applied CLAHE on L (pre).")

        # 2) Foreground detection on luminance (LAB L channel)
        L = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[..., 0]  # 0..255
        fg = L > threshold

        if not fg.any():
            if debug:
                print("[DEBUG] No foreground detected; returning original.")
            out = Image.fromarray(arr)
        else:
            ys = np.where(fg.any(axis=1))[0]
            top = max(int(ys[0]) - pad, 0)
            bottom = min(int(ys[-1]) + pad, H - 1)
            h_d = bottom - top

            # Square crop centered horizontally at image midpoint
            cx = W // 2
            left = max(cx - h_d // 2, 0)
            right = min(cx + h_d // 2, W - 1)

            if debug:
                print(f"[DEBUG] bounds_y: top={top}, bottom={bottom}  h_d={h_d}")
                print(
                    f"[DEBUG] center_x={cx}  crop_x=({left},{right})  box=({left},{top},{right+1},{bottom+1})"
                )

            out = Image.fromarray(arr).crop((left, top, right + 1, bottom + 1))

        arr_c = np.asarray(out)

        # 4) Optional POST illumination normalisation (recommended)
        if illum_norm == "post":
            # WB again here is typically unnecessary if applied pre; keep False unless you need it
            if white_balance and debug:
                print("[DEBUG] WB already applied pre; usually skip post-WB.")
            arr_c = _clahe_L(arr_c, clip=clahe_clip, tiles=clahe_tiles)
            if debug:
                print("[DEBUG] Applied CLAHE on L (post).")

        # 5) Color normalisation
        if color_norm == "neutralize":
            arr_c = _lab_neutralize_chroma_aware(
                arr_c,
                target_a=target_a,
                target_b=target_b,
                strength=neutralize_strength,
                chroma_floor=chroma_floor,
                chroma_soft=chroma_soft,
            )
            if debug:
                print("[DEBUG] Applied chroma-aware LAB neutralization.")
        elif color_norm == "reinhard":
            if ref_lab_means is None or ref_lab_stds is None:
                raise ValueError(
                    "Provide ref_lab_means and ref_lab_stds for 'reinhard' color_norm."
                )
            arr_c = _reinhard_lab_transfer(arr_c, ref_lab_means, ref_lab_stds)
            if debug:
                print("[DEBUG] Applied Reinhard LAB color transfer.")

        out = Image.fromarray(arr_c)

        # 6) Final resize
        if size is not None:
            out = out.resize((size, size), resample=Image.BICUBIC)
            if debug:
                print(f"[DEBUG] Resized to {size}x{size}")

        return out

    def get_fundus_image(
        self,
        data,
    ):
        """
        Only do square crop
        """
        img_path = data["image_path"]
        if self._fundus_cache_index is not None:
            cache_index = self._fundus_cache_index.get(
                canonicalize_fundus_cache_key(img_path)
            )
            if cache_index is not None:
                cached = np.array(self._get_fundus_cache()[cache_index], copy=True)
                return Image.fromarray(cached, mode="RGB")
            if self.fundus_require_cache:
                raise KeyError(f"Fundus image is absent from required cache: {img_path}")
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img)  # uint8 HxWx3
        H, W, _ = arr.shape
        cx = W // 2
        out = Image.fromarray(arr).crop((cx - H // 2, 0, cx + H // 2, H))
        return out

    def get_oct_image(self, data):
        """
        Load an OCT volume from a UKB zip and return (S, 1, H, W).
        """
        profile = {
            "oct_total_s": 0.0,
            "oct_zip_open_s": 0.0,
            "oct_list_s": 0.0,
            "oct_select_s": 0.0,
            "oct_read_s": 0.0,
            "oct_decode_resize_s": 0.0,
            "oct_stack_s": 0.0,
            "oct_slices": 0,
            "oct_bytes": 0,
        }
        total_start = time.perf_counter()
        self._last_oct_profile = None

        oct_path = data.get("oct_image_path", None)
        if oct_path is None or pd.isna(oct_path) or not os.path.isfile(oct_path):
            return None

        def slice_index(name):
            match = re.search(r"_(\d+)\.png$", os.path.basename(name))
            return int(match.group(1)) if match else -1

        zip_start = time.perf_counter()
        with zipfile.ZipFile(oct_path) as zf:
            profile["oct_zip_open_s"] = time.perf_counter() - zip_start

            list_start = time.perf_counter()
            names = [
                name
                for name in zf.namelist()
                if name.lower().endswith(".png") and not name.endswith("/")
            ]
            # names: all slice PNG filenames from the zip, usually length 128.
            names = sorted(names, key=slice_index)
            profile["oct_list_s"] = time.perf_counter() - list_start
            if not names:
                return None

            select_start = time.perf_counter()
            if self.oct_num_slices and len(names) != self.oct_num_slices:
                # Use one coherent train-only shift for the sampled B-scan grid.
                idx = select_oct_slice_indices(
                    len(names),
                    self.oct_num_slices,
                    split=self.split,
                    profile=self.oct_aug_profile,
                    no_aug=not self.oct_transform.active,
                )
                names = [names[i] for i in idx]
            profile["oct_select_s"] = time.perf_counter() - select_start

            slices = []
            for name in names:
                with zf.open(name) as f:
                    # img: one grayscale B-scan, native UKB example is 512 x 650.
                    read_start = time.perf_counter()
                    raw = f.read()
                    profile["oct_read_s"] += time.perf_counter() - read_start
                    profile["oct_bytes"] += len(raw)

                    decode_start = time.perf_counter()
                    img = Image.open(io.BytesIO(raw)).convert("L")
                    if self.oct_image_size is not None:
                        # img: one resized B-scan, (oct_image_size, oct_image_size).
                        img = img.resize(
                            (self.oct_image_size, self.oct_image_size),
                            resample=Image.BILINEAR,
                        )
                    # arr: (H, W), float32 in [0, 1].
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    # Each item appended is (1, H, W), where 1 is grayscale channel.
                    slices.append(torch.from_numpy(arr).unsqueeze(0))
                    profile["oct_decode_resize_s"] += time.perf_counter() - decode_start

        # return: (S, 1, H, W), e.g. (32, 1, 224, 224) for the OCT PBS.
        stack_start = time.perf_counter()
        oct_tensor = torch.stack(slices, dim=0)
        oct_tensor = self.oct_transform(oct_tensor)
        profile["oct_stack_s"] = time.perf_counter() - stack_start
        profile["oct_slices"] = len(slices)
        profile["oct_total_s"] = time.perf_counter() - total_start
        if self.profile_timing:
            self._last_oct_profile = profile
        return oct_tensor

    def get_label_col(self, data, label_col):
        value = data[label_col]
        if value is None or pd.isna(value):
            return None

        return torch.tensor([float(value)], dtype=torch.float32)

    def get_label_cols(self, data, label_cols):
        """
        Return fixed-length tensor with NaNs preserved.
        - None / non-numeric will be coerced to NaN
        """
        if len(label_cols) == 0:
            return torch.empty(0, dtype=torch.float32)
        s = data.loc[label_cols]
        s = pd.to_numeric(s, errors="coerce")  # None/str -> NaN
        vals = s.to_numpy(dtype=np.float32, copy=True)  # may contain NaN
        return torch.from_numpy(vals)  # (n_features,)

    def get_questionnaire(self, data):
        if self.questionnaire_tensorizer is None:
            return None
        return self.questionnaire_tensorizer.transform(
            parse_questionnaire_record(data.get("instance_questionnaire"))
        )

    def add_output_modality(self, output_dict, key, value):
        if not value is None:
            output_dict.update({key: value})

    def grab_modalities(self, data, modalities):
        output = {}
        # ---- Bulk Data ----
        if "fundus_image" in modalities:
            self.add_output_modality(
                output,
                "fundus_image",
                self.transform(
                    self.get_fundus_image_normalised(data)
                    if self.normalise_fundus_image
                    else self.get_fundus_image(data)
                ),
            )

        if "oct_image" in modalities:
            self.add_output_modality(output, "oct_image", self.get_oct_image(data))
            if self.profile_timing and self._last_oct_profile is not None:
                output.setdefault("__profile__", {}).update(self._last_oct_profile)

        if "gender" in modalities:
            self.add_output_modality(
                output, "gender", self.get_label_col(data, "patient_gender")
            )

        if "age" in modalities:
            self.add_output_modality(
                output, "age", self.get_label_col(data, "instance_age_at_time")
            )

        if "iop" in modalities:
            self.add_output_modality(
                output, "iop", self.get_label_col(data, "instance_iop")
            )

        if "vcdr" in modalities:
            self.add_output_modality(
                output, "vcdr", self.get_label_col(data, "instance_vcdr")
            )

        if "genotype" in modalities:
            self.add_output_modality(
                output,
                "genotype",
                self.get_genotype(data),
            )

        # ---- PRS ----
        if "prs" in modalities:
            self.add_output_modality(
                output,
                "prs",
                self.get_label_cols(
                    data,
                    PRS_COLS,
                ),
            )

        if "glaucoma_prs" in modalities:
            self.add_output_modality(
                output,
                "glaucoma_prs",
                self.get_label_col(
                    data, "patient_Enhanced PRS for primary open angle glaucoma (POAG)"
                ),
            )

        if "ad_prs" in modalities:
            self.add_output_modality(
                output,
                "ad_prs",
                self.get_label_col(
                    data, "patient_Enhanced PRS for alzheimer's disease (AD)"
                ),
            )

        if "pd_prs" in modalities:
            self.add_output_modality(
                output,
                "pd_prs",
                self.get_label_col(
                    data, "patient_Enhanced PRS for parkinson's disease (PD)"
                ),
            )

        if "ms_prs" in modalities:
            self.add_output_modality(
                output,
                "ms_prs",
                self.get_label_col(
                    data, "patient_Enhanced PRS for multiple sclerosis (MS)"
                ),
            )

        if "t2d_prs" in modalities:
            self.add_output_modality(
                output,
                "t2d_prs",
                self.get_label_col(
                    data, "patient_Enhanced PRS for type 2 diabetes (T2D)"
                ),
            )

        if "dr_t2d_prs" in modalities:
            self.add_output_modality(
                output,
                "dr_t2d_prs",
                self.get_label_col(data, "patient_dr_t2d_prs"),
            )

        if "cvd_prs" in modalities:
            self.add_output_modality(
                output,
                "cvd_prs",
                self.get_label_col(
                    data, "patient_Enhanced PRS for cardiovascular disease (CVD)"
                ),
            )

        if "metabolomics" in modalities:
            self.add_output_modality(
                output,
                "metabolomics",
                self.get_instance_omics_cols(
                    data,
                    self.metabolomics_instance_cols,
                ),
            )

        if "proteomics" in modalities:
            self.add_output_modality(
                output,
                "proteomics",
                self.get_instance_omics_cols(
                    data,
                    self.proteomics_instance_cols,
                ),
            )

        if "clinical_history" in modalities:
            self.add_output_modality(
                output,
                "clinical_history",
                self.get_clinical_history(data),
            )

        if "questionnaire" in modalities:
            self.add_output_modality(
                output,
                "questionnaire",
                self.get_questionnaire(data),
            )

        # ---- Categorised Modalities ----
        if "demographics" in modalities:
            self.add_output_modality(
                output,
                "demographics",
                self.get_label_cols(
                    data,
                    DEMOGRAPHICS_COLS,
                ),
            )

        if "anthropometrics_core" in modalities:
            self.add_output_modality(
                output,
                "anthropometrics_core",
                self.get_label_cols(
                    data,
                    ANTHROPOMETRICS_CORE_COLS,
                ),
            )

        if "anthropometrics" in modalities:
            self.add_output_modality(
                output,
                "anthropometrics",
                self.get_label_cols(
                    data,
                    ANTHROPOMETRICS_COLS,
                ),
            )

        if "family_history" in modalities:
            self.add_output_modality(
                output,
                "family_history",
                self.get_label_cols(
                    data,
                    FAMILY_HISTORY_COLS,
                ),
            )

        if "principal_components" in modalities:
            self.add_output_modality(
                output,
                "principal_components",
                self.get_label_cols(
                    data,
                    PRINCIPAL_COMPONENT_COLS,
                ),
            )

        if "lifestyle" in modalities:
            self.add_output_modality(
                output,
                "lifestyle",
                self.get_label_cols(
                    data,
                    LIFESTYLE_COLS,
                ),
            )

        if "mental_health" in modalities:
            self.add_output_modality(
                output,
                "mental_health",
                self.get_label_cols(
                    data,
                    MENTAL_HEALTH_COLS,
                ),
            )

        if "socioeconomic" in modalities:
            self.add_output_modality(
                output,
                "socioeconomic",
                self.get_label_cols(
                    data,
                    SOCIOECONOMIC_COLS,
                ),
            )

        if "vitals" in modalities:
            self.add_output_modality(
                output,
                "vitals",
                self.get_label_cols(
                    data,
                    VITALS_COLS,
                ),
            )

        if "medications" in modalities:
            self.add_output_modality(
                output,
                "medications",
                self.get_label_cols(
                    data,
                    MEDICATIONS_COLS,
                ),
            )

        if "incident_cvd_medications" in modalities:
            self.add_output_modality(
                output,
                "incident_cvd_medications",
                self.get_label_cols(data, INCIDENT_CVD_MEDICATIONS_COLS),
            )

        if "ancestry" in modalities:
            self.add_output_modality(
                output,
                "ancestry",
                self.get_label_cols(
                    data,
                    ANCESTRY_COLS,
                ),
            )

        # ---- Single array clinical fields ----

        if "height_cm" in modalities:
            self.add_output_modality(
                output, "height_cm", self.get_label_col(data, "instance_height_cm")
            )

        if "weight_kg" in modalities:
            self.add_output_modality(
                output, "weight_kg", self.get_label_col(data, "instance_weight_kg")
            )

        if "body_mass_index_bmi" in modalities:
            self.add_output_modality(
                output,
                "body_mass_index_bmi",
                self.get_label_col(data, "instance_body_mass_index_bmi"),
            )

        if "waist_circumference_cm" in modalities:
            self.add_output_modality(
                output,
                "waist_circumference_cm",
                self.get_label_col(data, "instance_waist_circumference_cm"),
            )

        if "hip_circumference_cm" in modalities:
            self.add_output_modality(
                output,
                "hip_circumference_cm",
                self.get_label_col(data, "instance_hip_circumference_cm"),
            )

        if "smoking_status" in modalities:
            self.add_output_modality(
                output,
                "smoking_status",
                self.get_label_col(data, "instance_smoking_status"),
            )

        if "packyears_of_smoking" in modalities:
            self.add_output_modality(
                output,
                "packyears_of_smoking",
                self.get_label_col(data, "instance_packyears_of_smoking"),
            )

        if "alcohol_intake_frequency" in modalities:
            self.add_output_modality(
                output,
                "alcohol_intake_frequency",
                self.get_label_col(data, "instance_alcohol_intake_frequency"),
            )

        if "alcohol_consumption_unitsweek" in modalities:
            self.add_output_modality(
                output,
                "alcohol_consumption_unitsweek",
                self.get_label_col(data, "instance_alcohol_consumption_unitsweek"),
            )

        if "physical_activity_met_minswk" in modalities:
            self.add_output_modality(
                output,
                "physical_activity_met_minswk",
                self.get_label_col(data, "instance_physical_activity_met_minswk"),
            )

        if "coffee_intake_cupsday" in modalities:
            self.add_output_modality(
                output,
                "coffee_intake_cupsday",
                self.get_label_col(data, "instance_coffee_intake_cupsday"),
            )

        if "tea_intake_cupsday" in modalities:
            self.add_output_modality(
                output,
                "tea_intake_cupsday",
                self.get_label_col(data, "instance_tea_intake_cupsday"),
            )

        if "salt_added_to_food_yesno" in modalities:
            self.add_output_modality(
                output,
                "salt_added_to_food_yesno",
                self.get_label_col(data, "instance_salt_added_to_food_yesno"),
            )

        if "fruit_intake_portionsday" in modalities:
            self.add_output_modality(
                output,
                "fruit_intake_portionsday",
                self.get_label_col(data, "instance_fruit_intake_portionsday"),
            )

        if "vegetable_intake_portionsday" in modalities:
            self.add_output_modality(
                output,
                "vegetable_intake_portionsday",
                self.get_label_col(data, "instance_vegetable_intake_portionsday"),
            )

        if "townsend_deprivation_index" in modalities:
            self.add_output_modality(
                output,
                "townsend_deprivation_index",
                self.get_label_col(data, "instance_townsend_deprivation_index"),
            )

        if "selfrated_health" in modalities:
            self.add_output_modality(
                output,
                "selfrated_health",
                self.get_label_col(data, "instance_selfrated_health"),
            )

        if "sleep_duration_hoursnight" in modalities:
            self.add_output_modality(
                output,
                "sleep_duration_hoursnight",
                self.get_label_col(data, "instance_sleep_duration_hoursnight"),
            )

        if "doctordiagnosed_diabetes" in modalities:
            self.add_output_modality(
                output,
                "doctordiagnosed_diabetes",
                self.get_label_col(data, "instance_doctordiagnosed_diabetes"),
            )

        if "type_of_milk_usually_consumed" in modalities:
            self.add_output_modality(
                output,
                "type_of_milk_usually_consumed",
                self.get_label_col(data, "instance_type_of_milk_usually_consumed"),
            )

        if "cheese_intake" in modalities:
            self.add_output_modality(
                output,
                "cheese_intake",
                self.get_label_col(data, "instance_cheese_intake"),
            )

        if "processed_meat_intake" in modalities:
            self.add_output_modality(
                output,
                "processed_meat_intake",
                self.get_label_col(data, "instance_processed_meat_intake"),
            )

        if "poultry_intake" in modalities:
            self.add_output_modality(
                output,
                "poultry_intake",
                self.get_label_col(data, "instance_poultry_intake"),
            )

        if "seen_doctor_nerves_anxiety_tension_or_depression" in modalities:
            self.add_output_modality(
                output,
                "seen_doctor_nerves_anxiety_tension_or_depression",
                self.get_label_col(
                    data, "instance_seen_doctor_nerves_anxiety_tension_or_depression"
                ),
            )

        if "ever_depressed_for_a_whole_week" in modalities:
            self.add_output_modality(
                output,
                "ever_depressed_for_a_whole_week",
                self.get_label_col(data, "instance_ever_depressed_for_a_whole_week"),
            )

        if "bipolar_and_major_depression_status" in modalities:
            self.add_output_modality(
                output,
                "bipolar_and_major_depression_status",
                self.get_label_col(
                    data, "instance_bipolar_and_major_depression_status"
                ),
            )

        if "comparative_body_size_at_age_10" in modalities:
            self.add_output_modality(
                output,
                "comparative_body_size_at_age_10",
                self.get_label_col(data, "instance_comparative_body_size_at_age_10"),
            )

        # === Qualifications ===
        if "qualifications_college_or_university_degree" in modalities:
            self.add_output_modality(
                output,
                "qualifications_college_or_university_degree",
                self.get_label_col(
                    data, "instance_qualifications_college_or_university_degree"
                ),
            )

        if "qualifications_a_levels_as_levels_or_equivalent" in modalities:
            self.add_output_modality(
                output,
                "qualifications_a_levels_as_levels_or_equivalent",
                self.get_label_col(
                    data, "instance_qualifications_a_levels_as_levels_or_equivalent"
                ),
            )

        if "qualifications_o_levels_gcse_or_equivalent" in modalities:
            self.add_output_modality(
                output,
                "qualifications_o_levels_gcse_or_equivalent",
                self.get_label_col(
                    data, "instance_qualifications_o_levels_gcse_or_equivalent"
                ),
            )

        if "qualifications_nvq_or_hnd_or_hnc_or_equivalent" in modalities:
            self.add_output_modality(
                output,
                "qualifications_nvq_or_hnd_or_hnc_or_equivalent",
                self.get_label_col(
                    data, "instance_qualifications_nvq_or_hnd_or_hnc_or_equivalent"
                ),
            )

        if "qualifications_other_professional_qualifications" in modalities:
            self.add_output_modality(
                output,
                "qualifications_other_professional_qualifications",
                self.get_label_col(
                    data, "instance_qualifications_other_professional_qualifications"
                ),
            )

        if "qualifications_prefer_not_to_answer" in modalities:
            self.add_output_modality(
                output,
                "qualifications_prefer_not_to_answer",
                self.get_label_col(
                    data, "instance_qualifications_prefer_not_to_answer"
                ),
            )

        # === Current Employment ===
        if "current_employment_paid_or_self_employed" in modalities:
            self.add_output_modality(
                output,
                "current_employment_paid_or_self_employed",
                self.get_label_col(
                    data, "instance_current_employment_paid_or_self_employed"
                ),
            )

        if "current_employment_retired" in modalities:
            self.add_output_modality(
                output,
                "current_employment_retired",
                self.get_label_col(data, "instance_current_employment_retired"),
            )

        if "current_employment_looking_after_home_and_or_family" in modalities:
            self.add_output_modality(
                output,
                "current_employment_looking_after_home_and_or_family",
                self.get_label_col(
                    data, "instance_current_employment_looking_after_home_and_or_family"
                ),
            )

        if "current_employment_unable_to_work_sickness_or_disability" in modalities:
            self.add_output_modality(
                output,
                "current_employment_unable_to_work_sickness_or_disability",
                self.get_label_col(
                    data,
                    "instance_current_employment_unable_to_work_sickness_or_disability",
                ),
            )

        if "current_employment_unemployed" in modalities:
            self.add_output_modality(
                output,
                "current_employment_unemployed",
                self.get_label_col(data, "instance_current_employment_unemployed"),
            )

        if "current_employment_unpaid_or_voluntary_work" in modalities:
            self.add_output_modality(
                output,
                "current_employment_unpaid_or_voluntary_work",
                self.get_label_col(
                    data, "instance_current_employment_unpaid_or_voluntary_work"
                ),
            )

        if "current_employment_full_or_part_time_student" in modalities:
            self.add_output_modality(
                output,
                "current_employment_full_or_part_time_student",
                self.get_label_col(
                    data, "instance_current_employment_full_or_part_time_student"
                ),
            )

        if "current_employment_prefer_not_to_answer" in modalities:
            self.add_output_modality(
                output,
                "current_employment_prefer_not_to_answer",
                self.get_label_col(
                    data, "instance_current_employment_prefer_not_to_answer"
                ),
            )

        # === Vascular/Heart Problems Diagnosed by Doctor ===
        if "vascular_heart_problems_heart_attack" in modalities:
            self.add_output_modality(
                output,
                "vascular_heart_problems_heart_attack",
                self.get_label_col(
                    data, "instance_vascular_heart_problems_heart_attack"
                ),
            )

        if "vascular_heart_problems_angina" in modalities:
            self.add_output_modality(
                output,
                "vascular_heart_problems_angina",
                self.get_label_col(data, "instance_vascular_heart_problems_angina"),
            )

        if "vascular_heart_problems_stroke" in modalities:
            self.add_output_modality(
                output,
                "vascular_heart_problems_stroke",
                self.get_label_col(data, "instance_vascular_heart_problems_stroke"),
            )

        if "vascular_heart_problems_high_blood_pressure" in modalities:
            self.add_output_modality(
                output,
                "vascular_heart_problems_high_blood_pressure",
                self.get_label_col(
                    data, "instance_vascular_heart_problems_high_blood_pressure"
                ),
            )

        if "vascular_heart_problems_prefer_not_to_answer" in modalities:
            self.add_output_modality(
                output,
                "vascular_heart_problems_prefer_not_to_answer",
                self.get_label_col(
                    data, "instance_vascular_heart_problems_prefer_not_to_answer"
                ),
            )

        # === Medications ===
        if "med_cholesterol" in modalities:
            self.add_output_modality(
                output,
                "med_cholesterol",
                self.get_label_col(data, "instance_med_cholesterol"),
            )

        if "med_blood_pressure" in modalities:
            self.add_output_modality(
                output,
                "med_blood_pressure",
                self.get_label_col(data, "instance_med_blood_pressure"),
            )

        if "med_diabetes" in modalities:
            self.add_output_modality(
                output,
                "med_diabetes",
                self.get_label_col(data, "instance_med_diabetes"),
            )

        if "med_hormone_replacement" in modalities:
            self.add_output_modality(
                output,
                "med_hormone_replacement",
                self.get_label_col(data, "instance_med_hormone_replacement"),
            )

        if "med_prefer_not_to_answer" in modalities:
            self.add_output_modality(
                output,
                "med_prefer_not_to_answer",
                self.get_label_col(data, "instance_med_prefer_not_to_answer"),
            )

        if "glaucoma_med" in modalities:
            self.add_output_modality(
                output,
                "glaucoma_med",
                self.get_label_col(data, "instance_glaucoma_med"),
            )

        # === Illnesses of Father ===
        if "father_heart_disease" in modalities:
            self.add_output_modality(
                output,
                "father_heart_disease",
                self.get_label_col(data, "instance_father_heart_disease"),
            )

        if "father_stroke" in modalities:
            self.add_output_modality(
                output,
                "father_stroke",
                self.get_label_col(data, "instance_father_stroke"),
            )

        if "father_lung_cancer" in modalities:
            self.add_output_modality(
                output,
                "father_lung_cancer",
                self.get_label_col(data, "instance_father_lung_cancer"),
            )

        if "father_bowel_cancer" in modalities:
            self.add_output_modality(
                output,
                "father_bowel_cancer",
                self.get_label_col(data, "instance_father_bowel_cancer"),
            )

        if "father_breast_cancer" in modalities:
            self.add_output_modality(
                output,
                "father_breast_cancer",
                self.get_label_col(data, "instance_father_breast_cancer"),
            )

        if "father_chronic_bronchitis_emphysema" in modalities:
            self.add_output_modality(
                output,
                "father_chronic_bronchitis_emphysema",
                self.get_label_col(
                    data, "instance_father_chronic_bronchitis_emphysema"
                ),
            )

        if "father_high_blood_pressure" in modalities:
            self.add_output_modality(
                output,
                "father_high_blood_pressure",
                self.get_label_col(data, "instance_father_high_blood_pressure"),
            )

        if "father_diabetes" in modalities:
            self.add_output_modality(
                output,
                "father_diabetes",
                self.get_label_col(data, "instance_father_diabetes"),
            )

        if "father_alzheimers_dementia" in modalities:
            self.add_output_modality(
                output,
                "father_alzheimers_dementia",
                self.get_label_col(data, "instance_father_alzheimers_dementia"),
            )

        if "father_parkinsons_disease" in modalities:
            self.add_output_modality(
                output,
                "father_parkinsons_disease",
                self.get_label_col(data, "instance_father_parkinsons_disease"),
            )

        if "father_severe_depression" in modalities:
            self.add_output_modality(
                output,
                "father_severe_depression",
                self.get_label_col(data, "instance_father_severe_depression"),
            )

        if "father_prostate_cancer" in modalities:
            self.add_output_modality(
                output,
                "father_prostate_cancer",
                self.get_label_col(data, "instance_father_prostate_cancer"),
            )

        if "father_hip_fracture" in modalities:
            self.add_output_modality(
                output,
                "father_hip_fracture",
                self.get_label_col(data, "instance_father_hip_fracture"),
            )

        if "father_prefer_not_to_answer" in modalities:
            self.add_output_modality(
                output,
                "father_prefer_not_to_answer",
                self.get_label_col(data, "instance_father_prefer_not_to_answer"),
            )

        # === Illnesses of Mother ===
        if "maternal_heart_disease" in modalities:
            self.add_output_modality(
                output,
                "maternal_heart_disease",
                self.get_label_col(data, "instance_maternal_heart_disease"),
            )

        if "maternal_stroke" in modalities:
            self.add_output_modality(
                output,
                "maternal_stroke",
                self.get_label_col(data, "instance_maternal_stroke"),
            )

        if "maternal_lung_cancer" in modalities:
            self.add_output_modality(
                output,
                "maternal_lung_cancer",
                self.get_label_col(data, "instance_maternal_lung_cancer"),
            )

        if "maternal_bowel_cancer" in modalities:
            self.add_output_modality(
                output,
                "maternal_bowel_cancer",
                self.get_label_col(data, "instance_maternal_bowel_cancer"),
            )

        if "maternal_breast_cancer" in modalities:
            self.add_output_modality(
                output,
                "maternal_breast_cancer",
                self.get_label_col(data, "instance_maternal_breast_cancer"),
            )

        if "maternal_chronic_bronchitis_emphysema" in modalities:
            self.add_output_modality(
                output,
                "maternal_chronic_bronchitis_emphysema",
                self.get_label_col(
                    data, "instance_maternal_chronic_bronchitis_emphysema"
                ),
            )

        if "maternal_high_blood_pressure" in modalities:
            self.add_output_modality(
                output,
                "maternal_high_blood_pressure",
                self.get_label_col(data, "instance_maternal_high_blood_pressure"),
            )

        if "maternal_diabetes" in modalities:
            self.add_output_modality(
                output,
                "maternal_diabetes",
                self.get_label_col(data, "instance_maternal_diabetes"),
            )

        if "maternal_alzheimers_dementia" in modalities:
            self.add_output_modality(
                output,
                "maternal_alzheimers_dementia",
                self.get_label_col(data, "instance_maternal_alzheimers_dementia"),
            )

        if "maternal_parkinsons_disease" in modalities:
            self.add_output_modality(
                output,
                "maternal_parkinsons_disease",
                self.get_label_col(data, "instance_maternal_parkinsons_disease"),
            )

        if "maternal_severe_depression" in modalities:
            self.add_output_modality(
                output,
                "maternal_severe_depression",
                self.get_label_col(data, "instance_maternal_severe_depression"),
            )

        if "maternal_prostate_cancer" in modalities:
            self.add_output_modality(
                output,
                "maternal_prostate_cancer",
                self.get_label_col(data, "instance_maternal_prostate_cancer"),
            )

        if "maternal_hip_fracture" in modalities:
            self.add_output_modality(
                output,
                "maternal_hip_fracture",
                self.get_label_col(data, "instance_maternal_hip_fracture"),
            )

        if "maternal_prefer_not_to_answer" in modalities:
            self.add_output_modality(
                output,
                "maternal_prefer_not_to_answer",
                self.get_label_col(data, "instance_maternal_prefer_not_to_answer"),
            )

        # === Illnesses of Siblings ===
        if "sibling_heart_disease" in modalities:
            self.add_output_modality(
                output,
                "sibling_heart_disease",
                self.get_label_col(data, "instance_sibling_heart_disease"),
            )

        if "sibling_stroke" in modalities:
            self.add_output_modality(
                output,
                "sibling_stroke",
                self.get_label_col(data, "instance_sibling_stroke"),
            )

        if "sibling_lung_cancer" in modalities:
            self.add_output_modality(
                output,
                "sibling_lung_cancer",
                self.get_label_col(data, "instance_sibling_lung_cancer"),
            )

        if "sibling_bowel_cancer" in modalities:
            self.add_output_modality(
                output,
                "sibling_bowel_cancer",
                self.get_label_col(data, "instance_sibling_bowel_cancer"),
            )

        if "sibling_breast_cancer" in modalities:
            self.add_output_modality(
                output,
                "sibling_breast_cancer",
                self.get_label_col(data, "instance_sibling_breast_cancer"),
            )

        if "sibling_chronic_bronchitis_emphysema" in modalities:
            self.add_output_modality(
                output,
                "sibling_chronic_bronchitis_emphysema",
                self.get_label_col(
                    data, "instance_sibling_chronic_bronchitis_emphysema"
                ),
            )

        if "sibling_high_blood_pressure" in modalities:
            self.add_output_modality(
                output,
                "sibling_high_blood_pressure",
                self.get_label_col(data, "instance_sibling_high_blood_pressure"),
            )

        if "sibling_diabetes" in modalities:
            self.add_output_modality(
                output,
                "sibling_diabetes",
                self.get_label_col(data, "instance_sibling_diabetes"),
            )

        if "sibling_alzheimers_dementia" in modalities:
            self.add_output_modality(
                output,
                "sibling_alzheimers_dementia",
                self.get_label_col(data, "instance_sibling_alzheimers_dementia"),
            )

        if "sibling_parkinsons_disease" in modalities:
            self.add_output_modality(
                output,
                "sibling_parkinsons_disease",
                self.get_label_col(data, "instance_sibling_parkinsons_disease"),
            )

        if "sibling_severe_depression" in modalities:
            self.add_output_modality(
                output,
                "sibling_severe_depression",
                self.get_label_col(data, "instance_sibling_severe_depression"),
            )

        if "sibling_prostate_cancer" in modalities:
            self.add_output_modality(
                output,
                "sibling_prostate_cancer",
                self.get_label_col(data, "instance_sibling_prostate_cancer"),
            )

        if "sibling_hip_fracture" in modalities:
            self.add_output_modality(
                output,
                "sibling_hip_fracture",
                self.get_label_col(data, "instance_sibling_hip_fracture"),
            )

        if "sibling_prefer_not_to_answer" in modalities:
            self.add_output_modality(
                output,
                "sibling_prefer_not_to_answer",
                self.get_label_col(data, "instance_sibling_prefer_not_to_answer"),
            )

        # === Mean Values ===
        if "systolic_bp" in modalities:
            self.add_output_modality(
                output, "systolic_bp", self.get_label_col(data, "instance_systolic_bp")
            )

        if "diastolic_bp" in modalities:
            self.add_output_modality(
                output,
                "diastolic_bp",
                self.get_label_col(data, "instance_diastolic_bp"),
            )

        if "pulse_rate" in modalities:
            self.add_output_modality(
                output, "pulse_rate", self.get_label_col(data, "instance_pulse_rate")
            )

        ## Above added the input modalities into the dict, the following adds labels.
        for label in self.possible_labels:
            self.add_output_modality(output, label, self.get_label_col(data, label))

        return output

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        item_start = time.perf_counter()
        data = self.df.iloc[idx]
        modalities = self.grab_modalities(data, self.all_modalities)
        modalities.update(
            {
                "idx": idx,
                "patient_eid": data.get("patient_eid"),
                "dataset": "ukb",
            }
        )
        if self.profile_timing:
            modalities.setdefault("__profile__", {})["getitem_total_s"] = (
                time.perf_counter() - item_start
            )
        return modalities

    def invert_normalisation(self, modalities):

        feature_to_col = {
            "age": "instance_age_at_time",
            "iop": "instance_iop",
            "vcdr": "instance_vcdr",
        }

        inverted_modalities = {}

        for key, value in modalities.items():

            if key not in self.mean_std_map:
                inverted_modalities[key] = value
                continue

            # if the key in feature_to_col, we need to get the corresponding column name to retrieve the mean and std
            if key in feature_to_col:
                key_for_mean_std = feature_to_col[key]
                # key_for_mean_std = col_name.replace("instance_", "")
            else:
                key_for_mean_std = key

            mean = self.mean_std_map[key_for_mean_std]["mean"]
            std = self.mean_std_map[key_for_mean_std]["std"]

            if isinstance(value, (int, float, np.integer, np.floating)):
                inverted_modalities[key] = float(value * std + mean)

            elif isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    v = value.item()
                    inverted_modalities[key] = float(v * std + mean)
                else:
                    inverted_modalities[key] = (value * std + mean).cpu().tolist()

            elif isinstance(value, np.ndarray):
                inverted_modalities[key] = (value * std + mean).tolist()

            else:
                inverted_modalities[key] = value

        return inverted_modalities



def resolve_split_quality_control(args, default_quality_control):
    """Resolve a global QC policy or an explicit per-split override."""
    requested = getattr(args, "quality_control_splits", None)
    if not requested:
        enabled = bool(default_quality_control)
        return {split: enabled for split in ("train", "val", "test")}

    if isinstance(requested, str):
        requested = [part.strip() for part in requested.split(",") if part.strip()]
    aliases = {"validation": "val"}
    normalized = {aliases.get(str(split).strip().lower(), str(split).strip().lower()) for split in requested}
    valid = {"train", "val", "test"}
    invalid = normalized - valid
    if invalid:
        raise ValueError(
            "quality_control_splits contains unsupported split(s): "
            f"{sorted(invalid)}; expected a subset of {sorted(valid)}"
        )
    return {split: split in normalized for split in ("train", "val", "test")}


def build_image_level_universal_datasets(args, **kwargs):
    dataset_kwargs = dict(kwargs)
    dataset_kwargs.setdefault(
        "processed_df_path",
        getattr(args, "ukb_processed_df_path", UKB_PROCERESS_DF_SAVED_PATH),
    )
    dataset_kwargs.setdefault(
        "require_any_requested_input",
        getattr(args, "require_any_requested_input", False),
    )
    dataset_kwargs["allow_legacy_iop_manifest"] = bool(
        getattr(args, "allow_legacy_iop_manifest", False)
    )
    logging.info(
        "Legacy IOP manifest compatibility: %s",
        dataset_kwargs["allow_legacy_iop_manifest"],
    )
    dataset_kwargs.setdefault("omics_qc", getattr(args, "omics_qc", True))
    dataset_kwargs.setdefault(
        "omics_missing_rate_threshold",
        getattr(args, "omics_missing_rate_threshold", 0.40),
    )
    dataset_kwargs.setdefault("omics_min_std", getattr(args, "omics_min_std", 1e-6))
    dataset_kwargs.setdefault(
        "omics_min_unique_values", getattr(args, "omics_min_unique_values", 10)
    )
    dataset_kwargs.setdefault(
        "omics_winsor_lower_quantile",
        getattr(args, "omics_winsor_lower_quantile", 0.005),
    )
    dataset_kwargs.setdefault(
        "omics_winsor_upper_quantile",
        getattr(args, "omics_winsor_upper_quantile", 0.995),
    )
    dataset_kwargs.setdefault(
        "clinical_history_min_count",
        getattr(args, "clinical_history_min_count", 10),
    )
    dataset_kwargs.setdefault(
        "clinical_history_max_len",
        getattr(args, "clinical_history_max_len", 128),
    )
    dataset_kwargs.setdefault(
        "clinical_history_mask_targets",
        getattr(args, "clinical_history_mask_targets", True),
    )
    dataset_kwargs.setdefault(
        "clinical_history_mask_map_path",
        getattr(args, "clinical_history_mask_map_path", DEFAULT_ICD10_MASK_MAP_PATH),
    )
    dataset_kwargs.setdefault(
        "clinical_history_target_diseases",
        getattr(args, "diseases", None),
    )
    dataset_kwargs.setdefault(
        "questionnaire_max_choices",
        getattr(args, "questionnaire_max_choices", 16),
    )
    dataset_kwargs.setdefault("oct_num_slices", getattr(args, "oct_num_slices", 128))
    dataset_kwargs.setdefault("oct_image_size", getattr(args, "oct_image_size", None))
    dataset_kwargs.setdefault(
        "oct_aug_profile",
        getattr(args, "oct_aug_profile", "oct_clinical_v1"),
    )
    dataset_kwargs.setdefault("profile_timing", getattr(args, "profile_timing", False))
    dataset_kwargs.setdefault(
        "algo_qc_path", getattr(args, "algo_qc_path", "data/ukb/algorithmic_qc.csv")
    )
    dataset_kwargs.setdefault(
        "smoke_test_max_rows_per_split",
        getattr(args, "smoke_test_max_rows_per_split", None),
    )
    dataset_kwargs.setdefault("smoke_test_seed", getattr(args, "seed", 42))
    dataset_kwargs.setdefault(
        "external_binary_phenotype_path",
        getattr(args, "external_binary_phenotype_path", None),
    )
    dataset_kwargs.setdefault(
        "external_binary_phenotype_id_col",
        getattr(args, "external_binary_phenotype_id_col", "IID"),
    )
    dataset_kwargs.setdefault(
        "external_binary_phenotype_label_col",
        getattr(args, "external_binary_phenotype_label_col", "phenotype"),
    )
    dataset_kwargs.setdefault(
        "external_binary_target_disease",
        getattr(args, "external_binary_target_disease", None),
    )
    dataset_kwargs.setdefault(
        "external_binary_restrict_cohort",
        getattr(args, "external_binary_restrict_cohort", True),
    )
    dataset_kwargs.setdefault(
        "external_binary_phenotype_date_col",
        getattr(args, "external_binary_phenotype_date_col", None),
    )
    dataset_kwargs.setdefault(
        "external_binary_instance_date_col",
        getattr(args, "external_binary_instance_date_col", None),
    )
    dataset_kwargs.setdefault("external_prs_path", getattr(args, "external_prs_path", None))
    dataset_kwargs.setdefault(
        "external_prs_id_col", getattr(args, "external_prs_id_col", "IID")
    )
    dataset_kwargs.setdefault(
        "external_prs_score_col", getattr(args, "external_prs_score_col", "SCORE1_AVG")
    )
    dataset_kwargs.setdefault(
        "require_external_prs", getattr(args, "require_external_prs", False)
    )
    dataset_kwargs.setdefault("fundus_cache_path", getattr(args, "fundus_cache_path", None))
    dataset_kwargs.setdefault(
        "fundus_cache_index_path", getattr(args, "fundus_cache_index_path", None)
    )
    dataset_kwargs.setdefault(
        "fundus_cache_metadata_path",
        getattr(args, "fundus_cache_metadata_path", None),
    )
    dataset_kwargs.setdefault(
        "fundus_require_cache", getattr(args, "fundus_require_cache", False)
    )
    dataset_kwargs.setdefault(
        "fundus_cache_allow_resize",
        getattr(args, "fundus_cache_allow_resize", False),
    )
    dataset_kwargs.setdefault(
        "fundus_aug_profile",
        getattr(args, "fundus_aug_profile", "kim_enhanced"),
    )

    split_quality_control = resolve_split_quality_control(
        args,
        dataset_kwargs.pop("quality_control", True),
    )
    logging.info(
        "Image QC policy by split: train=%s val=%s test=%s",
        split_quality_control["train"],
        split_quality_control["val"],
        split_quality_control["test"],
    )

    train_dataset = ImageLevelUKBUniversalDataset(
        image_size=args.image_size,
        split="train",
        quality_control=split_quality_control["train"],
        **dataset_kwargs,
    )
    val_dataset = ImageLevelUKBUniversalDataset(
        image_size=args.image_size,
        split="val",
        quality_control=split_quality_control["val"],
        **dataset_kwargs,
    )
    test_dataset = ImageLevelUKBUniversalDataset(
        image_size=args.image_size,
        split="test",
        quality_control=split_quality_control["test"],
        **dataset_kwargs,
    )
    return train_dataset, val_dataset, test_dataset
