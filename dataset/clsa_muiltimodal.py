import torch
import torch.nn as nn
import pandas as pd
import math
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from .paths import *
import logging
from .aug import get_default_aug
import numpy as np
import json
import cv2
import os
from pathlib import Path

from collections import Counter
from typing import Dict, List, Tuple, Optional
from .universal_image import (
    PRS_COLS,
    ANTHROPOMETRICS_COLS,
    ANCESTRY_COLS,
    CLSA_ANCESTRY_MAP,
    FAMILY_HISTORY_COLS,
    LIFESTYLE_COLS,
    MENTAL_HEALTH_COLS,
    PRINCIPAL_COMPONENT_COLS,
    SOCIOECONOMIC_COLS,
    VITALS_COLS,
    MEDICATIONS_COLS,
)

CLSA_DATA_ROOT = Path(os.environ.get("FORESIGHT_OMNI_CLSA_ROOT", "data/clsa"))
CLSA_PRS_DIR = str(CLSA_DATA_ROOT / "prs")
CLSA_PRS_LINKING_KEY_PATH = str(CLSA_DATA_ROOT / "gwas_linking_key.csv")
CLSA_PRS_FILES = {
    "patient_Enhanced PRS for primary open angle glaucoma (POAG)": "POAG_PRS.sscore",
    "patient_Enhanced PRS for alzheimer's disease (AD)": "ALZ_PRS.sscore",
    "patient_Enhanced PRS for parkinson's disease (PD)": "PD_PRS.sscore",
    "patient_Enhanced PRS for multiple sclerosis (MS)": "MS_PRS.sscore",
    "patient_Enhanced PRS for type 2 diabetes (T2D)": "T2D_PRS.sscore",
    "patient_Enhanced PRS for cardiovascular disease (CVD)": "CVD_PRS.sscore",
}

CLSA_LEGACY_DATA_ROOT = os.environ.get(
    "FORESIGHT_OMNI_CLSA_LEGACY_ROOT", str(CLSA_DATA_ROOT / "legacy")
)
CLSA_CURRENT_DATA_ROOT = os.environ.get(
    "FORESIGHT_OMNI_CLSA_IMAGE_ROOT", str(CLSA_DATA_ROOT / "images")
)
CLSA_ALGO_QC_PATH = str(CLSA_DATA_ROOT / "qc" / "algorithmic_qc.csv")
CLSA_DL_QC_PATH = str(CLSA_DATA_ROOT / "qc" / "dl_qc.csv")

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


# -------------------------
# Normalization utilities
# -------------------------


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


def get_image_file_names(entity_id, side: str):
    return f"190225_QIMR_SMacGregor_{entity_id}_{side}.jpg"


def _normalise_sample_id(x):
    if x is None or pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def normalise_clsa_image_path(path):
    """Map the retired CLSA storage prefix to its current reference location."""
    if not isinstance(path, str):
        return path
    if path == CLSA_LEGACY_DATA_ROOT:
        return CLSA_CURRENT_DATA_ROOT
    legacy_prefix = CLSA_LEGACY_DATA_ROOT + os.sep
    if path.startswith(legacy_prefix):
        return CLSA_CURRENT_DATA_ROOT + path[len(CLSA_LEGACY_DATA_ROOT) :]
    return path


CLSA_NUMERICAL_FEATURES = [
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
]

CLSA_CATEGORICAL_FEATURES = [
    "instance_comparative_body_size_at_age_10",
]


def _to_float_or_nan(x):
    if x is None or pd.isna(x):
        return np.nan
    try:
        return float(x)
    except Exception:
        return np.nan


def _clean_numeric(x, *, min_value=None, max_value=None, invalid_values=None):
    x = _to_float_or_nan(x)
    if np.isnan(x):
        return np.nan

    if invalid_values is not None and x in invalid_values:
        return np.nan

    if min_value is not None and x < min_value:
        return np.nan
    if max_value is not None and x > max_value:
        return np.nan
    return x


def _binary_yes_no(x):
    x = _to_float_or_nan(x)
    if np.isnan(x):
        return np.nan
    if x == 1:
        return 1.0
    if x == 2:
        return 0.0
    if x == 8:
        return -1.0  # Don't know/No answer → Do not know
    if x == 9:
        return -3.0  # Refused → Prefer not to answer
    return np.nan


# Assumes these exist in your repo (same as your previous version)
# - get_default_aug
# - _gray_world_white_balance
# - _clahe_L
# - _lab_neutralize_chroma_aware
# - _reinhard_lab_transfer
#
# If they live in different modules, import them accordingly.

CLSA_PREPROCESSED_FILE_PATH = str(CLSA_DATA_ROOT / "clsa_multimodal.parquet")


class CLSAMultimodalDataset(Dataset):
    """
    CLSA Dataset with FU1+FU2 support via a generic irreversible-disease labeler.

    Expected (recommended) dataframe columns after preprocessing:
      - split
      - image_path
      - SEX_ASK_COM (or your chosen sex col)
      - AGE_NMBR_COM (or your chosen baseline age col)

    For disease progression labels (example: glaucoma), you should provide:
      - baseline_visit_date (Timestamp)
      - followup1_visit_date (Timestamp or NaT)
      - has_glaucoma_in_followup1 (bool/0/1 or NaN)
      - followup2_visit_date (Timestamp or NaT)
      - has_glaucoma_in_followup2 (bool/0/1 or NaN)
      - last_update_date (Timestamp) OR set self.last_update

    You can also provide a direct diagnosis date column and use label_date_name.
    """

    def __init__(
        self,
        baseline_followup_combined_path=str(CLSA_DATA_ROOT / "clsa.csv"),
        transform=None,
        image_size=224,
        progression_label_years: list = (0, 2, 5, 10),
        split="train",
        possible_inputs: List[str] = None,
        possible_labels: List[str] = None,
        qc=True,
        enhanced=True,
        df_conditions=None,
        fine_tuning=False,
        processed_df=None,
        progression_label_ignorant_label_years: int = 1,
        mean_std_map=None,
        normalise_fundus_image: bool = False,
        algo_qc: bool = False,
        clsa_qc_strategy: str = "legacy",
        clinical_numerical_features: Optional[List[str]] = None,
        clinical_categorical_features: Optional[List[str]] = None,
        prs_dir: str = CLSA_PRS_DIR,
        prs_files: Optional[Dict[str, str]] = None,
        prs_linking_key_path: str = CLSA_PRS_LINKING_KEY_PATH,
        preprocesed_df_path: str = CLSA_PREPROCESSED_FILE_PATH,
        # ----- NEW: generic label configuration -----
        # per-disease optional flag column to force label=None (e.g. glaucoma_should_be_none)
    ) -> None:
        super().__init__()

        self.split = split
        self.progression_label_years = list(progression_label_years)
        self.possible_inputs = possible_inputs or []
        self.possible_labels = possible_labels or []
        self.progression_label_ignorant_label_years = (
            progression_label_ignorant_label_years
        )
        self.normalise_fundus_image = normalise_fundus_image
        self.qc = qc
        self.algo_qc = algo_qc
        self.clsa_qc_strategy = clsa_qc_strategy
        self.mean_std_map = mean_std_map or {}
        self.prs_dir = prs_dir
        self.prs_files = prs_files or dict(CLSA_PRS_FILES)
        self.prs_linking_key_path = prs_linking_key_path
        self.prs_cols = list(PRS_COLS)
        self.clinical_numerical_features = (
            clinical_numerical_features or CLSA_NUMERICAL_FEATURES
        )
        self.clinical_categorical_features = (
            clinical_categorical_features or CLSA_CATEGORICAL_FEATURES
        )
        self.all_modalities = list(set(self.possible_inputs + self.possible_labels))

        # Labeling config
        self.transform = (
            get_default_aug(split=split, image_size=image_size, enhanced=enhanced)
            if transform is None
            else transform
        )

        if processed_df is not None:
            self.df = processed_df.copy()
            self.__init_prs_features()
        else:
            self.df = pd.read_csv(baseline_followup_combined_path, low_memory=False)
            self.__init_event_ages()
            self.__init_gender()
            self.__init_multimodal_features()
            self.__init_progression_label()
            self.df.to_parquet(preprocesed_df_path, index=False)

        if "image_path" in self.df.columns:
            original_paths = self.df["image_path"].copy()
            self.df["image_path"] = self.df["image_path"].map(
                normalise_clsa_image_path
            )
            remapped_count = int(
                (original_paths.fillna("") != self.df["image_path"].fillna("")).sum()
            )
            if remapped_count:
                logging.info(
                    "[CLSA] Remapped %d image paths from %s to %s",
                    remapped_count,
                    CLSA_LEGACY_DATA_ROOT,
                    CLSA_CURRENT_DATA_ROOT,
                )

        # QC
        if self.clsa_qc_strategy not in {"legacy", "algorithmic_only"}:
            raise ValueError(
                "clsa_qc_strategy must be 'legacy' or 'algorithmic_only', got "
                f"{self.clsa_qc_strategy!r}."
            )

        if self.clsa_qc_strategy == "algorithmic_only":
            pre_qc_len = len(self.df)
            algo_qc_df = pd.read_csv(CLSA_ALGO_QC_PATH).rename(
                columns={"is_bad": "algo_qc_is_bad"}
            )
            algo_qc_df["image_path"] = algo_qc_df["image_path"].map(
                normalise_clsa_image_path
            )
            if algo_qc_df["image_path"].duplicated().any():
                raise ValueError("Algorithmic CLSA QC contains duplicate image paths.")
            self.df = self.df.merge(
                algo_qc_df[["image_path", "algo_qc_is_bad"]],
                on="image_path",
                how="left",
                validate="many_to_one",
            )
            missing_qc = int(self.df["algo_qc_is_bad"].isna().sum())
            if missing_qc:
                raise ValueError(
                    "Algorithmic-only CLSA QC requires an explicit per-image label; "
                    f"{missing_qc} rows were unmatched."
                )
            self.df = self.df[self.df["algo_qc_is_bad"].eq(False)]
            logging.info(
                "CLSA Dataset [%s] | QC strategy: algorithmic_only | "
                "After QC filter: %d | Passing Ratio: %.4f",
                split,
                len(self.df),
                len(self.df) / pre_qc_len,
            )
        elif self.qc:
            pre_qc_len = len(self.df)

            # Algorithmic QC
            if self.algo_qc:
                algo_qc_df = pd.read_csv(CLSA_ALGO_QC_PATH).rename(
                    columns={"is_bad": "algo_qc_is_bad"}
                )
                algo_qc_df["image_path"] = algo_qc_df["image_path"].map(
                    normalise_clsa_image_path
                )

                # Attach algorithmic QC labels by image path.
                self.df = self.df.merge(
                    algo_qc_df[["image_path", "algo_qc_is_bad"]],
                    on="image_path",
                    how="left",
                )
            else:
                self.df["algo_qc_is_bad"] = False

            # Attach deep-learning QC labels by image name.
            dl_qc_path = CLSA_DL_QC_PATH
            dl_qc_df = pd.read_csv(dl_qc_path)
            # In predicted_label, 0 means bad quality and 1 means good quality.
            dl_qc_df["dl_qc_is_bad"] = dl_qc_df["predicted_label"] == 0

            self.df["image_name"] = self.df["image_path"].apply(
                lambda x: os.path.basename(x) if isinstance(x, str) else None
            )
            self.df = self.df.merge(
                dl_qc_df[["image_name", "dl_qc_is_bad"]],
                on="image_name",
                how="left",
            )

            # Exclude images that either QC source explicitly flags as bad.
            # Missing QC labels are not treated as failures here: comparisons to
            # True return False for NaN values, so unmatched images are retained.
            self.df = self.df[
                ~(
                    (self.df["algo_qc_is_bad"] == True)
                    | (self.df["dl_qc_is_bad"] == True)
                )
            ]

            logging.info(
                f"CLSA Dataset [{split}] | After QC filter: {len(self.df)} | Passing Ratio: {len(self.df) / pre_qc_len:.4f}"
            )

        logging.info(f"CLSA Dataset [{self.split}] | Total: {len(self.df)}")

        # Fine-tuning split
        if split is not None:
            if fine_tuning:
                self.df = self.df[self.df["finetune_split"] == split]
            else:
                self.df = self.df[self.df["split"] == split]

        # Extra dataframe-level conditions
        if df_conditions is not None:
            for condition in df_conditions:
                original_len = len(self.df)
                self.df = self.df[condition(self.df)]
                removed_length = original_len - len(self.df)
                logging.info(
                    f"[CLSA] | [{self.split}] | [{condition.__name__}] | "
                    f"Original instances: {original_len} | After instances: {len(self.df)} | "
                    f"Removed instances: {removed_length}"
                )

        self.__normalise_numerical_features()
        logging.info(f"Mean/Std Map: {self.mean_std_map}")

    # -----------------
    # Basic init fields
    # -----------------
    def __init_gender(self):
        # Note: baseline suffix depends on your merged dataframe; adjust if needed
        if "SEX_ASK_COM" in self.df.columns:
            self.df["patient_gender"] = self.df["SEX_ASK_COM"].apply(
                lambda x: 0 if x == "F" else 1 if x == "M" else None
            )
        else:
            self.df["patient_gender"] = None

    def __init_age(self):
        logging.info(f"Initializing ages for CLSA Dataset [{self.split}]...")
        if "AGE_NMBR_COM" in self.df.columns:
            self.df["instance_age_at_time"] = self.df["AGE_NMBR_COM"].apply(
                lambda x: _clean_numeric(x, min_value=0, max_value=120)
            )
        elif "AGE_NMBR_TRM" in self.df.columns:
            self.df["instance_age_at_time"] = self.df["AGE_NMBR_TRM"].apply(
                lambda x: _clean_numeric(x, min_value=0, max_value=120)
            )
        else:
            self.df["instance_age_at_time"] = np.nan

    def __normalise_numerical_features(self):
        numerical_cols = list(
            dict.fromkeys(self.clinical_numerical_features + self.prs_cols)
        )
        for col in numerical_cols:
            if col not in self.df.columns:
                continue

            if col not in self.mean_std_map:
                raise KeyError(f"Missing mean/std for {col} in mean_std_map")

            if (
                "mean" not in self.mean_std_map[col]
                or "std" not in self.mean_std_map[col]
            ):
                raise KeyError(f"Incomplete mean/std for {col} in mean_std_map")

            mu = self.mean_std_map[col]["mean"]
            sd = self.mean_std_map[col]["std"]

            if pd.isna(sd) or sd == 0:
                continue

            series = pd.to_numeric(self.df[col], errors="coerce")
            self.df[col] = series.apply(
                lambda x: ((x - mu) / sd) if not pd.isna(x) else np.nan
            )

    def __init_prs_features(self):
        df = self.df.copy()
        for col in self.prs_cols:
            if col not in df.columns:
                df[col] = np.nan

        if "entity_id" not in df.columns:
            logging.warning("[CLSA] Missing entity_id column; skipping PRS merge.")
            self.df = df
            return

        if not os.path.exists(self.prs_linking_key_path):
            logging.warning(
                f"[CLSA] PRS linking key not found: {self.prs_linking_key_path}"
            )
            self.df = df
            return

        try:
            linking_df = pd.read_csv(self.prs_linking_key_path, low_memory=False)
        except Exception as e:
            logging.warning(
                f"[CLSA] Failed reading PRS linking key "
                f"{self.prs_linking_key_path}: {e}"
            )
            self.df = df
            return

        required_link_cols = ["entity_id", "ADM_GWAS3_COM"]
        missing_link_cols = [
            col for col in required_link_cols if col not in linking_df.columns
        ]
        if missing_link_cols:
            logging.warning(
                f"[CLSA] Missing linking-key columns {missing_link_cols} in "
                f"{self.prs_linking_key_path}. "
                f"Available columns: {list(linking_df.columns)}"
            )
            self.df = df
            return

        df["_entity_id_str"] = df["entity_id"].apply(_normalise_sample_id)
        linking_tmp = linking_df[["entity_id", "ADM_GWAS3_COM"]].copy()
        linking_tmp["_entity_id_str"] = linking_tmp["entity_id"].apply(
            _normalise_sample_id
        )
        linking_tmp["_prs_iid_str"] = linking_tmp["ADM_GWAS3_COM"].apply(
            _normalise_sample_id
        )
        linking_tmp = linking_tmp[["_entity_id_str", "_prs_iid_str"]].dropna(
            subset=["_entity_id_str", "_prs_iid_str"]
        )
        linking_tmp = linking_tmp.drop_duplicates(
            subset=["_entity_id_str"], keep="first"
        )

        df = df.merge(linking_tmp, on="_entity_id_str", how="left")

        for out_col, filename in self.prs_files.items():
            prs_path = os.path.join(self.prs_dir, filename)
            if not os.path.exists(prs_path):
                logging.warning(f"[CLSA] PRS file not found for {out_col}: {prs_path}")
                continue

            try:
                prs_df = pd.read_csv(prs_path, sep=r"\s+", engine="python")
            except Exception as e:
                logging.warning(f"[CLSA] Failed reading PRS file {prs_path}: {e}")
                continue

            required_cols = ["IID", "SCORE1_SUM"]
            missing_cols = [col for col in required_cols if col not in prs_df.columns]
            if missing_cols:
                logging.warning(
                    f"[CLSA] Missing required PRS columns {missing_cols} in {prs_path}. "
                    f"Available columns: {list(prs_df.columns)}"
                )
                continue

            prs_tmp = prs_df[["IID", "SCORE1_SUM"]].copy()
            prs_tmp["_prs_iid_str"] = prs_tmp["IID"].apply(_normalise_sample_id)
            prs_tmp[out_col] = pd.to_numeric(prs_tmp["SCORE1_SUM"], errors="coerce")
            prs_tmp = prs_tmp[["_prs_iid_str", out_col]].dropna(subset=["_prs_iid_str"])
            prs_tmp = prs_tmp.drop_duplicates(subset=["_prs_iid_str"], keep="first")

            df = df.drop(columns=[out_col], errors="ignore").merge(
                prs_tmp,
                on="_prs_iid_str",
                how="left",
            )
            logging.info(
                f"[CLSA] Loaded PRS {out_col} from {filename}: "
                f"link_key_matched={df['_prs_iid_str'].notna().sum()}/{len(df)}, "
                f"prs_matched={df[out_col].notna().sum()}/{len(df)}"
            )

            # throw error if linking key was matched but no PRS values found, as this likely indicates a format mismatch
            if df["_prs_iid_str"].notna().any() and df[out_col].notna().sum() == 0:
                raise ValueError(
                    f"[CLSA] Linking key matched for {out_col} but no valid PRS values found. "
                    f"Check if the PRS file {prs_path} has the expected format with 'IID' and 'SCORE1_SUM' columns."
                )

        df = df.drop(columns=["_entity_id_str", "_prs_iid_str"], errors="ignore")
        self.df = df

    def __pick_iop_for_side(self, row):
        side = row.get("side")
        right_iop = _clean_numeric(
            row.get("TON_IOPCC_R_COM"), min_value=1, max_value=80
        )
        left_iop = _clean_numeric(row.get("TON_IOPCC_L_COM"), min_value=1, max_value=80)

        if isinstance(side, str):
            side = side.lower()
            if side == "right":
                return right_iop
            if side == "left":
                return left_iop

        vals = [v for v in [left_iop, right_iop] if not np.isnan(v)]
        if len(vals) == 0:
            return np.nan
        return float(np.mean(vals))

    def __compute_packyears(self, row):
        # CLSA *_COM smoking fields are categorical bins, not raw cigarettes/day.
        # Use category midpoints so pack-years is on a numeric scale.
        cat_cigs_per_day = {
            1.0: 3.0,  # 1-5
            2.0: 8.0,  # 6-10
            3.0: 13.0,  # 11-15
            4.0: 18.0,  # 16-20
            5.0: 23.0,  # 21-25
            6.0: 28.0,  # 26+
        }

        def _to_cigs_per_day(col_name, value):
            if col_name in {"SMK_FRQDL_COM", "SMK_NBDL_COM"}:
                code = _to_float_or_nan(value)
                if np.isnan(code):
                    return np.nan
                return cat_cigs_per_day.get(code, np.nan)
            return _clean_numeric(value, min_value=0, max_value=200)

        def _first_valid(col_names, *, min_value, max_value):
            for c in col_names:
                if c in row.index:
                    v = _to_cigs_per_day(c, row.get(c))
                    if not np.isnan(v):
                        v = _clean_numeric(v, min_value=min_value, max_value=max_value)
                    if not np.isnan(v):
                        return v
            return np.nan

        current_cigs = _first_valid(
            ["SMK_FRQDL_NB_COM", "SMK_FRQDL_COM"], min_value=0, max_value=200
        )
        current_years = _first_valid(["SMK_YRDL_NB_COM"], min_value=0, max_value=120)
        past_cigs = _first_valid(
            ["SMK_NBDL_NB_COM", "SMK_NBDL_COM"], min_value=0, max_value=200
        )
        past_years = _first_valid(["SMK_TOTYR_NB_COM"], min_value=0, max_value=120)

        candidates = []
        if not np.isnan(current_cigs) and not np.isnan(current_years):
            candidates.append(current_cigs / 20.0 * current_years)
        if not np.isnan(past_cigs) and not np.isnan(past_years):
            candidates.append(past_cigs / 20.0 * past_years)

        if len(candidates) == 0:
            return np.nan
        return float(max(candidates))

    def __compute_alcohol_unitsweek(self, row):
        # Harmonize CLSA ALC_FREQ_COM categories to UKB-style frequency bins.
        # Note: despite the historical function name, this returns mapped category codes.
        freq = _to_float_or_nan(row.get("ALC_FREQ_COM"))
        if np.isnan(freq):
            return np.nan
        clsa_to_ukb = {
            1.0: 1.0,  # almost every day -> daily/almost daily
            2.0: 2.0,  # 4-5 times/week -> 3-4 times/week (closest)
            3.0: 3.0,  # 2-3 times/week -> once or twice/week (closest)
            4.0: 3.0,  # once/week -> once or twice/week
            5.0: 4.0,  # 2-3 times/month -> one to three times/month
            6.0: 4.0,  # about once/month -> one to three times/month
            7.0: 5.0,  # less than once/month -> special occasions only
            96.0: 6.0,  # never
            98.0: -1.0,  # don't know/no answer
            99.0: -3.0,  # refused/prefer not to answer
            77.0: np.nan,  # missing/not applicable
        }
        return clsa_to_ukb.get(freq, np.nan)

    def __compute_alcohol_drinker_status(self, row):
        ever = _to_float_or_nan(row.get("ALC_EVER_COM"))
        freq = _to_float_or_nan(row.get("ALC_FREQ_COM"))

        # UKB field 20117 is drinker status: current / previous / never.
        if np.isnan(ever):
            return np.nan
        if ever == 9:
            return -3.0
        if ever == 8:
            return -1.0  # Don't know → Do not know
        if ever == 2:
            return 0.0
        if ever != 1:
            return np.nan

        if np.isnan(freq):
            return np.nan
        if freq == 99:
            return -3.0
        if freq == 98:
            return -1.0  # Don't know → Do not know
        if freq == 77:
            return np.nan  # Not applicable
        if freq == 96:
            return 1.0
        if freq in {1, 2, 3, 4, 5, 6, 7}:
            return 2.0
        return np.nan

    def __current_hrt_status(self, row):
        gender = row.get("SEX_ASK_COM")
        if gender != "F":
            return 0.0

        age = _clean_numeric(row.get("AGE_NMBR_COM"), min_value=0, max_value=120)
        start_age = _clean_numeric(
            row.get("WHO_HRTAG_AG_COM"), min_value=0, max_value=120
        )
        duration_years = _clean_numeric(
            row.get("WHO_HRTYR_YR_COM"), min_value=0, max_value=80
        )

        if np.isnan(start_age) or np.isnan(duration_years) or np.isnan(age):
            return np.nan

        return float(start_age <= age <= (start_age + duration_years))

    def __init_multimodal_features(self):
        self.__init_age()

        df = self.df

        for col in (
            PRS_COLS
            + ANTHROPOMETRICS_COLS
            + ANCESTRY_COLS
            + FAMILY_HISTORY_COLS
            + PRINCIPAL_COMPONENT_COLS
            + LIFESTYLE_COLS
            + MENTAL_HEALTH_COLS
            + SOCIOECONOMIC_COLS
            + VITALS_COLS
            + MEDICATIONS_COLS
            + ["instance_height_cm", "instance_iop"]
        ):
            if col not in df.columns:
                df[col] = np.nan

        df["instance_iop"] = df.apply(self.__pick_iop_for_side, axis=1)

        # df["instance_height_cm"] = df["DXA_WB_HEIGHT_COM"].apply(
        #     lambda x: _clean_numeric(x, min_value=50, max_value=250, invalid_values={-1})
        # )

        df["instance_height_cm"] = np.nan

        df["instance_weight_kg"] = df["DXA_WB_WEIGHT_COM"].apply(
            lambda x: _clean_numeric(
                x, min_value=20, max_value=400, invalid_values={-1}
            )
        )
        df["instance_body_mass_index_bmi"] = df["HWT_DBMI_COM"].apply(
            lambda x: _clean_numeric(x, min_value=10, max_value=100)
        )
        df["instance_waist_circumference_cm"] = df["WHC_WAIST_CM_COM"].apply(
            lambda x: _clean_numeric(x, min_value=30, max_value=250)
        )
        df["instance_hip_circumference_cm"] = df["WHC_HIP_CM_COM"].apply(
            lambda x: _clean_numeric(x, min_value=30, max_value=250)
        )

        df["instance_systolic_bp"] = df["BP_SYSTOLIC_2_COM"].apply(
            lambda x: _clean_numeric(
                x, min_value=50, max_value=300, invalid_values={-888}
            )
        )
        df["instance_diastolic_bp"] = df["BP_DIASTOLIC_2_COM"].apply(
            lambda x: _clean_numeric(
                x, min_value=20, max_value=200, invalid_values={-888}
            )
        )
        df["instance_pulse_rate"] = df["BP_PULSE_2_COM"].apply(
            lambda x: _clean_numeric(
                x, min_value=20, max_value=250, invalid_values={-888}
            )
        )

        smoking_map = {1.0: 2.0, 2.0: 0.0, 3.0: 1.0, 8.0: -1.0, 9.0: -3.0}
        df["instance_smoking_status"] = pd.to_numeric(
            df["ICQ_SMOKE_COM"], errors="coerce"
        ).map(smoking_map)
        df["instance_packyears_of_smoking"] = df.apply(self.__compute_packyears, axis=1)

        df["instance_alcohol_intake_frequency"] = df.apply(
            self.__compute_alcohol_drinker_status, axis=1
        )
        df["instance_alcohol_consumption_unitsweek"] = df.apply(
            self.__compute_alcohol_unitsweek, axis=1
        )

        # selfrated_map = {1.0: 1.0, 2.0: 1.0, 3.0: 2.0, 4.0: 3.0, 5.0: 4.0}
        selfrated_map = {
            1.0: 1.0,
            2.0: 2.0,
            3.0: 2.0,
            4.0: 3.0,
            5.0: 4.0,
            8.0: -1.0,
            9.0: -3.0,
        }
        df["instance_selfrated_health"] = pd.to_numeric(
            df["GEN_HLTH_COM"], errors="coerce"
        ).map(selfrated_map)
        df["instance_sleep_duration_hoursnight"] = df["SLE_HOUR_NB_COM"].apply(
            lambda x: _clean_numeric(
                x, min_value=0, max_value=24, invalid_values={-88, 88, 98}
            )
        )
        df["instance_type_of_milk_usually_consumed"] = np.nan
        df["instance_cheese_intake"] = np.nan
        df["instance_processed_meat_intake"] = np.nan
        df["instance_poultry_intake"] = np.nan
        df["instance_comparative_body_size_at_age_10"] = np.nan

        def _combine_binary_any(row):
            vals = [_binary_yes_no(row.iloc[0]), _binary_yes_no(row.iloc[1])]
            real = [v for v in vals if not np.isnan(v)]
            if len(real) == 0:
                return np.nan
            if 1.0 in real:
                return 1.0  # OR: any Yes → Yes
            if all(v == 0.0 for v in real):
                return 0.0  # all No → No
            # At least one uncertain (8/9), no Yes → return most informative special code
            special = [v for v in real if v < 0]
            return float(max(special)) if special else 0.0  # -1.0 preferred over -3.0

        df["instance_seen_doctor_nerves_anxiety_tension_or_depression"] = df[
            ["CCC_ANXI_COM", "DPR_CLINDEP_COM"]
        ].apply(_combine_binary_any, axis=1)

        # maybe droping this one, because it's impossible to map the yes no from CCC_MOOD_COM to 6 codes in UKB.
        # df["instance_bipolar_and_major_depression_status"] = df["CCC_MOOD_COM"].apply(
        #     _binary_yes_no
        # )

        # dep = pd.to_numeric(df["DEP_FLDP_COM"], errors="coerce")
        # df["instance_ever_depressed_for_a_whole_week"] = dep.map(
        #     {1.0: 1.0, 2.0: 1.0, 3.0: 1.0, 4.0: 0.0, 8.0: -1.0, 9.0: -3.0}
        # )

        edu = pd.to_numeric(df["ED_UDR11_COM"], errors="coerce")
        df["instance_qualifications_college_or_university_degree"] = edu.isin(
            [9, 10, 11]
        ).astype(float)
        df["instance_qualifications_a_levels_as_levels_or_equivalent"] = edu.isin(
            [8]
        ).astype(float)
        df["instance_qualifications_o_levels_gcse_or_equivalent"] = edu.isin(
            [5]
        ).astype(float)
        df["instance_qualifications_nvq_or_hnd_or_hnc_or_equivalent"] = edu.isin(
            [6, 7]
        ).astype(float)
        df["instance_qualifications_other_professional_qualifications"] = edu.isin(
            [1, 2, 3, 4]
        ).astype(float)
        df["instance_qualifications_prefer_not_to_answer"] = edu.isin([99]).astype(
            float
        )
        valid_edu = edu.isin([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 99])
        unknown_edu = ~valid_edu
        for col in [
            "instance_qualifications_college_or_university_degree",
            "instance_qualifications_a_levels_as_levels_or_equivalent",
            "instance_qualifications_o_levels_gcse_or_equivalent",
            "instance_qualifications_nvq_or_hnd_or_hnc_or_equivalent",
            "instance_qualifications_other_professional_qualifications",
            "instance_qualifications_prefer_not_to_answer",
        ]:
            df.loc[unknown_edu, col] = np.nan

        work = pd.to_numeric(df["LBF_STTS_COM"], errors="coerce")
        retired = pd.to_numeric(df["RET_RTRD_COM"], errors="coerce")
        df["instance_current_employment_paid_or_self_employed"] = work.isin([1]).astype(
            float
        )
        df["instance_current_employment_unemployed"] = work.isin([2]).astype(float)
        df["instance_current_employment_looking_after_home_and_or_family"] = work.isin(
            [3]
        ).astype(float)
        df["instance_current_employment_retired"] = retired.isin([1]).astype(float)
        df["instance_current_employment_prefer_not_to_answer"] = (
            work.isin([8, 9]) | retired.isin([8, 9])
        ).astype(float)
        valid_work = work.isin([1, 2, 3, 8, 9])
        unknown_work = ~valid_work
        for col in [
            "instance_current_employment_paid_or_self_employed",
            "instance_current_employment_unemployed",
            "instance_current_employment_looking_after_home_and_or_family",
            "instance_current_employment_retired",
            "instance_current_employment_prefer_not_to_answer",
        ]:
            df.loc[unknown_work, col] = np.nan

        df["instance_med_blood_pressure"] = df["HBP_MED_COM"].apply(
            lambda x: 1.0 if _to_float_or_nan(x) == 1.0 else 0.0
        )
        df["instance_med_diabetes"] = df["DIA_MEDAGE_NB_COM"].apply(
            lambda x: float(not np.isnan(_clean_numeric(x, min_value=0, max_value=120)))
        )
        df["instance_med_hormone_replacement"] = df.apply(
            self.__current_hrt_status, axis=1
        ).fillna(0.0)

        # Ancestry: use first available wave column, map to harmonized codes
        ancestry_candidates = ["SPR_OUTPUT_ETHN_COF1", "SPR_OUTPUT_ETHN_COF2", "SPR_OUTPUT_ETHN_COM"]
        ancestry_source = next((c for c in ancestry_candidates if c in df.columns), None)
        if ancestry_source is not None:
            raw = pd.to_numeric(df[ancestry_source], errors="coerce")
            df["patient_ancestry"] = raw.apply(
                lambda x: float(CLSA_ANCESTRY_MAP[int(x)]) if (not pd.isna(x) and int(x) in CLSA_ANCESTRY_MAP) else np.nan
            )
        else:
            df["patient_ancestry"] = np.nan

        self.df = df
        self.__init_prs_features()

    # -------------------------
    # Generic irreversible label
    # -------------------------
    @staticmethod
    def _nz(x):
        return None if (x is None or (hasattr(pd, "isna") and pd.isna(x))) else x

    def has_disease_by_n_years_age(
        self,
        row,
        n: int,
        *,
        # baseline
        baseline_age_col: str = "age",
        # event (age at first diagnosis)
        event_age_col: str,
        # follow-up coverage (max age observed; if you have FU2 age, use it)
        followup_age_col: str = "AGE_NMBR_COF2",  # optional; can be None if you only do prevalent
        # optional flag to force None (your glaucoma_should_be_none logic)
        positive_buffer_years: float = 0.25,  # ~90 days
        ignorant_buffer_years: int = 1,
    ):
        """
        Tri-state cumulative label on AGE scale for irreversible diseases.

        True:
        - baseline prevalent (event_age <= baseline_age), OR
        - incident within n years (+ buffer): event_age <= baseline_age + n + buffer

        False:
        - no known event_age AND follow-up covers >= n + ignorant years

        None:
        - event in (n, n+y] window, OR
        - follow-up does not cover enough time to decide

        Notes:
        - CLSA has no exact dates: use ages.
        - followup_age_col should represent the latest observed age at follow-up (e.g., FU2 age).
        """
        base_age = row.get(baseline_age_col)
        evt_age = row.get(event_age_col)
        fu_age = row.get(followup_age_col) if followup_age_col is not None else None

        # normalize NaNs
        def _nz(x):
            return (
                None
                if (x is None or (hasattr(pd, "isna") and pd.isna(x)))
                else float(x)
            )

        base_age, evt_age, fu_age = map(_nz, (base_age, evt_age, fu_age))

        if base_age is None:
            return None

        # horizons on age scale
        n_cut = base_age + float(n)
        n_buf_cut = n_cut + float(positive_buffer_years)
        ny_cut = base_age + float(n + ignorant_buffer_years)

        # --- event known ---
        if evt_age is not None:
            # prevalent
            if evt_age <= base_age:
                return True
            # incident within n (+ buffer)
            if evt_age <= n_buf_cut:
                return True
            # event in (n, n+y] => uncertain
            if evt_age <= ny_cut:
                return None
            # event after n+y => can only say False if follow-up covers n+y
            if fu_age is not None and fu_age >= ny_cut:
                return False
            return None

        # --- no event recorded ---
        if fu_age is None:
            return None
        if fu_age >= ny_cut:
            return False
        if fu_age >= n_cut:
            return None
        return None

    def __init_event_ages(self):
        logging.info(f"Initializing event ages for CLSA Dataset [{self.split}]...")
        """
        Initialise age-at-diagnosis proxy columns for irreversible diseases in CLSA.

        This function is intended to be called once after `self.df` is loaded.
        It creates:
        - followup_age
        - glaucoma_event_age
        - diabetes_event_age
        - alzh_event_age
        - cvd_event_age

        Assumptions:
        - Baseline age is in column: 'age' (float/int).
        - Follow-up ages exist in one or more of: AGE_NMBR_COF1, AGE_NMBR_COF2
            (adjust the candidate list below to match your actual column names).
        - Disease age-at-diagnosis fields are the ones you listed (COF2 versions).

        Notes:
        - CLSA does not provide exact diagnosis dates; we use age scale.
        - Event age is taken as the minimum of relevant component ages (earliest onset proxy).
        - Basic sanity filters are applied (e.g., <0 or >120 set to NaN).
        """
        df = self.df

        def _min_cols(cols):
            cols = [c for c in cols if c in df.columns]
            if len(cols) == 0:
                return pd.Series(np.nan, index=df.index)
            return df[cols].min(axis=1, skipna=True)

        all_original_event_cols = [
            "ICQ_GLAUCAGE_NB_COF2",
            "DIA_AGE_NB_COF2",
            "PKD_AGE_NB_COF2",
            "CCC_ALZHAGE_NB_COF2",
            "STR_TIAAGE_NB_COF2",
            "STR_CVAAGE_NB_COF2",
            "IHD_ANGIAGE_NB_COF2",
            "IHD_AMIAGE_NB_COF2",
            "CCC_HEARTAGE_NB_COF2",
            "CCC_PADAGE_NB_COF2",
            "AOR_AORAGE_NB_COF2",
        ]

        # remove unrealistic numbers
        df[all_original_event_cols] = df[all_original_event_cols].apply(
            lambda s: s.where(s.between(0, 200))
        )

        # -------------------------
        # Glaucoma event age
        # -------------------------
        # You listed:
        #   ICQ_GLAUCAGE_NB_COF2
        #   DIA_AGE_NB_COF2  (only include here if it truly corresponds to glaucoma in your data)
        glaucoma_age_cols = [
            "ICQ_GLAUCAGE_NB_COF2",
            # "DIA_AGE_NB_COF2",  # uncomment ONLY if this is actually glaucoma age in your schema
        ]
        df["glaucoma_event_age"] = _min_cols(glaucoma_age_cols)

        # -------------------------
        # Diabetes event age
        # -------------------------
        diabetes_age_cols = [
            "DIA_AGE_NB_COF2",
        ]
        df["t2d_event_age"] = _min_cols(diabetes_age_cols)

        # -------------------------
        # Alzheimer's event age
        # -------------------------
        alzh_age_cols = [
            "CCC_ALZHAGE_NB_COF2",
        ]
        df["ad_event_age"] = _min_cols(alzh_age_cols)

        # -------------------------
        # PD event age
        # -------------------------
        alzh_age_cols = [
            "PKD_AGE_NB_COF2",
        ]
        df["pd_event_age"] = _min_cols(alzh_age_cols)

        # -------------------------
        # CVD composite event age
        # -------------------------
        # You listed the COF2 set:
        #   STR_TIAAGE_NB_COF2, STR_CVAAGE_NB_COF2, IHD_ANGIAGE_NB_COF2,
        #   IHD_AMIAGE_NB_COF2, CCC_HEARTAGE_NB_COF2
        cvd_component_cols = [
            "STR_TIAAGE_NB_COF2",
            "STR_CVAAGE_NB_COF2",
            "IHD_ANGIAGE_NB_COF2",
            "IHD_AMIAGE_NB_COF2",
            "CCC_HEARTAGE_NB_COF2",
            # Optional extras if present/desired:
            # "CCC_PADAGE_NB_COF2",
            # "AOR_AORAGE_NB_COF2",
        ]
        df["cvd_event_age"] = _min_cols(cvd_component_cols)

        # -------------------------
        # Sanity filters for ages
        # -------------------------
        # Set implausible ages to NaN.
        event_cols = [
            "glaucoma_event_age",
            "t2d_event_age",
            "ad_event_age",
            "cvd_event_age",
            "pd_event_age",
        ]

        for c in event_cols:
            if c not in df.columns:
                continue
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df.loc[(df[c] < 0) | (df[c] > 200), c] = np.nan

        # Logging summary
        for c in event_cols:
            if c in df.columns:
                logging.info(
                    f"[CLSA] {c}: non-missing={df[c].notna().sum()} | "
                    f"min={df[c].min(skipna=True)} | max={df[c].max(skipna=True)}"
                )

        self.df = df

    # Progression label creation
    # -------------------------
    def __init_progression_label(self):
        logging.info(
            f"Initializing progression labels for CLSA Dataset [{self.split}]..."
        )
        diseases = [
            "glaucoma",
            "ad",
            "pd",
            "t2d",
            "cvd",
        ]

        for disease in diseases:
            logging.info(f"Processing progression labels for disease: {disease}")
            event_age_col = f"{disease}_event_age"
            if event_age_col not in self.df.columns:
                logging.warning(f"[CLSA] Missing {event_age_col}, skipping {disease}.")
                continue

            for year in self.progression_label_years:
                logging.info(
                    f"Processing {disease} progression label for {year} years..."
                )
                out_col = f"has_{disease}_in_{year}_years"
                self.df[out_col] = self.df.apply(
                    lambda r: self.has_disease_by_n_years_age(
                        r,
                        n=year,
                        event_age_col=event_age_col,
                        positive_buffer_years=90 / 365,
                        ignorant_buffer_years=self.progression_label_ignorant_label_years,
                    ),
                    axis=1,
                )

                logging.info(f"CLSA Dataset [{self.split}] | {disease} in {year} years")
                logging.info(self.df[out_col].value_counts(dropna=False))

    # def __init_progression_label(self):
    #     diseases = ["glaucoma", "ad", "t2d", "cvd"]
    #     years = list(self.progression_label_years)
    #     pos_buf = 90 / 365

    #     for disease in diseases:
    #         event_age_col = f"{disease}_event_age"
    #         if event_age_col not in self.df.columns:
    #             logging.warning(f"[CLSA] Missing {event_age_col}, skipping {disease}.")
    #             continue

    #         # preallocate [N, Y] float32 (NaN = None/uncertain)
    #         labels = np.full((len(self.df), len(years)), np.nan, dtype=np.float32)

    #         # row-by-row compute (no pandas apply)
    #         for i in range(len(self.df)):
    #             r = self.df.iloc[i]
    #             for j, year in enumerate(years):
    #                 v = self.has_disease_by_n_years_age(
    #                     r,
    #                     n=year,
    #                     event_age_col=event_age_col,
    #                     positive_buffer_years=pos_buf,
    #                     ignorant_buffer_years=self.progression_label_ignorant_label_years,
    #                 )
    #                 labels[i, j] = np.nan if v is None else (1.0 if v else 0.0)

    #             if i % 50000 == 0 and i > 0:
    #                 logging.info(f"[CLSA] {disease}: processed {i}/{len(self.df)} rows")

    #         # write columns back (still compact float32)
    #         for j, year in enumerate(years):
    #             out_col = f"has_{disease}_in_{year}_years"
    #             self.df[out_col] = labels[:, j]  # float32 column
    #             logging.info(self.df[out_col].value_counts(dropna=False))

    # -----------------
    # torch dataset API
    # -----------------
    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def add_output_modality(output_dict, key, value):
        if value is not None:
            output_dict.update({key: value})

    @staticmethod
    def get_label_col(data, label_col):
        if label_col not in data:
            return None
        v = data[label_col]
        if v is None:
            return None
        try:
            if math.isnan(v):
                return None
        except Exception:
            pass
        return torch.tensor(data.loc[[label_col]]).float()

    @staticmethod
    def get_label_cols(data, label_cols):
        vals = pd.to_numeric(data.loc[label_cols], errors="coerce").to_numpy(
            dtype=np.float32, copy=True
        )
        return torch.from_numpy(vals)

    def grab_modalities(self, data, modalities):
        output = {}

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

        if "prs" in modalities:
            self.add_output_modality(output, "prs", self.get_label_cols(data, PRS_COLS))

        prs_singletons = {
            "glaucoma_prs": "patient_Enhanced PRS for primary open angle glaucoma (POAG)",
            "ad_prs": "patient_Enhanced PRS for alzheimer's disease (AD)",
            "pd_prs": "patient_Enhanced PRS for parkinson's disease (PD)",
            "ms_prs": "patient_Enhanced PRS for multiple sclerosis (MS)",
            "t2d_prs": "patient_Enhanced PRS for type 2 diabetes (T2D)",
            "cvd_prs": "patient_Enhanced PRS for cardiovascular disease (CVD)",
        }
        for modality, col in prs_singletons.items():
            if modality in modalities:
                self.add_output_modality(
                    output, modality, self.get_label_col(data, col)
                )

        if "clinical-cat" in modalities:
            self.add_output_modality(
                output,
                "clinical-cat",
                self.get_label_cols(data, self.clinical_categorical_features),
            )

        if "clinical-num" in modalities:
            self.add_output_modality(
                output,
                "clinical-num",
                self.get_label_cols(data, self.clinical_numerical_features),
            )

        if "anthropometrics" in modalities:
            self.add_output_modality(
                output,
                "anthropometrics",
                self.get_label_cols(data, ANTHROPOMETRICS_COLS),
            )

        if "family_history" in modalities:
            self.add_output_modality(
                output, "family_history", self.get_label_cols(data, FAMILY_HISTORY_COLS)
            )

        if "principal_components" in modalities:
            self.add_output_modality(
                output,
                "principal_components",
                self.get_label_cols(data, PRINCIPAL_COMPONENT_COLS),
            )

        if "lifestyle" in modalities:
            self.add_output_modality(
                output, "lifestyle", self.get_label_cols(data, LIFESTYLE_COLS)
            )

        if "mental_health" in modalities:
            self.add_output_modality(
                output, "mental_health", self.get_label_cols(data, MENTAL_HEALTH_COLS)
            )

        if "socioeconomic" in modalities:
            self.add_output_modality(
                output, "socioeconomic", self.get_label_cols(data, SOCIOECONOMIC_COLS)
            )

        if "vitals" in modalities:
            self.add_output_modality(
                output, "vitals", self.get_label_cols(data, VITALS_COLS)
            )

        if "medications" in modalities:
            self.add_output_modality(
                output, "medications", self.get_label_cols(data, MEDICATIONS_COLS)
            )

        if "ancestry" in modalities:
            self.add_output_modality(
                output, "ancestry", self.get_label_cols(data, ANCESTRY_COLS)
            )

        single_value_modalities = {
            "height_cm": "instance_height_cm",
            "weight_kg": "instance_weight_kg",
            "body_mass_index_bmi": "instance_body_mass_index_bmi",
            "waist_circumference_cm": "instance_waist_circumference_cm",
            "hip_circumference_cm": "instance_hip_circumference_cm",
            "comparative_body_size_at_age_10": "instance_comparative_body_size_at_age_10",
            "smoking_status": "instance_smoking_status",
            "packyears_of_smoking": "instance_packyears_of_smoking",
            "alcohol_intake_frequency": "instance_alcohol_intake_frequency",
            "alcohol_consumption_unitsweek": "instance_alcohol_consumption_unitsweek",
            "selfrated_health": "instance_selfrated_health",
            "sleep_duration_hoursnight": "instance_sleep_duration_hoursnight",
            "type_of_milk_usually_consumed": "instance_type_of_milk_usually_consumed",
            "cheese_intake": "instance_cheese_intake",
            "processed_meat_intake": "instance_processed_meat_intake",
            "poultry_intake": "instance_poultry_intake",
            "seen_doctor_nerves_anxiety_tension_or_depression": "instance_seen_doctor_nerves_anxiety_tension_or_depression",
            "ever_depressed_for_a_whole_week": "instance_ever_depressed_for_a_whole_week",
            "bipolar_and_major_depression_status": "instance_bipolar_and_major_depression_status",
            "systolic_bp": "instance_systolic_bp",
            "diastolic_bp": "instance_diastolic_bp",
            "pulse_rate": "instance_pulse_rate",
            "med_blood_pressure": "instance_med_blood_pressure",
            "med_diabetes": "instance_med_diabetes",
            "med_hormone_replacement": "instance_med_hormone_replacement",
        }

        for modality, col in single_value_modalities.items():
            if modality in modalities:
                self.add_output_modality(
                    output, modality, self.get_label_col(data, col)
                )

        for label in self.possible_labels:
            self.add_output_modality(output, label, self.get_label_col(data, label))

        return output

    # -----------------
    # image loading
    # -----------------
    def get_fundus_image_normalised(
        self,
        data,
        *,
        pad=16,
        threshold=40,
        size=None,
        debug=False,
        illum_norm="post",
        clahe_clip=2.0,
        clahe_tiles=(8, 8),
        white_balance=True,
        color_norm="none",
        target_a=0.0,
        target_b=0.0,
        neutralize_strength=0.5,
        chroma_floor=10.0,
        chroma_soft=8.0,
        ref_lab_means=None,
        ref_lab_stds=None,
    ):
        img_path = data["image_path"]
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img)
        H, W, _ = arr.shape

        if debug:
            print(f"[DEBUG] {img_path}  W={W} H={H}")

        if white_balance:
            arr = _gray_world_white_balance(arr)
        if illum_norm == "pre":
            arr = _clahe_L(arr, clip=clahe_clip, tiles=clahe_tiles)

        L = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)[..., 0]
        fg = L > threshold

        if not fg.any():
            out = Image.fromarray(arr)
        else:
            ys = np.where(fg.any(axis=1))[0]
            top = max(int(ys[0]) - pad, 0)
            bottom = min(int(ys[-1]) + pad, H - 1)
            h_d = bottom - top

            cx = W // 2
            left = max(cx - h_d // 2, 0)
            right = min(cx + h_d // 2, W - 1)

            out = Image.fromarray(arr).crop((left, top, right + 1, bottom + 1))

        arr_c = np.asarray(out)

        if illum_norm == "post":
            arr_c = _clahe_L(arr_c, clip=clahe_clip, tiles=clahe_tiles)

        if color_norm == "neutralize":
            arr_c = _lab_neutralize_chroma_aware(
                arr_c,
                target_a=target_a,
                target_b=target_b,
                strength=neutralize_strength,
                chroma_floor=chroma_floor,
                chroma_soft=chroma_soft,
            )
        elif color_norm == "reinhard":
            if ref_lab_means is None or ref_lab_stds is None:
                raise ValueError(
                    "Provide ref_lab_means and ref_lab_stds for 'reinhard'."
                )
            arr_c = _reinhard_lab_transfer(arr_c, ref_lab_means, ref_lab_stds)

        out = Image.fromarray(arr_c)

        if size is not None:
            out = out.resize((size, size), resample=Image.BICUBIC)

        return out

    def get_fundus_image(self, data):
        img_path = data["image_path"]
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img)
        H, W, _ = arr.shape

        cx = W // 2
        out = Image.fromarray(arr).crop((cx - H // 2, 0, cx + H // 2, H))
        return out

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.df.iloc[idx]
        all_modalities = list(set(self.possible_inputs + self.possible_labels))
        modalities = self.grab_modalities(data, all_modalities)
        modalities.update({"dataset": "clsa", "split": self.split, "idx": idx})
        return modalities

    # -----------------
    # sampling weights
    # -----------------
    def get_sampling_weights(self):
        logging.info("Computing sample weights...")

        possible_labels = self.possible_labels
        label_value_counts = {label: Counter() for label in possible_labels}

        df = self.df
        for label in possible_labels:
            if label in df.columns:
                label_value_counts[label].update(df[label].dropna().astype(int).values)

        label_class_weights = {}
        for label, counter in label_value_counts.items():
            total = sum(counter.values())
            if total == 0:
                continue
            label_class_weights[label] = {
                cls: total / count for cls, count in counter.items()
            }

        all_sample_weights = []
        for _, row in df.iterrows():
            weights = []
            for label in possible_labels:
                if label in row and pd.notna(row[label]):
                    value = int(row[label])
                    if (
                        label in label_class_weights
                        and value in label_class_weights[label]
                    ):
                        weights.append(label_class_weights[label][value])
            sample_weight = (sum(weights) / len(weights)) if weights else 1.0
            all_sample_weights.append(sample_weight)

        return all_sample_weights


def build_clsa_multimodal_datasets(args, **kwargs):
    train_transform = get_default_aug(image_size=args.image_size, split="train")
    test_transform = get_default_aug(image_size=args.image_size, split="test")

    print("Building CLSA datasets... | Image size:", args.image_size)

    train_dataset = CLSAMultimodalDataset(
        transform=train_transform,
        split="train",
        progression_label_years=kwargs.get("progression_label_years", None),
        possible_labels=kwargs.get("possible_labels", []),
    )

    val_dataset = CLSAMultimodalDataset(
        transform=test_transform,
        split="val",
        progression_label_years=kwargs.get("progression_label_years", None),
        possible_labels=kwargs.get("possible_labels", []),
    )

    test_dataset = CLSAMultimodalDataset(
        transform=test_transform,
        split="test",
        progression_label_years=kwargs.get("progression_label_years", None),
        possible_labels=kwargs.get("possible_labels", []),
    )

    train_dataset.possible_inputs = kwargs["possible_inputs"]
    train_dataset.possible_labels = kwargs["possible_labels"]

    val_dataset.possible_inputs = kwargs["possible_inputs"]
    val_dataset.possible_labels = kwargs["possible_labels"]

    test_dataset.possible_inputs = kwargs["possible_inputs"]
    test_dataset.possible_labels = kwargs["possible_labels"]

    return train_dataset, val_dataset, test_dataset
