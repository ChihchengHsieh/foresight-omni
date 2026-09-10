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
from dateutil.relativedelta import relativedelta

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


from collections import defaultdict, Counter

class CLSADataset(Dataset):
    def __init__(
        self,
        baseline_followup_combined_path=str(DATA_ROOT / "clsa" / "clsa.csv"),
        transform=None,
        image_size=224,
        progression_label_years: list = [0, 2, 5, 10, 13],
        split="train",
        possible_inputs: List[str] = [],
        possible_labels: List[str] = [],
        qc=True,
        enhanced=True,
        df_conditions=None,
        fine_tuning=False,
        fine_tuning_portion=0.1,
        processed_df=None,
        progression_label_ignorant_label_years: int = 3,
        mean_std_map={},
        normalise_fundus_image: bool = False,
        algo_qc: bool = False,
        algo_qc_path: str = str(DATA_ROOT / "clsa" / "qc" / "algorithmic_qc.csv"),
    ) -> None:
        super().__init__()
        self.split = split
        self.progression_label_years = progression_label_years
        self.possible_inputs = possible_inputs
        self.possible_labels = possible_labels
        self.progression_label_ignorant_label_years = (
            progression_label_ignorant_label_years
        )
        self.normalise_fundus_image = normalise_fundus_image
        self.transform = (
            get_default_aug(
                split=split,
                image_size=image_size,
                enhanced=enhanced,
            )
            if transform is None
            else transform
        )
        self.qc = qc
        self.algo_qc = algo_qc
        self.algo_qc_path = algo_qc_path
        self.mean_std_map = mean_std_map

        if processed_df is not None:
            self.df = processed_df
        else:
            self.df = pd.read_csv(baseline_followup_combined_path, low_memory=False)
            # self.df = self.df[:500]

            if split is not None and not fine_tuning:
                self.df = self.df[self.df["split"] == split]

            # initialise the finetuning dataset
            if fine_tuning:
                # then assert the split has to be train or test
                assert split in [
                    "train",
                    "test",
                ], "Fine-tuning can only be done on train or test split"
                # create a fine-tuning split column for self.df
                if "fine_tuning_split" not in self.df.columns:
                    # shuffle the dataframe
                    self.df = self.df.sample(frac=1, random_state=42).reset_index(
                        drop=True
                    )
                    # create a fine-tuning split column
                    self.df["fine_tuning_split"] = "train"
                    n_fine_tuning = int(len(self.df) * fine_tuning_portion)
                    self.df.loc[:n_fine_tuning, "fine_tuning_split"] = "test"
                    # obtain the desired split
                self.df = self.df[self.df["fine_tuning_split"] == split]

            if self.qc:
                # do the dl qc here
                if self.algo_qc:
                    # load the QC csv
                    qc_df = pd.read_csv(self.algo_qc_path)
                    self.df["algo_qc"] = ~qc_df["is_bad"]
                    self.df = self.df[self.df["algo_qc"]]
                    logging.info(
                        f"CLSA Dataset [{split}] | After algo QC filter: {len(self.df)}"
                    )

            logging.info(f"CLSA Dataset [{self.split}] | Total: {len(self.df)}")
            self.__init_gender()
            self.__init_age()
            self.__init_progression_label()

        if df_conditions is not None:
            for condition in df_conditions:
                original_len = len(self.df)
                self.df = self.df[condition(self.df)]
                removed_length = original_len - len(self.df)
                # print original and after length
                logging.info(
                    f"[UKB] | [{self.split}] | [{condition.__name__}] | Original instances: {original_len} | After instances: {len(self.df)} | Removed instances: {removed_length}"
                )

        # logging the mean_std_map
        logging.info(f"Mean/Std Map: {self.mean_std_map}")

    def __init_gender(
        self,
    ):
        self.df["patient_gender"] = self.df["SEX_ASK_COM"].apply(
            lambda x: 0 if x == "F" else 1 if x == "M" else None
        )

    def __init_age(self):
        self.df["instance_age_at_time"] = self.df["AGE_NMBR_COM"]

        assert (
            "instance_age_at_time" in self.mean_std_map
        ), "Mean/std for age not provided in mean_std_map"

        self.df["instance_age_at_time"] = self.df["instance_age_at_time"].apply(
            lambda x: (
                (x - self.mean_std_map["instance_age_at_time"]["mean"])
                / self.mean_std_map["instance_age_at_time"]["std"]
                if not pd.isna(x)
                else None
            )
        )

    def survival_analysis_has_glaucoma_in_n_years(
        self,
        row,
        n: int,
        *,
        positive_buffer_days: int = 90,  # small grace for positives
        ignorant_buffer_years: int = 3,  # uncertainty window
    ):
        """
        Tri-state glaucoma progression label using only:
        - instance_glaucoma_label
        - days_to_followup
        - has_glaucoma_in_followup

        Returns:
        True  -> glaucoma at/before n years
        False -> no glaucoma through >= n + ignorant years
        None  -> uncertain (censored, or event in uncertainty window)
        """

        baseline = row["instance_glaucoma_label"]
        days_to_followup = row["days_to_followup"]
        event_in_followup = row["has_glaucoma_in_followup"]

        # 1) Prevalent at baseline -> always True
        if baseline is True:
            return True

        # 2) No follow-up -> cannot determine
        if days_to_followup is None:
            return None

        # Convert year horizons to days
        n_cutoff_days = 365 * n
        n_buffer_cut_days = n_cutoff_days + positive_buffer_days
        ny_cutoff_days = 365 * (n + ignorant_buffer_years)

        # 3) If glaucoma is observed during follow-up
        if event_in_followup:
            # event_time <= days_to_followup
            if days_to_followup <= n_buffer_cut_days:
                # event is confidently within (0, n + buffer] → TRUE
                return True
            if days_to_followup <= ny_cutoff_days:
                # event might be in (n + buffer, n + y] → UNCERTAIN
                return None
            # days_to_followup > ny_cutoff_days:
            # event could be before or after n+y → cannot determine
            return None

        # 4) If NO glaucoma is observed during follow-up
        else:
            # If follow-up covers at least (n + y) years → confidently FALSE
            if days_to_followup >= ny_cutoff_days:
                return False
            # If follow-up reaches between n and n+y → UNCERTAIN
            if days_to_followup >= n_cutoff_days:
                return None
            # Follow-up does not even reach n → UNCERTAIN
            return None

    # def survival_analysis_has_glaucoma_in_n_years(self, row, n):

    #     # check if the patient has glaucoma now.
    #     current_glaucoma_label = row["instance_glaucoma_label"]

    #     # if the patient has the glaucoma now, then the patient has glaucoma in the n years
    #     if current_glaucoma_label:
    #         return True

    #     days_to_followup = row["days_to_followup"]
    #     has_glaucoma_in_followup = row["has_glaucoma_in_followup"]

    #     # check if "days_to_followup" is not None:
    #     if days_to_followup is None:
    #         # No followup data, we can't answer the question.
    #         return None

    #     # if we have the followup, then we have to check whether the patient has glaucoma in the followup
    #     if not has_glaucoma_in_followup:
    #         # if the patient doesn't have glaucoma in the followup, that means the patient is healthy in before the followup
    #         # And, we can only answer the range between the followup date and the current date
    #         if 365 * n <= days_to_followup:
    #             # if n*365 days is before the followup date, then the patient is healthy in the n years
    #             return False
    #         else:
    #             # Otherwise, we don't know whether the patient get the glaucoma in the future.
    #             return None
    #     else:
    #         # if the patient has glaucoma in the followup, then we can only answer the range after the followup date.
    #         if 365 * n >= days_to_followup:
    #             # if n*365 days is after the followup date, then the patient has glaucoma in the n years.
    #             return True
    #         else:
    #             # Otherwise, we don't know when's the patient get the glaucoma between current and the followup.
    #             return None

    def __init_progression_label(self):
        for year in self.progression_label_years:
            self.df[f"has_glaucoma_in_{year}_years"] = self.df.apply(
                lambda x: self.survival_analysis_has_glaucoma_in_n_years(
                    x,
                    year,
                    ignorant_buffer_years=self.progression_label_ignorant_label_years,
                ),
                axis=1,
            )
            logging.info(
                f"CLSA Dataset [{self.split}] | Glaucoma onset in {year} years"
            )
            logging.info(
                self.df[f"has_glaucoma_in_{year}_years"].value_counts(dropna=False)
            )

    def __len__(self) -> int:
        return len(self.df)

    def add_output_modality(self, output_dict, key, value):
        if not value is None:
            output_dict.update({key: value})

    def get_label_col(self, data, label_col):
        # check if the label_col is already in the data
        if label_col not in data:
            return None

        if data[label_col] is None or math.isnan(data[label_col]):
            return None
        return torch.tensor(data.loc[[label_col]]).float()

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
        img_path = data["image_path"]
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img)  # uint8 HxWx3
        H, W, _ = arr.shape

        cx = W // 2
        out = Image.fromarray(arr).crop((cx - H // 2, 0, cx + H // 2, H))
        return out

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        data = self.df.iloc[idx]
        modalities = {
            "fundus_image": self.transform(
                self.get_fundus_image_normalised(data)
                if self.normalise_fundus_image
                else self.get_fundus_image(data)
            ),
            "dataset": "clsa",
            "split": self.split,
            "idx": idx,
        }

        for label in self.possible_labels:
            self.add_output_modality(modalities, label, self.get_label_col(data, label))

        return modalities

    def get_sampling_weights(self):

        logging.info("Computing sample weights...")

        possible_labels = self.possible_labels
        # Step 1: Gather global label counts
        label_value_counts = {label: Counter() for label in possible_labels}

        df = self.df
        for label in possible_labels:
            if label in df.columns:
                label_value_counts[label].update(df[label].dropna().astype(int).values)

        # Step 2: Compute global weights per label
        label_class_weights = {}
        for label, counter in label_value_counts.items():
            total = sum(counter.values())
            if total == 0:
                continue
            label_class_weights[label] = {
                cls: total / count for cls, count in counter.items()
            }

        # Step 3: Assign weights to each sample
        all_sample_weights = []
        for idx, row in df.iterrows():
            weights = []
            for label in possible_labels:
                if label in row and pd.notna(row[label]):
                    value = int(row[label])
                    if (
                        label in label_class_weights
                        and value in label_class_weights[label]
                    ):
                        weights.append(label_class_weights[label][value])
            sample_weight = (
                sum(weights) / len(weights) if weights else 1.0
            )  # default weight
            all_sample_weights.append(sample_weight)

        return all_sample_weights


def build_clsa_datasets(args, **kwargs):
    train_transform = get_default_aug(image_size=args.image_size, split="train")
    test_transform = get_default_aug(image_size=args.image_size, split="test")

    print("Building CLSA datasets... | Image size:", args.image_size)

    train_dataset = CLSADataset(
        transform=train_transform,
        split="train",
        progression_label_years=kwargs.get("progression_label_years", None),
        possible_labels=kwargs.get("possible_labels", []),
    )

    val_dataset = CLSADataset(
        transform=test_transform,
        split="val",
        progression_label_years=kwargs.get("progression_label_years", None),
        possible_labels=kwargs.get("possible_labels", []),
    )

    test_dataset = CLSADataset(
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
