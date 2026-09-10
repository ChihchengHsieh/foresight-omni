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

def get_image_file_names(entity_id, side: str):
    return f"190225_QIMR_SMacGregor_{entity_id}_{side}.jpg"


from collections import defaultdict, Counter

class CLSAGenderDataset(Dataset):
    def __init__(
        self,
        baseline_followup_combined_path=str(DATA_ROOT / "clsa" / "clsa.csv"),
        transform=None,
        image_size=224,
        split="train",
        qc=True,
        enhanced=True,
        df_conditions=None,
        mean_std_map={},
        normalise_fundus_image: bool = False,
        algo_qc_path: str = str(DATA_ROOT / "clsa" / "qc" / "algorithmic_qc.csv"),
    ) -> None:
        super().__init__()
        self.split = split
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
        self.mean_std_map = mean_std_map

        self.df = pd.read_csv(baseline_followup_combined_path, low_memory=False)

        if split is not None:
            self.df = self.df[self.df["split"] == split]

        if self.qc:
            # load the QC csv
            qc_df = pd.read_csv(algo_qc_path)
            self.df["algo_qc"] = ~qc_df["is_bad"]
            self.df = self.df[self.df["algo_qc"]]
            logging.info(
                f"CLSA Dataset [{split}] | After algo QC filter: {len(self.df)}"
            )

        logging.info(f"CLSA Dataset [{self.split}] | Total: {len(self.df)}")
        self.__init_gender()
        # self.__init_age()

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

    def gender_logic(self, x):
        if x is None or pd.isna(x):
            return None

        if isinstance(x, str):
            x = x.strip().lower()
            return 0 if x == "f" else 1 if x == "m" else None

        return None

    def __init_gender(
        self,
    ):
        # first remove those cases who are not 'M' or 'F'
        self.df = self.df[self.df["SEX_ASK_COM"].isin(["M", "F"])]
        self.df["patient_gender"] = self.df["SEX_ASK_COM"].apply(
            lambda x: self.gender_logic(x)
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

    def __len__(self) -> int:
        return len(self.df)

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
            "patient_eid": data.get("entity_id"),
            "patient_gender": torch.tensor(data['patient_gender']),
        }

        return modalities
