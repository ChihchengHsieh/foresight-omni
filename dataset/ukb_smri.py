"""Scan-level UKB T1 structural-MRI dataset."""

from __future__ import annotations

import gzip
import math
import struct
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


NIFTI_DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


def smri_cache_path(cache_root: str | Path, eid: int, instance: int) -> Path:
    root = Path(cache_root)
    return root / f"{int(eid) % 1000:03d}" / f"{int(eid)}_{int(instance)}.npy"


def _load_nifti1_gz_from_zip(zip_path: str, member: str) -> np.ndarray:
    with zipfile.ZipFile(zip_path) as archive:
        if member not in archive.namelist():
            candidates = [
                name
                for name in archive.namelist()
                if name.endswith(".nii.gz") and "T1" in name
            ]
            raise FileNotFoundError(
                f"{member!r} is absent from {zip_path}; T1 candidates={candidates}"
            )
        with archive.open(member) as compressed_member:
            with gzip.GzipFile(fileobj=compressed_member) as nifti_stream:
                raw = nifti_stream.read()

    if len(raw) < 352:
        raise ValueError(f"Truncated NIfTI payload in {zip_path}:{member}")
    sizeof_hdr = struct.unpack("<i", raw[:4])[0]
    endian = "<"
    if sizeof_hdr != 348:
        sizeof_hdr = struct.unpack(">i", raw[:4])[0]
        endian = ">"
    if sizeof_hdr != 348:
        raise ValueError(f"Invalid NIfTI-1 header in {zip_path}:{member}")

    dim = struct.unpack(endian + "8h", raw[40:56])
    ndim = int(dim[0])
    shape = tuple(int(value) for value in dim[1 : 1 + ndim])
    if ndim != 3 or any(value <= 0 for value in shape):
        raise ValueError(f"Expected a 3D T1 NIfTI, got shape={shape}")
    datatype = struct.unpack(endian + "h", raw[70:72])[0]
    vox_offset = int(struct.unpack(endian + "f", raw[108:112])[0])
    slope = float(struct.unpack(endian + "f", raw[112:116])[0])
    intercept = float(struct.unpack(endian + "f", raw[116:120])[0])
    dtype = NIFTI_DTYPES.get(datatype)
    if dtype is None:
        raise ValueError(f"Unsupported NIfTI datatype={datatype}")
    dtype = np.dtype(dtype).newbyteorder(endian)
    count = int(np.prod(shape))
    required_bytes = vox_offset + count * dtype.itemsize
    if required_bytes > len(raw):
        raise ValueError(
            f"Truncated NIfTI data in {zip_path}:{member}; "
            f"need={required_bytes}, have={len(raw)}"
        )
    volume = np.frombuffer(
        raw, dtype=dtype, count=count, offset=vox_offset
    ).reshape(shape, order="F").astype(np.float32)
    if math.isfinite(slope) and slope not in (0.0, 1.0):
        volume *= slope
    if math.isfinite(intercept) and intercept != 0.0:
        volume += intercept
    return volume


def _crop_and_normalise(volume: np.ndarray, output_size: int) -> torch.Tensor:
    finite = np.isfinite(volume)
    foreground = finite & (np.abs(volume) > 1e-6)
    if not foreground.any():
        raise ValueError("T1 volume has no finite nonzero foreground")

    coordinates = np.where(foreground)
    starts = [max(int(axis.min()) - 2, 0) for axis in coordinates]
    stops = [
        min(int(axis.max()) + 3, volume.shape[index])
        for index, axis in enumerate(coordinates)
    ]
    volume = volume[
        starts[0] : stops[0], starts[1] : stops[1], starts[2] : stops[2]
    ]
    finite = np.isfinite(volume)
    foreground = finite & (np.abs(volume) > 1e-6)
    values = volume[foreground]
    lower, upper = np.percentile(values, [0.5, 99.5])
    clipped = np.clip(volume, lower, upper)
    mean = float(clipped[foreground].mean())
    std = float(clipped[foreground].std())
    if not math.isfinite(std) or std < 1e-6:
        raise ValueError(f"T1 foreground standard deviation is invalid: {std}")

    normalised = np.zeros_like(clipped, dtype=np.float32)
    normalised[foreground] = (clipped[foreground] - mean) / std
    tensor = torch.from_numpy(normalised.copy()).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(
        tensor,
        size=(output_size, output_size, output_size),
        mode="trilinear",
        align_corners=False,
    )
    return tensor.squeeze(0)


class UKBSmriDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        split: str,
        possible_inputs: Iterable[str],
        possible_labels: Iterable[str],
        balance_label_cols: Iterable[str],
        volume_size: int = 96,
        cache_root: str | None = None,
        require_cache: bool = False,
        augment: bool = False,
        smoke_test_max_rows_per_split: int | None = None,
        seed: int = 42,
        profile_timing: bool = False,
    ) -> None:
        super().__init__()
        self.manifest_path = str(manifest_path)
        self.split = split
        self.possible_inputs = list(possible_inputs)
        self.possible_labels = list(possible_labels)
        self.balance_label_cols = list(balance_label_cols)
        self.volume_size = int(volume_size)
        self.cache_root = str(cache_root) if cache_root else None
        self.require_cache = bool(require_cache)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.profile_timing = bool(profile_timing)
        self.mean_std_map = {}

        if self.possible_inputs != ["smri_image"]:
            raise ValueError(
                "UKBSmriDataset currently supports sMRI-only pretraining; "
                f"received inputs={self.possible_inputs}"
            )
        if self.volume_size < 32:
            raise ValueError(f"smri_volume_size must be >= 32, got {self.volume_size}")

        columns = [
            "patient_eid",
            "smri_instance",
            "smri_zip_path",
            "smri_member",
            "index_date",
            "split",
            *self.possible_labels,
        ]
        self.df = pd.read_parquet(self.manifest_path, columns=columns)
        missing = [column for column in columns if column not in self.df.columns]
        if missing:
            raise KeyError(f"sMRI manifest is missing columns: {missing}")
        self.df = self.df[self.df["split"] == split].copy()
        if self.possible_labels:
            self.df = self.df[
                self.df[self.possible_labels].notna().any(axis=1)
            ].copy()
        if smoke_test_max_rows_per_split is not None:
            cap = int(smoke_test_max_rows_per_split)
            if cap <= 0:
                raise ValueError("smoke_test_max_rows_per_split must be positive")
            if len(self.df) > cap:
                split_offset = {"train": 0, "val": 1, "test": 2}[split]
                self.df = self.df.sample(
                    n=cap, random_state=self.seed + split_offset
                )
        self.df = self.df.sort_values(
            ["patient_eid", "smri_instance"]
        ).reset_index(drop=True)
        self.patient_ids = self.df["patient_eid"].astype(str).tolist()
        if self.df.empty:
            raise ValueError(f"No usable sMRI rows remain for split={split}")

    def __len__(self) -> int:
        return len(self.df)

    def get_multi_class_labels_for_balance(self) -> torch.Tensor:
        values = self.df[self.balance_label_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        return torch.from_numpy(
            (values.fillna(0.0).to_numpy(dtype=np.float32) >= 0.5).astype(np.int64)
        )

    def get_sampling_weights(self) -> torch.Tensor:
        labels = self.get_multi_class_labels_for_balance().numpy().astype(np.float64)
        positive_counts = labels.sum(axis=0)
        valid = positive_counts > 0
        weights = np.ones(len(self.df), dtype=np.float64)
        if valid.any():
            weights += (labels[:, valid] / np.sqrt(positive_counts[valid])).sum(axis=1)
        weights /= weights.mean()
        return torch.from_numpy(weights)

    def __getitem__(self, index: int) -> dict:
        started = time.perf_counter()
        row = self.df.iloc[index]
        cache_path = None
        if self.cache_root:
            cache_path = smri_cache_path(
                self.cache_root, int(row["patient_eid"]), int(row["smri_instance"])
            )
        if cache_path is not None and cache_path.exists():
            cached = np.load(cache_path, allow_pickle=False)
            if cached.shape != (
                1,
                self.volume_size,
                self.volume_size,
                self.volume_size,
            ):
                raise ValueError(
                    f"Unexpected cached sMRI shape at {cache_path}: {cached.shape}"
                )
            volume = torch.from_numpy(cached.astype(np.float32, copy=False))
        else:
            if self.require_cache:
                raise FileNotFoundError(
                    f"Required cached sMRI volume is missing: {cache_path}"
                )
            volume = _crop_and_normalise(
                _load_nifti1_gz_from_zip(
                    str(row["smri_zip_path"]), str(row["smri_member"])
                ),
                self.volume_size,
            )
        if self.augment:
            # Non-lateral disease targets permit left-right reflection. Small
            # independent axis flips provide a conservative first augmentation.
            for axis in (1, 2, 3):
                if bool(torch.rand(()) < 0.5):
                    volume = torch.flip(volume, dims=(axis,))

        output = {
            "smri_image": volume,
            "idx": int(index),
            "patient_eid": int(row["patient_eid"]),
            "dataset": "ukb_smri",
        }
        for label in self.possible_labels:
            value = pd.to_numeric(pd.Series([row[label]]), errors="coerce").iloc[0]
            if pd.notna(value):
                output[label] = torch.tensor([float(value)], dtype=torch.float32)
        if self.profile_timing:
            elapsed = time.perf_counter() - started
            output["__profile__"] = {
                "getitem_total_s": elapsed,
                "smri_load_preprocess_s": elapsed,
            }
        return output


def build_smri_datasets(args, **kwargs):
    possible_inputs = list(kwargs["possible_inputs"])
    possible_labels = list(kwargs["possible_labels"])
    balance_label_cols = list(kwargs["balance_label_cols"])
    common = {
        "manifest_path": args.smri_manifest_path,
        "possible_inputs": possible_inputs,
        "possible_labels": possible_labels,
        "balance_label_cols": balance_label_cols,
        "volume_size": args.smri_volume_size,
        "cache_root": args.smri_cache_root,
        "require_cache": args.smri_require_cache,
        "smoke_test_max_rows_per_split": args.smoke_test_max_rows_per_split,
        "seed": args.seed,
        "profile_timing": args.profile_timing,
    }
    return (
        UKBSmriDataset(split="train", augment=not args.smri_no_aug, **common),
        UKBSmriDataset(split="val", augment=False, **common),
        UKBSmriDataset(split="test", augment=False, **common),
    )
