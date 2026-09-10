import torch
from torch.utils.data import Dataset
from .paths import *
from .aug import get_default_aug
import os
from .clsa import CLSADataset
from .universal_image import ImageLevelUKBUniversalDataset

from torchvision import datasets as tv_datasets
import logging
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from types import MethodType


# Prepare weights for weighted sampling
def get_class_weights(dataset):
    # TODO: generate the class weight based on the dataframe instead of going through the whole dataset.

    # Count occurrences of each class
    class_counts = {}
    for sample in dataset:
        # how's it possible without this label from any instance?
        label = (
            int(
                sample["has_glaucoma_in_0_years"].item()
                if isinstance(sample["has_glaucoma_in_0_years"], torch.Tensor)
                else sample["has_glaucoma_in_0_years"]
            )
            if "has_glaucoma_in_0_years" in sample
            else 0
        )
        class_counts[label] = class_counts.get(label, 0) + 1
    # Compute weights inversely proportional to class frequency
    total_count = sum(class_counts.values())
    class_weights = {cls: total_count / count for cls, count in class_counts.items()}
    return class_weights


def get_sample_weights(dataset, class_weights):
    # Assign weights to each sample based on its label
    sample_weights = [
        class_weights[
            (
                int(
                    sample["has_glaucoma_in_0_years"].item()
                    if isinstance(sample["has_glaucoma_in_0_years"], torch.Tensor)
                    else sample["has_glaucoma_in_0_years"]
                )
                if "has_glaucoma_in_0_years" in sample
                else 0
            )
        ]
        for sample in dataset
    ]
    return sample_weights


# Wrap the original dataset to return a dictionary
class CombinedDatasetWrapper(Dataset):
    def __init__(self, dataset, false_class=None, name=None, split=None):
        self.dataset = dataset
        self.false_class = false_class
        self.name = name
        self.split = split

        # create a pseudo dataframe for the dataset
        self.df = []
        for i in range(len(self.dataset)):
            image, label = self.dataset[i]
            self.df.append(self._build_item(image, label, i))

        self.df = pd.DataFrame(self.df)

    def _is_positive(self, label):
        if self.false_class is None:
            return bool(label)
        return int(label) != int(self.dataset.class_to_idx[self.false_class])

    def _build_item(self, image, label, idx):
        """Build the only supported target for prevalence-only public data."""
        is_positive = self._is_positive(label)
        return {
            "fundus_image": image,
            "has_glaucoma_in_0_years": is_positive,
            "dataset": self.name,
            "dataset_idx": idx,
            "split": self.split,
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        item = self._build_item(image, label, idx)
        for key in tuple(item):
            if key.startswith("has_glaucoma_in_"):
                item[key] = torch.tensor([float(item[key])], dtype=torch.float32)
        return item


def compute_multilabel_sample_weights(self):

    logging.info("Computing sample weights...")

    datasets = self.datasets
    possible_labels = self.possible_labels
    # Step 1: Gather global label counts
    label_value_counts = {label: Counter() for label in possible_labels}
    for dataset in datasets:
        df = dataset.df
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
    for dataset in datasets:
        df = dataset.df
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


def get_combined_multi_class_labels_for_balance(self):
    """Return one aligned multi-label balance row per ConcatDataset sample.

    Public glaucoma datasets only provide the prevalent-glaucoma target. Missing
    disease/horizon columns are therefore zero-filled, while UKB contributes its
    complete balance-label matrix. This is the interface required by the hybrid
    coverage/balance sampler.
    """
    balance_label_cols = list(
        getattr(self.datasets[0], "balance_label_cols", [])
    )
    if not balance_label_cols:
        raise ValueError("Combined dataset has no balance_label_cols")

    label_frames = []
    for dataset in self.datasets:
        values = dataset.df.reindex(columns=balance_label_cols).apply(
            pd.to_numeric, errors="coerce"
        )
        label_frames.append(values.fillna(0.0))

    combined_values = pd.concat(label_frames, ignore_index=True).to_numpy(
        dtype="float32"
    )
    if len(combined_values) != len(self):
        raise ValueError(
            "Combined balance-label rows do not match dataset length: "
            f"{len(combined_values)} != {len(self)}"
        )
    return torch.from_numpy(combined_values >= 0.5).long()


def get_combined_patient_ids(datasets):
    """Build lightweight, collision-safe IDs without materialising a joined df."""
    patient_ids = []
    for dataset_index, dataset in enumerate(datasets):
        if "patient_eid" in dataset.df.columns:
            component_ids = dataset.df["patient_eid"].astype(str).tolist()
        else:
            component_ids = [str(row_index) for row_index in range(len(dataset))]
        patient_ids.extend(
            f"{dataset_index}:{patient_id}" for patient_id in component_ids
        )
    return patient_ids


def load_combined_dataset(
    args,
    split: str,  # [train, val, test]
    include_public: bool = True,
    **kwargs,
):
    requested_processed_df_path = getattr(args, "ukb_processed_df_path", None)
    if not requested_processed_df_path:
        raise ValueError(
            "Combined UKB/public training requires an explicit "
            "--ukb_processed_df_path"
        )
    kwargs.setdefault("processed_df_path", requested_processed_df_path)
    kwargs.setdefault(
        "allow_legacy_iop_manifest",
        getattr(args, "allow_legacy_iop_manifest", False),
    )
    kwargs.setdefault(
        "smoke_test_max_rows_per_split",
        getattr(args, "smoke_test_max_rows_per_split", None),
    )
    kwargs.setdefault("smoke_test_seed", getattr(args, "seed", 42))
    kwargs.setdefault(
        "algo_qc_path", getattr(args, "algo_qc_path", "data/ukb/algorithmic_qc.csv")
    )
    kwargs.setdefault("omics_qc", getattr(args, "omics_qc", True))
    kwargs.setdefault("omics_missing_rate_threshold", getattr(args, "omics_missing_rate_threshold", 0.40))
    kwargs.setdefault("omics_min_std", getattr(args, "omics_min_std", 1e-6))
    kwargs.setdefault("omics_min_unique_values", getattr(args, "omics_min_unique_values", 10))
    kwargs.setdefault("omics_winsor_lower_quantile", getattr(args, "omics_winsor_lower_quantile", 0.005))
    kwargs.setdefault("omics_winsor_upper_quantile", getattr(args, "omics_winsor_upper_quantile", 0.995))
    kwargs.setdefault("clinical_history_min_count", getattr(args, "clinical_history_min_count", 10))
    kwargs.setdefault("clinical_history_max_len", getattr(args, "clinical_history_max_len", 128))
    kwargs.setdefault("clinical_history_mask_targets", getattr(args, "clinical_history_mask_targets", True))
    kwargs.setdefault("clinical_history_mask_map_path", getattr(args, "clinical_history_mask_map_path", None))
    kwargs.setdefault("clinical_history_target_diseases", getattr(args, "diseases", None))
    kwargs.setdefault("questionnaire_max_choices", getattr(args, "questionnaire_max_choices", 16))
    # Keep combined UKB/public runs aligned with the OCT preprocessing used by
    # the checkpoint. Without this override, ImageLevelUKBUniversalDataset
    # falls back to 128 slices even when the experiment is configured for 32.
    kwargs.setdefault("oct_num_slices", getattr(args, "oct_num_slices", 128))
    kwargs.setdefault("oct_image_size", getattr(args, "oct_image_size", None))
    kwargs.setdefault(
        "oct_aug_profile",
        getattr(args, "oct_aug_profile", "oct_clinical_v1"),
    )
    logging.info(
        "Combined UKB processed dataframe: %s",
        kwargs["processed_df_path"],
    )

    ukb_dataset = ImageLevelUKBUniversalDataset(
        image_size=args.image_size,
        split=split,
        # down_sampling=args.ukb_downsampling,
        **kwargs,
    )

    if not include_public:
        logging.info("Using UKB-only %s split for primary validation/test.", split)
        return ukb_dataset

    transform = get_default_aug(image_size=args.image_size, split=split)
    public_data_root = Path(args.public_data_root).expanduser()
    glaucoma_fundus_dataset = CombinedDatasetWrapper(
        tv_datasets.ImageFolder(
            public_data_root / "Glaucoma_fundus" / split,
            transform=transform,
        ),
        false_class="anormal_control",  # ['anormal_control', 'bearly_glaucoma', 'cadvanced_glaucoma']
        name="glaucoma_fundus",
        split=split,
    )
    logging.info("Glaucoma fundus dataset loaded.")

    papila_dataset = CombinedDatasetWrapper(
        tv_datasets.ImageFolder(
            public_data_root / "PAPILA" / split,
            transform=transform,
        ),
        false_class="anormal",  # ['anormal', 'bsuspectglaucoma', 'cglaucoma']
        name="papila",
        split=split,
    )
    logging.info("Papila dataset loaded.")

    # clsa = CLSADataset(
    #     transform=transform,
    #     split=split,
    #     progression_label_years=kwargs.get("progression_label_years", None),
    #     possible_labels=kwargs.get("possible_labels", []),
    # )

    logging.info("CLSA dataset loaded.")
    combined_dataset = torch.utils.data.ConcatDataset(
        [
            ukb_dataset,
            glaucoma_fundus_dataset,
            papila_dataset,
            # clsa,
        ]
    )
    combined_dataset.possible_inputs = ukb_dataset.possible_inputs
    combined_dataset.possible_labels = ukb_dataset.possible_labels
    combined_dataset.mean_std_map = ukb_dataset.mean_std_map
    combined_dataset.omics_qc_report = getattr(ukb_dataset, "omics_qc_report", {})
    combined_dataset.icd10_code_to_id = getattr(ukb_dataset, "icd10_code_to_id", {})
    combined_dataset.clinical_history_vocab_report = getattr(
        ukb_dataset, "clinical_history_vocab_report", {}
    )
    combined_dataset.questionnaire_num_fields = getattr(
        ukb_dataset, "questionnaire_num_fields", 0
    )
    combined_dataset.questionnaire_category_vocab_size = getattr(
        ukb_dataset, "questionnaire_category_vocab_size", 0
    )
    combined_dataset.patient_ids = get_combined_patient_ids(
        combined_dataset.datasets
    )

    logging.info("Datasets combined.")

    # Getting possible labels for balancing.
    possible_labels = kwargs.get("possible_labels", [])
    if len(possible_labels) == 0:
        raise ValueError(
            "No possible labels provided. Please provide a list of possible labels."
        )

    # def get_sampling_weights(self):
    #     sample_weights = []
    #     for dataset in self.datasets:
    #         class_weights = get_class_weights(dataset)
    #         sample_weights.extend(get_sample_weights(dataset, class_weights))
    #     return sample_weights

    # class_weights = get_class_weights(combined_dataset)
    # sample_weights = get_sample_weights(combined_dataset, class_weights)
    torch.utils.data.ConcatDataset.get_sampling_weights = (
        compute_multilabel_sample_weights
    )
    combined_dataset.get_multi_class_labels_for_balance = MethodType(
        get_combined_multi_class_labels_for_balance, combined_dataset
    )
    # combined_dataset.get_sampling_weights = lambda: sample_weights
    # sampler = torch.utils.data.WeightedRandomSampler(
    #     sample_weights, len(sample_weights)
    # )

    # combined_dataset.get_sampling_weights = lambda: compute_multilabel_sample_weights(
    #     [ukb_dataset, glaucoma_fundus_dataset, papila_dataset, clsa], possible_labels
    # )

    return combined_dataset

def load_public_only_dataset(
    args,
    split: str,  # [train, val, test]
    **kwargs,
):
    kwargs.setdefault("omics_qc", getattr(args, "omics_qc", True))
    kwargs.setdefault("clinical_history_min_count", getattr(args, "clinical_history_min_count", 10))
    kwargs.setdefault("clinical_history_max_len", getattr(args, "clinical_history_max_len", 128))
    kwargs.setdefault("clinical_history_mask_targets", getattr(args, "clinical_history_mask_targets", True))
    kwargs.setdefault("clinical_history_mask_map_path", getattr(args, "clinical_history_mask_map_path", None))
    kwargs.setdefault("clinical_history_target_diseases", getattr(args, "diseases", None))
    kwargs.setdefault("questionnaire_max_choices", getattr(args, "questionnaire_max_choices", 16))
    kwargs.setdefault("oct_num_slices", getattr(args, "oct_num_slices", 128))
    kwargs.setdefault("oct_image_size", getattr(args, "oct_image_size", None))
    kwargs.setdefault(
        "oct_aug_profile",
        getattr(args, "oct_aug_profile", "oct_clinical_v1"),
    )

    ukb_dataset = ImageLevelUKBUniversalDataset(
        image_size=args.image_size,
        split=split,
        # down_sampling=args.ukb_downsampling,
        **kwargs,
    )

    transform = get_default_aug(image_size=args.image_size, split=split)
    public_data_root = Path(args.public_data_root).expanduser()
    glaucoma_fundus_dataset = CombinedDatasetWrapper(
        tv_datasets.ImageFolder(
            public_data_root / "Glaucoma_fundus" / split,
            transform=transform,
        ),
        false_class="anormal_control",  # ['anormal_control', 'bearly_glaucoma', 'cadvanced_glaucoma']
        name="glaucoma_fundus",
        split=split,
    )
    logging.info("Glaucoma fundus dataset loaded.")

    papila_dataset = CombinedDatasetWrapper(
        tv_datasets.ImageFolder(
            public_data_root / "PAPILA" / split,
            transform=transform,
        ),
        false_class="anormal",  # ['anormal', 'bsuspectglaucoma', 'cglaucoma']
        name="papila",
        split=split,
    )
    logging.info("Papila dataset loaded.")

    # clsa = CLSADataset(
    #     transform=transform,
    #     split=split,
    #     progression_label_years=kwargs.get("progression_label_years", None),
    #     possible_labels=kwargs.get("possible_labels", []),
    # )

    logging.info("CLSA dataset loaded.")
    public_only_datasets = torch.utils.data.ConcatDataset(
        [
            glaucoma_fundus_dataset,
            papila_dataset,
            # clsa,
        ]
    )
    public_only_datasets.possible_inputs = ukb_dataset.possible_inputs
    public_only_datasets.possible_labels = ukb_dataset.possible_labels
    public_only_datasets.mean_std_map = ukb_dataset.mean_std_map
    public_only_datasets.omics_qc_report = getattr(ukb_dataset, "omics_qc_report", {})
    public_only_datasets.icd10_code_to_id = getattr(ukb_dataset, "icd10_code_to_id", {})
    public_only_datasets.clinical_history_vocab_report = getattr(
        ukb_dataset, "clinical_history_vocab_report", {}
    )
    public_only_datasets.questionnaire_num_fields = getattr(
        ukb_dataset, "questionnaire_num_fields", 0
    )
    public_only_datasets.questionnaire_category_vocab_size = getattr(
        ukb_dataset, "questionnaire_category_vocab_size", 0
    )

    logging.info("Datasets combined.")

    # Getting possible labels for balancing.
    possible_labels = kwargs.get("possible_labels", [])
    if len(possible_labels) == 0:
        raise ValueError(
            "No possible labels provided. Please provide a list of possible labels."
        )

    # def get_sampling_weights(self):
    #     sample_weights = []
    #     for dataset in self.datasets:
    #         class_weights = get_class_weights(dataset)
    #         sample_weights.extend(get_sample_weights(dataset, class_weights))
    #     return sample_weights

    # class_weights = get_class_weights(combined_dataset)
    # sample_weights = get_sample_weights(combined_dataset, class_weights)
    torch.utils.data.ConcatDataset.get_sampling_weights = (
        compute_multilabel_sample_weights
    )
    # combined_dataset.get_sampling_weights = lambda: sample_weights
    # sampler = torch.utils.data.WeightedRandomSampler(
    #     sample_weights, len(sample_weights)
    # )

    # combined_dataset.get_sampling_weights = lambda: compute_multilabel_sample_weights(
    #     [ukb_dataset, glaucoma_fundus_dataset, papila_dataset, clsa], possible_labels
    # )

    return public_only_datasets

def build_combined_datasets(args, **kwargs):
    logging.info(
        "Creating combined training, validation, and test datasets..."
    )
    train_dataset = load_combined_dataset(
        args,
        split="train",
        include_public=True,
        **kwargs,
    )
    val_dataset = load_combined_dataset(
        args,
        split="val",
        include_public=True,
        **kwargs,
    )
    test_dataset = load_combined_dataset(
        args,
        split="test",
        include_public=True,
        **kwargs,
    )
    logging.info(
        "Combined training, validation, and test datasets created."
    )
    return train_dataset, val_dataset, test_dataset


def build_public_only_datasets(args, **kwargs):
    logging.info("Creating combined datasets...")
    train_dataset = load_public_only_dataset(
        args,
        split="train",
        **kwargs,
    )
    val_dataset = load_public_only_dataset(
        args,
        split="val",
        **kwargs,
    )
    test_dataset = load_public_only_dataset(
        args,
        split="test",
        **kwargs,
    )
    logging.info("Combined datasets created.")
    return train_dataset, val_dataset, test_dataset
