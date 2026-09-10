import json
from pathlib import Path

import numpy as np
from dataset.glaucoma import build_glaucoma_dataset
from torchvision.datasets import CIFAR10
from .paths import *
from .aug import get_default_aug
import logging, torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset
from utils.sampler import (
    HybridCoverageBalancedSampler,
    ImbalancedDatasetSampler,
    MultiLabelImbalancedDatasetSampler,
    ParticipantEpochSampler,
)
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import WeightedRandomSampler


class TaskLabelView(Dataset):
    """Index view that exposes only one primary task plus shared auxiliaries."""

    def __init__(self, dataset, indices, *, active_primary_labels, all_primary_labels):
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        self.active_primary_labels = set(active_primary_labels)
        self.masked_primary_labels = set(all_primary_labels) - self.active_primary_labels
        self.df = dataset.df.iloc[self.indices].reset_index(drop=True)
        self.possible_inputs = dataset.possible_inputs
        self.possible_labels = [
            label
            for label in dataset.possible_labels
            if label not in self.masked_primary_labels
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        item = dict(self.dataset[self.indices[position]])
        for label in self.masked_primary_labels:
            item.pop(label, None)
        return item

    def __getattr__(self, name):
        return getattr(self.dataset, name)


class _TaskLoaderSamplerFacade:
    """Expose epoch/state hooks expected by the canonical training loop."""

    def __init__(self, task_loaders, active_batches_per_task=None):
        self.task_loaders = task_loaders
        self.active_batches_per_task = active_batches_per_task

    def __len__(self):
        if self.active_batches_per_task is not None:
            return sum(
                min(len(loader.sampler), self.active_batches_per_task * loader.batch_size)
                for loader in self.task_loaders.values()
            )
        return sum(len(loader.sampler) for loader in self.task_loaders.values())

    def set_epoch(self, epoch):
        for loader in self.task_loaders.values():
            if hasattr(loader.sampler, "set_epoch"):
                loader.sampler.set_epoch(epoch)

    def state_dict(self):
        return {
            name: loader.sampler.state_dict()
            for name, loader in self.task_loaders.items()
            if hasattr(loader.sampler, "state_dict")
        }

    def load_state_dict(self, state):
        for name, loader in self.task_loaders.items():
            if name in state and hasattr(loader.sampler, "load_state_dict"):
                loader.sampler.load_state_dict(state[name])

    def summary(self):
        summary = {"sampler": "AlternatingTaskLoader"}
        for name, loader in self.task_loaders.items():
            active_batches = (
                min(len(loader), self.active_batches_per_task)
                if self.active_batches_per_task is not None
                else len(loader)
            )
            summary[f"{name}_batches"] = active_batches
            summary[f"{name}_samples"] = min(
                len(loader.sampler), active_batches * loader.batch_size
            )
        return summary


class AlternatingTaskLoader:
    """Alternate equal numbers of masked task batches without replacement cycles."""

    def __init__(self, task_loaders):
        if not task_loaders:
            raise ValueError("AlternatingTaskLoader needs at least one task loader")
        self.task_loaders = dict(task_loaders)
        self.rounds = min(len(loader) for loader in self.task_loaders.values())
        self.sampler = _TaskLoaderSamplerFacade(
            self.task_loaders, active_batches_per_task=self.rounds
        )

    def __len__(self):
        return self.rounds * len(self.task_loaders)

    def __iter__(self):
        iterators = {name: iter(loader) for name, loader in self.task_loaders.items()}
        for _ in range(self.rounds):
            for name in self.task_loaders:
                yield next(iterators[name])


class SequentialTaskLoader:
    """Evaluate each masked task view once without cycling or shuffling."""

    def __init__(self, task_loaders):
        self.task_loaders = dict(task_loaders)
        self.sampler = _TaskLoaderSamplerFacade(self.task_loaders)

    def __len__(self):
        return sum(len(loader) for loader in self.task_loaders.values())

    def __iter__(self):
        for loader in self.task_loaders.values():
            yield from loader


def build_joint_task_loaders(
    args,
    train_dataset,
    val_dataset,
    test_dataset,
    *,
    task_sampling,
    primary_labels,
    **loader_kwargs,
):
    """Build endpoint-correct alternating train and sequential eval loaders."""
    if getattr(args, "distributed", False):
        raise NotImplementedError("Joint task loaders currently support one GPU process")

    batch_sizes = {
        "train": int(getattr(args, "train_batch_size", None) or args.batch_size),
        "val": int(getattr(args, "val_batch_size", None) or args.batch_size),
        "test": int(getattr(args, "test_batch_size", None) or args.batch_size),
    }
    split_datasets = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }
    split_task_loaders = {split: {} for split in split_datasets}

    for split, dataset in split_datasets.items():
        for task, sampling_mode in task_sampling.items():
            label = f"has_{task}_in_0_years"
            if label not in dataset.df.columns:
                raise KeyError(f"Joint task {task!r} is missing dataframe label {label!r}")
            indices = np.flatnonzero(dataset.df[label].notna().to_numpy()).tolist()
            if not indices:
                raise ValueError(f"Joint task {task!r} has no eligible rows in split={split}")
            view = TaskLabelView(
                dataset,
                indices,
                active_primary_labels=[label],
                all_primary_labels=primary_labels,
            )
            common = dict(
                batch_size=batch_sizes[split],
                collate_fn=lambda values: list(values),
                **loader_kwargs,
            )
            if split == "train" and sampling_mode == "participant":
                sampler = ParticipantEpochSampler(view, seed=int(args.seed))
                loader = DataLoader(view, sampler=sampler, **common)
            elif split == "train" and sampling_mode == "row":
                generator = torch.Generator()
                generator.manual_seed(int(args.seed))
                loader = DataLoader(view, shuffle=True, generator=generator, **common)
            else:
                loader = DataLoader(view, shuffle=False, **common)
            split_task_loaders[split][task] = loader
            logging.info(
                "Joint loader task=%s split=%s mode=%s rows=%d batches=%d",
                task,
                split,
                sampling_mode if split == "train" else "sequential",
                len(view),
                len(loader),
            )

    return (
        AlternatingTaskLoader(split_task_loaders["train"]),
        SequentialTaskLoader(split_task_loaders["val"]),
        SequentialTaskLoader(split_task_loaders["test"]),
    )


def _validate_split_lengths(train_dataset, val_dataset, test_dataset):
    train_len = len(train_dataset)
    val_len = len(val_dataset)
    test_len = len(test_dataset)

    if train_len <= 0 or val_len <= 0 or test_len <= 0:
        raise ValueError(
            "Empty dataset split detected before DataLoader creation. "
            f"train={train_len}, val={val_len}, test={test_len}. "
            "This usually means filtering removed all rows (for example, QC, modality availability, "
            "or label/split conditions)."
        )


def _build_validated_weighted_sampler(dataset, num_samples, split_name):
    weights = dataset.get_sampling_weights()
    weights = torch.as_tensor(weights, dtype=torch.double)

    if weights.numel() == 0:
        raise ValueError(
            f"Cannot build WeightedRandomSampler for '{split_name}': empty weights tensor."
        )

    if not torch.isfinite(weights).all():
        invalid_count = int((~torch.isfinite(weights)).sum().item())
        raise ValueError(
            f"Cannot build WeightedRandomSampler for '{split_name}': "
            f"weights contain {invalid_count} non-finite values (NaN/Inf)."
        )

    if float(weights.sum().item()) <= 0.0:
        raise ValueError(
            f"Cannot build WeightedRandomSampler for '{split_name}': "
            "sum(weights) <= 0. Check balancing labels, filtering, and default sampling weight settings."
        )

    return WeightedRandomSampler(num_samples=num_samples, weights=weights)


def build_validation_loaders(args, val_dataset, **kwargs):
    """Build a fixed fast panel and the complete natural validation loader."""
    val_batch_size = int(getattr(args, "val_batch_size", None) or args.batch_size)
    panel_size = min(int(getattr(args, "fast_val_size", 8000)), len(val_dataset))
    manifest_path = Path(args.output_dir) / "fast_validation_panel.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        positions = [int(i) for i in manifest["dataset_positions"]]
        if len(positions) != panel_size or any(i < 0 or i >= len(val_dataset) for i in positions):
            raise ValueError(
                f"Fast-validation manifest {manifest_path} does not match the current dataset."
            )
    else:
        labels = np.asarray(val_dataset.get_multi_class_labels_for_balance(), dtype=bool)
        patient_ids = getattr(val_dataset, "patient_ids", None)
        if patient_ids is None and hasattr(val_dataset, "df"):
            patient_ids = (
                val_dataset.df["patient_eid"].astype(str).tolist()
                if "patient_eid" in val_dataset.df.columns
                else None
            )
        if patient_ids is not None and len(patient_ids) != len(val_dataset):
            raise ValueError("Validation patient IDs do not match dataset length")
        counts = labels.sum(axis=0)
        rarity = np.zeros(len(val_dataset), dtype=np.float64)
        valid_cols = counts > 0
        if valid_cols.any():
            rarity = (labels[:, valid_cols] / np.sqrt(counts[valid_cols])).sum(axis=1)
        rng = np.random.default_rng(int(args.seed))
        tie_break = rng.random(len(val_dataset))
        ordered = np.lexsort((tie_break, -rarity))
        positions = []
        seen_patients = set()
        for position in ordered:
            patient = (
                str(patient_ids[position])
                if patient_ids is not None
                else str(position)
            )
            if patient in seen_patients:
                continue
            positions.append(int(position))
            seen_patients.add(patient)
            if len(positions) == panel_size:
                break
        if len(positions) < panel_size:
            for position in rng.permutation(len(val_dataset)):
                if int(position) not in positions:
                    positions.append(int(position))
                    if len(positions) == panel_size:
                        break
        manifest = {
            "seed": int(args.seed),
            "panel_size": panel_size,
            "dataset_size": len(val_dataset),
            "dataset_positions": positions,
            "patient_ids": (
                [str(patient_ids[i]) for i in positions]
                if patient_ids is not None
                else []
            ),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    fast_loader = DataLoader(
        Subset(val_dataset, positions),
        batch_size=val_batch_size,
        shuffle=False,
        **kwargs,
    )
    full_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        **kwargs,
    )
    return fast_loader, full_loader, manifest_path

def build_cifar10_dataset(args, **kwargs):
    train_dataset = CIFAR10(
        CIFAR10_PATH,
        train=True,
        transform=get_default_aug(args.image_size, "train"),
        download=True,
        **kwargs,
    )
    val_dataset = CIFAR10(
        CIFAR10_PATH,
        train=False,
        transform=get_default_aug(args.image_size, "val"),
        download=True,
        **kwargs,
    )
    test_dataset = CIFAR10(
        CIFAR10_PATH,
        train=False,
        transform=get_default_aug(args.image_size, "test"),
        download=True,
        **kwargs,
    )
    train_dataset.num_classes = 10
    val_dataset.num_classes = 10
    test_dataset.num_classes = 10
    return train_dataset, val_dataset, test_dataset


def build_loader(
    args,
    train_dataset,
    val_dataset,
    test_dataset,
    weight_sample_validation=False,
    eval_loader_kwargs=None,
    **kwargs
):
    _validate_split_lengths(train_dataset, val_dataset, test_dataset)

    participant_level_sampling = bool(
        getattr(args, "participant_level_sampling", False)
    )
    if participant_level_sampling and (
        getattr(args, "hybrid_sampler", False)
        or getattr(args, "imbalanced", False)
        or getattr(args, "num_samples", None) is not None
    ):
        raise ValueError(
            "participant_level_sampling is mutually exclusive with "
            "hybrid_sampler, imbalanced sampling, and num_samples."
        )

    train_batch_size = int(getattr(args, "train_batch_size", None) or args.batch_size)
    val_batch_size = int(getattr(args, "val_batch_size", None) or args.batch_size)
    test_batch_size = int(getattr(args, "test_batch_size", None) or args.batch_size)

    # 17565 -> val_size
    if args.distributed:
        print("Using distributed mode")
        if participant_level_sampling:
            raise NotImplementedError(
                "ParticipantEpochSampler currently supports one process per GPU job only."
            )
        if getattr(args, "hybrid_sampler", False):
            if args.distributed:
                raise NotImplementedError(
                    "HybridCoverageBalancedSampler currently supports one process per GPU job only."
                )
            train_sampler = HybridCoverageBalancedSampler(
                dataset=train_dataset,
                labels=train_dataset.get_multi_class_labels_for_balance(),
                batch_size=train_batch_size,
                coverage_per_batch=int(args.hybrid_coverage_per_batch),
                num_samples=args.num_samples or len(train_dataset),
                seed=args.seed,
            )
        elif args.imbalanced:
            print("Using imbalanced sampler")
            # train_sampler = ImbalancedDatasetSampler(
            #     dataset=train_dataset,
            #     labels=train_dataset.get_labels_for_balance(),
            #     num_samples=args.num_samples,
            # )
            # train_sampler = MultiLabelImbalancedDatasetSampler(
            #     dataset=train_dataset,
            #     num_samples=args.num_samples,
            #     labels=train_dataset.get_multi_class_labels_for_balance(),

            # )
            train_sampler = _build_validated_weighted_sampler(
                dataset=train_dataset,
                num_samples=args.num_samples if args.num_samples else len(train_dataset),
                split_name="train",
            )

        else:
            if args.num_samples:
                train_sampler = RandomSampler(
                    train_dataset,
                    num_samples=args.num_samples,
                )
            else:
                print("Not using imbalanced sampler")
                train_sampler = DistributedSampler(
                    train_dataset,
                    shuffle=True,
                )

        if weight_sample_validation:
            val_sampler = _build_validated_weighted_sampler(
                dataset=val_dataset,
                num_samples=int(args.num_samples * args.sample_val_ratio) if args.num_samples else len(val_dataset),
                split_name="val",
            )
        else:
            if args.sample_val_ratio:
                val_sampler = RandomSampler(
                    val_dataset,
                    num_samples=int(args.num_samples * args.sample_val_ratio),
                )
            else:
                val_sampler = DistributedSampler(
                    val_dataset, shuffle=False, drop_last=False
                )

        test_sampler = DistributedSampler(test_dataset, shuffle=False, drop_last=False)
    else:
        print("Not using distributed mode")
        if participant_level_sampling:
            print("Using one-row-per-participant epoch sampler")
            train_sampler = ParticipantEpochSampler(
                dataset=train_dataset,
                seed=args.seed,
            )
        elif getattr(args, "hybrid_sampler", False):
            print("Using hybrid coverage/balance sampler")
            train_sampler = HybridCoverageBalancedSampler(
                dataset=train_dataset,
                labels=train_dataset.get_multi_class_labels_for_balance(),
                batch_size=train_batch_size,
                coverage_per_batch=int(args.hybrid_coverage_per_batch),
                num_samples=args.num_samples or len(train_dataset),
                seed=args.seed,
            )
        elif args.imbalanced:
            print("Using imbalanced sampler")
            # train_sampler = ImbalancedDatasetSampler(
            #     dataset=train_dataset,
            #     labels=train_dataset.get_labels_for_balance(),
            #     num_samples=args.num_samples,
            # )
            # train_sampler = MultiLabelImbalancedDatasetSampler(
            #     dataset=train_dataset,
            #     labels=train_dataset.get_multi_class_labels_for_balance(),
            #     num_samples=args.num_samples,
            # )
            train_sampler = _build_validated_weighted_sampler(
                dataset=train_dataset,
                num_samples=args.num_samples if args.num_samples else len(train_dataset),
                split_name="train",
            )
        else:
            print("Not using imbalanced sampler")
            if args.num_samples:
                train_sampler = RandomSampler(
                    train_dataset,
                    num_samples=args.num_samples if args.num_samples else None,
                )
            else:
                train_sampler = None

        if weight_sample_validation:
            if args.num_samples and args.sample_val_ratio:
                val_sampler = _build_validated_weighted_sampler(
                    dataset=val_dataset,
                    num_samples=int(args.num_samples * args.sample_val_ratio) if args.num_samples else len(val_dataset),
                    split_name="val",
                )
            else:
                val_sampler = None
        else:
            if args.sample_val_ratio:
                val_sampler = RandomSampler(
                    val_dataset,
                    num_samples=int(args.num_samples * args.sample_val_ratio),
                )
            else:
                val_sampler = torch.utils.data.SequentialSampler(val_dataset)

        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    logging.info("Dataloader creating...")
    train_d = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        sampler=train_sampler,
        shuffle=True if train_sampler is None else False,
        **kwargs,
    )

    # Evaluation-specific worker settings override the shared loader settings,
    # but must not discard structural arguments such as collate_fn.  Losing the
    # list collator makes PyTorch's default collator require every optional
    # modality key to be present in every sample.
    eval_kwargs = dict(kwargs)
    if eval_loader_kwargs is not None:
        eval_kwargs.update(eval_loader_kwargs)

    val_d = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        sampler=val_sampler,
        shuffle=False,
        **eval_kwargs,
    )

    test_d = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        sampler=test_sampler,
        shuffle=False,
        **eval_kwargs,
    )

    return train_d, val_d, test_d
