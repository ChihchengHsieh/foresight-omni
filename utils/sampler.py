from typing import Callable, Optional

import pandas as pd
import torch, math
import numpy as np
import torch.utils.data
import torch.distributed as dist
from torch.utils.data.sampler import Sampler


class ParticipantEpochSampler(Sampler):
    """Select exactly one row per participant in every epoch.

    Multi-image participants retain view diversity because the selected row is
    redrawn deterministically from ``seed + epoch``. The resulting epoch has no
    participant replacement and therefore cannot amplify a small number of
    cases merely because they have more rows or receive repeated draws.
    """

    def __init__(self, dataset, seed: int = 42):
        if not hasattr(dataset, "df") or "patient_eid" not in dataset.df.columns:
            raise ValueError(
                "ParticipantEpochSampler requires dataset.df['patient_eid']."
            )

        patient_ids = dataset.df["patient_eid"]
        if patient_ids.isna().any():
            raise ValueError("ParticipantEpochSampler found missing patient_eid values.")

        self.seed = int(seed)
        self.epoch = 0
        self.patient_to_indices = {}
        for index, patient_id in enumerate(patient_ids.astype(str).tolist()):
            self.patient_to_indices.setdefault(patient_id, []).append(index)
        if not self.patient_to_indices:
            raise ValueError("ParticipantEpochSampler cannot sample an empty dataset.")

        self.patient_ids = list(self.patient_to_indices)
        self.last_selected_indices = []
        self.total_draws = 0
        self.seen_indices = set()

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
        selected = [
            int(rng.choice(self.patient_to_indices[patient_id]))
            for patient_id in self.patient_ids
        ]
        rng.shuffle(selected)
        self.last_selected_indices = selected
        self.total_draws += len(selected)
        self.seen_indices.update(selected)
        return iter(selected)

    def __len__(self):
        return len(self.patient_ids)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "total_draws": self.total_draws,
            "seen_indices": sorted(self.seen_indices),
            "last_selected_indices": list(self.last_selected_indices),
        }

    def load_state_dict(self, state):
        self.epoch = int(state.get("epoch", 0))
        self.seed = int(state.get("seed", self.seed))
        self.total_draws = int(state.get("total_draws", 0))
        self.seen_indices = {int(value) for value in state.get("seen_indices", [])}
        self.last_selected_indices = [
            int(value) for value in state.get("last_selected_indices", [])
        ]

    def summary(self):
        return {
            "sampler": self.__class__.__name__,
            "epoch": self.epoch,
            "unique_participants_per_epoch": len(self.patient_ids),
            "rows_selected_last_epoch": len(self.last_selected_indices),
            "participant_repeat_draws_last_epoch": 0,
            "total_draws": self.total_draws,
            "unique_rows_seen": len(self.seen_indices),
        }


class MultiLabelImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for a multi-label imbalanced dataset.

    Arguments:
        dataset: the dataset to sample from
        labels: a list of lists, where each sublist contains the labels for a particular sample
        indices: a list of indices, default is None, in which case all elements in the dataset will be considered
        num_samples: number of samples to draw, default is None, in which case all elements will be drawn
        callback_get_label: a callback-like function which takes two arguments - dataset and index, default is None
    """

    def __init__(
        self,
        dataset,
        labels: torch.Tensor,  # (len(dataset), #labels)
        indices: list[int] = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices are not provided, all elements in the dataset will be considered

        self.indices = list(range(len(dataset))) if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # convert list of lists to a DataFrame
        df = pd.DataFrame(labels, index=self.indices)

        label_to_count = df.apply(pd.Series.value_counts)
        weights = [1 / (label_to_count[col][df[col]]) for col in df.columns]

        self.weights = torch.DoubleTensor(np.array(weights).sum(axis=0))
        # # compute the inverse frequency of each label
        # label_counts = df.apply(pd.Series.value_counts).fillna(0).sum(axis=0)
        # weights = 1.0 / label_counts
        # # calculate sample weights as the sum of inverse frequencies of the labels
        # sample_weights = df.apply(lambda x: weights[x].sum(), axis=1)

        # self.weights = torch.DoubleTensor(sample_weights.to_list())

    def __iter__(self):
        return iter(
            self.indices[i]
            for i in torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=True,
            )
        )

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class HybridCoverageBalancedSampler(Sampler):
    """Batch-ordered sampler mixing exhaustive coverage with rare positives.

    The first ``coverage_per_batch`` positions of every logical batch advance
    through a shuffled permutation without replacement. The remaining
    positions draw from endpoint-specific positive queues, selecting endpoints
    with inverse-square-root frequency. A row is never repeated within a
    logical batch. State is serialisable so resumed runs preserve coverage.
    """

    def __init__(
        self,
        dataset,
        labels: torch.Tensor,
        batch_size: int = 64,
        coverage_per_batch: int = 48,
        num_samples: int | None = None,
        seed: int = 42,
    ):
        self.indices = np.arange(len(dataset), dtype=np.int64)
        self.labels = torch.as_tensor(labels).bool().cpu().numpy()
        if self.labels.ndim != 2 or self.labels.shape[0] != len(self.indices):
            raise ValueError("labels must have shape [len(dataset), num_endpoints]")
        self.batch_size = int(batch_size)
        self.coverage_per_batch = int(coverage_per_batch)
        if not 0 < self.coverage_per_batch <= self.batch_size:
            raise ValueError("coverage_per_batch must be in [1, batch_size]")
        self.balance_per_batch = self.batch_size - self.coverage_per_batch
        requested = len(self.indices) if num_samples is None else int(num_samples)
        self.num_samples = max(self.batch_size, requested)
        self.seed = int(seed)
        self.epoch = 0
        self.coverage_cycle = 0
        self.coverage_offset = 0
        self.total_draws = 0
        self.endpoint_draw_counts = np.zeros(self.labels.shape[1], dtype=np.int64)
        self.patient_ids = (
            dataset.df["patient_eid"].astype(str).tolist()
            if hasattr(dataset, "df") and "patient_eid" in dataset.df.columns
            else list(
                getattr(
                    dataset,
                    "patient_ids",
                    [str(index) for index in self.indices],
                )
            )
        )
        if len(self.patient_ids) != len(self.indices):
            raise ValueError("patient_ids must have one value per dataset row")
        self.seen_indices = set()
        self.seen_patients = set()
        self.endpoint_counts = self.labels.sum(axis=0).astype(np.int64)
        self.eligible_endpoints = np.flatnonzero(self.endpoint_counts > 0)
        if len(self.eligible_endpoints):
            endpoint_weights = 1.0 / np.sqrt(self.endpoint_counts[self.eligible_endpoints])
            self.endpoint_probabilities = endpoint_weights / endpoint_weights.sum()
        else:
            self.endpoint_probabilities = np.array([], dtype=np.float64)
        self._reset_runtime_state()

    def _rng(self, salt: int = 0):
        return np.random.default_rng(self.seed + self.epoch * 1_000_003 + salt)

    def _reset_runtime_state(self):
        self._coverage_order = self._rng(self.coverage_cycle).permutation(self.indices)
        self._positive_queues = {}
        self._positive_offsets = {}

    def _next_coverage(self):
        if self.coverage_offset >= len(self._coverage_order):
            self.coverage_cycle += 1
            self.coverage_offset = 0
            self._coverage_order = self._rng(self.coverage_cycle).permutation(self.indices)
        value = int(self._coverage_order[self.coverage_offset])
        self.coverage_offset += 1
        return value

    def _next_positive(self, endpoint: int, used: set[int], rng):
        queue = self._positive_queues.get(endpoint)
        offset = self._positive_offsets.get(endpoint, 0)
        if queue is None or offset >= len(queue):
            queue = rng.permutation(np.flatnonzero(self.labels[:, endpoint]))
            offset = 0
        while offset < len(queue):
            value = int(queue[offset])
            offset += 1
            if value not in used:
                self._positive_queues[endpoint] = queue
                self._positive_offsets[endpoint] = offset
                return value
        self._positive_queues[endpoint] = queue
        self._positive_offsets[endpoint] = offset
        return None

    def __iter__(self):
        rng = self._rng(17 + self.total_draws)
        output = []
        while len(output) < self.num_samples:
            used = set()
            batch = []
            coverage_target = min(
                self.coverage_per_batch,
                self.num_samples - len(output),
            )
            while len(batch) < coverage_target:
                value = self._next_coverage()
                if value not in used:
                    used.add(value)
                    batch.append(value)
            for _ in range(min(self.balance_per_batch, self.num_samples - len(output) - len(batch))):
                value = None
                if len(self.eligible_endpoints):
                    for _attempt in range(max(4, len(self.eligible_endpoints))):
                        endpoint = int(rng.choice(self.eligible_endpoints, p=self.endpoint_probabilities))
                        value = self._next_positive(endpoint, used, rng)
                        if value is not None:
                            self.endpoint_draw_counts[endpoint] += 1
                            break
                while value is None or value in used:
                    candidate = self._next_coverage()
                    if candidate not in used:
                        value = candidate
                used.add(value)
                batch.append(value)
            output.extend(batch)
        output = output[: self.num_samples]
        self.total_draws += len(output)
        self.seen_indices.update(output)
        self.seen_patients.update(self.patient_ids[index] for index in output)
        return iter(output)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "coverage_cycle": self.coverage_cycle,
            "coverage_offset": self.coverage_offset,
            "total_draws": self.total_draws,
            "seed": self.seed,
            "coverage_order": self._coverage_order.tolist(),
            "positive_queues": {
                int(key): value.tolist() for key, value in self._positive_queues.items()
            },
            "positive_offsets": dict(self._positive_offsets),
            "endpoint_draw_counts": self.endpoint_draw_counts.tolist(),
            "seen_indices": sorted(self.seen_indices),
            "seen_patients": sorted(self.seen_patients),
        }

    def load_state_dict(self, state):
        self.epoch = int(state.get("epoch", 0))
        self.coverage_cycle = int(state.get("coverage_cycle", 0))
        self.coverage_offset = int(state.get("coverage_offset", 0))
        self.total_draws = int(state.get("total_draws", 0))
        self.seed = int(state.get("seed", self.seed))
        coverage_order = state.get("coverage_order")
        if coverage_order is None:
            self._reset_runtime_state()
        else:
            self._coverage_order = np.asarray(coverage_order, dtype=np.int64)
            self._positive_queues = {
                int(key): np.asarray(value, dtype=np.int64)
                for key, value in state.get("positive_queues", {}).items()
            }
            self._positive_offsets = {
                int(key): int(value)
                for key, value in state.get("positive_offsets", {}).items()
            }
        self.endpoint_draw_counts = np.asarray(
            state.get("endpoint_draw_counts", np.zeros(self.labels.shape[1])),
            dtype=np.int64,
        )
        self.seen_indices = {int(value) for value in state.get("seen_indices", [])}
        self.seen_patients = {str(value) for value in state.get("seen_patients", [])}

    def summary(self):
        completed_rows = min(len(self.indices), self.coverage_offset)
        return {
            "total_draws": int(self.total_draws),
            "coverage_cycle": int(self.coverage_cycle),
            "coverage_offset": int(self.coverage_offset),
            "coverage_percent_current_cycle": 100.0 * completed_rows / len(self.indices),
            "unique_rows_seen": len(self.seen_indices),
            "unique_patients_seen": len(self.seen_patients),
            "row_coverage_percent": 100.0 * len(self.seen_indices) / len(self.indices),
            "patient_coverage_percent": 100.0
            * len(self.seen_patients)
            / len(set(self.patient_ids)),
            "repeat_draws": int(self.total_draws - len(self.seen_indices)),
            "endpoint_draw_counts": self.endpoint_draw_counts.tolist(),
        }


class ImbalancedDatasetSampler(Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list,
        indices: list = None,
        num_samples: int = None,
        callback_get_label: Callable = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices
        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = labels
        df.index = self.indices
        self.df = df
        df = df.sort_index()

        label_to_count = df["label"].value_counts()
        weights = 1.0 / label_to_count[df["label"]]
        self.weights = torch.DoubleTensor(weights.to_list())

    def __iter__(self):
        return iter(
            self.indices[i]
            for i in torch.multinomial(
                self.weights,
                self.num_samples,
                replacement=True,
            )
        )

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
