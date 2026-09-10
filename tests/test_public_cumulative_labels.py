import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from dataset.universal_public_dataset import (
    CombinedDatasetWrapper,
    build_combined_datasets,
)


class DummyImageDataset(torch.utils.data.Dataset):
    class_to_idx = {"control": 0, "glaucoma": 1}

    def __init__(self):
        self.samples = [
            (torch.zeros(3, 4, 4), 0),
            (torch.ones(3, 4, 4), 1),
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class PublicPrevalenceLabelTests(unittest.TestCase):
    def setUp(self):
        self.dataset = CombinedDatasetWrapper(
            DummyImageDataset(),
            false_class="control",
            name="public_glaucoma",
            split="train",
        )

    def test_prevalent_case_has_only_the_baseline_target(self):
        case = self.dataset[1]
        self.assertEqual(case["has_glaucoma_in_0_years"].item(), 1.0)
        for year in (2, 5, 10):
            self.assertNotIn(f"has_glaucoma_in_{year}_years", case)

    def test_control_has_only_the_baseline_target(self):
        control = self.dataset[0]
        self.assertEqual(control["has_glaucoma_in_0_years"].item(), 0.0)
        for year in (2, 5, 10):
            self.assertNotIn(f"has_glaucoma_in_{year}_years", control)

    @mock.patch("dataset.universal_public_dataset.load_combined_dataset")
    def test_public_data_follow_their_matching_split(self, load_dataset):
        load_dataset.side_effect = ["train", "val", "test"]
        result = build_combined_datasets(SimpleNamespace(), marker="value")
        self.assertEqual(result, ("train", "val", "test"))
        self.assertEqual(
            [call.kwargs["include_public"] for call in load_dataset.call_args_list],
            [True, True, True],
        )


if __name__ == "__main__":
    unittest.main()
