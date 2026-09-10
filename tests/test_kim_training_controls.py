import unittest

import torch
from torch import nn
from torchvision import transforms as T

from dataset.aug import get_default_aug
from training.stage2 import build_stage2_optimizer
from training.model_selection import selection_metric_improved


class _TinyProjectModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_seq = nn.ModuleDict(
            {"fundus_image": nn.Linear(3, 4)}
        )
        self.output_layers = nn.ModuleDict({"t2d": nn.Linear(4, 1)})


class KimTrainingControlsTest(unittest.TestCase):
    def test_kim_augmentation_matches_contract(self):
        transform = get_default_aug(578, split="train", profile="kim")
        self.assertIsInstance(transform, T.Compose)
        self.assertEqual(
            [type(item) for item in transform.transforms],
            [
                T.RandomResizedCrop,
                T.RandomHorizontalFlip,
                T.RandomRotation,
                T.ToTensor,
                T.Normalize,
            ],
        )
        self.assertEqual(tuple(transform.transforms[0].size), (578, 578))
        self.assertEqual(transform.transforms[0].scale, (0.9, 1.0))
        self.assertEqual(transform.transforms[1].p, 0.5)
        self.assertEqual(transform.transforms[2].degrees, [-10.0, 10.0])

    def test_kim_validation_is_deterministic(self):
        transform = get_default_aug(578, split="val", profile="kim")
        self.assertEqual(
            [type(item) for item in transform.transforms],
            [T.Resize, T.ToTensor, T.Normalize],
        )

    def test_kim_enhanced_is_the_default_training_profile(self):
        transform = get_default_aug(578, split="train")
        self.assertEqual(
            [type(item) for item in transform.transforms],
            [
                T.RandomResizedCrop,
                T.RandomHorizontalFlip,
                T.RandomRotation,
                T.RandomApply,
                T.ToTensor,
                T.Normalize,
            ],
        )
        self.assertEqual(transform.transforms[0].scale, (0.9, 1.0))
        self.assertEqual(transform.transforms[1].p, 0.5)
        self.assertEqual(transform.transforms[2].degrees, [-10.0, 10.0])
        self.assertEqual(transform.transforms[3].p, 0.5)
        jitter = transform.transforms[3].transforms[0]
        self.assertIsInstance(jitter, T.ColorJitter)
        self.assertEqual(jitter.brightness, (0.9, 1.1))
        self.assertEqual(jitter.contrast, (0.9, 1.1))
        self.assertEqual(jitter.saturation, (0.95, 1.05))
        self.assertEqual(jitter.hue, (-0.01, 0.01))

    def test_kim_enhanced_validation_is_deterministic(self):
        transform = get_default_aug(578, split="val")
        self.assertEqual(
            [type(item) for item in transform.transforms],
            [T.Resize, T.ToTensor, T.Normalize],
        )

    def test_adam_optimizer_preserves_backbone_and_head_groups(self):
        model = _TinyProjectModel()
        optimizer, backbone_names, head_names = build_stage2_optimizer(
            model,
            backbone_lr=1e-4,
            head_lr=1e-4,
            weight_decay=0.0,
            optimizer_type="adam",
        )
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertNotIsInstance(optimizer, torch.optim.AdamW)
        self.assertTrue(any("input_to_seq.fundus_image" in name for name in backbone_names))
        self.assertTrue(any("output_layers.t2d" in name for name in head_names))
        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1e-4, 1e-4])

    def test_validation_loss_selection_is_lower_is_better(self):
        self.assertTrue(selection_metric_improved(0.119, None, "val_loss"))
        self.assertTrue(selection_metric_improved(0.117, 0.119, "val_loss", 0.001))
        self.assertFalse(selection_metric_improved(0.1185, 0.119, "val_loss", 0.001))
        self.assertFalse(selection_metric_improved(0.121, 0.119, "val_loss", 0.001))

    def test_auroc_selection_remains_higher_is_better(self):
        self.assertTrue(selection_metric_improved(0.702, 0.700, "mean_auroc", 0.001))
        self.assertFalse(selection_metric_improved(0.699, 0.700, "mean_auroc", 0.001))


if __name__ == "__main__":
    unittest.main()
