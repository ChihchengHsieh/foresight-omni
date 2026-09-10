import unittest

import torch

from training.stage2 import (
    build_stage2_optimizer,
    build_stage2_scheduler,
    set_backbone_requires_grad,
    stage2_backbone_modality,
    stage2_epoch_range,
)


class _ToyFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_seq = torch.nn.ModuleDict(
            {
                "fundus_image": torch.nn.Linear(4, 4),
                "questionnaire": torch.nn.Linear(4, 4),
            }
        )
        self.transformer = torch.nn.Linear(4, 4)
        self.output_layers = torch.nn.ModuleDict({"dr": torch.nn.Linear(4, 2)})


class _ToyPartialFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_seq = torch.nn.ModuleDict(
            {
                "fundus_image": torch.nn.Sequential(
                    torch.nn.ModuleDict(
                        {
                            "feature_extractor": torch.nn.Sequential(
                                torch.nn.Linear(4, 4),
                                torch.nn.Linear(4, 4),
                            ),
                            "token_proj": torch.nn.Linear(4, 4),
                        }
                    )
                ),
                "questionnaire": torch.nn.Linear(4, 4),
            }
        )


class SelectiveStage2FreezingTests(unittest.TestCase):
    def test_eval_only_checkpoint_never_returns_training_epochs(self):
        self.assertEqual(
            list(stage2_epoch_range(1, 200, eval_only_checkpoint=True)),
            [],
        )
        self.assertEqual(
            list(stage2_epoch_range(18, 18, eval_only_checkpoint=True)),
            [],
        )
        self.assertEqual(
            list(stage2_epoch_range(3, 5, eval_only_checkpoint=False)),
            [3, 4, 5],
        )

    def test_modality_parser_accepts_wrapped_parameter_names(self):
        self.assertEqual(
            stage2_backbone_modality("module.input_to_seq.fundus_image.weight"),
            "fundus_image",
        )
        self.assertIsNone(stage2_backbone_modality("module.transformer.weight"))

    def test_only_selected_encoder_is_frozen(self):
        model = _ToyFusion()
        matched, changed = set_backbone_requires_grad(
            model,
            False,
            modalities={"fundus_image"},
        )
        self.assertEqual(matched, 2)
        self.assertEqual(changed, 2)
        self.assertTrue(
            all(not parameter.requires_grad for parameter in model.input_to_seq["fundus_image"].parameters())
        )
        self.assertTrue(
            all(parameter.requires_grad for parameter in model.input_to_seq["questionnaire"].parameters())
        )
        self.assertTrue(all(parameter.requires_grad for parameter in model.transformer.parameters()))

    def test_optimizer_excludes_permanently_frozen_encoder(self):
        model = _ToyFusion()
        set_backbone_requires_grad(
            model,
            False,
            modalities={"fundus_image"},
        )
        _, backbone_names, head_names = build_stage2_optimizer(
            model,
            backbone_lr=1e-5,
            head_lr=3e-4,
            weight_decay=1e-2,
            frozen_backbone_modalities={"fundus_image"},
            head_backbone_modalities={"questionnaire"},
        )
        self.assertFalse(any("fundus_image" in name for name in backbone_names + head_names))
        self.assertFalse(any("questionnaire" in name for name in backbone_names))
        self.assertTrue(any("questionnaire" in name for name in head_names))
        self.assertTrue(any("transformer" in name for name in head_names))
        self.assertTrue(any("output_layers.dr" in name for name in head_names))

    def test_partial_unfreeze_keeps_earlier_backbone_frozen(self):
        model = _ToyPartialFusion()
        set_backbone_requires_grad(model, False)
        matched, changed = set_backbone_requires_grad(
            model,
            True,
            name_patterns=("feature_extractor.1.", "token_proj."),
        )
        self.assertEqual(matched, 4)
        self.assertEqual(changed, 4)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.input_to_seq["fundus_image"][0]["feature_extractor"][0].parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.input_to_seq["fundus_image"][0]["feature_extractor"][1].parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.input_to_seq["fundus_image"][0]["token_proj"].parameters()
            )
        )
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.input_to_seq["questionnaire"].parameters()
            )
        )

    def test_selective_freeze_leaves_random_adapters_trainable(self):
        model = _ToyPartialFusion()
        matched, changed = set_backbone_requires_grad(
            model,
            False,
            name_patterns=("fundus_image.0.feature_extractor.",),
        )
        self.assertEqual(matched, 4)
        self.assertEqual(changed, 4)
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in model.input_to_seq["fundus_image"][0]["feature_extractor"].parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.input_to_seq["fundus_image"][0]["token_proj"].parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for parameter in model.input_to_seq["questionnaire"].parameters()
            )
        )

    def test_optimizer_assigns_only_pretrained_pattern_to_backbone_lr(self):
        model = _ToyPartialFusion()
        optimizer, backbone_names, head_names = build_stage2_optimizer(
            model,
            backbone_lr=1e-6,
            head_lr=1e-4,
            weight_decay=1e-2,
            backbone_param_patterns=("fundus_image.0.feature_extractor.",),
        )
        self.assertTrue(backbone_names)
        self.assertTrue(
            all("fundus_image.0.feature_extractor." in name for name in backbone_names)
        )
        self.assertTrue(any("fundus_image.0.token_proj." in name for name in head_names))
        self.assertTrue(any("questionnaire" in name for name in head_names))
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [1e-6, 1e-4],
        )

    def test_delayed_scheduler_starts_backbone_after_freeze(self):
        model = _ToyPartialFusion()
        optimizer, _, _ = build_stage2_optimizer(
            model,
            backbone_lr=1e-6,
            head_lr=1e-4,
            weight_decay=0.0,
            backbone_param_patterns=("fundus_image.0.feature_extractor.",),
        )
        scheduler = build_stage2_scheduler(
            "cosine",
            optimizer,
            steps_per_epoch=2,
            start_epoch=1,
            total_epochs=4,
            warmup_ratio=0.0,
            delayed_backbone_schedule=True,
            backbone_freeze_epochs=2,
            backbone_warmup_epochs=1,
        )
        self.assertEqual(scheduler.get_last_lr()[0], 0.0)
        self.assertEqual(scheduler.get_last_lr()[1], 1e-4)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
            self.assertEqual(scheduler.get_last_lr()[0], 0.0)
        optimizer.step()
        scheduler.step()
        self.assertGreater(scheduler.get_last_lr()[0], 0.0)


if __name__ == "__main__":
    unittest.main()
