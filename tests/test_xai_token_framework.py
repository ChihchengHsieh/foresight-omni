import unittest

import torch
from torch import nn

from dataset.universal_image import ImageLevelUKBUniversalDataset
from model.universal_dense_vit_improved import UniversalModel
from xai.image_attribution import (
    fundus_region_occlusion_variants,
    oct_slice_occlusion_variants,
    oct_within_slice_region_variants,
)
from xai.faithfulness import attach_deletion_logits, ranked_deletion_plan
from xai.provenance import (
    clinical_history_manifest,
    fundus_spatial_manifest,
    omics_feature_manifest,
)
from xai.raw_attribution import (
    conditional_modality_logits,
    integrated_gradients_for_modality,
    raw_encoded_logit_difference,
)
from xai.structured_attribution import (
    omics_reference_variants,
    remove_clinical_history_event,
)
from xai.token_attribution import encoded_token_manifest, masked_token_variants


class IdentityTransformer(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.dim = dim
        self.return_intermediate = False

    def forward(self, x, mask, pos, need_attn_weights=False):
        result = {"out": x}
        if need_attn_weights:
            result["attn_weights"] = torch.zeros(
                1, x.shape[0], 1, x.shape[1], x.shape[1], device=x.device
            )
        return result


class SumAttributionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_seq = nn.ModuleDict({"omics": nn.Identity()})

    def encode_modalities(self, samples):
        return [
            {"omics": self.input_to_seq["omics"](sample["omics"].unsqueeze(0))[0]}
            for sample in samples
        ]

    def forward_encoded_modalities(self, encoded, output_labels):
        return {
            "out": [
                {"cvd": torch.stack([item["omics"].sum()] * 4)}
                for item in encoded
            ]
        }

    def forward(self, samples, output_labels):
        return self.forward_encoded_modalities(
            self.encode_modalities(samples), output_labels
        )


class XAITokenFrameworkTests(unittest.TestCase):
    def build_model(self):
        return UniversalModel(
            transformer=IdentityTransformer(),
            input_to_seq=nn.ModuleDict({"vitals": nn.Identity()}),
            label_tokens_len={"cvd": 1},
            label_num_classes={"cvd": 4},
            modality_container_type="bracket",
            pos_enc_strategy="rotary",
        ).eval()

    def test_raw_and_cached_encoded_paths_are_identical(self):
        model = self.build_model()
        raw = [{"vitals": torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 3)}]
        with torch.no_grad():
            ordinary = model(raw, output_labels=[["cvd"]])["out"][0]["cvd"]
            encoded = model.encode_modalities(raw)
            cached = model.forward_encoded_modalities(
                encoded, output_labels=[["cvd"]]
            )["out"][0]["cvd"]
        torch.testing.assert_close(ordinary, cached)

    def test_token_manifest_and_masking_preserve_other_tokens(self):
        encoded = {"vitals": torch.arange(12, dtype=torch.float32).reshape(3, 4)}
        manifest = encoded_token_manifest(encoded)
        self.assertEqual(
            [item["feature"] for item in manifest],
            ["Systolic blood pressure", "Diastolic blood pressure", "Pulse rate"],
        )
        variants = masked_token_variants(encoded, manifest)
        self.assertTrue(torch.equal(variants[0]["vitals"][0], torch.zeros(4)))
        torch.testing.assert_close(variants[0]["vitals"][1:], encoded["vitals"][1:])
        torch.testing.assert_close(
            encoded["vitals"], torch.arange(12, dtype=torch.float32).reshape(3, 4)
        )

    def test_raw_encoded_consistency_diagnostic_and_conditional_variants(self):
        model = self.build_model()
        sample = {"vitals": torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 3)}
        self.assertLess(raw_encoded_logit_difference(model, sample, "cvd", 2), 1e-7)
        encoded = model.encode_modalities([sample])[0]
        variants = [sample["vitals"].clone(), sample["vitals"].clone()]
        variants[1][0].zero_()
        logits = conditional_modality_logits(
            model,
            encoded,
            "vitals",
            variants,
            "cvd",
            2,
            batch_size=2,
        )
        self.assertEqual(len(logits), 2)

    def test_integrated_gradients_is_complete_for_linear_raw_feature_model(self):
        model = SumAttributionModel().eval()
        value = torch.tensor([1.0, -2.0, 3.0])
        encoded = model.encode_modalities([{"omics": value}])[0]
        attribution, error = integrated_gradients_for_modality(
            model,
            encoded,
            "omics",
            value,
            torch.zeros_like(value),
            "cvd",
            2,
            steps=8,
        )
        torch.testing.assert_close(attribution, value)
        self.assertLess(error, 1e-6)

    def test_fundus_manifest_and_occlusion_are_spatially_traceable(self):
        image = torch.ones(3, 14, 14)
        manifest, variants = fundus_region_occlusion_variants(image, grid_size=7)
        self.assertEqual(manifest, fundus_spatial_manifest(49, 14, 14))
        self.assertEqual(len(variants), 49)
        self.assertTrue(torch.equal(variants[0][:, :2, :2], torch.zeros(3, 2, 2)))
        self.assertTrue(torch.equal(variants[0][:, 2:, 2:], image[:, 2:, 2:]))

    def test_oct_slice_occlusion_preserves_other_slices(self):
        volume = torch.ones(4, 1, 8, 8)
        manifest, variants = oct_slice_occlusion_variants(volume)
        self.assertEqual(len(manifest), 4)
        self.assertTrue(torch.equal(variants[2][2], torch.zeros(1, 8, 8)))
        torch.testing.assert_close(variants[2][[0, 1, 3]], volume[[0, 1, 3]])
        regions, region_variants = oct_within_slice_region_variants(
            volume, [1], grid_size=2
        )
        self.assertEqual(len(regions), 4)
        self.assertTrue(
            torch.equal(region_variants[0][1, :, :4, :4], torch.zeros(1, 4, 4))
        )

    def test_clinical_event_removal_recomputes_burden(self):
        history = torch.zeros(5, 6)
        history[:3, 0] = torch.tensor([4.0, 4.0, 8.0])
        history[:3, 1] = torch.log1p(torch.tensor([10.0, 20.0, 30.0]))
        history[:3, 4] = torch.log1p(torch.tensor(3.0))
        history[:3, 5] = torch.log1p(torch.tensor(2.0))
        removed = remove_clinical_history_event(history, 0)
        self.assertEqual(removed[:, 0].tolist(), [4.0, 8.0, 0.0, 0.0, 0.0])
        torch.testing.assert_close(removed[:2, 4], torch.log1p(torch.tensor([2.0, 2.0])))
        torch.testing.assert_close(removed[:2, 5], torch.log1p(torch.tensor([2.0, 2.0])))
        manifest = clinical_history_manifest(
            removed, id_to_code={4: "I10", 8: "E11"}
        )
        self.assertEqual([item["icd10_code"] for item in manifest], ["I10", "E11"])

    def test_dataset_clinical_provenance_rebuilds_exact_counterfactual(self):
        dataset = object.__new__(ImageLevelUKBUniversalDataset)
        dataset.clinical_history_max_len = 4
        dataset.icd10_code_to_id = {"<PAD>": 0, "<UNK>": 1, "I10": 2, "E11": 3}
        events = [
            {"code": "I10", "diagnosis_date": "2018-01-01"},
            {"code": "I10", "diagnosis_date": "2019-01-01"},
            {"code": "E11", "diagnosis_date": "2020-01-01"},
        ]
        dataset._ImageLevelUKBUniversalDataset__iter_preindex_clinical_history_events = (
            lambda _row: iter(events)
        )
        row = {"instance_assessment_centre_visit_date": "2021-01-01"}
        original, provenance = dataset.get_clinical_history_with_provenance(row)
        removed, removed_provenance = dataset.get_clinical_history_with_provenance(
            row, excluded_event_indices=[provenance[0]["source_event_index"]]
        )
        self.assertEqual(len(provenance), 3)
        self.assertEqual(len(removed_provenance), 2)
        torch.testing.assert_close(removed[:2, 4], torch.log1p(torch.tensor([2.0, 2.0])))
        # Removing E11 leaves only one unique ICD code.
        torch.testing.assert_close(removed[:2, 5], torch.log1p(torch.tensor([1.0, 1.0])))

    def test_omics_provenance_and_missingness_perturbations(self):
        values = torch.tensor([1.5, float("nan")])
        manifest = omics_feature_manifest(
            "metabolomics",
            ["met_0_ApoB", "met_0_GlycA"],
            values,
            mean_std_map={"met_0_ApoB": {"mean": 10.0, "std": 2.0}},
            feature_name_map={"0": "Apolipoprotein B"},
        )
        self.assertEqual(manifest[0]["original_value"], 13.0)
        self.assertFalse(manifest[1]["observed"])
        variants = omics_reference_variants(values)
        self.assertEqual(
            [(index, kind) for index, kind, _ in variants],
            [
                (0, "observed_to_reference"),
                (0, "observed_to_missing"),
                (1, "missing_to_reference"),
            ],
        )

    def test_ranked_deletion_plan_has_matched_top_low_random_controls(self):
        rows = [
            {"feature_index": 0, "signed_logit_contribution": 0.1},
            {"feature_index": 1, "signed_logit_contribution": -0.5},
            {"feature_index": 2, "signed_logit_contribution": 0.2},
        ]
        plan = ranked_deletion_plan(
            rows, "feature_index", counts=(1, 3, 5), seed=7
        )
        self.assertEqual(len(plan), 6)
        self.assertEqual(plan[0]["identities"], [1])
        self.assertEqual(plan[2]["identities"], [0])
        attached = attach_deletion_logits(plan, [0.5] * len(plan), 1.0)
        self.assertTrue(all(row["logit_drop"] == 0.5 for row in attached))


if __name__ == "__main__":
    unittest.main()
