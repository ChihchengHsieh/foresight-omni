import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "analysis"
    / "evaluate_leave_one_modality_out.py"
)
SPEC = importlib.util.spec_from_file_location("leave_one_modality_out", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class LeaveOneModalityOutAnalysisTest(unittest.TestCase):
    def test_train_prediction_layout(self):
        directory = Path("/tmp/example")
        self.assertEqual(
            MODULE.prediction_path(
                directory, "unused", "full_input", layout="train"
            ).name,
            "best_auroc_ukb_stage2_test_predictions.csv",
        )
        self.assertEqual(
            MODULE.prediction_path(
                directory, "unused", "fundus_image", layout="train"
            ).name,
            "best_auroc_without_fundus_image_ukb_stage2_test_predictions.csv",
        )
    def test_weighted_auc_with_unit_cluster_counts_matches_sklearn(self):
        target = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8)
        score = np.asarray([0.1, 0.8, 0.2, 0.9, 0.7, 0.6], dtype=float)
        cluster_codes = np.asarray([0, 0, 1, 2, 3, 3], dtype=int)
        counts = np.ones((1, 4), dtype=np.int16)
        observed = MODULE.weighted_auc_batch(target, score, cluster_codes, counts)[0]
        self.assertTrue(np.isclose(observed, roc_auc_score(target, score)))

    def test_paired_cluster_bootstrap_is_reproducible_and_paired(self):
        target = np.asarray([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
        full_score = np.asarray([0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6])
        ablated_score = np.asarray([0.2, 0.7, 0.3, 0.8, 0.4, 0.6, 0.5, 0.55])
        participants = np.asarray(["a", "a", "b", "c", "d", "d", "e", "f"])
        first = MODULE.paired_cluster_bootstrap(
            target,
            full_score,
            ablated_score,
            participants,
            n_bootstrap=200,
            seed=17,
        )
        second = MODULE.paired_cluster_bootstrap(
            target,
            full_score,
            ablated_score,
            participants,
            n_bootstrap=200,
            seed=17,
        )
        self.assertEqual(first, second)
        self.assertEqual(first["bootstrap_samples"], 200)
        self.assertTrue(
            np.isclose(
                first["delta_auroc"],
                first["ablated_auroc"] - first["full_auroc"],
            )
        )

    def test_historical_baseline_comparison_detects_equivalent_predictions(self):
        rows = []
        for disease in MODULE.DISEASES:
            row = {"index": 0, "disease": disease}
            for year in MODULE.HORIZONS:
                row[f"pred_{year}y"] = 0.25 + year / 100
                row[f"tgt_{year}y"] = 0.0
            rows.append(row)
        current = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as directory:
            reference_dir = Path(directory)
            for disease in MODULE.DISEASES:
                current.loc[current["disease"].eq(disease)].drop(
                    columns="disease"
                ).to_csv(reference_dir / f"{disease}.csv", index=False)
            result = MODULE.compare_full_input_to_reference(current, reference_dir)
        self.assertTrue(result["all_diseases_equivalent_at_1e-6"])

    def test_reference_target_masks_restore_nan_and_allow_reference_only_rows(self):
        rows = []
        for disease in MODULE.DISEASES:
            row = {"index": 0, "disease": disease}
            for year in MODULE.HORIZONS:
                row[f"tgt_{year}y"] = 0.0
            rows.append(row)
        current = pd.DataFrame(rows)

        with tempfile.TemporaryDirectory() as directory:
            reference_dir = Path(directory)
            for disease in MODULE.DISEASES:
                reference_rows = []
                for index in (0, 1):
                    row = {"index": index}
                    for year in MODULE.HORIZONS:
                        row[f"tgt_{year}y"] = np.nan if index == 0 else 1.0
                    reference_rows.append(row)
                pd.DataFrame(reference_rows).to_csv(
                    reference_dir / f"{disease}.csv", index=False
                )
            corrected, audit = MODULE.apply_reference_target_masks(
                current, reference_dir
            )

        self.assertTrue(corrected[[f"tgt_{year}y" for year in MODULE.HORIZONS]].isna().all().all())
        self.assertEqual(
            audit["by_disease"]["glaucoma"]["reference_only_rows"], 1
        )
        self.assertEqual(
            audit["by_disease"]["glaucoma"]["raw_target_cells_corrected"], 4
        )


if __name__ == "__main__":
    unittest.main()
