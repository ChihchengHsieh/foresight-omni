import unittest
from types import SimpleNamespace

from dataset.universal_image import resolve_split_quality_control


class SplitQualityControlTests(unittest.TestCase):
    def test_global_qc_is_preserved_without_override(self):
        args = SimpleNamespace(quality_control_splits=[])
        self.assertEqual(
            resolve_split_quality_control(args, True),
            {"train": True, "val": True, "test": True},
        )
        self.assertEqual(
            resolve_split_quality_control(args, False),
            {"train": False, "val": False, "test": False},
        )

    def test_validation_only_override(self):
        args = SimpleNamespace(quality_control_splits=["val"])
        self.assertEqual(
            resolve_split_quality_control(args, False),
            {"train": False, "val": True, "test": False},
        )

    def test_validation_alias_and_multiple_splits(self):
        args = SimpleNamespace(quality_control_splits="validation,test")
        self.assertEqual(
            resolve_split_quality_control(args, False),
            {"train": False, "val": True, "test": True},
        )

    def test_invalid_split_fails_closed(self):
        args = SimpleNamespace(quality_control_splits=["dev"])
        with self.assertRaisesRegex(ValueError, "unsupported split"):
            resolve_split_quality_control(args, False)


if __name__ == "__main__":
    unittest.main()
