"""Portable path defaults for the public Foresight-Omni code release.

The controlled UK Biobank and CLSA datasets are not distributed with this
repository. Set ``FORESIGHT_OMNI_DATA_ROOT`` or pass explicit paths through the
training command line/configuration.
"""

from __future__ import annotations

import os
from pathlib import Path


DATA_ROOT = Path(os.environ.get("FORESIGHT_OMNI_DATA_ROOT", "data")).expanduser()
LOG_PATH = str(Path(os.environ.get("FORESIGHT_OMNI_OUTPUT_ROOT", "outputs")).expanduser())
FUNDUS_DIR = str(DATA_ROOT / "ukb" / "fundus")
GLAUCOMA_SPREADSHEET_PATH = str(DATA_ROOT / "ukb" / "glaucoma.csv")
UNIVERSAL_SPREADSHEET_PATH = str(DATA_ROOT / "ukb" / "universal_side.csv")
INSTANCE_LEVEL_UNIVERSAL_SPREADSHEET_PATH = str(
    DATA_ROOT / "ukb" / "universal_instance_level.csv"
)
IMAGE_LEVEL_UNIVERSAL_SPREADSHEET_PATH = str(
    DATA_ROOT / "ukb" / "foresight_omni_manifest.parquet"
)
GENOTYPE_PATH = str(DATA_ROOT / "ukb" / "genotypes")

LOCAL_GENOTYPE_PATH = str(DATA_ROOT / "examples" / "example.ped")
LOCAL_UNIVERSAL_SPREADSHEET_PATH = str(DATA_ROOT / "examples" / "universal.csv")
LOCAL_GLAUCOMA_SPREADSHEET_PATH = str(DATA_ROOT / "examples" / "fundus.csv")
LOCAL_FUNDUS_EXAMPLE = str(DATA_ROOT / "examples" / "fundus.png")
CIFAR10_PATH = str(DATA_ROOT / "cifar10")
