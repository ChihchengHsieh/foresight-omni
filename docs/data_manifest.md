# Data manifest

The controlled cohort data are not included. The training code expects a prepared
Parquet manifest containing one row per image-level observation.

## Core columns

| Column | Meaning |
| --- | --- |
| `patient_eid` | Stable participant identifier used for disjoint splitting and clustered inference |
| `split` | Participant-level `train`, `val` or `test` assignment |
| `image_path` | Path to the corresponding fundus image |
| `image_quality` | Existing image-quality label, where 1 indicates inclusion |
| `has_<disease>_in_<horizon>_years` | Binary cumulative target for each disease and horizon |

The configured diseases are `glaucoma`, `ad`, `pd`, `cvd` and `t2d`; horizons are
0, 2, 5 and 10 years. Missing targets must remain missing rather than being
converted to controls.

Structured input column definitions are listed in `MODALITIES_TO_COLS` in
`dataset/universal_image.py`. They cover age, sex, intraocular pressure, polygenic
risk scores, anthropometrics, family history, principal components, lifestyle,
mental health, socioeconomic variables and vital signs. Medication columns are
not used by the manuscript model.

## Quality control

The manuscript configuration expects both an existing image-quality label in the
manifest and an aligned algorithmic QC CSV supplied through `algo_qc_path`. The
QC CSV must contain an `is_bad` Boolean column in the same verified image order as
the prepared manifest. Dataset construction fails when the manifest and QC table
have different row counts. Users must verify image-order alignment while preparing
the release manifest.

## Public glaucoma datasets

Arrange PAPILA and Glaucoma Fundus as torchvision `ImageFolder` datasets:

```text
data/public/
  PAPILA/
    train/anormal/
    train/bsuspectglaucoma/
    train/cglaucoma/
    val/...
    test/...
  Glaucoma_fundus/
    train/anormal_control/
    train/bearly_glaucoma/
    train/cadvanced_glaucoma/
    val/...
    test/...
```

Suspect or early-glaucoma classes are excluded from the manuscript analysis.
These datasets contribute only to the prevalent-glaucoma target because they do
not contain longitudinal follow-up labels.

## CLSA

Set `FORESIGHT_OMNI_CLSA_ROOT` to a directory containing the prepared CLSA
manifest, fundus images, QC files and PRS files. The default portable layout is:

```text
data/clsa/
  clsa_multimodal.parquet
  images/
  prs/
  gwas_linking_key.csv
  qc/algorithmic_qc.csv
  qc/dl_qc.csv
```

All CLSA train, validation and locked-test partitions must be participant-disjoint.
