# Foresight-Omni

Research code accompanying **Foresight-Omni: Multimodal Multi-Disease Risk
Forecasting Across Future Clinical Horizons**.

This repository contains the code required to construct, train, evaluate and
interpret the clean Foresight-Omni model used in the manuscript. It is a code-only
release. Checkpoints, participant-level data, prediction files and cohort-derived
artifacts are not included.

## Included

- multimodal transformer and modality encoders
- ImageNet-pretrained ResNet-18 fundus encoder with the manuscript freeze policy
- stochastic modality dropout and hybrid endpoint-aware sampling
- cumulative disease heads for 0, 2, 5 and 10 years
- participant-cluster bootstrap evaluation
- Grad-CAM, regional occlusion and structured-feature masking analyses
- matched structured-only configuration
- CLSA direct-evaluation and cohort-adaptation code

The experimental residual-anchor and medication settings are disabled in the
provided manuscript configurations and are not required to reproduce the reported
model.

## Repository layout

```text
configs/                  Manuscript and matched-control configurations
dataset/                  Dataset, preprocessing and augmentation components
engine/                   Training and evaluation loops
evaluators/               AUROC and temporal evaluation utilities
model/                    Encoders, transformer fusion and prediction heads
scripts/                  Statistical evaluation and XAI entry points
training/                 Optimizer, scheduler, checkpoint and selection logic
utils/                    Shared runtime utilities
xai/                      Attribution and faithfulness methods
tests/                    Focused regression tests
```

## Environment

Python 3.10 and CUDA-compatible PyTorch are recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data access

UK Biobank and CLSA data are controlled-access resources and are not redistributed
here. Researchers must obtain access directly from the respective data custodians.
The public PAPILA and Glaucoma Fundus datasets must also be downloaded from their
original repositories.

Prepare the datasets according to [docs/data_manifest.md](docs/data_manifest.md).
Paths can be supplied through the YAML files, command-line arguments, or these
environment variables:

```bash
export FORESIGHT_OMNI_DATA_ROOT=/path/to/data
export FORESIGHT_OMNI_PUBLIC_DATA_ROOT=/path/to/public_datasets
export FORESIGHT_OMNI_CLSA_ROOT=/path/to/clsa
```

No participant-level example data are included because synthetic examples would
not validate the controlled-cohort preprocessing or label construction.

## Manuscript model

The paper model is defined by `configs/foresight_omni.yaml`. The fully resolved
record from the selected run is provided in
`configs/foresight_omni_resolved.yaml`, with local paths replaced by portable
placeholders.

Key settings are:

- input modalities: fundus image plus 11 structured modality groups
- diseases: glaucoma, Alzheimer disease, Parkinson disease, cardiovascular
  disease and type 2 diabetes
- cumulative horizons: 0, 2, 5 and 10 years
- medications excluded
- ImageNet-pretrained ResNet-18 fundus feature extractor frozen throughout
  training
- four-layer, four-head fusion transformer with dimension 256
- AdamW, head learning rate 1e-4 and weight decay 1e-2
- cosine schedule with 10% warm-up
- stochastic modality dropout increasing to 0.15 by epoch 10
- hybrid endpoint-aware sampling, without task-specific loss weights
- Kim-style fundus augmentation with mild colour jitter
- maximum 40 epochs with validation-based early stopping

Train with:

```bash
python train.py \
  --config configs/foresight_omni.yaml \
  --name foresight_omni \
  --output_dir outputs/foresight_omni \
  --backbone_freeze_patterns 'fundus_image.0.feature_extractor.' \
  --backbone_unfreeze_patterns 'fundus_image.0.feature_extractor.' \
  --delayed_backbone_scheduler
```

The optional fundus cache arguments in `train.py` can be used to stage decoded
224 by 224 images on local scratch storage.

## Matched structured-only analysis

```bash
python train.py \
  --config configs/structured_only.yaml \
  --name foresight_omni_structured_only \
  --output_dir outputs/structured_only
```

## Paired AUROC comparison

Prediction CSVs written by the training pipeline can be compared with
participant-level cluster bootstrap confidence intervals:

```bash
python scripts/evaluate_paired_models.py \
  --predictions-a outputs/model_a/test_predictions.csv \
  --predictions-b outputs/model_b/test_predictions.csv \
  --name-a structured \
  --name-b multimodal \
  --min-positives 50 \
  --n-bootstrap 2000 \
  --output outputs/paired_comparison.csv
```

## Explainability analyses

The `xai/` package implements the attribution primitives. Reproducible analysis
entry points are provided under `scripts/analysis/` for:

- case-level fundus Grad-CAM and signed regional occlusion
- modality and encoded structured-feature masking
- cohort-level structured-feature summaries
- cohort-average retinal maps and attribution faithfulness checks

Each script exposes its required inputs through `--help`. Attribution outputs are
associations with model predictions and should not be interpreted as causal
effects.

## Checkpoints

Model checkpoints are intentionally not included in this repository. The current
release provides source code and configurations only.

## Reproducibility notes

Model selection uses validation endpoints with at least 10 positive and at least
one negative observation. Manuscript mean summaries use endpoints with at least
50 positive test observations. Reported confidence intervals use participant-level
cluster bootstrap resampling where participant identifiers are available.

## Citation

Citation metadata will be updated with the final journal reference and archival
DOI when available. See `CITATION.cff`.

## License

No software license has yet been assigned. Please contact the authors before
redistributing or reusing the code. A permanent license will be selected before
the archival release.
