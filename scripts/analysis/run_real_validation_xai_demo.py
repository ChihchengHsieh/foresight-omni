#!/usr/bin/env python3
"""Generate a de-identified token-occlusion XAI demo on real validation cases."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset.universal_image import build_image_level_universal_datasets
from dataset.universal_image import MODALITIES_TO_COLS
from engine.universal_dense_onehead import get_fixed_inputs_and_labels
from model.universal_dense.input_output_projs import (
    build_universal_input_output_projs_batch,
)
from model.universal_dense_vit_improved import build_universal_dense_vit
from train import get_args_parser
from utils.checkpoint import load_checkpoint_from_path
from utils.tensor import nested_to_device
from utils.yaml_config import add_config_arguments, parse_configured_args
from xai.token_attribution import (
    drop_modalities,
    encoded_token_manifest,
    masked_token_variants,
)


CASE_SPECS = (
    ("glaucoma", 5, 1),
    ("cvd", 5, 1),
    ("t2d", 5, 0),
)


def parse_case_specs(value: str):
    """Parse disease:year:label triplets while preserving the legacy default."""
    if not value:
        return CASE_SPECS
    parsed = []
    for raw_item in value.split(","):
        disease, year, label = (part.strip() for part in raw_item.split(":"))
        parsed.append((disease, int(year), int(label)))
    return tuple(parsed)


def parse_disease_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def select_paired_case(
    dataset,
    disease,
    desired_label_0y,
    desired_label_10y,
    used_positions,
    used_participants=None,
):
    """Select one deterministic participant with valid 0-year and 10-year labels."""
    label_0y = f"has_{disease}_in_0_years"
    label_10y = f"has_{disease}_in_10_years"
    frame = dataset.df
    participant_col = "patient_eid" if "patient_eid" in frame.columns else None
    candidate_positions = []
    for position in range(len(frame)):
        row = frame.iloc[position]
        if position in used_positions:
            continue
        participant = str(row[participant_col]) if participant_col else None
        if used_participants is not None and participant in used_participants:
            continue
        value_0y = row[label_0y]
        value_10y = row[label_10y]
        if pd.isna(value_0y) or pd.isna(value_10y):
            continue
        if (
            int(float(value_0y)) == desired_label_0y
            and int(float(value_10y)) == desired_label_10y
        ):
            candidate_positions.append(position)
    if not candidate_positions:
        raise RuntimeError(
            f"No paired validation candidate for {disease}: "
            f"0y={desired_label_0y}, 10y={desired_label_10y}"
        )
    position = candidate_positions[0]
    used_positions.add(position)
    if used_participants is not None and participant_col is not None:
        used_participants.add(str(frame.iloc[position][participant_col]))
    return position, dataset[position]


def disease_labels(args) -> list[str]:
    return [
        f"has_{disease}_in_{year}_years"
        for disease in args.diseases
        for year in args.progression_label_years
    ]


def build_datasets(args, labels):
    return build_image_level_universal_datasets(
        args,
        possible_inputs=args.input_modalities,
        possible_labels=labels,
        balance_label_cols=labels,
        quality_control=not args.no_quality_control,
        progression_label_years=args.progression_label_years,
        clinical_numerical_features=args.clinical_num_modalities,
        clinical_categorical_features=args.clinical_cat_modalities,
        enhanced_aug=args.enhanced_aug,
        no_aug=args.no_aug,
        numerical_label_cols=[],
        progression_label_ignorant_label_years=args.progression_label_ignorant_label_years,
        normalise_fundus_image=args.normalise_fundus_image,
        algo_qc=args.algo_qc,
        external_binary_phenotype_path=args.external_binary_phenotype_path,
        external_binary_phenotype_id_col=args.external_binary_phenotype_id_col,
        external_binary_phenotype_label_col=args.external_binary_phenotype_label_col,
        external_binary_target_disease=args.external_binary_target_disease,
        external_prs_path=args.external_prs_path,
        external_prs_id_col=args.external_prs_id_col,
        external_prs_score_col=args.external_prs_score_col,
        require_external_prs=args.require_external_prs,
    )


def build_model(args, train_dataset, labels, device):
    if "clinical_history" in args.input_modalities:
        args.clinical_history_vocab_size = len(train_dataset.icd10_code_to_id)
    if "questionnaire" in args.input_modalities:
        args.questionnaire_num_fields = train_dataset.questionnaire_num_fields
        args.questionnaire_category_vocab_size = (
            train_dataset.questionnaire_category_vocab_size
        )
    input_to_seq, _ = build_universal_input_output_projs_batch(
        args,
        device,
        possible_input_modalities=args.input_modalities,
        binary_label_cols=labels,
        numerical_label_cols=[],
        omics_num_tokens=args.omics_num_tokens,
        genotype_autoencoder_intermediate_dims=args.genotype_autoencoder_intermediate_dims,
        genotype_autoencoder_patch_sizes=args.genotype_autoencoder_patch_sizes,
        genotype_last_patch_size=args.genotype_last_patch_size,
        genotype_length=args.genotype_length,
        image_encoder_type=args.image_encoder_type,
    )
    model = build_universal_dense_vit(
        args,
        input_to_seq=input_to_seq,
        label_num_classes={disease: len(args.progression_label_years) for disease in args.diseases},
        label_tokens_len={disease: int(args.disease_label_tokens) for disease in args.diseases},
    ).to(device)
    checkpoint = load_checkpoint_from_path(args.xai_checkpoint, device)
    state = {
        key.replace("fundus-image", "fundus_image"): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def select_case(
    dataset,
    disease,
    year,
    desired_label,
    used_positions,
    used_participants=None,
):
    label = f"has_{disease}_in_{year}_years"
    frame = dataset.df
    participant_col = "patient_eid" if "patient_eid" in frame.columns else None
    candidate_positions = [
        position
        for position, value in enumerate(frame[label].tolist())
        if position not in used_positions
        and pd.notna(value)
        and int(float(value)) == desired_label
        and (
            used_participants is None
            or participant_col is None
            or str(frame.iloc[position][participant_col]) not in used_participants
        )
    ]
    if not candidate_positions:
        raise RuntimeError(f"No validation candidate for {label}={desired_label}")
    # Selection is deterministic and based only on the prespecified label stratum,
    # never on prediction or explanation appearance.
    position = candidate_positions[0]
    used_positions.add(position)
    if used_participants is not None and participant_col is not None:
        used_participants.add(str(frame.iloc[position][participant_col]))
    return position, dataset[position], label


def logits_for_variants(model, variants, disease, horizon_index, batch_size):
    values = []
    with torch.no_grad():
        for start in range(0, len(variants), batch_size):
            batch = variants[start : start + batch_size]
            output = model.forward_encoded_modalities(
                batch,
                output_labels=[[disease] for _ in batch],
            )["out"]
            values.extend(
                float(item[disease][horizon_index].detach().cpu()) for item in output
            )
    return values


DIRECT_SOURCE_COLUMNS = {
    "age": ["instance_age_at_time"],
    "iop": ["instance_iop"],
    "gender": ["patient_gender"],
    "vcdr": ["instance_vcdr"],
}

RAW_UNITS = {
    "instance_age_at_time": "years",
    "instance_iop": "mmHg",
    "instance_height_cm": "cm",
    "instance_weight_kg": "kg",
    "instance_body_mass_index_bmi": "kg/m^2",
    "instance_waist_circumference_cm": "cm",
    "instance_hip_circumference_cm": "cm",
    "instance_systolic_bp": "mmHg",
    "instance_diastolic_bp": "mmHg",
    "instance_pulse_rate": "bpm",
}


def token_source_columns(modality, count, dataset):
    if modality in DIRECT_SOURCE_COLUMNS:
        columns = DIRECT_SOURCE_COLUMNS[modality]
    elif modality in MODALITIES_TO_COLS:
        columns = list(MODALITIES_TO_COLS[modality])
    elif modality == "prs":
        columns = list(getattr(dataset, "prs_cols", []))
    else:
        columns = []
    return columns if len(columns) == count else [None] * count


def original_value(dataset, position, source_column, modality):
    if source_column is None or source_column not in dataset.df.columns:
        return None, "not directly recoverable"
    value = pd.to_numeric(
        pd.Series([dataset.df.iloc[position][source_column]]), errors="coerce"
    ).iloc[0]
    if pd.isna(value):
        return None, "missing"
    stats = getattr(dataset, "mean_std_map", {}).get(source_column)
    if stats:
        mean = float(stats["mean"])
        std = float(stats["std"])
        value = float(value) * std + mean
    else:
        value = float(value)
    if modality not in {"prs", "principal_components"} and value < 0:
        display = f"UKB missing code ({value:g})"
    elif modality in {"family_history", "mental_health", "socioeconomic"} and value in {
        0.0,
        1.0,
    }:
        display = "Yes" if value == 1.0 else "No"
    else:
        unit = RAW_UNITS.get(source_column, "")
        precision = 1 if abs(value) >= 10 else 3
        display = f"{value:.{precision}f}{f' {unit}' if unit else ''}"
    return value, display


def explain_case(
    model, dataset, position, item, disease, year, target_label, case_id, args, device
):
    labels = disease_labels(args)
    sample, targets = get_fixed_inputs_and_labels(item, args.input_modalities, labels)
    sample = nested_to_device(sample, device)
    encoded = model.encode_modalities([sample])[0]
    horizon_index = args.progression_label_years.index(year)
    baseline_logit = logits_for_variants(
        model, [encoded], disease, horizon_index, args.xai_mask_batch_size
    )[0]
    baseline_probability = float(torch.sigmoid(torch.tensor(baseline_logit)))

    manifest = encoded_token_manifest(encoded, dataset=dataset)
    modality_counts = {key: int(value.shape[0]) for key, value in encoded.items()}
    source_columns = {
        modality: token_source_columns(modality, count, dataset)
        for modality, count in modality_counts.items()
    }
    token_variants = masked_token_variants(encoded, manifest)
    masked_logits = logits_for_variants(
        model, token_variants, disease, horizon_index, args.xai_mask_batch_size
    )
    token_rows = []
    for item_meta, masked_logit in zip(manifest, masked_logits):
        row = dict(item_meta)
        source_column = source_columns[row["modality"]][int(row["token_index"])]
        raw_value, raw_value_display = original_value(
            dataset, position, source_column, row["modality"]
        )
        row.update(
            {
                "case": case_id,
                "disease": disease,
                "horizon_years": year,
                "baseline_logit": baseline_logit,
                "baseline_probability": baseline_probability,
                "masked_logit": masked_logit,
                "signed_logit_contribution": baseline_logit - masked_logit,
                "source_column": source_column,
                "raw_value": raw_value,
                "raw_value_display": raw_value_display,
            }
        )
        token_rows.append(row)

    dropped = drop_modalities(encoded)
    dropped_logits = logits_for_variants(
        model,
        [variant for _, variant in dropped],
        disease,
        horizon_index,
        args.xai_mask_batch_size,
    )
    modality_rows = []
    for (modality, _), masked_logit in zip(dropped, dropped_logits):
        modality_rows.append(
            {
                "case": case_id,
                "disease": disease,
                "horizon_years": year,
                "modality": modality,
                "baseline_logit": baseline_logit,
                "baseline_probability": baseline_probability,
                "masked_logit": masked_logit,
                "signed_logit_contribution": baseline_logit - masked_logit,
            }
        )

    ranked = sorted(
        token_rows,
        key=lambda row: abs(row["signed_logit_contribution"]),
        reverse=True,
    )
    removal_counts = [1, 3, 5]
    faithfulness_variants = []
    valid_counts = []
    for count in removal_counts:
        count = min(count, len(ranked))
        variant = dict(encoded)
        grouped = defaultdict(list)
        for row in ranked[:count]:
            grouped[row["modality"]].append(int(row["token_index"]))
        for modality, indices in grouped.items():
            changed = encoded[modality].clone()
            changed[indices] = 0
            variant[modality] = changed
        faithfulness_variants.append(variant)
        valid_counts.append(count)
    faithfulness_logits = logits_for_variants(
        model,
        faithfulness_variants,
        disease,
        horizon_index,
        args.xai_mask_batch_size,
    )

    return {
        "case": case_id,
        "disease": disease,
        "horizon_years": year,
        "observed_label": int(target_label),
        "baseline_logit": baseline_logit,
        "baseline_probability": baseline_probability,
        "available_modalities": list(encoded),
        "token_rows": token_rows,
        "modality_rows": modality_rows,
        "faithfulness": [
            {
                "top_tokens_removed": count,
                "masked_logit": masked_logit,
                "logit_drop": baseline_logit - masked_logit,
            }
            for count, masked_logit in zip(valid_counts, faithfulness_logits)
        ],
    }


def plot_cases(explanations, output_path):
    fig, axes = plt.subplots(len(explanations), 2, figsize=(14, 4.6 * len(explanations)))
    if len(explanations) == 1:
        axes = [axes]
    for row_axes, explanation in zip(axes, explanations):
        token_ax, modality_ax = row_axes
        top = sorted(
            explanation["token_rows"],
            key=lambda row: abs(row["signed_logit_contribution"]),
            reverse=True,
        )[:10][::-1]
        colors = ["#b2182b" if row["signed_logit_contribution"] > 0 else "#2166ac" for row in top]
        token_ax.barh(
            [row["feature"][:58] for row in top],
            [row["signed_logit_contribution"] for row in top],
            color=colors,
        )
        token_ax.axvline(0, color="black", linewidth=0.8)
        token_ax.set_xlabel("Prediction logit change when token is masked")
        token_ax.set_title(
            f"{explanation['case']}: {explanation['disease'].upper()} "
            f"{explanation['horizon_years']}y, risk={explanation['baseline_probability']:.3f}"
        )

        modality = sorted(
            explanation["modality_rows"],
            key=lambda row: abs(row["signed_logit_contribution"]),
        )
        modality_ax.barh(
            [row["modality"] for row in modality],
            [row["signed_logit_contribution"] for row in modality],
            color=["#b2182b" if row["signed_logit_contribution"] > 0 else "#2166ac" for row in modality],
        )
        modality_ax.axvline(0, color="black", linewidth=0.8)
        modality_ax.set_xlabel("Prediction logit change when modality is removed")
        modality_ax.set_title("Modality-level perturbation")
    fig.suptitle(
        "Real validation-case XAI prototype\nRed supports the prediction; blue suppresses it",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cases_by_disease(explanations, figure_dir):
    diseases = sorted({item["disease"] for item in explanations})
    for disease in diseases:
        selected = [item for item in explanations if item["disease"] == disease]
        plot_cases(selected, figure_dir / f"case_token_occlusion_{disease}.png")


def write_report(explanations, output_dir, checkpoint):
    lines = [
        "# Real Validation-Case Token XAI Demonstration",
        "",
        "This is a de-identified validation-cohort prototype using a frozen previously trained model. It does not inspect or select cases from the locked test set.",
        "",
        f"- Checkpoint: `{checkpoint}`",
        "- Primary attribution: zero-embedding token occlusion at the fusion-transformer input",
        "- Modality attribution: remove-one-modality perturbation",
        "- Case selection: deterministic, prespecified outcome strata; never selected by prediction or explanation appearance",
        "- Interpretation: signed model sensitivity, not biological causality",
        "",
        "![Token and modality attribution demonstration](figures/real_validation_xai_demo.png)",
        "",
    ]
    for explanation in explanations:
        lines.extend(
            [
                f"## {explanation['case']}: {explanation['disease'].upper()} at {explanation['horizon_years']} years",
                "",
                f"Observed label: **{explanation['observed_label']}**. Predicted risk: **{explanation['baseline_probability']:.3f}**.",
                "",
                "| Rank | Evidence token | Original value | Modality | Signed logit contribution |",
                "| ---: | --- | --- | --- | ---: |",
            ]
        )
        top = sorted(
            explanation["token_rows"],
            key=lambda row: abs(row["signed_logit_contribution"]),
            reverse=True,
        )[:10]
        for rank, row in enumerate(top, 1):
            lines.append(
                f"| {rank} | {row['feature']} | {row.get('raw_value_display', 'not available')} | `{row['modality']}` | {row['signed_logit_contribution']:+.4f} |"
            )
        lines.extend(["", "Faithfulness preview:", ""])
        for result in explanation["faithfulness"]:
            lines.append(
                f"- Mask top {result['top_tokens_removed']} token(s): logit change {result['logit_drop']:+.4f}."
            )
        lines.append("")
    lines.extend(
        [
            "## Limitations and next implementation",
            "",
            "This first demonstration ranks fusion content tokens using a zero-embedding perturbation. It deliberately excludes bracket tokens. OCT latent tokens and grouped omics/history tokens are not yet anatomically or feature resolved. The paper framework should next add multiple-reference integrated gradients, random-token deletion controls, Grad-CAM/region occlusion for fundus, slice/region attribution for OCT, and cohort-level stability summaries.",
            "",
        ]
    )
    (output_dir / "real_validation_xai_demo.md").write_text("\n".join(lines))


def main(args):
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    labels = disease_labels(args)
    train_dataset, val_dataset, _ = build_datasets(args, labels)
    model = build_model(args, train_dataset, labels, device)

    explanations = []
    used_positions = set()
    used_participants = set()
    paired_diseases = parse_disease_list(args.xai_paired_0_10_diseases)
    incident_diseases = set(parse_disease_list(args.xai_paired_incident_diseases))
    if paired_diseases:
        case_number = 0
        for disease in paired_diseases:
            desired_0y = 0 if disease in incident_diseases else 1
            position, item = select_paired_case(
                val_dataset,
                disease,
                desired_0y,
                1,
                used_positions,
                used_participants,
            )
            for year in (0, 10):
                case_number += 1
                target_name = f"has_{disease}_in_{year}_years"
                observed = int(float(item[target_name].item()))
                explanations.append(
                    explain_case(
                        model,
                        val_dataset,
                        position,
                        item,
                        disease,
                        year,
                        observed,
                        f"Case {case_number}",
                        args,
                        device,
                    )
                )
    else:
        for case_number, (disease, year, desired_label) in enumerate(
            parse_case_specs(args.xai_case_specs), 1
        ):
            position, item, target_name = select_case(
                val_dataset,
                disease,
                year,
                desired_label,
                used_positions,
                used_participants,
            )
            observed = int(float(item[target_name].item()))
            explanations.append(
                explain_case(
                    model,
                    val_dataset,
                    position,
                    item,
                    disease,
                    year,
                    observed,
                    f"Case {case_number}",
                    args,
                    device,
                )
            )

    token_rows = [row for case in explanations for row in case["token_rows"]]
    modality_rows = [row for case in explanations for row in case["modality_rows"]]
    pd.DataFrame(token_rows).to_csv(output_dir / "token_attributions.csv", index=False)
    pd.DataFrame(modality_rows).to_csv(output_dir / "modality_attributions.csv", index=False)
    serializable = [
        {key: value for key, value in case.items() if key not in {"token_rows", "modality_rows"}}
        for case in explanations
    ]
    (output_dir / "explanations.json").write_text(json.dumps(serializable, indent=2) + "\n")
    # A single vertically stacked atlas becomes invalid for large screening
    # pools. Disease-specific pages below remain bounded and are sufficient for
    # visual review; the complete source rows are always written above.
    if len(explanations) <= 50:
        plot_cases(explanations, figure_dir / "real_validation_xai_demo.png")
    else:
        (figure_dir / "combined_atlas_omitted.txt").write_text(
            f"Combined atlas omitted for {len(explanations)} cases; "
            "use the disease-specific pages and source CSV files.\n"
        )
    plot_cases_by_disease(explanations, figure_dir)
    write_report(explanations, output_dir, args.xai_checkpoint)
    print(f"Saved XAI demo to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Real validation-case XAI demo",
        parents=[get_args_parser()],
    )
    add_config_arguments(parser)
    parser.add_argument("--xai_checkpoint", required=True)
    parser.add_argument("--xai_mask_batch_size", type=int, default=64)
    parser.add_argument(
        "--xai_case_specs",
        default="",
        help="Comma-separated disease:year:label triplets for prespecified cases.",
    )
    parser.add_argument(
        "--xai_paired_0_10_diseases",
        default="",
        help="Comma-separated diseases for same-participant 0-year and 10-year XAI.",
    )
    parser.add_argument(
        "--xai_paired_incident_diseases",
        default="ad",
        help="Paired diseases requiring a 0-year negative and 10-year positive label.",
    )
    main(parse_configured_args(parser))
