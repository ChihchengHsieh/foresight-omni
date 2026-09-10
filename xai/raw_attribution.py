from __future__ import annotations

from typing import Iterable, Sequence

import torch


def clone_sample(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Clone tensor values while preserving non-tensor sample metadata."""
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else value
        for key, value in sample.items()
    }


def target_logit_from_output(
    output: dict,
    disease: str,
    horizon_index: int,
) -> torch.Tensor:
    return output["out"][0][disease][horizon_index]


def raw_target_logit(
    model,
    sample: dict[str, torch.Tensor],
    disease: str,
    horizon_index: int,
) -> torch.Tensor:
    output = model([sample], output_labels=[[disease]])
    return target_logit_from_output(output, disease, horizon_index)


def encoded_target_logits(
    model,
    encoded_variants: Sequence[dict[str, torch.Tensor]],
    disease: str,
    horizon_index: int,
    batch_size: int = 64,
) -> list[float]:
    """Evaluate cached encoded variants without re-running unchanged encoders."""
    values: list[float] = []
    with torch.no_grad():
        for start in range(0, len(encoded_variants), batch_size):
            batch = encoded_variants[start : start + batch_size]
            output = model.forward_encoded_modalities(
                batch,
                output_labels=[[disease] for _ in batch],
            )["out"]
            values.extend(
                float(item[disease][horizon_index].detach().cpu())
                for item in output
            )
    return values


def encode_modality_variants(
    model,
    encoded_base: dict[str, torch.Tensor],
    modality: str,
    raw_variants: Sequence[torch.Tensor],
    encoder_batch_size: int = 64,
) -> list[dict[str, torch.Tensor]]:
    """Encode one perturbed modality and reuse all other modality encodings."""
    if modality not in model.input_to_seq:
        raise KeyError(f"Model has no encoder for modality [{modality}]")
    if modality not in encoded_base:
        raise KeyError(f"Baseline sample has no modality [{modality}]")

    encoder = model.input_to_seq[modality]
    try:
        encoder_device = next(encoder.parameters()).device
    except StopIteration:
        encoder_device = encoded_base[modality].device
    variants: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for start in range(0, len(raw_variants), encoder_batch_size):
            raw_batch = torch.stack(
                list(raw_variants[start : start + encoder_batch_size]), dim=0
            ).to(encoder_device)
            encoded_batch = encoder(raw_batch.float())
            for encoded_value in encoded_batch:
                variant = dict(encoded_base)
                variant[modality] = encoded_value
                variants.append(variant)
    return variants


def conditional_modality_logits(
    model,
    encoded_base: dict[str, torch.Tensor],
    modality: str,
    raw_variants: Sequence[torch.Tensor],
    disease: str,
    horizon_index: int,
    batch_size: int = 64,
) -> list[float]:
    encoded_variants = encode_modality_variants(
        model,
        encoded_base,
        modality,
        raw_variants,
        encoder_batch_size=batch_size,
    )
    return encoded_target_logits(
        model,
        encoded_variants,
        disease,
        horizon_index,
        batch_size=batch_size,
    )


def raw_encoded_logit_difference(
    model,
    sample: dict[str, torch.Tensor],
    disease: str,
    horizon_index: int,
) -> float:
    """Regression diagnostic for the raw and cached-encoder inference paths."""
    with torch.no_grad():
        raw = raw_target_logit(model, sample, disease, horizon_index)
        encoded = model.encode_modalities([sample])
        cached = target_logit_from_output(
            model.forward_encoded_modalities(
                encoded,
                output_labels=[[disease]],
            ),
            disease,
            horizon_index,
        )
    return float((raw - cached).abs().detach().cpu())


def integrated_gradients_for_modality(
    model,
    encoded_base: dict[str, torch.Tensor],
    modality: str,
    raw_value: torch.Tensor,
    reference: torch.Tensor,
    disease: str,
    horizon_index: int,
    steps: int = 32,
) -> tuple[torch.Tensor, float]:
    """Integrated gradients through one encoder and the final fusion prediction.

    Other modality encodings are held fixed. The returned completeness error is
    |sum(attributions) - (f(x) - f(reference))|.
    """
    if steps < 2:
        raise ValueError("Integrated gradients requires at least two steps")
    if raw_value.shape != reference.shape:
        raise ValueError(
            f"Input/reference shape mismatch: {raw_value.shape} != {reference.shape}"
        )

    encoder = model.input_to_seq[modality]
    total_gradient = torch.zeros_like(raw_value, dtype=torch.float32)
    alphas = torch.linspace(
        0.0,
        1.0,
        steps,
        device=raw_value.device,
        dtype=raw_value.dtype,
    )

    for step_index, alpha in enumerate(alphas):
        interpolated = (
            reference + alpha * (raw_value - reference)
        ).detach().requires_grad_(True)
        encoded_value = encoder(interpolated.unsqueeze(0).float())[0]
        encoded = dict(encoded_base)
        encoded[modality] = encoded_value
        output = model.forward_encoded_modalities(
            [encoded], output_labels=[[disease]]
        )
        logit = target_logit_from_output(output, disease, horizon_index)
        gradient = torch.autograd.grad(logit, interpolated, retain_graph=False)[0]
        weight = 0.5 if step_index in {0, steps - 1} else 1.0
        total_gradient += weight * gradient.detach().float()

    # Trapezoidal integration is less biased than a simple endpoint mean.
    average_gradient = total_gradient / float(steps - 1)
    attribution = (raw_value - reference).float() * average_gradient

    with torch.no_grad():
        input_encoded = dict(encoded_base)
        input_encoded[modality] = encoder(raw_value.unsqueeze(0).float())[0]
        reference_encoded = dict(encoded_base)
        reference_encoded[modality] = encoder(reference.unsqueeze(0).float())[0]
        input_logit = target_logit_from_output(
            model.forward_encoded_modalities(
                [input_encoded], output_labels=[[disease]]
            ),
            disease,
            horizon_index,
        )
        reference_logit = target_logit_from_output(
            model.forward_encoded_modalities(
                [reference_encoded], output_labels=[[disease]]
            ),
            disease,
            horizon_index,
        )
        completeness_error = (
            attribution.sum() - (input_logit - reference_logit)
        ).abs()
    return attribution, float(completeness_error.detach().cpu())
