"""Compatibility helpers for loading checkpoints across model revisions."""

from __future__ import annotations

from collections.abc import Iterable


def parse_component_pretrained_spec(spec: str) -> tuple[list[str], str]:
    """Parse ``modality+modality=/checkpoint/path`` component specifications."""
    if "=" not in spec:
        raise ValueError(
            "Component checkpoint must use 'modality[+modality]=/checkpoint/path': "
            f"{spec!r}"
        )
    raw_modalities, raw_path = spec.split("=", 1)
    modalities = [item.strip() for item in raw_modalities.split("+") if item.strip()]
    path = raw_path.strip()
    if not modalities or not path:
        raise ValueError(
            "Component checkpoint must include modalities and a path: " f"{spec!r}"
        )
    return modalities, path


def load_pretrained_components(
    model,
    specs: Iterable[str],
    *,
    device,
    checkpoint_loader,
    logger,
) -> dict[str, int]:
    """Load only named ``input_to_seq`` branches from independent checkpoints.

    The shared fusion transformer and output heads deliberately remain at their
    new-run initialization. Shape mismatches are fatal because silently mixing
    incompatible tokenizer definitions would invalidate a pretrained-encoder run.
    """
    model_state = model.state_dict()
    claimed_modalities: set[str] = set()
    loaded_tensors = 0

    for spec in specs:
        modalities, checkpoint_path = parse_component_pretrained_spec(spec)
        overlap = claimed_modalities.intersection(modalities)
        if overlap:
            raise ValueError(
                "A modality may be initialized from only one component checkpoint; "
                f"duplicated: {sorted(overlap)}"
            )

        checkpoint = checkpoint_loader(checkpoint_path, device)
        source_state = checkpoint["model"]
        loaded_for_spec = 0

        for modality in modalities:
            prefix = f"input_to_seq.{modality}."
            source_keys = [key for key in source_state if key.startswith(prefix)]
            if not source_keys:
                raise ValueError(
                    f"Checkpoint {checkpoint_path!r} has no encoder branch {prefix!r}"
                )

            for key in source_keys:
                if key not in model_state:
                    raise ValueError(
                        f"Target full model has no parameter for pretrained key {key!r}"
                    )
                source_tensor = source_state[key]
                target_tensor = model_state[key]
                if target_tensor.shape != source_tensor.shape:
                    raise ValueError(
                        f"Shape mismatch for {key}: checkpoint "
                        f"{tuple(source_tensor.shape)} vs full model "
                        f"{tuple(target_tensor.shape)}"
                    )
                target_tensor.copy_(
                    source_tensor.to(
                        device=target_tensor.device, dtype=target_tensor.dtype
                    )
                )
                loaded_for_spec += 1

        claimed_modalities.update(modalities)
        loaded_tensors += loaded_for_spec
        logger.info(
            "Loaded pretrained encoder components %s from %s (%d tensors)",
            modalities,
            checkpoint_path,
            loaded_for_spec,
        )

    model.load_state_dict(model_state, strict=True)
    return {
        "component_checkpoints": len(list(specs)),
        "modalities": len(claimed_modalities),
        "tensors": loaded_tensors,
    }


def inflate_checkpoint_tensor(target, source):
    """Copy a source tensor into the overlapping block of a target-shaped tensor."""
    if target.ndim != source.ndim:
        return None

    inflated = target.clone()
    slices = tuple(slice(0, min(t, s)) for t, s in zip(target.shape, source.shape))
    inflated[slices].copy_(source[slices].to(device=target.device, dtype=target.dtype))
    return inflated
