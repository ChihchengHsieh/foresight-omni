"""Stage 2 parameter freezing, optimizer grouping, and scheduling."""

import math

import torch
from transformers import get_cosine_schedule_with_warmup


STAGE2_BACKBONE_KEYWORDS = ("input_to_seq.",)


def stage2_epoch_range(current_epoch, total_epochs, *, eval_only_checkpoint=False):
    """Return no training epochs for checkpoint-only evaluation."""
    if eval_only_checkpoint:
        return range(0)
    return range(current_epoch, total_epochs + 1)


def stage2_backbone_modality(param_name):
    """Return the modality owning an ``input_to_seq`` parameter, if any."""
    marker = "input_to_seq."
    if marker not in param_name:
        return None
    suffix = param_name.split(marker, 1)[1]
    return suffix.split(".", 1)[0] or None


def is_stage2_backbone_param(param_name, modalities=None):
    """Return whether a parameter belongs to a pretrained modality encoder.

    The shared transformer, modality containers, output tokens, and prediction
    heads are fusion-specific parameters. They must remain trainable from the
    start when independently pretrained ``input_to_seq`` branches are loaded.
    """
    modality = stage2_backbone_modality(param_name)
    if modality is None:
        return False
    return modalities is None or modality in set(modalities)


def set_backbone_requires_grad(
    model,
    requires_grad,
    *,
    modalities=None,
    exclude_modalities=(),
    name_patterns=(),
):
    """Toggle Stage 2 backbone gradients and return matched/changed counts."""
    excluded = set(exclude_modalities)
    patterns = tuple(str(pattern) for pattern in name_patterns if str(pattern))
    matched = 0
    changed = 0
    for name, parameter in model.named_parameters():
        modality = stage2_backbone_modality(name)
        if (
            modality is not None
            and (modalities is None or modality in set(modalities))
            and modality not in excluded
            and (not patterns or any(pattern in name for pattern in patterns))
        ):
            matched += 1
            if parameter.requires_grad != requires_grad:
                parameter.requires_grad = requires_grad
                changed += 1
    return matched, changed


def build_stage2_optimizer(
    model,
    backbone_lr,
    head_lr,
    weight_decay,
    *,
    frozen_backbone_modalities=(),
    head_backbone_modalities=(),
    backbone_param_patterns=(),
    optimizer_type="adamw",
):
    """Build differential-learning-rate AdamW groups for Stage 2."""
    frozen_modalities = set(frozen_backbone_modalities)
    head_modalities = set(head_backbone_modalities)
    backbone_patterns = tuple(
        str(pattern) for pattern in backbone_param_patterns if str(pattern)
    )
    backbone_params = []
    head_params = []
    backbone_names = []
    head_names = []

    for name, parameter in model.named_parameters():
        modality = stage2_backbone_modality(name)
        if modality in frozen_modalities:
            continue
        matches_backbone_pattern = (
            not backbone_patterns
            or any(pattern in name for pattern in backbone_patterns)
        )
        if (
            is_stage2_backbone_param(name)
            and modality not in head_modalities
            and matches_backbone_pattern
        ):
            backbone_params.append(parameter)
            backbone_names.append(name)
        else:
            head_params.append(parameter)
            head_names.append(name)

    param_groups = []
    if backbone_params:
        param_groups.append(
            {
                "params": backbone_params,
                "lr": backbone_lr,
                "weight_decay": weight_decay,
                "group_name": "stage2_backbone",
            }
        )
    if head_params:
        param_groups.append(
            {
                "params": head_params,
                "lr": head_lr,
                "weight_decay": weight_decay,
                "group_name": "stage2_head",
            }
        )

    optimizer_type = str(optimizer_type).lower()
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(param_groups)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(param_groups)
    else:
        raise ValueError(f"Unsupported Stage 2 optimizer: {optimizer_type}")
    return optimizer, backbone_names, head_names


def build_stage2_scheduler(
    scheduler_type,
    optimizer,
    steps_per_epoch,
    start_epoch,
    total_epochs,
    warmup_ratio,
    *,
    delayed_backbone_schedule=False,
    backbone_freeze_epochs=0,
    backbone_warmup_epochs=1,
):
    """Build the Stage 2 schedule over epochs remaining in this invocation."""
    if scheduler_type == "none":
        return None

    remaining_epochs = total_epochs - start_epoch + 1
    if remaining_epochs <= 0:
        raise ValueError(
            "Stage 2 has no remaining epochs: "
            f"start={start_epoch}, total={total_epochs}"
        )

    steps_per_epoch = math.ceil(steps_per_epoch)
    training_steps = remaining_epochs * steps_per_epoch
    warmup_steps = int(warmup_ratio * training_steps)
    if delayed_backbone_schedule:
        frozen_epochs_remaining = min(
            remaining_epochs,
            max(0, int(backbone_freeze_epochs) - int(start_epoch) + 1),
        )
        backbone_freeze_steps = frozen_epochs_remaining * steps_per_epoch
        backbone_active_steps = training_steps - backbone_freeze_steps
        backbone_warmup_steps = min(
            max(0, int(backbone_warmup_epochs)) * steps_per_epoch,
            backbone_active_steps,
        )

        def cosine_factor(
            current_step,
            *,
            delay_steps=0,
            local_warmup_steps=0,
            warmup_step_offset=0,
        ):
            if current_step < delay_steps:
                return 0.0
            local_step = current_step - delay_steps
            local_training_steps = training_steps - delay_steps
            if local_training_steps <= 0:
                return 0.0
            if local_step < local_warmup_steps:
                return float(local_step + warmup_step_offset) / float(
                    max(1, local_warmup_steps)
                )
            progress = float(local_step - local_warmup_steps) / float(
                max(1, local_training_steps - local_warmup_steps)
            )
            progress = min(1.0, max(0.0, progress))
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        lr_lambdas = []
        for group in optimizer.param_groups:
            if group.get("group_name") == "stage2_backbone":
                lr_lambdas.append(
                    lambda step, delay=backbone_freeze_steps, warmup=backbone_warmup_steps: cosine_factor(
                        step,
                        delay_steps=delay,
                        local_warmup_steps=warmup,
                        warmup_step_offset=1,
                    )
                )
            else:
                lr_lambdas.append(
                    lambda step, warmup=warmup_steps: cosine_factor(
                        step,
                        local_warmup_steps=warmup,
                    )
                )
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambdas)

    return get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=training_steps,
        num_cycles=0.5,
    )
