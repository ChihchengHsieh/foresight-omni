from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F

from xai.provenance import fundus_spatial_manifest
from xai.raw_attribution import target_logit_from_output


def fundus_region_occlusion_variants(
    image: torch.Tensor,
    grid_size: int = 7,
    reference: torch.Tensor | None = None,
) -> tuple[list[dict], list[torch.Tensor]]:
    if image.ndim != 3:
        raise ValueError(f"Expected fundus image shape (C, H, W), got {image.shape}")
    if reference is None:
        reference = torch.zeros_like(image)
    if reference.shape != image.shape:
        raise ValueError("Fundus reference must match image shape")

    manifest = fundus_spatial_manifest(
        grid_size * grid_size,
        int(image.shape[-2]),
        int(image.shape[-1]),
    )
    variants = []
    for region in manifest:
        variant = image.clone()
        variant[
            :,
            region["y0"] : region["y1"],
            region["x0"] : region["x1"],
        ] = reference[
            :,
            region["y0"] : region["y1"],
            region["x0"] : region["x1"],
        ]
        variants.append(variant)
    return manifest, variants


def oct_slice_occlusion_variants(
    volume: torch.Tensor,
    reference: torch.Tensor | None = None,
    window: int = 1,
) -> tuple[list[dict], list[torch.Tensor]]:
    if volume.ndim != 4:
        raise ValueError(f"Expected OCT volume shape (S, C, H, W), got {volume.shape}")
    if window < 1:
        raise ValueError("OCT occlusion window must be >= 1")
    if reference is None:
        reference = torch.zeros_like(volume)
    if reference.shape != volume.shape:
        raise ValueError("OCT reference must match volume shape")

    manifest = []
    variants = []
    for start in range(volume.shape[0]):
        end = min(start + window, volume.shape[0])
        variant = volume.clone()
        variant[start:end] = reference[start:end]
        manifest.append(
            {
                "slice_start": start,
                "slice_end_exclusive": end,
                "feature": (
                    f"OCT B-scan {start + 1}"
                    if end == start + 1
                    else f"OCT B-scans {start + 1}-{end}"
                ),
            }
        )
        variants.append(variant)
    return manifest, variants


def oct_within_slice_region_variants(
    volume: torch.Tensor,
    slice_indices: Sequence[int],
    grid_size: int = 7,
    reference: torch.Tensor | None = None,
) -> tuple[list[dict], list[torch.Tensor]]:
    """Occlude spatial cells within selected OCT B-scans."""
    if volume.ndim != 4:
        raise ValueError(f"Expected OCT volume shape (S, C, H, W), got {volume.shape}")
    if reference is None:
        reference = torch.zeros_like(volume)
    if reference.shape != volume.shape:
        raise ValueError("OCT reference must match volume shape")

    height, width = int(volume.shape[-2]), int(volume.shape[-1])
    manifest = []
    variants = []
    for slice_index in slice_indices:
        slice_index = int(slice_index)
        if not 0 <= slice_index < volume.shape[0]:
            raise IndexError(f"OCT slice index out of range: {slice_index}")
        for row in range(grid_size):
            for column in range(grid_size):
                y0 = round(row * height / grid_size)
                y1 = round((row + 1) * height / grid_size)
                x0 = round(column * width / grid_size)
                x1 = round((column + 1) * width / grid_size)
                variant = volume.clone()
                variant[slice_index, :, y0:y1, x0:x1] = reference[
                    slice_index, :, y0:y1, x0:x1
                ]
                manifest.append(
                    {
                        "slice_index": slice_index,
                        "row": row,
                        "column": column,
                        "y0": y0,
                        "y1": y1,
                        "x0": x0,
                        "x1": x1,
                        "feature": (
                            f"OCT B-scan {slice_index + 1} spatial cell "
                            f"r{row + 1}c{column + 1}"
                        ),
                    }
                )
                variants.append(variant)
    return manifest, variants


def grad_cam_for_module(
    model,
    sample: dict[str, torch.Tensor],
    disease: str,
    horizon_index: int,
    target_module,
    output_size: Sequence[int] | None = None,
) -> torch.Tensor:
    """Compute a target-specific Grad-CAM map for a convolutional module."""
    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        captured["activations"] = output

    def backward_hook(_module, _grad_input, grad_output):
        captured["gradients"] = grad_output[0]

    forward_handle = target_module.register_forward_hook(forward_hook)
    backward_handle = target_module.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        output = model([sample], output_labels=[[disease]])
        logit = target_logit_from_output(output, disease, horizon_index)
        logit.backward()
        activations = captured["activations"]
        gradients = captured["gradients"]
        if activations.ndim != 4 or gradients.ndim != 4:
            raise ValueError(
                "Grad-CAM target must produce a four-dimensional CNN feature map"
            )
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        heatmap = (weights * activations).sum(dim=1, keepdim=True)
        heatmap = torch.relu(heatmap)
        if output_size is not None:
            heatmap = F.interpolate(
                heatmap,
                size=tuple(output_size),
                mode="bilinear",
                align_corners=False,
            )
        heatmap = heatmap[0, 0]
        maximum = heatmap.max()
        if maximum > 0:
            heatmap = heatmap / maximum
        return heatmap.detach()
    finally:
        forward_handle.remove()
        backward_handle.remove()
