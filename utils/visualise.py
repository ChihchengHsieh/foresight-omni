import re
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Optional
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import transforms
from matplotlib.ticker import FixedLocator, FixedFormatter  # put at top of file
from matplotlib import transforms
from textwrap import shorten, fill
from matplotlib import transforms


def _denorm_img(img, mean_std=None):
    """
    img: torch.Tensor (C,H,W) in model space. Returns numpy (H,W,3) in [0,1].
    mean_std: (mean, std) or None
    """
    x = img.detach().float().cpu()
    if mean_std is not None:
        mean, std = mean_std
        mean = torch.as_tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
        std = torch.as_tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
        x = x * std + mean
    # clamp and to numpy HWC
    x = x.clamp(0, 1)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    return x.permute(1, 2, 0).cpu().numpy()


@torch.no_grad()
def _normalize_map(m):
    m = m - m.min()
    mx = m.max()
    if mx > 0:
        m = m / mx
    return m


def _overlay_heatmap(base_img, heat, alpha=0.6, cmap="turbo"):
    """
    base_img: numpy (H,W,3) in [0,1]
    heat: numpy (H,W) in [0,1]
    returns a matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=200)
    ax.imshow(base_img)
    ax.imshow(heat, cmap=cmap, alpha=alpha, interpolation="bilinear")
    ax.axis("off")
    return fig


import math
import torch
import torch.nn.functional as F


def _fundus_gradcam_for_label_from_tokens(
    model,
    sample_dict,
    label_key: str,
    *,
    fundus_key: str = "fundus_image",
    image_size: int = 224,
    mean_std: tuple | None = None,
    debug: bool = False,
):
    """
    Token-level Grad-CAM for fundus encoder output tokens.
    Works for CNN tokenisation (e.g., 7x7=49 tokens) and ViT (e.g., 14x14=196 tokens),
    as long as tokens correspond to a square grid (H*W tokens).
    """

    assert (
        fundus_key in model.input_to_seq
    ), f"{fundus_key} not found in model.input_to_seq"

    cache = {"tokens": None, "fired": False}

    def _hook(_m, _inp, out):
        cache["fired"] = True
        tok = out[0] if isinstance(out, (tuple, list)) else out
        cache["tokens"] = tok
        if debug:
            print(
                "[HOOK] tok.shape:",
                tuple(tok.shape),
                "requires_grad:",
                tok.requires_grad,
                "grad_fn:",
                tok.grad_fn,
            )

    h = model.input_to_seq[fundus_key].register_forward_hook(_hook)

    model.zero_grad(set_to_none=True)
    model.eval()

    with torch.enable_grad():
        outs = model(
            [sample_dict], output_labels=[[label_key]], need_attn_weights=False
        )
        logit = outs["out"][0][label_key].mean()

        h.remove()

        tokens = cache["tokens"]
        if debug:
            print("[DEBUG] grad_enabled:", torch.is_grad_enabled())
            print("[DEBUG] hook fired:", cache["fired"])
            print(
                "[DEBUG] logit.requires_grad:",
                logit.requires_grad,
                "grad_fn:",
                logit.grad_fn,
            )

        if tokens is None:
            raise RuntimeError("Fundus hook did not fire / tokens is None")

        grads = torch.autograd.grad(
            logit, tokens, retain_graph=False, create_graph=False, allow_unused=True
        )[0]

        if grads is None:
            raise RuntimeError(
                "Grad-CAM failed: grads is None. "
                "=> logit 沒有依賴 tokens（可能 forward 有 detach / no_grad / gating skip）。"
            )

        # tokens/grads: (B, L, D) or (B, 1, D) etc.
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
            grads = grads.unsqueeze(0)

        B, L, D = tokens.shape

        # token saliency: (B, L)
        sal = F.relu((tokens * grads).sum(-1))
        sal0 = sal[0]  # (L,)

        # 推 grid
        side = int(math.isqrt(L))
        if side * side != L:
            raise RuntimeError(
                f"Token count L={L} is not a perfect square (cannot reshape to HxW)."
            )

        heat_small = sal0.reshape(side, side).detach().float()
        heat_small = (heat_small - heat_small.min()) / (
            heat_small.max() - heat_small.min() + 1e-8
        )

        heat = (
            F.interpolate(
                heat_small[None, None],
                size=(image_size, image_size),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            .cpu()
            .numpy()
        )

    # base image for overlay
    img = sample_dict[fundus_key]
    if img.dim() == 4:
        img = img[0]
    base_img = _denorm_img(img, mean_std)

    return heat, base_img, outs["out"][0]


def get_fundus_target_layer(model, fundus_key="fundus_image"):
    # 你的 model.input_to_seq[fundus_key][0] 是 ConvTokenisation
    tok = model.input_to_seq[fundus_key][0]
    fe = tok.feature_extractor  # Sequential(...) like ResNet stem + layers

    # 依你印出的結構，layer4 在 index 7
    # 最後一個 BasicBlock 通常是 fe[7][-1]
    return fe[7][-1]  # BasicBlock


def gradcam_fundus_conv(
    model,
    sample_dict,
    label_key,
    *,
    fundus_key="fundus_image",
    mean_std=None,
    target_layer=None,
    eps=1e-8,
    debug=False,
):
    model.eval()
    device = next(model.parameters()).device

    if target_layer is None:
        target_layer = get_fundus_target_layer(model, fundus_key)

    cache = {"act": None, "grad": None, "fired_fwd": False, "fired_bwd": False}

    def fwd_hook(m, inp, out):
        cache["fired_fwd"] = True
        cache["act"] = out  # (B,C,H,W)
        if debug:
            print("[fwd] act:", tuple(out.shape), "requires_grad:", out.requires_grad)

    def bwd_hook(m, grad_in, grad_out):
        cache["fired_bwd"] = True
        cache["grad"] = grad_out[0]  # (B,C,H,W)
        if debug:
            g = grad_out[0]
            print("[bwd] grad:", tuple(g.shape), "mean:", g.abs().mean().item())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    # forward
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        outs = model(
            [sample_dict], output_labels=[[label_key]], need_attn_weights=False
        )
        logit = outs["out"][0][label_key].mean()
        if debug:
            print("[logit] req_grad:", logit.requires_grad, "grad_fn:", logit.grad_fn)

        logit.backward()

    h1.remove()
    h2.remove()

    if not cache["fired_fwd"]:
        raise RuntimeError("Forward hook 沒觸發：target_layer 不在 forward path 上。")
    if cache["grad"] is None:
        raise RuntimeError("Backward hook 沒拿到 grad：可能被 no_grad / detach 了。")

    act = cache["act"]  # (1,C,H,W)
    grad = cache["grad"]  # (1,C,H,W)

    # Grad-CAM: weights = GAP over spatial dims
    weights = grad.mean(dim=(2, 3), keepdim=True)  # (1,C,1,1)
    cam = (weights * act).sum(dim=1, keepdim=True)  # (1,1,H,W)
    cam = F.relu(cam)

    # normalize to [0,1]
    cam = cam - cam.min()
    cam = cam / (cam.max() + eps)

    # upsample to input image size (必要步驟，否則無法 overlay)
    img = sample_dict[fundus_key]
    if img.dim() == 4:
        img = img[0]
    H, W = img.shape[-2], img.shape[-1]
    cam_up = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
    heat = cam_up.detach().cpu().numpy()

    # base img for overlay
    base_img = _denorm_img(img, mean_std)  # 你原本的 function

    return heat, base_img, outs["out"][0]


def _fundus_gradcam_for_label(
    model,
    sample_dict,  # one sample: samples[inst_idx]
    label_key: str,
    *,
    fundus_key: str = "fundus_image",
    image_size: int = 224,
    patch_size: int = 16,
    mean_std: tuple | None = None,  # (mean,std) used at input normalization
    debug=True,
):
    """
    Returns (heatmap(H,W) in [0,1], base_img(H,W,3) in [0,1]).
    Uses Grad-CAM on the output of model.input_to_seq[fundus_key] (token embeddings).
    Score per token = ReLU( (A * dA).sum(-1) ). Reshaped to (H_patches, W_patches) then upsampled.
    """
    assert (
        fundus_key in model.input_to_seq
    ), f"{fundus_key} not found in model.input_to_seq"

    cache = {"tokens": None, "fired": False}

    def _hook(_m, _inp, out):
        cache["fired"] = True

        tok = out
        # out 可能是 (B, L, D) 或 (B, 1, D) 或 (B, D)
        if isinstance(tok, (tuple, list)):
            tok = tok[0]

        # ✅ 不要在這裡 tok = tok[0] 先把 batch 拿掉
        #    先保留完整 (B, L, D)，等下算 grad 再切
        cache["tokens"] = tok

        if debug:
            print("\n[HOOK] fired")
            print("[HOOK] tok.shape:", tuple(tok.shape))
            print("[HOOK] tok.requires_grad:", tok.requires_grad)
            print("[HOOK] tok.grad_fn:", tok.grad_fn)

        # retain_grad 只有在 tok.requires_grad=True 且有 graph 才有效
        try:
            tok.retain_grad()
        except Exception as e:
            if debug:
                print("[HOOK] retain_grad failed:", repr(e))

    handle = model.input_to_seq[fundus_key].register_forward_hook(_hook)

    batch_1 = [sample_dict]
    model.zero_grad(set_to_none=True)
    model.eval()

    # --- CRITICAL: force gradients enabled (overrides any outer no_grad/inference_mode) ---
    with torch.enable_grad():
        outs = model(batch_1, output_labels=[[label_key]], need_attn_weights=False)
        logit = outs["out"][0][label_key].mean()

        tokens = cache["tokens"]
        handle.remove()

        if debug:
            print("\n[DEBUG] grad_enabled:", torch.is_grad_enabled())
            print("[DEBUG] logit.requires_grad:", logit.requires_grad)
            print("[DEBUG] logit.grad_fn:", logit.grad_fn)
            print("[DEBUG] hook fired:", cache["fired"])

        assert (
            tokens is not None
        ), "Fundus encoder hook did not fire. Check that sample has the key and the encoder is called."

        # Option A (recommended): explicit autograd.grad (more robust than tokens.grad)
        grads = torch.autograd.grad(
            logit,
            tokens,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]

        if debug:
            print("[DEBUG] grads is None:", grads is None)
            if grads is not None:
                print("[DEBUG] grads.shape:", tuple(grads.shape))
                print(
                    "[DEBUG] grads |mean|:",
                    grads.abs().mean().item(),
                    " |max|:",
                    grads.abs().max().item(),
                )

        if grads is None:
            # ✅ 第二層診斷：看看 fundus encoder 參數有沒有梯度（能快速判斷是否整條路徑 detach）
            any_param = next(iter(model.input_to_seq[fundus_key].parameters()), None)
            if any_param is not None:
                g2 = torch.autograd.grad(logit, any_param, allow_unused=True)[0]
                if debug:
                    print("[DEBUG] grad wrt fundus encoder param is None:", g2 is None)
            raise RuntimeError(
                "Grad-CAM failed: grads is None.\n"
                "=> 代表 logit 與 fundus tokens 沒連上。\n"
                "請看上面的 [DEBUG] logit.grad_fn / tok.grad_fn，通常是 forward 裡有 detach/no_grad/inference_mode 或 tokens 不是同一次 forward 的輸出。"
            )

    # Grad-CAM score per token
    sal_tok = F.relu((tokens * grads).sum(-1))  # (L,)

    # reshape to patch grid
    H_p = W_p = image_size // patch_size
    if sal_tok.numel() != H_p * W_p:
        sal_core = sal_tok[-(H_p * W_p) :]
    else:
        sal_core = sal_tok

    heat_small = sal_core.reshape(H_p, W_p).detach().cpu().numpy()
    heat_small = _normalize_map(heat_small)

    # upsample to image size
    heat = torch.tensor(heat_small)[None, None]
    heat = (
        F.interpolate(
            heat, size=(image_size, image_size), mode="bilinear", align_corners=False
        )[0, 0]
        .cpu()
        .numpy()
    )
    heat = _normalize_map(heat)

    # get the original image for overlay
    img = sample_dict[fundus_key]
    if img.dim() == 4:
        img = img[0]
    base_img = _denorm_img(img, mean_std)

    return heat, base_img, outs["out"][0]


def _std_attn_stack(attn_weights):
    """
    attn_weights can be:
      - list of length n_layers, each (B, H, L, L) or (B, L, L)
      - tensor (n_layers, B, H, L, L) or (n_layers, B, L, L)
    Returns tensor (n_layers, B, H, L, L)
    """
    if isinstance(attn_weights, list):
        A = torch.stack(attn_weights, dim=0)
    else:
        A = attn_weights
    if A.dim() == 4:  # (n_layers,B,L,L) -> add head dim
        A = A.unsqueeze(2)
    return A  # (n_layers,B,H,L,L)


@torch.no_grad()
def _rollout_from_stack(stack, head_fusion="mean", add_identity=False, eps=1e-6):
    """
    stack: (n_layers, B, H, L, L) (already label-conditioned if using gradients)
    Returns: (B, L, L)
    """
    nL, B, H, L, _ = stack.shape
    # fuse heads
    if head_fusion == "max":
        A = stack.max(dim=2).values
    else:
        A = stack.mean(dim=2)
    # row-normalize
    A = A / (A.sum(dim=-1, keepdim=True) + eps)
    if add_identity:
        I = torch.eye(L, device=A.device).expand(B, L, L)
        A = (A + I.unsqueeze(0)) / 2
    # chain
    R = A[0]
    for l in range(1, nL):
        R = R @ A[l]
    return R  # (B,L,L)


def attention_x_gradient_rollout(
    model,
    batch,  # one-instance batch: [samples[inst_idx]]
    label_key: str,  # e.g. "has_glaucoma_in_3_years"
    *,
    head_fusion: str = "max",  # "mean" or "max"
    add_identity: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns R_label: (B, L, L) : label-conditioned rollout using A ⊙ ReLU(dA).
    Strongly normalized so each intermediate and final matrix is row-stochastic.
    """

    model.eval()

    # ---- 1) forward with attention PROBS that keep grad ----
    # IMPORTANT: your transformer must *not* .detach() attn probs when need_attn_weights=True
    outs = model(batch, output_labels=[[label_key]], need_attn_weights=True)

    A_raw = outs[
        "attn_weights"
    ]  # list[tensor(B,H,L,L)] or tensor(nL,B,H,L,L) or (nL,B,L,L)

    # standardize to (nL, B, H, L, L)
    if isinstance(A_raw, list):
        A_stack = torch.stack(A_raw, dim=0)
    else:
        A_stack = A_raw
    if A_stack.dim() == 4:
        A_stack = A_stack.unsqueeze(2)  # add head dim
    nL, B, H, L, _ = A_stack.shape

    # make sure grads will flow
    if isinstance(A_raw, list):
        for t in A_raw:
            t.requires_grad_(True)
            t.retain_grad()
    else:
        A_stack.requires_grad_(True)
        A_stack.retain_grad()

    # ---- 2) backprop from chosen label logit ----
    logit = outs["out"][0][label_key].mean()
    model.zero_grad(set_to_none=True)
    logit.backward(retain_graph=False)

    # ---- 3) saliency weighting + per-head row normalization ----
    if isinstance(A_raw, list):
        sal_layers = []
        for t in A_raw:
            S = t * torch.relu(t.grad)  # (B,H,L,L), nonnegative
            # row-normalize per head
            S = S / (S.sum(dim=-1, keepdim=True) + eps)
            sal_layers.append(S)
        sal = torch.stack(sal_layers, dim=0)  # (nL,B,H,L,L)
    else:
        S = A_stack * torch.relu(A_stack.grad)  # (nL,B,H,L,L)
        S = S / (S.sum(dim=-1, keepdim=True) + eps)
        sal = S

    # ---- 4) fuse heads, row-normalize again ----
    if head_fusion == "max":
        A_f = sal.max(dim=2).values  # (nL,B,L,L)
    else:
        A_f = sal.mean(dim=2)  # (nL,B,L,L)

    A_f = A_f / (A_f.sum(dim=-1, keepdim=True) + eps)

    # optional identity mixing (some like it off for attribution)
    if add_identity:
        I = torch.eye(L, device=A_f.device).view(1, 1, L, L).expand(nL, B, L, L)
        A_f = (A_f + I) / 2.0
        A_f = A_f / (A_f.sum(dim=-1, keepdim=True) + eps)

    # ---- 5) rollout (matrix product across layers) ----
    R = A_f[0]  # (B,L,L)
    for l in range(1, nL):
        R = R @ A_f[l]  # chain row-stochastic matrices

    # final row normalization (safety) -> rows sum to 1
    R = R / (R.sum(dim=-1, keepdim=True) + eps)

    # nonnegativity (numerical guard)
    R = torch.clamp(R, min=0)

    return R


# --- capture per-layer attention tensors and their grads ---
class AttnGrabber:
    def __init__(self):
        self.A = []
        self.handles = []

    def hook(self, module, inp, out):
        # out is attention probs of shape (B, H, L, L)
        out.retain_grad()
        self.A.append(out)

    def attach(self, transformer):
        # register on modules that output attention probs
        # adjust the module type/name to match your transformer
        for m in transformer.modules():
            if hasattr(m, "attn_drop"):  # heuristic: inside attention block
                h = m.register_forward_hook(self.hook)
                self.handles.append(h)
        return self

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def attention_rollout_from_stack(
    stack, head_fusion="mean", add_identity=False, eps=1e-6
):
    # stack: (n_layers, B, H, L, L) or (n_layers, B, L, L)
    A = stack
    if A.dim() == 4:  # no heads
        A = A.unsqueeze(2)
    nL, B, H, L, _ = A.shape
    # fuse heads
    if head_fusion == "max":
        A = A.max(dim=2).values
    else:
        A = A.mean(dim=2)
    # normalize rows to sum to 1 (stability)
    A = A / (A.sum(dim=-1, keepdim=True) + eps)
    if add_identity:
        I = torch.eye(L, device=A.device).unsqueeze(0).unsqueeze(0).expand(nL, B, L, L)
        A = (A + I) / 2
    # chain across layers
    R = A[0]
    for l in range(1, nL):
        R = R @ A[l]
    return R  # (B, L, L)


# ---- find the fundus segment ----
def get_segment(modality_segments, name="fundus_image"):
    for seg in modality_segments:
        if seg["name"] == name:
            return seg
    raise KeyError(
        f"Modality '{name}' not found in segments: {[m['name'] for m in modality_segments]}"
    )


# ---- roll out → label-conditioned vector (L,) ----
@torch.no_grad()
def label_conditioned_vector(R_single: torch.Tensor, label_spans, reduce="mean"):
    """
    R_single: (L, L) attention rollout for ONE instance
    label_spans: list of (start, end) ranges for selected labels
    returns v: (L,) attention FROM the selected labels TO all tokens
    """
    if (
        isinstance(label_spans, (tuple, list))
        and len(label_spans) == 2
        and isinstance(label_spans[0], int)
    ):
        label_spans = [label_spans]  # allow single span
    rows = [R_single[s:e, :] for (s, e) in label_spans]
    if not rows:
        return torch.zeros(R_single.shape[1], device=R_single.device)
    M = torch.cat(rows, dim=0)  # (n_label_tokens, L)
    if reduce == "mean":
        return M.mean(dim=0)
    if reduce == "sum":
        return M.sum(dim=0)
    if reduce == "max":
        return M.max(dim=0).values
    raise ValueError("reduce must be 'mean'|'sum'|'max'")


# ---- vector slice → 2D patch grid ----
def vector_to_patch_grid(v_slice: torch.Tensor, grid_hw=None):
    """
    v_slice: (N,) attention for exactly the fundus tokens (no brackets).
    grid_hw: (H_p, W_p). If None, tries to infer a square grid.
    Returns: (H_p, W_p) numpy array normalized to [0,1] (not colored).
    """
    n = int(v_slice.numel())
    if grid_hw is None:
        g = int(round(math.sqrt(n)))
        if g * g != n:
            raise ValueError(
                f"Can't infer square grid from {n} tokens; pass grid_hw=(H_p, W_p)."
            )
        H_p, W_p = g, g
    else:
        H_p, W_p = grid_hw
        if H_p * W_p != n:
            raise ValueError(f"grid_hw={grid_hw} but n_tokens={n}.")
    A = v_slice.detach().float().cpu().view(H_p, W_p).numpy()
    # normalize to [0,1] for colormap
    a_min, a_max = float(A.min()), float(A.max())
    if a_max > a_min:
        A = (A - a_min) / (a_max - a_min)
    else:
        A = np.zeros_like(A)
    return A


# ---- unnormalize and format image ----
def prepare_image(img_tensor, mean=None, std=None):
    """
    img_tensor: (C,H,W) or (H,W,C) or (H,W)
    mean/std: list of 3 or 1 (values used in preprocessing), if provided.
    Returns HxWx3 uint8 for plotting.
    """
    x = img_tensor.detach().cpu().numpy()
    if x.ndim == 3 and x.shape[0] in (1, 3):  # (C,H,W)
        x = np.transpose(x, (1, 2, 0))
    if x.ndim == 2:  # grayscale -> 3 channels
        x = np.stack([x, x, x], axis=-1)

    x = x.astype(np.float32)
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, -1)
        std = np.array(std, dtype=np.float32).reshape(1, 1, -1)
        x = x * std + mean

    # robust to either 0..1 or 0..255 inputs
    if x.max() <= 1.5:
        x = np.clip(x, 0.0, 1.0) * 255.0
    x = np.clip(x, 0, 255).astype(np.uint8)
    if x.shape[2] == 1:
        x = np.repeat(x, 3, axis=2)
    return x


# ---- overlay heatmap on image ----
def overlay_attention_on_image(
    img_uint8,
    heat_2d_01,
    alpha=0.45,
    cmap="magma",
    title=None,
    show=True,
    save_path=None,
):
    """
    img_uint8: HxWx3 uint8
    heat_2d_01: HxW float in [0,1]
    """
    H, W, _ = img_uint8.shape
    fig = plt.figure(figsize=(W / 50, H / 50), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img_uint8)
    ax.imshow(heat_2d_01, cmap=cmap, alpha=alpha, interpolation="bilinear")  # smooth
    ax.axis("off")
    # if title: ax.set_title(title, y=0.98, fontsize=11)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.0)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_fundus_attention_overlay(
    viz,
    inst_idx: int,
    samples,  # your original batch list; we read the image from here
    *,
    label_key: str | None = None,  # if None → uses ALL label tokens (mean)
    reduce: str = "mean",
    fundus_key: str = "fundus_image",
    image_index_in_batch: int = 0,  # which sample's image to use (matches inst_idx)
    image_mean_std: (
        tuple | None
    ) = None,  # e.g., (mean, std) used in preprocessing, or None
    image_size: int = 224,
    patch_size: int = 16,
    alpha: float = 0.45,
    cmap: str = "magma",
    title: str | None = None,
    save_path: str | None = None,
    exclude_brackets: bool = True,
    show: bool = True,
):
    """
    Uses viz['rollout'][inst_idx], viz['label_start_end'], and the fundus span to create a heatmap overlay.
    """
    R = viz["rollout"][inst_idx]  # (L, L)
    label_dict = viz["label_start_end"][inst_idx]
    modality_segments, _ = build_segments_from_viz(
        viz, inst_idx=inst_idx, exclude_brackets=exclude_brackets
    )

    # fundus segment and grid size
    fundus_seg = get_segment(modality_segments, fundus_key)
    H_p = W_p = image_size // patch_size

    # choose label spans
    if label_key is None:
        # use ALL label tokens
        spans = list(label_dict.values())
        pretty_label = "all_labels"
    else:
        spans = [label_dict[label_key]]
        pretty_label = label_key.replace("_", " ")

    # label-conditioned vector and slice for fundus tokens
    v = label_conditioned_vector(R, spans, reduce=reduce)  # (L,)
    s, e = int(fundus_seg["start"]), int(fundus_seg["end"])
    v_fundus = v[s:e]  # (H_p*W_p,)

    # reshape to grid & upsample to image size
    heat_grid = vector_to_patch_grid(
        v_fundus, grid_hw=(H_p, W_p)
    )  # (H_p, W_p) in [0,1]
    heat_ups = torch.tensor(heat_grid).unsqueeze(0).unsqueeze(0)  # (1,1,H_p,W_p)
    heat_ups = torch.nn.functional.interpolate(
        heat_ups, size=(image_size, image_size), mode="bilinear", align_corners=False
    )
    heat_ups = heat_ups.squeeze().cpu().numpy()  # (H,W) in [0,1]

    # get the image from your samples
    # Expecting samples[inst_idx]['fundus_image'] to be (C,H,W) or (H,W) torch tensor
    img_t = samples[image_index_in_batch][fundus_key]
    mean, std = image_mean_std if image_mean_std is not None else (None, None)
    img = prepare_image(img_t, mean=mean, std=std)

    # plot / save
    if title is None:
        title = f"{pretty_label} → {fundus_key}"
    return overlay_attention_on_image(
        img,
        heat_ups,
        alpha=alpha,
        cmap=cmap,
        title=title,
        save_path=save_path,
        show=show,
    )


def _autosize_from_span_len(span_len: int, *, min_w=6.5, per_col_w=0.22, height=3.8):
    # width scales with the number of columns in the modality
    return (max(min_w, per_col_w * max(1, span_len)), height)


def plot_vector_to_modality_row(
    v,  # (L,)
    modality_segment,  # {'name','start','end'}
    *,
    MODALITIES_TO_COLS=None,
    rotation: int = 90,
    dpi=300,
    show=False,
    return_fig=True,
    title=None,
):
    s, e = int(modality_segment["start"]), int(modality_segment["end"])
    name = modality_segment["name"]
    row = v[s:e].unsqueeze(0)  # (1, mod_len)

    fig = plt.figure(figsize=_autosize_from_span_len(e - s, height=5), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(row.detach().cpu().numpy(), aspect="auto", interpolation="nearest")
    fig.colorbar(im, ax=ax, label="Attention")
    ax.set_yticks([0])
    ax.set_yticklabels([name])
    ax.set_xlabel(f"{name} tokens")
    ax.set_title(f"Selected label(s) → {name}" if title is None else title)

    if MODALITIES_TO_COLS is not None and name in MODALITIES_TO_COLS:
        cols = MODALITIES_TO_COLS[name]
        n = len(cols)
        step = max(1, int(round(n / 20)))
        ticks = list(range(0, n, step))
        labels = [cols[i].replace("_", " ") for i in ticks]
        top = ax.secondary_xaxis("top")
        top.set_ticks(ticks, labels=labels)
        top.tick_params(axis="x", labelsize=7, pad=8)
        for t in top.get_xticklabels():
            t.set_rotation(rotation)
            t.set_ha("center")
            t.set_va("top")
            t.set_color("white")

    fig.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return fig


def plot_label_to_modality_heatmap(
    R_single,  # (L, L) rollout for one instance
    label_span,  # (ls, le)
    modality_segment,  # {'name','start','end'} (already bracket-excluded)
    *,
    MODALITIES_TO_COLS=None,  # optional dict for per-token names
    rotation: int = 90,
    vmax=None,
    dpi=300,
    show=False,
    return_fig=True,
    title=None,
):
    ls, le = label_span
    s, e = int(modality_segment["start"]), int(modality_segment["end"])
    name = modality_segment["name"]
    sub = R_single[ls:le, s:e]  # (label_len, mod_len)

    fig = plt.figure(figsize=_autosize_from_span_len(e - s), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        sub.detach().cpu().numpy(), aspect="auto", interpolation="nearest", vmax=vmax
    )
    fig.colorbar(im, ax=ax, label="Attention")

    ax.set_xlabel(f"{name} tokens")
    ax.set_ylabel("Label tokens")
    ax.set_title(f"Label → {name} (rollout)" if title is None else title)

    # token tick labels (optional)
    if MODALITIES_TO_COLS is not None and name in MODALITIES_TO_COLS:
        cols = MODALITIES_TO_COLS[name]
        # thin to avoid crowding
        n = len(cols)
        step = max(1, int(round(n / 20)))
        ticks = list(range(0, n, step))
        labels = [cols[i].replace("_", " ") for i in ticks]
        top = ax.secondary_xaxis("top")
        top.set_ticks(ticks, labels=labels)
        top.tick_params(axis="x", labelsize=7, pad=10)
        for t in top.get_xticklabels():
            t.set_rotation(rotation)
            t.set_ha("center")
            t.set_va("top")
            t.set_color("white")
            # t.set_va("bottom")

    fig.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return fig


def find_label_keys(
    label_dict: Dict[str, Tuple[int, int]],
    disease: str,
    horizon: Optional[int] = None,
) -> List[str]:
    """Keys look like 'has_<disease>_in_<years>_years'."""
    patt = re.compile(rf"^has_{re.escape(disease)}_in_(\d+)_years$")
    out = []
    for k in label_dict.keys():
        m = patt.match(k)
        if m is None:
            continue
        if horizon is None or int(m.group(1)) == horizon:
            out.append(k)
    return out


def aggregate_vector_to_modalities(
    v: torch.Tensor,  # (L,)
    modality_segments: List[
        Dict
    ],  # [{'name','start','end'}] (already bracket-excluded)
    mode: str = "sum",  # sum | mean | max
) -> Dict[str, float]:
    out = {}
    for m in modality_segments:
        s, e = int(m["start"]), int(m["end"])
        if e <= s:
            out[m["name"]] = 0.0
            continue
        x = v[s:e]
        out[m["name"]] = float(
            x.sum() if mode == "sum" else (x.mean() if mode == "mean" else x.max())
        )
    return out


def aggregate_vector_to_features(
    v: torch.Tensor,  # (L,)
    modality_segments: List[Dict],
    MODALITIES_TO_COLS: Dict[str, List[str]],
    *,
    target_modalities: Optional[List[str]] = None,
    mode: str = "sum",  # sum | mean | max
    on_mismatch: str = "truncate",  # truncate | warn | error
) -> Dict[str, float]:
    """Returns {feature_name: value} over selected modalities using exact per-token names."""
    name2seg = {m["name"]: (int(m["start"]), int(m["end"])) for m in modality_segments}
    feats = {}
    for mod, cols in MODALITIES_TO_COLS.items():
        if target_modalities and mod not in target_modalities:
            continue
        if mod not in name2seg:
            continue
        s, e = name2seg[mod]
        span_len = max(0, e - s)
        if span_len == 0:
            continue
        if span_len != len(cols):
            msg = f"[feature-agg] mismatch {mod}: span={span_len} cols={len(cols)}"
            if on_mismatch == "error":
                raise ValueError(msg)
            if on_mismatch == "warn":
                print("WARN:", msg, "→ truncating to min length")
        k = min(span_len, len(cols))
        x = v[s : s + k]
        for i, fname in enumerate(cols[:k]):
            feats[f"{mod}:{fname}"] = (
                float(
                    x[i].item()
                    if mode == "max"
                    else (x[i].item() if mode == "sum" else x[i].item())
                )
                if mode in ("sum", "max", "mean") and False
                else float(x[i].item())
            )
    # Note: per-feature is per-token; if you wanted mean/sum across multi-token features,
    # you'd group here. Since each column maps 1→1 to a token, we just read x[i].
    return feats


def plot_disease_to_modalities_bar(
    v: torch.Tensor,
    modality_segments: List[Dict],
    disease_label: str,
    agg: str = "sum",
    top_k: Optional[int] = None,
):
    md = aggregate_vector_to_modalities(v, modality_segments, mode=agg)
    items = sorted(md.items(), key=lambda kv: kv[1], reverse=True)
    if top_k:
        items = items[:top_k]
    mods, vals = zip(*items) if items else ([], [])
    fig = plt.figure(figsize=(max(7, 0.6 * len(items)), 3.2))
    ax = fig.add_subplot(111)
    ax.bar(mods, vals)
    ax.set_xticklabels(mods, rotation=90, ha="right")
    ax.set_ylabel(f"Attention ({agg})")
    ax.set_title(f"{disease_label} → modalities")
    fig.tight_layout()
    plt.show()


def plot_disease_to_features_bar(
    v: torch.Tensor,
    modality_segments: List[Dict],
    MODALITIES_TO_COLS: Dict[str, List[str]],
    disease_label: str,
    target_modalities: Optional[List[str]] = None,
    top_k: int = 30,
):
    feats = aggregate_vector_to_features(
        v,
        modality_segments,
        MODALITIES_TO_COLS,
        target_modalities=target_modalities,
        mode="sum",
        on_mismatch="warn",
    )
    items = sorted(feats.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    names, vals = zip(*items) if items else ([], [])
    fig = plt.figure(figsize=(10, max(3.2, 0.22 * len(items))))
    ax = fig.add_subplot(111)
    ax.barh(range(len(items)), vals)
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels([n.replace("_", " ") for n in names])
    ax.invert_yaxis()
    ax.set_xlabel("Attention (sum)")
    ax.set_title(f"{disease_label} → top-{top_k} features")
    fig.tight_layout()
    plt.show()


def export_disease_report(
    viz,
    MODALITIES_TO_COLS: Dict[str, List[str]],
    disease: str,
    horizon: Optional[int] = None,
    inst_idx: int = 0,
    pdf_path: str = "disease_attention.pdf",
    token_modalities_to_show: Optional[List[str]] = None,
):
    R = viz["rollout"][inst_idx]
    labels = viz["label_start_end"][inst_idx]
    modality_segments, _ = build_segments_from_viz(
        viz, inst_idx=inst_idx, exclude_brackets=True
    )
    label_segments = build_label_segments_from_viz(viz, inst_idx=inst_idx)

    keys = find_label_keys(labels, disease=disease, horizon=horizon)
    if not keys:
        raise KeyError(f"No labels for disease={disease}, horizon={horizon}")
    spans = [labels[k] for k in keys]
    v = label_conditioned_vector(R, spans, reduce="mean")
    pretty = f"{disease} ({'all horizons' if horizon is None else f'{horizon}y'})"

    with PdfPages(pdf_path) as pdf:
        # Heatmap (brackets + per-token annotations)
        for k in keys:
            fig = plot_token_heatmap_for_label_with_brackets(
                R,
                labels[k],
                modality_segments=modality_segments,
                label_segments=label_segments,
                highlight_label_key=k,
                MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                token_modalities_to_show=token_modalities_to_show,
                figsize=_autosize_from_modalities(modality_segments),
                dpi=160,
                show=False,
                return_fig=True,
            )
            fig.suptitle(k.replace("_", " "), y=0.985, fontsize=11)
            pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)

        # Modality bar
        md = aggregate_vector_to_modalities(v, modality_segments, mode="sum")
        items = sorted(md.items(), key=lambda kv: kv[1], reverse=True)
        mods, vals = zip(*items) if items else ([], [])
        fig = plt.figure(figsize=(max(8, 0.6 * len(items)), 3.2))
        ax = fig.add_subplot(111)
        ax.bar(mods, vals)
        ax.set_xticklabels(mods, rotation=90, ha="right")
        ax.set_ylabel("Attention (sum)")
        ax.set_title(f"{pretty} → modalities")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

        # Feature bar
        feats = aggregate_vector_to_features(
            v,
            modality_segments,
            MODALITIES_TO_COLS,
            target_modalities=token_modalities_to_show,
            mode="sum",
        )
        items = sorted(feats.items(), key=lambda kv: kv[1], reverse=True)[:30]
        names, vals = zip(*items) if items else ([], [])
        fig = plt.figure(figsize=(10, max(3.2, 0.22 * len(items))))
        ax = fig.add_subplot(111)
        ax.barh(range(len(items)), vals)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([n.replace("_", " ") for n in names])
        ax.invert_yaxis()
        ax.set_xlabel("Attention (sum)")
        ax.set_title(f"{pretty} → top features")
        fig.tight_layout()
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
        plt.close(fig)

    print(f"Saved disease report → {pdf_path}")


from matplotlib import patheffects as path_effects


# --- (A) build token centers from your modality segments and MODALITIES_TO_COLS ---
def build_token_positions_for_modalities(
    modality_segments: list[dict],
    MODALITIES_TO_COLS: dict[str, list[str]],
    *,
    on_mismatch: str = "truncate",  # "truncate" | "warn" | "error"
) -> dict[str, list[tuple[int, str]]]:
    """
    Returns: { modality_name: [(token_index, token_name), ...], ... }
    Uses the adjusted spans (i.e., after dropping bracket tokens).
    """
    name2seg = {m["name"]: (int(m["start"]), int(m["end"])) for m in modality_segments}
    out = {}

    for mod, cols in MODALITIES_TO_COLS.items():
        if mod not in name2seg:
            # not present in this instance → skip silently
            continue
        s, e = name2seg[mod]
        span_len = max(0, e - s)
        if span_len == 0:
            continue

        if span_len != len(cols):
            msg = (
                f"[token-annot] mismatch for '{mod}': span_len={span_len} "
                f"but {len(cols)} names. "
            )
            if on_mismatch == "error":
                raise ValueError(msg)
            elif on_mismatch == "warn":
                print("WARN:", msg + "Annotating min(len(span), len(cols)).")
            # fallback: align on the min length
            k = min(span_len, len(cols))
            positions = list(range(s, s + k))
            names = cols[:k]
        else:
            positions = list(range(s, e))
            names = cols

        out[mod] = list(zip(positions, names))
    return out


# --- (B) annotate tokens (vertical names) above the heatmap for selected modalities ---
def annotate_modality_tokens(
    ax,
    token_map: dict[
        str, list[tuple[int, str]]
    ],  # from build_token_positions_for_modalities
    *,
    which_modalities: list[str] | None = None,  # if None → annotate all in token_map
    max_tokens_per_mod: int = 24,  # auto-thin to at most this many labels per modality
    rotation: int = 90,
    fontsize: int = 6,
    color: str = "white",
    pad_axes_y: float = 0.06,  # how far ABOVE the axes to place the labels
    tick_len_axes: float = 0.018,  # short tick length (axes coords)
    wrap_width: int = 128,  # wrap/shorten long column names
):
    """
    Draws tiny tick marks at each token index and vertical text labels above the image.
    Uses a blended transform so x is in data coords, y is in axes coords.
    """
    if which_modalities is None:
        mods = list(token_map.keys())
    else:
        mods = [m for m in which_modalities if m in token_map]

    bt = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    y_text = 0.9
    # y_text = 1.0 + pad_axes_y
    y_tick0, y_tick1 = 1.0, 1.0 + tick_len_axes

    for mod in mods:
        pairs = token_map[mod]  # [(idx, name), ...]
        if not pairs:
            continue
        n = len(pairs)
        step = max(1, (n + max_tokens_per_mod - 1) // max_tokens_per_mod)  # thin
        sampled = pairs[::step]

        # ticks
        # for idx, _ in sampled:
        #     ax.plot(
        #         [idx + 1, idx + 1],
        #         [y_tick0, y_tick1],
        #         transform=bt,
        #         color="black",
        #         lw=0.8,
        #         clip_on=False,
        #         zorder=12,
        #     )

        # labels
        for idx, nm in sampled:
            # compact but readable: first wrap, then shorten a bit if needed
            # text = fill(nm.replace("_", " "), width=wrap_width)
            text = nm.replace("_", " ")
            # text = shorten(text, width=wrap_width + 4, placeholder="…")
            txt = ax.text(
                idx + 1,
                y_text,
                text,
                transform=bt,
                rotation=rotation,
                ha="center",
                va="top",
                fontsize=fontsize,
                color=color,
                clip_on=False,
                zorder=12,
            )
            txt.set_path_effects(
                [
                    path_effects.Stroke(linewidth=2, foreground="black"),
                    path_effects.Normal(),
                ]
            )


def _thinned_names(names, centers, max_labels=12, wrap_width=14):
    n = len(names)
    step = max(1, (n + max_labels - 1) // max_labels)
    idx = list(range(0, n, step))
    return [centers[i] + 1 for i in idx], [
        fill(names[i].replace("_", " "), width=wrap_width) for i in idx
    ]


def annotate_axis_with_input_and_label_brackets(
    ax,
    *,
    modality_segments: list[dict],  # [{'name','start','end'}]
    label_segments: list[dict],  # [{'name','start','end'}]
    highlight_label: (
        str | None
    ) = None,  # exact key to emphasize (e.g. 'has_glaucoma_in_3_years')
    show_vertical_delimiters: bool = True,
    top_max_modality_names: int = 12,
    top_wrap_width: int = 14,
    bottom_max_label_names: int = 10,
    bottom_wrap_width: int = 18,
    top_rotation: int = 90,
    top_label_pad: int = 12,
    top_fontsize: int = 8,
    bottom_rotation: int = 90,
    bottom_label_pad: int = 10,
    bottom_fontsize: int = 7,
):
    """
    Draws:
      - modality brackets at top (black rails)
      - label brackets at bottom (dark red rails)
      - optional top & bottom thinned tick labels
      - vertical delimiters at all boundaries (inputs + labels)
    """
    bt = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    # ----- Inputs (modalities) -----
    mod_bounds, mod_centers, mod_names = [], [], []
    for m in modality_segments:
        s, e = int(m["start"]), int(m["end"])
        mod_bounds.append(s)
        mod_centers.append((s + e - 1) / 2)
        mod_names.append(m["name"])
    if modality_segments:
        mod_bounds.append(int(modality_segments[-1]["end"]))

    # top rails (inside axes so they never clip)
    y_top = 0.985
    for i in range(len(mod_bounds) - 1):
        x0, x1 = mod_bounds[i], mod_bounds[i + 1] - 1
        ax.plot(
            [x0, x1],
            [y_top, y_top],
            transform=bt,
            color="black",
            lw=2.0,
            zorder=10,
            clip_on=False,
        )
        ax.plot(
            [x0, x0],
            [y_top, y_top + 0.025],
            transform=bt,
            color="black",
            lw=2.0,
            zorder=10,
            clip_on=False,
        )
        ax.plot(
            [x1, x1],
            [y_top, y_top + 0.025],
            transform=bt,
            color="black",
            lw=2.0,
            zorder=10,
            clip_on=False,
        )

    # TOP secondary x-axis (modalities) : vertical tick text
    if mod_names:
        ticks, labels = _thinned_names(
            mod_names,
            mod_centers,
            max_labels=top_max_modality_names,
            wrap_width=top_wrap_width,
        )
        top = ax.secondary_xaxis("top")
        top.xaxis.set_major_locator(FixedLocator(ticks))
        top.xaxis.set_major_formatter(FixedFormatter(labels))
        top.tick_params(axis="x", labelsize=top_fontsize, pad=top_label_pad)
        for t in top.get_xticklabels():
            t.set_rotation(top_rotation)
            t.set_ha("center")
            # vertical labels at top should sit above; 'bottom' or 'center' both fine
            t.set_va("bottom")

    # ----- Labels (output tokens) -----
    lab_bounds, lab_centers, lab_names = [], [], []
    for L in label_segments:
        s, e = int(L["start"]), int(L["end"])
        lab_bounds.append(s)
        lab_centers.append((s + e - 1) / 2)
        lab_names.append(L["name"])
    if label_segments:
        lab_bounds.append(int(label_segments[-1]["end"]))

    # bottom rails
    y_bottom = 0.015
    for i in range(len(lab_bounds) - 1):
        x0, x1 = lab_bounds[i], lab_bounds[i + 1] - 1
        # use darker red; if this span is the "current" label, make it bold
        color = "darkred"
        lw = 2.0
        if highlight_label is not None:
            # if this span’s center falls within the highlight label, boost style
            Lname = label_segments[i]["name"]
            if Lname == highlight_label:
                color, lw = "#b00020", 3.0
        ax.plot(
            [x0, x1],
            [y_bottom, y_bottom],
            transform=bt,
            color=color,
            lw=lw,
            zorder=10,
            clip_on=False,
        )
        ax.plot(
            [x0, x0],
            [y_bottom - 0.025, y_bottom],
            transform=bt,
            color=color,
            lw=lw,
            zorder=10,
            clip_on=False,
        )
        ax.plot(
            [x1, x1],
            [y_bottom - 0.025, y_bottom],
            transform=bt,
            color=color,
            lw=lw,
            zorder=10,
            clip_on=False,
        )

    # bottom (label) tick labels (thinned)
    if lab_names:
        ticks, labels = _thinned_names(
            lab_names,
            lab_centers,
            max_labels=bottom_max_label_names,
            wrap_width=bottom_wrap_width,
        )
        bot = ax.secondary_xaxis("bottom")
        bot.xaxis.set_major_locator(FixedLocator(ticks))
        bot.xaxis.set_major_formatter(FixedFormatter(labels))
        bot.tick_params(axis="x", labelsize=bottom_fontsize, pad=bottom_label_pad)
        for t in bot.get_xticklabels():
            t.set_rotation(bottom_rotation)
            t.set_ha("center")
            # vertical labels at bottom should sit below; 'top' keeps them outside the axes
            t.set_va("top")
            t.set_color("darkred")

    # ----- Vertical delimiters (inputs + labels) -----
    if show_vertical_delimiters:
        ymin, ymax = ax.get_ylim()
        for x in mod_bounds + lab_bounds:
            ax.vlines(x - 0.5, ymin, ymax, color="k", alpha=0.25, lw=0.8, zorder=5)


def build_label_segments_from_viz(viz: dict, inst_idx: int = 0):
    """Return [{'name': str, 'start': int, 'end': int}, ...] for labels."""
    label_se = viz["label_start_end"][inst_idx]
    return [{"name": k, "start": s, "end": e} for k, (s, e) in label_se.items()]


def _autosize_from_modalities(modalities, *, min_w=10.0, per_mod_w=15, min_h=10):
    """
    Compute a reasonable figsize from how many modalities are shown.
    """
    n = max(1, len(modalities))
    width = max(min_w, per_mod_w * n)  # grow width with modality count
    height = min_h  # label-rows are usually 1; keep moderate height
    return (width, height)


def _wrap_name(name: str, width: int = 14) -> str:
    return fill(name.replace("_", " "), width=width)


def annotate_axis_with_brackets(
    ax: plt.Axes,
    *,
    modalities: list[dict],  # [{'name': str, 'start': int, 'end': int}, ...]
    show_vertical_delimiters: bool = True,
    bracket_y_pad_axes: float = 0.02,  # gap above heatmap (axes coords)
    max_labels: int = 8,  # max modality names shown on top axis
    wrap_width: int = 14,
    delimiter_kwargs: dict | None = None,
    bracket_kwargs: dict | None = None,
    tick_kwargs: dict | None = None,
):
    if delimiter_kwargs is None:
        delimiter_kwargs = dict(color="k", alpha=0.25, lw=0.8)
    if bracket_kwargs is None:
        bracket_kwargs = dict(color="k", lw=1.2)
    if tick_kwargs is None:
        tick_kwargs = dict(color="k", lw=1.2)

    # x in data coords, y in axes coords
    bt = transforms.blended_transform_factory(ax.transData, ax.transAxes)

    # boundaries & centers
    bounds, centers, names = [], [], []
    for m in modalities:
        s, e = int(m["start"]), int(m["end"])
        bounds.append(s)
        centers.append((s + e - 1) / 2)
        names.append(m["name"])
    bounds.append(int(modalities[-1]["end"]))  # final right bound

    # vertical delimiters through heatmap
    if show_vertical_delimiters:
        ymin, ymax = ax.get_ylim()
        for x in bounds:
            ax.vlines(x - 0.5, ymin, ymax, **delimiter_kwargs)

    # bracket rails just above heatmap
    y = 1.0 + bracket_y_pad_axes
    for i in range(len(bounds) - 1):
        x0, x1 = bounds[i], bounds[i + 1] - 1
        # rail
        ax.plot(
            [x0, x1],
            [y, y],
            transform=bt,
            color="red",
            lw=2.0,
            zorder=10,
            clip_on=False,
        )

        # end ticks (short verticals)
        ax.plot(
            [x0, x0],
            [y, y + 0.025],
            transform=bt,
            color="red",
            lw=2.0,
            zorder=10,
            clip_on=False,
        )
        ax.plot(
            [x1, x1],
            [y, y + 0.025],
            transform=bt,
            color="red",
            lw=2.0,
            zorder=10,
            clip_on=False,
        )

    # top x-axis with thinned, wrapped labels
    n = len(centers)
    step = max(1, (n + max_labels - 1) // max_labels)
    idx = list(range(0, n, step))
    tick_positions = [centers[i] for i in idx]
    tick_labels = [_wrap_name(names[i], width=wrap_width) for i in idx]
    top = ax.secondary_xaxis("top")
    top.set_ticks([centers[i] for i in idx])
    top.xaxis.set_major_locator(FixedLocator(tick_positions))
    top.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    top.tick_params(axis="x", labelsize=8, pad=10)
    # top.set_ticks(tick_positions, labels=tick_labels)
    for tick in top.get_xticklabels():
        tick.set_ha("center")


def plot_token_heatmap_for_label_with_brackets(
    rollout,
    label_span,
    modality_segments,  # from build_segments_from_viz(..., exclude_brackets=True)
    label_segments,  # from build_label_segments_from_viz(...)
    *,
    highlight_label_key: str | None = None,
    MODALITIES_TO_COLS: dict | None = None,
    token_modalities_to_show: (
        list[str] | None
    ) = None,  # e.g., ["prs","anthropometrics"]
    vmax=None,
    figsize=None,
    dpi=160,
    show=False,
    return_fig=True,
):
    import matplotlib.pyplot as plt

    ls, le = label_span
    sub = rollout[ls:le, :]

    if figsize is None:
        figsize = _autosize_from_modalities(modality_segments)

    fig = plt.figure(figsize=figsize, dpi=dpi, constrained_layout=False)
    fig.subplots_adjust(top=2)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        sub.detach().cpu().numpy(), aspect="auto", interpolation="nearest", vmax=vmax
    )
    fig.colorbar(im, ax=ax, label="Attention")

    ax.set_xlabel("Sequence tokens")
    ax.set_ylabel("Label tokens")
    ax.set_title("Label-token → all tokens (rollout)", pad=8)

    # brackets: inputs (top) + labels (bottom)
    annotate_axis_with_input_and_label_brackets(
        ax,
        modality_segments=modality_segments,
        label_segments=label_segments,
        highlight_label=highlight_label_key,
        show_vertical_delimiters=True,
        top_max_modality_names=12,
        top_wrap_width=14,
        bottom_max_label_names=10,
        bottom_wrap_width=22,
        # you already added vertical tick text there; keep as-is
    )

    # NEW: per-token annotations for selected modalities (above the plot)
    if MODALITIES_TO_COLS is not None:
        tokmap = build_token_positions_for_modalities(
            modality_segments, MODALITIES_TO_COLS, on_mismatch="warn"
        )
        annotate_modality_tokens(
            ax,
            tokmap,
            which_modalities=token_modalities_to_show,  # None → annotate all present in tokmap
            max_tokens_per_mod=24,
            rotation=90,
            fontsize=6,
            color="white",
            pad_axes_y=0.5,
            tick_len_axes=0.02,
            wrap_width=36,
        )

    # give room for top token labels and bottom label names
    fig.subplots_adjust(top=0.90, bottom=0.22)

    if show:
        plt.show()
    if return_fig:
        return fig


# Optional: your disease grouping helper for the final section
def group_labels_by_disease(label_dict):
    groups = defaultdict(list)
    patt = re.compile(r"^has_(.+?)_in_(\d+)_years$")
    for lbl, (s, e) in label_dict.items():
        m = patt.match(lbl)
        disease = m.group(1) if m else lbl
        groups[disease].append((lbl, (s, e)))
    return dict(groups)


def build_segments_from_viz(
    viz: dict, inst_idx: int = 0, exclude_brackets: bool = False
):
    """Return modality & label segments from viz (drops bracket tokens if requested)."""
    input_se = viz["input_start_end"][inst_idx]  # OrderedDict[str, (s,e)]
    label_se = viz["label_start_end"][inst_idx]  # OrderedDict[str, (s,e)]
    cont_type = "bracket"  # your model uses bracket

    def adj(span):
        s, e = span
        if exclude_brackets and cont_type == "bracket" and (e - s) >= 2:
            return s + 1, e - 1
        return s, e

    modalities = []
    for name, (s, e) in input_se.items():
        s2, e2 = adj((s, e))
        if e2 > s2:
            modalities.append({"name": name, "start": s2, "end": e2})

    labels = [{"name": name, "start": s, "end": e} for name, (s, e) in label_se.items()]
    return modalities, labels


import matplotlib.pyplot as plt


def annotate_axis_with_spans(
    ax: plt.Axes,
    *,
    modalities: List[Dict],
    label_spans: List[Dict] | None = None,
    show_label_spans: bool = False,
    max_modality_names: int = 20,
    modality_alpha: float = 0.10,
    label_alpha: float = 0.06,
):
    """
    Shades vertical bands for each modality (and optionally label tokens).
    Also writes a subset of modality names at band centers to avoid clutter.
    """
    L = int(ax.images[0].get_array().shape[-1]) if ax.images else None

    # draw modality spans
    for m in modalities:
        s, e = m["start"], m["end"]
        ax.axvspan(s, e - 1, alpha=modality_alpha, lw=0.0)

    # optional: draw label spans lightly at the far-right (since labels are appended)
    if show_label_spans and label_spans:
        for lab in label_spans:
            s, e = lab["start"], lab["end"]
            ax.axvspan(s, e - 1, color="k", alpha=label_alpha, lw=0.0)

    # write some modality names at centers
    if len(modalities) > 0:
        step = max(1, len(modalities) // max_modality_names)
        chosen = modalities[::step]
        centers = [(m["start"] + m["end"] - 1) / 2 for m in chosen]
        names = [m["name"] for m in chosen]
        ax.set_xticks(centers, names, rotation=30, ha="right")


def plot_token_heatmap_for_label_with_spans(
    rollout,  # (L, L) tensor for a single instance
    label_span: tuple,  # (start, end) for the label
    modalities: List[Dict],
    label_spans: List[Dict] | None = None,
    *,
    show_label_spans: bool = False,
    vmax=None,
    figsize=(7.5, 4),
    show=True,
    return_fig=False,
):
    ls, le = label_span
    sub = rollout[ls:le, :]  # (L_label, L)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        sub.detach().cpu().numpy(), aspect="auto", interpolation="nearest", vmax=vmax
    )
    fig.colorbar(im, ax=ax, label="Attention")

    ax.set_xlabel("Sequence tokens (modality-shaded)")
    ax.set_ylabel("Label tokens")
    ax.set_title("Label-token → all tokens (rollout)")

    annotate_axis_with_spans(
        ax,
        modalities=modalities,
        label_spans=label_spans,
        show_label_spans=show_label_spans,
        max_modality_names=22,  # tune for readability
        modality_alpha=0.10,
        label_alpha=0.05,
    )

    fig.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return fig


def find_label_keys(
    label_dict: Dict[str, Tuple[int, int]],
    disease: str,
    horizon: Optional[int] = None,
) -> List[str]:
    """
    Returns all keys matching disease (and optional horizon).
    Keys look like: 'has_glaucoma_in_3_years'
    """
    patt = re.compile(rf"^has_{re.escape(disease)}_in_(\d+)_years$")
    out = []
    for k in label_dict.keys():
        m = patt.match(k)
        if m is None:
            continue
        if horizon is None or int(m.group(1)) == horizon:
            out.append(k)
    # keep original ordering
    return out


def get_label_span(viz, batch_idx: int, label_key: str) -> Tuple[int, int]:
    return viz["label_start_end"][batch_idx][label_key]


def get_combined_label_span_rows(
    R_single: torch.Tensor,
    label_spans: List[Tuple[int, int]],
    reduce: str = "mean",  # "mean" | "sum" | "max"
) -> torch.Tensor:
    """
    Given rollout R (L,L) for one instance and multiple label spans,
    return a single row-vector (L,) by pooling across the label rows.
    """
    rows = []
    for s, e in label_spans:
        rows.append(R_single[s:e, :])  # (len, L) (often len=1)
    M = (
        torch.cat(rows, dim=0)
        if rows
        else torch.zeros(0, R_single.shape[1], device=R_single.device)
    )
    if M.numel() == 0:
        return torch.zeros(R_single.shape[1], device=R_single.device)
    if reduce == "mean":
        return M.mean(dim=0)
    elif reduce == "sum":
        return M.sum(dim=0)
    elif reduce == "max":
        return M.max(dim=0).values
    else:
        raise ValueError("reduce must be 'mean'|'sum'|'max'")


def plot_heat_for_disease(
    R_single: torch.Tensor,
    label_dict: Dict[str, Tuple[int, int]],
    disease: str,
    horizon: Optional[int] = None,
    reduce: str = "mean",
):
    # gather keys & spans
    keys = find_label_keys(label_dict, disease=disease, horizon=horizon)
    if not keys:
        raise KeyError(
            f"No labels found for disease='{disease}' horizon={horizon}. "
            f"Available keys include: {list(label_dict.keys())[:5]} ..."
        )
    spans = [label_dict[k] for k in keys]

    if len(spans) == 1:
        # single label → use the stock helper
        plot_token_heatmap_for_label(R_single, spans[0])
    else:
        # average/sum/max across horizons, then show as a single-row heatmap
        pooled = get_combined_label_span_rows(R_single, spans, reduce=reduce)  # (L,)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 2.2))
        plt.imshow(
            pooled.unsqueeze(0).cpu().numpy(), aspect="auto", interpolation="nearest"
        )
        plt.colorbar(label="Attention")
        plt.yticks([0], [f"{disease} ({reduce} over {len(spans)} horizons)"])
        plt.xlabel("Sequence tokens")
        plt.tight_layout()
        plt.show()


def _autosize_row_from_modalities(
    modality_segments,
    *,
    min_w: float = 11.0,
    per_mod_w: float = 1.25,
    height: float = 2.8,  # compact, 1-row figure (e.g., disease-aggregated)
):
    n = max(1, len(modality_segments))
    return (max(min_w, per_mod_w * n), height)


def export_all_to_pdf_backup(
    viz,
    inst_idx=0,
    pdf_path="attention_report.pdf",
    reduce="mean",
    MODALITIES_TO_COLS=None,
):
    from matplotlib.backends.backend_pdf import PdfPages

    R = viz["rollout"][inst_idx]
    label_dict = viz["label_start_end"][inst_idx]
    mod_map = viz["per_instance"][inst_idx]

    modality_segments, _ = build_segments_from_viz(
        viz, inst_idx=inst_idx, exclude_brackets=False
    )
    label_segments = build_label_segments_from_viz(viz, inst_idx=inst_idx)
    groups = group_labels_by_disease(label_dict)

    with PdfPages(pdf_path) as pdf:
        # per-label heatmaps
        for lbl, span in label_dict.items():
            fig = plot_token_heatmap_for_label_with_brackets(
                R,
                span,
                modality_segments=modality_segments,
                label_segments=label_segments,
                highlight_label_key=lbl,
                MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                token_modalities_to_show=MODALITIES_TO_COLS.keys(),  # e.g., ["prs","anthropometrics","family_history"]
                vmax=None,
                figsize=_autosize_from_modalities(modality_segments),
                dpi=400,
                show=False,
                return_fig=True,
            )
            fig.suptitle(lbl.replace("_", " "), y=0.985, fontsize=11)
            pdf.savefig(fig)
            plt.close(fig)

        # 2) Per-label modality bars (unchanged)
        for lbl, md in mod_map.items():
            fig = plot_label_modality_bar(lbl, md, show=False, return_fig=True)
            fig.subplots_adjust(top=0.9)
            pdf.savefig(fig)
            plt.close(fig)

        # 3) Disease-aggregated rows (unchanged)
        for disease, items in groups.items():
            spans = [span for _, span in items]
            fig = plot_disease_aggregated_row(
                R,
                spans,
                disease=disease,
                reduce=reduce,
                figsize=_autosize_from_modalities(modality_segments),
                show=False,
                return_fig=True,
            )
            fig.subplots_adjust(top=0.88, bottom=0.22)
            pdf.savefig(fig)
            plt.close(fig)

    print(f"Saved: {pdf_path}")


# def export_all_to_pdf(viz, inst_idx=0, pdf_path="attention_report.pdf", reduce="mean"):
#     R = viz["rollout"][inst_idx]
#     label_dict = viz["label_start_end"][inst_idx]
#     mod_map = viz["per_instance"][inst_idx]
#     groups = group_labels_by_disease(label_dict)

#     with PdfPages(pdf_path) as pdf:
#         # per-label heatmaps
#         for lbl, span in label_dict.items():
#             fig = plot_token_heatmap_for_label(R, span, show=False, return_fig=True)
#             pdf.savefig(fig); plt.close(fig)

#         # per-label modality bars
#         for lbl, md in mod_map.items():
#             fig = plot_label_modality_bar(lbl, md, show=False, return_fig=True)
#             pdf.savefig(fig); plt.close(fig)

#         # disease-aggregated rows
#         for disease, items in groups.items():
#             spans = [span for _, span in items]
#             fig = plot_disease_aggregated_row(R, spans, disease=disease, reduce=reduce, show=False, return_fig=True)
#             pdf.savefig(fig); plt.close(fig)

#     print(f"Saved: {pdf_path}")


def plot_all_label_modality_bars(viz, inst_idx=0):
    mod_map = viz["per_instance"][inst_idx]  # {label: {modality: value}}
    for label, md in mod_map.items():
        items = sorted(md.items(), key=lambda kv: kv[1], reverse=True)
        mods, vals = zip(*items) if items else ([], [])
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 3.2))
        plt.bar(mods, vals)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Attention (rollout, aggregated)")
        plt.title(f"Instance {inst_idx}: {label} → modalities")
        plt.tight_layout()
        plt.show()


def plot_all_diseases_aggregated(viz, inst_idx=0, reduce="mean"):
    R = viz["rollout"][inst_idx]
    label_dict = viz["label_start_end"][inst_idx]
    groups = group_labels_by_disease(label_dict)

    for disease, items in groups.items():
        spans = [span for _, span in items]
        pooled = get_combined_label_span_rows(R, spans, reduce=reduce)  # (L,)
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 2.2))
        plt.imshow(
            pooled.unsqueeze(0).cpu().numpy(), aspect="auto", interpolation="nearest"
        )
        plt.colorbar(label="Attention")
        plt.yticks([0], [f"{disease} ({reduce} over {len(spans)} horizons)"])
        plt.xlabel("Sequence tokens")
        plt.tight_layout()
        plt.show()


def plot_all_labels_heatmaps(viz, inst_idx=0, vmax=None):
    R = viz["rollout"][inst_idx]  # (L, L)
    label_dict = viz["label_start_end"][inst_idx]
    for lbl, span in label_dict.items():
        print(f"[instance {inst_idx}] {lbl} -> span={span}")
        plot_token_heatmap_for_label(R, span, vmax=vmax)


# ---------- A. Shape helpers ----------


def _ensure_4d_per_layer(x: torch.Tensor) -> torch.Tensor:
    """
    Make a single-layer attention tensor 4D as (B, H, L, L).
    Accepts (B, H, L, L) or (B, L, L) and returns (B, H, L, L) with H=1 for headless.
    """
    if x.dim() == 4:
        B, A, L, L2 = x.shape
        assert L == L2, f"Last two dims must be equal (got {x.shape})"
        # Heuristic: if A equals L it could be (B, L, L, L) which is unusual; assume A is H
        return x  # already (B, H, L, L)
    elif x.dim() == 3:
        B, L, L2 = x.shape
        assert L == L2, f"Last two dims must be equal (got {x.shape})"
        return x.unsqueeze(1)  # (B, 1, L, L)
    else:
        raise ValueError(f"Unexpected attention per-layer shape: {tuple(x.shape)}")


def _standardize_attn_shape(attn_weights) -> torch.Tensor:
    """
    Accepts:
      - list/tuple of (B, H, L, L) or (B, L, L)  -> stacks to (n_layers, B, H, L, L)
      - tensor (n_layers, B, H, L, L)            -> returned as-is
      - tensor (n_layers, B, L, L)               -> add H=1
      - tensor (B, H, L, L)                      -> add n_layers=1
      - tensor (B, L, L)                         -> add H=1 and n_layers=1
    Returns: (n_layers, B, H, L, L)
    """
    if isinstance(attn_weights, (list, tuple)):
        # one tensor per layer
        layers = [_ensure_4d_per_layer(x) for x in attn_weights]
        A = torch.stack(layers, dim=0)  # (n_layers, B, H, L, L)
        return A

    # Tensor paths
    A = attn_weights
    if A.dim() == 5:
        # (n_layers, B, H, L, L)
        return A
    elif A.dim() == 4:
        # Could be (n_layers, B, L, L) or (B, H, L, L)
        n0, n1, n2, n3 = A.shape
        if n2 == n3 and n0 != n1:
            # assume (n_layers, B, L, L) -> add H=1
            return A.unsqueeze(2)
        else:
            # assume (B, H, L, L) -> add n_layers=1
            return A.unsqueeze(0)
    elif A.dim() == 3:
        # (B, L, L) -> add H=1 and n_layers=1
        return A.unsqueeze(0).unsqueeze(2)
    else:
        raise ValueError(f"Unexpected attention shape: {tuple(A.shape)}")


# ---------- B. Attention Rollout (per: Abnar & Zuidema, 2020 style) ----------


@torch.no_grad()
def attention_rollout(
    attn_weights: torch.Tensor,
    head_fusion: str = "mean",  # ["mean", "max"]
    add_identity: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute attention rollout over layers.
    Input: attn_weights -> (n_layers, B, n_heads, L, L)  (probabilities along the last dim)
    Output: rollout -> (B, L, L), where entry (i,j) ≈ how much token i depends on token j after all layers.
    """
    A = _standardize_attn_shape(attn_weights)  # (n_layers, B, H, L, L)
    n_layers, B, H, L, _ = A.shape

    # fuse heads
    if head_fusion == "mean":
        A = A.mean(dim=2)  # (n_layers, B, L, L)
    elif head_fusion == "max":
        A = A.max(dim=2).values
    else:
        raise ValueError("head_fusion must be 'mean' or 'max'")

    # (optional) add identity and renormalize per layer to keep it stochastic
    if add_identity:
        I = (
            torch.eye(L, device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)
        )  # (1,1,L,L)
        A = A + I
        A = A / (A.sum(dim=-1, keepdim=True) + eps)

    # rollout = A_L * ... * A_1  (left-multiplying so rows remain "query/target" tokens)
    R = A[0]
    for l in range(1, n_layers):
        R = torch.bmm(A[l], R)  # (B, L, L)
    return R  # (B, L, L)


# ---------- C. Aggregate attention from label tokens to input tokens ----------


def _token_range_sum(M: torch.Tensor, start: int, end: int, dim: int) -> torch.Tensor:
    """
    Sum over token indices [start:end) along the given dimension.
    """
    sl = slice(start, end)
    return M.narrow(dim, start, end - start).sum(dim=dim)


def _exclude_brackets(span: Tuple[int, int], container_type: str) -> Tuple[int, int]:
    """
    If using 'bracket' containers, drop the first and last tokens from a modality span.
    """
    s, e = span
    if container_type == "bracket" and (e - s) >= 2:
        return s + 1, e - 1
    return s, e


@torch.no_grad()
def aggregate_modality_attention_to_labels(
    rollout: torch.Tensor,  # (B, L, L) from attention_rollout
    input_start_end: List[OrderedDict],
    label_start_end: List[OrderedDict],
    modality_container_type: str = "bracket",
    reduce_label_tokens: str = "mean",  # ["mean", "sum", "max"]
    reduce_modality_tokens: str = "sum",  # ["sum", "mean", "max"]
) -> List[Dict[str, Dict[str, float]]]:
    """
    For each instance in batch, produce:
      result[inst][label_name][modality_name] = scalar attention mass
    Interprets rollout[row, col] as "row (query) attends to col (key)".
    We aggregate *rows* over label-token rows, and *cols* over modality-token cols.

    Returns: list of dicts (length B).
    """
    B, L, _ = rollout.shape
    out: List[Dict[str, Dict[str, float]]] = []

    for b in range(B):
        inst_map: Dict[str, Dict[str, float]] = {}
        for lbl, (ls, le) in label_start_end[b].items():
            # label token rows
            ls_, le_ = ls, le
            label_slice = rollout[b, ls_:le_, :]  # (L_label, L)

            # reduce over label-token rows
            if reduce_label_tokens == "mean":
                L_rows = label_slice.mean(dim=0)  # (L,)
            elif reduce_label_tokens == "sum":
                L_rows = label_slice.sum(dim=0)
            elif reduce_label_tokens == "max":
                L_rows = label_slice.max(dim=0).values
            else:
                raise ValueError("reduce_label_tokens must be 'mean'|'sum'|'max'")

            inst_map[lbl] = {}

            for mod, (ms, me) in input_start_end[b].items():
                ms_, me_ = _exclude_brackets((ms, me), modality_container_type)

                if me_ <= ms_:
                    inst_map[lbl][mod] = 0.0
                    continue

                # reduce over modality-token columns
                M_cols = L_rows[ms_:me_]  # (L_mod,)
                if reduce_modality_tokens == "sum":
                    v = float(M_cols.sum().item())
                elif reduce_modality_tokens == "mean":
                    v = float(M_cols.mean().item())
                elif reduce_modality_tokens == "max":
                    v = float(M_cols.max().item())
                else:
                    raise ValueError(
                        "reduce_modality_tokens must be 'sum'|'mean'|'max'"
                    )

                inst_map[lbl][mod] = v

        out.append(inst_map)
    return out


# ---------- D. Pretty plotting ----------


def plot_label_modality_heatmap(
    mod_attn: Dict[str, Dict[str, float]],
    title: str = "Label → Modality attention",
    sort_modalities_by_value_for_label: Optional[str] = None,
    figsize=(6, 3.5),
):
    """
    mod_attn: {label: {modality: value}}
    Makes one figure per label (simple bar chart).
    """
    for label, md in mod_attn.items():
        items = list(md.items())
        if (
            sort_modalities_by_value_for_label is None
            or sort_modalities_by_value_for_label != label
        ):
            items = sorted(items, key=lambda kv: kv[0])  # alpha by modality
        else:
            items = sorted(items, key=lambda kv: kv[1], reverse=True)

        mods, vals = zip(*items) if items else ([], [])
        plt.figure(figsize=figsize)
        plt.bar(mods, vals)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Attention (rollout, aggregated)")
        plt.title(f"{title}: {label}")
        plt.tight_layout()
        plt.show()


def plot_token_heatmap_for_label(
    rollout: torch.Tensor,  # (L, L) single instance
    label_span: tuple,
    token_names: list | None = None,
    figsize=(6, 4),
    vmax: float | None = None,
    show: bool = True,
    return_fig: bool = False,
):
    ls, le = label_span
    sub = rollout[ls:le, :]  # (L_label, L)

    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        sub.detach().cpu().numpy(), aspect="auto", interpolation="nearest", vmax=vmax
    )
    cbar = fig.colorbar(im, ax=ax, label="Attention")
    ax.set_xlabel("Sequence tokens")
    ax.set_ylabel("Label tokens")
    ax.set_title("Label-token → all tokens (rollout)")
    if token_names is not None:
        # optionally set xticks here
        pass
    fig.tight_layout()

    if show:
        plt.show()

    if return_fig:
        return fig


def plot_disease_aggregated_row(
    R_single: torch.Tensor,
    spans: list[tuple],
    disease: str,
    reduce="mean",
    figsize=(6, 2.2),
    show=True,
    return_fig=False,
):
    import matplotlib.pyplot as plt

    pooled = get_combined_label_span_rows(R_single, spans, reduce=reduce)  # (L,)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    im = ax.imshow(
        pooled.unsqueeze(0).detach().cpu().numpy(),
        aspect="auto",
        interpolation="nearest",
    )
    fig.colorbar(im, ax=ax, label="Attention")
    ax.set_yticks([0], [f"{disease} ({reduce} over {len(spans)} horizons)"])
    ax.set_xlabel("Sequence tokens")
    fig.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return fig


def plot_label_modality_bar(
    label: str, md: dict[str, float], figsize=(7, 3.2), show=True, return_fig=False
):
    import matplotlib.pyplot as plt

    items = sorted(md.items(), key=lambda kv: kv[1], reverse=True)
    mods, vals = zip(*items) if items else ([], [])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.bar(mods, vals)
    ax.set_xticklabels(mods, rotation=30, ha="right")
    ax.set_ylabel("Attention (rollout, aggregated)")
    ax.set_title(f"{label} → modalities")
    fig.tight_layout()
    if show:
        plt.show()
    if return_fig:
        return fig

from engine.universal_dense_onehead import seperate_onehead_outputs

@torch.no_grad()
def visualize_attention_by_modality(
    model: nn.Module,
    batch: List[Dict[str, torch.Tensor]],
    output_labels: List[List[str]],
    head_fusion: str = "mean",
    reduce_label_tokens: str = "mean",
    reduce_modality_tokens: str = "sum",
    exclude_brackets: bool = True,
    do_plots: bool = True,
    diseases = None,
    progression_label_years= None,
):
    """
    - Runs the model once with need_attn_weights=True.
    - Computes attention rollout.
    - Aggregates rollout attention from each label to each modality.
    - Optionally plots per-label bar charts.

    Returns:
        dict with keys:
          'rollout': (B, L, L) tensor,
          'per_instance': list of {label: {modality: value}},
          'input_start_end', 'label_start_end'
    """
    device = next(model.parameters()).device
    outputs = model(batch, output_labels=output_labels, need_attn_weights=True)
    outputs['out'] = seperate_onehead_outputs(outputs['out'], diseases, progression_label_years)


    assert (
        "attn_weights" in outputs
    ), "Model didn't return attn_weights; set need_attn_weights=True."
    attn = outputs[
        "attn_weights"
    ]  # expect (n_layers, B, H, L, L) or (n_layers, B, L, L)
    input_start_end = outputs.get("input_start_end")
    label_start_end = outputs.get("label_start_end")
    modality_container_type = getattr(model, "modality_container_type", "bracket")

    # (If you didn't add the optional tweak above, you can recompute spans by re-running
    #  model.to_seq / attach_container / concat_modalities / append_output_tokens here.)

    # rollout across layers
    R = attention_rollout(attn, head_fusion=head_fusion)  # (B, L, L)

    # aggregate per instance
    per_instance = aggregate_modality_attention_to_labels(
        R,
        input_start_end=input_start_end,
        label_start_end=label_start_end,
        modality_container_type=(
            modality_container_type if exclude_brackets else "splitter"
        ),
        reduce_label_tokens=reduce_label_tokens,
        reduce_modality_tokens=reduce_modality_tokens,
    )

    # Simple plots
    if do_plots:
        for b_idx, mod_map in enumerate(per_instance):
            plot_label_modality_heatmap(
                mod_map,
                title=f"[Instance {b_idx}] Label → Modality attention",
                sort_modalities_by_value_for_label=None,
            )

    return {
        "rollout": R,  # (B, L, L)
        "per_instance": per_instance,
        "input_start_end": input_start_end,
        "label_start_end": label_start_end,
        "outs": outputs,
    }


def export_all_to_pdf(
    viz,
    inst_idx=0,
    pdf_path="attention_report.pdf",
    *,
    reduce="mean",
    MODALITIES_TO_COLS=None,
    token_modalities_to_show=None,
    disease: str | None = None,
    horizon: int | None = None,
    top_k_features: int = 30,
    dpi: int = 300,
    include_per_modality_heatmaps: bool = True,
    modality_subset: list[str] | None = None,
    # fundus overlays
    include_fundus_overlays: bool = True,
    include_overall_fundus_overlay: bool = True,
    include_per_label_fundus_overlays: bool = True,
    samples=None,
    fundus_key: str = "fundus_image",
    image_size: int = 224,
    patch_size: int = 16,
    image_mean_std: tuple | None = None,
    overlay_alpha: float = 0.45,
    overlay_cmap: str = "magma",
    # attention backend
    attn_mode: str = "rollout",  # "rollout" | "attn_x_grad"
    model=None,  # required if attn_mode="attn_x_grad"
    attn_head_fusion: str = "max",
    attn_add_identity: bool = False,
):
    """
    Export a PDF report with attention visualizations.

    - viz: output of visualize_attention_by_modality
    - attn_mode:
        "rollout"     -> use viz["rollout"][inst_idx] (your current method)
        "attn_x_grad" -> recompute label-specific rollout with gradients
    """

    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.pyplot as plt

    label_dict = viz["label_start_end"][inst_idx]
    per_label_mod_map = viz["per_instance"][inst_idx]
    modality_segments, _ = build_segments_from_viz(
        viz, inst_idx=inst_idx, exclude_brackets=True
    )
    label_segments = build_label_segments_from_viz(viz, inst_idx=inst_idx)

    name2seg = {m["name"]: m for m in modality_segments}

    def _mods_iter():
        if modality_subset:
            return [name2seg[m] for m in modality_subset if m in name2seg]
        return modality_segments

    def _save(pdf, fig, top=0.90, bottom=0.24):
        fig.tight_layout()
        fig.subplots_adjust(top=top, bottom=bottom)
        pdf.savefig(fig, bbox_inches="tight", pad_inches=0.28)
        plt.close(fig)

    # cache to avoid recomputing Attn×Grad per label
    attn_cache = {}

    def _R_for_label(label_key: str | None):
        if attn_mode == "attn_x_grad" and label_key is not None:
            if label_key not in attn_cache:
                assert (
                    model is not None and samples is not None
                ), "model and samples are required for attn_x_grad"
                batch_1 = [samples[inst_idx]]
                R = attention_x_gradient_rollout(
                    model,
                    batch_1,
                    label_key,
                    head_fusion=attn_head_fusion,
                    add_identity=attn_add_identity,
                )[
                    0
                ]  # (L,L)
                attn_cache[label_key] = R
            return attn_cache[label_key]
        # fallback: global rollout
        return viz["rollout"][inst_idx]

    with PdfPages(pdf_path) as pdf:
        if disease is None:
            # ============ ALL LABELS ============
            for lbl, span in label_dict.items():
                R = _R_for_label(lbl)
                # full sequence page
                fig = plot_token_heatmap_for_label_with_brackets(
                    R,
                    span,
                    modality_segments=modality_segments,
                    label_segments=label_segments,
                    highlight_label_key=lbl,
                    MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                    token_modalities_to_show=(
                        list(MODALITIES_TO_COLS.keys()) if MODALITIES_TO_COLS else None
                    ),
                    figsize=_autosize_from_modalities(modality_segments),
                    dpi=dpi,
                    show=False,
                    return_fig=True,
                )
                fig.suptitle(lbl.replace("_", " "), y=0.985, fontsize=11)
                _save(pdf, fig)

                # per-modality pages
                if include_per_modality_heatmaps:
                    for seg in _mods_iter():
                        s, e = int(seg["start"]), int(seg["end"])
                        if e <= s:
                            continue
                        fig = plot_label_to_modality_heatmap(
                            R,
                            span,
                            seg,
                            MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                            rotation=90,
                            dpi=dpi,
                            show=False,
                            return_fig=True,
                            title=f"{lbl.replace('_',' ')} → {seg['name']}",
                        )
                        _save(pdf, fig, top=0.88, bottom=0.20)

                # per-label fundus overlay
                if (
                    include_fundus_overlays
                    and include_per_label_fundus_overlays
                    and samples is not None
                ):
                    fig = plot_fundus_attention_overlay(
                        viz,
                        inst_idx=inst_idx,
                        samples=samples,
                        label_key=lbl,
                        reduce=reduce,
                        fundus_key=fundus_key,
                        image_size=image_size,
                        patch_size=patch_size,
                        image_mean_std=image_mean_std,
                        alpha=overlay_alpha,
                        cmap=overlay_cmap,
                        show=False,
                    )
                    fig.suptitle(
                        f"Fundus overlay: {lbl.replace('_',' ')}", y=0.985, fontsize=11
                    )
                    _save(pdf, fig, top=0.92, bottom=0.02)

            # aggregated bars
            for lbl, md in per_label_mod_map.items():
                items = sorted(md.items(), key=lambda kv: kv[1], reverse=True)
                mods, vals = zip(*items) if items else ([], [])
                fig = plt.figure(
                    figsize=(max(8, 0.6 * max(1, len(items))), 3.2), dpi=dpi
                )
                ax = fig.add_subplot(111)
                ax.bar(mods, vals)
                ax.set_xticklabels(mods, rotation=90, ha="right")
                ax.set_ylabel("Attention (rollout, aggregated)")
                ax.set_title(f"{lbl} → modalities")
                _save(pdf, fig, top=0.90, bottom=0.20)

            # disease-aggregated rows
            groups = group_labels_by_disease(label_dict)
            for disease_name, items in groups.items():
                spans = [span for _, span in items]
                R = _R_for_label(None)  # global rollout
                fig = plot_disease_aggregated_row(
                    R,
                    spans,
                    disease=disease_name,
                    reduce=reduce,
                    figsize=_autosize_row_from_modalities(modality_segments),
                    show=False,
                    return_fig=True,
                )
                _save(pdf, fig, top=0.88, bottom=0.22)

            # overall fundus overlay
            if (
                include_fundus_overlays
                and include_overall_fundus_overlay
                and samples is not None
            ):
                fig = plot_fundus_attention_overlay(
                    viz,
                    inst_idx=inst_idx,
                    samples=samples,
                    label_key=None,
                    reduce=reduce,
                    fundus_key=fundus_key,
                    image_size=image_size,
                    patch_size=patch_size,
                    image_mean_std=image_mean_std,
                    alpha=overlay_alpha,
                    cmap=overlay_cmap,
                    show=False,
                )
                fig.suptitle("Fundus overlay (all labels pooled)", y=0.985, fontsize=11)
                _save(pdf, fig, top=0.92, bottom=0.02)

        else:
            # ============ DISEASE-SPECIFIC ============
            keys = find_label_keys(label_dict, disease=disease, horizon=horizon)
            if not keys:
                raise KeyError(f"No labels for disease='{disease}', horizon={horizon}")
            spans = [label_dict[k] for k in keys]

            for k in keys:
                R = _R_for_label(k)
                print(
                    "R min/max/mean:", R.min().item(), R.max().item(), R.mean().item()
                )
                print("Any nonzero?", (R.abs() > 1e-8).any().item())
                fig = plot_token_heatmap_for_label_with_brackets(
                    R,
                    label_dict[k],
                    modality_segments=modality_segments,
                    label_segments=label_segments,
                    highlight_label_key=k,
                    MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                    token_modalities_to_show=(
                        list(MODALITIES_TO_COLS.keys()) if MODALITIES_TO_COLS else None
                    ),
                    figsize=_autosize_from_modalities(modality_segments),
                    dpi=dpi,
                    show=False,
                    return_fig=True,
                )
                fig.suptitle(k.replace("_", " "), y=0.985, fontsize=11)
                _save(pdf, fig)

                if (
                    include_fundus_overlays
                    and include_per_label_fundus_overlays
                    and samples is not None
                ):
                    fig = plot_fundus_attention_overlay(
                        viz,
                        inst_idx=inst_idx,
                        samples=samples,
                        label_key=k,
                        reduce=reduce,
                        fundus_key=fundus_key,
                        image_size=image_size,
                        patch_size=patch_size,
                        image_mean_std=image_mean_std,
                        alpha=overlay_alpha,
                        cmap=overlay_cmap,
                        show=False,
                    )
                    fig.suptitle(
                        f"Fundus overlay: {k.replace('_',' ')}", y=0.985, fontsize=11
                    )
                    _save(pdf, fig, top=0.92, bottom=0.02)

            # pooled disease vector + modality maps
            R = _R_for_label(
                keys[0]
            )  # pick one label to seed vector (or average if you want)
            v = label_conditioned_vector(R, spans, reduce=reduce)

            if include_per_modality_heatmaps:
                pretty = f"{disease} ({'all horizons' if horizon is None else f'{horizon}y'})"
                for seg in _mods_iter():
                    s, e = int(seg["start"]), int(seg["end"])
                    if e <= s:
                        continue
                    fig = plot_vector_to_modality_row(
                        v,
                        seg,
                        MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                        rotation=90,
                        dpi=dpi,
                        show=False,
                        return_fig=True,
                        title=f"{pretty} → {seg['name']}",
                    )
                    _save(pdf, fig, top=0.88, bottom=0.18)

            # bar chart of modalities
            md = aggregate_vector_to_modalities(v, modality_segments, mode="sum")
            items = sorted(md.items(), key=lambda kv: kv[1], reverse=True)
            mods, vals = zip(*items) if items else ([], [])
            pretty = (
                f"{disease} ({'all horizons' if horizon is None else f'{horizon}y'})"
            )
            fig = plt.figure(figsize=(max(8, 0.6 * max(1, len(items))), 3.2), dpi=dpi)
            ax = fig.add_subplot(111)
            ax.bar(mods, vals)
            ax.set_xticklabels(mods, rotation=90, ha="right")
            ax.set_ylabel("Attention (sum)")
            ax.set_title(f"{pretty} → modalities")
            _save(pdf, fig, top=0.90, bottom=0.20)

            # bar chart of top-k features
            feats = aggregate_vector_to_features(
                v,
                modality_segments,
                MODALITIES_TO_COLS or {},
                target_modalities=modality_subset,
                mode="sum",
                on_mismatch="warn",
            )
            items = sorted(feats.items(), key=lambda kv: kv[1], reverse=True)[
                :top_k_features
            ]
            names, vals = zip(*items) if items else ([], [])
            fig = plt.figure(figsize=(10, max(3.2, 0.22 * max(1, len(items)))), dpi=dpi)
            ax = fig.add_subplot(111)
            ax.barh(range(len(items)), vals)
            ax.set_yticks(range(len(items)))
            ax.set_yticklabels([n.replace("_", " ") for n in names])
            ax.invert_yaxis()
            ax.set_xlabel("Attention (sum)")
            ax.set_title(f"{pretty} → top {top_k_features} features")
            _save(pdf, fig, top=0.92, bottom=0.20)

            if (
                include_fundus_overlays
                and include_overall_fundus_overlay
                and samples is not None
            ):
                fig = plot_fundus_attention_overlay(
                    viz,
                    inst_idx=inst_idx,
                    samples=samples,
                    label_key=None,
                    reduce=reduce,
                    fundus_key=fundus_key,
                    image_size=image_size,
                    patch_size=patch_size,
                    image_mean_std=image_mean_std,
                    alpha=overlay_alpha,
                    cmap=overlay_cmap,
                    show=False,
                )
                fig.suptitle(f"Fundus overlay: {pretty}", y=0.985, fontsize=11)
                _save(pdf, fig, top=0.92, bottom=0.02)

    print(f"Saved: {pdf_path}")


def export_all_to_pdf_gradcam(
    viz,
    inst_idx=0,
    pdf_path="attention_report_gradcam.pdf",
    *,
    model=None,
    samples=None,
    disease: str | None = None,
    horizon: int | None = None,
    dpi: int = 300,
    # fundus params
    fundus_key: str = "fundus_image",
    image_size: int = 224,
    patch_size: int = 16,
    image_mean_std: (
        tuple | None
    ) = None,  # (mean,std) used during training normalization
    overlay_alpha: float = 0.65,
    overlay_cmap: str = "turbo",
    # optional feature aggregation (non-image) to include after each page
    include_topk_features: bool = False,
    MODALITIES_TO_COLS=None,
    modality_subset: list[str] | None = None,
    top_k_features: int = 30,
    reduce: str = "mean",
):
    """
    Export a PDF where each page shows the **Grad-CAM fundus overlay** for the selected label(s).
    - If disease=None: do all labels for this instance.
    - Else: only labels matching disease (+ optional horizon).
    """
    assert (
        model is not None and samples is not None
    ), "Pass model= and samples= for Grad-CAM."
    label_dict = viz["label_start_end"][inst_idx]

    # pick labels to render
    if disease is None:
        items = list(label_dict.items())
    else:
        keys = find_label_keys(label_dict, disease=disease, horizon=horizon)
        if not keys:
            raise KeyError(f"No labels for disease='{disease}', horizon={horizon}")
        items = [(k, label_dict[k]) for k in keys]

    # for optional feature aggregation
    modality_segments, _ = build_segments_from_viz(
        viz, inst_idx=inst_idx, exclude_brackets=True
    )

    with PdfPages(pdf_path) as pdf:
        for lbl, _span in items:
            # 1) Grad-CAM on fundus for this label
            heat, base_img = _fundus_gradcam_for_label(
                model,
                samples[inst_idx],
                lbl,
                fundus_key=fundus_key,
                image_size=image_size,
                patch_size=patch_size,
                mean_std=image_mean_std,
            )
            fig = _overlay_heatmap(
                base_img, heat, alpha=overlay_alpha, cmap=overlay_cmap
            )
            fig.suptitle(lbl.replace("_", " "), y=0.995, fontsize=12)
            pdf.savefig(fig, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)

            # 2) (optional) top-k non-image features for the same label (uses your rollout vector path)
            if include_topk_features:
                # we reuse your rollout to build a label-conditioned vector on the full sequence
                R = viz["rollout"][inst_idx]
                v = label_conditioned_vector(
                    R, [label_dict[lbl]], reduce=reduce
                )  # (L,)
                feats = aggregate_vector_to_features(
                    v,
                    modality_segments,
                    MODALITIES_TO_COLS or {},
                    target_modalities=modality_subset,
                    mode="sum",
                    on_mismatch="warn",
                )
                items_sorted = sorted(
                    feats.items(), key=lambda kv: kv[1], reverse=True
                )[:top_k_features]
                names, vals = zip(*items_sorted) if items_sorted else ([], [])

                fig = plt.figure(
                    figsize=(10, max(3.2, 0.22 * max(1, len(names)))), dpi=dpi
                )
                ax = fig.add_subplot(111)
                ax.barh(range(len(names)), vals)
                ax.set_yticks(range(len(names)))
                ax.set_yticklabels([n.replace("_", " ") for n in names])
                ax.invert_yaxis()
                ax.set_xlabel("Attention (sum)")
                ax.set_title(
                    f"Top {top_k_features} non-image features: {lbl.replace('_',' ')}"
                )
                fig.tight_layout()
                pdf.savefig(fig, bbox_inches="tight", pad_inches=0.25)
                plt.close(fig)

    print(f"Saved Grad-CAM report: {pdf_path}")


def _safe_name(s: str) -> str:
    s = s.strip().lower().replace(" ", "_")
    # keep letters, numbers, underscore, hyphen, plus
    return re.sub(r"[^a-z0-9_\-+]", "", s)


class Debugger:
    pass


import os, re, torch, numpy as np, matplotlib.pyplot as plt
import torch.nn.functional as F

# ---------- your helpers are assumed to exist ----------
# build_token_positions_for_modalities(...)
# annotate_modality_tokens(...)

# ---------- utility helpers ----------


def _safe_name(s: str) -> str:
    s = str(s).strip().lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_\-+]", "", s)


def gradcam_tokens_for_label(
    model, sample, label_key: str, modality_key: str, *, normalize: bool = True
):
    """
    Generic Grad-CAM over the token embeddings produced by model.input_to_seq[modality_key].
    Returns a 1D numpy array of length num_tokens. If normalize=True, scales to [0,1].
    """
    assert (
        modality_key in model.input_to_seq
    ), f"{modality_key} not found in model.input_to_seq"

    cache = {"tok": None}

    def _hook(_m, _in, out):
        tok = out[0] if out.dim() == 3 else out  # (L,D) or (B,L,D)->(L,D)
        tok.retain_grad()
        cache["tok"] = tok

    h = model.input_to_seq[modality_key].register_forward_hook(_hook)

    model.zero_grad(set_to_none=True)
    model.eval()
    outs = model([sample], output_labels=[[label_key]], need_attn_weights=False)
    logit = outs["out"][0][label_key].mean()
    logit.backward()
    h.remove()

    tok = cache["tok"]
    assert tok is not None, f"Hook for {modality_key} did not fire."

    sal = F.relu((tok * tok.grad).sum(-1))  # (L,)
    sal_np = sal.detach().cpu().numpy().astype(np.float32)

    if normalize:
        m = float(sal_np.max())
        if m > 0:
            sal_np /= m

    return sal_np


def _save_modality_row_with_annotations(
    cam_1d: np.ndarray,
    modality_name: str,
    modality_segments: list[dict],
    MODALITIES_TO_COLS: dict[str, list[str]] | None,
    title: str,
    out_path: str,
    *,
    dpi: int = 300,
    annotate_tokens: bool = True,
    annot_which_modalities: list[str] | None = None,
    annot_max_tokens_per_mod: int = 24,
    annot_rotation: int = 90,
    annot_fontsize: int = 6,
    annot_color: str = "white",
    annot_pad_axes_y: float = 0.06,
    annot_tick_len_axes: float = 0.018,
    annot_wrap_width: int = 36,
):
    """
    Draw a 1xL Grad-CAM row and (optionally) annotate token names above it.
    We convert global token positions into local 1..L coordinates so that
    the annotations align with the row image.
    """
    L = len(cam_1d)
    fig = plt.figure(figsize=((0.35 * L) + 2, 8), dpi=dpi)
    ax = fig.add_subplot(111)
    # Make x-domain exactly 1..L to match annotate_modality_tokens (which uses idx+1)
    im = ax.imshow(
        cam_1d[None, :],
        aspect="auto",
        cmap="viridis",
        # vmin=0,
        # vmax=1,  # cmap for other modalities
        # extent=[1, L, 0, 1]
    )
    ax.set_yticks([])
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_xticklabels([])

    # X ticks optional (often hidden when annotated)
    # ax.set_xticks(range(1, L+1))
    # ax.set_xticklabels(range(1, L+1), rotation=90, va="bottom")
    # ax.xaxis.set_tick_params(pad=6)

    # Only annotate if we have the mapping
    if annotate_tokens and MODALITIES_TO_COLS:
        # Build GLOBAL positions map once
        token_map_global = build_token_positions_for_modalities(
            modality_segments, MODALITIES_TO_COLS, on_mismatch="truncate"
        )
        if modality_name in token_map_global and token_map_global[modality_name]:
            # Convert to LOCAL positions 1..L
            # Figure out this modality's start index from segments
            name2seg = {
                m["name"]: (int(m["start"]), int(m["end"])) for m in modality_segments
            }
            if modality_name in name2seg:
                s_global, e_global = name2seg[modality_name]
                # Prepare a local token_map where each (idx,name) -> (idx - s_global + 1, name)
                pairs = token_map_global[modality_name]
                pairs_local = []
                for idx, nm in pairs:
                    local = idx - (s_global + 1)
                    # if 1 <= local <= L:
                    pairs_local.append((local, nm))
                if pairs_local:
                    token_map_local = {modality_name: pairs_local}
                    annotate_modality_tokens(
                        ax,
                        token_map_local,
                        which_modalities=(
                            [modality_name]
                            if annot_which_modalities is None
                            else annot_which_modalities
                        ),
                        max_tokens_per_mod=annot_max_tokens_per_mod,
                        rotation=annot_rotation,
                        fontsize=annot_fontsize,
                        color=annot_color,
                        pad_axes_y=annot_pad_axes_y,
                        tick_len_axes=annot_tick_len_axes,
                        wrap_width=annot_wrap_width,
                    )

    fig.colorbar(im, ax=ax, pad=0.02, label="Grad-CAM")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def save_modality_grid(cam_1d, grid_hw, title="", out_path="grid.png", dpi=300):
    """Reshape Grad-CAM vector to grid and save (no token annotations for grids)."""
    H_p, W_p = grid_hw
    cam_2d = cam_1d.reshape(H_p, W_p)
    fig = plt.figure(figsize=(max(4, 0.4 * W_p), max(3, 0.4 * H_p)), dpi=dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(cam_2d, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="Grad-CAM")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return out_path


def save_cross_modal_token_row_with_annotations(
    cams_by_mod: dict[str, np.ndarray],  # {"mod": (L_mod,)}
    modality_order: list[str],  # concat order
    token_names_by_mod: dict[str, list[str]] | None,
    title: str,
    out_path: str,
    *,
    dpi: int = 300,
    cmap: str = "magma",
    normalize_per_mod_for_display: bool = True,  # keeps weak-but-structured mods visible
    annot_color: str = "white",
    annot_rotation: int = 90,
    annot_fontsize: int = 5,
    annot_wrap_width: int = 34,
    # NEW: group brackets
    show_group_brackets: bool = True,
    bracket_color: str = "black",
    bracket_linewidth: float = 1.5,
    bracket_y0: float = 1.02,  # axes coords: vertical tick bottom
    bracket_y1: float = 1.10,  # axes coords: horizontal bar height
    bracket_label_fontsize: int = 8,
    bracket_label_color: str = "black",
):
    """
    Concatenate all modality token CAMs into one long row and annotate EVERY token.
    Also draws a bracket ( └───┘ ) above the token span for each modality group.
    Uses annotate_modality_tokens(ax, token_map, ...).
    """
    row_vals = []
    token_map = {}  # {mod: [(global_idx0_based, token_name), ...]}
    group_spans = []  # [(mod, start_idx0, end_idx0_exclusive), ...]
    token_offset = 0  # 0-based; annotate_modality_tokens adds +1 internally

    for mod in modality_order:
        if mod not in cams_by_mod:
            continue
        v = np.asarray(cams_by_mod[mod], dtype=np.float32)
        if v.size == 0:
            continue
        if normalize_per_mod_for_display and v.max() > 0:
            v = v / v.max()

        # names
        if token_names_by_mod and mod in token_names_by_mod:
            names = token_names_by_mod[mod]
        else:
            names = [f"{mod}_t{i+1}" for i in range(len(v))]

        k = min(len(v), len(names))
        v = v[:k]
        names = names[:k]

        # build token map (0-based indices; annotator will use idx+1)
        pairs = [(token_offset + i - 1, nm) for i, nm in enumerate(names)]
        token_map[mod] = pairs

        # remember span for bracket: [start, end)
        group_spans.append((mod, token_offset - 1, token_offset + len(v) - 1))

        row_vals.append(v)
        token_offset += len(v)

    if not row_vals:
        print("[cross-modal] nothing to plot.")
        return None

    row = np.concatenate(row_vals)
    L = row.shape[0]

    # Plot with pixel CENTERS at integer positions 1..L
    fig = plt.figure(figsize=(max(10, 0.25 * L), 8), dpi=dpi)
    fig.subplots_adjust(top=0.85)
    ax = fig.add_subplot(111)
    Debugger.row = row
    Debugger.token_map = token_map
    im = ax.imshow(
        row[None, :],
        aspect="auto",
        cmap=cmap,
        # vmin=0, vmax=1,
        # extent=[0.5, L + 0.5, 0, 1]
    )
    # ax.set_xlim(0.5, L + 0.5)
    ax.set_title(title, y=1.25)
    ax.set_yticks([])
    ax.get_xaxis().set_visible(False)

    # Annotate EVERY token name
    annotate_modality_tokens(
        ax,
        token_map,
        which_modalities=list(token_map.keys()),
        max_tokens_per_mod=10**9,  # draw ALL tokens (no thinning)
        rotation=annot_rotation,
        fontsize=annot_fontsize,
        color=annot_color,
        pad_axes_y=0.06,
        tick_len_axes=0.018,
        wrap_width=annot_wrap_width,
    )

    # Draw brackets per group (in data-x, axes-y)
    if show_group_brackets and group_spans:
        bt = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        for mod, s0, e0 in group_spans:
            # Convert to 1-based centers -> edges are at 0.5 and +0.5 already via extent.
            x_left = s0 + 0.5
            x_right = e0 + 0.5

            # vertical ticks
            ax.plot(
                [x_left, x_left],
                [bracket_y0, bracket_y1],
                transform=bt,
                color=bracket_color,
                lw=bracket_linewidth,
                clip_on=False,
                zorder=20,
            )
            ax.plot(
                [x_right, x_right],
                [bracket_y0, bracket_y1],
                transform=bt,
                color=bracket_color,
                lw=bracket_linewidth,
                clip_on=False,
                zorder=20,
            )
            # horizontal bar
            ax.plot(
                [x_left, x_right],
                [bracket_y1, bracket_y1],
                transform=bt,
                color=bracket_color,
                lw=bracket_linewidth,
                clip_on=False,
                zorder=20,
            )

            # label centered above the bracket
            txt = ax.text(
                (x_left + x_right) / 2.0,
                bracket_y1 + 0.03,
                mod,
                transform=bt,
                ha="center",
                va="bottom",
                fontsize=bracket_label_fontsize,
                color=bracket_label_color,
                clip_on=False,
                zorder=21,
            )
            # add outline for readability
            # txt.set_path_effects([
            #     path_effects.Stroke(linewidth=1.2, foreground="black"),
            #     path_effects.Normal()
            # ])

    # tiny colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Grad-CAM (per-mod normalized)", rotation=90)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    return out_path


# ---------- main exporter with token annotations ----------


def export_all_to_images_gradcam_backup(
    viz,
    inst_idx=0,
    *,
    model=None,
    samples=None,
    disease: str | None = None,
    horizon: int | None = None,
    dpi: int = 300,
    out_dir: str = "gradcam_images",
    img_format: str = "png",
    filename_prefix: str | None = None,
    # fundus
    fundus_key: str = "fundus_image",
    image_size: int = 224,
    patch_size: int = 16,
    image_mean_std: tuple | None = None,
    overlay_alpha: float = 0.65,
    overlay_cmap: str = "magma",
    # Grad-CAM for all modalities
    other_modalities_gradcam: bool = True,
    modalities_for_gradcam: (
        list[str] | None
    ) = None,  # None -> all model.input_to_seq keys
    modality_token_grids: dict[str, tuple[int, int]] | None = None,
    # NEW: token annotation config (for 1-D rows)
    MODALITIES_TO_COLS: dict[str, list[str]] | None = None,
    annotate_tokens: bool = True,
    annot_max_tokens_per_mod: int = 24,
    annot_rotation: int = 90,
    annot_fontsize: int = 6,
    annot_color: str = "black",
    annot_pad_axes_y: float = 0.06,
    annot_tick_len_axes: float = 0.018,
    annot_wrap_width: int = 36,
):
    """
    Compute and save Grad-CAM visualizations for fundus and all other modalities.
    For 1-D rows, token names are annotated using your helper functions.
    """
    assert model is not None and samples is not None
    os.makedirs(out_dir, exist_ok=True)

    prefix = (filename_prefix or "").strip()
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    label_dict = viz["label_start_end"][inst_idx]
    # we need segments to translate global token indices to local row coords
    modality_segments, _ = build_segments_from_viz(
        viz, inst_idx=inst_idx, exclude_brackets=True
    )

    saved = {"fundus": [], "modalities": []}

    # pick labels
    if disease is None:
        items = list(label_dict.items())
    else:
        keys = find_label_keys(label_dict, disease=disease, horizon=horizon)
        items = [(k, label_dict[k]) for k in keys]

    all_mods = list(getattr(model, "input_to_seq").keys())
    mods_to_do = [m for m in (modalities_for_gradcam or all_mods)]

    for lbl, _span in items:
        label_safe = _safe_name(lbl)
        print(f"[GradCAM] processing {lbl}")

        # --- fundus (image overlay) ---
        if fundus_key in model.input_to_seq and fundus_key in mods_to_do:
            heat, base_img = _fundus_gradcam_for_label(
                model,
                samples[inst_idx],
                lbl,
                fundus_key=fundus_key,
                image_size=image_size,
                patch_size=patch_size,
                mean_std=image_mean_std,
            )
            fig = _overlay_heatmap(
                base_img, heat, alpha=overlay_alpha, cmap=overlay_cmap
            )
            fig.suptitle(lbl.replace("_", " "), y=0.995, fontsize=12)
            out_path = os.path.join(
                out_dir,
                f"{prefix}inst{inst_idx}__{label_safe}__{_safe_name(fundus_key)}.{img_format}",
            )
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)
            saved["fundus"].append(out_path)

        # --- other modalities (row/grid + token annotations for rows) ---
        if other_modalities_gradcam:
            for mod in mods_to_do:
                if mod == fundus_key and fundus_key in model.input_to_seq:
                    # already handled as an image overlay above
                    continue
                if mod not in model.input_to_seq:
                    continue

                try:
                    cam = gradcam_tokens_for_label(model, samples[inst_idx], lbl, mod)
                except Exception as e:
                    print(f"[warn] {mod}: Grad-CAM failed ({e})")
                    continue

                pretty_title = f"{lbl.replace('_',' ')} → {mod}"
                mod_safe = _safe_name(mod)
                out_path = os.path.join(
                    out_dir,
                    f"{prefix}inst{inst_idx}__{label_safe}__{mod_safe}.{img_format}",
                )

                if modality_token_grids and mod in modality_token_grids:
                    # 2D visualization (no token names overlaid here)
                    save_modality_grid(
                        cam,
                        modality_token_grids[mod],
                        title=pretty_title,
                        out_path=out_path,
                        dpi=dpi,
                    )
                else:
                    print("1D token vis")
                    # 1D row with token annotations (if provided)
                    _save_modality_row_with_annotations(
                        cam_1d=cam,
                        modality_name=mod,
                        modality_segments=modality_segments,
                        MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                        title=pretty_title,
                        out_path=out_path,
                        dpi=dpi,
                        annotate_tokens=annotate_tokens,
                        annot_which_modalities=[
                            mod
                        ],  # only annotate this modality on this figure
                        annot_max_tokens_per_mod=annot_max_tokens_per_mod,
                        annot_rotation=annot_rotation,
                        annot_fontsize=annot_fontsize,
                        annot_color=annot_color,
                        annot_pad_axes_y=annot_pad_axes_y,
                        annot_tick_len_axes=annot_tick_len_axes,
                        annot_wrap_width=annot_wrap_width,
                    )

                saved["modalities"].append(out_path)

    print(
        f"\nSaved {len(saved['fundus'])} fundus + {len(saved['modalities'])} other-modality Grad-CAM(s) to '{out_dir}'."
    )
    return saved


def export_all_to_images_gradcam(
    viz,
    inst_idx=0,
    *,
    model=None,
    samples=None,
    disease: str | None = None,
    horizon: int | None = None,
    dpi: int = 300,
    out_dir: str = "gradcam_images",
    img_format: str = "png",
    filename_prefix: str | None = None,
    # fundus
    fundus_key: str = "fundus_image",
    image_size: int = 224,
    patch_size: int = 16,
    image_mean_std: tuple | None = None,
    overlay_alpha: float = 0.65,
    overlay_cmap: str = "magma",
    # Grad-CAM for all modalities
    other_modalities_gradcam: bool = True,
    modalities_for_gradcam: list[str] | None = None,
    modality_token_grids: dict[str, tuple[int, int]] | None = None,
    # token annotation for 1-D rows
    MODALITIES_TO_COLS: dict[str, list[str]] | None = None,
    annotate_tokens: bool = True,
    annot_max_tokens_per_mod: int = 24,
    annot_rotation: int = 90,
    annot_fontsize: int = 6,
    annot_color: str = "black",
    annot_pad_axes_y: float = 0.06,
    annot_tick_len_axes: float = 0.018,
    annot_wrap_width: int = 36,
    # NEW: cross-modal concatenated row
    emit_cross_modal_row: bool = True,
    cross_modal_row_cmap: str = "magma",
    cross_modal_row_fontsize: int = 5,
):
    """
    Compute and save Grad-CAM visualizations for fundus and all other modalities.
    Adds one concatenated cross-modal token heat row with EVERY token annotated.
    """
    assert model is not None and samples is not None
    os.makedirs(out_dir, exist_ok=True)

    prefix = (filename_prefix or "").strip()
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    label_dict = viz["label_start_end"][inst_idx]
    modality_segments, _ = build_segments_from_viz(
        viz, inst_idx=inst_idx, exclude_brackets=True
    )

    saved = {"fundus": [], "modalities": [], "cross_modal_row": []}

    # pick labels
    if disease is None:
        items = list(label_dict.items())
    else:
        keys = find_label_keys(label_dict, disease=disease, horizon=horizon)
        items = [(k, label_dict[k]) for k in keys]

    all_mods = list(getattr(model, "input_to_seq").keys())
    mods_to_do = [m for m in (modalities_for_gradcam or all_mods)]

    for lbl, _span in items:
        label_safe = _safe_name(lbl)
        print(f"[GradCAM] processing {lbl}")

        # for the cross-modal row (collect raw cams + names)
        cams_by_mod = {}
        names_by_mod = {}

        # --- fundus overlay ---
        if fundus_key in model.input_to_seq and fundus_key in mods_to_do:
            heat, base_img, pred = _fundus_gradcam_for_label(
                model,
                samples[inst_idx],
                lbl,
                fundus_key=fundus_key,
                image_size=image_size,
                patch_size=patch_size,
                mean_std=image_mean_std,
            )
            fig = _overlay_heatmap(
                base_img, heat, alpha=overlay_alpha, cmap=overlay_cmap
            )
            fig.suptitle(lbl.replace("_", " "), y=0.995, fontsize=12)
            out_path = os.path.join(
                out_dir,
                f"{prefix}inst{inst_idx}__{label_safe}__{_safe_name(fundus_key)}_overlay.{img_format}",
            )
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)

            saved["fundus"].append(out_path)

        # save logits and samples as txt
        pred_out_path = os.path.join(
            out_dir,
            f"{prefix}inst{inst_idx}__{label_safe}__{_safe_name(fundus_key)}_pred.txt",
        )
        with open(pred_out_path, "w") as f:
            f.write(f"Predictions for label '{lbl}':\n")
            for k, v in pred.items():
                f.write(f"{k}: {F.sigmoid(v).item()}\n")
            f.write("\nSample data:\n")
            for key, value in samples[inst_idx].items():
                f.write(f"{key}: {value}\n")

        # --- other modalities (and cache for cross-modal row) ---
        if other_modalities_gradcam:
            for mod in mods_to_do:
                if mod not in model.input_to_seq:
                    continue

                # raw (unnormalized) CAM for cross-modal fairness
                try:
                    cam_raw = gradcam_tokens_for_label(
                        model, samples[inst_idx], lbl, mod, normalize=False
                    )
                except Exception as e:
                    print(f"[warn] {mod}: Grad-CAM failed ({e})")
                    continue

                # Save per-modality visualization (1-D row or grid)
                pretty_title = f"{lbl.replace('_',' ')} → {mod}"
                mod_safe = _safe_name(mod)
                out_path = os.path.join(
                    out_dir,
                    f"{prefix}inst{inst_idx}__{label_safe}__{mod_safe}.{img_format}",
                )

                if modality_token_grids and mod in modality_token_grids:
                    save_modality_grid(
                        cam_raw,
                        modality_token_grids[mod],
                        title=pretty_title,
                        out_path=out_path,
                        dpi=dpi,
                    )
                else:
                    _save_modality_row_with_annotations(
                        cam_1d=cam_raw
                        / (cam_raw.max() + 1e-12),  # normalize for display
                        modality_name=mod,
                        modality_segments=modality_segments,
                        MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                        title=pretty_title,
                        out_path=out_path,
                        dpi=dpi,
                        annotate_tokens=annotate_tokens,
                        annot_which_modalities=[mod],
                        annot_max_tokens_per_mod=annot_max_tokens_per_mod,
                        annot_rotation=annot_rotation,
                        annot_fontsize=annot_fontsize,
                        annot_color=annot_color,
                        annot_pad_axes_y=annot_pad_axes_y,
                        annot_tick_len_axes=annot_tick_len_axes,
                        annot_wrap_width=annot_wrap_width,
                    )

                saved["modalities"].append(out_path)

                # cache for cross-modal row (raw values)
                cams_by_mod[mod] = cam_raw

                # token names for annotation in the big row
                if MODALITIES_TO_COLS and mod in MODALITIES_TO_COLS:
                    names_by_mod[mod] = MODALITIES_TO_COLS[mod]
                else:
                    names_by_mod[mod] = [f"{mod}_t{i+1}" for i in range(len(cam_raw))]

        # --- emit concatenated cross-modal row once per label ---
        if emit_cross_modal_row and cams_by_mod:
            concat_order = [
                m for m in mods_to_do if m in cams_by_mod and len(cams_by_mod[m]) > 0
            ]
            title = f"All tokens across modalities : {lbl.replace('_',' ')}"
            out_row = os.path.join(
                out_dir,
                f"{prefix}inst{inst_idx}__{label_safe}__token_importance_row.{img_format}",
            )
            save_cross_modal_token_row_with_annotations(
                cams_by_mod=cams_by_mod,
                modality_order=concat_order,
                token_names_by_mod=names_by_mod,
                title=title,
                out_path=out_row,
                dpi=dpi,
                cmap=cross_modal_row_cmap,
                normalize_per_mod_for_display=True,
                annot_color="white",  # readable on magma
                annot_rotation=90,
                annot_fontsize=cross_modal_row_fontsize,
                annot_wrap_width=34,
            )
            saved["cross_modal_row"].append(out_row)

    print(
        f"\nSaved {len(saved['fundus'])} fundus + {len(saved['modalities'])} modality CAM(s)"
        f" + {len(saved['cross_modal_row'])} cross-modal row(s) to '{out_dir}'."
    )
    return saved


import os
import numpy as np
import torch
import torch.nn.functional as F


def gradcam_tokens_for_label_v2(
    model,
    sample,
    label_key: str,
    modality_key: str,
    *,
    normalize: bool = True,
    debug: bool = False,
):
    assert (
        modality_key in model.input_to_seq
    ), f"{modality_key} not found in model.input_to_seq"

    cache = {"tok": None, "fired": False}

    def _hook(_m, _inp, out):
        cache["fired"] = True
        tok = out[0] if isinstance(out, (tuple, list)) else out
        cache["tok"] = tok
        if debug:
            print(
                f"[HOOK:{modality_key}] tok.shape={tuple(tok.shape)} req_grad={tok.requires_grad} grad_fn={tok.grad_fn}"
            )

    h = model.input_to_seq[modality_key].register_forward_hook(_hook)

    model.zero_grad(set_to_none=True)
    model.eval()

    with torch.enable_grad():
        outs = model([sample], output_labels=[[label_key]], need_attn_weights=False)
        logit = outs["out"][0][label_key].mean()

        h.remove()

        tok = cache["tok"]
        if tok is None:
            raise RuntimeError(f"Hook for {modality_key} did not fire.")

        # 讓 tok 一律變成 (B,L,D)
        if tok.dim() == 2:  # (B,D) -> (B,1,D)
            tok = tok.unsqueeze(1)
        elif tok.dim() == 3:
            pass
        else:
            raise RuntimeError(f"Unexpected tok dim={tok.dim()} for {modality_key}")

        grads = torch.autograd.grad(
            logit, tok, retain_graph=False, create_graph=False, allow_unused=True
        )[0]

        if grads is None:
            raise RuntimeError(
                f"Grad-CAM failed for {modality_key}: grads is None (logit not connected to tokens)."
            )

        # 取 batch=0
        tok0 = tok[0]  # (L,D)
        grad0 = grads[0]  # (L,D)

        sal = F.relu((tok0 * grad0).sum(-1))  # (L,)
        sal_np = sal.detach().cpu().numpy().astype(np.float32)

        if normalize:
            m = float(sal_np.max())
            if m > 0:
                sal_np /= m

        return sal_np


def export_all_to_images_gradcam_cnn(
    viz,
    inst_idx=0,
    *,
    model=None,
    samples=None,
    disease: str | None = None,
    horizon: int | None = None,
    dpi: int = 300,
    out_dir: str = "gradcam_images",
    img_format: str = "png",
    filename_prefix: str | None = None,
    # fundus
    fundus_key: str = "fundus_image",
    image_size: int = 224,
    patch_size: int = 16,
    image_mean_std: tuple | None = None,
    overlay_alpha: float = 0.65,
    overlay_cmap: str = "magma",
    # Grad-CAM for all modalities
    other_modalities_gradcam: bool = True,
    modalities_for_gradcam: list[str] | None = None,
    modality_token_grids: dict[str, tuple[int, int]] | None = None,
    # token annotation for 1-D rows
    MODALITIES_TO_COLS: dict[str, list[str]] | None = None,
    annotate_tokens: bool = True,
    annot_max_tokens_per_mod: int = 24,
    annot_rotation: int = 90,
    annot_fontsize: int = 6,
    annot_color: str = "black",
    annot_pad_axes_y: float = 0.06,
    annot_tick_len_axes: float = 0.018,
    annot_wrap_width: int = 36,
    # NEW: cross-modal concatenated row
    emit_cross_modal_row: bool = True,
    cross_modal_row_cmap: str = "magma",
    cross_modal_row_fontsize: int = 5,
):
    """
    Compute and save Grad-CAM visualizations for fundus and all other modalities.
    Adds one concatenated cross-modal token heat row with EVERY token annotated.
    """
    assert model is not None and samples is not None
    os.makedirs(out_dir, exist_ok=True)

    prefix = (filename_prefix or "").strip()
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    label_dict = viz["label_start_end"][inst_idx]
    modality_segments, _ = build_segments_from_viz(
        viz, inst_idx=inst_idx, exclude_brackets=True
    )

    saved = {"fundus": [], "modalities": [], "cross_modal_row": []}

    # pick labels
    if disease is None:
        items = list(label_dict.items())
    else:
        keys = find_label_keys(label_dict, disease=disease, horizon=horizon)
        items = [(k, label_dict[k]) for k in keys]

    all_mods = list(getattr(model, "input_to_seq").keys())
    mods_to_do = [m for m in (modalities_for_gradcam or all_mods)]

    for lbl, _span in items:
        label_safe = _safe_name(lbl)
        print(f"[GradCAM] processing {lbl}")

        # for the cross-modal row (collect raw cams + names)
        cams_by_mod = {}
        names_by_mod = {}

        # --- fundus overlay ---
        if fundus_key in model.input_to_seq and fundus_key in mods_to_do:
            # heat, base_img, pred = _fundus_gradcam_for_label(
            #     model,
            #     samples[inst_idx],
            #     lbl,
            #     fundus_key=fundus_key,
            #     image_size=image_size,
            #     patch_size=patch_size,
            #     mean_std=image_mean_std,
            # )

            heat, base_img, pred = gradcam_fundus_conv(
                model,
                samples[inst_idx],
                lbl,
                fundus_key=fundus_key,
                mean_std=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                debug=True,
            )
            print("heat:", heat.shape, heat.min(), heat.max())
            # heat, base_img, pred = _fundus_gradcam_for_label_from_tokens(
            #     model,
            #     samples[inst_idx],
            #     lbl,
            #     fundus_key=fundus_key,
            #     image_size=image_size,
            #     mean_std=image_mean_std,
            #     debug=False,
            # )
            fig = _overlay_heatmap(
                base_img, heat, alpha=overlay_alpha, cmap=overlay_cmap
            )
            fig.suptitle(lbl.replace("_", " "), y=0.995, fontsize=12)
            out_path = os.path.join(
                out_dir,
                f"{prefix}inst{inst_idx}__{label_safe}__{_safe_name(fundus_key)}_overlay.{img_format}",
            )
            fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.25)
            plt.close(fig)

            saved["fundus"].append(out_path)

        # save logits and samples as txt
        pred_out_path = os.path.join(
            out_dir,
            f"{prefix}inst{inst_idx}__{label_safe}__{_safe_name(fundus_key)}_pred.txt",
        )
        with open(pred_out_path, "w") as f:
            f.write(f"Predictions for label '{lbl}':\n")
            for k, v in pred.items():
                f.write(f"{k}: {F.sigmoid(v).item()}\n")
            f.write("\nSample data:\n")
            for key, value in samples[inst_idx].items():
                f.write(f"{key}: {value}\n")

        # --- other modalities (and cache for cross-modal row) ---
        if other_modalities_gradcam:
            for mod in mods_to_do:
                if mod not in model.input_to_seq:
                    continue
                # raw (unnormalized) CAM for cross-modal fairness
                try:
                    cam_raw = gradcam_tokens_for_label_v2(
                        model, samples[inst_idx], lbl, mod, normalize=False
                    )
                except Exception as e:
                    print(f"[warn] {mod}: Grad-CAM failed ({e})")
                    continue

                # Save per-modality visualization (1-D row or grid)
                pretty_title = f"{lbl.replace('_',' ')} → {mod}"
                mod_safe = _safe_name(mod)
                out_path = os.path.join(
                    out_dir,
                    f"{prefix}inst{inst_idx}__{label_safe}__{mod_safe}.{img_format}",
                )

                if modality_token_grids and mod in modality_token_grids:
                    save_modality_grid(
                        cam_raw,
                        modality_token_grids[mod],
                        title=pretty_title,
                        out_path=out_path,
                        dpi=dpi,
                    )
                else:
                    _save_modality_row_with_annotations(
                        cam_1d=cam_raw
                        / (cam_raw.max() + 1e-12),  # normalize for display
                        modality_name=mod,
                        modality_segments=modality_segments,
                        MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                        title=pretty_title,
                        out_path=out_path,
                        dpi=dpi,
                        annotate_tokens=annotate_tokens,
                        annot_which_modalities=[mod],
                        annot_max_tokens_per_mod=annot_max_tokens_per_mod,
                        annot_rotation=annot_rotation,
                        annot_fontsize=annot_fontsize,
                        annot_color=annot_color,
                        annot_pad_axes_y=annot_pad_axes_y,
                        annot_tick_len_axes=annot_tick_len_axes,
                        annot_wrap_width=annot_wrap_width,
                    )

                saved["modalities"].append(out_path)

                # cache for cross-modal row (raw values)
                cams_by_mod[mod] = cam_raw

                # token names for annotation in the big row
                if MODALITIES_TO_COLS and mod in MODALITIES_TO_COLS:
                    names_by_mod[mod] = MODALITIES_TO_COLS[mod]
                else:
                    names_by_mod[mod] = [f"{mod}_t{i+1}" for i in range(len(cam_raw))]

        # --- emit concatenated cross-modal row once per label ---
        if emit_cross_modal_row and cams_by_mod:
            concat_order = [
                m for m in mods_to_do if m in cams_by_mod and len(cams_by_mod[m]) > 0
            ]
            title = f"All tokens across modalities : {lbl.replace('_',' ')}"
            out_row = os.path.join(
                out_dir,
                f"{prefix}inst{inst_idx}__{label_safe}__token_importance_row.{img_format}",
            )
            save_cross_modal_token_row_with_annotations(
                cams_by_mod=cams_by_mod,
                modality_order=concat_order,
                token_names_by_mod=names_by_mod,
                title=title,
                out_path=out_row,
                dpi=dpi,
                cmap=cross_modal_row_cmap,
                normalize_per_mod_for_display=True,
                annot_color="white",  # readable on magma
                annot_rotation=90,
                annot_fontsize=cross_modal_row_fontsize,
                annot_wrap_width=34,
            )
            saved["cross_modal_row"].append(out_row)

    print(
        f"\nSaved {len(saved['fundus'])} fundus + {len(saved['modalities'])} modality CAM(s)"
        f" + {len(saved['cross_modal_row'])} cross-modal row(s) to '{out_dir}'."
    )
    return saved


def export_all_to_images(
    viz,
    inst_idx=0,
    *,
    reduce="mean",
    MODALITIES_TO_COLS=None,
    token_modalities_to_show=None,
    disease: str | None = None,
    horizon: int | None = None,
    top_k_features: int = 30,
    dpi: int = 300,
    include_per_modality_heatmaps: bool = True,
    modality_subset: list[str] | None = None,
    # fundus overlays
    include_fundus_overlays: bool = True,
    include_overall_fundus_overlay: bool = True,
    include_per_label_fundus_overlays: bool = True,
    samples=None,
    fundus_key: str = "fundus_image",
    image_size: int = 224,
    patch_size: int = 16,
    image_mean_std: tuple | None = None,
    overlay_alpha: float = 0.45,
    overlay_cmap: str = "magma",
    # attention backend
    attn_mode: str = "rollout",  # "rollout" | "attn_x_grad"
    model=None,  # required if attn_mode="attn_x_grad"
    attn_head_fusion: str = "max",
    attn_add_identity: bool = False,
    # new outputs
    out_dir: str = "attention_images",
    img_format: str = "png",  # "png" | "jpg" | "pdf" (vector) | etc.
    filename_prefix: str | None = None,
):
    """
    Save an *image-per-figure* report (instead of a single PDF) with attention visualizations.

    Mirrors `export_all_to_pdf` but writes files to `out_dir`:
      - Token heatmaps per label
      - Per-modality heatmaps (optional)
      - Per-label fundus overlays (optional)
      - Overall fundus overlay (optional)
      - Per-label aggregated modality bars
      - Disease-aggregated rows & bars (when `disease` is specified)

    Returns:
      {
        "token_heatmaps": [...],
        "per_modality": [...],
        "fundus_per_label": [...],
        "fundus_overall": [...],  # length 0 or 1
        "modality_bars": [...],
        "disease_rows": [...],
        "disease_per_modality": [...],
        "topk_feature_bars": [...],
      }
    """
    os.makedirs(out_dir, exist_ok=True)
    prefix = (filename_prefix or "").strip()
    if prefix and not prefix.endswith("_"):
        prefix += "_"

    label_dict = viz["label_start_end"][inst_idx]
    per_label_mod_map = viz["per_instance"][inst_idx]
    modality_segments, _ = build_segments_from_viz(
        viz, inst_idx=inst_idx, exclude_brackets=True
    )
    label_segments = build_label_segments_from_viz(viz, inst_idx=inst_idx)

    name2seg = {m["name"]: m for m in modality_segments}

    def _mods_iter():
        if modality_subset:
            return [name2seg[m] for m in modality_subset if m in name2seg]
        return modality_segments

    def _save_fig(fig, path, *, top=None, bottom=None):
        if top is not None or bottom is not None:
            try:
                fig.tight_layout()
                if top is not None or bottom is not None:
                    fig.subplots_adjust(
                        top=(top if top is not None else 0.9),
                        bottom=(bottom if bottom is not None else 0.2),
                    )
            except Exception:
                # Some complex layouts may not cooperate with tight_layout
                pass
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.28)
        plt.close(fig)

    # cache to avoid recomputing Attn×Grad per label
    attn_cache = {}

    def _R_for_label(label_key: str | None):
        if attn_mode == "attn_x_grad" and label_key is not None:
            if label_key not in attn_cache:
                assert (
                    model is not None and samples is not None
                ), "model and samples are required for attn_x_grad"
                batch_1 = [samples[inst_idx]]
                R = attention_x_gradient_rollout(
                    model,
                    batch_1,
                    label_key,
                    head_fusion=attn_head_fusion,
                    add_identity=attn_add_identity,
                )[
                    0
                ]  # (L,L)
                attn_cache[label_key] = R
            return attn_cache[label_key]
        # fallback: global rollout
        return viz["rollout"][inst_idx]

    # manifest of all saved files
    saved = {
        "token_heatmaps": [],
        "per_modality": [],
        "fundus_per_label": [],
        "fundus_overall": [],
        "modality_bars": [],
        "disease_rows": [],
        "disease_per_modality": [],
        "topk_feature_bars": [],
    }

    # Determine which token modalities to show (pass-through to your plotting fn)
    token_mods = token_modalities_to_show
    if token_mods is None and MODALITIES_TO_COLS:
        token_mods = list(MODALITIES_TO_COLS.keys())

    if disease is None:
        # ============ ALL LABELS ============
        for lbl, span in label_dict.items():
            lbl_pretty = lbl.replace("_", " ")
            lbl_safe = _safe_name(lbl)

            # Full sequence heatmap
            R = _R_for_label(lbl)
            fig = plot_token_heatmap_for_label_with_brackets(
                R,
                span,
                modality_segments=modality_segments,
                label_segments=label_segments,
                highlight_label_key=lbl,
                MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                token_modalities_to_show=token_mods,
                figsize=_autosize_from_modalities(modality_segments),
                dpi=dpi,
                show=False,
                return_fig=True,
            )
            fig.suptitle(lbl_pretty, y=0.985, fontsize=11)
            out_path = os.path.join(
                out_dir,
                f"{prefix}inst{inst_idx}__{lbl_safe}__token_heatmap.{img_format}",
            )
            _save_fig(fig, out_path, top=0.90, bottom=0.24)
            saved["token_heatmaps"].append(out_path)

            # Per-modality heatmaps
            if include_per_modality_heatmaps:
                for seg in _mods_iter():
                    s, e = int(seg["start"]), int(seg["end"])
                    if e <= s:
                        continue
                    seg_name = _safe_name(seg["name"])
                    fig = plot_label_to_modality_heatmap(
                        R,
                        span,
                        seg,
                        MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                        rotation=90,
                        dpi=dpi,
                        show=False,
                        return_fig=True,
                        title=f"{lbl_pretty} → {seg['name']}",
                    )
                    out_path = os.path.join(
                        out_dir,
                        f"{prefix}inst{inst_idx}__{lbl_safe}__to_{seg_name}.{img_format}",
                    )
                    _save_fig(fig, out_path, top=0.88, bottom=0.20)
                    saved["per_modality"].append(out_path)

            # Per-label fundus overlay
            if (
                include_fundus_overlays
                and include_per_label_fundus_overlays
                and samples is not None
            ):
                fig = plot_fundus_attention_overlay(
                    viz,
                    inst_idx=inst_idx,
                    samples=samples,
                    label_key=lbl,
                    reduce=reduce,
                    fundus_key=fundus_key,
                    image_size=image_size,
                    patch_size=patch_size,
                    image_mean_std=image_mean_std,
                    alpha=overlay_alpha,
                    cmap=overlay_cmap,
                    show=False,
                )
                fig.suptitle(f"Fundus overlay: {lbl_pretty}", y=0.985, fontsize=11)
                out_path = os.path.join(
                    out_dir, f"{prefix}inst{inst_idx}__{lbl_safe}__fundus.{img_format}"
                )
                _save_fig(fig, out_path, top=0.92, bottom=0.02)
                saved["fundus_per_label"].append(out_path)

        # Aggregated modality bars per label
        for lbl, md in per_label_mod_map.items():
            lbl_safe = _safe_name(lbl)
            items = sorted(md.items(), key=lambda kv: kv[1], reverse=True)
            mods, vals = zip(*items) if items else ([], [])
            fig = plt.figure(figsize=(max(8, 0.6 * max(1, len(items))), 3.2), dpi=dpi)
            ax = fig.add_subplot(111)
            ax.bar(mods, vals)
            ax.set_xticklabels(mods, rotation=90, ha="right")
            ax.set_ylabel("Attention (rollout, aggregated)")
            ax.set_title(f"{lbl} → modalities")
            out_path = os.path.join(
                out_dir,
                f"{prefix}inst{inst_idx}__{lbl_safe}__modality_bars.{img_format}",
            )
            _save_fig(fig, out_path, top=0.90, bottom=0.20)
            saved["modality_bars"].append(out_path)

        # Disease-aggregated rows
        groups = group_labels_by_disease(label_dict)
        for disease_name, items in groups.items():
            spans = [span for _, span in items]
            R = _R_for_label(None)  # global rollout
            fig = plot_disease_aggregated_row(
                R,
                spans,
                disease=disease_name,
                reduce=reduce,
                figsize=_autosize_row_from_modalities(modality_segments),
                show=False,
                return_fig=True,
            )
            out_path = os.path.join(
                out_dir,
                f"{prefix}inst{inst_idx}__{_safe_name(disease_name)}__row.{img_format}",
            )
            _save_fig(fig, out_path, top=0.88, bottom=0.22)
            saved["disease_rows"].append(out_path)

        # Overall fundus overlay
        if (
            include_fundus_overlays
            and include_overall_fundus_overlay
            and samples is not None
        ):
            fig = plot_fundus_attention_overlay(
                viz,
                inst_idx=inst_idx,
                samples=samples,
                label_key=None,
                reduce=reduce,
                fundus_key=fundus_key,
                image_size=image_size,
                patch_size=patch_size,
                image_mean_std=image_mean_std,
                alpha=overlay_alpha,
                cmap=overlay_cmap,
                show=False,
            )
            fig.suptitle("Fundus overlay (all labels pooled)", y=0.985, fontsize=11)
            out_path = os.path.join(
                out_dir, f"{prefix}inst{inst_idx}__fundus_overall.{img_format}"
            )
            _save_fig(fig, out_path, top=0.92, bottom=0.02)
            saved["fundus_overall"].append(out_path)

    else:
        # ============ DISEASE-SPECIFIC ============
        keys = find_label_keys(label_dict, disease=disease, horizon=horizon)
        if not keys:
            raise KeyError(f"No labels for disease='{disease}', horizon={horizon}")
        spans = [label_dict[k] for k in keys]
        pretty = f"{disease} ({'all horizons' if horizon is None else f'{horizon}y'})"

        for k in keys:
            k_pretty = k.replace("_", " ")
            k_safe = _safe_name(k)

            R = _R_for_label(k)
            fig = plot_token_heatmap_for_label_with_brackets(
                R,
                label_dict[k],
                modality_segments=modality_segments,
                label_segments=label_segments,
                highlight_label_key=k,
                MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                token_modalities_to_show=token_mods,
                figsize=_autosize_from_modalities(modality_segments),
                dpi=dpi,
                show=False,
                return_fig=True,
            )
            fig.suptitle(k_pretty, y=0.985, fontsize=11)
            out_path = os.path.join(
                out_dir, f"{prefix}inst{inst_idx}__{k_safe}__token_heatmap.{img_format}"
            )
            _save_fig(fig, out_path, top=0.90, bottom=0.24)
            saved["token_heatmaps"].append(out_path)

            if (
                include_fundus_overlays
                and include_per_label_fundus_overlays
                and samples is not None
            ):
                fig = plot_fundus_attention_overlay(
                    viz,
                    inst_idx=inst_idx,
                    samples=samples,
                    label_key=k,
                    reduce=reduce,
                    fundus_key=fundus_key,
                    image_size=image_size,
                    patch_size=patch_size,
                    image_mean_std=image_mean_std,
                    alpha=overlay_alpha,
                    cmap=overlay_cmap,
                    show=False,
                )
                fig.suptitle(f"Fundus overlay: {k_pretty}", y=0.985, fontsize=11)
                out_path = os.path.join(
                    out_dir, f"{prefix}inst{inst_idx}__{k_safe}__fundus.{img_format}"
                )
                _save_fig(fig, out_path, top=0.92, bottom=0.02)
                saved["fundus_per_label"].append(out_path)

        # pooled disease vector + modality maps
        R = _R_for_label(keys[0])  # (optionally average across keys instead)
        v = label_conditioned_vector(R, spans, reduce=reduce)

        # per-modality rows (vector→modality)
        if include_per_modality_heatmaps:
            for seg in _mods_iter():
                s, e = int(seg["start"]), int(seg["end"])
                if e <= s:
                    continue
                seg_name = _safe_name(seg["name"])
                fig = plot_vector_to_modality_row(
                    v,
                    seg,
                    MODALITIES_TO_COLS=MODALITIES_TO_COLS,
                    rotation=90,
                    dpi=dpi,
                    show=False,
                    return_fig=True,
                    title=f"{pretty} → {seg['name']}",
                )
                out_path = os.path.join(
                    out_dir,
                    f"{prefix}inst{inst_idx}__{_safe_name(disease)}__to_{seg_name}.{img_format}",
                )
                _save_fig(fig, out_path, top=0.88, bottom=0.18)
                saved["disease_per_modality"].append(out_path)

        # bar chart of modalities (vector aggregated)
        md = aggregate_vector_to_modalities(v, modality_segments, mode="sum")
        items = sorted(md.items(), key=lambda kv: kv[1], reverse=True)
        mods, vals = zip(*items) if items else ([], [])
        fig = plt.figure(figsize=(max(8, 0.6 * max(1, len(items))), 3.2), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.bar(mods, vals)
        ax.set_xticklabels(mods, rotation=90, ha="right")
        ax.set_ylabel("Attention (sum)")
        ax.set_title(f"{pretty} → modalities")
        out_path = os.path.join(
            out_dir,
            f"{prefix}inst{inst_idx}__{_safe_name(disease)}__modality_bars.{img_format}",
        )
        _save_fig(fig, out_path, top=0.90, bottom=0.20)
        saved["modality_bars"].append(out_path)

        # bar chart of top-k features (vector aggregated)
        feats = aggregate_vector_to_features(
            v,
            modality_segments,
            MODALITIES_TO_COLS or {},
            target_modalities=modality_subset,
            mode="sum",
            on_mismatch="warn",
        )
        items = sorted(feats.items(), key=lambda kv: kv[1], reverse=True)[
            :top_k_features
        ]
        names, vals = zip(*items) if items else ([], [])
        fig = plt.figure(figsize=(10, max(3.2, 0.22 * max(1, len(items)))), dpi=dpi)
        ax = fig.add_subplot(111)
        ax.barh(range(len(items)), vals)
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels([n.replace("_", " ") for n in names])
        ax.invert_yaxis()
        ax.set_xlabel("Attention (sum)")
        ax.set_title(f"{pretty} → top {top_k_features} features")
        out_path = os.path.join(
            out_dir,
            f"{prefix}inst{inst_idx}__{_safe_name(disease)}__topk_features.{img_format}",
        )
        _save_fig(fig, out_path, top=0.92, bottom=0.20)
        saved["topk_feature_bars"].append(out_path)

        # overall fundus overlay (for the disease slice)
        if (
            include_fundus_overlays
            and include_overall_fundus_overlay
            and samples is not None
        ):
            fig = plot_fundus_attention_overlay(
                viz,
                inst_idx=inst_idx,
                samples=samples,
                label_key=None,
                reduce=reduce,
                fundus_key=fundus_key,
                image_size=image_size,
                patch_size=patch_size,
                image_mean_std=image_mean_std,
                alpha=overlay_alpha,
                cmap=overlay_cmap,
                show=False,
            )
            fig.suptitle(f"Fundus overlay: {pretty}", y=0.985, fontsize=11)
            out_path = os.path.join(
                out_dir,
                f"{prefix}inst{inst_idx}__{_safe_name(disease)}__fundus_overall.{img_format}",
            )
            _save_fig(fig, out_path, top=0.92, bottom=0.02)
            saved["fundus_overall"].append(out_path)

    print(
        f"Saved images to '{out_dir}'. Counts: "
        f"token_heatmaps={len(saved['token_heatmaps'])}, "
        f"per_modality={len(saved['per_modality'])}, "
        f"fundus_per_label={len(saved['fundus_per_label'])}, "
        f"fundus_overall={len(saved['fundus_overall'])}, "
        f"modality_bars={len(saved['modality_bars'])}, "
        f"disease_rows={len(saved['disease_rows'])}, "
        f"disease_per_modality={len(saved['disease_per_modality'])}, "
        f"topk_feature_bars={len(saved['topk_feature_bars'])}"
    )
    return saved
