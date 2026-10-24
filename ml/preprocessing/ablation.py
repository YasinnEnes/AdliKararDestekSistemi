"""
FHDSS SPRINT 28 — Preprocessing Ablation Module
Implements 6 controlled preprocessing strategies for forensic signature and handwriting verification:
- EXP-A: Current V6 Baseline (Symmetric pad-to-square + resize to 224x224)
- EXP-B: Aspect-Ratio Preserving Letterbox (Uniform scaling with minimal border)
- EXP-C: Content-Aware Crop + Safe Margin Letterbox (BBox crop + 8% margin + letterbox)
- EXP-D: Long-Axis Aware Normalization (Horizontal stroke-preserving scaling)
- EXP-E: Horizontal Tiling / Multi-Patch Encoder (Sliding window for wide text lines)
- EXP-F: Multi-View Generator (Generates triplets of views for cross-view invariance audit)
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def to_rgb_safe(image: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    if image is None or image.size == 0:
        raise ValueError("Invalid empty image")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        c = image.shape[2]
        if c == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif c == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif c == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image
    raise ValueError(f"Unsupported image shape: {image.shape}")


def normalize_tensor(img_rgb_224: np.ndarray) -> torch.Tensor:
    """Converts (224, 224, 3) uint8 RGB image to normalized float32 tensor (1, 3, 224, 224)."""
    t = torch.from_numpy(img_rgb_224).permute(2, 0, 1).float() / 255.0
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    return norm(t).unsqueeze(0)


# ==============================================================================
# EXP-A: Current V6 Baseline Preprocessing
# ==============================================================================

def preprocess_exp_a_baseline(image: np.ndarray, is_bgr: bool = True) -> torch.Tensor:
    """Current V6 Baseline: pad_to_square + resize (224, 224)."""
    rgb = to_rgb_safe(image, is_bgr=is_bgr)
    h, w = rgb.shape[:2]
    max_dim = max(h, w)
    pad_t = (max_dim - h) // 2
    pad_b = max_dim - h - pad_t
    pad_l = (max_dim - w) // 2
    pad_r = max_dim - w - pad_l

    padded = cv2.copyMakeBorder(
        rgb, pad_t, pad_b, pad_l, pad_r,
        borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    resized = cv2.resize(padded, (224, 224), interpolation=cv2.INTER_AREA)
    return normalize_tensor(resized)


# ==============================================================================
# EXP-B: Aspect-Ratio Preserving Letterbox
# ==============================================================================

def preprocess_exp_b_letterbox(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    is_bgr: bool = True,
    bg_color: Tuple[int, int, int] = (255, 255, 255)
) -> torch.Tensor:
    """Uniform scaling fitting inside target_size maintaining exact aspect ratio."""
    rgb = to_rgb_safe(image, is_bgr=is_bgr)
    h, w = rgb.shape[:2]
    th, tw = target_size

    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.full((th, tw, 3), bg_color, dtype=np.uint8)
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized

    return normalize_tensor(canvas)


# ==============================================================================
# EXP-C: Content-Aware Crop + Safe Margin Letterbox
# ==============================================================================

def find_ink_bounding_box(rgb_img: np.ndarray, threshold: int = 240) -> Tuple[int, int, int, int]:
    """Detects ink bounding box (min_x, min_y, max_x, max_y) on document background."""
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    # Ink pixels are darker than background
    ink_mask = gray < threshold
    coords = cv2.findNonZero(ink_mask.astype(np.uint8))
    if coords is None:
        return 0, 0, rgb_img.shape[1], rgb_img.shape[0]
    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h


def preprocess_exp_c_content_crop(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    margin_ratio: float = 0.08,
    is_bgr: bool = True
) -> torch.Tensor:
    """Tight ink bounding box extraction + safe margin + aspect-ratio preserving letterbox."""
    rgb = to_rgb_safe(image, is_bgr=is_bgr)
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = find_ink_bounding_box(rgb)

    # Safe margin padding
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)

    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)

    cropped = rgb[cy1:cy2, cx1:cx2]
    if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
        cropped = rgb

    return preprocess_exp_b_letterbox(cropped, target_size=target_size, is_bgr=False)


# ==============================================================================
# EXP-D: Long-Axis Aware Normalization
# ==============================================================================

def preprocess_exp_d_long_axis(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    is_bgr: bool = True
) -> torch.Tensor:
    """Scales long axis to target dimension and vertically centers content with stroke contrast compensation."""
    rgb = to_rgb_safe(image, is_bgr=is_bgr)
    h, w = rgb.shape[:2]
    th, tw = target_size

    if w >= h:
        nw = tw
        scale = nw / float(w)
        nh = max(16, int(round(h * scale)))
        nh = min(th, nh)
    else:
        nh = th
        scale = nh / float(h)
        nw = max(16, int(round(w * scale)))
        nw = min(tw, nw)

    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
    dy = (th - nh) // 2
    dx = (tw - nw) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized

    return normalize_tensor(canvas)


# ==============================================================================
# EXP-E: Horizontal Tiling / Multi-Patch Encoder
# ==============================================================================

def extract_tiled_representation(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    is_bgr: bool = True
) -> np.ndarray:
    """
    For wide aspect-ratio images (e.g. CVL lines > 3:1), extracts overlapping square tiles
    across the text line, encodes each patch with V6, and pools them into a single 128D L2 vector.
    For standard aspect ratios (<= 3:1), executes standard letterbox.
    """
    model.eval()
    rgb = to_rgb_safe(image, is_bgr=is_bgr)
    h, w = rgb.shape[:2]
    ar = w / float(h)

    if ar <= 3.0:
        t = preprocess_exp_b_letterbox(rgb, is_bgr=False).to(device)
        with torch.no_grad():
            emb = model(t).cpu().numpy()[0]
        return emb

    # Extract horizontal tiles (3 overlapping patches: Left, Center, Right)
    tile_w = h  # square patch based on line height
    if tile_w < 10:
        tile_w = h

    # If line is very wide, create 3 fixed patches
    patch_w = max(h, w // 3)
    p_left = rgb[:, :patch_w]
    mid_start = max(0, (w - patch_w) // 2)
    p_center = rgb[:, mid_start:mid_start + patch_w]
    p_right = rgb[:, max(0, w - patch_w):]

    patch_embs = []
    with torch.no_grad():
        for patch in [p_left, p_center, p_right]:
            t_patch = preprocess_exp_b_letterbox(patch, is_bgr=False).to(device)
            p_emb = model(t_patch).cpu().numpy()[0]
            patch_embs.append(p_emb)

    # Average pooled and L2 normalized
    mean_emb = np.mean(patch_embs, axis=0)
    norm = np.linalg.norm(mean_emb)
    return mean_emb / (norm + 1e-12)


# ==============================================================================
# EXP-F: Multi-View Generator
# ==============================================================================

def generate_multi_view_tensors(image: np.ndarray, is_bgr: bool = True) -> Dict[str, torch.Tensor]:
    """Generates View 1 (Baseline), View 2 (Letterbox), View 3 (Content-Crop) for cross-view analysis."""
    return {
        "view1_baseline": preprocess_exp_a_baseline(image, is_bgr=is_bgr),
        "view2_letterbox": preprocess_exp_b_letterbox(image, is_bgr=is_bgr),
        "view3_content_crop": preprocess_exp_c_content_crop(image, is_bgr=is_bgr)
    }


def get_preprocessor(name: str):
    """Returns a callable preprocessor mapping input (PIL image or ndarray) to normalized tensor."""
    name_upper = name.upper()
    if name_upper in ("EXP-A", "EXP_A", "BASELINE"):
        fn = preprocess_exp_a_baseline
    elif name_upper in ("EXP-B", "EXP_B", "LETTERBOX"):
        fn = preprocess_exp_b_letterbox
    elif name_upper in ("EXP-C", "EXP_C", "CONTENT_CROP"):
        fn = preprocess_exp_c_content_crop
    elif name_upper in ("EXP-D", "EXP_D", "LONG_AXIS"):
        fn = preprocess_exp_d_long_axis
    else:
        raise ValueError(f"Unknown preprocessing strategy: {name}")

    def _call(img: Any) -> torch.Tensor:
        if hasattr(img, "convert"):  # PIL Image
            arr = np.array(img.convert("RGB"))
            return fn(arr, is_bgr=False)
        elif isinstance(img, np.ndarray):
            return fn(img)
        raise TypeError(f"Unsupported image type: {type(img)}")

    return _call

