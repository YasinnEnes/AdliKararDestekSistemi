"""
FHDSS SPRINT 29 — Multi-Aspect-Ratio (MAR) Preprocessing & Information-Preserving Pipeline
Implements:
- Ink / Foreground Bounding Box with adaptive thresholding
- Safe margin expansion (controlled aspect-ratio preservation)
- Controlled aspect-ratio letterboxing to 224x224
- Forensic metadata tracking:
  * foreground occupancy
  * resized foreground height / width
  * padding ratio
  * quality warning flags (occupancy < 5%, fg_height < 24px, padding_ratio > 90%)
- Multi-Aspect-Ratio Augmentation (small rotation, scale, translation, aspect-ratio jitter, brightness/contrast, noise - NO FLIPS!)
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
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


def detect_foreground_bbox(
    img_rgb: np.ndarray,
    threshold: int = 240
) -> Tuple[int, int, int, int]:
    """
    Finds ink/foreground bounding box (x1, y1, x2, y2).
    Uses Otsu adaptive fallback if standard thresholding detects no foreground.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    mask = (gray < threshold).astype(np.uint8)
    coords = cv2.findNonZero(mask)
    if coords is None:
        # Fallback to Otsu thresholding
        _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(otsu_mask)
        if coords is None:
            return 0, 0, img_rgb.shape[1], img_rgb.shape[0]

    x, y, w, h = cv2.boundingRect(coords)
    return x, y, x + w, y + h


def compute_geometry_descriptors(img_rgb: np.ndarray) -> Dict[str, Any]:
    """
    Computes all forensic geometric parameters required by Section 6 of Sprint 29:
    - width, height, aspect ratio
    - foreground bbox (x1, y1, x2, y2)
    - foreground occupancy (ink pixels / bbox area)
    - vertical occupancy (bbox height / image height)
    - horizontal occupancy (bbox width / image width)
    - white pixel ratio (pixels >= 240 / total image area)
    - foreground centroid (cx, cy normalized to [0, 1])
    - stroke density (ink pixels / total image area)
    - connected component count
    """
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    total_pixels = float(h * w)

    # Ink mask
    ink_mask = (gray < 235).astype(np.uint8)
    ink_pixels = int(np.sum(ink_mask))
    white_pixels = int(np.sum(gray >= 240))

    x1, y1, x2, y2 = detect_foreground_bbox(img_rgb, threshold=240)
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    bbox_area = float(bw * bh)

    # Sub-mask within bbox
    ink_in_bbox = int(np.sum(ink_mask[y1:y2, x1:x2]))
    fg_occupancy = ink_in_bbox / bbox_area if bbox_area > 0 else 0.0
    vert_occupancy = bh / float(h)
    horiz_occupancy = bw / float(w)
    white_pixel_ratio = white_pixels / total_pixels
    stroke_density = ink_pixels / total_pixels

    # Centroid of ink
    moments = cv2.moments(ink_mask)
    if moments["m00"] > 0:
        cx = float(moments["m10"] / moments["m00"]) / float(w)
        cy = float(moments["m01"] / moments["m00"]) / float(h)
    else:
        cx = (x1 + x2) / (2.0 * w)
        cy = (y1 + y2) / (2.0 * h)

    # Connected component count
    num_labels, _ = cv2.connectedComponents(ink_mask)
    cc_count = max(0, num_labels - 1)

    return {
        "width": w,
        "height": h,
        "aspect_ratio": round(w / float(max(1, h)), 4),
        "bbox": [x1, y1, x2, y2],
        "bbox_width": bw,
        "bbox_height": bh,
        "foreground_occupancy": round(float(fg_occupancy), 4),
        "vertical_occupancy": round(float(vert_occupancy), 4),
        "horizontal_occupancy": round(float(horiz_occupancy), 4),
        "white_pixel_ratio": round(float(white_pixel_ratio), 4),
        "centroid_x": round(float(cx), 4),
        "centroid_y": round(float(cy), 4),
        "stroke_density": round(float(stroke_density), 4),
        "connected_component_count": int(cc_count),
    }


def preprocess_exp_f_multi_aspect_ratio(
    image: np.ndarray,
    target_size: Tuple[int, int] = (224, 224),
    margin_ratio: float = 0.08,
    is_bgr: bool = True
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Sprint 29 EXP-F Strategy:
    1. Detect foreground bounding box
    2. Add safe margin (default 8% of width and height)
    3. Crop foreground safely
    4. Aspect-ratio preserving controlled resize into (224, 224)
    5. Symmetric letterbox / padding
    6. Normalize to ImageNet contract
    7. Track forensic quality metrics & flags (Section 8):
       - foreground occupancy
       - resized foreground height
       - resized foreground width
       - padding ratio
       - quality warning flag: (occupancy < 0.05 or fg_height < 24 or padding_ratio > 0.90)
    """
    rgb = to_rgb_safe(image, is_bgr=is_bgr)
    h, w = rgb.shape[:2]
    th, tw = target_size

    x1, y1, x2, y2 = detect_foreground_bbox(rgb, threshold=240)
    bw, bh = x2 - x1, y2 - y1

    # Safe margin
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)
    cx1 = max(0, x1 - mx)
    cy1 = max(0, y1 - my)
    cx2 = min(w, x2 + mx)
    cy2 = min(h, y2 + my)

    cropped = rgb[cy1:cy2, cx1:cx2]
    if cropped.size == 0 or cropped.shape[0] < 5 or cropped.shape[1] < 5:
        cropped = rgb
        cx1, cy1, cx2, cy2 = 0, 0, w, h

    ch, cw = cropped.shape[:2]

    # Aspect-ratio preserving controlled resize
    scale = min(tw / float(cw), th / float(ch))
    nw = max(1, int(round(cw * scale)))
    nh = max(1, int(round(ch * scale)))
    nw = min(tw, nw)
    nh = min(th, nh)

    resized_crop = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_AREA)

    # Canvas creation (letterbox)
    canvas = np.full((th, tw, 3), 255, dtype=np.uint8)
    dx = (tw - nw) // 2
    dy = (th - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized_crop

    # Forensic quality metrics
    canvas_area = float(th * tw)
    content_area = float(nw * nh)
    padding_ratio = 1.0 - (content_area / canvas_area)

    # Gray of canvas to calculate foreground occupancy in input
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_RGB2GRAY)
    ink_pixels_in_canvas = int(np.sum(canvas_gray < 235))
    foreground_occupancy = ink_pixels_in_canvas / canvas_area

    # Quality warning flag check
    has_warning = bool(
        foreground_occupancy < 0.05 or
        nh < 24 or
        padding_ratio > 0.90
    )
    warning_reasons = []
    if foreground_occupancy < 0.05:
        warning_reasons.append("LOW_OCCUPANCY_<5%")
    if nh < 24:
        warning_reasons.append("LOW_HEIGHT_<24PX")
    if padding_ratio > 0.90:
        warning_reasons.append("HIGH_PADDING_>90%")

    meta = {
        "foreground_occupancy": round(float(foreground_occupancy), 4),
        "resized_foreground_height": nh,
        "resized_foreground_width": nw,
        "padding_ratio": round(float(padding_ratio), 4),
        "has_warning": has_warning,
        "warning_reasons": warning_reasons,
        "scale": round(float(scale), 4)
    }

    # Normalize
    t = torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0
    norm = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = norm(t).unsqueeze(0)

    return tensor, meta


class Sprint29MARAugmentation:
    """
    Sprint 29 Multi-Aspect-Ratio Augmentation (Section 9):
    - NO FLIP (strictly prohibited)
    - Rotation: [-5°, +5°]
    - Brightness: 0.90 - 1.10
    - Contrast: 0.90 - 1.10
    - Aspect-ratio jitter: 0.85 - 1.15
    - Scale: 0.90 - 1.10
    - Translation: ±5%
    - Small Gaussian noise: sigma <= 1.5
    - Deterministic with random seed control
    """
    def __init__(self, is_training: bool = True, seed: Optional[int] = None):
        self.is_training = is_training
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.np_rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()
        self.normalizer = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, img_bgr: np.ndarray) -> Tuple[torch.Tensor, Dict[str, Any]]:
        rgb = to_rgb_safe(img_bgr, is_bgr=True)
        h, w = rgb.shape[:2]

        if not self.is_training:
            tensor, meta = preprocess_exp_f_multi_aspect_ratio(rgb, is_bgr=False)
            return tensor.squeeze(0), meta

        processed = rgb.copy()

        # 1. Aspect-Ratio Jitter (0.85 - 1.15) & Scale (0.90 - 1.10)
        ar_jitter = self.rng.uniform(0.85, 1.15)
        scale = self.rng.uniform(0.90, 1.10)
        new_w = max(16, int(round(w * scale * math.sqrt(ar_jitter))))
        new_h = max(16, int(round(h * scale / math.sqrt(ar_jitter))))
        processed = cv2.resize(processed, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        curr_h, curr_w = processed.shape[:2]

        # 2. Rotation [-5°, +5°] & Translation (±5%)
        angle = self.rng.uniform(-5.0, 5.0)
        tx = self.rng.uniform(-0.05 * curr_w, 0.05 * curr_w)
        ty = self.rng.uniform(-0.05 * curr_h, 0.05 * curr_h)

        M = cv2.getRotationMatrix2D((curr_w / 2.0, curr_h / 2.0), angle, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        processed = cv2.warpAffine(
            processed, M, (curr_w, curr_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        # 3. Brightness (0.90 - 1.10) & Contrast (0.90 - 1.10)
        contrast = self.rng.uniform(0.90, 1.10)
        brightness_shift = self.rng.uniform(-10.0, 10.0)
        processed = np.clip(processed.astype(np.float32) * contrast + brightness_shift, 0, 255).astype(np.uint8)

        # 4. Small Gaussian noise (30% probability)
        if self.rng.random() < 0.30:
            noise = self.np_rng.normal(0, 1.5, processed.shape).astype(np.float32)
            processed = np.clip(processed.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Apply EXP-F multi-aspect-ratio formatting to final 224x224
        tensor, meta = preprocess_exp_f_multi_aspect_ratio(processed, is_bgr=False)
        return tensor.squeeze(0), meta
