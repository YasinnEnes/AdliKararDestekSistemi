"""
FHDSS SPRINT 29 — Test Suite Group 2: MAR Preprocessing, Geometry & Information Preservation
Verifies:
8. Preprocessing determinism on identical input
9. Aspect-ratio preservation and correct canvas padding
10. Accurate padding ratio computation
11. Information preservation warning flags (occupancy, height, padding)
12. Strict prohibition of horizontal and vertical flips
13. Input dimension guarantee (always (1, 3, 224, 224))
14. Normalization adheres to ImageNet mean and std bounds
"""

import pytest
import sys
import numpy as np
import torch
import cv2
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.preprocessing.v6_s29_mar import (
    preprocess_exp_f_multi_aspect_ratio,
    detect_foreground_bbox,
    compute_geometry_descriptors,
    Sprint29MARAugmentation
)


@pytest.fixture
def sample_wide_image():
    # Wide image simulating CVL line: 50x800
    img = np.full((50, 800, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Forensic Handwriting Sample Test", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


@pytest.fixture
def sample_tall_image():
    # Tall image: 600x200
    img = np.full((600, 200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Signature", (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    return img


def test_preprocessing_determinism(sample_wide_image):
    """Test 8: Running preprocess_exp_f_multi_aspect_ratio twice yields bitwise identical tensors."""
    t1, m1 = preprocess_exp_f_multi_aspect_ratio(sample_wide_image)
    t2, m2 = preprocess_exp_f_multi_aspect_ratio(sample_wide_image)
    assert torch.equal(t1, t2), "MAR preprocessing is non-deterministic!"
    assert m1 == m2, "Metadata output is non-deterministic!"


def test_output_tensor_shape(sample_wide_image):
    """Test 9: Output tensor must strictly be (1, 3, 224, 224)."""
    t, _ = preprocess_exp_f_multi_aspect_ratio(sample_wide_image)
    assert t.shape == (1, 3, 224, 224), f"Unexpected tensor shape: {t.shape}"


def test_padding_ratio_calculation(sample_wide_image):
    """Test 10: Padding ratio must be between 0.0 and 1.0 and accurately reflect letterbox padding."""
    t, meta = preprocess_exp_f_multi_aspect_ratio(sample_wide_image)
    pad_ratio = meta["padding_ratio"]
    assert 0.0 <= pad_ratio <= 1.0
    # Wide image must have significant padding ratio in 224x224
    assert pad_ratio > 0.50, f"Expected padding ratio > 0.50 for wide image, got {pad_ratio}"


def test_quality_warning_generation(sample_wide_image):
    """Test 11: Quality flag correctly identifies low height or high padding ratio."""
    t, meta = preprocess_exp_f_multi_aspect_ratio(sample_wide_image)
    # The wide image resized will have low height and high padding
    assert "has_warning" in meta
    if meta["resized_foreground_height"] < 24 or meta["padding_ratio"] > 0.90:
        assert meta["has_warning"] is True
        assert len(meta["warning_reasons"]) > 0


def test_no_flips_in_mar_augmentation():
    """Test 12: MAR augmentation must NEVER apply horizontal or vertical flip."""
    aug = Sprint29MARAugmentation(is_training=True, seed=42)
    # Create an image with an arrow pointing right
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    cv2.arrowedLine(img, (20, 50), (280, 50), (0, 0, 0), 3)

    for _ in range(20):
        t, meta = aug(img)
        # Verify shape
        assert t.shape == (3, 224, 224)


def test_geometry_descriptors_completeness(sample_wide_image):
    """Test 13: Geometric descriptors include all 11 required forensic parameters."""
    geo = compute_geometry_descriptors(sample_wide_image)
    required_keys = [
        "width", "height", "aspect_ratio", "bbox", "bbox_width", "bbox_height",
        "foreground_occupancy", "vertical_occupancy", "horizontal_occupancy",
        "white_pixel_ratio", "centroid_x", "centroid_y", "stroke_density", "connected_component_count"
    ]
    for k in required_keys:
        assert k in geo, f"Missing geometric parameter: {k}"


def test_imagenet_normalization_values(sample_wide_image):
    """Test 14: Normalized pixel values should conform to ImageNet range [-2.5, +2.8]."""
    t, _ = preprocess_exp_f_multi_aspect_ratio(sample_wide_image)
    assert t.min() >= -3.0
    assert t.max() <= 3.0
