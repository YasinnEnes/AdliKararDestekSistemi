"""
FHDSS SPRINT 28 — Test: Content-Aware Crop (EXP-C)
Verifies:
- find_ink_bounding_box correctly isolates ink from white document background.
- Margin padding prevents cutting off edge strokes.
"""

import numpy as np
import pytest

from ml.preprocessing.ablation import find_ink_bounding_box, preprocess_exp_c_content_crop


def test_ink_bounding_box_detection():
    # 200 x 300 white image with ink in the center: [50:150, 100:200]
    img = np.full((200, 300, 3), 255, dtype=np.uint8)
    img[50:150, 100:200] = 30  # dark ink

    x1, y1, x2, y2 = find_ink_bounding_box(img)
    assert 95 <= x1 <= 105, f"Expected x1 near 100, got {x1}"
    assert 45 <= y1 <= 55, f"Expected y1 near 50, got {y1}"
    assert 195 <= x2 <= 205, f"Expected x2 near 200, got {x2}"
    assert 145 <= y2 <= 155, f"Expected y2 near 150, got {y2}"


def test_content_crop_tensor_shape():
    img = np.full((150, 600, 3), 255, dtype=np.uint8)
    img[40:110, 100:500] = 20
    t = preprocess_exp_c_content_crop(img)
    assert t.shape == (1, 3, 224, 224)
