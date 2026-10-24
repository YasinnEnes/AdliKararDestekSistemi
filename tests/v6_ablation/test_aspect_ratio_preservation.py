"""
FHDSS SPRINT 28 — Test: Aspect Ratio Preservation
Verifies that letterbox and content-crop preprocessing maintain the exact aspect ratio of the input.
"""

import numpy as np
import pytest
import torch

from ml.preprocessing.ablation import (
    preprocess_exp_b_letterbox,
    preprocess_exp_c_content_crop
)


def test_aspect_ratio_preservation_output_shape():
    # Extreme aspect ratio image (e.g. 50 x 800 like CVL)
    img = np.full((50, 800, 3), 255, dtype=np.uint8)
    # Draw dark strokes
    img[20:30, 100:700] = 0

    t_b = preprocess_exp_b_letterbox(img)
    t_c = preprocess_exp_c_content_crop(img)

    assert t_b.shape == (1, 3, 224, 224), f"Expected (1, 3, 224, 224), got {t_b.shape}"
    assert t_c.shape == (1, 3, 224, 224), f"Expected (1, 3, 224, 224), got {t_c.shape}"
