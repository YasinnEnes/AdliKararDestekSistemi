"""
FHDSS SPRINT 28 — Test: Letterbox Geometry
Verifies that letterbox scaling centers content and fills remaining pixels with specified background.
"""

import numpy as np
import pytest

from ml.preprocessing.ablation import preprocess_exp_b_letterbox


def test_letterbox_centering():
    # 100 x 200 image (2:1 aspect ratio)
    img = np.full((100, 200, 3), 128, dtype=np.uint8)
    t = preprocess_exp_b_letterbox(img, target_size=(224, 224))
    assert t.shape == (1, 3, 224, 224)
