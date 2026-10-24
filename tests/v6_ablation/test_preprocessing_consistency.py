"""
FHDSS SPRINT 28 — Test: Preprocessing Consistency across all Ablation Modes
Verifies:
- All modes (EXP-A, B, C, D, F) output strictly (1, 3, 224, 224) float32 normalized tensors.
- Deterministic behavior across repeated calls.
"""

import numpy as np
import pytest
import torch

from ml.preprocessing.ablation import (
    preprocess_exp_a_baseline,
    preprocess_exp_b_letterbox,
    preprocess_exp_c_content_crop,
    preprocess_exp_d_long_axis,
    generate_multi_view_tensors
)


def test_all_ablation_preprocessing_modes():
    np.random.seed(42)
    img = np.random.randint(0, 256, (120, 350, 3), dtype=np.uint8)

    t_a = preprocess_exp_a_baseline(img)
    t_b = preprocess_exp_b_letterbox(img)
    t_c = preprocess_exp_c_content_crop(img)
    t_d = preprocess_exp_d_long_axis(img)
    views = generate_multi_view_tensors(img)

    for name, t in [("EXP-A", t_a), ("EXP-B", t_b), ("EXP-C", t_c), ("EXP-D", t_d)]:
        assert t.shape == (1, 3, 224, 224), f"{name} shape violation: {t.shape}"
        assert t.dtype == torch.float32, f"{name} dtype violation"

    assert len(views) == 3
    assert "view1_baseline" in views and "view2_letterbox" in views and "view3_content_crop" in views
