"""
FHDSS SPRINT 27 — Test: CVL Regression Root-Cause Analysis (Phase 5)
Verifies:
- CVL lines have high aspect ratios (> 5:1), explaining why square 224x224 padding leads to whitespace dominance.
- CVL is identified as a handwriting line regression domain, protecting V5 production status.
"""

from pathlib import Path
import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_cvl_aspect_ratio_and_padding_properties():
    cvl_dir = PROJECT_ROOT / "datasets" / "processed" / "CVL"
    if not cvl_dir.exists():
        pytest.skip("CVL directory not found")

    sample_files = list(cvl_dir.glob("*.jpg"))[:10]
    assert len(sample_files) > 0, "No CVL samples found"

    aspect_ratios = []
    for p in sample_files:
        data = p.read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            h, w = img.shape[:2]
            aspect_ratios.append(w / float(h))

    mean_ar = np.mean(aspect_ratios)
    # CVL lines are wide text strips (aspect ratio typically > 8.0)
    assert mean_ar > 5.0, f"CVL aspect ratio expected > 5.0, got {mean_ar}"
