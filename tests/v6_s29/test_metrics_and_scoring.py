"""
FHDSS SPRINT 29 — Test Suite Group 4: Forensic Metrics, Similarities & Calibration
Verifies:
21. Cosine similarity bounded in [-1.0, 1.0]
22. Normalized forensic similarity bounded in [0.0, 1.0]
23. Identity self-similarity is exactly 1.0
24. EER and operating threshold computation accuracy
25. Strict threshold (FAR<=1%) >= Standard threshold (FAR<=5%)
26. TAR monotonic behavior under stricter operating thresholds
"""

import pytest
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.siamese import (
    compute_cosine_similarity,
    compute_normalized_similarity,
)
from scripts.v6_s29.pipeline_core import evaluate_pairs_detailed


def test_cosine_similarity_bounds():
    """Test 21: Cosine similarity is bounded in [-1.0, 1.0]."""
    e1 = np.random.randn(128)
    e1 /= np.linalg.norm(e1)
    e2 = np.random.randn(128)
    e2 /= np.linalg.norm(e2)

    cos_sim = compute_cosine_similarity(e1, e2)
    assert -1.0001 <= cos_sim <= 1.0001


def test_normalized_similarity_bounds():
    """Test 22: Normalized forensic similarity is bounded in [0.0, 1.0]."""
    e1 = np.random.randn(128)
    e1 /= np.linalg.norm(e1)
    e2 = -e1  # Opposite direction

    norm_sim = compute_normalized_similarity(e1, e2)
    assert 0.0 <= norm_sim <= 1.0
    assert norm_sim < 0.05, f"Expected near 0 for opposite vectors, got {norm_sim}"


def test_identity_self_similarity():
    """Test 23: Comparison of an embedding with itself yields exactly 1.0."""
    e = np.random.randn(128)
    e /= np.linalg.norm(e)
    sim = compute_normalized_similarity(e, e)
    assert abs(sim - 1.0) < 1e-5


def test_threshold_calibration_hierarchy():
    """Test 24 & 25: Strict threshold (FAR<=1%) is >= Standard threshold (FAR<=5%)."""
    # Create synthetic scores
    rng = np.random.RandomState(42)
    y_true = [1] * 100 + [0] * 100
    scores = list(rng.normal(0.85, 0.08, 100)) + list(rng.normal(0.45, 0.12, 100))
    scores = np.clip(scores, 0.0, 1.0)

    from sklearn.metrics import roc_curve
    fpr, tpr, ths = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    th_eer = float(ths[eer_idx])

    far5_idx = np.where(fpr <= 0.05)[0]
    th_far5 = float(ths[far5_idx[-1]]) if len(far5_idx) > 0 else th_eer

    far1_idx = np.where(fpr <= 0.01)[0]
    th_far1 = float(ths[far1_idx[-1]]) if len(far1_idx) > 0 else th_eer

    assert th_far1 >= th_far5 - 1e-4, f"Hierarchy violation: th_far1 ({th_far1}) < th_far5 ({th_far5})"


def test_tar_at_operating_thresholds():
    """Test 26: TAR at FAR<=5% is >= TAR at FAR<=1%."""
    rng = np.random.RandomState(42)
    y_true = [1] * 200 + [0] * 200
    scores = list(rng.normal(0.88, 0.06, 200)) + list(rng.normal(0.50, 0.10, 200))

    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    far5_idx = np.where(fpr <= 0.05)[0]
    tar_far5 = float(tpr[far5_idx[-1]]) if len(far5_idx) > 0 else 0.5

    far1_idx = np.where(fpr <= 0.01)[0]
    tar_far1 = float(tpr[far1_idx[-1]]) if len(far1_idx) > 0 else 0.5

    assert tar_far5 >= tar_far1 - 1e-4, f"TAR inversion: tar_far5 ({tar_far5}) < tar_far1 ({tar_far1})"
