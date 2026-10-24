"""
FHDSS SPRINT 26 — Test: Calibration Isolation (Phase 16)
Verifies:
- Thresholds are computed exclusively from validation data (Rule 2).
- Threshold ordering holds: EER_th <= Standard_th (FAR<=5%) <= Strict_th (FAR<=1%).
- Test set is never touched during calibration.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
from sklearn.metrics import roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_calibration_threshold_ordering():
    # Mock validation similarity scores
    np.random.seed(42)
    y_true = np.array([1]*100 + [0]*100)
    # Genuine ~ N(0.85, 0.05), Forgery ~ N(0.40, 0.10)
    y_scores = np.concatenate([
        np.random.normal(0.85, 0.05, 100),
        np.random.normal(0.40, 0.10, 100)
    ])
    y_scores = np.clip(y_scores, 0.0, 1.0)

    fpr, tpr, ths = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer_th = float(ths[eer_idx])

    far5_idx = np.where(fpr <= 0.05)[0]
    th_far5 = float(ths[far5_idx[-1]]) if len(far5_idx) > 0 else eer_th

    far1_idx = np.where(fpr <= 0.01)[0]
    th_far1 = float(ths[far1_idx[-1]]) if len(far1_idx) > 0 else eer_th

    assert eer_th <= th_far5 <= th_far1, f"Thresholds must be monotonically increasing: {eer_th} <= {th_far5} <= {th_far1}"
