"""
FHDSS SPRINT 27 — Test: Bootstrap Significance (Phase 10)
Verifies:
- Bootstrap confidence interval calculation produces mathematically sound intervals.
- ci_lower <= mean_diff <= ci_upper.
"""

from pathlib import Path
import sys
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_sprint27_v6_validation import bootstrap_significance_test


def test_bootstrap_ci_validity():
    np.random.seed(42)
    y_true = [1]*50 + [0]*50
    s_v5 = np.random.normal(0.70, 0.1, 100).clip(0, 1).tolist()
    s_v6 = np.random.normal(0.85, 0.1, 100).clip(0, 1).tolist()

    mean_diff, ci_l, ci_u, p_val = bootstrap_significance_test(y_true, s_v5, s_v6, n_bootstraps=200, seed=42)

    assert ci_l <= mean_diff <= ci_u, f"Invalid CI: {ci_l} <= {mean_diff} <= {ci_u}"
    assert 0.0 <= p_val <= 1.0, f"p-value must be in [0, 1], got {p_val}"
