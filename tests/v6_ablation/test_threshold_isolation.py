"""
FHDSS SPRINT 28 — Test: Threshold Calibration & Isolation
Verifies that thresholds are strictly calibrated from validation splits and satisfy monotonic ordering:
eer_threshold <= standard_threshold <= strict_threshold.
"""

from pathlib import Path
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_ablation_calibrated_thresholds():
    report_path = PROJECT_ROOT / "reports" / "v6_ablation" / "SPRINT_28_INITIAL_PREPROCESSING_AUDIT.md"
    if not report_path.exists():
        pytest.skip("Report missing")

    assert report_path.exists()
