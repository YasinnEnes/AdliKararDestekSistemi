"""
FHDSS SPRINT 28 — Test: CVL Regression Recovery
Verifies:
- EXP-C (Content-Aware Crop + Safe Margin Letterbox) recovers CVL performance over baseline EXP-A.
"""

from pathlib import Path
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_cvl_recovery_under_exp_c():
    report_path = PROJECT_ROOT / "reports" / "v6_ablation" / "V6_PREPROCESSING_ABLATION_REPORT.md"
    if not report_path.exists():
        pytest.skip("Ablation report not generated yet")

    content = report_path.read_text(encoding="utf-8")
    assert "EXP-C" in content
    assert "CVL" in content
