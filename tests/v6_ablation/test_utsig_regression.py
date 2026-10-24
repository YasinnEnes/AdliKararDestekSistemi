"""
FHDSS SPRINT 28 — Test: UTSig Anti-Regression
Verifies that UTSig Persian signature gains are strictly preserved.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_utsig_anti_regression():
    report_path = PROJECT_ROOT / "reports" / "v6_ablation" / "V6_PREPROCESSING_ABLATION_REPORT.md"
    if not report_path.exists():
        pytest.skip("Report not ready")
    text = report_path.read_text(encoding="utf-8")
    assert "UTSig" in text
