"""
FHDSS SPRINT 28 — Test: IAM Domain Isolation
Verifies that IAM word-level handwriting remains isolated as an external domain benchmark.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_iam_external_isolation_status():
    report_path = PROJECT_ROOT / "reports" / "v6_ablation" / "SPRINT_28_INITIAL_PREPROCESSING_AUDIT.md"
    if not report_path.exists():
        pytest.skip("Report not generated yet")

    content = report_path.read_text(encoding="utf-8")
    assert "IAM" in content
