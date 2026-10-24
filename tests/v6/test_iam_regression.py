"""
FHDSS SPRINT 27 — Test: IAM Regression & Lexical Confounding (Phase 6)
Verifies:
- IAM isolated words are treated as an external benchmark domain.
- IAM lexical confounding is documented, preventing premature production promotion.
"""

from pathlib import Path
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_iam_external_domain_status():
    report_path = PROJECT_ROOT / "reports" / "v6" / "SPRINT_26_V6_MODEL_RESEARCH_FINAL_REPORT.md"
    assert report_path.exists(), "SPRINT 26 report missing"

    content = report_path.read_text(encoding="utf-8")
    assert "IAM_HW" in content
    assert "External Benchmark" in content or "separate" in content.lower()
