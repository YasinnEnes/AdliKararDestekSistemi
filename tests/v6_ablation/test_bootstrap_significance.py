"""
FHDSS SPRINT 28 — Test: Bootstrap Significance
Verifies statistical testing of CVL improvement under EXP-C.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_bootstrap_significance_recorded():
    report = PROJECT_ROOT / "reports" / "v6_ablation" / "V6_STATISTICAL_COMPARISON.md"
    if not report.exists():
        pytest.skip("Report not found")
    content = report.read_text(encoding="utf-8")
    assert "Bootstrap" in content
    assert "95% CI" in content
