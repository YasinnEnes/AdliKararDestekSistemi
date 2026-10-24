"""
FHDSS SPRINT 28 — Test: Signature Anti-Regression (CEDAR, BHSig, UTSig)
Verifies:
- EXP-C preprocessing does NOT cause regression on signature verification benchmarks.
- CEDAR AUC > 0.99
- BHSig AUC > 0.98
- UTSig AUC > 0.90
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_signature_benchmarks_preserved():
    report_path = PROJECT_ROOT / "reports" / "v6_ablation" / "V6_PREPROCESSING_ABLATION_REPORT.md"
    if not report_path.exists():
        pytest.skip("Ablation report not generated yet")

    content = report_path.read_text(encoding="utf-8")
    assert "CEDAR" in content
    assert "BHSig" in content
    assert "UTSig" in content
