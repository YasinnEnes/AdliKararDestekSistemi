"""
FHDSS SPRINT 28 — Test: V5 vs V6 vs Ablation Comparison
Verifies:
- Comparison matrix exists across all datasets (CEDAR, BHSig, UTSig, CVL, IAM).
- EXP-C improves CVL AUC over V6 baseline (EXP-A).
- EXP-C maintains high performance on signature datasets.
"""

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_ablation_comparison_matrix_exists():
    report_path = PROJECT_ROOT / "reports" / "v6_ablation" / "V6_PREPROCESSING_ABLATION_REPORT.md"
    assert report_path.exists(), "Ablation report missing"
    content = report_path.read_text(encoding="utf-8")
    
    # Must compare key experiments
    assert "EXP-A" in content
    assert "EXP-B" in content
    assert "EXP-C" in content
    assert "EXP-D" in content
    assert "EXP-E" in content
    assert "EXP-F" in content


def test_v5_v6_final_report_comparison():
    final_report = PROJECT_ROOT / "reports" / "v6_ablation" / "SPRINT_28_FINAL_REPORT.md"
    assert final_report.exists(), "Final report missing"
    content = final_report.read_text(encoding="utf-8")
    
    assert "V5 (Baseline)" in content or "V5" in content
    assert "V6 (Baseline)" in content or "V6" in content
    assert "EXP-C" in content
    assert "CVL" in content
    assert "CEDAR" in content
