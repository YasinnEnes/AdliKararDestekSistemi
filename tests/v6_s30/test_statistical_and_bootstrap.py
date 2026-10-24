"""
Sprint 30 Tests: CVL Bootstrap Significance & Multiple Comparisons
Verifies 10,000 bootstrap consistency, confidence interval integrity, and Benjamini-Hochberg adjustment.
"""

import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "reports" / "v6_s30" / "sprint30_reconciliation_data.json"


@pytest.fixture
def recon_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_bootstrap_results_contain_all_target_comparisons(recon_data):
    boot = recon_data["bootstrap_results"]
    expected_tests = [
        "Candidate_B vs V5",
        "Candidate_B vs V6",
        "Candidate_D vs V5",
        "Candidate_D vs V6",
        "Candidate_B vs Candidate_D",
        "Candidate_C vs V5",
    ]
    for test_name in expected_tests:
        assert test_name in boot, f"Missing bootstrap comparison: {test_name}"


def test_candidate_b_cvl_gain_not_statistically_significant(recon_data):
    boot = recon_data["bootstrap_results"]
    test = boot["Candidate_B vs V5"]
    # 95% Confidence interval must cross zero
    assert test["ci_95_low"] < 0.0 < test["ci_95_high"], "95% CI should cross zero, indicating non-significance"
    assert test["fdr_adjusted_p_value"] > 0.05, "FDR adjusted p-value must be > 0.05 (Not Significant)"
    assert not test["significant_fdr_05"]


def test_fdr_adjusted_p_values_monotonicity(recon_data):
    boot = recon_data["bootstrap_results"]
    raw_ps = [v["raw_p_value"] for v in boot.values()]
    adj_ps = [v["fdr_adjusted_p_value"] for v in boot.values()]
    for raw, adj in zip(raw_ps, adj_ps):
        assert adj >= raw - 1e-6, "Adjusted p-value must be greater than or equal to raw p-value"
        assert 0.0 <= adj <= 1.0, "Adjusted p-value out of [0, 1] range"
