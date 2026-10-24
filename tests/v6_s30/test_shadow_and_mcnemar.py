"""
Sprint 30 Tests: Ground-Truth Shadow Benchmark & McNemar's Test
Verifies 500-pair ground truth integrity, 4-quadrant decomposition, and McNemar test statistics.
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


def test_shadow_pair_count_and_accuracy(recon_data):
    shadow = recon_data["shadow_summary"]
    assert shadow["total_pairs"] == 500, "Shadow test must have exactly 500 pairs"
    assert shadow["v5_accuracy"] > 0.80, "V5 accuracy in shadow benchmark must be > 80%"
    assert shadow["v5_accuracy"] > shadow["cand_b_accuracy"], "V5 accuracy must strictly exceed Candidate B"
    assert shadow["v5_accuracy"] > shadow["cand_d_accuracy"], "V5 accuracy must strictly exceed Candidate D"


def test_four_quadrant_partition_conservation(recon_data):
    shadow = recon_data["shadow_summary"]
    for cand_key in ["candidate_b", "candidate_d"]:
        cand_data = shadow[cand_key]
        total_q = (
            cand_data["true_improvement_count"]
            + cand_data["true_regression_count"]
            + cand_data["agreement_correct_count"]
            + cand_data["agreement_wrong_count"]
        )
        assert total_q == 500, f"{cand_key} quadrant counts must sum to exactly 500! Got {total_q}"
        assert cand_data["true_regression_count"] > cand_data["true_improvement_count"], (
            f"{cand_key} regressions must exceed improvements"
        )


def test_mcnemar_test_structure_and_significance(recon_data):
    mcnemar = recon_data["mcnemar_results"]
    assert "candidate_b_vs_v5" in mcnemar
    assert "candidate_d_vs_v5" in mcnemar

    mb = mcnemar["candidate_b_vs_v5"]
    assert mb["chi2_stat"] > 3.841, "Chi2 must exceed critical threshold for df=1 (3.841)"
    assert mb["p_value"] < 0.05, "P-value must be statistically significant (< 0.05)"
    assert mb["significant_05"]
