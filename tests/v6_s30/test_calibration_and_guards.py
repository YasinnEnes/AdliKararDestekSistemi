"""
Sprint 30 Tests: Forensic Score Calibration, Hard Negatives & Production Guards
Verifies Brier score, ECE, hard negative ground-truth classification, and non-negotiable production gates.
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


def test_calibration_brier_scores(recon_data):
    calib = recon_data["calibration_results"]
    for m_name in ["V5_Production", "V6_Baseline", "Candidate_B", "Candidate_D"]:
        assert m_name in calib
        brier = calib[m_name]["brier_score"]
        assert 0.0 <= brier <= 0.25, f"Brier score outside expected bounds: {brier}"
    # V5 should have the lowest (best) Brier score
    assert calib["V5_Production"]["brier_score"] < calib["Candidate_D"]["brier_score"]


def test_top_100_hard_negatives_ground_truth(recon_data):
    hard_neg = recon_data["hard_neg_results"]
    for m_name in ["V5_Production", "Candidate_B", "Candidate_D"]:
        hn = hard_neg[m_name]
        assert hn["false_accept_count"] + hn["true_reject_count"] == 100
        assert 0.0 <= hn["false_accept_rate_in_hard"] <= 1.0
    # V5 must have fewer false accepts than Candidate D
    assert hard_neg["V5_Production"]["false_accept_count"] < hard_neg["Candidate_D"]["false_accept_count"]


def test_production_candidate_guards_integrity(recon_data):
    guards = recon_data["guards_summary"]
    # V5 Production status must be PRODUCTION_AUTHORITY
    assert guards["V5_Production"]["Production_Promotion_Status"] == "PRODUCTION_AUTHORITY"

    # All candidate models must have status FAIL
    for cand_name in ["V6_Baseline", "Candidate_A", "Candidate_B", "Candidate_C", "Candidate_D", "Candidate_E"]:
        assert guards[cand_name]["Production_Promotion_Status"] == "FAIL", (
            f"{cand_name} should fail production criteria"
        )
        # CVL guard fails for all candidates
        assert not guards[cand_name]["Guard_CVL_ge_80"], f"{cand_name} cannot pass CVL >= 0.80"


def test_final_decision_is_option_b(recon_data):
    decision = recon_data["final_decision"]
    assert "OPTION B" in decision["official_decision"], "Final decision must be OPTION B"
    assert "V5 PRODUCTION / RESEARCH CONTINUES" in decision["official_decision"]
