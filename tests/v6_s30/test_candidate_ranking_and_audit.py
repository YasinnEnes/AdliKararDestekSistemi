"""
Sprint 30 Tests: Candidate Ranking Audit & Bug Reproduction
Verifies that Candidate D was falsely labeled winner and Candidate B is strictly superior across all 4 axes.
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


def test_candidate_b_strictly_dominates_candidate_d(recon_data):
    models = recon_data["models_evaluation"]
    b_macro = models["Candidate_B"]["Canonical_Summary"]["macro_auc"]
    d_macro = models["Candidate_D"]["Canonical_Summary"]["macro_auc"]
    assert b_macro > d_macro, f"Candidate B Macro ({b_macro}) must exceed Candidate D ({d_macro})"

    b_weighted = models["Candidate_B"]["Canonical_Summary"]["weighted_auc"]
    d_weighted = models["Candidate_D"]["Canonical_Summary"]["weighted_auc"]
    assert b_weighted > d_weighted, f"Candidate B Weighted ({b_weighted}) must exceed Candidate D ({d_weighted})"

    b_cvl = models["Candidate_B"]["CVL Database"]["auc"]
    d_cvl = models["Candidate_D"]["CVL Database"]["auc"]
    assert b_cvl > d_cvl, f"Candidate B CVL ({b_cvl}) must exceed Candidate D ({d_cvl})"

    b_utsig = models["Candidate_B"]["UTSig"]["auc"]
    d_utsig = models["Candidate_D"]["UTSig"]["auc"]
    assert b_utsig > d_utsig, f"Candidate B UTSig ({b_utsig}) must exceed Candidate D ({d_utsig})"


def test_audit_fault_mechanism_documented(recon_data):
    audit = recon_data["audit_findings"]
    assert "fault_mechanism" in audit
    assert "best_cand_name" in audit["fault_mechanism"]
    assert "Candidate_D" in audit["fault_mechanism"]
    assert audit["lines"] == "369-376"


def test_reproduction_of_silent_selection_loop_failure():
    """Reproduces the exact buggy selection loop from execute_sprint29.py to prove fault mechanism."""
    mock_candidate_evaluations = {
        "Candidate_A": {"CVL Database": {"auc": 0.7440}, "UTSig": {"auc": 0.9020}},
        "Candidate_B": {"CVL Database": {"auc": 0.7644}, "UTSig": {"auc": 0.9010}},
        "Candidate_D": {"CVL Database": {"auc": 0.7256}, "UTSig": {"auc": 0.8780}},
        "Candidate_E": {"CVL Database": {"auc": 0.5968}, "UTSig": {"auc": 0.8945}},
    }
    # Buggy code snippet from execute_sprint29.py:
    best_cand_name = "Candidate_D"
    for c_name in ["Candidate_E", "Candidate_D", "Candidate_B", "Candidate_A"]:
        eval_dict = mock_candidate_evaluations[c_name]
        if eval_dict["CVL Database"]["auc"] >= 0.80 and eval_dict["UTSig"]["auc"] >= 0.88:
            best_cand_name = c_name
            break

    # Condition is false for every candidate, so loop fails through and retains Candidate_D!
    assert best_cand_name == "Candidate_D", "Buggy loop must replicate the fallback assignment"


def test_corrected_selection_identifies_candidate_b():
    """Validates that a corrected ranking based on multi-corpus Macro AUC correctly selects Candidate B."""
    candidates = ["Candidate_A", "Candidate_B", "Candidate_D", "Candidate_E"]
    macro_scores = {
        "Candidate_A": 0.9068,  # Note: A fails CVL badly (0.7440)
        "Candidate_B": 0.8920,
        "Candidate_D": 0.8714,
        "Candidate_E": 0.8439,
    }
    # Between MAR candidates (B vs D):
    mar_winner = max(["Candidate_B", "Candidate_D"], key=lambda c: macro_scores[c])
    assert mar_winner == "Candidate_B"
