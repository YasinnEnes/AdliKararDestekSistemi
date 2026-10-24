"""
Sprint 30 Tests: Canonical Benchmark Consistency & Metric Reproducibility
Verifies S29_CANONICAL_BENCHMARK structure, dataset coverage, Macro/Weighted formulas, and metric invariants.
"""

import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "reports" / "v6_s30" / "sprint30_reconciliation_data.json"


@pytest.fixture
def recon_data():
    assert DATA_PATH.exists(), f"Missing {DATA_PATH}"
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_canonical_corpora_inclusion(recon_data):
    models_eval = recon_data["models_evaluation"]
    canonical_required = {"CEDAR", "BHSig-260", "UTSig", "CVL Database"}
    for m_name, m_eval in models_eval.items():
        assert canonical_required.issubset(set(m_eval.keys())), f"{m_name} missing canonical corpora!"


def test_iam_monitoring_domain_separation(recon_data):
    models_eval = recon_data["models_evaluation"]
    for m_name, m_eval in models_eval.items():
        assert "IAM_HW" in m_eval, f"{m_name} missing IAM_HW monitoring domain!"
        # In canonical summary formula, IAM must NOT be part of canonical weights formula
        formula = m_eval["Canonical_Summary"]["weights_formula"]
        assert "IAM" not in formula, f"IAM must not be included in canonical weights formula: {formula}"


def test_macro_auc_mathematical_consistency(recon_data):
    models_eval = recon_data["models_evaluation"]
    canonical_corpora = ["CEDAR", "BHSig-260", "UTSig", "CVL Database"]
    for m_name, m_eval in models_eval.items():
        expected_macro = np.mean([m_eval[ds]["auc"] for ds in canonical_corpora])
        actual_macro = m_eval["Canonical_Summary"]["macro_auc"]
        assert abs(actual_macro - expected_macro) < 1e-3, f"{m_name} Macro AUC mismatch! {actual_macro} != {expected_macro}"


def test_weighted_auc_mathematical_consistency(recon_data):
    models_eval = recon_data["models_evaluation"]
    weights = {"CEDAR": 0.30, "BHSig-260": 0.30, "UTSig": 0.25, "CVL Database": 0.15}
    for m_name, m_eval in models_eval.items():
        expected_weighted = sum(weights[ds] * m_eval[ds]["auc"] for ds in weights)
        actual_weighted = m_eval["Canonical_Summary"]["weighted_auc"]
        assert abs(actual_weighted - expected_weighted) < 1e-3, f"{m_name} Weighted AUC mismatch! {actual_weighted} != {expected_weighted}"


def test_metric_bounds_and_validity(recon_data):
    models_eval = recon_data["models_evaluation"]
    for m_name, m_eval in models_eval.items():
        for ds in ["CEDAR", "BHSig-260", "UTSig", "CVL Database", "IAM_HW"]:
            metrics = m_eval[ds]
            assert 0.0 <= metrics["auc"] <= 1.0, f"Invalid AUC in {m_name}/{ds}"
            assert 0.0 <= metrics["eer"] <= 1.0, f"Invalid EER in {m_name}/{ds}"
            assert 0.0 <= metrics["pr_auc"] <= 1.0, f"Invalid PR-AUC in {m_name}/{ds}"
            assert 0.0 <= metrics["f1"] <= 1.0, f"Invalid F1 in {m_name}/{ds}"
            assert 0.0 <= metrics["accuracy"] <= 1.0, f"Invalid Accuracy in {m_name}/{ds}"


def test_threshold_reproducibility(recon_data):
    models_eval = recon_data["models_evaluation"]
    for m_name, m_eval in models_eval.items():
        for ds in ["CEDAR", "BHSig-260", "UTSig", "CVL Database"]:
            th = m_eval[ds]["eer_th"]
            assert 0.0 <= th <= 1.0, f"Operating threshold outside [0, 1] in {m_name}/{ds}: {th}"
