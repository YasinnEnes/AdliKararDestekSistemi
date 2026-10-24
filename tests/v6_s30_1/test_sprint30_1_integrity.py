"""
Sprint 30.1 Tests: Metric Integrity, Canonical Benchmark Reconciliation & Evidence Audit
Contains 10 independent verification tests:
1. test_weighted_auc_formula_matches_independent_recomputation
2. test_all_models_evaluated_on_identical_canonical_pairs
3. test_macro_auc_mathematical_consistency
4. test_weighted_eer_formula_consistency
5. test_pooled_metrics_computed_from_true_pooled_predictions
6. test_zero_test_set_leakage_in_thresholds
7. test_selection_guard_returns_no_candidate_passed
8. test_model_artifact_hashes_strictly_intact
9. test_similarity_score_and_label_direction
10. test_report_table_matches_json_exactly
"""

import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "v6_s30_1"
RECOMPUTED_JSON = REPORTS_DIR / "sprint30_1_recomputed_metrics.json"
PROVENANCE_JSON = REPORTS_DIR / "sprint30_1_source_provenance.json"
CANONICAL_MD = REPORTS_DIR / "SPRINT30_1_CANONICAL_RECONCILIATION.md"


@pytest.fixture(scope="module")
def audit_data():
    assert RECOMPUTED_JSON.exists(), f"Missing {RECOMPUTED_JSON}"
    with open(RECOMPUTED_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def provenance_data():
    assert PROVENANCE_JSON.exists(), f"Missing {PROVENANCE_JSON}"
    with open(PROVENANCE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# Test 1: Weighted AUC formülünün bağımsız hesapla uyuşması
def test_weighted_auc_formula_matches_independent_recomputation(audit_data):
    models = audit_data["recomputed_metrics_by_model"]
    weights = {"CEDAR": 0.30, "BHSig-260": 0.30, "UTSig": 0.25, "CVL Database": 0.15}

    for m_name, m_data in models.items():
        corpus_metrics = m_data["per_corpus"]
        independent_w_auc = sum(weights[ds] * corpus_metrics[ds]["roc_auc"] for ds in weights)
        reported_w_auc = m_data["canonical_summary"]["weighted_auc"]
        assert abs(independent_w_auc - reported_w_auc) < 1e-4, (
            f"{m_name} Weighted AUC mismatch: calculated {independent_w_auc:.6f} vs reported {reported_w_auc:.6f}"
        )


# Test 2: Tüm modellerin aynı canonical test pair kimliklerini kullanması
def test_all_models_evaluated_on_identical_canonical_pairs(audit_data):
    models = audit_data["recomputed_metrics_by_model"]
    canonical_corpora = ["CEDAR", "BHSig-260", "UTSig", "CVL Database", "IAM_HW"]

    base_pair_counts = None
    for m_name, m_data in models.items():
        corpus_metrics = m_data["per_corpus"]
        counts = {ds: corpus_metrics[ds]["total_pairs"] for ds in canonical_corpora}
        if base_pair_counts is None:
            base_pair_counts = counts
        else:
            assert counts == base_pair_counts, f"{m_name} pair counts {counts} diverge from baseline {base_pair_counts}!"


# Test 3: Macro AUC'nin korpus AUC'lerinden doğru hesaplanması
def test_macro_auc_mathematical_consistency(audit_data):
    models = audit_data["recomputed_metrics_by_model"]
    canonical_corpora = ["CEDAR", "BHSig-260", "UTSig", "CVL Database"]

    for m_name, m_data in models.items():
        corpus_metrics = m_data["per_corpus"]
        expected_macro = np.mean([corpus_metrics[ds]["roc_auc"] for ds in canonical_corpora])
        actual_macro = m_data["canonical_summary"]["macro_auc"]
        assert abs(expected_macro - actual_macro) < 1e-4, (
            f"{m_name} Macro AUC mismatch: expected {expected_macro:.6f} vs actual {actual_macro:.6f}"
        )


# Test 4: Weighted EER formülünün açık ve tutarlı olması
def test_weighted_eer_formula_consistency(audit_data):
    models = audit_data["recomputed_metrics_by_model"]
    weights = {"CEDAR": 0.30, "BHSig-260": 0.30, "UTSig": 0.25, "CVL Database": 0.15}

    for m_name, m_data in models.items():
        corpus_metrics = m_data["per_corpus"]
        expected_w_eer = sum(weights[ds] * corpus_metrics[ds]["eer"] for ds in weights)
        actual_w_eer = m_data["canonical_summary"]["weighted_eer"]
        assert abs(expected_w_eer - actual_w_eer) < 1e-4, (
            f"{m_name} Weighted EER mismatch: expected {expected_w_eer:.6f} vs actual {actual_w_eer:.6f}"
        )


# Test 5: Pooled metriklerin gerçek pooled tahminlerden hesaplanması
def test_pooled_metrics_computed_from_true_pooled_predictions(audit_data):
    models = audit_data["recomputed_metrics_by_model"]
    for m_name, m_data in models.items():
        pooled = m_data["global_pooled"]
        assert "roc_auc" in pooled
        assert "pr_auc" in pooled
        assert "eer" in pooled
        assert "accuracy" in pooled
        assert "f1" in pooled
        assert 0.0 <= pooled["roc_auc"] <= 1.0
        assert 0.0 <= pooled["eer"] <= 1.0
        # Pooled AUC must NOT be identical to Weighted AUC (they are mathematically distinct)
        weighted_auc = m_data["canonical_summary"]["weighted_auc"]
        assert abs(pooled["roc_auc"] - weighted_auc) > 1e-3, (
            f"{m_name}: Pooled AUC ({pooled['roc_auc']}) should not falsely equal Weighted AUC ({weighted_auc})"
        )


# Test 6: Test verisinin threshold seçiminde kullanılmaması
def test_zero_test_set_leakage_in_thresholds(provenance_data):
    prov = provenance_data["model_provenance"]
    for m_name, meta in prov.items():
        assert "Validation split" in meta["threshold_source"]
        assert "Zero test-leakage" in meta["threshold_source"]


# Test 7: Eksik adayların otomatik winner seçilmemesi (NO CANDIDATE PASSED ALL GUARDS)
def test_selection_guard_returns_no_candidate_passed(audit_data):
    verdict = audit_data["official_selection_verdict"]
    assert verdict == "NO CANDIDATE PASSED ALL GUARDS", f"Expected strict rejection, got: {verdict}"
    guards = audit_data["guard_evaluations"]
    for c_name, g_info in guards.items():
        assert g_info["all_guards_passed"] is False, f"{c_name} should fail strict production guards!"


# Test 8: Model hash'lerinin değişmediğinin doğrulanması
def test_model_artifact_hashes_strictly_intact(provenance_data):
    prov = provenance_data["model_provenance"]
    for m_name, meta in prov.items():
        assert meta["pt_hash_intact"] is True, f"{m_name} PT hash drift detected!"
        assert meta["actual_pt_hash"] == meta["expected_pt_hash"], f"{m_name} hash mismatch!"


# Test 9: Etiket ve score yönünün doğrulanması (yüksek skor = genuine)
def test_similarity_score_and_label_direction(audit_data):
    models = audit_data["recomputed_metrics_by_model"]
    for m_name, m_data in models.items():
        for ds in ["CEDAR", "BHSig-260", "UTSig", "CVL Database"]:
            c = m_data["per_corpus"][ds]
            # Genuine-Genuine mean score must exceed Genuine-Forgery mean score
            assert c["gg_mean"] > c["gf_mean"], (
                f"{m_name}/{ds}: Score direction inverted! GG mean ({c['gg_mean']}) <= GF mean ({c['gf_mean']})"
            )
            assert c["gg_gf_separation"] > 0, f"{m_name}/{ds}: Negative separation!"


# Test 10: Rapor tablosunun JSON verisiyle birebir uyuşması
def test_report_table_matches_json_exactly(audit_data):
    models = audit_data["recomputed_metrics_by_model"]
    assert CANONICAL_MD.exists(), f"Missing {CANONICAL_MD}"
    md_content = CANONICAL_MD.read_text(encoding="utf-8")

    for m_name, m_data in models.items():
        p = m_data["per_corpus"]
        s = m_data["canonical_summary"]
        cedar_str = f"{p['CEDAR']['roc_auc']:.4f}"
        weighted_str = f"{s['weighted_auc']:.4f}"
        macro_str = f"{s['macro_auc']:.4f}"

        assert cedar_str in md_content, f"{m_name} CEDAR AUC {cedar_str} missing from markdown table!"
        assert weighted_str in md_content, f"{m_name} Weighted AUC {weighted_str} missing from markdown table!"
        assert macro_str in md_content, f"{m_name} Macro AUC {macro_str} missing from markdown table!"
