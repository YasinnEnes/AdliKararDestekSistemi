"""
Sprint 30.2 Tests: Low-Similarity Root-Cause Audit & Inference Robustness Validation
Validates findings from Matrices A through G:
1. test_v5_model_artifacts_and_hashes_intact
2. test_matrix_a_identity_and_determinism
3. test_matrix_b_scale_sensitivity_root_cause
4. test_matrix_c_full_page_whitespace_dilution
5. test_matrix_d_rotation_robustness_and_gating
6. test_matrix_e_stroke_quality_and_terminal_fragmentation
7. test_matrix_f_n2_covariance_ill_conditioning
8. test_matrix_g_gating_ablation_score_suppression
9. test_experiment_results_json_schema_completeness
10. test_ui_percentage_and_band_mapping_consistency
"""

import json
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports" / "v6_s30_2"
RESULTS_JSON = REPORTS_DIR / "sprint30_2_experiment_results.json"
HASHES_JSON = REPORTS_DIR / "sprint30_2_artifact_hashes.json"


@pytest.fixture(scope="module")
def experiment_data():
    assert RESULTS_JSON.exists(), f"Missing {RESULTS_JSON}"
    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def artifact_hashes():
    assert HASHES_JSON.exists(), f"Missing {HASHES_JSON}"
    with open(HASHES_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


# Test 1: V5 Model ağırlıklarının ve manifestosunun bütünlüğü
def test_v5_model_artifacts_and_hashes_intact(artifact_hashes):
    assert artifact_hashes["v5_production_pt"]["verified"] is True
    assert artifact_hashes["v5_production_onnx"]["verified"] is True
    assert artifact_hashes["v5_production_manifest"]["verified"] is True


# Test 2: Matrix A — Deterministiklik ve Özdeşlik Testi
def test_matrix_a_identity_and_determinism(experiment_data):
    identity_runs = [e for e in experiment_data["experiments"] if e["category"] == "A_IDENTITY"]
    assert len(identity_runs) >= 3, "Matrix A missing required runs"
    for r in identity_runs:
        layers = r["layers"]
        assert layers["layer_4_deep_metric"]["normalized_similarity"] == 1.0
        assert layers["layer_5_mahalanobis"]["morphological_similarity"] == 1.0
        assert layers["layer_7_gating"]["gating_triggered"] is False
        assert layers["layer_8_final_decision"]["final_hybrid_score"] == 1.0
        assert layers["layer_8_final_decision"]["ui_percentage"] == 100.0


# Test 3: Matrix B — Ölçekleme / Piksel Öznitelik Sapması Kök Nedeni
def test_matrix_b_scale_sensitivity_root_cause(experiment_data):
    scale_runs = {e["parameters"]["scale"]: e for e in experiment_data["experiments"] if e["category"] == "B_SCALE"}
    assert 1.0 in scale_runs and 2.5 in scale_runs
    
    # 2.5x ölçekte derin model aynı imzayı tanır (Deep >= 0.98), fakat piksel boyutu büyüdüğü için
    # morfolojik benzerlik çöker ve gating tetiklenir
    run_25 = scale_runs[2.5]["layers"]
    assert run_25["layer_4_deep_metric"]["normalized_similarity"] >= 0.98
    assert run_25["layer_5_mahalanobis"]["morphological_similarity"] < 0.40
    assert run_25["layer_7_gating"]["gating_triggered"] is True
    assert run_25["layer_8_final_decision"]["final_hybrid_score"] < 0.55


# Test 4: Matrix C — Tam Sayfa / Beyaz Alan Seyrelmesi (Whitespace Dilution)
def test_matrix_c_full_page_whitespace_dilution(experiment_data):
    roi_runs = {e["parameters"]["case"]: e for e in experiment_data["experiments"] if e["category"] == "C_ROI_WHITESPACE"}
    assert "FULL_A4_PAGE" in roi_runs and "ORIGINAL_CROP" in roi_runs

    full_page = roi_runs["FULL_A4_PAGE"]["layers"]
    orig_crop = roi_runs["ORIGINAL_CROP"]["layers"]

    # Tam sayfada derin benzerlik en az 0.20 puan düşer
    assert orig_crop["layer_4_deep_metric"]["normalized_similarity"] == 1.0
    assert full_page["layer_4_deep_metric"]["normalized_similarity"] < 0.75
    assert full_page["layer_8_final_decision"]["final_hybrid_score"] < 0.85


# Test 5: Matrix D — Dönme Toleransı ve Açı Eşiği
def test_matrix_d_rotation_robustness_and_gating(experiment_data):
    rot_runs = {e["parameters"]["angle_deg"]: e for e in experiment_data["experiments"] if e["category"] == "D_ROTATION"}
    # ±6 dereceye kadar derin model ve nihai skor >= 0.90
    assert rot_runs[6.0]["layers"]["layer_8_final_decision"]["final_hybrid_score"] >= 0.90
    assert rot_runs[-6.0]["layers"]["layer_8_final_decision"]["final_hybrid_score"] >= 0.90
    # -12 derecede eğim çelişkisi gating tetikler
    assert rot_runs[-12.0]["layers"]["layer_7_gating"]["gating_triggered"] is True


# Test 6: Matrix E — Çizgi Kalitesi ve İskelet Parçalanması
def test_matrix_e_stroke_quality_and_terminal_fragmentation(experiment_data):
    quality_runs = {e["parameters"]["case"]: e for e in experiment_data["experiments"] if e["category"] == "E_LINE_QUALITY"}
    
    # Gürültülü ve düşük kontrastlı görselde uç nokta (terminal) sayısı katlanır
    noise_run = quality_runs["SCANNER_NOISE"]["layers"]
    assert noise_run["layer_3_morphological_features"]["terminal_count"] >= 30
    assert noise_run["layer_5_mahalanobis"]["morphological_similarity"] < 0.50
    assert noise_run["layer_8_final_decision"]["final_hybrid_score"] < 0.65


# Test 7: Matrix F — N=2 Kovaryans Kararsızlığı ve Çoklu Referans
def test_matrix_f_n2_covariance_ill_conditioning(experiment_data):
    ref_runs = {e["parameters"]["ref_count"]: e for e in experiment_data["experiments"] 
                if e["category"] == "F_REFERENCE_COUNT" and e["parameters"]["pair_type"] == "GENUINE_GENUINE"}
    
    # N=2 durumunda Mahalanobis mesafesi rank-1 tekilliği yüzünden sıçrar
    assert ref_runs[2]["layers"]["layer_5_mahalanobis"]["mahalanobis_distance"] > 20.0
    # N=1 durumu prior varyans kullandığı için daha kararlıdır
    assert ref_runs[1]["layers"]["layer_5_mahalanobis"]["mahalanobis_distance"] < 3.0


# Test 8: Matrix G — Gating Etkisi ve Skor Baskılama
def test_matrix_g_gating_ablation_score_suppression(experiment_data):
    meta = experiment_data["metadata"]
    assert meta["gating_triggered_count"] >= 10
    assert meta["max_score_suppression_delta"] >= 0.25


# Test 9: JSON Şema Bütünlüğü
def test_experiment_results_json_schema_completeness(experiment_data):
    for exp in experiment_data["experiments"]:
        layers = exp["layers"]
        for layer_key in [
            "layer_1_input_shapes", "layer_2_preprocessing", "layer_3_morphological_features",
            "layer_4_deep_metric", "layer_5_mahalanobis", "layer_6_pre_gating",
            "layer_7_gating", "layer_8_final_decision"
        ]:
            assert layer_key in layers, f"{exp['experiment_id']} missing {layer_key}"


# Test 10: UI Yüzde ve Bant Eşlemesi
def test_ui_percentage_and_band_mapping_consistency(experiment_data):
    for exp in experiment_data["experiments"]:
        score = exp["layers"]["layer_8_final_decision"]["final_hybrid_score"]
        pct = exp["layers"]["layer_8_final_decision"]["ui_percentage"]
        band = exp["layers"]["layer_8_final_decision"]["ui_band"]
        assert abs(pct - round(score * 100.0, 1)) < 0.2
        if pct >= 98.0:
            assert band == "SUPERIMPOSITION_ANOMALY"
        elif pct >= 85.0:
            assert band == "NATURAL_VARIATION_MATCH"
        elif pct >= 70.0:
            assert band == "SKILLED_SIMULATION_ZONE"
        elif pct >= 50.0:
            assert band == "INCONCLUSIVE_DISGUISE_ZONE"
        elif pct >= 25.0:
            assert band == "LOW_SIMILARITY_NON_SOURCE"
        else:
            assert band == "COMPLETE_DISPARITY"
