import pytest
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from ml.dynamic.timeseries import (
    BiometricPoint,
    BiometricTrajectory,
    DynamicTimeSeriesAnalyzer,
)
from ml.explainability.counterfactual import CounterfactualExplainer
from ml.models.comparison import (
    ComparisonReport,
    DifferenceType,
    FeatureDifferenceDetail,
    ForensicComparisonEngine,
)


@pytest.fixture
def client():
    return TestClient(app)


def test_dynamic_time_series_profile_extraction():
    """Biyometrik yörüngeden hız, ivme, sarsıntı ve pürüzsüzlük çıkarımını test eder."""
    analyzer = DynamicTimeSeriesAnalyzer()
    traj = analyzer.generate_synthetic_tablet_trajectory(num_points=100, is_forgery=False)

    profile = analyzer.extract_profile(traj)
    assert profile.total_points == 100
    assert profile.duration_sec > 0.5
    assert profile.mean_velocity > 0.0
    assert profile.mean_acceleration >= 0.0
    assert profile.mean_jerk >= 0.0
    assert 0.0 <= profile.kinematic_smoothness <= 1.0
    assert 0.0 <= profile.in_air_time_ratio <= 1.0


def test_dtw_distance_and_similarity():
    """Özdeş ve taklit dinamik imzalar arasındaki DTW mesafesini ve benzerlik puanını test eder."""
    analyzer = DynamicTimeSeriesAnalyzer()
    traj_ref1 = analyzer.generate_synthetic_tablet_trajectory(num_points=120, is_forgery=False, seed=42)
    traj_ref2 = analyzer.generate_synthetic_tablet_trajectory(num_points=120, is_forgery=False, seed=42)
    traj_forgery = analyzer.generate_synthetic_tablet_trajectory(num_points=120, is_forgery=True, seed=99)

    # 1. Özdeş imzalar -> Çok yüksek benzerlik (DTW ~ 0.0, Simülasyon > 0.90)
    res_identical = analyzer.compare(traj_ref1, traj_ref2)
    assert res_identical.dtw_distance < 0.1
    assert res_identical.dtw_normalized_similarity > 0.90
    assert res_identical.is_biometrically_consistent is True

    # 2. Taklit / Çelişkili imza -> Düşük benzerlik
    res_forgery = analyzer.compare(traj_forgery, traj_ref1)
    assert res_forgery.dtw_distance > res_identical.dtw_distance
    assert res_forgery.dtw_normalized_similarity < res_identical.dtw_normalized_similarity


def test_counterfactual_what_if_simulation():
    """Karşıtlı 'Ne Olurdu?' simülatörünün hızını ve karar sınırlarını test eder."""
    explainer = CounterfactualExplainer()

    # Çelişkili bir karşılaştırma raporu kurgula
    diff_slant = FeatureDifferenceDetail(
        feature_key="slant_angle_deg",
        feature_name_tr="Yazı Eğim Açısı",
        questioned_value=25.0,
        reference_mean=5.0,
        reference_std=2.0,
        absolute_difference=20.0,
        z_score=10.0,
        diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
        explanation="Kritik eğim sapması",
    )
    diff_tremor = FeatureDifferenceDetail(
        feature_key="tremor_index_psd",
        feature_name_tr="FFT Titreme İndeksi",
        questioned_value=0.15,
        reference_mean=0.02,
        reference_std=0.01,
        absolute_difference=0.13,
        z_score=13.0,
        diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
        explanation="Mikroskobik titreme",
    )

    comp_report = ComparisonReport(
        similarity_score=0.20,
        euclidean_distance=16.4,
        mahalanobis_distance=16.4,
        concordant_features=[],
        natural_variations=[],
        significant_differences=[diff_slant, diff_tremor],
        preliminary_opinion="Kuvvetle Çelişkili",
        reference_count=5,
        audit_hash_sha256="TEST_HASH",
    )

    # Simülasyon: Eğim açısını -20.0 derece düzelterek referans ortalamasına çek
    custom_p = {"slant_angle_deg": -20.0, "tremor_index_psd": -0.13}
    sim_res = explainer.simulate_what_if(comp_report, custom_p, deep_similarity_score=0.85)

    assert sim_res["simulated_hybrid_score"] > 0.50
    assert sim_res["execution_time_ms"] < 250.0  # 250ms performans kriteri


def test_counterfactual_scenarios_generation():
    """Karşıtlı senaryo üreticisinin adli gating cezasını kaldıran senaryolar ürettiğini test eder."""
    explainer = CounterfactualExplainer()

    diff_slant = FeatureDifferenceDetail(
        feature_key="slant_angle_deg",
        feature_name_tr="Yazı Eğim Açısı",
        questioned_value=18.0,
        reference_mean=4.0,
        reference_std=2.0,
        absolute_difference=14.0,
        z_score=7.0,
        diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
        explanation="Aşırı eğim sapması",
    )

    comp_report = ComparisonReport(
        similarity_score=0.35,
        euclidean_distance=7.0,
        mahalanobis_distance=7.0,
        concordant_features=[],
        natural_variations=[],
        significant_differences=[diff_slant],
        preliminary_opinion="Çelişkili",
        reference_count=3,
        audit_hash_sha256="TEST_HASH",
    )

    result = explainer.generate_counterfactual_scenarios(comp_report, current_hybrid_score=0.40)

    assert len(result.top_scenarios) >= 1
    assert result.top_scenarios[0].simulated_hybrid_score > result.original_score
    assert len(result.top_scenarios[0].required_perturbations) >= 1


def test_biometrics_api_endpoint(client):
    """Biyometrik REST API karşılaştırma ve demo uç noktalarını test eder."""
    # 1. Demo uç noktası
    demo_resp = client.get("/api/v1/biometrics/synthetic-demo")
    assert demo_resp.status_code == 200
    demo_data = demo_resp.json()
    assert "dtw_distance" in demo_data
    assert "verbal_assessment_tr" in demo_data
    assert "questioned_profile" in demo_data

    # 2. Compare uç noktası
    q_pts = [
        {"x": 10.0, "y": 20.0, "t": 0.01, "pressure": 0.5},
        {"x": 15.0, "y": 25.0, "t": 0.02, "pressure": 0.6},
        {"x": 20.0, "y": 30.0, "t": 0.03, "pressure": 0.7},
    ]
    ref_pts = [
        {"x": 10.5, "y": 20.2, "t": 0.01, "pressure": 0.52},
        {"x": 15.2, "y": 25.1, "t": 0.02, "pressure": 0.59},
        {"x": 20.1, "y": 30.3, "t": 0.03, "pressure": 0.68},
    ]

    compare_resp = client.post(
        "/api/v1/biometrics/compare",
        json={"questioned_points": q_pts, "reference_points": ref_pts},
    )
    assert compare_resp.status_code == 200
    comp_data = compare_resp.json()
    assert comp_data["dtw_normalized_similarity"] > 0.50
    assert comp_data["is_biometrically_consistent"] is True
