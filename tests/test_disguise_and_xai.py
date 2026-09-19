import numpy as np
import pytest
import torch

from cv.features.extractors import (
    ForensicFeatureReport,
    LineQualityFeatures,
    StructuralSpatialFeatures,
)
from ml.explainability.explainer import (
    DualExplanationReport,
    DualForensicExplainer,
    TabularFeatureAttribution,
)
from ml.models.comparison import (
    ComparisonReport,
    DifferenceType,
    FeatureDifferenceDetail,
)
from ml.models.disguise import (
    ForensicDisguiseDetector,
    IndicatorLevel,
    SimulationIndicators,
)
from ml.models.siamese import ForensicConvNeXtEncoder


# ==============================================================================
# 1. Taklit ve Gizleme (Simulation & Disguise) Testleri
# ==============================================================================

@pytest.fixture
def genuine_feature_report() -> ForensicFeatureReport:
    """Akıcı, doğal nöromusküler hareketlerle üretilmiş orijinal el yazısı öznitelikleri."""
    struct = StructuralSpatialFeatures(
        slant_angle_deg=14.0,
        baseline_angle_deg=-1.5,
        baseline_deviation_px=2.5,
        loop_count=2,
        crossing_count=3,
        terminal_count=4,
        avg_stroke_length_px=75.0,
        inter_stroke_spacing_px=12.0,
    )
    kinematics = LineQualityFeatures(
        tremor_index_psd=0.03,        # Düşük titreme
        smoothness_energy=0.02,       # Pürüzsüz hat
        mean_velocity_proxy=2.20,     # Akıcı balistik hız
        velocity_variance_proxy=0.25, # Düşük duraksama
        acceleration_proxy=0.15,
    )
    vec = [14.0, -1.5, 2.5, 2.0, 3.0, 4.0, 75.0, 12.0, 0.03, 0.02, 2.20, 0.25, 0.15]
    return ForensicFeatureReport(structural=struct, line_quality=kinematics, feature_vector=vec, metadata={})


@pytest.fixture
def forged_simulated_feature_report() -> ForensicFeatureReport:
    """Yavaşlatılmış, yapay titremeli ve duraksamalı taklit imza öznitelikleri."""
    struct = StructuralSpatialFeatures(
        slant_angle_deg=-30.0,         # Kasıtlı radikal eğim değişimi
        baseline_angle_deg=8.0,
        baseline_deviation_px=14.0,    # Aşırı satır dalgalanması
        loop_count=6,
        crossing_count=10,
        terminal_count=12,
        avg_stroke_length_px=35.0,
        inter_stroke_spacing_px=28.0,
    )
    kinematics = LineQualityFeatures(
        tremor_index_psd=0.38,        # Yüksek frekanslı taklit titremesi
        smoothness_energy=0.25,       # Yüksek bükülme enerjisi (yavaş çizim)
        mean_velocity_proxy=1.10,     # Yavaşlatılmış hareket
        velocity_variance_proxy=0.85, # Yoğun duraksama / tereddüt
        acceleration_proxy=0.65,
    )
    vec = [-30.0, 8.0, 14.0, 6.0, 10.0, 12.0, 35.0, 28.0, 0.38, 0.25, 1.10, 0.85, 0.65]
    return ForensicFeatureReport(structural=struct, line_quality=kinematics, feature_vector=vec, metadata={})


def test_disguise_detector_genuine_profile(genuine_feature_report: ForensicFeatureReport):
    """Akıcı orijinal profilde taklit anomalisinin tetiklenmediğini doğrular."""
    detector = ForensicDisguiseDetector()
    indicators = detector.analyze(genuine_feature_report)

    assert isinstance(indicators, SimulationIndicators)
    assert indicators.indicator_level == IndicatorLevel.NO_STRONG_EVIDENCE
    assert indicators.overall_anomaly_score < 0.25
    assert indicators.tremor_anomaly_detected is False
    assert indicators.kinematic_decoupling_detected is False
    assert "doğal" in indicators.anomaly_details[0].lower()


def test_disguise_detector_simulated_forgery(forged_simulated_feature_report: ForensicFeatureReport):
    """Taklit ve zorlanmış profilde güçlü anomali tespit edildiğini doğrular."""
    detector = ForensicDisguiseDetector()

    # Kasıtlı eğim farkı içeren mock karşılaştırma raporu
    mock_diff = FeatureDifferenceDetail(
        feature_key="slant_angle_deg",
        feature_name_tr="Yazı Eğim Açısı",
        questioned_value=-30.0,
        reference_mean=14.0,
        reference_std=3.0,
        absolute_difference=44.0,
        z_score=14.67,
        diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
        explanation="Radikal eğim sapması",
    )
    comp_report = ComparisonReport(
        similarity_score=0.20,
        euclidean_distance=5.2,
        mahalanobis_distance=8.1,
        concordant_features=[],
        natural_variations=[],
        significant_differences=[mock_diff],
        preliminary_opinion="SAHTECİLİK",
        reference_count=3,
        audit_hash_sha256="mock_hash",
    )

    indicators = detector.analyze(forged_simulated_feature_report, comparison_report=comp_report)

    assert indicators.indicator_level == IndicatorLevel.STRONG_INDICATORS
    assert indicators.overall_anomaly_score >= 0.50
    assert indicators.tremor_anomaly_detected is True
    assert indicators.kinematic_decoupling_detected is True
    assert indicators.structural_disguise_detected is True
    assert len(indicators.anomaly_details) >= 2


# ==============================================================================
# 2. Çift Seviyeli Açıklanabilirlik (Dual XAI) Testleri
# ==============================================================================

@pytest.fixture
def sample_comparison_report() -> ComparisonReport:
    """XAI testleri için hem uyumlu hem de sapan özellikler içeren rapor."""
    concordant = [
        FeatureDifferenceDetail(
            feature_key="loop_count",
            feature_name_tr="Kapalı Döngü Sayısı",
            questioned_value=2.0,
            reference_mean=2.1,
            reference_std=0.8,
            absolute_difference=0.1,
            z_score=0.12,
            diff_type=DifferenceType.CONCORDANT,
            explanation="Uyumlu",
        ),
        FeatureDifferenceDetail(
            feature_key="slant_angle_deg",
            feature_name_tr="Yazı Eğim Açısı",
            questioned_value=12.0,
            reference_mean=11.5,
            reference_std=3.0,
            absolute_difference=0.5,
            z_score=0.17,
            diff_type=DifferenceType.CONCORDANT,
            explanation="Uyumlu",
        ),
    ]

    significant = [
        FeatureDifferenceDetail(
            feature_key="tremor_index_psd",
            feature_name_tr="Titreme İndeksi",
            questioned_value=0.35,
            reference_mean=0.04,
            reference_std=0.05,
            absolute_difference=0.31,
            z_score=6.20,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Belirgin titreme",
        ),
        FeatureDifferenceDetail(
            feature_key="smoothness_energy",
            feature_name_tr="Eğrilik & Pürüzsüzlük",
            questioned_value=0.22,
            reference_mean=0.03,
            reference_std=0.04,
            absolute_difference=0.19,
            z_score=4.75,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Pürüzsüzlük sapması",
        ),
    ]

    return ComparisonReport(
        similarity_score=0.55,
        euclidean_distance=2.1,
        mahalanobis_distance=3.5,
        concordant_features=concordant,
        natural_variations=[],
        significant_differences=significant,
        preliminary_opinion="KARARSIZ",
        reference_count=3,
        audit_hash_sha256="mock_xai_hash",
    )


def test_dual_explainer_tabular_attributions(sample_comparison_report: ComparisonReport):
    """Tabüler XAI'ın ilk 5 pozitif ve ilk 5 negatif belirleyici özelliği doğru ayrıştırdığını test eder."""
    explainer = DualForensicExplainer()
    positives, negatives = explainer.explain_tabular(sample_comparison_report, top_k=5)

    assert len(positives) > 0
    assert len(negatives) > 0

    # Pozitiflerin yönü ve skoru
    for p in positives:
        assert isinstance(p, TabularFeatureAttribution)
        assert p.direction == "POSITIVE"
        assert p.impact_score > 0.0

    # Negatiflerin yönü ve skoru
    for n in negatives:
        assert isinstance(n, TabularFeatureAttribution)
        assert n.direction == "NEGATIVE"
        assert n.impact_score < 0.0


def test_dual_explainer_gradcam_generation():
    """ConvNeXt-Tiny modeli üzerinde Layer Grad-CAM ısı haritası üretimini doğrular."""
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    explainer = DualForensicExplainer()

    # 1 adet 3-kanallı 224x224 adli görsel tensörü
    image_tensor = torch.randn(1, 3, 224, 224)
    heatmap = explainer.generate_gradcam(encoder, image_tensor)

    # Çıktı matris boyutları ve değer aralığı
    assert heatmap.shape == (224, 224)
    assert np.min(heatmap) >= 0.0
    assert np.max(heatmap) <= 1.0

    # Renkli bindirme (overlay) testi
    dummy_img = np.full((224, 224, 3), 220, dtype=np.uint8)
    overlay = explainer.generate_heatmap_overlay(dummy_img, heatmap)
    assert overlay.shape == (224, 224, 3)
    assert overlay.dtype == np.uint8


def test_dual_explainer_full_report(sample_comparison_report: ComparisonReport):
    """Hem tabüler hem görsel XAI'ı birleştiren DualExplanationReport üretimini test eder."""
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    explainer = DualForensicExplainer()

    image_tensor = torch.randn(1, 3, 224, 224)
    report = explainer.explain(sample_comparison_report, encoder=encoder, image_tensor=image_tensor)

    assert isinstance(report, DualExplanationReport)
    assert len(report.top_positive_features) > 0
    assert len(report.top_negative_features) > 0
    assert "Grad-CAM" in report.visual_attention_summary
    assert report.audit_hash_sha256 is not None
