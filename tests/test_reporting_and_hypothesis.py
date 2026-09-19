import os
from pathlib import Path
import pytest

from backend.app.services.report_service import (
    ForensicReportService,
    ReportGenerationResult,
)
from cv.features.complexity import (
    ComplexityCategory,
    ComplexityComponents,
    ComplexityResult,
)
from cv.preprocessing.suitability import (
    SuitabilityMetrics,
    SuitabilityResult,
    SuitabilityStatus,
)
from ml.evaluation.hypothesis import (
    ENFSIConclusionScale,
    ForensicHypothesisEvaluator,
    ForensicSimilarityBand,
    ForensicSimilarityScaleEvaluation,
    HypothesisEvaluationReport,
)
from ml.explainability.explainer import (
    DualExplanationReport,
    TabularFeatureAttribution,
)
from ml.models.comparison import (
    ComparisonReport,
    DifferenceType,
    FeatureDifferenceDetail,
)
from ml.models.disguise import (
    IndicatorLevel,
    SimulationIndicators,
)
from ml.models.fusion import (
    HybridFusionResult,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def mock_suitability_result() -> SuitabilityResult:
    metrics = SuitabilityMetrics(
        blur_laplacian_var=185.4,
        rms_contrast=42.1,
        michelson_contrast=0.88,
        snr_db=18.5,
        stroke_density=0.045,
        connected_components_count=4,
        width=500,
        height=500,
    )
    return SuitabilityResult(
        status=SuitabilityStatus.SUITABLE,
        overall_score=0.92,
        is_admissible=True,
        metrics=metrics,
        reasons=["Netlik ve kontrast mükemmel."],
        recommendations=[],
    )


@pytest.fixture
def mock_genuine_fusion_result() -> HybridFusionResult:
    return HybridFusionResult(
        hybrid_similarity_score=0.91,
        morphological_similarity=0.90,
        deep_metric_similarity=0.92,
        gating_penalty=0.0,
        gating_triggered=False,
        concordance_level="KUVVETLİ ORİJİNAL DESTEĞİ",
        forensic_explanation="Tüm katmanlar referans yazar profiliyle tam uyumludur.",
        audit_hash_sha256="mock_fusion_hash_genuine",
    )


@pytest.fixture
def mock_forged_fusion_result() -> HybridFusionResult:
    return HybridFusionResult(
        hybrid_similarity_score=0.22,
        morphological_similarity=0.20,
        deep_metric_similarity=0.25,
        gating_penalty=0.25,
        gating_triggered=True,
        concordance_level="KUVVETLİ SAHTECİLİK / FARKLI YAZAR DESTEĞİ",
        forensic_explanation="Kritik yapısal sapmalar sebebiyle gating kalkanı devrededir.",
        audit_hash_sha256="mock_fusion_hash_forged",
    )


@pytest.fixture
def mock_full_case_bundle(mock_suitability_result, mock_genuine_fusion_result):
    case_info = {
        "case_id": "DOSYA_2026_999",
        "authority": "Ankara 4. Ağır Ceza Mahkemesi",
        "examiner_id": "Adli Belge Uzmanı Dr. Ahmet Yılmaz",
        "questioned_doc_id": "SENET_NO_77819",
        "reference_doc_ids": ["MUKAYESE_01", "MUKAYESE_02", "MUKAYESE_03", "MUKAYESE_04"],
        "date_utc": "2026-03-30T10:00:00Z",
    }

    comparison = ComparisonReport(
        similarity_score=0.90,
        euclidean_distance=0.35,
        mahalanobis_distance=1.12,
        concordant_features=[
            FeatureDifferenceDetail(
                feature_key="slant_angle_deg",
                feature_name_tr="Yazı Eğim Açısı",
                questioned_value=72.4,
                reference_mean=71.8,
                reference_std=1.2,
                absolute_difference=0.6,
                z_score=0.5,
                diff_type=DifferenceType.CONCORDANT,
                explanation="Yazı eğimi doğal varyasyon sınırları içerisinde tam uyumludur.",
            )
        ],
        natural_variations=[],
        significant_differences=[],
        preliminary_opinion="KUVVETLİ UYUM",
        reference_count=4,
        audit_hash_sha256="mock_comp_hash",
    )

    complexity = ComplexityResult(
        overall_complexity_index=0.74,
        category=ComplexityCategory.HIGH,
        components=ComplexityComponents(
            turning_points=42,
            crossings=18,
            loops=5,
            skeleton_length_px=1450,
            spatial_entropy=0.82,
        ),
        feature_contributions={
            "loops_pct": 28.0,
            "crossings_pct": 22.0,
            "spatial_entropy_pct": 20.0,
            "turning_points_pct": 18.0,
            "skeleton_length_pct": 12.0,
        },
        forgery_vulnerability="Düşük taklit riski, yüksek ayırt edicilik.",
        confidence=0.95,
    )

    disguise = SimulationIndicators(
        indicator_level=IndicatorLevel.NO_STRONG_EVIDENCE,
        overall_anomaly_score=0.08,
        tremor_anomaly_detected=False,
        kinematic_decoupling_detected=False,
        structural_disguise_detected=False,
        anomaly_details=["Balistik nöromusküler akış doğaldır."],
        confidence=0.92,
        audit_hash_sha256="mock_disguise_hash",
    )

    xai = DualExplanationReport(
        top_positive_features=[
            TabularFeatureAttribution(
                feature_key="slant_angle_deg",
                feature_name_tr="Yazı Eğim Açısı",
                impact_score=0.34,
                direction="POSITIVE",
                description="Yazı eğim açısı referans yazar profiliyle tam örtüşmektedir.",
            )
        ],
        top_negative_features=[],
        visual_attention_summary="Model dikkati imzanın karakteristik başlangıç ilmeğine odaklanmıştır.",
        audit_hash_sha256="mock_xai_hash",
    )

    evaluator = ForensicHypothesisEvaluator()
    hypothesis = evaluator.evaluate(
        hybrid_result=mock_genuine_fusion_result,
        reference_count=4,
        suitability_result=mock_suitability_result,
        suspect_name="Mehmet Özkan",
    )

    return {
        "case_info": case_info,
        "suitability": mock_suitability_result,
        "comparison": comparison,
        "complexity": complexity,
        "disguise": disguise,
        "fusion": mock_genuine_fusion_result,
        "xai": xai,
        "hypothesis": hypothesis,
    }


# ==============================================================================
# 1. ENFSI Hipotez Değerlendirme Testleri
# ==============================================================================

def test_hypothesis_evaluator_genuine_high_confidence(mock_genuine_fusion_result, mock_suitability_result):
    """Yüksek benzerlik ve yeterli referans numunesi durumunda H1 güçlü desteğini test eder."""
    evaluator = ForensicHypothesisEvaluator()
    report = evaluator.evaluate(
        hybrid_result=mock_genuine_fusion_result,
        reference_count=8,
        suitability_result=mock_suitability_result,
        suspect_name="Test Şahıs",
    )

    assert isinstance(report, HypothesisEvaluationReport)
    assert report.support_level in (
        ENFSIConclusionScale.EXTREMELY_STRONG_SUPPORT_FOR_H1,
        ENFSIConclusionScale.VERY_STRONG_SUPPORT_FOR_H1,
    )
    assert report.log_likelihood_ratio >= 3.0
    assert report.uncertainty_score < 0.20
    assert len(report.forensic_limitations) == 0
    assert "H1" in report.verbal_conclusion_tr
    assert report.similarity_scale_evaluation is not None
    assert report.similarity_scale_evaluation.band == ForensicSimilarityBand.NATURAL_VARIATION_MATCH


def test_hypothesis_evaluator_sample_size_penalty(mock_genuine_fusion_result, mock_suitability_result):
    """Yalnızca 1 referans belge olduğunda belirsizlik artışını ve kısıt şerhini test eder."""
    evaluator = ForensicHypothesisEvaluator()
    report = evaluator.evaluate(
        hybrid_result=mock_genuine_fusion_result,
        reference_count=1,  # Yetersiz numune
        suitability_result=mock_suitability_result,
    )

    assert report.uncertainty_score >= 0.40
    assert len(report.forensic_limitations) >= 1
    assert any("KRİTİK KISIT" in lim or "1 adet referans" in lim for lim in report.forensic_limitations)
    # Belirsizlik sebebiyle Log(LR) hafif büzülmüş olmalıdır
    assert report.log_likelihood_ratio < 3.5


def test_hypothesis_evaluator_forgery_case(mock_forged_fusion_result, mock_suitability_result):
    """Sahtecilik / farklı yazar durumunda H2 güçlü desteğini test eder."""
    evaluator = ForensicHypothesisEvaluator()
    report = evaluator.evaluate(
        hybrid_result=mock_forged_fusion_result,
        reference_count=5,
        suitability_result=mock_suitability_result,
    )

    assert report.support_level in (
        ENFSIConclusionScale.VERY_STRONG_SUPPORT_FOR_H2,
        ENFSIConclusionScale.STRONG_SUPPORT_FOR_H2,
    )
    assert report.log_likelihood_ratio <= -2.0
    assert "H2" in report.verbal_conclusion_tr
    assert report.similarity_scale_evaluation is not None
    assert report.similarity_scale_evaluation.band == ForensicSimilarityBand.COMPLETE_DISPARITY


# ==============================================================================
# 2. 6 Kademeli Kriminalojik Benzerlik ve Tetabuk Analizi Testleri
# ==============================================================================

def test_forensic_similarity_scale_bands(mock_suitability_result):
    """6 benzerlik bandının ve %98+ Tetabuk sahtecilik anomalisinin doğru çalıştığını test eder."""
    evaluator = ForensicHypothesisEvaluator()

    # 1. Band: %99 (Tetabuk / İzdüşüm ve Dolandırıcılık Anomalisi)
    res_tetabuk = HybridFusionResult(
        hybrid_similarity_score=0.99,
        morphological_similarity=0.99,
        deep_metric_similarity=0.99,
        gating_penalty=0.0,
        gating_triggered=False,
        concordance_level="ANORMAL ÇAKIŞMA",
        forensic_explanation="Birebir çakışma.",
        audit_hash_sha256="hash1",
    )
    rep_tetabuk = evaluator.evaluate(res_tetabuk, reference_count=5, suitability_result=mock_suitability_result)
    assert rep_tetabuk.similarity_scale_evaluation.band == ForensicSimilarityBand.SUPERIMPOSITION_ANOMALY
    assert rep_tetabuk.similarity_scale_evaluation.is_superimposition_anomaly is True
    assert rep_tetabuk.support_level == ENFSIConclusionScale.VERY_STRONG_SUPPORT_FOR_H2
    assert rep_tetabuk.log_likelihood_ratio < 0.0
    assert any("tetabuk" in lim.lower() for lim in rep_tetabuk.forensic_limitations)

    # 2. Band: %88 (Doğal Varyasyon ve Aidiyet Sınırları)
    res_natural = HybridFusionResult(
        hybrid_similarity_score=0.88,
        morphological_similarity=0.87,
        deep_metric_similarity=0.89,
        gating_penalty=0.0,
        gating_triggered=False,
        concordance_level="DOĞAL UYUM",
        forensic_explanation="Doğal varyasyon.",
        audit_hash_sha256="hash2",
    )
    rep_natural = evaluator.evaluate(res_natural, reference_count=5, suitability_result=mock_suitability_result)
    assert rep_natural.similarity_scale_evaluation.band == ForensicSimilarityBand.NATURAL_VARIATION_MATCH
    assert rep_natural.similarity_scale_evaluation.is_superimposition_anomaly is False

    # 3. Band: %76 (Kritik Ayrım Alanı - Nitelikli Taklit Şüphesi)
    res_skilled = HybridFusionResult(
        hybrid_similarity_score=0.76,
        morphological_similarity=0.75,
        deep_metric_similarity=0.77,
        gating_penalty=0.0,
        gating_triggered=False,
        concordance_level="KRİTİK ALAN",
        forensic_explanation="Nitelikli taklit şüphesi.",
        audit_hash_sha256="hash3",
    )
    rep_skilled = evaluator.evaluate(res_skilled, reference_count=5, suitability_result=mock_suitability_result)
    assert rep_skilled.similarity_scale_evaluation.band == ForensicSimilarityBand.SKILLED_SIMULATION_ZONE
    assert rep_skilled.similarity_scale_evaluation.requires_micro_inspection is True

    # 4. Band: %58 (Kararsız / İstekli Bozma Alanı)
    res_inconclusive = HybridFusionResult(
        hybrid_similarity_score=0.58,
        morphological_similarity=0.55,
        deep_metric_similarity=0.60,
        gating_penalty=0.0,
        gating_triggered=False,
        concordance_level="KARARSIZ",
        forensic_explanation="Kararsız.",
        audit_hash_sha256="hash4",
    )
    rep_inconclusive = evaluator.evaluate(res_inconclusive, reference_count=5, suitability_result=mock_suitability_result)
    assert rep_inconclusive.similarity_scale_evaluation.band == ForensicSimilarityBand.INCONCLUSIVE_DISGUISE_ZONE

    # 5. Band: %35 (Düşük Benzerlik - Farklı Şahıs)
    res_low = HybridFusionResult(
        hybrid_similarity_score=0.35,
        morphological_similarity=0.30,
        deep_metric_similarity=0.40,
        gating_penalty=0.0,
        gating_triggered=False,
        concordance_level="DÜŞÜK UYUM",
        forensic_explanation="Farklı yazar.",
        audit_hash_sha256="hash5",
    )
    rep_low = evaluator.evaluate(res_low, reference_count=5, suitability_result=mock_suitability_result)
    assert rep_low.similarity_scale_evaluation.band == ForensicSimilarityBand.LOW_SIMILARITY_NON_SOURCE

    # 6. Band: %10 (Tam Uyumsuzluk)
    res_disp = HybridFusionResult(
        hybrid_similarity_score=0.10,
        morphological_similarity=0.08,
        deep_metric_similarity=0.12,
        gating_penalty=0.0,
        gating_triggered=False,
        concordance_level="TAM UYUMSUZLUK",
        forensic_explanation="Rastgele sahte.",
        audit_hash_sha256="hash6",
    )
    rep_disp = evaluator.evaluate(res_disp, reference_count=5, suitability_result=mock_suitability_result)
    assert rep_disp.similarity_scale_evaluation.band == ForensicSimilarityBand.COMPLETE_DISPARITY


# ==============================================================================
# 3. Adli Raporlama Servisi (Report Service) Testleri
# ==============================================================================

def test_report_service_generation_html_and_hash(mock_full_case_bundle, tmp_path):
    """Adli raporun HTML formatında hatasız üretildiğini, benzerlik skalasını ve SHA-256 imzasını doğrular."""
    service = ForensicReportService(output_dir=tmp_path)
    bundle = mock_full_case_bundle

    result = service.generate_report(
        case_info=bundle["case_info"],
        suitability=bundle["suitability"],
        comparison=bundle["comparison"],
        complexity=bundle["complexity"],
        disguise=bundle["disguise"],
        fusion=bundle["fusion"],
        xai=bundle["xai"],
        hypothesis=bundle["hypothesis"],
        isolation_hash="mock_phase_a_sha256_seal",
    )

    assert isinstance(result, ReportGenerationResult)
    assert os.path.exists(result.html_path)
    assert len(result.report_sha256) == 64  # SHA-256

    # Üretilen HTML dosyasının içeriğini kontrol et
    with open(result.html_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "T.C. ADLİ BİLİRİŞİ KARAR DESTEK RAPORU" in content
    assert bundle["case_info"]["case_id"] in content
    assert "ENFSI Standartlarında Bayesyen Hipotez Değerlendirmesi" in content
    assert "Kriminalojik Benzerlik ve Tetabuk (İzdüşüm) Skalası Analizi" in content
    assert result.report_sha256 in content
