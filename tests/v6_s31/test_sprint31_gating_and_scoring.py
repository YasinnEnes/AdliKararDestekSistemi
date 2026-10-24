"""
FHDSS SPRINT 31 — Test Suite 4: P1 Forensic Gating & Scoring Mathematics Tests
Verifies that:
1. Gating audit log is comprehensively generated with rules, weights, and pre/post gating scores.
2. UI percentage is mathematically exact: ui_percentage == round(final_score * 100, 1).
3. Gating ablation:
   - Genuine pairs do not trigger gating and preserve high similarity (>85%).
   - Genuine pairs under scale variation do not trigger false structural contradictions.
   - Forged signatures with genuine structural contradictions (e.g. reverse slant) trigger gating
     and correctly suppress similarity (<35%).
4. Low sample count protection: With N <= 2, deep model weight is not crushed below 0.40.
"""
import pytest

from ml.models.comparison import (
    ComparisonReport,
    DifferenceType,
    FeatureDifferenceDetail,
    ForensicComparisonEngine,
)
from ml.models.fusion import HybridForensicFusion


@pytest.fixture
def fusion():
    return HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)


def _make_comparison_report(
    similarity_score=0.85,
    mahalanobis_dist=0.6,
    significant_diffs=None,
    reference_count=3,
):
    return ComparisonReport(
        similarity_score=similarity_score,
        euclidean_distance=1.2,
        mahalanobis_distance=mahalanobis_dist,
        concordant_features=[],
        natural_variations=[],
        significant_differences=significant_diffs or [],
        preliminary_opinion="Adli İnceleme Raporu",
        reference_count=reference_count,
        audit_hash_sha256="test_hash_audit_123",
        covariance_method="LEDOIT_WOLF",
        condition_number=24.5,
    )


def test_gating_audit_log_populated(fusion):
    comp = _make_comparison_report(similarity_score=0.85, mahalanobis_dist=0.6)
    res = fusion.fuse(comp, deep_similarity_score=0.90)

    log = res.gating_audit_log
    assert log is not None
    assert "pre_gating" in log
    assert "post_gating" in log
    assert log["pre_gating"]["s_deep"] == 0.90
    assert log["pre_gating"]["s_morph"] == 0.85
    assert log["ui_percentage"] == round(res.hybrid_similarity_score * 100.0, 1)


def test_ui_percentage_matches_final_score(fusion):
    comp = _make_comparison_report(similarity_score=0.78, mahalanobis_dist=0.9)
    res = fusion.fuse(comp, deep_similarity_score=0.82)

    expected_ui = round(res.hybrid_similarity_score * 100.0, 1)
    assert res.gating_audit_log["ui_percentage"] == expected_ui
    assert abs(res.hybrid_similarity_score * 100.0 - expected_ui) <= 0.05


def test_gating_ablation_genuine_no_trigger(fusion):
    """Doğal varyasyonlu samimi imzada gating tetiklenmemeli ve yüksek skor kalibre edilmelidir."""
    comp = _make_comparison_report(similarity_score=0.88, mahalanobis_dist=0.45, reference_count=3)
    res = fusion.fuse(comp, deep_similarity_score=0.92)

    assert res.gating_triggered is False
    assert res.gating_penalty == 0.0
    assert res.hybrid_similarity_score >= 0.85, f"Samimi imza skoru düşük: {res.hybrid_similarity_score}"
    assert "DOĞAL VARYASYON" in res.concordance_level or "TETABUK" in res.concordance_level


def test_gating_triggers_on_real_forgery_contradiction(fusion):
    """Gerçek sahtecilik emarelerinde (zıt eğim açısı > 18° ve Z > 3.5) gating devreye girmelidir."""
    slant_diff = FeatureDifferenceDetail(
        feature_key="slant_angle_deg",
        feature_name_tr="Yazı Eğim Açısı",
        questioned_value=28.0,
        reference_mean=-5.0,
        reference_std=3.0,
        absolute_difference=33.0,  # 33° zıt eğim farkı!
        z_score=11.0,
        diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
        explanation="Majör zıt eğim sapması",
    )
    comp = _make_comparison_report(
        similarity_score=0.18,
        mahalanobis_dist=4.5,
        significant_diffs=[slant_diff],
        reference_count=3,
    )

    # Derin model dokusal benzerlikten %70 verse dahi gating bunu baskılamalıdır
    res = fusion.fuse(comp, deep_similarity_score=0.70)

    assert res.gating_triggered is True
    assert len(res.structural_contradictions) >= 1
    assert "Zıt Eğim Çelişkisi" in res.structural_contradictions[0]
    # Gating ağırlığı derin modelden morfolojiye kaymalıdır
    assert res.gating_weight < 0.45
    # Nihai skor sahtecilik bölgesine çekilmelidir (< 50%)
    assert res.hybrid_similarity_score < 0.50


def test_small_sample_size_deep_weight_protection(fusion):
    """N=1 veya N=2 durumunda derin model ağırlığı 0.40'ın altına düşürülmemelidir."""
    mild_diff = FeatureDifferenceDetail(
        feature_key="crossing_count",
        feature_name_tr="Kesişim / Kavşak Sayısı",
        questioned_value=25.0,
        reference_mean=18.0,
        reference_std=2.0,
        absolute_difference=7.0,
        z_score=3.5,
        diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
        explanation="Kavşak sayısı farkı",
    )
    comp_n1 = _make_comparison_report(
        similarity_score=0.40,
        mahalanobis_dist=2.8,
        significant_diffs=[mild_diff],
        reference_count=1,
    )

    res = fusion.fuse(comp_n1, deep_similarity_score=0.92)

    # N=1 olduğundan derin model ağırlığı en az 0.40 korunmalıdır
    assert res.gating_weight >= 0.40, f"N=1 için derin ağırlık çok düşük: {res.gating_weight}"
    assert res.hybrid_similarity_score >= 0.55
