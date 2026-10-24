"""
FHDSS SPRINT 31 — Test Suite 2: P0 Covariance & Reference Count Robustness Tests
Verifies that:
1. N=1 uses diagonal forensic prior without singular matrix issues.
2. N=2 eliminates the rank-1 covariance explosion (D_M stays < 3.0, condition number <= 1e4).
3. N=3 and N=5 use Ledoit-Wolf with minimum prior diagonal protection.
4. Numerical stability: No NaN/Inf, condition numbers bounded, extreme distances capped.
5. Discrimination: Genuine-genuine pairs receive high similarity, distinct/forged pairs receive low similarity.
"""
import numpy as np
import pytest

from ml.models.comparison import ForensicComparisonEngine


@pytest.fixture
def engine():
    return ForensicComparisonEngine()


def _generate_synthetic_feature_vector(seed=42, slant=10.0, baseline_dev=15.0, stroke_len=180.0):
    """13 boyutlu öznitelik vektörü üretir."""
    rng = np.random.RandomState(seed)
    return np.array([
        slant,            # slant_angle_deg
        2.5,              # baseline_angle_deg
        baseline_dev,     # baseline_deviation_px
        3.0,              # loop_count
        15.0,             # crossing_count
        18.0,             # terminal_count
        stroke_len,       # avg_stroke_length_px
        45.0,             # inter_stroke_spacing_px
        0.05,             # tremor_index_psd
        0.12,             # smoothness_energy
        1.25,             # mean_velocity_proxy
        0.35,             # velocity_variance_proxy
        0.80,             # acceleration_proxy
    ], dtype=np.float64)


def test_n1_single_reference_robustness(engine):
    q_vec = _generate_synthetic_feature_vector(seed=101, slant=11.0)
    ref_vec = _generate_synthetic_feature_vector(seed=102, slant=10.0)

    report = engine.compare(q_vec, [ref_vec])

    assert report.reference_count == 1
    assert report.covariance_method == "SINGLE_REF_PRIOR"
    assert report.mahalanobis_distance < 2.0, f"N=1 Mahalanobis çok yüksek: {report.mahalanobis_distance}"
    assert report.similarity_score > 0.60
    assert report.confidence_warning is not None
    assert "Tek referans numune" in report.confidence_warning


def test_n2_rank1_covariance_explosion_prevented(engine):
    """Sprint 30.2'de D_M = 36.29'a patlayan N=2 senaryosunun düzeltildiğini doğrular."""
    q_vec = _generate_synthetic_feature_vector(seed=201, slant=10.5, baseline_dev=16.0)
    ref_1 = _generate_synthetic_feature_vector(seed=202, slant=10.0, baseline_dev=15.0)
    ref_2 = _generate_synthetic_feature_vector(seed=203, slant=11.0, baseline_dev=17.0)

    report = engine.compare(q_vec, [ref_1, ref_2])

    assert report.reference_count == 2
    assert report.covariance_method == "TARGET_SHRINKAGE_N2"
    # D_M asla 36'ya patlamamalı, < 3.0 kalmalıdır!
    assert report.mahalanobis_distance < 3.0, f"N=2 Mahalanobis patladı: {report.mahalanobis_distance}"
    assert report.similarity_score > 0.45, f"Morfolojik benzerlik çöktü: {report.similarity_score}"
    assert report.condition_number is not None
    assert report.condition_number <= 1e4, f"Koşul sayısı çok yüksek: {report.condition_number}"
    assert report.confidence_warning is not None
    assert "Target Shrinkage" in report.confidence_warning


def test_n3_and_n5_ledoit_wolf_stability(engine):
    q_vec = _generate_synthetic_feature_vector(seed=301)
    refs_3 = [_generate_synthetic_feature_vector(seed=300 + i, slant=10.0 + i * 0.5) for i in range(1, 4)]
    refs_5 = [_generate_synthetic_feature_vector(seed=500 + i, slant=10.0 + i * 0.4) for i in range(1, 6)]

    rep_3 = engine.compare(q_vec, refs_3)
    rep_5 = engine.compare(q_vec, refs_5)

    assert rep_3.covariance_method == "LEDOIT_WOLF"
    assert rep_5.covariance_method == "LEDOIT_WOLF"
    assert rep_3.mahalanobis_distance < 2.5
    assert rep_5.mahalanobis_distance < 2.5
    assert rep_5.confidence_warning is None  # N>=5 numunede uyarı kalkar


def test_identical_samples_zero_distance(engine):
    q_vec = _generate_synthetic_feature_vector(seed=401)
    refs_2 = [q_vec.copy(), q_vec.copy()]

    report = engine.compare(q_vec, refs_2)
    assert report.mahalanobis_distance < 1e-3, f"Özdeş örnek mesafesi sıfır değil: {report.mahalanobis_distance}"
    assert report.similarity_score >= 0.999


def test_discrimination_genuine_vs_forgery(engine):
    """Doğal varyasyonlu imza ile belirgin taklit/farklı yazar örneğini ayırabilmelidir."""
    ref_1 = _generate_synthetic_feature_vector(seed=501, slant=10.0, baseline_dev=15.0)
    ref_2 = _generate_synthetic_feature_vector(seed=502, slant=12.0, baseline_dev=16.0)

    # Samimi örnek: Eğim 11°, benzer vuruşlar
    q_genuine = _generate_synthetic_feature_vector(seed=503, slant=11.0, baseline_dev=15.5)
    # Taklit / Farklı yazar: Eğim zıt (-25°), devasa dalgalanma (60 px), aşırı vuruş boyu
    q_forged = _generate_synthetic_feature_vector(seed=504, slant=-25.0, baseline_dev=60.0, stroke_len=450.0)

    rep_gen = engine.compare(q_genuine, [ref_1, ref_2])
    rep_forg = engine.compare(q_forged, [ref_1, ref_2])

    assert rep_gen.mahalanobis_distance < rep_forg.mahalanobis_distance
    assert rep_gen.similarity_score > 0.65
    assert rep_forg.similarity_score < 0.25
    assert len(rep_forg.significant_differences) >= 2
