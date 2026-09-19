from typing import List, Tuple
import cv2
import numpy as np
import pytest

from cv.features.complexity import (
    ComplexityCategory,
    ComplexityResult,
    SignatureComplexityScorer,
)
from cv.features.extractors import (
    ForensicFeatureExtractor,
    ForensicFeatureReport,
    LineQualityFeatures,
    StructuralSpatialFeatures,
)
from ml.models.comparison import (
    ComparisonReport,
    DifferenceType,
    ForensicComparisonEngine,
)


# ==============================================================================
# 1. Found & Rogers İmza Karmaşıklığı Testleri
# ==============================================================================

@pytest.fixture
def synthetic_simple_line() -> np.ndarray:
    """Yalnızca tek bir düz çizgiden oluşan basit (LOW complexity) imza taslağı."""
    canvas = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(canvas, (30, 100), (170, 100), 255, 1)
    return canvas


@pytest.fixture
def synthetic_complex_signature() -> Tuple[np.ndarray, np.ndarray]:
    """Çoklu döngüler, kesişimler ve dönüşler içeren karmaşık (HIGH complexity) imza ve ikili görsel."""
    skel = np.zeros((300, 300), dtype=np.uint8)
    binary = np.full((300, 300), 255, dtype=np.uint8)

    # 1. Döngüler (Çemberler)
    cv2.circle(skel, (80, 120), 30, 255, 1)
    cv2.circle(binary, (80, 120), 30, 0, 2)

    cv2.circle(skel, (180, 140), 25, 255, 1)
    cv2.circle(binary, (180, 140), 25, 0, 2)

    # 2. Kesişen hatlar ve dalgalı çizgiler
    cv2.line(skel, (40, 180), (260, 80), 255, 1)
    cv2.line(binary, (40, 180), (260, 80), 0, 2)

    cv2.line(skel, (50, 60), (250, 220), 255, 1)
    cv2.line(binary, (50, 60), (250, 220), 0, 2)

    # 3. Zikzak dalgalar
    pts = np.array([[40, 240], [90, 200], [140, 250], [190, 190], [240, 240]], np.int32)
    cv2.polylines(skel, [pts], False, 255, 1)
    cv2.polylines(binary, [pts], False, 0, 2)

    return skel, binary


def test_complexity_simple_line(synthetic_simple_line: np.ndarray):
    """Basit tek çizginin LOW karmaşıklık kategorisi aldığını doğrular."""
    scorer = SignatureComplexityScorer()
    result = scorer.score(synthetic_simple_line)

    assert isinstance(result, ComplexityResult)
    assert result.category == ComplexityCategory.LOW
    assert result.overall_complexity_index < 0.35
    assert result.components.loops == 0
    assert result.components.crossings == 0
    assert "YÜKSEK RİSK" in result.forgery_vulnerability


def test_complexity_complex_signature(synthetic_complex_signature: Tuple[np.ndarray, np.ndarray]):
    """Çoklu döngülü ve kesişimli imzanın yüksek karmaşıklık skoru aldığını doğrular."""
    skel, binary = synthetic_complex_signature
    scorer = SignatureComplexityScorer()
    result = scorer.score(skeleton_image=skel, binary_image=binary)

    assert isinstance(result, ComplexityResult)
    assert result.overall_complexity_index >= 0.50
    assert result.category in (ComplexityCategory.MEDIUM, ComplexityCategory.HIGH)
    assert result.components.loops >= 2
    assert result.components.crossings >= 2
    assert result.components.spatial_entropy > 0.40

    # Katkı yüzdeleri toplamı ~100 olmalı
    total_pct = sum(result.feature_contributions.values())
    assert abs(total_pct - 100.0) < 1.0


# ==============================================================================
# 2. Temel Adli Karşılaştırma Motoru (Forensic Comparison Engine) Testleri
# ==============================================================================

@pytest.fixture
def genuine_writer_profiles() -> Tuple[List[float], List[List[float]]]:
    """Aynı yazara ait 1 sorgulanan ve 3 referans öznitelik vektörü üretir."""
    base_vector = np.array([
        12.0,   # slant_angle_deg
        -2.5,   # baseline_angle_deg
        3.2,    # baseline_deviation_px
        2.0,    # loop_count
        4.0,    # crossing_count
        6.0,    # terminal_count
        85.0,   # avg_stroke_length_px
        14.0,   # inter_stroke_spacing_px
        0.04,   # tremor_index_psd
        0.03,   # smoothness_energy
        2.10,   # mean_velocity_proxy
        0.30,   # velocity_variance_proxy
        0.18,   # acceleration_proxy
    ], dtype=np.float64)

    # %2-%5 arası hafif doğal varyasyon simülasyonu
    np.random.seed(42)
    ref_1 = (base_vector * (1.0 + np.random.normal(0, 0.03, len(base_vector)))).tolist()
    ref_2 = (base_vector * (1.0 + np.random.normal(0, 0.04, len(base_vector)))).tolist()
    ref_3 = (base_vector * (1.0 + np.random.normal(0, 0.02, len(base_vector)))).tolist()

    # Sorgulanan (Questioned) belge de aynı yazarın el ürünü
    questioned = (base_vector * (1.0 + np.random.normal(0, 0.03, len(base_vector)))).tolist()

    return questioned, [ref_1, ref_2, ref_3]


@pytest.fixture
def forged_writer_profile(genuine_writer_profiles: Tuple[List[float], List[List[float]]]) -> Tuple[List[float], List[List[float]]]:
    """Referans yazar profili ile sahte/farklı yazara ait sorgulanan profil üretir."""
    _, references = genuine_writer_profiles

    # Sahte imza: Çok farklı eğim, yüksek titreme (tremor), farklı döngü ve vuruş sayısı
    forged_questioned = [
        -25.0,  # slant_angle_deg (ters eğim!)
        10.0,   # baseline_angle_deg
        12.5,   # baseline_deviation_px (aşırı dalgalanma)
        6.0,    # loop_count (farklı ilmek sayısı)
        12.0,   # crossing_count
        15.0,   # terminal_count
        40.0,   # avg_stroke_length_px
        30.0,   # inter_stroke_spacing_px
        0.45,   # tremor_index_psd (aşırı titreme / sahtecilik kanıtı)
        0.35,   # smoothness_energy
        1.20,   # mean_velocity_proxy
        0.80,   # velocity_variance_proxy
        0.60,   # acceleration_proxy
    ]

    return forged_questioned, references


def test_comparison_genuine_concordant(genuine_writer_profiles: Tuple[List[float], List[List[float]]]):
    """Aynı yazarın belgelerinde yüksek benzerlik ve 0 anlamlı fark oluştuğunu test eder."""
    questioned, references = genuine_writer_profiles
    engine = ForensicComparisonEngine()
    report = engine.compare(questioned, references)

    assert isinstance(report, ComparisonReport)
    assert report.similarity_score >= 0.80
    assert len(report.significant_differences) == 0
    assert len(report.concordant_features) >= 10
    assert "ORİJİNAL" in report.preliminary_opinion
    assert report.audit_hash_sha256 is not None


def test_comparison_forgery_significant_differences(forged_writer_profile: Tuple[List[float], List[List[float]]]):
    """Farklı/sahte yazar örneğinde düşük benzerlik ve çok sayıda anlamlı fark tespit edildiğini test eder."""
    questioned, references = forged_writer_profile
    engine = ForensicComparisonEngine()
    report = engine.compare(questioned, references)

    assert isinstance(report, ComparisonReport)
    assert report.similarity_score < 0.50
    assert len(report.significant_differences) >= 3
    assert "SAHTECİLİK" in report.preliminary_opinion or "FARKLI YAZAR" in report.preliminary_opinion

    # Titreme ve eğim farklarının significant differences listesinde olduğunu doğrula
    sig_keys = [d.feature_key for d in report.significant_differences]
    assert "slant_angle_deg" in sig_keys
    assert "tremor_index_psd" in sig_keys


def test_comparison_with_report_objects():
    """compare metodunun doğrudan ForensicFeatureReport nesneleriyle çalıştığını doğrular."""
    struct = StructuralSpatialFeatures(
        slant_angle_deg=10.0,
        baseline_angle_deg=0.0,
        baseline_deviation_px=2.0,
        loop_count=1,
        crossing_count=2,
        terminal_count=3,
        avg_stroke_length_px=50.0,
        inter_stroke_spacing_px=10.0,
    )
    kinematics = LineQualityFeatures(
        tremor_index_psd=0.02,
        smoothness_energy=0.01,
        mean_velocity_proxy=2.0,
        velocity_variance_proxy=0.2,
        acceleration_proxy=0.1,
    )
    vec = [10.0, 0.0, 2.0, 1.0, 2.0, 3.0, 50.0, 10.0, 0.02, 0.01, 2.0, 0.2, 0.1]
    
    q_rep = ForensicFeatureReport(structural=struct, line_quality=kinematics, feature_vector=vec, metadata={})
    r_rep = ForensicFeatureReport(structural=struct, line_quality=kinematics, feature_vector=vec, metadata={})

    engine = ForensicComparisonEngine()
    report = engine.compare(q_rep, [r_rep])

    assert report.similarity_score > 0.95
    assert len(report.concordant_features) == 13
