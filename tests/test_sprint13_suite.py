"""
FHDSS (Adli Belge ve El Yazısı Karar Destek Sistemi)
Sprint 13 Doğrulama Test Paketi: Kararlılık, Ön İşleme Normalizasyonu ve Skor Kalibrasyonu.
"""

import cv2
import numpy as np
import pytest
import torch

from cv.preprocessing.pipeline import ForensicImagePreprocessor, PreprocessingArtifacts
from ml.models.comparison import ComparisonReport, DifferenceType, FeatureDifferenceDetail
from ml.models.fusion import HybridForensicFusion, HybridFusionResult


def test_task1_bounding_box_crop_and_center_invariance():
    """GÖREV 1: Farklı konumlardaki ve aşırı boşluklu aynı imzanın doğru hizalanıp merkezlendiğini doğrular."""
    preprocessor = ForensicImagePreprocessor()

    # Temel imza deseni
    base_sig = np.full((100, 200), 255, dtype=np.uint8)
    cv2.putText(base_sig, "Adli Imza", (20, 60), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.2, 0, 2)

    # 1. Tuvalin sol üst köşesinde
    canvas1 = np.full((600, 600), 255, dtype=np.uint8)
    canvas1[30:130, 40:240] = base_sig

    # 2. Tuvalin sağ alt köşesinde
    canvas2 = np.full((700, 800), 255, dtype=np.uint8)
    canvas2[450:550, 480:680] = base_sig

    centered_1, meta_1 = preprocessor.bounding_box_crop_and_center(canvas1, target_size=(224, 224))
    centered_2, meta_2 = preprocessor.bounding_box_crop_and_center(canvas2, target_size=(224, 224))

    assert centered_1.shape == (224, 224)
    assert centered_2.shape == (224, 224)
    assert meta_1["has_content"] is True
    assert meta_2["has_content"] is True

    # Merkezleme sonrası iki görselin MAE farkı düşük olmalıdır (hizalama başarısı)
    mae = np.mean(np.abs(centered_1.astype(np.float32) - centered_2.astype(np.float32)))
    assert mae < 20.0, f"Hizalama MAE farkı ({mae:.2f}) beklenenden yüksek."


def test_task2_background_illumination_and_speckle_cleaning():
    """GÖREV 2: Tarama sararması/aydınlatma dalgalanması ve izole toz noktacıklarının temizlendiğini doğrular."""
    preprocessor = ForensicImagePreprocessor(denoise_enabled=True, clean_speckle_noise=True)

    # Gradyanlı ve sararmış zemin simülasyonu
    h, w = 250, 250
    bg = np.tile(np.linspace(170, 235, w, dtype=np.uint8), (h, 1))

    # Ana yazı vuruşu ekle
    cv2.putText(bg, "Test", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.4, 20, 3)

    # Ayrık 1-2 piksel toz noktaları ekle
    bg[20, 20] = 0
    bg[20, 21] = 0
    bg[220, 220] = 0

    artifacts = preprocessor.process(bg)

    assert isinstance(artifacts, PreprocessingArtifacts)
    assert artifacts.normalized_crop.shape == (224, 224)
    # Köşe piksellerinde zemin temizliği (255 beyaz)
    assert artifacts.binarized[10, 10] == 255
    # İzole toz noktası temizlenmiş olmalı
    assert artifacts.binarized[20, 20] == 255


def test_task3_gradual_penalty_curve_no_harsh_drop():
    """GÖREV 3: Kademeli ceza eğrisinin küçük sapmalarda skoru sertçe sıfırlamadığını doğrular."""
    fusion = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)

    # 1 adet sınır değerinde sapma (z = 2.6)
    minor_diff = [
        FeatureDifferenceDetail(
            feature_key="inter_stroke_spacing_px",
            feature_name_tr="Vuruşlar Arası Mesafe",
            questioned_value=22.0,
            reference_mean=18.0,
            reference_std=1.5,
            absolute_difference=4.0,
            z_score=2.60,
            diff_type=DifferenceType.NATURAL_VARIATION,
            explanation="Vuruş aralığında hafif varyasyon",
        )
    ]

    report = ComparisonReport(
        similarity_score=0.90,
        euclidean_distance=0.30,
        mahalanobis_distance=0.45,
        concordant_features=[],
        natural_variations=minor_diff,
        significant_differences=minor_diff,
        preliminary_opinion="DOĞAL VARYASYON",
        reference_count=3,
        audit_hash_sha256="mock_gradual_hash",
    )

    res = fusion.fuse(morphological_report=report, deep_similarity_score=0.91)

    # Kademeli ceza ufak olmalı (<= 0.08) ve skor yüksek (%80+) kalmalı
    assert res.gating_penalty <= 0.08, f"Ceza beklenenden yüksek: {res.gating_penalty}"
    assert res.hybrid_similarity_score >= 0.80, f"Skor gereksiz çöktü: {res.hybrid_similarity_score}"
    assert res.calibration_status == "CALIBRATED_PROBABILISTIC"


def test_task4_probabilistic_score_calibration():
    """GÖREV 4: Olasılıksal kalibrasyonun samimi eşleşmeleri %85-%98 aralığına yerleştirdiğini doğrular."""
    fusion = HybridForensicFusion()

    concordant_report = ComparisonReport(
        similarity_score=0.91,
        euclidean_distance=0.20,
        mahalanobis_distance=0.30,
        concordant_features=[],
        natural_variations=[],
        significant_differences=[],
        preliminary_opinion="KUVVETLİ ORİJİNAL",
        reference_count=3,
        audit_hash_sha256="mock_concordant_hash",
    )

    res = fusion.fuse(morphological_report=concordant_report, deep_similarity_score=0.90)

    # Doğal varyasyon bandında (%85 - %98) kümelenme
    assert 0.85 <= res.hybrid_similarity_score <= 0.98
    assert "ORİJİNAL" in res.concordance_level


def test_task5_contradiction_severity_and_penalty_details():
    """GÖREV 5: Ağır yapısal çelişkilerde (zıt eğim) kademeli ceza dökümünün üretildiğini doğrular."""
    fusion = HybridForensicFusion()

    severe_diff = [
        FeatureDifferenceDetail(
            feature_key="slant_angle_deg",
            feature_name_tr="Yazı Eğim Açısı",
            questioned_value=-28.0,
            reference_mean=14.0,
            reference_std=2.0,
            absolute_difference=42.0,
            z_score=-21.0,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Zıt eğim çelişkisi",
        )
    ]

    report = ComparisonReport(
        similarity_score=0.20,
        euclidean_distance=5.0,
        mahalanobis_distance=8.0,
        concordant_features=[],
        natural_variations=[],
        significant_differences=severe_diff,
        preliminary_opinion="KUVVETLİ SAHTECİLİK",
        reference_count=3,
        audit_hash_sha256="mock_severe_hash",
    )

    res = fusion.fuse(morphological_report=report, deep_similarity_score=0.88)

    assert res.gating_triggered is True
    assert res.gating_penalty >= 0.20
    assert len(res.penalty_details) > 0
    assert res.hybrid_similarity_score < 0.40
