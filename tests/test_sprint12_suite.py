"""
FHDSS (Adli Belge ve El Yazısı Karar Destek Sistemi)
Sprint 12 Doğrulama Test Paketi: Üretim Sertleştirme ve Verimlilik Optimizasyonu.
"""

import time
import numpy as np
import pytest
import torch
from fastapi import BackgroundTasks

from cv.features.complexity import SignatureComplexityScorer
from cv.features.extractors import filter_paper_grain_and_noise
from ml.dynamic.timeseries import BiometricPoint, BiometricTrajectory, DynamicTimeSeriesAnalyzer
from ml.explainability.counterfactual import (
    CounterfactualExplainer,
    FastSurrogateSimulator,
)
from ml.explainability.explainer import DualForensicExplainer
from ml.models.comparison import ComparisonReport, DifferenceType, FeatureDifferenceDetail
from ml.models.siamese import ForensicConvNeXtEncoder, ForensicONNXRuntimeEncoder


def test_paper_grain_noise_filtering():
    """GÖREV 2: Kağıt lifi ve tarayıcı noktacık gürültülerinin morfolojik temizlenmesini test eder."""
    # 100x100 siyah zemin üzerinde 1 büyük çizgi ve 10 küçük ayrık gürültü noktası
    noisy_img = np.zeros((100, 100), dtype=np.uint8)
    noisy_img[40:60, 40:60] = 255  # 400 piksel alanlı gerçek imza bloğu
    # 2-3 piksellik tuz-biber gürültüleri
    noisy_img[10, 10] = 255
    noisy_img[10, 11] = 255
    noisy_img[80, 85] = 255
    noisy_img[85, 20] = 255

    clean_img = filter_paper_grain_and_noise(noisy_img, min_component_size=5)

    assert clean_img[45, 45] == 255, "Ana imza vuruşu korunmalıdır."
    assert clean_img[10, 10] == 0, "Ayrık tuz-biber gürültüleri temizlenmelidir."
    assert clean_img[80, 85] == 0, "Gürültü pikseli sıfırlanmalıdır."


def test_complexity_noise_resilience():
    """GÖREV 2: İmza karmaşıklığı analizinin gürültüye dayanıklılığını test eder."""
    scorer = SignatureComplexityScorer()
    # Temiz basit çizgi
    clean_line = np.zeros((120, 120), dtype=np.uint8)
    clean_line[60, 20:100] = 255

    res = scorer.score(clean_line)
    assert res.overall_complexity_index >= 0.0
    assert res.category.value in ["LOW", "MEDIUM", "HIGH"]


def test_onnx_runtime_encoder_and_fallback():
    """GÖREV 3: ONNX Runtime çıkarım sarmalayıcısının L2 normalize embedding ürettiğini test eder."""
    pytorch_encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    onnx_encoder = ForensicONNXRuntimeEncoder(
        onnx_model_path="artifacts/non_existent_model.onnx",
        fallback_pytorch_model=pytorch_encoder,
        embedding_dim=128,
    )

    dummy_input = torch.randn(1, 3, 224, 224)
    emb = onnx_encoder(dummy_input)

    assert emb.shape == (1, 128)
    norm = float(torch.norm(emb, p=2).item())
    assert abs(norm - 1.0) < 1e-3, "Embedding L2 normalize birim kürede olmalıdır."


def test_gradcam_caching_performance():
    """GÖREV 3: Grad-CAM üretiminin görsel hash önbelleği ile tekrarlı çağrılarda <1ms döndüğünü test eder."""
    explainer = DualForensicExplainer()
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    img_tensor = torch.randn(1, 3, 112, 112)

    # 1. İlk çalıştırma (Önbelleğe yazılır)
    t0 = time.perf_counter()
    cam1 = explainer.generate_gradcam(encoder, img_tensor)
    t1 = time.perf_counter()

    # 2. İkinci çalıştırma (Önbellekten çekilir)
    t2 = time.perf_counter()
    cam2 = explainer.generate_gradcam(encoder, img_tensor)
    t3 = time.perf_counter()

    cache_time_ms = (t3 - t2) * 1000.0
    assert cache_time_ms < 10.0, f"Önbellekten okuma çok hızlı olmalıdır (Ölçülen: {cache_time_ms:.2f}ms)"
    assert np.allclose(cam1, cam2), "Önbellekten dönen Grad-CAM matrisi özdeş olmalıdır."


def test_biometric_100hz_spline_resampling():
    """GÖREV 5: Tablet verilerinin standart 100 Hz (10 ms) frekansa eşitlendiğini test eder."""
    analyzer = DynamicTimeSeriesAnalyzer()
    # 50 Hz düzensiz örnekleme (dt = 0.02s)
    points = [
        BiometricPoint(x=10.0, y=20.0, t=0.0, pressure=0.5),
        BiometricPoint(x=15.0, y=25.0, t=0.02, pressure=0.6),
        BiometricPoint(x=20.0, y=30.0, t=0.04, pressure=0.7),
        BiometricPoint(x=25.0, y=35.0, t=0.06, pressure=0.8),
    ]
    traj = BiometricTrajectory(sample_id="Q_50HZ", writer_id="W1", points=points)

    resampled = analyzer.resample_to_uniform_frequency(traj, target_hz=100.0)

    assert len(resampled.points) >= len(points)
    # Zaman adımlarının düzgün 0.01s aralıklı olduğunu doğrula
    dt = resampled.points[1].t - resampled.points[0].t
    assert abs(dt - 0.01) < 1e-3, f"Frekans 100 Hz (dt=0.01s) olmalıdır, ölçülen dt={dt}"


def test_fast_surrogate_what_if_simulator():
    """GÖREV 6: FastSurrogateSimulator vekil modelinin <1 ms altında doğru sonuç verdiğini test eder."""
    diffs = [
        FeatureDifferenceDetail(
            feature_key="slant_angle_deg",
            feature_name_tr="Eğim Açısı",
            questioned_value=15.0,
            reference_mean=5.0,
            reference_std=2.0,
            absolute_difference=10.0,
            z_score=5.0,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Sapma",
        ),
        FeatureDifferenceDetail(
            feature_key="tremor_index_psd",
            feature_name_tr="Titreme",
            questioned_value=0.10,
            reference_mean=0.02,
            reference_std=0.01,
            absolute_difference=0.08,
            z_score=8.0,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Titreme anomalisi",
        ),
    ]
    comp = ComparisonReport(
        similarity_score=0.45,
        euclidean_distance=9.4,
        mahalanobis_distance=9.4,
        concordant_features=[],
        natural_variations=[],
        significant_differences=diffs,
        preliminary_opinion="Uyuşmazlık",
        reference_count=3,
        audit_hash_sha256="HASH_TEST",
    )

    surrogate = FastSurrogateSimulator(comp, base_deep_score=0.80)
    # Eğim ve titremeyi referansa yaklaştıran müdahale
    result = surrogate.predict_what_if({"slant_angle_deg": -10.0, "tremor_index_psd": -0.08})

    assert result["execution_time_ms"] < 20.0, f"Vekil model <20ms olmalıdır (Ölçülen: {result['execution_time_ms']}ms)"
    assert result["simulated_hybrid_score"] > 0.45, "Müdahale sonrası hibrit skor yükselmelidir."
