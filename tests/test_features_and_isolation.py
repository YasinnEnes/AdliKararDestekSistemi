import math
import cv2
import numpy as np
import pytest

from backend.app.core.context_isolation import (
    ContextIsolationOrchestrator,
    IsolationState,
    IsolationViolationError,
)
from cv.features.extractors import (
    ForensicFeatureExtractor,
    ForensicFeatureReport,
    LineQualityKinematicsExtractor,
    StructuralSpatialExtractor,
)


# ==============================================================================
# 1. Bilişsel İzolasyon (Context Isolation) Testleri
# ==============================================================================

def test_context_isolation_happy_path():
    """Phase A -> Freeze -> Phase B -> Complete adımlarının kurallara uygun çalıştığını doğrular."""
    orchestrator = ContextIsolationOrchestrator(case_id="DOSYA_2026_001", examiner_id="BILIRKISI_42")
    assert orchestrator.state == IsolationState.PHASE_A_INIT

    # 1. Phase A Başlat
    orchestrator.start_phase_a(questioned_sample_id="Q_BELGE_01", document_hash="sha256_mock_hash_123")
    assert orchestrator.state == IsolationState.PHASE_A_ANALYSIS

    # 2. Phase A Özniteliklerini Çıkar ve Dondur (Freeze)
    features_mock = {"slant_angle_deg": 14.5, "tremor_index_psd": 0.04, "loop_count": 3}
    frozen = orchestrator.freeze_phase_a(features=features_mock, examiner_notes="İmza akıcı, sol eğimli.")
    
    assert orchestrator.state == IsolationState.PHASE_A_FROZEN
    assert frozen.integrity_hash_sha256 is not None
    assert len(frozen.integrity_hash_sha256) == 64  # SHA-256 uzunluğu
    assert orchestrator.verify_integrity() is True

    # 3. Phase B (Referans Belgeler) Kilidini Aç
    unlocked = orchestrator.unlock_phase_b(reference_sample_ids=["REF_01", "REF_02", "REF_03"])
    assert unlocked is True
    assert orchestrator.state == IsolationState.PHASE_B_ACTIVE

    # 4. Değerlendirmeyi Tamamla
    orchestrator.complete_evaluation()
    assert orchestrator.state == IsolationState.EVALUATION_COMPLETED


def test_context_isolation_violation_premature_phase_b():
    """Phase A dondurulmadan Phase B açılmak istendiğinde IsolationViolationError fırlatıldığını doğrular."""
    orchestrator = ContextIsolationOrchestrator(case_id="DOSYA_2026_002")
    orchestrator.start_phase_a(questioned_sample_id="Q_BELGE_02", document_hash="hash_abc")

    # Phase A henüz dondurulmadı (ANALYSIS durumunda) -> Phase B açılamaz!
    with pytest.raises(IsolationViolationError) as excinfo:
        orchestrator.unlock_phase_b(reference_sample_ids=["REF_01"])

    assert "Phase A dondurulmadan" in str(excinfo.value)


def test_context_isolation_tamper_detection():
    """Dondurulan Phase A kanıtı sonradan manipüle edilirse bütünlük doğrulamasının yakaladığını test eder."""
    orchestrator = ContextIsolationOrchestrator(case_id="DOSYA_2026_003")
    orchestrator.start_phase_a(questioned_sample_id="Q_03", document_hash="hash_xyz")
    orchestrator.freeze_phase_a(features={"slant": 10.0})

    # Manipülasyon simülasyonu
    orchestrator.frozen_evidence.features["slant"] = 999.0

    # Bütünlük denetimi başarısız olmalı
    assert orchestrator.verify_integrity() is False

    # Manipüle edilmiş durumda Phase B açılamaz
    with pytest.raises(IsolationViolationError) as excinfo:
        orchestrator.unlock_phase_b(reference_sample_ids=["REF_01"])

    assert "bütünlük doğrulaması başarısız" in str(excinfo.value).lower()


# ==============================================================================
# 2. Adli Öznitelik Çıkarım (Feature Extraction) Testleri
# ==============================================================================

@pytest.fixture
def synthetic_cross_skeleton() -> np.ndarray:
    """1 kesişim (junction) ve 4 uç nokta (terminal) içeren sentetik '+' iskeleti."""
    canvas = np.zeros((100, 100), dtype=np.uint8)
    canvas[50, 20:80] = 255  # Yatay çizgi
    canvas[20:80, 50] = 255  # Dikey çizgi
    return canvas


@pytest.fixture
def synthetic_t_skeleton() -> np.ndarray:
    """1 kesişim (junction) ve 3 uç nokta (terminal) içeren sentetik 'T' iskeleti."""
    canvas = np.zeros((100, 100), dtype=np.uint8)
    canvas[30, 20:80] = 255  # Yatay üst çizgi
    canvas[30:80, 50] = 255  # Dikey gövde
    return canvas


def test_topological_points_cross_and_t(synthetic_cross_skeleton: np.ndarray, synthetic_t_skeleton: np.ndarray):
    """Topolojik uç nokta (terminal) ve kesişim (crossing) analizini doğrular."""
    extractor = StructuralSpatialExtractor()

    # '+' şekli: 4 uç, en az 1 kesişim
    term_cross, junc_cross = extractor.extract_topological_points(synthetic_cross_skeleton)
    assert term_cross == 4
    assert junc_cross >= 1

    # 'T' şekli: 3 uç, en az 1 kesişim
    term_t, junc_t = extractor.extract_topological_points(synthetic_t_skeleton)
    assert term_t == 3
    assert junc_t >= 1


def test_slant_angle_slanted_line():
    """45 derece eğimli çizginin eğim açısının doğru hesaplandığını doğrular."""
    canvas = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(canvas, (30, 30), (170, 170), 255, 1)

    extractor = StructuralSpatialExtractor()
    slant = extractor.calculate_slant_angle(canvas)

    # 45 derece doğrultusunda olmalı (küçük toleransla)
    assert abs(abs(slant) - 45.0) < 5.0


def test_tremor_and_smoothness_distinction():
    """Pürüzsüz akıcı çizgi ile titrek (tremor) çizginin FFT spektral farkını doğrular."""
    # 1. Pürüzsüz Çizgi
    smooth_canvas = np.zeros((300, 300), dtype=np.uint8)
    pts_smooth = []
    for x in range(30, 270):
        y = int(150 + 40 * math.sin(x / 40.0))
        pts_smooth.append((x, y))
    for i in range(len(pts_smooth) - 1):
        cv2.line(smooth_canvas, pts_smooth[i], pts_smooth[i+1], 255, 1)

    # 2. Titrek / Kesintili Çizgi (Yüksek frekanslı salınım eklenmiş)
    shaky_canvas = np.zeros((300, 300), dtype=np.uint8)
    pts_shaky = []
    for x in range(30, 270):
        y = int(150 + 40 * math.sin(x / 40.0) + 8 * math.sin(x * 1.8))
        pts_shaky.append((x, y))
    for i in range(len(pts_shaky) - 1):
        cv2.line(shaky_canvas, pts_shaky[i], pts_shaky[i+1], 255, 1)

    extractor = LineQualityKinematicsExtractor()
    res_smooth = extractor.extract(smooth_canvas)
    res_shaky = extractor.extract(shaky_canvas)

    # Titrek çizginin Tremor PSD indeksi ve eğrilik enerjisi belirgin biçimde daha yüksek olmalıdır
    assert res_shaky.tremor_index_psd > res_smooth.tremor_index_psd
    assert res_shaky.smoothness_energy > res_smooth.smoothness_energy


def test_forensic_feature_extractor_full():
    """Tüm özniteliklerin çıkarılarak ForensicFeatureReport yapısında paketlendiğini doğrular."""
    canvas = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(canvas, (30, 50), (170, 50), 255, 1)
    cv2.line(canvas, (100, 50), (100, 150), 255, 1)
    
    extractor = ForensicFeatureExtractor()
    report = extractor.extract_all(skeleton_image=canvas, binary_image=canvas)

    assert isinstance(report, ForensicFeatureReport)
    assert len(report.feature_vector) == 13
    assert report.structural.terminal_count > 0
    assert report.metadata["total_feature_dimensions"] == 13
