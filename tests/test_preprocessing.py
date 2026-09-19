import cv2
import numpy as np
import pytest

from cv.preprocessing.pipeline import ForensicImagePreprocessor, PreprocessingArtifacts
from cv.preprocessing.suitability import (
    SuitabilityAssessor,
    SuitabilityResult,
    SuitabilityStatus,
)


@pytest.fixture
def synthetic_clean_handwriting() -> np.ndarray:
    """Yüksek kaliteli, net ve yeterli kontrastlı sentetik el yazısı görseli üretir."""
    # 400x400 Beyaz Zemin (255)
    img = np.full((400, 400), 255, dtype=np.uint8)

    # Net, siyah mürekkep çizgileri ve el yazısı simülasyonu
    cv2.putText(img, "Adli Inceleme", (50, 100), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, 0, 2, cv2.LINE_AA)
    cv2.putText(img, "Tubitak BilirKisi", (50, 200), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1.0, 0, 2, cv2.LINE_AA)
    cv2.line(img, (50, 240), (350, 240), 0, 2, cv2.LINE_AA)
    cv2.circle(img, (200, 320), 40, 0, 2, cv2.LINE_AA)

    return img


@pytest.fixture
def synthetic_blurred_handwriting(synthetic_clean_handwriting: np.ndarray) -> np.ndarray:
    """Aşırı odak/hareket bulanıklığı eklenmiş sentetik görüntü."""
    return cv2.GaussianBlur(synthetic_clean_handwriting, (31, 31), 10.0)


@pytest.fixture
def synthetic_low_contrast_handwriting(synthetic_clean_handwriting: np.ndarray) -> np.ndarray:
    """Kontrastı aşırı derecede sıkıştırılmış/soluk sentetik görüntü."""
    # Piksel değerlerini dar bir aralığa ([120, 135]) sıkıştır
    normalized = synthetic_clean_handwriting.astype(np.float32) / 255.0
    compressed = 120.0 + (normalized * 15.0)
    return compressed.astype(np.uint8)


@pytest.fixture
def synthetic_blank_page() -> np.ndarray:
    """Tamamen boş, yazı içermeyen beyaz sayfa."""
    return np.full((400, 400), 255, dtype=np.uint8)


# ==============================================================================
# Suitability Assessment Testleri
# ==============================================================================

def test_suitability_clean_image(synthetic_clean_handwriting: np.ndarray):
    """Temiz ve net görüntünün SUITABLE olarak kabul edildiğini doğrular."""
    assessor = SuitabilityAssessor()
    result = assessor.assess(synthetic_clean_handwriting)

    assert isinstance(result, SuitabilityResult)
    assert result.status == SuitabilityStatus.SUITABLE
    assert result.is_admissible is True
    assert result.overall_score >= 0.80
    assert result.metrics.stroke_density > 0.003
    assert result.metrics.blur_laplacian_var >= assessor.blur_threshold_suitable


def test_suitability_blurred_image(synthetic_blurred_handwriting: np.ndarray):
    """Bulanık görüntünün INSUFFICIENT olarak reddedildiğini ve gerekçelendirildiğini doğrular."""
    assessor = SuitabilityAssessor()
    result = assessor.assess(synthetic_blurred_handwriting)

    assert result.status in (SuitabilityStatus.INSUFFICIENT, SuitabilityStatus.PARTIALLY_SUITABLE)
    assert result.metrics.blur_laplacian_var < assessor.blur_threshold_suitable
    assert any("bulanık" in reason.lower() for reason in result.reasons)


def test_suitability_low_contrast_image(synthetic_low_contrast_handwriting: np.ndarray):
    """Düşük kontrastlı görüntünün tespit edilip gerekçelendirildiğini doğrular."""
    assessor = SuitabilityAssessor()
    result = assessor.assess(synthetic_low_contrast_handwriting)

    assert result.metrics.rms_contrast < assessor.rms_contrast_min
    assert any("kontrast" in reason.lower() for reason in result.reasons)


def test_suitability_blank_image(synthetic_blank_page: np.ndarray):
    """Boş belgenin NOT_COMPARABLE olarak reddedildiğini doğrular."""
    assessor = SuitabilityAssessor()
    result = assessor.assess(synthetic_blank_page)

    assert result.status == SuitabilityStatus.NOT_COMPARABLE
    assert result.is_admissible is False
    assert any("el yazısı/mürekkep" in reason.lower() for reason in result.reasons)


# ==============================================================================
# Preprocessing Pipeline Determinizm ve Bütünlük Testleri
# ==============================================================================

def test_pipeline_determinism(synthetic_clean_handwriting: np.ndarray):
    """Aynı görüntünün ardışık iki kez işlendiğinde birebir aynı çıktıları verdiğini (determinizm) doğrular."""
    preprocessor = ForensicImagePreprocessor()

    artifacts_1 = preprocessor.process(synthetic_clean_handwriting)
    artifacts_2 = preprocessor.process(synthetic_clean_handwriting)

    assert np.array_equal(artifacts_1.grayscale, artifacts_2.grayscale)
    assert np.array_equal(artifacts_1.clahe, artifacts_2.clahe)
    assert np.array_equal(artifacts_1.deskewed, artifacts_2.deskewed)
    assert np.array_equal(artifacts_1.binarized, artifacts_2.binarized)
    assert np.array_equal(artifacts_1.skeleton, artifacts_2.skeleton)


def test_pipeline_non_destructive_original(synthetic_clean_handwriting: np.ndarray):
    """Orijinal girdi matrisinin işlem sırasında hiçbir şekilde değiştirilmediğini (immutable) doğrular."""
    preprocessor = ForensicImagePreprocessor()
    original_copy = synthetic_clean_handwriting.copy()

    artifacts = preprocessor.process(synthetic_clean_handwriting)

    # Girdi dizisi ile işlem sonrasındaki durum birebir aynı kalmalı
    assert np.array_equal(synthetic_clean_handwriting, original_copy)
    assert np.array_equal(artifacts.original, original_copy)


def test_skeletonization_properties(synthetic_clean_handwriting: np.ndarray):
    """İskeletleştirme çıktısının ikili (0/255) ve adli yazı hatlarını içerdiğini doğrular."""
    preprocessor = ForensicImagePreprocessor()
    artifacts = preprocessor.process(synthetic_clean_handwriting)

    skeleton = artifacts.skeleton
    unique_vals = set(np.unique(skeleton))

    # Yalnızca 0 ve 255 değerleri olmalı
    assert unique_vals.issubset({0, 255})
    # Yazı içeren görüntüde en azından iskelet pikselleri olmalı
    assert 255 in unique_vals
    assert np.count_nonzero(skeleton == 255) > 50


def test_deskewing_execution(synthetic_clean_handwriting: np.ndarray):
    """Hafif döndürülmüş görüntünün deskewing fonksiyonundan hatasız geçtiğini doğrular."""
    preprocessor = ForensicImagePreprocessor()
    
    # 5 derece döndürülmüş test görüntüsü
    h, w = synthetic_clean_handwriting.shape[:2]
    rot_mat = cv2.getRotationMatrix2D((w/2, h/2), 5.0, 1.0)
    rotated = cv2.warpAffine(synthetic_clean_handwriting, rot_mat, (w, h), borderValue=255)

    deskewed, angle = preprocessor.deskew(rotated)
    assert deskewed.shape == rotated.shape
    assert isinstance(angle, float)


def test_bounding_box_crop_and_center_alignment(synthetic_clean_handwriting: np.ndarray):
    """
    Farklı konumlarda (sol üst, sağ alt vb.) yer alan aynı imza/yazının
    bounding_box_crop_and_center ile hedef tuvalde (224, 224) tam merkeze oturtulduğunu doğrular.
    """
    preprocessor = ForensicImagePreprocessor()

    # 1. Yazıyı sol üst köşeye yerleştir
    canvas_topleft = np.full((500, 500), 255, dtype=np.uint8)
    canvas_topleft[20:120, 20:200] = cv2.resize(synthetic_clean_handwriting, (180, 100))

    # 2. Aynı yazıyı sağ alt köşeye yerleştir
    canvas_bottomright = np.full((600, 700), 255, dtype=np.uint8)
    canvas_bottomright[400:500, 450:630] = cv2.resize(synthetic_clean_handwriting, (180, 100))

    # Normalizasyon çalıştır
    centered_1, meta_1 = preprocessor.bounding_box_crop_and_center(canvas_topleft, target_size=(224, 224))
    centered_2, meta_2 = preprocessor.bounding_box_crop_and_center(canvas_bottomright, target_size=(224, 224))

    assert centered_1.shape == (224, 224)
    assert centered_2.shape == (224, 224)
    assert meta_1["has_content"] is True
    assert meta_2["has_content"] is True

    # İki farklı konumdaki girdi, merkezlendikten sonra yüksek piksel korelasyonu/benzerliği göstermelidir
    diff = np.abs(centered_1.astype(np.float32) - centered_2.astype(np.float32))
    mae = np.mean(diff)
    assert mae < 25.0  # Konum farkı normalize edilmiş ve hizalanmış olmalıdır


def test_background_noise_cleaning():
    """Tarama sararması ve arka plan aydınlatma gradyanının başarıyla temizlendiğini test eder."""
    preprocessor = ForensicImagePreprocessor()

    # Gradyanlı/gölgeli arka plan (soldan sağa 180'den 240'a aydınlanma)
    h, w = 300, 300
    gradient = np.tile(np.linspace(180, 240, w, dtype=np.uint8), (h, 1))

    # Üzerine siyah mürekkep yazısı ekle
    cv2.putText(gradient, "Imza Test", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 30, 3)

    cleaned = preprocessor.remove_background_noise(gradient)
    bin_res = preprocessor.binarize(cleaned)

    # Temizlenmiş ikili görselde zemin beyaz (255), yazı siyah (0) olmalı ve zemin gölgesi temizlenmeli
    assert bin_res.shape == (300, 300)
    # Köşedeki zemin pikseli temiz beyaz olmalı
    assert bin_res[10, 10] == 255
    assert bin_res[290, 290] == 255

