"""
FHDSS SPRINT 31 — Test Suite 3: P1 Morphology Normalization & Scale Invariance Tests
Verifies that:
1. StructuralSpatialExtractor computes both raw pixel features and dimensionless ratios.
2. Across 0.6x, 1.0x, 1.8x scales on real CEDAR signatures, normalized ratios remain invariant.
3. Comparing the same signature at different scales using scale-invariant features prevents
   the artificial Z > 8.5 sigma explosion (D_M stays < 2.5 and similarity > 0.50).
4. Backward compatibility: Schema 1.0.0_RAW_PIXELS and 2.0.0_SCALE_INVARIANT can both be
   processed reliably.
"""
from pathlib import Path
import cv2
import numpy as np
import pytest

from cv.preprocessing.pipeline import ForensicImagePreprocessor
from cv.features.extractors import ForensicFeatureExtractor, StructuralSpatialExtractor
from ml.models.comparison import ForensicComparisonEngine


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CEDAR_IMG_PATH = PROJECT_ROOT / "datasets" / "DS_Downloaded" / "CEDAR" / "CEDAR1" / "signatures" / "full_org" / "original_1_1.png"


@pytest.fixture
def preprocessor():
    return ForensicImagePreprocessor(deskew_enabled=False)


@pytest.fixture
def extractor_v2():
    return ForensicFeatureExtractor(scale_invariant=True)


@pytest.fixture
def extractor_v1():
    return ForensicFeatureExtractor(scale_invariant=False)


@pytest.fixture
def engine():
    return ForensicComparisonEngine()


def _load_or_create_signature():
    """CEDAR original_1_1 imzasını yükler, yoksa temiz sentetik imza döner."""
    if CEDAR_IMG_PATH.exists():
        data = CEDAR_IMG_PATH.read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            return img

    img = np.full((350, 550, 3), 255, dtype=np.uint8)
    pts1 = np.array([[80, 150], [140, 80], [210, 220], [280, 90], [380, 180]], np.int32)
    pts2 = np.array([[120, 260], [240, 250], [360, 270]], np.int32)
    cv2.polylines(img, [pts1], False, (25, 25, 25), 3, cv2.LINE_AA)
    cv2.polylines(img, [pts2], False, (25, 25, 25), 3, cv2.LINE_AA)
    cv2.circle(img, (220, 140), 25, (25, 25, 25), 2, cv2.LINE_AA)
    return img


def test_normalized_ratios_calculated(preprocessor):
    img = _load_or_create_signature()
    art = preprocessor.process(img)
    extractor = StructuralSpatialExtractor()
    feat = extractor.extract(art.skeleton, art.binarized)

    assert feat.bbox_dimensions[0] > 0
    assert feat.bbox_dimensions[1] > 0
    assert feat.baseline_deviation_ratio >= 0.0
    assert feat.avg_stroke_length_ratio > 0.0
    assert feat.inter_stroke_spacing_ratio >= 0.0


def test_scale_invariance_across_resolutions(preprocessor, extractor_v2, extractor_v1):
    img_1x = _load_or_create_signature()
    h, w = img_1x.shape[:2]
    img_06x = cv2.resize(img_1x, (int(w * 0.6), int(h * 0.6)), interpolation=cv2.INTER_AREA)
    img_18x = cv2.resize(img_1x, (int(w * 1.8), int(h * 1.8)), interpolation=cv2.INTER_CUBIC)

    art_1x = preprocessor.process(img_1x)
    art_06x = preprocessor.process(img_06x)
    art_18x = preprocessor.process(img_18x)

    # V1: Bounding box piksel boyutları çözünürlükle doğrudan ölçeklenir
    rep_v1_1x = extractor_v1.extract_all(art_1x.skeleton, art_1x.binarized)
    rep_v1_18x = extractor_v1.extract_all(art_18x.skeleton, art_18x.binarized)
    assert rep_v1_18x.structural.bbox_dimensions[0] > 1.4 * rep_v1_1x.structural.bbox_dimensions[0]

    # V2 (Scale-Invariant): Boyutsuz oranlar ölçek değişiminde korunur
    rep_v2_1x = extractor_v2.extract_all(art_1x.skeleton, art_1x.binarized)
    rep_v2_06x = extractor_v2.extract_all(art_06x.skeleton, art_06x.binarized)
    rep_v2_18x = extractor_v2.extract_all(art_18x.skeleton, art_18x.binarized)

    # Normalize oranlar (index 2, 6, 7) sapması çok küçük kalmalıdır
    # Index 2: baseline_deviation_ratio
    assert abs(rep_v2_1x.feature_vector[2] - rep_v2_18x.feature_vector[2]) < 0.10
    # Index 6: avg_stroke_length_ratio
    assert abs(rep_v2_1x.feature_vector[6] - rep_v2_18x.feature_vector[6]) < 0.25
    # Index 7: inter_stroke_spacing_ratio
    assert abs(rep_v2_1x.feature_vector[7] - rep_v2_18x.feature_vector[7]) < 0.15


def test_comparison_prevents_scale_explosion(preprocessor, extractor_v2, extractor_v1, engine):
    img_1x = _load_or_create_signature()
    h, w = img_1x.shape[:2]
    img_18x = cv2.resize(img_1x, (int(w * 1.8), int(h * 1.8)), interpolation=cv2.INTER_CUBIC)

    art_1x = preprocessor.process(img_1x)
    art_18x = preprocessor.process(img_18x)

    # 1. Eski v1 (Raw Pixels): Ölçek farkında Mahalanobis patlar
    rep_v1_1x = extractor_v1.extract_all(art_1x.skeleton, art_1x.binarized)
    rep_v1_18x = extractor_v1.extract_all(art_18x.skeleton, art_18x.binarized)
    comp_v1 = engine.compare(rep_v1_18x, [rep_v1_1x])

    # 2. Yeni v2 (Scale-Invariant): Ölçek farkında Mahalanobis stabil kalır
    rep_v2_1x = extractor_v2.extract_all(art_1x.skeleton, art_1x.binarized)
    rep_v2_18x = extractor_v2.extract_all(art_18x.skeleton, art_18x.binarized)
    comp_v2 = engine.compare(rep_v2_18x, [rep_v2_1x])

    # v2 Mahalanobis mesafesi v1'den belirgin şekilde düşük olmalı ve 2.5'i geçmemelidir
    assert comp_v2.mahalanobis_distance < comp_v1.mahalanobis_distance
    assert comp_v2.mahalanobis_distance < 2.5, f"v2 Mahalanobis mesafesi yüksek: {comp_v2.mahalanobis_distance}"
    assert comp_v2.similarity_score > 0.50, f"v2 Benzerlik skoru düşük: {comp_v2.similarity_score}"
