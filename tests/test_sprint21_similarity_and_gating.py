import math
import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cv.preprocessing.pipeline import ForensicImagePreprocessor
from cv.features.extractors import ForensicFeatureExtractor, ForensicFeatureReport
from ml.models.comparison import ForensicComparisonEngine, ComparisonReport, DifferenceType, FeatureDifferenceDetail
from ml.models.fusion import HybridForensicFusion, HybridFusionResult, ScoreStatus, ScoreProvenance
from ml.models.siamese import ForensicConvNeXtEncoder, compute_cosine_similarity, compute_normalized_similarity
from ml.preprocessing.contract import preprocess_for_inference
from scripts.debug_similarity import create_demo_img


def test_deskew_disabled_by_default_in_pipeline():
    """1. Pipeline varsayılan olarak serbest imzalarda deskew'i kapalı tutmalıdır."""
    preprocessor = ForensicImagePreprocessor()
    img = np.full((200, 400, 3), 255, dtype=np.uint8)
    cv2.line(img, (50, 100), (350, 100), (20, 20, 20), 2)
    
    art = preprocessor.process(img)
    # deskew_enabled=False olduğunda deskew_angle_deg 0.0 olmalı
    assert art.metadata.get("deskew_angle_deg") == 0.0


def test_single_reference_mahalanobis_does_not_explode():
    """2. Tek referanslı karşılaştırmada genişletilmiş adli prior devreye girmeli, Mahalanobis patlamamalıdır."""
    engine = ForensicComparisonEngine()
    extractor = ForensicFeatureExtractor()
    preprocessor = ForensicImagePreprocessor(deskew_enabled=False)

    img_q = create_demo_img(slant_angle=14.5, with_tremor=False)
    img_r1 = create_demo_img(slant_angle=14.0, with_tremor=False)

    art_q = preprocessor.process(img_q)
    art_r = preprocessor.process(img_r1)

    feat_q = extractor.extract_all(art_q.skeleton, art_q.binarized)
    feat_r = extractor.extract_all(art_r.skeleton, art_r.binarized)

    comp = engine.compare(feat_q, [feat_r])
    # Tek referansta bile Mahalanobis 15-20'ye patlamamalı, < 4.0 kalmalıdır
    assert comp.mahalanobis_distance < 4.0, f"Mahalanobis tek referansta patladı: {comp.mahalanobis_distance}"
    assert comp.similarity_score > 0.35, f"Morfolojik benzerlik çok düşük: {comp.similarity_score}"


def test_continuous_gating_never_zeroes_out_high_deep_similarity():
    """3. Sürekli Gating (Continuous Gating): Yüksek derin benzerlikte skor asla 0.0000 olmamalıdır."""
    fusion = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)

    # Yapay olarak morfolojik uyuşmazlık içeren bir rapor oluştur
    diff = FeatureDifferenceDetail(
        feature_key="slant_angle_deg",
        feature_name_tr="Yazı Eğim Açısı",
        questioned_value=25.0,
        reference_mean=10.0,
        reference_std=2.0,
        absolute_difference=15.0,
        z_score=7.5,
        diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
        explanation="Ciddi eğim sapması",
    )
    fake_comp = ComparisonReport(
        similarity_score=0.15,
        euclidean_distance=5.0,
        mahalanobis_distance=7.5,
        concordant_features=[],
        natural_variations=[],
        significant_differences=[diff],
        preliminary_opinion="Belirgin Uyuşmazlık",
        reference_count=1,
        audit_hash_sha256="test_hash",
    )

    # Derin model ise %95 benzerlik tespit etmiş olsun
    res = fusion.fuse(fake_comp, deep_similarity_score=0.95)

    # Skor asla 0.0 olamaz! Sürekli gating ile dengelenmeli
    assert res.hybrid_similarity_score > 0.25, f"Skor sıfırlandı: {res.hybrid_similarity_score}"
    assert res.gating_weight > 0.15, f"Gating ağırlığı sıfırlandı: {res.gating_weight}"
    assert res.score_provenance is not None
    assert res.score_provenance.status in [ScoreStatus.VALID_SCORE, ScoreStatus.GATING_REJECTION]
    assert res.score_provenance.final_score == res.hybrid_similarity_score


def test_score_provenance_audit_object_integrity():
    """4. Her analiz sonucunda tam Score Provenance nesnesi ve gerekçesi bulunmalıdır."""
    fusion = HybridForensicFusion()
    preprocessor = ForensicImagePreprocessor(deskew_enabled=False)
    extractor = ForensicFeatureExtractor()
    engine = ForensicComparisonEngine()

    img = create_demo_img(14.5)
    art = preprocessor.process(img)
    feat = extractor.extract_all(art.skeleton, art.binarized)
    comp = engine.compare(feat, [feat])

    res = fusion.fuse(comp, deep_similarity_score=1.0)
    prov = res.score_provenance

    assert prov is not None
    assert prov.deep_similarity_raw == 1.0
    assert prov.morphological_similarity == 1.0
    assert prov.mahalanobis_distance == 0.0
    assert prov.final_score >= 0.985
    assert prov.status == ScoreStatus.VALID_SCORE
