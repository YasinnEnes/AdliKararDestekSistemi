import numpy as np
import pytest
import torch
import torch.nn.functional as F

from cv.features.extractors import (
    ForensicFeatureReport,
    LineQualityFeatures,
    StructuralSpatialFeatures,
)
from ml.models.comparison import ComparisonReport, DifferenceType, FeatureDifferenceDetail
from ml.models.fusion import HybridForensicFusion, HybridFusionResult
from ml.models.siamese import (
    ArcFaceMarginLoss,
    ForensicConvNeXtEncoder,
    SupervisedContrastiveLoss,
    compute_embedding_distance,
    set_deterministic_mode,
)


# ==============================================================================
# 1. ConvNeXt-Tiny Derin Metrik Kodlayıcı Testleri
# ==============================================================================

def test_convnext_encoder_dimensions_and_l2_norm():
    """ConvNeXt-Tiny encoder'ın (B, 128) L2-normalize vektör ürettiğini doğrular."""
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    encoder.eval()

    # 4 adet 3-kanallı 224x224 adli görsel tensörü
    x = torch.randn(4, 3, 224, 224)
    with torch.inference_mode():
        embeddings = encoder(x)

    assert embeddings.shape == (4, 128)
    
    # L2 normunun 1.0 (birim küre) olduğunu doğrula
    norms = torch.norm(embeddings, p=2, dim=-1)
    expected_norms = torch.ones(4)
    assert torch.allclose(norms, expected_norms, atol=1e-5)


def test_convnext_encoder_grayscale_and_numpy_support():
    """Tek kanal (Grayscale) ve numpy dizisi girdilerinin başarıyla işlendiğini doğrular."""
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    encoder.eval()

    # 1. Tek kanal tensör (2, 1, 128, 128)
    x_gray = torch.randn(2, 1, 128, 128)
    with torch.inference_mode():
        emb_gray = encoder(x_gray)
    assert emb_gray.shape == (2, 128)

    # 2. Numpy 2D görsel (H, W)
    np_img_2d = np.random.randint(0, 255, size=(100, 200), dtype=np.uint8)
    emb_np_2d = encoder.extract_embedding(np_img_2d)
    assert emb_np_2d.shape == (1, 128)
    assert abs(torch.norm(emb_np_2d, p=2).item() - 1.0) < 1e-4

    # 3. Numpy 3D görsel (H, W, 3)
    np_img_3d = np.random.randint(0, 255, size=(100, 200, 3), dtype=np.uint8)
    emb_np_3d = encoder.extract_embedding(np_img_3d)
    assert emb_np_3d.shape == (1, 128)


def test_deterministic_inference():
    """Modelin aynı girdi tensörü için deterministik olarak özdeş embeddingler ürettiğini doğrular."""
    set_deterministic_mode(42)
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    encoder.eval()

    x = torch.randn(2, 3, 128, 128)
    with torch.inference_mode():
        emb1 = encoder(x)
        emb2 = encoder(x)

    assert torch.allclose(emb1, emb2, atol=1e-6)


def test_arcface_loss_computation_and_backward():
    """ArcFace marjin kaybının hesaplandığını ve gradyan akışını doğrular."""
    num_classes = 5
    embedding_dim = 128
    loss_fn = ArcFaceMarginLoss(num_classes=num_classes, embedding_dim=embedding_dim, scale=16.0, margin=0.30)

    # Gradient gerektiren normalize embedding tensörleri
    raw_emb = torch.randn(4, embedding_dim, requires_grad=True)
    embeddings = F.normalize(raw_emb, p=2, dim=-1)
    labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    loss = loss_fn(embeddings, labels)

    assert loss.item() > 0.0
    assert torch.isfinite(loss)

    # Geriye yayılım (backward pass) testi
    loss.backward()
    assert raw_emb.grad is not None
    assert torch.all(torch.isfinite(raw_emb.grad))


def test_supcon_loss_computation():
    """Supervised Contrastive (SupCon) kaybının çalıştığını doğrular."""
    loss_fn = SupervisedContrastiveLoss(temperature=0.1)

    # 4 örnek: ilk 2'si yazar 0, sonraki 2'si yazar 1
    raw_emb = torch.randn(4, 128, requires_grad=True)
    embeddings = F.normalize(raw_emb, p=2, dim=-1)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss = loss_fn(embeddings, labels)
    assert loss.item() >= 0.0
    assert torch.isfinite(loss)

    loss.backward()
    assert raw_emb.grad is not None


def test_compute_embedding_distance_helpers():
    """Kosinüs benzerliği ve açısal mesafe yardımcı fonksiyonlarını test eder."""
    torch.manual_seed(42)
    e1 = F.normalize(torch.randn(1, 128), p=2, dim=-1)

    # 1. Özdeş Vektörler (cos=1.0, ang_dist=0.0, norm_sim=1.0)
    cos_sim, ang_dist, norm_sim = compute_embedding_distance(e1, e1)
    assert abs(cos_sim - 1.0) < 1e-4
    assert abs(ang_dist - 0.0) < 1e-4
    assert abs(norm_sim - 1.0) < 1e-4

    # 2. Zıt Vektörler (cos=-1.0, ang_dist=1.0, norm_sim=0.0)
    cos_opp, ang_opp, norm_opp = compute_embedding_distance(e1, -e1)
    assert abs(cos_opp - (-1.0)) < 1e-4
    assert abs(ang_opp - 1.0) < 1e-4
    assert abs(norm_opp - 0.0) < 1e-4


# ==============================================================================
# 2. Adli Denetimli Hibrit Füzyon (Hybrid Fusion) Testleri
# ==============================================================================

@pytest.fixture
def mock_genuine_morphological_report() -> ComparisonReport:
    """Yüksek morfolojik uyumlu ve 0 anlamlı fark içeren adli karşılaştırma raporu."""
    return ComparisonReport(
        similarity_score=0.92,
        euclidean_distance=0.25,
        mahalanobis_distance=0.35,
        concordant_features=[],
        natural_variations=[],
        significant_differences=[],
        preliminary_opinion="KUVVETLİ ORİJİNAL DESTEĞİ",
        reference_count=3,
        audit_hash_sha256="mock_hash_genuine",
    )


@pytest.fixture
def mock_forged_morphological_report() -> ComparisonReport:
    """Düşük morfolojik uyumlu ve kritik yapısal çelişkiler içeren sahtecilik raporu."""
    sig_diffs = [
        FeatureDifferenceDetail(
            feature_key="slant_angle_deg",
            feature_name_tr="Yazı Eğim Açısı",
            questioned_value=-25.0,
            reference_mean=12.0,
            reference_std=3.0,
            absolute_difference=37.0,
            z_score=-12.33,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Zıt yönde eğim açısı sapması",
        ),
        FeatureDifferenceDetail(
            feature_key="loop_count",
            feature_name_tr="Kapalı Döngü Sayısı",
            questioned_value=6.0,
            reference_mean=2.0,
            reference_std=0.8,
            absolute_difference=4.0,
            z_score=5.00,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Farklı ilmek/döngü sayısı",
        ),
        FeatureDifferenceDetail(
            feature_key="tremor_index_psd",
            feature_name_tr="Titreme / Tremor İndeksi",
            questioned_value=0.45,
            reference_mean=0.04,
            reference_std=0.05,
            absolute_difference=0.41,
            z_score=8.20,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Yüksek kinematik titreme ve duraksama",
        ),
    ]

    return ComparisonReport(
        similarity_score=0.25,
        euclidean_distance=4.50,
        mahalanobis_distance=7.50,
        concordant_features=[],
        natural_variations=[],
        significant_differences=sig_diffs,
        preliminary_opinion="KUVVETLİ SAHTECİLİK DESTEĞİ",
        reference_count=3,
        audit_hash_sha256="mock_hash_forged",
    )


def test_hybrid_fusion_genuine_concordance(mock_genuine_morphological_report: ComparisonReport):
    """Hem morfolojik hem derin modelin uyumlu olduğu orijinal belge füzyonunu doğrular."""
    fusion = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)
    
    # Derin metrik skoru da yüksek (%91)
    result = fusion.fuse(
        morphological_report=mock_genuine_morphological_report,
        deep_similarity_score=0.91,
    )

    assert isinstance(result, HybridFusionResult)
    assert result.hybrid_similarity_score >= 0.85
    assert result.gating_triggered is False
    assert result.gating_penalty == 0.0
    assert "ORİJİNAL" in result.concordance_level
    assert result.audit_hash_sha256 is not None


def test_hybrid_fusion_gated_penalty_on_contradiction(mock_forged_morphological_report: ComparisonReport):
    """
    Derin model yanılsama ile yüksek benzerlik verse dahi,
    yapısal çelişkiler (zıt eğim, farklı ilmek) sebebiyle adli gating kalkanının devreye girdiğini test eder.
    """
    fusion = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)

    # Derin model yanıltıcı biçimde %90 benzerlik vermiş olsun
    result = fusion.fuse(
        morphological_report=mock_forged_morphological_report,
        deep_similarity_score=0.90,
    )

    assert isinstance(result, HybridFusionResult)
    # Gating tetiklenmiş olmalı ve ceza uygulanmalı
    assert result.gating_triggered is True
    assert result.gating_penalty >= 0.20
    assert len(result.structural_contradictions) >= 2
    # Nihai skor sahtecilik seviyesine (< 0.40) düşürülmüş olmalı
    assert result.hybrid_similarity_score < 0.40
    assert "SAHTECİLİK" in result.concordance_level or "TAM UYUMSUZLUK" in result.concordance_level
    assert "gating kalkanı devreye girmiş" in result.forensic_explanation


def test_hybrid_fusion_direct_feature_reports():
    """Doğrudan ForensicFeatureReport nesneleriyle hibrit füzyonun çalıştığını doğrular."""
    structural_q = StructuralSpatialFeatures(
        slant_angle_deg=65.0,
        baseline_angle_deg=2.0,
        baseline_deviation_px=3.0,
        loop_count=3,
        crossing_count=5,
        terminal_count=4,
        avg_stroke_length_px=80.0,
        inter_stroke_spacing_px=15.0,
    )
    line_quality_q = LineQualityFeatures(
        tremor_index_psd=0.03,
        smoothness_energy=0.02,
        mean_velocity_proxy=2.0,
        velocity_variance_proxy=0.25,
        acceleration_proxy=0.15,
    )
    q_rep = ForensicFeatureReport(
        structural=structural_q,
        line_quality=line_quality_q,
        feature_vector=[65.0, 2.0, 3.0, 3.0, 5.0, 4.0, 80.0, 15.0, 0.03, 0.02, 2.0, 0.25, 0.15],
    )

    structural_ref = StructuralSpatialFeatures(
        slant_angle_deg=64.0,
        baseline_angle_deg=2.1,
        baseline_deviation_px=3.1,
        loop_count=3,
        crossing_count=5,
        terminal_count=4,
        avg_stroke_length_px=79.0,
        inter_stroke_spacing_px=14.8,
    )
    ref_rep1 = ForensicFeatureReport(
        structural=structural_ref,
        line_quality=line_quality_q,
        feature_vector=[64.0, 2.1, 3.1, 3.0, 5.0, 4.0, 79.0, 14.8, 0.03, 0.02, 2.0, 0.25, 0.15],
    )

    fusion = HybridForensicFusion()
    result = fusion.fuse(
        morphological_report=q_rep,
        deep_similarity_score=0.92,
        reference_reports=[ref_rep1],
    )

    assert isinstance(result, HybridFusionResult)
    assert 0.0 <= result.hybrid_similarity_score <= 1.0
    assert "ORİJİNAL" in result.concordance_level or "UYUM" in result.concordance_level


def test_gradual_penalty_smooth_decay():
    """
    Minik bir tekil sapmanın (ör. hafif eğim veya vuruş farkı) skoru aniden sıfırlamadığını,
    kademeli (smooth) ve orantılı bir puan kesintisi yaptığını doğrular.
    """
    fusion = HybridForensicFusion()

    # 1 adet ufak sınır sapması içeren hafif rapor (z = 2.7)
    minor_diff = [
        FeatureDifferenceDetail(
            feature_key="slant_angle_deg",
            feature_name_tr="Yazı Eğim Açısı",
            questioned_value=16.0,
            reference_mean=12.0,
            reference_std=1.5,
            absolute_difference=4.0,
            z_score=2.67,
            diff_type=DifferenceType.NATURAL_VARIATION,
            explanation="Hafif açı sapması",
        )
    ]

    minor_report = ComparisonReport(
        similarity_score=0.88,
        euclidean_distance=0.45,
        mahalanobis_distance=0.60,
        concordant_features=[],
        natural_variations=minor_diff,
        significant_differences=minor_diff,
        preliminary_opinion="DOĞAL VARYASYON",
        reference_count=3,
        audit_hash_sha256="mock_minor_hash",
    )

    result = fusion.fuse(
        morphological_report=minor_report,
        deep_similarity_score=0.89,
    )

    assert isinstance(result, HybridFusionResult)
    # Skor sert biçimde < %50'ye çökmemeli; %75 - %95 bandında kalmalı
    assert result.hybrid_similarity_score >= 0.75
    # Ceza ufak (<= 0.10) olmalı
    assert result.gating_penalty <= 0.10
    assert result.calibration_status == "CALIBRATED_PROBABILISTIC"


def test_probabilistic_score_calibration_bands():
    """Doğal varyasyonlu yüksek uyum durumunda skorun %85 - %98 bandında kalibre edildiğini doğrular."""
    fusion = HybridForensicFusion()

    perfect_report = ComparisonReport(
        similarity_score=0.94,
        euclidean_distance=0.10,
        mahalanobis_distance=0.15,
        concordant_features=[],
        natural_variations=[],
        significant_differences=[],
        preliminary_opinion="KUVVETLİ ORİJİNAL",
        reference_count=3,
        audit_hash_sha256="mock_perfect_hash",
    )

    result = fusion.fuse(
        morphological_report=perfect_report,
        deep_similarity_score=0.93,
    )

    assert result.hybrid_similarity_score >= 0.85
    assert result.hybrid_similarity_score <= 0.98
    assert "ORİJİNAL" in result.concordance_level

