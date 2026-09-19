import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.core.guard import (
    AllowedForensicScope,
    PsychologicalInferenceGuard,
    PsychologicalInferenceViolationError,
)
from backend.app.main import app
from ml.datasets.forensic_dataset import (
    ForensicAugmentation,
    ForensicDataPipeline,
    ForensicPairDataset,
    ForensicSample,
    PairType,
)
from ml.evaluation.benchmark import ForensicBenchmarkEngine, MetricSummary
from ml.models.siamese import ForensicConvNeXtEncoder, compute_embedding_distance

client = TestClient(app)


# ==============================================================================
# 1. Dataset ve Sıfır Sızıntı (Disjoint-Writer) Testleri
# ==============================================================================

def test_dataset_disjoint_writer_split():
    """Yazarların Train/Val/Test kümelerine sıfır sızıntıyla (ayrık) bölündüğünü test eder."""
    corpus = ForensicDataPipeline.generate_synthetic_forensic_corpus(
        num_writers=10,
        samples_per_writer=4,
        forgeries_per_writer=2,
        seed=101,
    )
    assert len(corpus) == 10 * (4 + 2)

    train, val, test = ForensicDataPipeline.split_by_disjoint_writers(
        corpus, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20, seed=101
    )

    train_writers = set(s.writer_id for s in train)
    val_writers = set(s.writer_id for s in val)
    test_writers = set(s.writer_id for s in test)

    # Kümelerin kesişimi kesinlikle boş olmalı (Zero Data Leakage)
    assert len(train_writers.intersection(val_writers)) == 0
    assert len(train_writers.intersection(test_writers)) == 0
    assert len(val_writers.intersection(test_writers)) == 0
    assert len(train_writers) + len(val_writers) + len(test_writers) == 10


def test_dataset_pair_generation():
    """Pozitif ve Negatif adli çiftlerin doğru etiketlerle üretildiğini test eder."""
    corpus = ForensicDataPipeline.generate_synthetic_forensic_corpus(
        num_writers=6,
        samples_per_writer=4,
        forgeries_per_writer=2,
    )
    pairs = ForensicDataPipeline.create_pairs(corpus, max_pairs_per_writer=5)
    assert len(pairs) > 0

    pair_types = set(p[3] for p in pairs)
    assert PairType.GENUINE_GENUINE in pair_types
    assert PairType.GENUINE_FORGERY in pair_types

    for s_a, s_b, label, p_type in pairs:
        if p_type == PairType.GENUINE_GENUINE:
            assert label == 1.0
            assert s_a.writer_id == s_b.writer_id
            assert not s_a.is_forgery and not s_b.is_forgery
        elif p_type == PairType.GENUINE_FORGERY:
            assert label == 0.0
            assert s_a.is_forgery or s_b.is_forgery
        elif p_type == PairType.DIFFERENT_WRITERS:
            assert label == 0.0
            assert s_a.writer_id != s_b.writer_id


# ==============================================================================
# 2. ConvNeXt-Tiny Derin Model ve Normalizasyon Testleri
# ==============================================================================

def test_convnext_embedding_dimension_and_l2_norm():
    """ConvNeXt modelinin 128 boyutlu L2-normalize birim küre tensörü ürettiğini test eder."""
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    model.eval()

    dummy_input = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        embeddings = model(dummy_input)

    assert embeddings.shape == (4, 128)

    # Her bir satırın L2 normu 1.0 olmalı
    norms = torch.norm(embeddings, p=2, dim=1)
    for n in norms:
        assert pytest.approx(n.item(), abs=1e-4) == 1.0


def test_embedding_distance_bounds():
    """compute_embedding_distance fonksiyonunun [0.0, 1.0] aralığında çalıştığını test eder."""
    emb1 = torch.randn(128)
    emb1 = emb1 / torch.norm(emb1, p=2)

    # Tam aynı vektör
    cos, ang, sim = compute_embedding_distance(emb1, emb1)
    assert pytest.approx(cos, abs=1e-3) == 1.0
    assert pytest.approx(ang, abs=1e-3) == 0.0
    assert pytest.approx(sim, abs=1e-3) == 1.0

    # Zıt vektör
    cos_neg, ang_neg, sim_neg = compute_embedding_distance(emb1, -emb1)
    assert pytest.approx(cos_neg, abs=1e-3) == -1.0
    assert pytest.approx(sim_neg, abs=1e-3) == 0.0


# ==============================================================================
# 3. Üçlü Model Karşılaştırma & Benchmark Testleri
# ==============================================================================

def test_three_models_benchmark_evaluation():
    """Model A (Deep), Model B (Morph) ve Model C (Hybrid) benchmark motorunun çalıştığını test eder."""
    corpus = ForensicDataPipeline.generate_synthetic_forensic_corpus(
        num_writers=6,
        samples_per_writer=4,
        forgeries_per_writer=2,
    )
    pairs = ForensicDataPipeline.create_pairs(corpus, max_pairs_per_writer=4)

    engine = ForensicBenchmarkEngine()
    report = engine.run_benchmark(pairs, save_results=False)

    assert "model_a_deep_only" in report.models
    assert "model_b_morph_only" in report.models
    assert "model_c_hybrid_fhdss" in report.models

    for model_key, summary in report.models.items():
        assert 0.0 <= summary.eer <= 1.0
        assert 0.5 <= summary.auc <= 1.0
        assert 0.0 <= summary.accuracy <= 1.0
        assert 0.0 <= summary.f1_score <= 1.0


# ==============================================================================
# 4. Kapsam Koruması (PsychologicalInferenceGuard) Testleri
# ==============================================================================

def test_scope_guard_allowed_cases():
    """Geçerli adli inceleme kapsamlarının guard tarafından onaylandığını test eder."""
    assert PsychologicalInferenceGuard.validate_text("authorship_comparison") == (True, None)
    assert PsychologicalInferenceGuard.validate_text("forensic_comparison") == (True, None)
    assert PsychologicalInferenceGuard.validate_text("morphological_analysis") == (True, None)
    assert PsychologicalInferenceGuard.validate_text("Şüpheli senet üzerindeki keşideci imzası incelemesi") == (True, None)


def test_scope_guard_forbidden_cases():
    """Yasaklı psikanaliz ve kişilik analizi isteklerinin guard tarafından engellendiğini test eder."""
    # İngilizce terimler
    is_valid, forbidden = PsychologicalInferenceGuard.validate_text("personality traits analysis request")
    assert is_valid is False
    assert forbidden in ["personality", "personality_traits"]

    is_valid, forbidden = PsychologicalInferenceGuard.validate_text("psychological_state assessment")
    assert is_valid is False
    assert forbidden == "psychological_state"

    # Türkçe terimler
    is_valid, forbidden = PsychologicalInferenceGuard.validate_text("Yazarın kişilik ve karakter analizi talebi")
    assert is_valid is False
    assert any(k in forbidden for k in ["kisilik", "kişilik", "karakter"])

    is_valid, forbidden = PsychologicalInferenceGuard.validate_text("İmzadan ruhsal durum ve psikolojik durum tespiti")
    assert is_valid is False
    assert forbidden in ["ruhsal_durum", "psikolojik_durum", "psikoloji"]


def test_api_rejection_for_psychological_inquiries():
    """POST /api/v1/cases uç noktasının kişilik analizi isteklerini HTTP 400 ile reddettiğini test eder."""
    payload = {
        "case_id": "VAKA_FORBIDDEN_001",
        "name": "Şüphelinin Kişilik Analizi Dosyası",
        "description": "El yazısından karakter ve psikolojik durum tespiti talebi",
        "examination_type": "SIGNATURE",
    }

    response = client.post("/api/v1/cases", json=payload)
    assert response.status_code == 400
    data = response.json()
    assert data["detail"]["error"] == "OUT_OF_FORENSIC_SCOPE"
    assert (
        "kişilik" in data["detail"]["detected_forbidden_term"]
        or "karakter" in data["detail"]["detected_forbidden_term"]
        or "psikolojik" in data["detail"]["detected_forbidden_term"]
    )
