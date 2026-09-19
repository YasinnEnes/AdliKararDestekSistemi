"""
FHDSS - Sprint 18: Gerçek Veri Setleri, Kalite Denetimi ve Model Öncesi Test Paketi
Sprint 18.11 Pre-Training Test Suite
"""

import hashlib
import json
import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from ml.datasets.forensic_dataset import (
    ForensicAugmentation,
    ForensicDataPipeline,
    ForensicSample,
    ForensicWriterDataset,
)
from ml.models.registry import model_registry
from ml.models.siamese import (
    ArcFaceMarginLoss,
    ForensicConvNeXtEncoder,
    SupervisedContrastiveLoss,
    compute_normalized_similarity,
)
from ml.preprocessing.contract import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    pad_to_square,
    preprocess_for_inference,
    to_rgb,
)
from ml.training.dataset_split_validator import (
    detect_writer_overlap,
    generate_split_report,
    validate_writer_disjointness,
)
from ml.training.preflight import PreflightChecker
from ml.training.train import (
    BalancedBatchSampler,
    ForensicModelTrainer,
    TrainingConfig,
    set_deterministic_seed,
)
from scripts.ingest_all_datasets import ForensicDatasetIngestionPipeline


@pytest.fixture
def cedar_path() -> Path:
    p = Path(__file__).resolve().parent.parent / "datasets" / "DS_Downloaded" / "CEDAR" / "CEDAR1" / "signatures"
    return p


# ------------------------------------------------------------------------------
# 1. Dataset Envanteri & Manifest Testi (Sprint 18.1)
# ------------------------------------------------------------------------------
def test_18_1_dataset_inventory_manifest_structure():
    inv_path = Path(__file__).resolve().parent.parent / "datasets" / "manifests" / "dataset_inventory.json"
    assert inv_path.exists(), "dataset_inventory.json manifest dosyası eksik!"

    with open(inv_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("total_datasets") == 6
    ds_names = [d["name"] for d in data.get("datasets", [])]
    for required_name in ["CEDAR", "BHSig-260", "UTSig", "CVL Database", "IAM_HW", "SVC2004"]:
        assert required_name in ds_names, f"{required_name} veri seti envanterde yer almıyor!"


# ------------------------------------------------------------------------------
# 2. Dataset Ingestion & Heuristic Parsing Testi (Sprint 18.2)
# ------------------------------------------------------------------------------
def test_18_2_dataset_ingestion_parsing():
    pipeline = ForensicDatasetIngestionPipeline()
    # Test BHSig-260 parsing
    bhsig_files = ["Reconstructed_BHSig100/Forged/B-S-25-F-01.tif", "Reconstructed_BHSig100/Genuine/H-S-42-G-01.tif"]
    writers_bhsig = pipeline.discover_writers_and_samples("BHSig-260", bhsig_files)
    assert "BHSig_B-S-25" in writers_bhsig
    assert "BHSig_H-S-42" in writers_bhsig
    assert writers_bhsig["BHSig_B-S-25"][0]["is_forgery"] is True
    assert writers_bhsig["BHSig_H-S-42"][0]["is_forgery"] is False

    # Test CEDAR parsing
    cedar_files = ["full_org/original_10_2.png", "full_forg/forgeries_10_2.png"]
    writers_cedar = pipeline.discover_writers_and_samples("CEDAR", cedar_files)
    assert "CEDAR_W_010" in writers_cedar
    assert len(writers_cedar["CEDAR_W_010"]) == 2


# ------------------------------------------------------------------------------
# 3. CEDAR Doğrulama ve Recursive Yükleme Testi (Sprint 18.3)
# ------------------------------------------------------------------------------
def test_18_3_cedar_recursive_loading_and_labels(cedar_path):
    if not cedar_path.exists():
        pytest.skip("CEDAR fiziksel verisi mevcut değil.")

    samples = ForensicDataPipeline.load_from_directory(cedar_path)
    assert len(samples) == 2640, f"CEDAR 2,640 görsel içermeli, bulunan: {len(samples)}"
    writers = set(s.writer_id for s in samples)
    assert len(writers) == 55, f"CEDAR 55 yazar içermeli, bulunan: {len(writers)}"

    genuines = [s for s in samples if not s.is_forgery]
    forgeries = [s for s in samples if s.is_forgery]
    assert len(genuines) == 1320, "Orijinal imza sayısı 1320 olmalı"
    assert len(forgeries) == 1320, "Sahte imza sayısı 1320 olmalı"


# ------------------------------------------------------------------------------
# 4. Writer Leakage & Disjointness Enforcement Testi (Sprint 18.4)
# ------------------------------------------------------------------------------
def test_18_4_writer_disjointness_enforcement_and_leakage_rejection():
    # Temiz ayrım
    train = [ForensicSample(sample_id="s1", writer_id="W1"), ForensicSample(sample_id="s2", writer_id="W2")]
    val = [ForensicSample(sample_id="s3", writer_id="W3")]
    test = [ForensicSample(sample_id="s4", writer_id="W4")]

    assert validate_writer_disjointness(train, val, test) is True
    report = generate_split_report(train, val, test)
    assert report["is_disjoint"] is True
    assert report["overlaps"]["total_leaked_count"] == 0

    # Sızıntı enjeksiyonu: W1 val kümesine de eklenirse hata fırlatmalı
    leaked_val = val + [ForensicSample(sample_id="s_leak", writer_id="W1")]
    with pytest.raises(ValueError) as exc:
        validate_writer_disjointness(train, leaked_val, test)
    assert "Writer Leakage" in str(exc.value)


# ------------------------------------------------------------------------------
# 5. Duplicate Detection Testi (Sprint 18.5)
# ------------------------------------------------------------------------------
def test_18_5_duplicate_detection_integrity(cedar_path):
    if not cedar_path.exists():
        pytest.skip("CEDAR fiziksel verisi mevcut değil.")

    samples = ForensicDataPipeline.load_from_directory(cedar_path)
    paths = [s.image_path for s in samples]
    assert len(paths) == len(set(paths)), "CEDAR datasetinde mükerrer dosya yolu bulundu!"


# ------------------------------------------------------------------------------
# 6. Corrupted Image Handling Testi (Sprint 18.5)
# ------------------------------------------------------------------------------
def test_18_5_corrupted_image_handling(tmp_path):
    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"NOT_A_REAL_IMAGE_DATA")

    bad_sample = ForensicSample(sample_id="bad1", writer_id="W_BAD", image_path=str(corrupt_file))
    dataset = ForensicWriterDataset([bad_sample])

    with pytest.raises(ValueError):
        _ = dataset[0]


# ------------------------------------------------------------------------------
# 7. BalancedBatchSampler Gerçek Veri Testi (Sprint 18.6)
# ------------------------------------------------------------------------------
def test_18_6_balanced_batch_sampler_on_real_data(cedar_path):
    if not cedar_path.exists():
        pytest.skip("CEDAR fiziksel verisi mevcut değil.")

    samples = ForensicDataPipeline.load_from_directory(cedar_path)
    train_samples, _, _ = ForensicDataPipeline.split_by_disjoint_writers(samples, 0.7, 0.15, 0.15, seed=42)

    ds = ForensicWriterDataset(train_samples, transform=ForensicAugmentation(is_training=False))
    sampler = BalancedBatchSampler(ds, p_writers=4, k_samples=4, seed=42)

    batch_indices = next(iter(sampler))
    assert len(batch_indices) == 16, "Batch boyutu 4x4 = 16 olmalıdır."

    batch_writers = [ds.samples[i].writer_id for i in batch_indices]
    writer_counts = {w: batch_writers.count(w) for w in set(batch_writers)}
    assert len(writer_counts) == 4, "Batch içerisinde tam 4 farklı yazar olmalıdır."
    for w, count in writer_counts.items():
        assert count == 4, f"Yazar {w} için 4 numune olmalı, bulunan: {count}"


# ------------------------------------------------------------------------------
# 8. Training Configuration Sözleşmesi Testi (Sprint 18.7)
# ------------------------------------------------------------------------------
def test_18_7_training_config_dataclass_and_serialization():
    cfg = TrainingConfig(
        dataset_name="CEDAR",
        batch_size=16,
        p_writers=4,
        k_samples=4,
        num_epochs=10,
        lr=1e-4,
    )
    d = cfg.to_dict()
    assert d["dataset_name"] == "CEDAR"
    assert d["batch_size"] == 16
    assert d["num_epochs"] == 10
    assert d["training_type"] == "real_dataset"


# ------------------------------------------------------------------------------
# 9. Reproducibility & Deterministik Tohum Testi (Sprint 18.8)
# ------------------------------------------------------------------------------
def test_18_8_reproducibility_deterministic_seed():
    set_deterministic_seed(42)
    t1 = torch.randn(5, 5)
    set_deterministic_seed(42)
    t2 = torch.randn(5, 5)
    assert torch.allclose(t1, t2), "Tohum sıfırlandıktan sonra aynı tensör üretilmeli!"


# ------------------------------------------------------------------------------
# 10. ArcFace Gradient ve Optimizer Testi (Sprint 18.9)
# ------------------------------------------------------------------------------
def test_18_9_arcface_optimizer_and_gradient():
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False, in_channels=3)
    loss_fn = ArcFaceMarginLoss(num_classes=8, embedding_dim=128)
    opt = torch.optim.AdamW(list(encoder.parameters()) + list(loss_fn.parameters()), lr=1e-3)

    x = torch.randn(4, 3, 224, 224)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    opt.zero_grad()
    emb = encoder(x)
    loss = loss_fn(emb, y)
    loss.backward()

    assert loss_fn.weight.grad is not None, "ArcFace weight gradient üretmeli!"
    assert torch.norm(loss_fn.weight.grad).item() > 0, "Gradient normu 0'dan büyük olmalı!"


# ------------------------------------------------------------------------------
# 11. SupCon Loss Testi (Sprint 18.9)
# ------------------------------------------------------------------------------
def test_18_9_supcon_loss_and_positive_pairs():
    supcon = SupervisedContrastiveLoss(temperature=0.07)
    # 4 numune, 2 sınıf (0 ve 1) -> 2 pozitif çift garantisi
    embeddings = torch.nn.functional.normalize(torch.randn(4, 128), dim=-1)
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    loss = supcon(embeddings, labels)
    assert not torch.isnan(loss), "SupCon loss NaN olamaz!"
    assert loss.item() > 0.0, "SupCon loss pozitif olmalıdır."


# ------------------------------------------------------------------------------
# 12. Embedding Finite-Value Testi (Sprint 18.9)
# ------------------------------------------------------------------------------
def test_18_9_embedding_finite_values_no_nan_inf():
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False, in_channels=3)
    encoder.eval()

    test_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        emb = encoder(test_input)

    assert not torch.isnan(emb).any(), "Embedding tensörü NaN içeremez!"
    assert not torch.isinf(emb).any(), "Embedding tensörü Inf içeremez!"
    # L2 normu 1.0 olmalı
    norms = torch.norm(emb, p=2, dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4), "L2 normalizasyon 1.0 olmalı!"


# ------------------------------------------------------------------------------
# 13. Sentetik Checkpoint Ayrımı & Güvenlik Uyarısı (Sprint 18.10)
# ------------------------------------------------------------------------------
def test_18_10_synthetic_checkpoint_separation_and_warning():
    best_pt = Path(__file__).resolve().parent.parent / "artifacts" / "weights" / "convnext_tiny_forensic_best.pt"
    if best_pt.exists():
        ckpt = torch.load(best_pt, map_location="cpu")
        assert ckpt.get("training_type") == "synthetic_test", "Mevcut baseline checkpoint 'synthetic_test' olarak mühürlenmeli!"

    assert model_registry.is_synthetic_checkpoint() is True
    assert model_registry.get_model_training_type() == "synthetic_test"


# ------------------------------------------------------------------------------
# 14. 13 Aşamalı Preflight Otomasyon Testi (Sprint 18.12)
# ------------------------------------------------------------------------------
def test_18_12_preflight_automated_checks():
    checker = PreflightChecker()
    success = checker.run_all_checks()
    assert success is True, "Preflight 13 aşamalı denetimi başarıyla geçmelidir!"
    assert len(checker.results) == 13
    assert all(r["passed"] for r in checker.results)
