"""
FHDSS - SPRINT 17 Doğrulama ve Regresyon Test Paketi.
Model Eğitimi, Validation, Preprocessing ve Inference Zincirinin Kritik Uyumlandırılması.
"""

import json
import math
import os
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ml.datasets.forensic_dataset import (
    ForensicAugmentation,
    ForensicDataPipeline,
    ForensicSample,
    ForensicWriterDataset,
)
from ml.models.fusion import HybridForensicFusion
from ml.models.siamese import (
    ArcFaceMarginLoss,
    ForensicConvNeXtEncoder,
    SupervisedContrastiveLoss,
    compute_cosine_similarity,
    compute_normalized_similarity,
)
from ml.preprocessing.contract import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    ForensicPreprocessingContract,
    pad_to_square,
    preprocess_for_inference,
    to_rgb,
)
from ml.training.train import (
    BalancedBatchSampler,
    ForensicModelTrainer,
)


# ==============================================================================
# SPRINT 17.1 Testleri: ArcFace Optimizer Entegrasyonu
# ==============================================================================

def test_17_1_arcface_optimizer_weight_updates():
    """ArcFace criterion parametrelerinin optimizer'a eklenip step ile güncellendiğini doğrular."""
    torch.manual_seed(42)
    device = torch.device("cpu")
    encoder = ForensicConvNeXtEncoder(embedding_dim=64, pretrained=False).to(device)
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-2)

    criterion = ArcFaceMarginLoss(num_classes=5, embedding_dim=64).to(device)
    assert criterion.weight.requires_grad is True

    # ArcFace ağırlıklarını optimizer'a param-group olarak ekle
    optimizer.add_param_group({"params": list(criterion.parameters()), "lr": 1e-2})
    assert len(optimizer.param_groups) == 2

    initial_weight = criterion.weight.clone().detach()

    # Dummy forward & backward
    dummy_input = torch.randn(4, 3, 224, 224)
    dummy_labels = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    optimizer.zero_grad()
    embeddings = encoder(dummy_input)
    loss = criterion(embeddings, dummy_labels)
    loss.backward()

    # Gradient oluştuğunu doğrula
    assert criterion.weight.grad is not None
    assert torch.sum(torch.abs(criterion.weight.grad)).item() > 0.0

    optimizer.step()

    # Ağırlığın güncellendiğini doğrula
    updated_weight = criterion.weight.detach()
    diff = torch.sum(torch.abs(updated_weight - initial_weight)).item()
    assert diff > 1e-5, "ArcFace sınıf ağırlıkları optimizer.step() sonrasında güncellenmedi!"


# ==============================================================================
# SPRINT 17.2 Testleri: Açık Küme (Open-Set) Validasyon Sistemi
# ==============================================================================

def test_17_2_open_set_validation_disjoint_writers():
    """Train ve validation yazar kümeleri disjoint olduğunda ve val_classes > train_classes iken çökmediğini doğrular."""
    corpus = ForensicDataPipeline.generate_synthetic_forensic_corpus(
        num_writers=8, samples_per_writer=4, forgeries_per_writer=2, seed=42
    )

    # 3 Train yazarı, 5 Validation yazarı (val_classes > train_classes durumu)
    all_writers = sorted(list(set(s.writer_id for s in corpus)))
    train_writers = set(all_writers[:3])
    val_writers = set(all_writers[3:])

    train_samples = [s for s in corpus if s.writer_id in train_writers]
    val_samples = [s for s in corpus if s.writer_id in val_writers]

    assert len(val_writers) > len(train_writers)
    assert train_writers.isdisjoint(val_writers)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ForensicModelTrainer(
            embedding_dim=64,
            loss_type="arcface",
            weights_dir=tmp_dir,
            seed=42,
        )
        # Open-set validasyon fonksiyonunu doğrudan test et
        val_metrics = trainer.evaluate_open_set_validation(val_samples)

        assert "val_eer" in val_metrics
        assert "val_auc" in val_metrics
        assert "optimal_threshold" in val_metrics
        assert 0.0 <= val_metrics["val_eer"] <= 1.0
        assert 0.0 <= val_metrics["val_auc"] <= 1.0
        assert 0.0 <= val_metrics["optimal_threshold"] <= 1.0


def test_17_2_open_set_validation_empty_samples():
    """Boş validation kümesinde fonksiyonun kontrollü şekilde 0.5 döndürdüğünü doğrular."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ForensicModelTrainer(embedding_dim=64, weights_dir=tmp_dir)
        metrics = trainer.evaluate_open_set_validation([])
        assert metrics["val_loss"] == 0.5
        assert metrics["val_eer"] == 0.5
        assert metrics["val_auc"] == 0.5


# ==============================================================================
# SPRINT 17.3 Testleri: BalancedBatchSampler
# ==============================================================================

def test_17_3_balanced_batch_sampler():
    """Her batch'te en az 2 aynı yazar örneği olduğunu ve dengeli örnekleme yapıldığını doğrular."""
    corpus = ForensicDataPipeline.generate_synthetic_forensic_corpus(
        num_writers=8, samples_per_writer=6, forgeries_per_writer=1, seed=42
    )
    dataset = ForensicWriterDataset(corpus, transform=ForensicAugmentation(is_training=False))

    # P=3 writers, K=4 samples -> batch_size=12
    sampler = BalancedBatchSampler(dataset, p_writers=3, k_samples=4, seed=42)
    loader = DataLoader(dataset, batch_sampler=sampler)

    for batch_images, batch_labels, _ in loader:
        assert batch_images.size(0) == 12
        labels_list = batch_labels.tolist()
        unique_labels = set(labels_list)
        assert len(unique_labels) == 3

        # Her seçilen yazardan tam 4 adet örnek olmalı
        for lbl in unique_labels:
            assert labels_list.count(lbl) == 4

        # SupCon kaybının sıfır dönmediğini doğrula
        criterion = SupervisedContrastiveLoss(temperature=0.07)
        dummy_embs = torch.nn.functional.normalize(torch.randn(12, 64), p=2, dim=-1)
        loss = criterion(dummy_embs, batch_labels)
        assert float(loss.item()) > 0.0


def test_17_3_balanced_batch_sampler_small_dataset():
    """Yetersiz numunesi olan küçük veri setlerinde sampler'ın çökmeden yerine koyma yaptığını doğrular."""
    sample1 = ForensicSample(sample_id="s1", writer_id="W1", image_data=np.full((100, 100), 255, dtype=np.uint8))
    sample2 = ForensicSample(sample_id="s2", writer_id="W2", image_data=np.full((100, 100), 255, dtype=np.uint8))
    dataset = ForensicWriterDataset([sample1, sample2], transform=ForensicAugmentation(is_training=False))

    sampler = BalancedBatchSampler(dataset, p_writers=2, k_samples=2, seed=42)
    assert len(sampler) >= 1
    for batch_idxs in sampler:
        assert len(batch_idxs) == 4
        # W1 ve W2'nin indexleri (0 ve 1) en az 2'şer kez tekrarlanmış olmalı
        assert batch_idxs.count(0) == 2
        assert batch_idxs.count(1) == 2


# ==============================================================================
# SPRINT 17.4 Testleri: Preprocessing Sözleşmesi & Train/Serving Skew
# ==============================================================================

def test_17_4_preprocessing_contract_consistency():
    """Eğitim (ForensicAugmentation) ve Çıkarım (preprocess_for_inference) tensörlerinin birebir uyumunu doğrular."""
    raw_img = np.zeros((300, 500, 3), dtype=np.uint8)
    cv2.circle(raw_img, (250, 150), 40, (255, 255, 255), -1)

    # 1. Çıkarım Tensörü
    inf_tensor = preprocess_for_inference(raw_img)  # (1, 3, 224, 224)
    assert inf_tensor.shape == (1, 3, 224, 224)
    assert inf_tensor.dtype == torch.float32

    # 2. Eğitim / Validasyon Tensörü (Augmentation off)
    train_aug = ForensicAugmentation(is_training=False, target_size=(224, 224))
    train_tensor = train_aug(raw_img).unsqueeze(0)  # (1, 3, 224, 224)

    # İkisi de RGB olmalı ve ImageNet normalizasyonu içermeli (değerler < 0 içermeli)
    assert inf_tensor.min() < 0.0
    assert train_tensor.min() < 0.0

    # Deterministik fark kontrolü (sayısal fark sıfıra yakın olmalı)
    max_diff = torch.max(torch.abs(inf_tensor - train_tensor)).item()
    assert max_diff < 1e-4, f"Eğitim ve Çıkarım ön-işleme çıktıları arasında fark var: {max_diff}"


def test_17_4_to_rgb_and_aspect_ratio():
    """to_rgb ve pad_to_square fonksiyonlarının doğruluğunu test eder."""
    gray = np.full((100, 200), 128, dtype=np.uint8)
    rgb = to_rgb(gray)
    assert rgb.shape == (100, 200, 3)

    padded = pad_to_square(rgb)
    assert padded.shape == (200, 200, 3)
    # Üst ve alt kenarların beyaz (255) dolgulandığını doğrula
    assert np.all(padded[0, 0] == [255, 255, 255])


# ==============================================================================
# SPRINT 17.5 Testleri: Benzerlik Metriği Standardizasyonu
# ==============================================================================

def test_17_5_similarity_metric_standardization():
    """Kosinüs [-1, 1] ile normalize adli benzerlik [0, 1] eşlemesini doğrular."""
    v1 = torch.tensor([1.0, 0.0, 0.0])
    v_identical = torch.tensor([1.0, 0.0, 0.0])
    v_orthogonal = torch.tensor([0.0, 1.0, 0.0])
    v_opposite = torch.tensor([-1.0, 0.0, 0.0])

    # 1. Birebir aynı (cos = 1 -> norm = 1.0)
    assert pytest.approx(compute_cosine_similarity(v1, v_identical), 1e-4) == 1.0
    assert pytest.approx(compute_normalized_similarity(v1, v_identical), 1e-4) == 1.0

    # 2. Dik / Alakasız (cos = 0 -> norm = 0.5)
    assert pytest.approx(compute_cosine_similarity(v1, v_orthogonal), 1e-4) == 0.0
    assert pytest.approx(compute_normalized_similarity(v1, v_orthogonal), 1e-4) == 0.5

    # 3. Zıt (cos = -1 -> norm = 0.0)
    assert pytest.approx(compute_cosine_similarity(v1, v_opposite), 1e-4) == -1.0
    assert pytest.approx(compute_normalized_similarity(v1, v_opposite), 1e-4) == 0.0


def test_17_5_fusion_nan_inf_protection():
    """Füzyon katmanının NaN ve Inf değerlerine karşı korumalı çalıştığını doğrular."""
    fusion = HybridForensicFusion(weight_morphological=0.5, weight_deep=0.5)
    dummy_rep = {"similarity_score": 0.85, "significant_differences": []}

    # NaN girişi
    res_nan = fusion.fuse(dummy_rep, deep_similarity_score=float("nan"))
    assert not math.isnan(res_nan.hybrid_similarity_score)
    assert 0.0 <= res_nan.hybrid_similarity_score <= 1.0

    # Inf girişi
    res_inf = fusion.fuse(dummy_rep, deep_similarity_score=float("inf"))
    assert not math.isinf(res_inf.hybrid_similarity_score)
    assert 0.0 <= res_inf.hybrid_similarity_score <= 1.0


# ==============================================================================
# SPRINT 17.6 Testleri: Model Export → ONNX → Manifest Otomasyonu
# ==============================================================================

def test_17_6_onnx_export_and_manifest_automation():
    """Eğitilen modelin ONNX'e ihraç edildiğini, manifest ve SHA-256'nın güncellendiğini doğrular."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_dir = Path(tmp_dir)
        trainer = ForensicModelTrainer(
            embedding_dim=64,
            weights_dir=weights_dir,
            seed=42,
        )

        # Sahte bir checkpoint kaydet
        pt_path = weights_dir / "convnext_tiny_forensic_best.pt"
        torch.save({"model_state_dict": trainer.model.state_dict()}, str(pt_path))

        val_metrics = {
            "val_eer": 0.042,
            "val_auc": 0.981,
            "tar_at_eer": 0.958,
            "far_at_eer": 0.042,
            "frr_at_eer": 0.042,
        }

        export_res = trainer.export_and_validate_onnx(
            checkpoint_path=pt_path,
            val_metrics=val_metrics,
            numerical_tolerance=1e-3,
        )

        assert export_res["parity_passed"] is True
        assert os.path.exists(export_res["onnx_path"])
        assert len(export_res["pt_sha256"]) == 64
        assert len(export_res["onnx_sha256"]) == 64

        # Manifest içeriğini doğrula
        manifest_path = Path(export_res["manifest_path"])
        assert manifest_path.exists()
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        assert manifest["weights"]["pytorch_checkpoint"]["sha256"] == export_res["pt_sha256"]
        assert manifest["weights"]["onnx_model"]["sha256"] == export_res["onnx_sha256"]
        assert manifest["metrics"]["eer"] == 0.042
        assert manifest["metrics"]["auc"] == 0.981
        assert manifest["export_status"] == "SUCCESS"


# ==============================================================================
# SPRINT 17.7 Testleri: CEDAR Recursive Discovery
# ==============================================================================

def test_17_7_cedar_recursive_discovery():
    """Derin klasör yapısındaki CEDAR full_org ve full_forg dizinlerinin özyinelemeli bulunduğunu doğrular."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        base = Path(tmp_dir)
        # CEDAR/CEDAR1/signatures/full_org hiyerarşisi oluştur
        org_dir = base / "CEDAR" / "CEDAR1" / "signatures" / "full_org"
        forg_dir = base / "CEDAR" / "CEDAR1" / "signatures" / "full_forg"
        org_dir.mkdir(parents=True, exist_ok=True)
        forg_dir.mkdir(parents=True, exist_ok=True)

        # Sahte görsel dosyaları yaz
        dummy_img = np.full((50, 50), 255, dtype=np.uint8)
        cv2.imwrite(str(org_dir / "original_1_1.png"), dummy_img)
        cv2.imwrite(str(org_dir / "original_1_2.png"), dummy_img)
        cv2.imwrite(str(forg_dir / "forgeries_1_1.png"), dummy_img)

        # Kök dizinden yükle
        samples = ForensicDataPipeline.load_from_directory(base / "CEDAR")
        assert len(samples) == 3

        genuines = [s for s in samples if not s.is_forgery]
        forgeries = [s for s in samples if s.is_forgery]
        assert len(genuines) == 2
        assert len(forgeries) == 1
        assert genuines[0].writer_id == "YAZAR_001"


# ==============================================================================
# SPRINT 17.8 Testleri: Dataset Ingestion Pipeline Kontrolü
# ==============================================================================

def test_17_8_ingest_all_datasets_support():
    """scripts/ingest_all_datasets.py ETL yöneticisinde CEDAR desteğini ve klasör taramasını doğrular."""
    from scripts.ingest_all_datasets import ForensicDatasetIngestionPipeline
    with tempfile.TemporaryDirectory() as tmp_dir:
        pipeline = ForensicDatasetIngestionPipeline(base_dir=Path(tmp_dir))
        assert "CEDAR" in pipeline.SUPPORTED_DATASETS
        assert "BHSig-260" in pipeline.SUPPORTED_DATASETS


# ==============================================================================
# SPRINT 17.9 & 17.10 Testleri: Uçtan Uca ML Pipeline & Train-Serving Contract
# ==============================================================================

def test_17_9_end_to_end_ml_pipeline_run():
    """Discovery -> Split -> Balanced Sampler -> Train -> Validation -> ONNX Export -> Manifest zincirini test eder."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        weights_dir = Path(tmp_dir) / "weights"
        trainer = ForensicModelTrainer(
            embedding_dim=64,
            loss_type="supcon",
            weights_dir=weights_dir,
            seed=42,
        )

        corpus = ForensicDataPipeline.generate_synthetic_forensic_corpus(
            num_writers=6, samples_per_writer=4, forgeries_per_writer=1, seed=42
        )
        train_s, val_s, _ = ForensicDataPipeline.split_by_disjoint_writers(
            corpus, train_ratio=0.67, val_ratio=0.33, test_ratio=0.0, seed=42
        )

        results = trainer.fit(
            train_samples=train_s,
            val_samples=val_s,
            num_epochs=1,
            batch_size=8,
            use_balanced_sampler=True,
        )

        assert os.path.exists(results["best_checkpoint_path"])
        assert "val_eer" in results["history"]
        assert "val_auc" in results["history"]
        assert results["export_result"]["parity_passed"] is True
        assert os.path.exists(results["export_result"]["onnx_path"])
