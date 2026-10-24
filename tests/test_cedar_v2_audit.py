import os
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ml.datasets.forensic_dataset import (
    ForensicAugmentation,
    ForensicDataPipeline,
    ForensicSample,
    ForensicStructuredBatchSampler,
    ForensicWriterDataset,
)
from ml.models.siamese import (
    ForensicConvNeXtEncoder,
    ForensicHybridLoss,
    compute_cosine_similarity,
    compute_normalized_similarity,
)
from ml.preprocessing.contract import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    pad_to_square,
    preprocess_for_inference,
    to_rgb,
)
from ml.training.dataset_split_validator import validate_writer_disjointness

CEDAR_DIR = Path("datasets/DS_Downloaded/CEDAR/CEDAR1/signatures")


# ==============================================================================
# 1. DATASET TESTS
# ==============================================================================
def test_1_cedar_real_dataset_integrity():
    assert CEDAR_DIR.exists(), f"CEDAR veri seti bulunamadi: {CEDAR_DIR}"
    samples = ForensicDataPipeline.load_from_directory(CEDAR_DIR)

    assert len(samples) == 2640, f"CEDAR toplam 2640 ornek olmali, bulunan: {len(samples)}"
    writers = set(s.writer_id for s in samples)
    assert len(writers) == 55, f"CEDAR 55 yazar icermeli, bulunan: {len(writers)}"

    genuines = [s for s in samples if not s.is_forgery]
    forgeries = [s for s in samples if s.is_forgery]
    assert len(genuines) == 1320, f"1320 genuine olmali, bulunan: {len(genuines)}"
    assert len(forgeries) == 1320, f"1320 forgery olmali, bulunan: {len(forgeries)}"

    # Her yazar icin tam 24 genuine, 24 forgery
    for w in sorted(list(writers)):
        w_gen = [s for s in genuines if s.writer_id == w]
        w_forg = [s for s in forgeries if s.writer_id == w]
        assert len(w_gen) == 24, f"Yazar {w} icin 24 genuine olmali, bulunan: {len(w_gen)}"
        assert len(w_forg) == 24, f"Yazar {w} icin 24 forgery olmali, bulunan: {len(w_forg)}"


# ==============================================================================
# 2. SAMPLER TESTS (En az 100 batch uret ve dogrula)
# ==============================================================================
def test_2_structured_forensic_sampler_100_batches():
    samples = ForensicDataPipeline.load_from_directory(CEDAR_DIR)
    train_s, _, _ = ForensicDataPipeline.split_by_disjoint_writers(samples, 0.70, 0.15, 0.15, seed=42)

    dataset = ForensicWriterDataset(train_s, return_forgery_flag=True)
    sampler = ForensicStructuredBatchSampler(
        dataset,
        p_writers=4,
        genuines_per_writer=2,
        forgeries_per_writer=2,
        num_batches=100,
        seed=42,
    )

    batches = list(sampler)
    assert len(batches) == 100, f"Tam 100 batch uretilmeli, uretilen: {len(batches)}"

    for b_idx, batch_indices in enumerate(batches):
        assert len(batch_indices) == 16, f"Batch {b_idx}: 16 ornek olmali, bulunan: {len(batch_indices)}"

        batch_samples = [dataset.samples[i] for i in batch_indices]
        writers = set(s.writer_id for s in batch_samples)
        assert len(writers) == 4, f"Batch {b_idx}: tam 4 yazar olmali, bulunan: {len(writers)}"

        for w in writers:
            w_samples = [s for s in batch_samples if s.writer_id == w]
            w_gen = [s for s in w_samples if not s.is_forgery]
            w_forg = [s for s in w_samples if s.is_forgery]

            assert len(w_gen) == 2, f"Batch {b_idx}, Yazar {w}: tam 2 genuine olmali, bulunan: {len(w_gen)}"
            assert len(w_forg) == 2, f"Batch {b_idx}, Yazar {w}: tam 2 forgery olmali, bulunan: {len(w_forg)}"
            # Orijinal ve sahte orneklerin ayni yazara ait oldugunu assert et
            assert w_gen[0].writer_id == w_forg[0].writer_id == w


# ==============================================================================
# 3. LABEL SEMANTICS TESTS
# ==============================================================================
def test_3_label_semantics_and_metadata_preservation():
    samples = ForensicDataPipeline.load_from_directory(CEDAR_DIR)
    dataset = ForensicWriterDataset(samples, return_forgery_flag=True)

    # ForensicSample.writer_id degistirilmemeli
    w001_samples = [s for s in samples if s.writer_id == "YAZAR_001"]
    assert any(not s.is_forgery for s in w001_samples), "Genuine numune bulunmali"
    assert any(s.is_forgery for s in w001_samples), "Forgery numune bulunmali"

    # Dataset getitem kontrolu
    gen_sample = next(s for s in samples if not s.is_forgery and s.writer_id == "YAZAR_001")
    forg_sample = next(s for s in samples if s.is_forgery and s.writer_id == "YAZAR_001")

    gen_idx = dataset.samples.index(gen_sample)
    forg_idx = dataset.samples.index(forg_sample)

    _, gen_label, gen_id, gen_is_forg = dataset[gen_idx]
    _, forg_label, forg_id, forg_is_forg = dataset[forg_idx]

    # Metadata korunur
    assert gen_sample.writer_id == forg_sample.writer_id == "YAZAR_001"
    assert gen_is_forg is False
    assert forg_is_forg is True
    # Her ikisi de ayni yazar etiketini tasir ancak is_forgery bayragi sayesinde loss icinde ayrilir
    assert gen_label == forg_label


# ==============================================================================
# 4. LOSS TESTS (Sentetik Test: G1, G2 genuine; F1 forgery)
# ==============================================================================
def test_4_hybrid_loss_pulls_genuine_and_pushes_forgery():
    """
    G1 = writer 001 genuine
    G2 = writer 001 genuine
    F1 = writer 001 forgery
    Beklenen:
    G1-G2 = positive (sim artmali)
    G1-F1 = hard negative (sim azalmali)
    G2-F1 = hard negative (sim azalmali)
    """
    torch.manual_seed(42)
    criterion = ForensicHybridLoss(temperature=0.07, margin=0.40, hard_neg_weight=1.5)

    # Baslangic embeddingleri (L2-normalize)
    emb = torch.tensor([
        [0.80, 0.60, 0.00],  # G1
        [0.60, 0.80, 0.00],  # G2
        [0.75, 0.65, 0.10],  # F1 (yetenekli sahte - baslangicta G1'e cok yakin!)
    ], dtype=torch.float32, requires_grad=True)

    emb_norm = F.normalize(emb, p=2, dim=-1)
    labels = torch.tensor([0, 0, 0], dtype=torch.long)
    is_forgery = torch.tensor([False, False, True], dtype=torch.bool)

    initial_sim_g1_g2 = float(torch.sum(emb_norm[0] * emb_norm[1]).item())
    initial_sim_g1_f1 = float(torch.sum(emb_norm[0] * emb_norm[2]).item())
    initial_sim_g2_f1 = float(torch.sum(emb_norm[1] * emb_norm[2]).item())

    # Optimizer ile birkac optimizasyon adimi calistir
    optimizer = torch.optim.SGD([emb], lr=0.1)

    for _ in range(15):
        optimizer.zero_grad()
        norm_emb = F.normalize(emb, p=2, dim=-1)
        loss, gen_loss, hn_loss = criterion(norm_emb, labels, is_forgery)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_emb = F.normalize(emb, p=2, dim=-1)
        final_sim_g1_g2 = float(torch.sum(final_emb[0] * final_emb[1]).item())
        final_sim_g1_f1 = float(torch.sum(final_emb[0] * final_emb[2]).item())
        final_sim_g2_f1 = float(torch.sum(final_emb[1] * final_emb[2]).item())

    # 1. G1-G2 pozitif cift: benzerlik artmali
    assert final_sim_g1_g2 > initial_sim_g1_g2, (
        f"G1-G2 benzerligi artmaliydi: {initial_sim_g1_g2:.4f} -> {final_sim_g1_g2:.4f}"
    )

    # 2. G1-F1 hard negative cift: benzerlik azalmali
    assert final_sim_g1_f1 < initial_sim_g1_f1, (
        f"G1-F1 benzerligi azalmaliydi: {initial_sim_g1_f1:.4f} -> {final_sim_g1_f1:.4f}"
    )

    # 3. G2-F1 hard negative cift: benzerlik azalmali
    assert final_sim_g2_f1 < initial_sim_g2_f1, (
        f"G2-F1 benzerligi azalmaliydi: {initial_sim_g2_f1:.4f} -> {final_sim_g2_f1:.4f}"
    )

    # Label collision geri donmedi: Pozitif benzerlik hard negative benzerliginden buyuk olmali
    assert final_sim_g1_g2 > final_sim_g1_f1
    assert final_sim_g1_g2 > final_sim_g2_f1


# ==============================================================================
# 5. PRETRAINED MODEL TESTS
# ==============================================================================
def test_5_convnext_tiny_pretrained_weights_enforced():
    # Pretrained=True kesin olarak yuklenebilmeli
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=True)
    assert encoder.is_pretrained is True

    # Agirliklarin uninitialized olmadigini dogrula (sifir veya trivial olmamali)
    first_conv_weight = encoder.backbone.features[0][0].weight
    assert first_conv_weight.shape[0] == 96
    assert first_conv_weight.std().item() > 0.01

    # Freeze / Unfreeze fonksiyonel dogrulugu
    encoder.freeze_backbone()
    for p in encoder.backbone.parameters():
        assert p.requires_grad is False
    for p in encoder.projection_head.parameters():
        assert p.requires_grad is True

    encoder.unfreeze_backbone()
    for p in encoder.backbone.parameters():
        assert p.requires_grad is True


# ==============================================================================
# 6. PREPROCESSING TESTS
# ==============================================================================
def test_6_preprocessing_contract_and_parity():
    # Sentetik BGR test gorseli (300 x 500)
    img_bgr = np.random.randint(0, 255, (300, 500, 3), dtype=np.uint8)

    aug_eval = ForensicAugmentation(is_training=False, target_size=(224, 224))
    t_eval = aug_eval(img_bgr)
    t_inf = preprocess_for_inference(img_bgr)

    # 1. Boyut: (3, 224, 224)
    assert t_eval.shape == (3, 224, 224)
    assert t_inf.shape == (1, 3, 224, 224)

    # 2. Inference ve Augmentation(is_training=False) arasinda sifir fark (Parity)
    diff = torch.max(torch.abs(t_eval - t_inf[0])).item()
    assert diff < 1e-6, f"Preprocessing tutarsizligi tespit edildi! Max fark: {diff}"

    # 3. Aspect Ratio Preservation (Kare tamamlama kontrolu)
    padded = pad_to_square(img_bgr, background_value=(255, 255, 255))
    assert padded.shape[0] == padded.shape[1] == 500
    # Eklenen dolgu beyaz olmali
    assert padded[0, 0, 0] == 255


# ==============================================================================
# 7. WRITER LEAKAGE TESTS
# ==============================================================================
def test_7_zero_writer_leakage_split():
    samples = ForensicDataPipeline.load_from_directory(CEDAR_DIR)
    train_s, val_s, test_s = ForensicDataPipeline.split_by_disjoint_writers(
        samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    train_writers = set(s.writer_id for s in train_s)
    val_writers = set(s.writer_id for s in val_s)
    test_writers = set(s.writer_id for s in test_s)

    # Yazar sayilari
    assert len(train_writers) == 38, f"Train yazar sayisi 38 olmali, bulunan: {len(train_writers)}"
    assert len(val_writers) == 8, f"Val yazar sayisi 8 olmali, bulunan: {len(val_writers)}"
    assert len(test_writers) == 9, f"Test yazar sayisi 9 olmali, bulunan: {len(test_writers)}"

    # Kesisimler bos kume olmali
    assert len(train_writers.intersection(val_writers)) == 0, "TRAIN ve VAL kesisiyor!"
    assert len(train_writers.intersection(test_writers)) == 0, "TRAIN ve TEST kesisiyor!"
    assert len(val_writers.intersection(test_writers)) == 0, "VAL ve TEST kesisiyor!"

    # validate_writer_disjointness fonksiyonu hata firlatmamali
    validate_writer_disjointness(train_s, val_s, test_s)
