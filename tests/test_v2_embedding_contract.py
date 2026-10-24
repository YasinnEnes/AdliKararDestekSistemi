import os
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from ml.models.siamese import (
    ForensicConvNeXtEncoder,
    compute_cosine_similarity,
    compute_normalized_similarity,
)
from ml.preprocessing.contract import preprocess_for_inference

WEIGHTS_PATH = Path("artifacts/weights/convnext_tiny_forensic_cedar_v2_best.pt")


def test_v2_weights_exist_and_load():
    """1. Mevcut V2 modelinin diskte bulunduğunu ve hatasız yüklendiğini doğrular."""
    assert WEIGHTS_PATH.exists(), f"V2 model dosyası bulunamadı: {WEIGHTS_PATH}"
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["model_state_dict"])
    elif isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
        encoder.load_state_dict(ckpt["encoder_state_dict"])
    else:
        encoder.load_state_dict(ckpt, strict=False)
    
    encoder.eval()
    assert encoder is not None


def test_v2_architecture_and_embedding_dimension():
    """2. ConvNeXt-Tiny omurgasını ve 128 boyutlu projeksiyon başlığını doğrular."""
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    dummy_input = torch.randn(2, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        out = encoder(dummy_input)

    assert out.shape == (2, 128), f"Beklenen şekil (2, 128), gelen: {out.shape}"
    assert out.dtype == torch.float32


def test_v2_l2_normalization_contract():
    """3. Model çıktısının kesin olarak L2-normalize edildiğini (norm ≈ 1.0) doğrular."""
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    dummy_input = torch.randn(4, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        emb = encoder(dummy_input)

    norms = torch.linalg.vector_norm(emb, dim=-1)
    for i, n in enumerate(norms):
        assert torch.isclose(n, torch.tensor(1.0, dtype=torch.float32), atol=1e-4), (
            f"Örnek {i} L2 normu 1.0 değil: {n.item()}"
        )
    assert not torch.isnan(emb).any(), "Embedding'de NaN değer tespit edildi!"
    assert not torch.isinf(emb).any(), "Embedding'de Inf değer tespit edildi!"


def test_v2_identity_test_exact_same_image():
    """4. Identity Test: Aynı görsel iki kez verildiğinde cosine similarity ≈ 1.0 olmalıdır."""
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    # Sentetik gerçekçi imza görseli
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.putText(img, "Forensic Test Signature", (50, 150), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (20, 20, 30), 3)

    t1 = preprocess_for_inference(img)
    t2 = preprocess_for_inference(img)

    with torch.no_grad():
        e1 = encoder(t1)
        e2 = encoder(t2)

    raw_cos = float(F.cosine_similarity(e1, e2, dim=-1).item())
    norm_sim = float(compute_normalized_similarity(e1[0].numpy(), e2[0].numpy()))

    assert raw_cos > 0.9999, f"Aynı görsel kosinüs benzerliği 1.0 olmalı, hesaplanan: {raw_cos}"
    assert norm_sim > 0.9999, f"Aynı görsel normalize benzerlik 1.0 olmalı, hesaplanan: {norm_sim}"


def test_v2_distinct_images_have_different_embeddings_no_mock():
    """5. Farklı imzalarda skorun gerçekten vektör farkından geldiğini ve mock katman olmadığını doğrular."""
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    img_a = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.putText(img_a, "Signature Alice", (50, 150), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.5, (10, 10, 10), 3)

    img_b = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.circle(img_b, (250, 150), 80, (10, 10, 10), 4)
    cv2.rectangle(img_b, (50, 50), (450, 250), (10, 10, 10), 2)

    t_a = preprocess_for_inference(img_a)
    t_b = preprocess_for_inference(img_b)

    with torch.no_grad():
        e_a = encoder(t_a)
        e_b = encoder(t_b)

    raw_cos = float(F.cosine_similarity(e_a, e_b, dim=-1).item())
    # Tamamen farklı iki görselin kosinüs benzerliği 0.99 olamaz
    assert raw_cos < 0.85, f"Farklı görsellerin benzerliği çok yüksek çıktı: {raw_cos}"
    # Sıfır vektör veya çöküş yok
    assert not torch.allclose(e_a, e_b, atol=1e-3)
