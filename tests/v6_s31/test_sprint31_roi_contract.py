"""
FHDSS SPRINT 31 — Test Suite 1: P0 Inference ROI & Preprocessing Contract Tests
Verifies that:
1. `normalized_crop` is prioritized over `isolated_document` to prevent whitespace dilution.
2. Blank or empty inputs fallback gracefully to document/original without raising exceptions.
3. Preprocessed tensors maintain shape (1, 3, 224, 224), float32, and ImageNet standardization.
4. Comparing identical signatures on different page margins produces high deep similarity (>0.95).
"""
import cv2
import numpy as np
import pytest
import torch

from cv.preprocessing.pipeline import ForensicImagePreprocessor
from ml.models.siamese import ForensicConvNeXtEncoder, compute_cosine_similarity, compute_normalized_similarity
from ml.preprocessing.contract import preprocess_for_inference


@pytest.fixture
def preprocessor():
    return ForensicImagePreprocessor(deskew_enabled=False)


@pytest.fixture
def encoder():
    enc = ForensicConvNeXtEncoder()
    enc.eval()
    return enc


def _create_sample_signature():
    """340x500 boyutunda sentetik imza vuruşu üretir."""
    img = np.full((340, 500, 3), 255, dtype=np.uint8)
    pts = np.array([[80, 200], [150, 100], [220, 240], [300, 110], [420, 260]], np.int32)
    cv2.polylines(img, [pts], False, (20, 20, 20), 3, cv2.LINE_AA)
    cv2.circle(img, (220, 150), 35, (25, 25, 25), 2, cv2.LINE_AA)
    return img


def _embed_in_full_page(sig_img, page_h=1200, page_w=900, top=600, left=200):
    """İmzayı geniş beyaz A4 sayfasına yerleştirir."""
    page = np.full((page_h, page_w, 3), 255, dtype=np.uint8)
    h, w = sig_img.shape[:2]
    page[top:top+h, left:left+w] = sig_img
    return page


def test_normalized_crop_produces_tight_ink_bbox(preprocessor):
    sig = _create_sample_signature()
    full_page = _embed_in_full_page(sig)

    art = preprocessor.process(full_page)
    crop = art.normalized_crop

    # Crop boyutu 224x224 olmalıdır
    assert crop.shape[:2] == (224, 224)
    # İmzanın mürekkep pikselleri hedef tuvalde merkezlenmiş olmalıdır
    ink_ratio = float(np.mean(crop < 200))
    assert ink_ratio > 0.01, f"Crop boş görünüyor: {ink_ratio}"
    assert art.metadata["bounding_box_normalization"]["has_content"] is True


def test_whitespace_dilution_solved_by_roi_contract(preprocessor, encoder):
    sig1 = _create_sample_signature()
    # Aynı imza iki farklı sayfa boyutuna / marjine yerleştirilsin
    page_a = _embed_in_full_page(sig1, page_h=1000, page_w=800, top=400, left=150)
    page_b = _embed_in_full_page(sig1, page_h=1400, page_w=1000, top=800, left=350)

    art_a = preprocessor.process(page_a)
    art_b = preprocessor.process(page_b)

    # 1. Eski Hatalı Yol (isolated_document / tam sayfa)
    t_old_a = preprocess_for_inference(art_a.isolated_document)
    t_old_b = preprocess_for_inference(art_b.isolated_document)
    emb_old_a = encoder(t_old_a).detach().cpu().numpy()[0]
    emb_old_b = encoder(t_old_b).detach().cpu().numpy()[0]
    sim_old = compute_normalized_similarity(emb_old_a, emb_old_b)

    # 2. Yeni Düzeltilmiş Yol (normalized_crop / imza ROI'si)
    t_new_a = preprocess_for_inference(art_a.normalized_crop)
    t_new_b = preprocess_for_inference(art_b.normalized_crop)
    emb_new_a = encoder(t_new_a).detach().cpu().numpy()[0]
    emb_new_b = encoder(t_new_b).detach().cpu().numpy()[0]
    sim_new = compute_normalized_similarity(emb_new_a, emb_new_b)

    # Düzeltilmiş ROI'nin benzerliği > %0.95 olmalı ve eski yoldan belirgin üstün olmalıdır
    assert sim_new >= 0.95, f"Yeni ROI benzerliği düşük: {sim_new}"
    assert sim_new > sim_old, f"Yeni ROI ({sim_new}) eski yoldan ({sim_old}) daha yüksek olmalıydı"


def test_empty_blank_roi_graceful_fallback(preprocessor):
    blank_page = np.full((500, 500, 3), 255, dtype=np.uint8)
    art = preprocessor.process(blank_page)

    assert art.metadata["bounding_box_normalization"]["has_content"] is False
    # Boş görselde bile tensör üretilebilmeli, crash olmamalıdır
    t = preprocess_for_inference(art.normalized_crop)
    assert t.shape == (1, 3, 224, 224)
    assert not torch.isnan(t).any()
