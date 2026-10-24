"""
Unit tests for Sprint 22 V4 Preprocessing Contract (tests/test_v4_preprocessing_contract.py)
"""

import numpy as np
import pytest
import torch

from ml.preprocessing.contract import (
    V4ForensicAugmentation,
    inference_preprocess,
    pad_to_square,
    preprocess_for_inference,
    to_rgb,
    training_preprocess,
)


@pytest.fixture
def sample_bgr_signature() -> np.ndarray:
    """Simüle edilmiş 350x180 boyutunda imza matrisi."""
    img = np.full((180, 350, 3), 255, dtype=np.uint8)
    # İmza benzeri siyah vuruşlar
    img[80:100, 50:300] = [20, 20, 20]
    img[40:140, 150:170] = [10, 10, 10]
    return img


@pytest.fixture
def sample_grayscale_signature() -> np.ndarray:
    """Tek kanallı gri seviye imza."""
    img = np.full((200, 400), 255, dtype=np.uint8)
    img[90:110, 60:350] = 30
    return img


def test_v4_preprocessing_contract_shape_and_dtype(sample_bgr_signature):
    """Eğitim ve çıkarım ön işlemesinin tam olarak [3, 224, 224] float32 ürettiğini doğrular."""
    train_t = training_preprocess(sample_bgr_signature, is_training=True)
    val_t = training_preprocess(sample_bgr_signature, is_training=False)
    infer_t = inference_preprocess(sample_bgr_signature)

    assert train_t.shape == (3, 224, 224)
    assert train_t.dtype == torch.float32
    assert val_t.shape == (3, 224, 224)
    assert val_t.dtype == torch.float32
    assert infer_t.shape == (1, 3, 224, 224)
    assert infer_t.dtype == torch.float32


def test_v4_eval_train_exact_parity(sample_bgr_signature):
    """is_training=False durumunda training_preprocess ile inference_preprocess'in özdeşliğini doğrular."""
    val_t = training_preprocess(sample_bgr_signature, is_training=False)
    infer_t = inference_preprocess(sample_bgr_signature)[0]

    max_diff = float(torch.max(torch.abs(val_t - infer_t)))
    assert max_diff < 1e-6, f"Parite hatası: max_diff={max_diff}"


def test_v4_grayscale_and_pad_to_square(sample_grayscale_signature):
    """Gri seviye görsellerin 3 kanala dönüştürüldüğünü ve kareye tamamlandığını doğrular."""
    infer_t = inference_preprocess(sample_grayscale_signature)
    assert infer_t.shape == (1, 3, 224, 224)
    assert not torch.isnan(infer_t).any()
    assert not torch.isinf(infer_t).any()


def test_v4_augmentation_limits(sample_bgr_signature):
    """V4 veri artırmasının biyomekaniği bozmadan dar aralıkta kaldığını doğrular."""
    aug = V4ForensicAugmentation(is_training=True)
    tensors = [aug(sample_bgr_signature) for _ in range(5)]
    for t in tensors:
        assert t.shape == (3, 224, 224)
        assert not torch.isnan(t).any()
        assert not torch.isinf(t).any()
