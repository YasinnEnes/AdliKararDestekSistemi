"""
FHDSS — Sprint 24: V5 Multi-Domain Model & Pipeline Verification Suite
Tests:
1. Rule 1: V4 baseline checkpoints remain intact and unmodified.
2. V5 PyTorch checkpoint exists, is loadable, and has 128D L2-normalized output.
3. V5 ONNX model exists, is loadable, and matches PyTorch with MAE < 1e-4.
4. Model manifest contains valid V5 registration with matching SHA-256 hashes.
5. V5 multi-domain dataset loader strictly enforces zero writer leakage.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

from ml.datasets.v5_multidomain_loader import load_and_split_v5_corpora
from ml.models.siamese import ForensicConvNeXtEncoder
from ml.training.multidataset_split_validator import validate_multidataset_disjointness


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def test_rule1_v4_baseline_preserved():
    """Kural 1: V4 baseline checkpoint ve ONNX silinmemiş ve bozulmamıştır."""
    v4_pt = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v4_best.pt"
    v4_onnx = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v4.onnx"

    assert v4_pt.exists(), "V4 PyTorch checkpoint bulunamadı!"
    assert v4_onnx.exists(), "V4 ONNX modeli bulunamadı!"

    expected_pt_hash = "3f0640047c170874cd0369f9c19451dd9fa857a24f8410d7c3f5c2393b5e2e5e"
    assert compute_sha256(v4_pt) == expected_pt_hash, "V4 PyTorch checkpoint değiştirilmiş!"


def test_v5_pytorch_checkpoint_integrity_and_l2_norm():
    """V5 PyTorch checkpoint yüklenir, 128D çıkarım yapar ve L2-norm = 1.0 sağlar."""
    v5_pt = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5_best.pt"
    assert v5_pt.exists(), "V5 PyTorch checkpoint bulunamadı!"

    ckpt = torch.load(v5_pt, map_location="cpu")
    assert "model_state_dict" in ckpt
    assert ckpt["model_version"] == "v5.0.0-multidomain"
    assert ckpt["warm_start_source"] == "convnext_tiny_forensic_multidataset_v4_best.pt"

    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    dummy = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = model(dummy)

    assert out.shape == (4, 128), f"Beklenen (4, 128), alınan {out.shape}"
    norms = torch.norm(out, p=2, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), "L2 normalizasyonu başarısız!"


@pytest.mark.skipif(not HAS_ONNX, reason="onnxruntime kütüphanesi yüklü değil")
def test_v5_onnx_model_and_numerical_parity():
    """V5 ONNX modeli yüklenir ve PyTorch ile MAE < 1e-4 sayısal parite sağlar."""
    v5_pt = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5_best.pt"
    v5_onnx = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5.onnx"

    assert v5_onnx.exists(), "V5 ONNX modeli bulunamadı!"

    ckpt = torch.load(v5_pt, map_location="cpu")
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    sess = ort.InferenceSession(str(v5_onnx), providers=["CPUExecutionProvider"])
    test_inputs = np.random.randn(8, 3, 224, 224).astype(np.float32)

    with torch.no_grad():
        pt_out = model(torch.from_numpy(test_inputs)).numpy()

    onnx_out = sess.run(None, {"input": test_inputs})[0]
    max_abs_diff = float(np.max(np.abs(pt_out - onnx_out)))

    assert max_abs_diff < 1e-4, f"ONNX parite toleransı aşıldı: {max_abs_diff}"


def test_v5_model_manifest_registration():
    """Model manifestinde V5 modeli kayıtlıdır ve SHA-256 hashleri dosyalarla eşleşir."""
    manifest_p = PROJECT_ROOT / "artifacts" / "weights" / "model_manifest.json"
    assert manifest_p.exists(), "model_manifest.json bulunamadı!"

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    v5_entry = None
    for m in manifest.get("registered_models", []):
        if m.get("model_id") == "convnext_tiny_forensic_multidataset_v5":
            v5_entry = m
            break

    assert v5_entry is not None, "V5 modeli model_manifest.json içine kaydedilmemiş!"
    assert v5_entry["status"] == "PRODUCTION_READY"
    assert v5_entry["warm_start_source"] == "convnext_tiny_forensic_multidataset_v4_best.pt"

    v5_pt = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5_best.pt"
    v5_onnx = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5.onnx"

    assert v5_entry["sha256"] == compute_sha256(v5_pt)
    assert v5_entry["onnx_sha256"] == compute_sha256(v5_onnx)


def test_v5_corpora_zero_writer_leakage():
    """V5 çoklu alan veri setinde sıfır yazar sızıntısı kuralı tam olarak doğrulanır."""
    train_s, val_s, test_s, summary = load_and_split_v5_corpora(seed=42)

    assert summary["status"] == "ZERO_LEAKAGE_CONFIRMED"
    assert len(summary["corpora"]) == 5

    # Writer kümeleri ayrıklığı
    train_w = set(s.writer_id for s in train_s)
    val_w = set(s.writer_id for s in val_s)
    test_w = set(s.writer_id for s in test_s)

    assert len(train_w.intersection(val_w)) == 0, "Train ve Val arasında yazar sızıntısı!"
    assert len(train_w.intersection(test_w)) == 0, "Train ve Test arasında yazar sızıntısı!"
    assert len(val_w.intersection(test_w)) == 0, "Val ve Test arasında yazar sızıntısı!"
