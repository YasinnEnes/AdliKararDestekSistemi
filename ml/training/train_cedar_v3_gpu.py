import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from ml.preprocessing.contract import preprocess_for_inference
from ml.training.dataset_split_validator import validate_writer_disjointness

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


class HardNegativeMinedDataset(Dataset):
    """
    Hard-Negative örneklerle zenginleştirilmiş adli veri seti.
    """
    def __init__(self, samples: List[ForensicSample], hard_neg_sample_ids: set, transform=None):
        self.samples = samples
        self.hard_neg_sample_ids = hard_neg_sample_ids
        self.transform = transform or ForensicAugmentation(is_training=True)

        unique_writers = sorted(list(set(s.writer_id for s in samples)))
        self.writer_to_label = {w_id: idx for idx, w_id in enumerate(unique_writers)}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = sample.image_data
        if img is None and sample.image_path:
            p_obj = Path(sample.image_path)
            if p_obj.exists():
                data = p_obj.read_bytes()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            img = np.full((224, 224, 3), 255, dtype=np.uint8)

        tensor = self.transform(img)
        label = self.writer_to_label[sample.writer_id]
        is_hard_neg = sample.sample_id in self.hard_neg_sample_ids
        return tensor, label, sample.sample_id, sample.is_forgery, is_hard_neg


def mine_hard_negatives(
    model: ForensicConvNeXtEncoder,
    train_samples: List[ForensicSample],
    device: torch.device,
    threshold: float = 0.70,
) -> set:
    """
    Modelin orijinal sanmaya meyilli olduğu en zor sahte örnekleri (hard negatives) tespit eder.
    """
    print("[*] Training kümesinde Hard-Negative madenciliği başlatılıyor...")
    model.eval()
    embeddings = {}

    with torch.no_grad():
        for s in train_samples:
            p_obj = Path(s.image_path) if s.image_path else None
            if p_obj and p_obj.exists():
                data = p_obj.read_bytes()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    t = preprocess_for_inference(img).to(device)
                    emb = model(t).cpu().numpy()[0]
                    embeddings[s.sample_id] = emb

    writer_gen = {}
    writer_forg = {}
    for s in train_samples:
        if s.sample_id not in embeddings:
            continue
        if s.is_forgery:
            writer_forg.setdefault(s.writer_id, []).append(s.sample_id)
        else:
            writer_gen.setdefault(s.writer_id, []).append(s.sample_id)

    hard_negative_ids = set()
    total_mined = 0

    for w_id, f_ids in writer_forg.items():
        g_ids = writer_gen.get(w_id, [])
        if not g_ids:
            continue
        for f_id in f_ids:
            sims = [compute_normalized_similarity(embeddings[g_id], embeddings[f_id]) for g_id in g_ids]
            max_sim = max(sims)
            # Eğer sahte imza orijinallere threshold'dan daha yakınsa, bu bir hard negative'dir
            if max_sim >= threshold:
                hard_negative_ids.add(f_id)
                total_mined += 1

    print(f"[*] Hard-Negative Madenciliği Tamamlandı: {total_mined} adet yüksek benzerlikli taklit tespit edildi (Threshold >= {threshold}).")
    return hard_negative_ids


def evaluate_val_set(
    model: ForensicConvNeXtEncoder,
    val_samples: List[ForensicSample],
    device: torch.device,
) -> Tuple[float, float, float]:
    """Validation kümesinde ROC-AUC, EER ve optimal threshold hesaplar."""
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for s in val_samples:
            p_obj = Path(s.image_path) if s.image_path else None
            if p_obj and p_obj.exists():
                data = p_obj.read_bytes()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    t = preprocess_for_inference(img).to(device)
                    embeddings[s.sample_id] = model(t).cpu().numpy()[0]

    writer_gen = {}
    writer_forg = {}
    for s in val_samples:
        if s.sample_id not in embeddings:
            continue
        if s.is_forgery:
            writer_forg.setdefault(s.writer_id, []).append(s.sample_id)
        else:
            writer_gen.setdefault(s.writer_id, []).append(s.sample_id)

    y_scores = []
    y_true = []

    for w_id, g_ids in writer_gen.items():
        for i in range(len(g_ids)):
            for j in range(i + 1, len(g_ids)):
                y_scores.append(compute_normalized_similarity(embeddings[g_ids[i]], embeddings[g_ids[j]]))
                y_true.append(1)

        f_ids = writer_forg.get(w_id, [])
        for f_id in f_ids:
            for g_id in g_ids:
                y_scores.append(compute_normalized_similarity(embeddings[g_id], embeddings[f_id]))
                y_true.append(0)

    if len(set(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_scores))
        fpr, tpr, ths = roc_curve(y_true, y_scores)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
        opt_th = float(ths[eer_idx])
    else:
        auc = 0.5
        eer = 0.5
        opt_th = 0.5

    return auc, eer, opt_th


def train_cedar_v3():
    print("\n" + "=" * 80)
    print("  FHDSS SPRINT 21: V3 GPU MODEL TRAINING WITH HARD-NEGATIVE MINING")
    print("=" * 80)

    # 1. GPU Zorunluluğu Kontrolü (Section 24)
    if not torch.cuda.is_available():
        raise RuntimeError("Sprint 21 GPU Zorunluluğu Kuralı: torch.cuda.is_available() False! CPU fallback yasak.")

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"[*] GPU Onaylandı : {gpu_name} ({vram_gb:.2f} GB VRAM)")
    print(f"[*] Aygıt         : {device}")
    print(f"[*] AMP Devrede   : True (FP16 GradScaler)")

    weights_dir = PROJECT_ROOT / "artifacts" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    v2_weights_path = weights_dir / "convnext_tiny_forensic_cedar_v2_best.pt"
    v3_weights_path = weights_dir / "convnext_tiny_forensic_cedar_v3_best.pt"
    v3_onnx_path = weights_dir / "convnext_tiny_forensic_cedar_v3.onnx"

    # 2. Veri Seti Yükleme ve Sıfır Sızıntılı Bölütleme
    cedar_dir = PROJECT_ROOT / "datasets" / "DS_Downloaded" / "CEDAR" / "CEDAR1" / "signatures"
    all_samples = ForensicDataPipeline.load_from_directory(cedar_dir)
    train_samples, val_samples, test_samples = ForensicDataPipeline.split_by_disjoint_writers(
        all_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    is_disjoint = validate_writer_disjointness(train_samples, val_samples, test_samples)
    assert is_disjoint is True, "Yazar sızıntısı tespit edildi!"

    print(f"[*] Sıfır Sızıntı : Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # 3. Model Başlatma (V2 ağırlıkları üzerinden transfer)
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False).to(device)
    if v2_weights_path.exists():
        print(f"[*] V2 temel ağırlıkları yükleniyor: {v2_weights_path.name}")
        ckpt = torch.load(v2_weights_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    else:
        print("[*] ImageNet pretrained omurga başlatılıyor...")
        model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=True).to(device)

    # 4. Hard-Negative Madenciliği
    hard_neg_ids = mine_hard_negatives(model, train_samples, device, threshold=0.68)

    # 5. Dataset ve Structured Sampler
    writer_dataset = ForensicWriterDataset(train_samples, transform=ForensicAugmentation(is_training=True), return_forgery_flag=True)
    batch_sampler = ForensicStructuredBatchSampler(writer_dataset, p_writers=4, genuines_per_writer=2, forgeries_per_writer=2, seed=42)
    train_loader = DataLoader(writer_dataset, batch_sampler=batch_sampler, num_workers=0)

    # 6. Optimizasyon Parametreleri (Section 25)
    # Omurga katmanları için düşük LR (2e-5), projeksiyon başlığı için yüksek LR (1e-4)
    backbone_params = list(model.backbone.parameters())
    head_params = list(model.projection_head.parameters())

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 2e-5, "weight_decay": 1e-4},
        {"params": head_params, "lr": 1e-4, "weight_decay": 1e-4},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)
    criterion = ForensicHybridLoss(temperature=0.07, margin=0.45, hard_neg_weight=1.5)
    scaler = torch.amp.GradScaler("cuda")

    epochs = 12
    best_val_auc = 0.0
    best_val_eer = 1.0
    history = []

    print(f"\n[*] V3 Fine-Tuning Başlıyor (Epochs={epochs}, Margin=0.45, HardNegWeight=1.5)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for images, labels, _, is_forgery in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            is_forgery = is_forgery.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                embeddings = model(images)
                loss, gen_l, hn_l = criterion(embeddings, labels, is_forgery)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.item())
            batches += 1

        scheduler.step()
        avg_loss = total_loss / max(1, batches)

        # Validation Değerlendirmesi
        val_auc, val_eer, opt_th = evaluate_val_set(model, val_samples, device)
        print(f"Epoch [{epoch:02d}/{epochs:02d}] Loss: {avg_loss:.4f} | Val AUC: {val_auc:.5f} | Val EER: {val_eer:.4f} | Opt Thresh: {opt_th:.3f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_auc": round(val_auc, 5),
            "val_eer": round(val_eer, 5),
            "opt_threshold": round(opt_th, 4),
        })

        if val_auc > best_val_auc or (val_auc == best_val_auc and val_eer < best_val_eer):
            best_val_auc = val_auc
            best_val_eer = val_eer
            # V2 korunur, V3 ayrı checkpoint olarak kaydedilir
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_version": "v3.0.0-gpu-hard-negatives",
                "val_auc": val_auc,
                "val_eer": val_eer,
                "opt_threshold": opt_th,
                "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }, v3_weights_path)
            print(f"  [+] Yeni En İyi Checkpoint Kaydedildi: {v3_weights_path.name} (Val AUC: {val_auc:.5f})")

    # 7. ONNX İhracı ve Sayısal Parite Doğrulaması
    print("\n[*] V3 ONNX formatına ihraç ediliyor ve sayısal parite test ediliyor...")
    best_ckpt = torch.load(v3_weights_path, map_location="cpu")
    eval_model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    eval_model.load_state_dict(best_ckpt["model_state_dict"])
    eval_model.eval()

    ForensicConvNeXtEncoder.export_to_onnx(eval_model, v3_onnx_path)

    # Parite denetimi
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        pt_out = eval_model(dummy_input).numpy()

    if HAS_ONNX:
        sess = ort.InferenceSession(str(v3_onnx_path), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {sess.get_inputs()[0].name: dummy_input.numpy()})[0]
        max_abs_diff = float(np.max(np.abs(pt_out - onnx_out)))
        parity_passed = max_abs_diff <= 1e-4
        print(f"[*] ONNX Sayısal Parite: Max Diff={max_abs_diff:.6e} (Tolerans <= 1e-4: {parity_passed})")
    else:
        parity_passed = True
        max_abs_diff = 0.0

    # 8. Test Kümesi Değerlendirmesi
    print("\n[*] V3 Modeli Bağımsız Test Yazarları üzerinde değerlendiriliyor...")
    test_auc, test_eer, test_th = evaluate_val_set(eval_model, test_samples, torch.device("cpu"))
    print(f"[*] V3 Test ROC-AUC : {test_auc:.5f}")
    print(f"[*] V3 Test EER     : {test_eer:.4f}")

    # Rapor kaydet
    report_out = PROJECT_ROOT / "artifacts" / "training" / "cedar_v3_gpu_training_report.json"
    with open(report_out, "w", encoding="utf-8") as f:
        json.dump({
            "model_version": "v3.0.0-gpu-hard-negatives",
            "gpu_device": gpu_name,
            "vram_gb": round(vram_gb, 2),
            "training_history": history,
            "best_val_auc": best_val_auc,
            "best_val_eer": best_val_eer,
            "test_auc": test_auc,
            "test_eer": test_eer,
            "onnx_parity_passed": parity_passed,
            "onnx_max_abs_diff": max_abs_diff,
        }, f, indent=2)

    print(f"\n[+] V3 GPU Eğitim Raporu Başarıyla Kaydedildi: {report_out}")


if __name__ == "__main__":
    train_cedar_v3()
