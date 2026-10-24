import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.forensic_dataset import ForensicDataPipeline, ForensicSample
from ml.models.siamese import ForensicConvNeXtEncoder, compute_cosine_similarity, compute_normalized_similarity
from ml.preprocessing.contract import preprocess_for_inference
from ml.training.dataset_split_validator import validate_writer_disjointness


def compute_distribution_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p05": round(float(np.percentile(arr, 5)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
    }


def extract_embeddings(model: ForensicConvNeXtEncoder, samples: List[ForensicSample], device: torch.device) -> Dict[str, np.ndarray]:
    model.eval()
    embeddings = {}
    with torch.no_grad():
        for s in samples:
            img = s.image_data
            if img is None and s.image_path and os.path.exists(s.image_path):
                data = Path(s.image_path).read_bytes()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                t = preprocess_for_inference(img).to(device)
                emb = model(t).cpu().numpy()[0]
                embeddings[s.sample_id] = emb
    return embeddings


def evaluate_cedar_distributions():
    print("\n" + "=" * 80)
    print("  FHDSS SPRINT 21: CEDAR TEST DISTRIBUTION AUDIT & CALIBRATION")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Aygıt: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    weights_path = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_cedar_v2_best.pt"
    assert weights_path.exists(), f"V2 ağırlığı bulunamadı: {weights_path}"

    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # CEDAR veri setini yükle ve sıfır sızıntılı olarak böl
    cedar_dir = PROJECT_ROOT / "datasets" / "DS_Downloaded" / "CEDAR" / "CEDAR1" / "signatures"
    assert cedar_dir.exists(), f"CEDAR dizini bulunamadı: {cedar_dir}"

    all_samples = ForensicDataPipeline.load_from_directory(cedar_dir)
    train_samples, val_samples, test_samples = ForensicDataPipeline.split_by_disjoint_writers(
        all_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    is_disjoint = validate_writer_disjointness(train_samples, val_samples, test_samples)
    train_w = len(set(s.writer_id for s in train_samples))
    val_w = len(set(s.writer_id for s in val_samples))
    test_w = len(set(s.writer_id for s in test_samples))
    print(f"[*] Writer Disjointness ({is_disjoint}): Train Writers={train_w}, Val Writers={val_w}, Test Writers={test_w}")
    assert is_disjoint is True, "Yazar sızıntısı tespit edildi!"

    print(f"[*] Numuneler: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # 1. Validation Embedding Çıkarımı
    print("\n[*] Validation kümesinde embedding çıkarımı ve threshold kalibrasyonu başlatılıyor...")
    val_embeddings = extract_embeddings(model, val_samples, device)

    val_writer_gen: Dict[str, List[str]] = {}
    val_writer_forg: Dict[str, List[str]] = {}
    for s in val_samples:
        if s.sample_id not in val_embeddings:
            continue
        if s.is_forgery:
            val_writer_forg.setdefault(s.writer_id, []).append(s.sample_id)
        else:
            val_writer_gen.setdefault(s.writer_id, []).append(s.sample_id)

    val_scores: List[float] = []
    val_labels: List[int] = []

    # Val GG
    for w, g_ids in val_writer_gen.items():
        for i in range(len(g_ids)):
            for j in range(i + 1, len(g_ids)):
                sim = compute_normalized_similarity(val_embeddings[g_ids[i]], val_embeddings[g_ids[j]])
                val_scores.append(sim)
                val_labels.append(1)

    # Val GF
    for w, g_ids in val_writer_gen.items():
        f_ids = val_writer_forg.get(w, [])
        for f_id in f_ids:
            for g_id in g_ids[:max(1, len(g_ids) // 2)]:
                sim = compute_normalized_similarity(val_embeddings[g_id], val_embeddings[f_id])
                val_scores.append(sim)
                val_labels.append(0)

    # Val GDW
    rng = random.Random(42)
    val_writers = sorted(list(val_writer_gen.keys()))
    for w in val_writers:
        other_w = [ow for ow in val_writers if ow != w]
        for g_id in val_writer_gen[w]:
            if other_w:
                ow = rng.choice(other_w)
                og_ids = val_writer_gen.get(ow, [])
                if og_ids:
                    sim = compute_normalized_similarity(val_embeddings[g_id], val_embeddings[rng.choice(og_ids)])
                    val_scores.append(sim)
                    val_labels.append(0)

    val_scores_arr = np.array(val_scores, dtype=np.float64)
    val_labels_arr = np.array(val_labels, dtype=np.int32)

    # Validation Threshold Sweep (Optimal EER Noktası)
    thresholds = np.linspace(0.0, 1.0, 101)
    best_diff = float("inf")
    opt_val_threshold = 0.5
    opt_val_eer = 0.5
    for th in thresholds:
        preds = (val_scores_arr >= th).astype(int)
        tp = np.sum((preds == 1) & (val_labels_arr == 1))
        fp = np.sum((preds == 1) & (val_labels_arr == 0))
        fn = np.sum((preds == 0) & (val_labels_arr == 1))
        tn = np.sum((preds == 0) & (val_labels_arr == 0))
        far = fp / max(1, (fp + tn))
        frr = fn / max(1, (fn + tp))
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            opt_val_threshold = float(th)
            opt_val_eer = float((far + frr) / 2.0)

    val_auc = float(roc_auc_score(val_labels_arr, val_scores_arr))
    print(f"[*] Validation ROC-AUC : {val_auc:.5f}")
    print(f"[*] Validation EER     : {opt_val_eer:.5f} (Optimal Validation Threshold: {opt_val_threshold:.3f})")

    # Lojistik Regresyon Kalibrasyonu (Validation setine fit edilir)
    calibrator = LogisticRegression(C=1.0)
    calibrator.fit(val_scores_arr.reshape(-1, 1), val_labels_arr)
    print(f"[*] Logistic Calibrator Fit: Intercept={calibrator.intercept_[0]:.4f}, Coef={calibrator.coef_[0][0]:.4f}")

    # 2. Test Kümesi Değerlendirmesi (Bağımsız 9 Test Yazarı, 432 Görsel)
    print("\n[*] Bağımsız Test Yazarları üzerinde GG, GF ve GDW çiftleri üretiliyor...")
    test_embeddings = extract_embeddings(model, test_samples, device)

    test_writer_gen: Dict[str, List[str]] = {}
    test_writer_forg: Dict[str, List[str]] = {}
    for s in test_samples:
        if s.sample_id not in test_embeddings:
            continue
        if s.is_forgery:
            test_writer_forg.setdefault(s.writer_id, []).append(s.sample_id)
        else:
            test_writer_gen.setdefault(s.writer_id, []).append(s.sample_id)

    scores_gg: List[float] = []
    scores_gf: List[float] = []
    scores_gdw: List[float] = []

    test_scores: List[float] = []
    test_labels: List[int] = []

    # Genuine - Genuine (GG)
    for w, g_ids in test_writer_gen.items():
        for i in range(len(g_ids)):
            for j in range(i + 1, len(g_ids)):
                sim = compute_normalized_similarity(test_embeddings[g_ids[i]], test_embeddings[g_ids[j]])
                scores_gg.append(sim)
                test_scores.append(sim)
                test_labels.append(1)

    # Genuine - Skilled Forgery (GF)
    for w, g_ids in test_writer_gen.items():
        f_ids = test_writer_forg.get(w, [])
        for f_id in f_ids:
            for g_id in g_ids:
                sim = compute_normalized_similarity(test_embeddings[g_id], test_embeddings[f_id])
                scores_gf.append(sim)
                test_scores.append(sim)
                test_labels.append(0)

    # Genuine - Different Writer (GDW)
    test_writers = sorted(list(test_writer_gen.keys()))
    for w in test_writers:
        other_w = [ow for ow in test_writers if ow != w]
        for g_id in test_writer_gen[w]:
            for ow in other_w:
                target_g_ids = test_writer_gen.get(ow, [])
                if target_g_ids:
                    # Deterministik ilk 2'yi al
                    for target_id in target_g_ids[:2]:
                        sim = compute_normalized_similarity(test_embeddings[g_id], test_embeddings[target_id])
                        scores_gdw.append(sim)
                        test_scores.append(sim)
                        test_labels.append(0)

    # İstatistikleri hesapla
    stats_gg = compute_distribution_stats(scores_gg)
    stats_gf = compute_distribution_stats(scores_gf)
    stats_gdw = compute_distribution_stats(scores_gdw)

    print("\n" + "=" * 80)
    print("  CEDAR TEST KÜMESİ DAĞILIM İSTATİSTİKLERİ")
    print("=" * 80)
    print(f"{'Metrik':<10s} | {'Genuine-Genuine (GG)':<22s} | {'Genuine-Forgery (GF)':<22s} | {'Diff-Writer (GDW)':<22s}")
    print("-" * 80)
    for k in ["count", "mean", "std", "median", "p05", "p25", "p50", "p75", "p95", "min", "max"]:
        print(f"{k:<10s} | {str(stats_gg.get(k)):<22s} | {str(stats_gf.get(k)):<22s} | {str(stats_gdw.get(k)):<22s}")

    # Test Seti Metrikleri (Validation Threshold Kullanılarak)
    test_scores_arr = np.array(test_scores, dtype=np.float64)
    test_labels_arr = np.array(test_labels, dtype=np.int32)

    test_auc = float(roc_auc_score(test_labels_arr, test_scores_arr))
    preds_at_val_th = (test_scores_arr >= opt_val_threshold).astype(int)
    tp = np.sum((preds_at_val_th == 1) & (test_labels_arr == 1))
    fp = np.sum((preds_at_val_th == 1) & (test_labels_arr == 0))
    fn = np.sum((preds_at_val_th == 0) & (test_labels_arr == 1))
    tn = np.sum((preds_at_val_th == 0) & (test_labels_arr == 0))

    accuracy = (tp + tn) / len(test_labels_arr)
    far = fp / max(1, (fp + tn))
    frr = fn / max(1, (fn + tp))
    precision = tp / max(1, (tp + fp))
    recall = tp / max(1, (tp + fn))
    f1 = 2 * precision * recall / max(1e-6, (precision + recall))

    # Kalibre Edilmiş Skor Dağılımı
    calibrated_test_scores = calibrator.predict_proba(test_scores_arr.reshape(-1, 1))[:, 1]
    calibrated_auc = float(roc_auc_score(test_labels_arr, calibrated_test_scores))

    print("\n" + "=" * 80)
    print("  TEST KÜMESİ BAŞARIM METRİKLERİ (Validation'dan Belirlenen Eşik: {:.3f})".format(opt_val_threshold))
    print("=" * 80)
    print(f"[*] Test ROC-AUC           : {test_auc:.5f} (Kalibre Sonrası AUC: {calibrated_auc:.5f})")
    print(f"[*] Test Doğruluğu (Acc)  : {accuracy * 100:.2f}%")
    print(f"[*] FAR (Hatalı Kabul)    : {far * 100:.2f}%")
    print(f"[*] FRR (Hatalı Red)      : {frr * 100:.2f}%")
    print(f"[*] Precision / Recall / F1: {precision:.4f} / {recall:.4f} / {f1:.4f}")
    print(f"[*] Toplam Test Çifti     : {len(test_scores_arr)} (Pozitif GG={len(scores_gg)}, Negatif GF+GDW={len(scores_gf)+len(scores_gdw)})")

    # Sonuçları JSON olarak kaydet
    out_path = PROJECT_ROOT / "artifacts" / "training" / "cedar_distribution_report_sprint21.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report_data = {
        "validation_metrics": {
            "val_auc": val_auc,
            "optimal_val_threshold": opt_val_threshold,
            "val_eer": opt_val_eer,
        },
        "logistic_calibration_params": {
            "intercept": float(calibrator.intercept_[0]),
            "coef": float(calibrator.coef_[0][0]),
        },
        "test_distribution_stats": {
            "genuine_genuine": stats_gg,
            "genuine_forgery": stats_gf,
            "different_writer": stats_gdw,
        },
        "test_metrics_at_val_threshold": {
            "test_auc": round(test_auc, 5),
            "calibrated_auc": round(calibrated_auc, 5),
            "accuracy": round(float(accuracy), 5),
            "far": round(float(far), 5),
            "frr": round(float(frr), 5),
            "precision": round(float(precision), 5),
            "recall": round(float(recall), 5),
            "f1_score": round(float(f1), 5),
            "total_pairs": len(test_scores_arr),
        }
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n[+] Rapor kaydedildi: {out_path}")


if __name__ == "__main__":
    evaluate_cedar_distributions()
