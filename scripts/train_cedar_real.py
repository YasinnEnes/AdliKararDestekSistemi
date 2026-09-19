"""
FHDSS - Sprint 19: Gerçek CEDAR Veri Seti ile Model Eğitimi ve Açık Küme Değerlendirmesi
(End-to-End Real Forensic Model Training, Validation, Test Evaluation, and ONNX Deployment)
"""

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

# Proje kök dizinini sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.forensic_dataset import (
    ForensicAugmentation,
    ForensicDataPipeline,
    ForensicSample,
    ForensicWriterDataset,
    PairType,
)
from ml.models.siamese import (
    ForensicConvNeXtEncoder,
    SupervisedContrastiveLoss,
    compute_cosine_similarity,
    compute_normalized_similarity,
)
from ml.preprocessing.contract import preprocess_for_inference
from ml.training.dataset_split_validator import (
    generate_split_report,
    validate_writer_disjointness,
)
from ml.training.preflight import PreflightChecker
from ml.training.train import (
    BalancedBatchSampler,
    ForensicModelTrainer,
    TrainingConfig,
    get_git_commit_hash,
    set_deterministic_seed,
)

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


def calculate_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def evaluate_test_set(
    model: ForensicConvNeXtEncoder,
    test_samples: List[ForensicSample],
    device: torch.device,
    seed: int = 42,
    max_pairs_per_writer: int = 24,
) -> Dict[str, Any]:
    """
    Modelin hiç görmediği bağımsız test yazarları (unseen writers) üzerinde
    adli açık küme (open-set) doğrulama metriklerini hesaplar.
    """
    model.eval()
    pairs = ForensicDataPipeline.create_pairs(test_samples, max_pairs_per_writer=max_pairs_per_writer, seed=seed)
    
    genuine_scores: List[float] = []
    impostor_scores: List[float] = []
    y_true: List[int] = []
    y_scores: List[float] = []
    raw_cosines: List[float] = []

    with torch.no_grad():
        for s_a, s_b, label, p_type in pairs:
            # Görselleri yükle
            img_a = s_a.image_data
            if img_a is None and s_a.image_path:
                import cv2
                data_a = Path(s_a.image_path).read_bytes()
                img_a = cv2.imdecode(np.frombuffer(data_a, np.uint8), cv2.IMREAD_COLOR)

            img_b = s_b.image_data
            if img_b is None and s_b.image_path:
                import cv2
                data_b = Path(s_b.image_path).read_bytes()
                img_b = cv2.imdecode(np.frombuffer(data_b, np.uint8), cv2.IMREAD_COLOR)

            if img_a is None or img_b is None:
                continue

            t_a = preprocess_for_inference(img_a).to(device)
            t_b = preprocess_for_inference(img_b).to(device)

            emb_a = model(t_a).cpu().numpy()[0]
            emb_b = model(t_b).cpu().numpy()[0]

            raw_cos = float(compute_cosine_similarity(emb_a, emb_b))
            norm_sim = float(compute_normalized_similarity(emb_a, emb_b))

            raw_cosines.append(raw_cos)
            y_scores.append(norm_sim)
            lbl = int(label)
            y_true.append(lbl)

            if lbl == 1:
                genuine_scores.append(norm_sim)
            else:
                impostor_scores.append(norm_sim)

    # ROC-AUC ve EER hesabı
    if HAS_SKLEARN and len(set(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_scores))
        fpr_arr, tpr_arr, thresholds = roc_curve(y_true, y_scores)
        fnr_arr = 1.0 - tpr_arr
        eer_idx = int(np.nanargmin(np.abs(fpr_arr - fnr_arr)))
        eer = float((fpr_arr[eer_idx] + fnr_arr[eer_idx]) / 2.0)
        opt_thresh = float(thresholds[eer_idx]) if eer_idx < len(thresholds) else 0.5
        tar_at_eer = float(tpr_arr[eer_idx])
        far_at_eer = float(fpr_arr[eer_idx])
        frr_at_eer = float(fnr_arr[eer_idx])
    else:
        auc = 0.5
        eer = 0.5
        opt_thresh = 0.5
        tar_at_eer = 0.5
        far_at_eer = 0.5
        frr_at_eer = 0.5

    # Belirlenen sabit eşiklerde detaylı metrik analizi (0.50, 0.60, 0.70, 0.80, 0.90)
    fixed_thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]
    threshold_analysis = []

    y_arr = np.array(y_true)
    s_arr = np.array(y_scores)
    positives = (y_arr == 1)
    negatives = (y_arr == 0)
    num_pos = max(1, int(np.sum(positives)))
    num_neg = max(1, int(np.sum(negatives)))

    for th in fixed_thresholds:
        pred_pos = (s_arr >= th)
        tp = int(np.sum(pred_pos & positives))
        fp = int(np.sum(pred_pos & negatives))
        fn = int(np.sum((~pred_pos) & positives))
        tn = int(np.sum((~pred_pos) & negatives))

        tar = round(tp / num_pos, 4)
        far = round(fp / num_neg, 4)
        frr = round(fn / num_pos, 4)
        acc = round((tp + tn) / (num_pos + num_neg), 4)

        threshold_analysis.append({
            "threshold": th,
            "TAR": tar,
            "FAR": far,
            "FRR": frr,
            "Accuracy": acc,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
        })

    return {
        "total_test_pairs": len(y_true),
        "genuine_pairs_count": len(genuine_scores),
        "impostor_pairs_count": len(impostor_scores),
        "roc_auc": round(auc, 4),
        "eer": round(eer, 4),
        "optimal_eer_threshold": round(opt_thresh, 4),
        "tar_at_eer": round(tar_at_eer, 4),
        "far_at_eer": round(far_at_eer, 4),
        "frr_at_eer": round(frr_at_eer, 4),
        "genuine_similarity_mean": round(float(np.mean(genuine_scores)), 4) if genuine_scores else 0.0,
        "genuine_similarity_std": round(float(np.std(genuine_scores)), 4) if genuine_scores else 0.0,
        "impostor_similarity_mean": round(float(np.mean(impostor_scores)), 4) if impostor_scores else 0.0,
        "impostor_similarity_std": round(float(np.std(impostor_scores)), 4) if impostor_scores else 0.0,
        "raw_cosine_min": round(float(np.min(raw_cosines)), 4) if raw_cosines else 0.0,
        "raw_cosine_max": round(float(np.max(raw_cosines)), 4) if raw_cosines else 0.0,
        "threshold_analysis": threshold_analysis,
    }


def main():
    print("\n" + "=" * 80)
    print("  FHDSS - SPRINT 19: GERÇEK CEDAR VERİ SETİ İLE MODEL EĞİTİMİ VE DEĞERLENDİRME")
    print("=" * 80 + "\n")

    set_deterministic_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Çalışma Cihazı: {device}")

    # 1. Preflight Kontrolü
    preflight = PreflightChecker()
    pre_ok = preflight.run_all_checks()
    if not pre_ok:
        print("\n[!] KRİTİK HATA: Preflight denetimi başarısız oldu. Model eğitimi durduruldu.")
        sys.exit(1)

    # 2. CEDAR Veri Setini Yükle
    cedar_dir = PROJECT_ROOT / "datasets" / "DS_Downloaded" / "CEDAR" / "CEDAR1" / "signatures"
    print(f"\n[*] CEDAR veri seti yükleniyor: {cedar_dir}")
    samples = ForensicDataPipeline.load_from_directory(cedar_dir)
    print(f"[OK] Toplam {len(samples)} imza görseli yüklendi.")

    # 3. Sıfır Sızıntılı Yazar Ayrımı (Train / Val / Test)
    train_samples, val_samples, test_samples = ForensicDataPipeline.split_by_disjoint_writers(
        samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42
    )
    validate_writer_disjointness(train_samples, val_samples, test_samples)
    split_report = generate_split_report(train_samples, val_samples, test_samples)

    print("\n[*] Yazar Ayrım İstatistikleri (Zero Writer Leakage):")
    for s_name, s_info in split_report["splits"].items():
        print(f"    - {s_name.upper():<6}: {s_info['writer_count']} yazar, {s_info['sample_count']} numune")

    # Split manifestini kaydet
    artifacts_training = PROJECT_ROOT / "artifacts" / "training"
    artifacts_training.mkdir(parents=True, exist_ok=True)
    split_manifest_path = artifacts_training / "cedar_split_manifest.json"
    with open(split_manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": "CEDAR",
            "seed": 42,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "report": split_report,
            "train_writers": sorted(list(set(s.writer_id for s in train_samples))),
            "val_writers": sorted(list(set(s.writer_id for s in val_samples))),
            "test_writers": sorted(list(set(s.writer_id for s in test_samples))),
        }, f, indent=2)
    print(f"[OK] cedar_split_manifest.json kaydedildi: {split_manifest_path}")

    # 4. Eğitim Yapılandırması (10 Epoch, Batch 16 = 4x4, lr=1e-4, SupCon)
    log_dir = artifacts_training / "cedar_real_training"
    log_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = PROJECT_ROOT / "artifacts" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    save_pt_name = "convnext_tiny_forensic_cedar_real_best.pt"
    save_onnx_name = "convnext_tiny_forensic_cedar_real.onnx"

    config = TrainingConfig(
        dataset_name="CEDAR",
        data_dir=str(cedar_dir),
        num_epochs=10,
        batch_size=16,
        p_writers=4,
        k_samples=4,
        lr=1e-4,
        weight_decay=1e-4,
        loss_type="supcon",
        embedding_dim=128,
        seed=42,
        training_type="real_dataset",
        checkpoint_dir=str(weights_dir),
        save_filename=save_pt_name,
    )

    with open(log_dir / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    # 5. Sıfırdan Model Eğitimi (Sentetik ağırlık kullanılmaz!)
    print("\n" + "-" * 80)
    print("  GERÇEK CEDAR MODEL EĞİTİMİ BAŞLATILIYOR (10 EPOCH)")
    print("-" * 80)

    trainer = ForensicModelTrainer(
        embedding_dim=config.embedding_dim,
        lr=config.lr,
        weight_decay=config.weight_decay,
        loss_type=config.loss_type,
        weights_dir=weights_dir,
        device=device,
        seed=config.seed,
    )

    training_start = time.time()
    train_results = trainer.fit(
        train_samples=train_samples,
        val_samples=val_samples,
        num_epochs=config.num_epochs,
        batch_size=config.batch_size,
        save_filename=save_pt_name,
        training_type="real_dataset",
        dataset_name="CEDAR",
        config=config,
    )
    total_training_sec = round(time.time() - training_start, 2)
    print(f"\n[OK] Eğitim {total_training_sec} saniyede tamamlandı.")

    # 6. ONNX İhracı ve Doğrulama
    best_pt_path = weights_dir / save_pt_name
    onnx_path = weights_dir / save_onnx_name

    print(f"\n[*] Model ONNX formatına ihraç ediliyor -> {onnx_path.name}")
    onnx_res = trainer.export_and_validate_onnx(
        checkpoint_path=best_pt_path,
        val_metrics=train_results.get("best_val_metrics", {}),
        onnx_filename=save_onnx_name,
        numerical_tolerance=1e-4,
        training_type="real_dataset",
        dataset_name="CEDAR",
        config_dict=config.to_dict(),
    )
    print(f"[OK] ONNX İhracı ve Sayısal Doğrulama: {onnx_res['export_status']} (Maks Fark: {onnx_res['max_abs_diff']:.2e})")

    # 7. Bağımsız TEST Kümesinde Açık Küme Değerlendirmesi (9 Görülmemiş Yazar)
    print("\n" + "-" * 80)
    print("  BAĞIMSIZ TEST KÜMESİNDE AÇIK KÜME (OPEN-SET) ADLİ DEĞERLENDİRME")
    print("-" * 80)

    test_eval = evaluate_test_set(
        model=trainer.model,
        test_samples=test_samples,
        device=device,
        seed=42,
    )

    print(f"[*] Toplam Test Çifti Sayısı : {test_eval['total_test_pairs']}")
    print(f"[*] ROC-AUC Skoru            : {test_eval['roc_auc']:.4f}")
    print(f"[*] EER (Equal Error Rate)   : {test_eval['eer']:.4f} (%{test_eval['eer']*100:.2f})")
    print(f"[*] Optimum EER Eşik Değeri  : {test_eval['optimal_eer_threshold']:.4f}")
    print(f"[*] TAR @ EER                : {test_eval['tar_at_eer']:.4f}")
    print(f"[*] FAR @ EER                : {test_eval['far_at_eer']:.4f}")
    print(f"[*] FRR @ EER                : {test_eval['frr_at_eer']:.4f}")
    print(f"[*] Orijinal Çift Benzerliği : {test_eval['genuine_similarity_mean']:.4f} ± {test_eval['genuine_similarity_std']:.4f}")
    print(f"[*] Sahte/Farklı Benzerliği  : {test_eval['impostor_similarity_mean']:.4f} ± {test_eval['impostor_similarity_std']:.4f}")

    print("\n[*] Eşik Değeri Kalibrasyon Analizi (Threshold Sweep):")
    print("    Eşik  |  TAR   |  FAR   |  FRR   | Doğruluk (Accuracy)")
    print("   -------+--------+--------+--------+--------------------")
    for row in test_eval["threshold_analysis"]:
        print(f"    {row['threshold']:.2f}  | {row['TAR']:.4f} | {row['FAR']:.4f} | {row['FRR']:.4f} | {row['Accuracy']:.4f}")

    # Test sonuçlarını kaydet
    test_eval_path = log_dir / "test_evaluation_report.json"
    with open(test_eval_path, "w", encoding="utf-8") as f:
        json.dump(test_eval, f, indent=2)
    print(f"\n[OK] test_evaluation_report.json kaydedildi: {test_eval_path}")

    # Eğitim geçmişini kaydet
    with open(log_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(train_results.get("history", {}), f, indent=2)

    with open(log_dir / "best_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "val_eer": train_results.get("best_val_eer"),
            "val_auc": train_results.get("best_val_auc"),
            "test_roc_auc": test_eval["roc_auc"],
            "test_eer": test_eval["eer"],
            "test_tar_at_eer": test_eval["tar_at_eer"],
            "elapsed_seconds": total_training_sec,
        }, f, indent=2)

    # 8. Model Manifestosunu Güncelle (Sentetik Baseline'ı Kaybetmeden)
    manifest_path = weights_dir / "model_manifest.json"
    pt_sha = calculate_sha256(best_pt_path)
    onnx_sha = calculate_sha256(onnx_path)

    existing_manifest = {}
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        except Exception:
            existing_manifest = {}

    # Sentetik modeli koru
    synthetic_entry = existing_manifest.get("synthetic_baseline", {
        "model_id": "convnext_tiny_synthetic_baseline",
        "training_type": "synthetic_test",
        "pt_filename": "convnext_tiny_forensic_best.pt",
        "sha256": existing_manifest.get("weights", {}).get("pytorch_checkpoint", {}).get("sha256", ""),
        "notes": "Boru hattı doğrulama amaçlı sentetik baseline model.",
    })

    updated_manifest = {
        "manifest_version": "2.0.0",
        "last_updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "active_primary_model": "convnext_tiny_forensic_cedar_real",
        "training_type": "real_dataset",
        "dataset_name": "CEDAR",
        "dataset_statistics": {
            "total_samples": 2640,
            "total_writers": 55,
            "train_writers": 38,
            "val_writers": 8,
            "test_writers": 9,
            "train_samples": len(train_samples),
            "val_samples": len(val_samples),
            "test_samples": len(test_samples),
        },
        "model_architecture": "ConvNeXt-Tiny",
        "embedding_dimension": 128,
        "source_code_commit": get_git_commit_hash(),
        "random_seed": 42,
        "preprocessing_version": "v1.2-adli-pipeline",
        "weights": {
            "pytorch_checkpoint": {
                "filename": save_pt_name,
                "sha256": pt_sha,
                "size_bytes": best_pt_path.stat().st_size,
                "training_type": "real_dataset",
            },
            "onnx_model": {
                "filename": save_onnx_name,
                "sha256": onnx_sha,
                "size_bytes": onnx_path.stat().st_size,
            },
            "synthetic_baseline_pt": {
                "filename": "convnext_tiny_forensic_best.pt",
                "sha256": synthetic_entry.get("sha256", ""),
                "training_type": "synthetic_test",
            },
        },
        "test_metrics": {
            "roc_auc": test_eval["roc_auc"],
            "eer": test_eval["eer"],
            "optimal_threshold": test_eval["optimal_eer_threshold"],
            "tar_at_eer": test_eval["tar_at_eer"],
            "far_at_eer": test_eval["far_at_eer"],
            "frr_at_eer": test_eval["frr_at_eer"],
        },
        "export_numerical_parity": {
            "max_abs_diff": onnx_res["max_abs_diff"],
            "passed": onnx_res["parity_passed"],
        },
        "synthetic_baseline": synthetic_entry,
        "registered_models": [
            {
                "model_id": "convnext_tiny_forensic_cedar_real",
                "pt_filename": save_pt_name,
                "onnx_filename": save_onnx_name,
                "training_type": "real_dataset",
                "dataset": "CEDAR",
                "pt_sha256": pt_sha,
                "onnx_sha256": onnx_sha,
                "test_roc_auc": test_eval["roc_auc"],
                "test_eer": test_eval["eer"],
                "status": "PRIMARY_DEPLOYED",
            },
            {
                "model_id": "convnext_tiny_synthetic_baseline",
                "pt_filename": "convnext_tiny_forensic_best.pt",
                "training_type": "synthetic_test",
                "status": "RETAINED_BASELINE",
            },
        ],
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(updated_manifest, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] model_manifest.json güncellendi: {manifest_path}")

    # 9. Model Sicili (Registry) Doğrulaması
    from ml.models.registry import model_registry
    model_registry._manifest_cache = None
    is_real_type = model_registry.get_model_training_type(save_pt_name)
    val_ok, _, _ = model_registry.verify_weights_integrity(best_pt_path)

    print(f"\n[*] Model Sicili Doğrulaması: Model Tipi={is_real_type}, Bütünlük Kontrolü={val_ok}")

    print("\n" + "=" * 80)
    print("  REAL MODEL TRAINING COMPLETED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
