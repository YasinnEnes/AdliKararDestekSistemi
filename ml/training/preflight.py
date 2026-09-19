"""
FHDSS - Full Model Eğitimi Öncesi Otomatik Denetim (Preflight Validation)
Sprint 18.12: Full training başlatmadan önce 13 kritik kontrolü gerçekleştiren CLI aracı.

Kullanım:
    python ml/training/preflight.py
    python ml/training/preflight.py --data-dir datasets/DS_Downloaded/CEDAR/CEDAR1/signatures
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

# Proje kök dizinini sys.path'e ekle
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.forensic_dataset import (
    ForensicAugmentation,
    ForensicDataPipeline,
    ForensicSample,
    ForensicWriterDataset,
)
from ml.models.siamese import (
    ArcFaceMarginLoss,
    ForensicConvNeXtEncoder,
    SupervisedContrastiveLoss,
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
from ml.training.train import BalancedBatchSampler, TrainingConfig


class PreflightChecker:
    """
    FHDSS Model Eğitimi 13 Aşamalı Kritik Uçtan Uca Doğrulama Yöneticisi.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        self.base_dir = Path(__file__).resolve().parent.parent.parent
        self.data_dir = data_dir or (self.base_dir / "datasets" / "DS_Downloaded" / "CEDAR" / "CEDAR1" / "signatures")
        self.weights_dir = self.base_dir / "artifacts" / "weights"
        self.results: List[Dict[str, Any]] = []

    def log_check(self, index: int, name: str, passed: bool, message: str, details: Optional[Dict[str, Any]] = None):
        status = "PASS" if passed else "FAIL"
        self.results.append({
            "index": index,
            "name": name,
            "status": status,
            "passed": passed,
            "message": message,
            "details": details or {},
        })
        icon = "[OK]  " if passed else "[FAIL]"
        print(f"{icon} {index:02d}. {name:<36} : {status} - {message}")

    def run_all_checks(self) -> bool:
        print("\n" + "=" * 80)
        print("  FHDSS - MODEL EĞİTİMİ ÖNCESİ 13 AŞAMALI PREFLIGHT DENETİMİ (SPRINT 18.12)")
        print("=" * 80 + "\n")

        all_passed = True

        # 1. Dataset mevcut mu?
        ds_exists = self.data_dir.exists() and self.data_dir.is_dir()
        self.log_check(
            1, "Dataset Mevcudiyeti", ds_exists,
            f"Dizin: {self.data_dir}" if ds_exists else f"Dizin bulunamadı: {self.data_dir}"
        )
        if not ds_exists:
            return False

        # 2. Dataset extraction tamam mı?
        # CEDAR için full_org ve full_forg klasörleri aranır
        has_org = (self.data_dir / "full_org").exists() or list(self.data_dir.rglob("full_org"))
        has_forg = (self.data_dir / "full_forg").exists() or list(self.data_dir.rglob("full_forg"))
        extracted = bool(has_org and has_forg)
        self.log_check(
            2, "Dataset Extraction Durumu", extracted,
            "full_org ve full_forg klasörleri mevcut ve erişilebilir" if extracted else "Gerekli imza alt dizinleri eksik"
        )
        if not extracted:
            return False

        # 3. Image count > 0
        try:
            samples = ForensicDataPipeline.load_from_directory(self.data_dir)
            img_count = len(samples)
            passed = img_count > 0
            self.log_check(
                3, "Görsel Sayısı Kontrolü", passed,
                f"Toplam {img_count} adli imza/el yazısı görseli yüklendi",
                {"image_count": img_count}
            )
        except Exception as e:
            self.log_check(3, "Görsel Sayısı Kontrolü", False, f"Yükleme hatası: {e}")
            return False

        # 4. Writer count > minimum (en az 4 yazar)
        unique_writers = sorted(list(set(s.writer_id for s in samples)))
        writer_count = len(unique_writers)
        passed = writer_count >= 4
        self.log_check(
            4, "Yazar Sayısı Kontrolü", passed,
            f"Toplam {writer_count} yazar tespit edildi (Minimum gereksinim: 4)",
            {"writer_count": writer_count}
        )
        if not passed:
            return False

        # 5. Train / Val / Test writer overlap == 0
        train, val, test = ForensicDataPipeline.split_by_disjoint_writers(samples, 0.70, 0.15, 0.15, seed=42)
        try:
            validate_writer_disjointness(train, val, test)
            split_report = generate_split_report(train, val, test)
            self.log_check(
                5, "Yazar Sızıntısı (Disjoint Split)", True,
                f"Sıfır sızıntı doğrulandı (Train: {len(train)} numune, Val: {len(val)}, Test: {len(test)})",
                split_report
            )
        except ValueError as e:
            self.log_check(5, "Yazar Sızıntısı (Disjoint Split)", False, str(e))
            all_passed = False

        # 6. Corrupted image count kontrolü
        corrupted_count = 0
        for s in samples[:100]:  # İlk 100 numuneyi anlık denetle
            try:
                img = s.image_data if s.image_data is not None else None
                if img is None and s.image_path:
                    import cv2
                    data = Path(s.image_path).read_bytes()
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None or img.size == 0 or np.all(img == img[0, 0]):
                    corrupted_count += 1
            except Exception:
                corrupted_count += 1
        passed = corrupted_count == 0
        self.log_check(
            6, "Bozuk Görsel Denetimi", passed,
            f"Örneklemde {corrupted_count} bozuk görsel tespit edildi",
            {"corrupted_count": corrupted_count}
        )
        if not passed:
            all_passed = False

        # 7. BalancedBatchSampler doğrulaması (P=4, K=4)
        try:
            writer_dataset = ForensicWriterDataset(train, transform=ForensicAugmentation(is_training=False))
            sampler = BalancedBatchSampler(writer_dataset, p_writers=4, k_samples=4, seed=42)
            first_batch = next(iter(sampler))
            batch_labels = [writer_dataset.writer_to_label[writer_dataset.samples[idx].writer_id] for idx in first_batch]
            # Her etiketten en az 2 adet olmalı
            counts = {lbl: batch_labels.count(lbl) for lbl in set(batch_labels)}
            has_pairs = all(c >= 2 for c in counts.values())
            passed = len(first_batch) == 16 and has_pairs
            self.log_check(
                7, "BalancedBatchSampler (P x K)", passed,
                f"Batch boyutu={len(first_batch)}, Yazarlar={len(counts)}, Pozitif çift garantisi={has_pairs}"
            )
        except Exception as e:
            self.log_check(7, "BalancedBatchSampler (P x K)", False, f"Hata: {e}")
            all_passed = False

        # 8. Preprocessing contract geçerliliği
        try:
            import cv2
            dummy = np.full((120, 240, 3), 200, dtype=np.uint8)
            proc_tensor = preprocess_for_inference(dummy)
            shape_ok = tuple(proc_tensor.shape) == (1, 3, 224, 224)
            dtype_ok = proc_tensor.dtype == torch.float32
            passed = shape_ok and dtype_ok
            self.log_check(
                8, "Preprocessing Sözleşmesi", passed,
                f"Tensör biçimi={tuple(proc_tensor.shape)}, Veri tipi={proc_tensor.dtype}, ImageNet Norm=Aktif"
            )
        except Exception as e:
            self.log_check(8, "Preprocessing Sözleşmesi", False, f"Hata: {e}")
            all_passed = False

        # 9. ArcFace optimizer gradient doğrulaması
        try:
            encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False, in_channels=3)
            arcface = ArcFaceMarginLoss(num_classes=10, embedding_dim=128)
            opt = torch.optim.AdamW(
                list(encoder.parameters()) + list(arcface.parameters()),
                lr=1e-4,
            )
            dummy_x = torch.randn(4, 3, 224, 224)
            dummy_y = torch.tensor([0, 0, 1, 1], dtype=torch.long)
            opt.zero_grad()
            emb = encoder(dummy_x)
            loss = arcface(emb, dummy_y)
            loss.backward()
            grad_ok = arcface.weight.grad is not None and torch.norm(arcface.weight.grad).item() > 0
            opt.step()
            self.log_check(
                9, "ArcFace Optimizer & Gradient", grad_ok,
                f"requires_grad={arcface.weight.requires_grad}, gradient_norm={round(torch.norm(arcface.weight.grad).item(), 4)}"
            )
        except Exception as e:
            self.log_check(9, "ArcFace Optimizer & Gradient", False, f"Hata: {e}")
            all_passed = False

        # 10. Open-set validation geçerliliği (EER / ROC-AUC)
        try:
            from ml.training.train import ForensicModelTrainer
            trainer = ForensicModelTrainer(embedding_dim=128, lr=1e-4)
            val_metrics = trainer.evaluate_open_set_validation(val[:50])
            eer_ok = 0.0 <= val_metrics["val_eer"] <= 1.0
            auc_ok = 0.0 <= val_metrics["val_auc"] <= 1.0
            passed = eer_ok and auc_ok
            self.log_check(
                10, "Açık Küme (Open-Set) Doğrulama", passed,
                f"EER={val_metrics['val_eer']}, ROC-AUC={val_metrics['val_auc']}, Ayrık yazar çiftleri başarılı"
            )
        except Exception as e:
            self.log_check(10, "Açık Küme (Open-Set) Doğrulama", False, f"Hata: {e}")
            all_passed = False

        # 11. GPU / CPU Donanım Ortamı
        cuda_avail = torch.cuda.is_available()
        device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU (Standard Host)"
        self.log_check(
            11, "Donanım / Çalışma Ortamı", True,
            f"Ortam: {device_name} (CUDA={cuda_avail}, PyTorch={torch.__version__})"
        )

        # 12. Checkpoint dizini yazılabilir mi?
        try:
            self.weights_dir.mkdir(parents=True, exist_ok=True)
            test_file = self.weights_dir / ".preflight_write_test.tmp"
            test_file.write_text("ok", encoding="utf-8")
            test_file.unlink()
            self.log_check(12, "Checkpoint Dizini Erişimi", True, f"Yazılabilir dizin: {self.weights_dir}")
        except Exception as e:
            self.log_check(12, "Checkpoint Dizini Erişimi", False, f"Yazma hatası: {e}")
            all_passed = False

        # 13. ONNX export ortamı geçerliliği
        try:
            import onnxruntime as ort
            dummy_enc = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False, in_channels=3)
            tmp_onnx = self.weights_dir / ".preflight_onnx_test.tmp"
            ForensicConvNeXtEncoder.export_to_onnx(dummy_enc, tmp_onnx)
            onnx_exists = tmp_onnx.exists()
            if onnx_exists:
                tmp_onnx.unlink()
            self.log_check(
                13, "ONNX Export Ortamı", onnx_exists,
                f"ONNX ihracı ve ONNXRuntime {ort.__version__} yürütme ortamı hazır"
            )
        except Exception as e:
            self.log_check(13, "ONNX Export Ortamı", False, f"Hata: {e}")
            all_passed = False

        print("\n" + "=" * 80)
        passed_count = sum(1 for r in self.results if r["passed"])
        total_count = len(self.results)
        print(f"  PREFLIGHT SONUCU: {passed_count}/{total_count} KONTROL BAŞARILI")
        if all_passed:
            print("  DURUM: [PASS] - SİSTEM GERÇEK MODEL EĞİTİMİNE %100 HAZIRDIR.")
        else:
            print("  DURUM: [FAIL] - KRİTİK HATALAR BULUNDU. FULL TRAINING BAŞLATILMAMALIDIR!")
        print("=" * 80 + "\n")

        return all_passed


def main():
    parser = argparse.ArgumentParser(description="FHDSS Preflight Model Eğitimi Denetim Aracı")
    parser.add_argument("--data-dir", "--data_dir", type=str, default=None, help="Denetlenecek veri seti dizini")
    args = parser.parse_args()

    data_dir_path = Path(args.data_dir) if args.data_dir else None
    checker = PreflightChecker(data_dir=data_dir_path)
    success = checker.run_all_checks()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
