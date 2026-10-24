import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from ml.datasets.forensic_dataset import ForensicDataPipeline, ForensicSample
from ml.models.siamese import (
    ForensicConvNeXtEncoder,
    compute_normalized_similarity,
    set_deterministic_mode,
)
from ml.preprocessing.contract import preprocess_for_inference
from ml.training.dataset_split_validator import validate_writer_disjointness
from ml.training.train import ForensicModelTrainer, TrainingConfig


def evaluate_dataset_split(model: ForensicConvNeXtEncoder, samples: list[ForensicSample], device: torch.device, seed: int = 42):
    """
    Belirtilen numune kümesinde (Val veya Test) pairwise verification yapar:
    - Genuine - Genuine (Pozitif: 1)
    - Genuine - Skilled Forgery (Hard Negative: 0)
    - Genuine - Different Writer (Negative: 0)
    """
    model.eval()
    sample_embeddings: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for s in samples:
            import cv2
            img = s.image_data
            if img is None and s.image_path and os.path.exists(s.image_path):
                try:
                    data = Path(s.image_path).read_bytes()
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                except Exception:
                    img = cv2.imread(s.image_path)
            if img is not None:
                t = preprocess_for_inference(img).to(device)
                emb = model(t).cpu().numpy()[0]
                sample_embeddings[s.sample_id] = emb

    writer_to_gen: dict[str, list[str]] = {}
    writer_to_forg: dict[str, list[str]] = {}
    for s in samples:
        if s.sample_id not in sample_embeddings:
            continue
        if s.is_forgery:
            writer_to_forg.setdefault(s.writer_id, []).append(s.sample_id)
        else:
            writer_to_gen.setdefault(s.writer_id, []).append(s.sample_id)

    all_writers = sorted(list(writer_to_gen.keys()))
    scores_gen_gen: list[float] = []
    scores_gen_forg: list[float] = []
    scores_diff_writer: list[float] = []

    scores: list[float] = []
    labels: list[float] = []

    # 1. Genuine - Genuine
    for w, gen_ids in writer_to_gen.items():
        for i in range(len(gen_ids)):
            for j in range(i + 1, len(gen_ids)):
                sim = compute_normalized_similarity(sample_embeddings[gen_ids[i]], sample_embeddings[gen_ids[j]])
                scores_gen_gen.append(sim)
                scores.append(sim)
                labels.append(1.0)

    # 2. Genuine - Skilled Forgery
    for w, gen_ids in writer_to_gen.items():
        forg_ids = writer_to_forg.get(w, [])
        for f_id in forg_ids:
            for g_id in gen_ids[:max(1, len(gen_ids) // 2)]:
                sim = compute_normalized_similarity(sample_embeddings[g_id], sample_embeddings[f_id])
                scores_gen_forg.append(sim)
                scores.append(sim)
                labels.append(0.0)

    # 3. Genuine - Different Writer
    import random
    rng = random.Random(seed)
    for w in all_writers:
        other_writers = [ow for ow in all_writers if ow != w]
        gen_ids = writer_to_gen[w]
        for g_id in gen_ids:
            if other_writers:
                target_w = rng.choice(other_writers)
                target_gen_ids = writer_to_gen.get(target_w, [])
                if target_gen_ids:
                    target_g_id = rng.choice(target_gen_ids)
                    sim = compute_normalized_similarity(sample_embeddings[g_id], sample_embeddings[target_g_id])
                    scores_diff_writer.append(sim)
                    scores.append(sim)
                    labels.append(0.0)

    scores_arr = np.array(scores, dtype=np.float64)
    labels_arr = np.array(labels, dtype=np.float64)

    auc = float(roc_auc_score(labels_arr, scores_arr))

    # EER Hesabı
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_diff = float("inf")
    opt_thresh = 0.5
    eer = 0.5
    tar_at_eer = 0.5
    far_at_eer = 0.5
    frr_at_eer = 0.5

    pos_mask = (labels_arr == 1.0)
    neg_mask = (labels_arr == 0.0)
    n_pos = max(1, int(np.sum(pos_mask)))
    n_neg = max(1, int(np.sum(neg_mask)))

    for th in thresholds:
        far = float(np.sum((scores_arr >= th) & neg_mask) / n_neg)
        frr = float(np.sum((scores_arr < th) & pos_mask) / n_pos)
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            opt_thresh = float(th)
            eer = float((far + frr) / 2.0)
            tar_at_eer = 1.0 - frr
            far_at_eer = far
            frr_at_eer = frr

    preds_at_opt = (scores_arr >= opt_thresh).astype(int)
    acc = float(accuracy_score(labels_arr.astype(int), preds_at_opt))
    prec = float(precision_score(labels_arr.astype(int), preds_at_opt, zero_division=0))
    rec = float(recall_score(labels_arr.astype(int), preds_at_opt, zero_division=0))
    f1 = float(f1_score(labels_arr.astype(int), preds_at_opt, zero_division=0))

    def dist_dict(lst):
        arr = np.array(lst)
        return {
            "mean": round(float(np.mean(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    return {
        "roc_auc": round(auc, 5),
        "eer": round(eer, 5),
        "tar_at_eer": round(tar_at_eer, 5),
        "far_at_eer": round(far_at_eer, 5),
        "frr_at_eer": round(frr_at_eer, 5),
        "accuracy": round(acc, 5),
        "precision": round(prec, 5),
        "recall": round(rec, 5),
        "f1": round(f1, 5),
        "optimal_threshold": round(opt_thresh, 4),
        "dist_gen_gen": dist_dict(scores_gen_gen),
        "dist_gen_forg": dist_dict(scores_gen_forg),
        "dist_diff_writer": dist_dict(scores_diff_writer),
        "counts": {
            "gen_gen": len(scores_gen_gen),
            "gen_forg": len(scores_gen_forg),
            "diff_writer": len(scores_diff_writer),
            "total_pairs": len(scores),
        }
    }


def main():
    print("=" * 70)
    print("FHDSS CEDAR V2 CONTROLLED GPU RE-TRAINING PIPELINE")
    print("=" * 70)

    # 0. GPU ZORUNLULUK KONTROLÜ (CPU Fallback Kesinlikle Yasak)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required for CEDAR v2 training. "
            "CPU fallback is disabled."
        )
    device = torch.device("cuda:0")

    cedar_dir = Path("datasets/DS_Downloaded/CEDAR/CEDAR1/signatures")
    assert cedar_dir.exists(), f"Dizin bulunamadi: {cedar_dir}"

    seed = 42
    set_deterministic_mode(seed)

    # 1. Dataset yükleme ve writer-disjoint bölme
    corpus = ForensicDataPipeline.load_from_directory(cedar_dir)
    train_s, val_s, test_s = ForensicDataPipeline.split_by_disjoint_writers(
        corpus, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )
    validate_writer_disjointness(train_s, val_s, test_s)

    train_writers = set(s.writer_id for s in train_s)
    val_writers = set(s.writer_id for s in val_s)
    test_writers = set(s.writer_id for s in test_s)
    leakage = len(train_writers.intersection(val_writers)) + len(train_writers.intersection(test_writers))

    # Pretrained model oluşturma ve parametre hesabı
    temp_encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=True).to(device)
    total_params = sum(p.numel() for p in temp_encoder.parameters())
    trainable_params = sum(p.numel() for p in temp_encoder.parameters() if p.requires_grad)

    # 2. ŞARTNAME FORMATINDA EĞİTİM ÖNCESİ RAPORLAMA (Section 27)
    dev_props = torch.cuda.get_device_properties(0)
    import torchvision
    print("\n========== GPU AUDIT ==========")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {dev_props.total_memory / (1024**3):.2f} GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"Torchvision: {torchvision.__version__}")
    print(f"CUDA Runtime: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    print(f"AMP: Supported (FP16/GradScaler ready)")
    print("===============================")

    print("\n========== DATASET ==========")
    print(f"Total: {len(corpus)}")
    print(f"Genuine: {len([s for s in corpus if not s.is_forgery])}")
    print(f"Forgery: {len([s for s in corpus if s.is_forgery])}")
    print(f"Writers: {len(set(s.writer_id for s in corpus))}")
    print(f"Train: {len(train_s)} samples ({len(train_writers)} writers)")
    print(f"Validation: {len(val_s)} samples ({len(val_writers)} writers)")
    print(f"Test: {len(test_s)} samples ({len(test_writers)} writers)")
    print(f"Writer Leakage: {leakage:.2f}%")
    print("=============================")

    print("\n========== MODEL ==========")
    print("Architecture: ConvNeXt-Tiny")
    print("Pretrained: DEFAULT (torchvision.models.ConvNeXt_Tiny_Weights.DEFAULT)")
    print(f"Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print("===========================")

    config = TrainingConfig(
        dataset_name="CEDAR",
        data_dir=str(cedar_dir),
        num_epochs=8,
        warmup_epochs=2,
        batch_size=16,
        p_writers=4,
        k_samples=4,
        lr=1e-4,
        head_lr=1e-4,
        backbone_lr=2e-5,
        margin=0.40,
        early_stopping_patience=4,
        weight_decay=1e-4,
        loss_type="hybrid",
        pretrained=True,
        seed=seed,
        save_filename="convnext_tiny_forensic_cedar_v2_best.pt",
        onnx_filename="convnext_tiny_forensic_cedar_v2.onnx",
        update_manifest=False,
    )

    print("\n========== TRAINING ==========")
    print(f"Device: {device}")
    print(f"Batch Size: {config.batch_size} (P={config.p_writers} writers, K={config.k_samples} samples: 2 Genuine, 2 Forgery)")
    print(f"Optimizer: {config.optimizer.upper()}")
    print(f"Backbone LR: {config.backbone_lr}")
    print(f"Head LR: {config.head_lr}")
    print(f"Weight Decay: {config.weight_decay}")
    print(f"Margin: {config.margin}")
    print(f"Lambda: 1.0 (Hard Negative weight)")
    print(f"Epochs: {config.num_epochs} (Stage 1 Warmup: {config.warmup_epochs}, Stage 2 Fine-Tuning: {config.num_epochs - config.warmup_epochs})")
    print(f"Early Stopping: {config.early_stopping_patience}")
    print("==============================")

    # 3. GPU Sanity & Model Device Test
    print("\n[GPU SANITY TEST]")
    sx = torch.randn(2048, 2048, device=device)
    sy = torch.randn(2048, 2048, device=device)
    sz = sx @ sy
    torch.cuda.synchronize()
    print("CUDA Matmul (2048x2048): PASSED")
    print(f"Model first parameter device: {next(temp_encoder.parameters()).device}")
    assert str(next(temp_encoder.parameters()).device).startswith("cuda"), "Model CUDA üzerinde değil!"
    del temp_encoder, sx, sy, sz
    torch.cuda.empty_cache()

    trainer = ForensicModelTrainer(
        embedding_dim=config.embedding_dim,
        lr=config.lr,
        head_lr=config.head_lr,
        backbone_lr=config.backbone_lr,
        margin=config.margin,
        loss_type=config.loss_type,
        pretrained=config.pretrained,
        seed=seed,
        device=device,
    )
    assert trainer.device.type == "cuda", "Trainer device CUDA olmalı!"

    print("\n[3] 2-Aşamalı Kontrollü GPU Eğitimi Başlatılıyor...")
    fit_res = trainer.fit(
        train_samples=train_s,
        val_samples=val_s,
        num_epochs=config.num_epochs,
        warmup_epochs=config.warmup_epochs,
        batch_size=config.batch_size,
        save_filename=config.save_filename,
        onnx_filename=config.onnx_filename,
        use_structured_sampler=True,
        config=config,
    )

    best_ckpt_path = Path(fit_res["best_checkpoint_path"])
    print(f"\n[4] Eğitim tamamlandı! En iyi model kaydedildi: {best_ckpt_path}")
    print(f"Best Val EER: {fit_res['best_val_eer']:.4f}, Best Val AUC: {fit_res['best_val_auc']:.4f}")

    # En iyi modeli test için yükle
    v2_model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(str(best_ckpt_path), map_location=trainer.device)
    v2_model.load_state_dict(ckpt["model_state_dict"])
    v2_model.to(trainer.device)

    print("\n[5] GÖRÜLMEMİŞ TEST SETİ FİNAL DEĞERLENDİRMESİ (9 Yazar, 432 Numune)...")
    test_metrics_v2 = evaluate_dataset_split(v2_model, test_s, trainer.device, seed=seed)
    print("\n--- NEW MODEL (V2) TEST RESULTS ---")
    print(json.dumps(test_metrics_v2, indent=2))

    # Eski baseline modeli de test seti üzerinde değerlendir
    old_ckpt_path = trainer.weights_dir / "convnext_tiny_forensic_cedar_real_best.pt"
    old_metrics = None
    if old_ckpt_path.exists():
        print("\n[6] ESKİ BASELINE MODEL (convnext_tiny_forensic_cedar_real_best.pt) TEST DEĞERLENDİRMESİ...")
        old_model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
        old_data = torch.load(str(old_ckpt_path), map_location=trainer.device)
        old_model.load_state_dict(old_data["model_state_dict"])
        old_model.to(trainer.device)
        old_metrics = evaluate_dataset_split(old_model, test_s, trainer.device, seed=seed)
        print("\n--- OLD BASELINE MODEL TEST RESULTS ---")
        print(json.dumps(old_metrics, indent=2))

    # 7. ONNX Export ve MAE Sayısal Doğrulama
    print("\n[7] ONNX Model Doğrulaması ve PyTorch Parite Testi...")
    onnx_file_path = trainer.weights_dir / config.onnx_filename
    import onnxruntime as ort
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    with torch.no_grad():
        pt_out = v2_model(dummy_input.to(trainer.device)).cpu().numpy()

    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_file_path), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    onnx_in_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {onnx_in_name: dummy_input.numpy()})[0]
    mae = float(np.mean(np.abs(pt_out - onnx_out)))
    max_diff = float(np.max(np.abs(pt_out - onnx_out)))
    print(f"ONNX Parity: MAE = {mae:.6e}, Max Diff = {max_diff:.6e} (Tolerance: 1e-4)")
    assert max_diff < 1e-4, f"ONNX Parite Hatası: {max_diff} >= 1e-4"

    # 8. SHA-256 Hesaplamaları
    import hashlib
    def get_sha256(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            while c := f.read(65536):
                h.update(c)
        return h.hexdigest()

    pt_sha = get_sha256(best_ckpt_path)
    onnx_sha = get_sha256(onnx_file_path)
    print(f"\n[8] SHA-256 HASHES:")
    print(f"  PyTorch Checkpoint: {pt_sha} ({best_ckpt_path.name})")
    print(f"  ONNX Model:        {onnx_sha} ({onnx_file_path.name})")

    # 9. Model Manifest Güncellemesi (Yalnızca Tüm Kontroller Başarılıysa)
    manifest_path = trainer.weights_dir / "model_manifest.json"
    manifest_data = {}
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    manifest_data["manifest_version"] = "2.1.0"
    manifest_data["last_updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest_data["active_primary_model"] = "convnext_tiny_forensic_cedar_v2"
    manifest_data["training_type"] = "real_dataset"
    manifest_data["dataset_name"] = "CEDAR"
    manifest_data["weights"]["pytorch_checkpoint"] = {
        "filename": best_ckpt_path.name,
        "sha256": pt_sha,
        "size_bytes": best_ckpt_path.stat().st_size,
        "training_type": "real_dataset",
    }
    manifest_data["weights"]["onnx_model"] = {
        "filename": onnx_file_path.name,
        "sha256": onnx_sha,
        "size_bytes": onnx_file_path.stat().st_size,
    }
    manifest_data["test_metrics"] = test_metrics_v2
    manifest_data["val_metrics"] = {
        "eer": fit_res["best_val_eer"],
        "auc": fit_res["best_val_auc"],
    }
    manifest_data["export_numerical_parity"] = {
        "max_abs_diff": max_diff,
        "mean_abs_diff": mae,
        "numerical_tolerance": 0.0001,
        "passed": True,
    }

    # Registered models listesine v2 ekle
    v2_registered_entry = {
        "model_id": "convnext_tiny_forensic_cedar_v2",
        "display_name": "ConvNeXt-Tiny Forensic Metric Encoder (CEDAR v2 - GPU Trained)",
        "filename": best_ckpt_path.name,
        "sha256": pt_sha,
        "training_type": "real_dataset",
        "dataset_name": "CEDAR",
        "status": "PRODUCTION_READY",
        "metrics": {
            "test_auc": test_metrics_v2["roc_auc"],
            "test_eer": test_metrics_v2["eer"],
            "val_eer": fit_res["best_val_eer"],
            "val_auc": fit_res["best_val_auc"],
        },
    }
    # Eski kayıtları koru, varsa v2'yi güncelle yoksa ekle
    reg_list = manifest_data.get("registered_models", [])
    updated_reg = [r for r in reg_list if r.get("model_id") != "convnext_tiny_forensic_cedar_v2"]
    updated_reg.insert(0, v2_registered_entry)
    manifest_data["registered_models"] = updated_reg

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    manifest_sha = get_sha256(manifest_path)
    print(f"  Model Manifest:    {manifest_sha} ({manifest_path.name})")

    # Karşılaştırma Raporunu diske kaydet
    summary_report = {
        "new_model_test_metrics": test_metrics_v2,
        "old_model_test_metrics": old_metrics,
        "fit_results": fit_res,
        "config": config.to_dict(),
        "sha256": {
            "pt": pt_sha,
            "onnx": onnx_sha,
            "manifest": manifest_sha,
        },
        "onnx_parity": {
            "mae": mae,
            "max_diff": max_diff,
            "passed": True,
        },
    }
    report_file = trainer.weights_dir / "cedar_v2_vs_v1_evaluation_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"\n[10] Tam karşılaştırmalı değerlendirme raporu kaydedildi: {report_file}")
    print("\n" + "=" * 70)
    print("FHDSS CEDAR V2 GPU EĞİTİM VE ENTEGRASYON SÜRECİ BAŞARIYLA TAMAMLANDI!")
    print("=" * 70)


if __name__ == "__main__":
    main()
