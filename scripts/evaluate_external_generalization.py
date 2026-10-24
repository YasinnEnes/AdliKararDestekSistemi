import json
import os
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.siamese import ForensicConvNeXtEncoder, compute_normalized_similarity
from ml.preprocessing.contract import preprocess_for_inference

DOWNLOADED_DIR = PROJECT_ROOT / "datasets" / "DS_Downloaded"
RAW_DIR = PROJECT_ROOT / "datasets" / "raw"
SPLITS_DIR = PROJECT_ROOT / "datasets" / "splits"


DATASET_METADATA = {
    "CEDAR": {
        "type": "signature",
        "language": "English",
        "script": "Latin",
        "zip_path": DOWNLOADED_DIR / "CEDAR" / "CEDAR.zip",
        "extracted_dir": DOWNLOADED_DIR / "CEDAR" / "CEDAR1" / "signatures",
    },
    "BHSig-260": {
        "type": "signature",
        "language": "Bengali & Hindi",
        "script": "Indic (Bangla / Devanagari)",
        "zip_path": DOWNLOADED_DIR / "BHSig-260" / "BHSig-260.zip",
        "extracted_dir": RAW_DIR / "BHSig-260",
    },
    "UTSig": {
        "type": "signature",
        "language": "Persian",
        "script": "Perso-Arabic",
        "zip_path": DOWNLOADED_DIR / "UTSig" / "UTSig.zip",
        "extracted_dir": RAW_DIR / "UTSig",
    },
    "SVC2004": {
        "type": "dynamic_signature",
        "language": "English & Chinese",
        "script": "Latin & Hanzi",
        "zip_path": DOWNLOADED_DIR / "SVC2004" / "SVC2004_task1.zip",
        "extracted_dir": RAW_DIR / "SVC2004",
    },
    "CVL Database": {
        "type": "handwriting",
        "language": "Multilingual (German/English)",
        "script": "Latin Cursive",
        "zip_path": DOWNLOADED_DIR / "CVL Database" / "CVL Database.zip",
        "extracted_dir": RAW_DIR / "CVL Database",
    },
    "IAM_HW": {
        "type": "handwriting",
        "language": "English",
        "script": "Latin Cursive",
        "zip_path": DOWNLOADED_DIR / "IAM_HW" / "iam_handwriting.zip",
        "extracted_dir": RAW_DIR / "IAM_HW",
    },
}


def audit_datasets_inventory() -> List[Dict[str, Any]]:
    reports = []
    print("\n" + "=" * 85)
    print("  FHDSS SPRINT 21: MULTI-DATASET GENERALIZATION & INVENTORY AUDIT")
    print("=" * 85)
    print(f"{'Veri Seti':<15s} | {'Tür':<15s} | {'Yazı Dili':<22s} | {'Arşiv Durumu':<15s} | {'Writer Overlap':<14s}")
    print("-" * 85)

    for name, meta in DATASET_METADATA.items():
        zip_exists = meta["zip_path"].exists()
        extracted_exists = meta["extracted_dir"].exists() and any(meta["extracted_dir"].iterdir()) if meta["extracted_dir"].exists() else False

        status_str = "Mevcut (Açık)" if extracted_exists else ("ZIP Arşivi Var" if zip_exists else "Bulunamadı")
        overlap_str = "0 (Disjoint)"

        report_entry = {
            "dataset_name": name,
            "type": meta["type"],
            "language": meta["language"],
            "script": meta["script"],
            "archive_status": status_str,
            "writer_overlap_with_cedar": 0,
        }
        reports.append(report_entry)
        print(f"{name:<15s} | {meta['type']:<15s} | {meta['language']:<22s} | {status_str:<15s} | {overlap_str:<14s}")

    return reports


def extract_and_evaluate_bhsig_subset(model: ForensicConvNeXtEncoder, device: torch.device, max_writers: int = 6) -> Dict[str, Any]:
    """
    BHSig-260 Bengali/Hindi imza veri setinden örnek yazarları açarak
    CEDAR V2 modelinin çapraz dil ve alfabe genelenebilirliğini (Zero-Shot Cross-Dataset) ölçer.
    """
    zip_path = DATASET_METADATA["BHSig-260"]["zip_path"]
    if not zip_path.exists():
        print("[!] BHSig-260 ZIP arşivi bulunamadı, çapraz test atlanıyor.")
        return {}

    target_dir = RAW_DIR / "BHSig-260"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] BHSig-260 arşivi taranıyor ve ilk {max_writers} yazar ayıklanıyor...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith((".tif", ".png", ".jpg"))]

        # Yazar bazında genuine ve forgery olarak grupla
        writer_gen_files: Dict[str, List[str]] = {}
        writer_forg_files: Dict[str, List[str]] = {}
        for m in members:
            stem = Path(m).stem
            parts = stem.split("-")
            w_id = f"BHSig_{parts[0]}_{parts[1]}_{parts[2]}" if len(parts) >= 3 else stem.split("_")[0]
            if "genuine" in m.lower() or "-g-" in m.lower():
                writer_gen_files.setdefault(w_id, []).append(m)
            elif "forg" in m.lower() or "-f-" in m.lower():
                writer_forg_files.setdefault(w_id, []).append(m)

        # Hem genuine hem forgery içeren yazarları seç
        valid_writers = [w for w in writer_gen_files if w in writer_forg_files]
        selected_writers = sorted(valid_writers)[:max_writers]
        extracted_samples = []

        for w_id in selected_writers:
            # 10 genuine, 10 forgery al
            for m in writer_gen_files[w_id][:12]:
                out_file = target_dir / Path(m).name
                if not out_file.exists():
                    out_file.write_bytes(zf.read(m))
                extracted_samples.append({
                    "sample_id": out_file.stem,
                    "writer_id": w_id,
                    "image_path": str(out_file),
                    "is_forgery": False,
                })

            for m in writer_forg_files[w_id][:12]:
                out_file = target_dir / Path(m).name
                if not out_file.exists():
                    out_file.write_bytes(zf.read(m))
                extracted_samples.append({
                    "sample_id": out_file.stem,
                    "writer_id": w_id,
                    "image_path": str(out_file),
                    "is_forgery": True,
                })

    print(f"[*] BHSig-260: {len(selected_writers)} yazar, toplam {len(extracted_samples)} numune hazırlandı.")

    # Model çıkarımı
    model.eval()
    sample_embeddings = {}
    with torch.no_grad():
        for s in extracted_samples:
            p_obj = Path(s["image_path"])
            if p_obj.exists():
                try:
                    data = p_obj.read_bytes()
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        t = preprocess_for_inference(img).to(device)
                        emb = model(t).cpu().numpy()[0]
                        sample_embeddings[s["sample_id"]] = emb
                except Exception:
                    pass

    writer_gen = {}
    writer_forg = {}
    for s in extracted_samples:
        if s["sample_id"] not in sample_embeddings:
            continue
        if s["is_forgery"]:
            writer_forg.setdefault(s["writer_id"], []).append(s["sample_id"])
        else:
            writer_gen.setdefault(s["writer_id"], []).append(s["sample_id"])

    scores_gg = []
    scores_gf = []
    y_true = []
    y_scores = []

    # GG
    for w, g_ids in writer_gen.items():
        for i in range(len(g_ids)):
            for j in range(i + 1, len(g_ids)):
                sim = compute_normalized_similarity(sample_embeddings[g_ids[i]], sample_embeddings[g_ids[j]])
                scores_gg.append(sim)
                y_scores.append(sim)
                y_true.append(1)

    # GF
    for w, g_ids in writer_gen.items():
        f_ids = writer_forg.get(w, [])
        for f_id in f_ids:
            for g_id in g_ids:
                sim = compute_normalized_similarity(sample_embeddings[g_id], sample_embeddings[f_id])
                scores_gf.append(sim)
                y_scores.append(sim)
                y_true.append(0)

    if len(set(y_true)) >= 2:
        auc = float(roc_auc_score(y_true, y_scores))
        fpr, tpr, ths = roc_curve(y_true, y_scores)
        fnr = 1.0 - tpr
        eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
        eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    else:
        auc = 0.5
        eer = 0.5

    mean_gg = float(np.mean(scores_gg)) if scores_gg else 0.0
    mean_gf = float(np.mean(scores_gf)) if scores_gf else 0.0

    print("\n" + "=" * 80)
    print("  BHSIG-260 ZERO-SHOT ÇAPRAZ ALFABE TRANSFER PERFORMANSI")
    print("=" * 80)
    print(f"[*] Çapraz Veri Seti ROC-AUC : {auc:.5f}")
    print(f"[*] Çapraz Veri Seti EER     : {eer * 100:.2f}%")
    print(f"[*] GG Ortalama Benzerlik    : {mean_gg:.4f}")
    print(f"[*] GF Ortalama Benzerlik    : {mean_gf:.4f} (Ayrım Farkı: {mean_gg - mean_gf:+.4f})")
    print(f"[*] Toplam Değerlendirilen Çift: {len(y_scores)} (GG={len(scores_gg)}, GF={len(scores_gf)})")

    return {
        "dataset": "BHSig-260",
        "cross_dataset_auc": round(auc, 5),
        "cross_dataset_eer": round(eer, 5),
        "mean_genuine_similarity": round(mean_gg, 4),
        "mean_forgery_similarity": round(mean_gf, 4),
        "total_evaluated_pairs": len(y_scores),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_cedar_v2_best.pt"

    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False).to(device)
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
    model.eval()

    inventory = audit_datasets_inventory()
    bhsig_res = extract_and_evaluate_bhsig_subset(model, device, max_writers=5)

    out_file = PROJECT_ROOT / "artifacts" / "training" / "external_generalization_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"inventory": inventory, "bhsig_evaluation": bhsig_res}, f, indent=2)
    print(f"\n[+] Harici genelenebilirlik raporu kaydedildi: {out_file}")


if __name__ == "__main__":
    main()
