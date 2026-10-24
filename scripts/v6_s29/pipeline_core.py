"""
FHDSS SPRINT 29 — Comprehensive Pipeline Runner
Covers:
- Phase 2: Baseline Reproducibility (V6 rerun across all 5 corpora, verification against Sprint 28)
- Phase 3: Dataset Geometry Analysis (Width, Height, AR, BBox, Occupancy, Stroke Density, CC Count, P5-P95)
- Phase 4: Preprocessing Ablation on Frozen V6 (EXP-A, EXP-C, EXP-E, EXP-F MAR)
- Phase 5: Information-Preserving Rule & Quality Flags Tracking
- Phase 6: Warm-Start Fine-Tuning (Candidates A, B, C, D, E)
- Phase 7: Cross-Domain & Embedding Quality Analysis (Centroid Distances, GG vs GF Separation)
- Phase 8: Statistical Significance (1000 Bootstrap Iterations, 95% CI, p-values)
- Phase 9: ONNX Parity & Determinism (10 consecutive runs, zero drift)
- Phase 10: V5 Shadow Comparison & Disagreement Analysis (100 pairs)
- Phase 11: Production Promotion Audit & Report Generation
"""

import sys

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.forensic_dataset import ForensicSample
from ml.datasets.v5_multidomain_loader import load_and_split_v5_corpora
from ml.models.siamese import (
    ForensicConvNeXtEncoder,
    ForensicHybridLoss,
    compute_cosine_similarity,
    compute_normalized_similarity,
)
from ml.preprocessing.ablation import (
    extract_tiled_representation,
    find_ink_bounding_box,
    preprocess_exp_a_baseline,
    preprocess_exp_b_letterbox,
    preprocess_exp_c_content_crop,
    preprocess_exp_d_long_axis,
)
from ml.preprocessing.v6_s29_mar import (
    Sprint29MARAugmentation,
    compute_geometry_descriptors,
    detect_foreground_bbox,
    preprocess_exp_f_multi_aspect_ratio,
)

# Safe print
_orig_print = print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    try:
        _orig_print(*args, **kwargs)
    except UnicodeEncodeError:
        safe_args = [str(a).encode("ascii", "replace").decode("ascii") for a in args]
        _orig_print(*safe_args, **kwargs)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==============================================================================
# Statistical & Distribution Utilities
# ==============================================================================

def compute_percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"min": 0.0, "p5": 0.0, "p25": 0.0, "median": 0.0, "mean": 0.0, "p75": 0.0, "p95": 0.0, "max": 0.0, "std": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "min": round(float(np.min(arr)), 4),
        "p5": round(float(np.percentile(arr, 5)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
        "max": round(float(np.max(arr)), 4),
        "std": round(float(np.std(arr)), 4),
    }


# ==============================================================================
# Embedding Extraction for Ablation Modes
# ==============================================================================

def extract_embeddings_for_mode(
    model: nn.Module,
    samples: List[ForensicSample],
    device: torch.device,
    mode: str = "EXP-A"
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, Any]]]:
    """
    Extracts L2 normalized embeddings (128D) under specified preprocessing mode.
    Returns: (embeddings_dict, metadata_dict)
    """
    model.eval()
    embeddings = {}
    metadata = {}

    for s in samples:
        try:
            data = Path(s.image_path).read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue

            meta = {}
            if mode == "EXP-A":
                t = preprocess_exp_a_baseline(img, is_bgr=True).to(device)
                with torch.no_grad():
                    emb = model(t).cpu().numpy()[0]
            elif mode == "EXP-C":
                t = preprocess_exp_c_content_crop(img, is_bgr=True).to(device)
                with torch.no_grad():
                    emb = model(t).cpu().numpy()[0]
            elif mode == "EXP-E":
                emb = extract_tiled_representation(model, img, device, is_bgr=True)
            elif mode == "EXP-F":
                t, meta = preprocess_exp_f_multi_aspect_ratio(img, is_bgr=True)
                t = t.to(device)
                with torch.no_grad():
                    emb = model(t).cpu().numpy()[0]
            else:
                raise ValueError(f"Unknown mode: {mode}")

            # Ensure unit length
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            embeddings[s.sample_id] = emb
            metadata[s.sample_id] = meta
        except Exception:
            continue

    return embeddings, metadata


# ==============================================================================
# Full Forensic Evaluation on Pairs
# ==============================================================================

def evaluate_pairs_detailed(
    embeddings: Dict[str, np.ndarray],
    samples: List[ForensicSample],
    dataset_filter: Optional[str] = None
) -> Dict[str, Any]:
    writer_gen: Dict[str, List[str]] = {}
    writer_forg: Dict[str, List[str]] = {}

    for s in samples:
        if s.sample_id not in embeddings:
            continue
        ds = s.metadata.get("dataset", "UNKNOWN")
        if dataset_filter and ds != dataset_filter:
            continue
        if s.is_forgery:
            writer_forg.setdefault(s.writer_id, []).append(s.sample_id)
        else:
            writer_gen.setdefault(s.writer_id, []).append(s.sample_id)

    gg_scores, gf_scores, gdw_scores = [], [], []
    y_true, y_scores = [], []
    pair_details = []

    # 1. Genuine-Genuine (Target = 1)
    for w, g_ids in writer_gen.items():
        for i in range(len(g_ids)):
            for j in range(i + 1, min(i + 6, len(g_ids))):
                sim = compute_normalized_similarity(embeddings[g_ids[i]], embeddings[g_ids[j]])
                gg_scores.append(sim)
                y_scores.append(sim)
                y_true.append(1)
                pair_details.append({
                    "pair_type": "GG", "writer": w, "s1": g_ids[i], "s2": g_ids[j], "score": sim, "target": 1
                })

    # 2. Genuine-Forgery (Target = 0)
    for w, g_ids in writer_gen.items():
        f_ids = writer_forg.get(w, [])
        for f_id in f_ids[:5]:
            for g_id in g_ids[:5]:
                sim = compute_normalized_similarity(embeddings[g_id], embeddings[f_id])
                gf_scores.append(sim)
                y_scores.append(sim)
                y_true.append(0)
                pair_details.append({
                    "pair_type": "GF", "writer": w, "s1": g_id, "s2": f_id, "score": sim, "target": 0
                })

    # 3. Genuine-Different Writer (Target = 0)
    writers = sorted(list(writer_gen.keys()))
    rng = random.Random(42)
    diff_pairs = []
    for i in range(len(writers)):
        for j in range(i + 1, min(i + 4, len(writers))):
            diff_pairs.append((writers[i], writers[j]))
    rng.shuffle(diff_pairs)
    for w1, w2 in diff_pairs[:600]:
        g1 = writer_gen.get(w1, [])
        g2 = writer_gen.get(w2, [])
        if g1 and g2:
            sim = compute_normalized_similarity(embeddings[g1[0]], embeddings[g2[0]])
            gdw_scores.append(sim)
            y_scores.append(sim)
            y_true.append(0)
            pair_details.append({
                "pair_type": "GDW", "writer": f"{w1}_{w2}", "s1": g1[0], "s2": g2[0], "score": sim, "target": 0
            })

    if len(set(y_true)) < 2:
        return {"auc": 0.5, "eer": 0.5, "delta_gg_gf": 0.0, "total_pairs": len(y_true)}

    auc = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, ths = roc_curve(y_true, y_scores)
    fnr = 1.0 - tpr
    eer_idx = int(np.nanargmin(np.abs(fpr - fnr)))
    eer = float((fpr[eer_idx] + fnr[eer_idx]) / 2.0)
    eer_th = float(ths[eer_idx])

    far_at_eer = float(fpr[eer_idx])
    frr_at_eer = float(fnr[eer_idx])
    tar_at_eer = float(tpr[eer_idx])

    far5_idx = np.where(fpr <= 0.05)[0]
    th_far5 = float(ths[far5_idx[-1]]) if len(far5_idx) > 0 else eer_th
    tar_far5 = float(tpr[far5_idx[-1]]) if len(far5_idx) > 0 else (1.0 - eer)

    far1_idx = np.where(fpr <= 0.01)[0]
    th_far1 = float(ths[far1_idx[-1]]) if len(far1_idx) > 0 else eer_th
    tar_far1 = float(tpr[far1_idx[-1]]) if len(far1_idx) > 0 else (1.0 - eer)

    gg_stats = compute_percentiles(gg_scores)
    gf_stats = compute_percentiles(gf_scores)
    gdw_stats = compute_percentiles(gdw_scores)

    mean_gg = gg_stats["mean"]
    mean_gf = gf_stats["mean"]
    mean_gdw = gdw_stats["mean"]
    delta_gg_gf = round(mean_gg - mean_gf, 4)

    y_pred = [1 if s >= eer_th else 0 for s in y_scores]
    acc = float(accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    return {
        "auc": round(auc, 4),
        "eer": round(eer, 4),
        "eer_th": round(eer_th, 4),
        "far_at_eer": round(far_at_eer, 4),
        "frr_at_eer": round(frr_at_eer, 4),
        "tar_at_eer": round(tar_at_eer, 4),
        "th_far5": round(th_far5, 4),
        "tar_far5": round(tar_far5, 4),
        "th_far1": round(th_far1, 4),
        "tar_far1": round(tar_far1, 4),
        "acc": round(acc, 4),
        "f1": round(float(f1), 4),
        "prec": round(float(prec), 4),
        "rec": round(float(rec), 4),
        "mean_gg": round(mean_gg, 4),
        "mean_gf": round(mean_gf, 4),
        "mean_gdw": round(mean_gdw, 4),
        "delta_gg_gf": delta_gg_gf,
        "gg_stats": gg_stats,
        "gf_stats": gf_stats,
        "gdw_stats": gdw_stats,
        "total_pairs": len(y_true),
        "gg_count": len(gg_scores),
        "gf_count": len(gf_scores),
        "gdw_count": len(gdw_scores),
        "y_true": y_true,
        "y_scores": y_scores,
        "pair_details": pair_details
    }


# ==============================================================================
# SPRINT 29 Dataset for Multi-Aspect-Ratio Fine-Tuning
# ==============================================================================

class Sprint29ForensicDataset(Dataset):
    def __init__(
        self,
        samples: List[ForensicSample],
        is_training: bool = True,
        mode: str = "EXP-F",
        domain_balanced: bool = False
    ):
        self.samples = samples
        self.is_training = is_training
        self.mode = mode
        self.domain_balanced = domain_balanced
        self.mar_augmenter = Sprint29MARAugmentation(is_training=is_training)

        unique_writers = sorted(list(set(s.writer_id for s in samples)))
        self.writer_to_label = {w: i for i, w in enumerate(unique_writers)}

        # Group by dataset for domain-balanced sampling if needed
        self.samples_by_dataset = {}
        for s in samples:
            ds = s.metadata.get("dataset", "UNKNOWN")
            self.samples_by_dataset.setdefault(ds, []).append(s)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_img(self, path: str) -> np.ndarray:
        try:
            data = Path(path).read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is not None:
                return img
        except Exception:
            pass
        return np.full((224, 224, 3), 255, dtype=np.uint8)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = self._load_img(s.image_path)

        if self.mode == "EXP-A":
            # Baseline pad-to-square
            if self.is_training:
                # Basic non-flip augmentation
                t = preprocess_exp_a_baseline(img, is_bgr=True).squeeze(0)
            else:
                t = preprocess_exp_a_baseline(img, is_bgr=True).squeeze(0)
            meta = {}
        elif self.mode == "EXP-C":
            t = preprocess_exp_c_content_crop(img, is_bgr=True).squeeze(0)
            meta = {}
        elif self.mode == "EXP-F":
            t, meta = self.mar_augmenter(img)
        else:
            t, meta = preprocess_exp_f_multi_aspect_ratio(img, is_bgr=True)
            t = t.squeeze(0)

        label = self.writer_to_label[s.writer_id]
        is_forg = s.is_forgery
        ds = s.metadata.get("dataset", "UNKNOWN")
        return t, label, is_forg, s.sample_id, ds


# ==============================================================================
# Model Fine-Tuning Routine (Two Stages)
# ==============================================================================

def train_candidate_model(
    base_model_path: Path,
    train_samples: List[ForensicSample],
    val_samples: List[ForensicSample],
    device: torch.device,
    mode: str = "EXP-F",
    candidate_name: str = "Candidate_D",
    epochs_stage1: int = 2,
    epochs_stage2: int = 3,
    domain_balanced: bool = False,
    seed: int = 42
) -> nn.Module:
    set_seed(seed)
    print(f"\n" + "=" * 80)
    print(f"  TRAINING [{candidate_name}] (Mode: {mode}, Domain-Balanced: {domain_balanced})")
    print("=" * 80)

    # 1. Warm start from V6 checkpoint (PRETRAINED=FALSE SCRATCH PROHIBITED)
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False).to(device)
    ckpt = torch.load(base_model_path, map_location=device)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    print(f"  [+] Warm-started from: {base_model_path.name}")

    # Exclude IAM words from signature/handwriting fine-tuning (Rule 22)
    train_pool = [s for s in train_samples if s.metadata.get("dataset") != "IAM_HW"]
    print(f"  [+] Training Pool: {len(train_pool)} samples across CEDAR, BHSig, UTSig, CVL.")

    # Datasets and DataLoader
    train_ds = Sprint29ForensicDataset(train_pool, is_training=True, mode=mode, domain_balanced=domain_balanced)
    train_loader = DataLoader(train_ds, batch_size=24, shuffle=True, num_workers=0)

    criterion = ForensicHybridLoss(temperature=0.06, margin=0.50, hard_neg_weight=2.5)

    # STAGE 1: Freeze Backbone, train projection head only
    print(f"\n  [*] Stage 1: Backbone Frozen (Head Training, {epochs_stage1} epochs, LR=1e-4)...")
    model.freeze_backbone()
    opt_head = torch.optim.AdamW(model.projection_head.parameters(), lr=1e-4, weight_decay=1e-4)

    for ep in range(1, epochs_stage1 + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for tensors, labels, is_forg, _, _ in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            is_forg = is_forg.to(device)
            opt_head.zero_grad()
            embs = model(tensors)
            loss, _, _ = criterion(embs, labels, is_forg)
            loss.backward()
            opt_head.step()
            total_loss += loss.item()
            batches += 1
            if batches >= 60:
                break
        print(f"      Stage 1 - Epoch {ep}/{epochs_stage1}: Mean Loss = {total_loss / max(1, batches):.4f}")

    # STAGE 2: Unfreeze Backbone (Differential LR)
    print(f"\n  [*] Stage 2: Backbone Unfrozen (Differential LR, {epochs_stage2} epochs)...")
    model.unfreeze_backbone()
    opt_all = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 2e-5},
        {"params": model.projection_head.parameters(), "lr": 1e-4}
    ], weight_decay=1e-4)

    # Cosine Annealing scheduler (min_lr = 1e-6)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt_all, T_max=epochs_stage2 * 70, eta_min=1e-6)

    best_val_eer = 999.0
    best_weights = copy.deepcopy(model.state_dict())

    # Pre-select validation pool for fast early stopping check
    val_sub = [s for s in val_samples if s.metadata.get("dataset") in ["CEDAR", "BHSig-260", "UTSig", "CVL Database"]][:500]

    for ep in range(1, epochs_stage2 + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for tensors, labels, is_forg, _, _ in train_loader:
            tensors, labels = tensors.to(device), labels.to(device)
            is_forg = is_forg.to(device)
            opt_all.zero_grad()
            embs = model(tensors)
            loss, _, _ = criterion(embs, labels, is_forg)
            loss.backward()
            opt_all.step()
            scheduler.step()
            total_loss += loss.item()
            batches += 1
            if batches >= 70:
                break

        # Validation check for early stopping
        model.eval()
        val_embs, _ = extract_embeddings_for_mode(model, val_sub, device, mode=mode)
        val_eval = evaluate_pairs_detailed(val_embs, val_sub)
        val_eer = val_eval["eer"]
        val_auc = val_eval["auc"]
        print(f"      Stage 2 - Epoch {ep}/{epochs_stage2}: Mean Loss = {total_loss / max(1, batches):.4f} | Val EER = {val_eer:.4f} | Val AUC = {val_auc:.4f}")

        if val_eer < best_val_eer:
            best_val_eer = val_eer
            best_weights = copy.deepcopy(model.state_dict())
            print(f"      -> New best model saved (EER = {best_val_eer:.4f})")

    model.load_state_dict(best_weights)
    model.eval()
    return model


print("FHDSS Sprint 29 Pipeline module loaded successfully.")
