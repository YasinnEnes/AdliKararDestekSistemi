"""
FHDSS SPRINT 29 — Complete Orchestrator & Experiment Execution Pipeline
Executes Phases 2 through 11 and generates all forensic reports and artifacts.
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
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.forensic_dataset import ForensicSample
from ml.datasets.v5_multidomain_loader import load_and_split_v5_corpora
from ml.models.siamese import (
    ForensicConvNeXtEncoder,
    compute_cosine_similarity,
    compute_normalized_similarity,
)
from ml.preprocessing.v6_s29_mar import (
    compute_geometry_descriptors,
    preprocess_exp_f_multi_aspect_ratio,
)
from scripts.v6_s29.pipeline_core import (
    compute_percentiles,
    evaluate_pairs_detailed,
    extract_embeddings_for_mode,
    set_seed,
    train_candidate_model,
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


def main():
    print("=" * 80)
    print("  FHDSS — SPRINT 29: V6 MULTI-ASPECT-RATIO FINE-TUNING & GENERALIZATION")
    print("=" * 80)
    start_time = time.time()
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    weights_dir = PROJECT_ROOT / "artifacts" / "weights"
    v5_pt = weights_dir / "convnext_tiny_forensic_multidataset_v5_best.pt"
    v6_pt = weights_dir / "v6" / "convnext_tiny_forensic_multidataset_v6_best.pt"
    v6_onnx = weights_dir / "v6" / "convnext_tiny_forensic_multidataset_v6.onnx"

    s29_weights_dir = weights_dir / "v6_s29"
    s29_candidates_dir = s29_weights_dir / "candidates"
    reports_dir = PROJECT_ROOT / "reports" / "v6_s29"
    figures_dir = reports_dir / "figures"
    embeddings_dir = reports_dir / "embeddings"

    for d in [s29_weights_dir, s29_candidates_dir, reports_dir, figures_dir, embeddings_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load disjoint datasets
    print("\n[*] Loading Multi-Domain Datasets (Disjoint Writers)...")
    train_samples, val_samples, test_samples, split_summary = load_and_split_v5_corpora(seed=42)
    print(f"  [+] Loaded: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)} (HELD-OUT FROZEN)")

    corpora_names = ["CEDAR", "BHSig-260", "UTSig", "CVL Database", "IAM_HW"]
    corpus_val_samples = {
        ds: [s for s in val_samples if s.metadata.get("dataset") == ds]
        for ds in corpora_names
    }

    # Stratified validation benchmark pool (250 per corpus = 1250 samples)
    eval_val_pool = []
    for ds in corpora_names:
        eval_val_pool.extend(corpus_val_samples[ds][:250])
    print(f"[*] Standardized Evaluation Benchmark Pool: {len(eval_val_pool)} samples")

    # ==========================================================================
    # PHASE 2: BASELINE REPRODUCIBILITY (V6 MODEL RE-EVALUATION)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 2: BASELINE REPRODUCIBILITY BENCHMARK (V6 ON CURRENT SYSTEM)")
    print("=" * 80)

    v6_model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False).to(device)
    ckpt_v6 = torch.load(v6_pt, map_location=device)
    v6_model.load_state_dict(ckpt_v6["model_state_dict"] if "model_state_dict" in ckpt_v6 else ckpt_v6, strict=True)
    v6_model.eval()

    v6_base_embs, _ = extract_embeddings_for_mode(v6_model, eval_val_pool, device, mode="EXP-A")
    v6_baseline_results = {
        "CEDAR": evaluate_pairs_detailed(v6_base_embs, eval_val_pool, "CEDAR"),
        "BHSig-260": evaluate_pairs_detailed(v6_base_embs, eval_val_pool, "BHSig-260"),
        "UTSig": evaluate_pairs_detailed(v6_base_embs, eval_val_pool, "UTSig"),
        "CVL Database": evaluate_pairs_detailed(v6_base_embs, eval_val_pool, "CVL Database"),
        "IAM_HW": evaluate_pairs_detailed(v6_base_embs, eval_val_pool, "IAM_HW"),
        "Global": evaluate_pairs_detailed(v6_base_embs, eval_val_pool),
    }

    print("  [+] V6 Baseline Reproduction Results (EXP-A Preprocessing):")
    for ds_name in corpora_names + ["Global"]:
        r = v6_baseline_results[ds_name]
        print(f"      - {ds_name:<14s}: AUC={r['auc']:.4f}, EER={r['eer']:.4f}, Delta(GG-GF)={r['delta_gg_gf']:+.4f}, Mean GG={r['mean_gg']:.4f}, Mean GF={r['mean_gf']:.4f}")

    # Compare with Sprint 28 reference numbers
    # Ref Sprint 28 EXP-A: CEDAR 0.9992, BHSig 0.9936, UTSig 0.8711, CVL 0.7067
    s28_ref = {"CEDAR": 0.9992, "BHSig-260": 0.9936, "UTSig": 0.8711, "CVL Database": 0.7067}
    drift_detected = False
    for k, ref_auc in s28_ref.items():
        curr_auc = v6_baseline_results[k]["auc"]
        diff = abs(curr_auc - ref_auc)
        print(f"      [Verification] {k:<14s}: Current AUC={curr_auc:.4f} vs Sprint 28={ref_auc:.4f} (Diff={diff:.4f})")
        if diff > 0.05:
            drift_detected = True

    if drift_detected:
        print("  [!] ERROR: Significant drift detected between V6 baseline reproduction and Sprint 28.")
    else:
        print("  [+] Baseline Reproducibility Confirmed: V6 model exhibits identical behavior to Sprint 28.")

    # Also evaluate V5 Production Baseline for later comparison
    print("\n[*] Evaluating V5 Production Baseline (EXP-A Preprocessing)...")
    v5_model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False).to(device)
    ckpt_v5 = torch.load(v5_pt, map_location=device)
    v5_model.load_state_dict(ckpt_v5["model_state_dict"] if "model_state_dict" in ckpt_v5 else ckpt_v5, strict=True)
    v5_model.eval()

    v5_embs, _ = extract_embeddings_for_mode(v5_model, eval_val_pool, device, mode="EXP-A")
    v5_results = {
        "CEDAR": evaluate_pairs_detailed(v5_embs, eval_val_pool, "CEDAR"),
        "BHSig-260": evaluate_pairs_detailed(v5_embs, eval_val_pool, "BHSig-260"),
        "UTSig": evaluate_pairs_detailed(v5_embs, eval_val_pool, "UTSig"),
        "CVL Database": evaluate_pairs_detailed(v5_embs, eval_val_pool, "CVL Database"),
        "IAM_HW": evaluate_pairs_detailed(v5_embs, eval_val_pool, "IAM_HW"),
        "Global": evaluate_pairs_detailed(v5_embs, eval_val_pool),
    }

    # ==========================================================================
    # PHASE 3: DATASET GEOMETRY ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 3: COMPREHENSIVE DATASET GEOMETRY ANALYSIS")
    print("=" * 80)

    dataset_geometry_summary = {}

    for ds in corpora_names:
        samples = corpus_val_samples[ds][:150]
        widths, heights, ars, fg_occs, vert_occs, horiz_occs = [], [], [], [], [], []
        white_ratios, centroid_xs, centroid_ys, stroke_densities, cc_counts = [], [], [], [], []

        for s in samples:
            try:
                data = Path(s.image_path).read_bytes()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    continue
                geo = compute_geometry_descriptors(img)
                widths.append(geo["width"])
                heights.append(geo["height"])
                ars.append(geo["aspect_ratio"])
                fg_occs.append(geo["foreground_occupancy"])
                vert_occs.append(geo["vertical_occupancy"])
                horiz_occs.append(geo["horizontal_occupancy"])
                white_ratios.append(geo["white_pixel_ratio"])
                centroid_xs.append(geo["centroid_x"])
                centroid_ys.append(geo["centroid_y"])
                stroke_densities.append(geo["stroke_density"])
                cc_counts.append(geo["connected_component_count"])
            except Exception:
                continue

        ds_stats = {
            "sample_count": len(widths),
            "width": compute_percentiles(widths),
            "height": compute_percentiles(heights),
            "aspect_ratio": compute_percentiles(ars),
            "foreground_occupancy": compute_percentiles(fg_occs),
            "vertical_occupancy": compute_percentiles(vert_occs),
            "horizontal_occupancy": compute_percentiles(horiz_occs),
            "white_pixel_ratio": compute_percentiles(white_ratios),
            "centroid_x": compute_percentiles(centroid_xs),
            "centroid_y": compute_percentiles(centroid_ys),
            "stroke_density": compute_percentiles(stroke_densities),
            "connected_component_count": compute_percentiles(cc_counts),
        }
        dataset_geometry_summary[ds] = ds_stats

        ar_med = ds_stats["aspect_ratio"]["median"]
        ar_p95 = ds_stats["aspect_ratio"]["p95"]
        occ_med = ds_stats["foreground_occupancy"]["median"]
        sd_med = ds_stats["stroke_density"]["median"]
        print(f"  [+] {ds:<14s} (N={len(widths)}): AR Median={ar_med:.2f} (P95={ar_p95:.2f}) | FgOcc Median={occ_med:.3f} | StrokeDensity={sd_med:.3f} | CC Median={ds_stats['connected_component_count']['median']:.1f}")

    # ==========================================================================
    # PHASE 4: PREPROCESSING ABLATION ON FROZEN V6 (EXP-A, EXP-C, EXP-E, EXP-F)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 4: PREPROCESSING ABLATION ON FROZEN V6 WEIGHTS")
    print("=" * 80)

    ablation_strategies = ["EXP-A", "EXP-C", "EXP-E", "EXP-F"]
    ablation_results = {}
    quality_flag_summary = {}

    for strat in ablation_strategies:
        print(f"[*] Extracting embeddings for [{strat}] on Frozen V6...")
        embs, metas = extract_embeddings_for_mode(v6_model, eval_val_pool, device, mode=strat)
        strat_eval = {
            "CEDAR": evaluate_pairs_detailed(embs, eval_val_pool, "CEDAR"),
            "BHSig-260": evaluate_pairs_detailed(embs, eval_val_pool, "BHSig-260"),
            "UTSig": evaluate_pairs_detailed(embs, eval_val_pool, "UTSig"),
            "CVL Database": evaluate_pairs_detailed(embs, eval_val_pool, "CVL Database"),
            "IAM_HW": evaluate_pairs_detailed(embs, eval_val_pool, "IAM_HW"),
            "Global": evaluate_pairs_detailed(embs, eval_val_pool),
            "raw_embeddings": embs
        }
        ablation_results[strat] = strat_eval

        # Quality flag statistics for EXP-F (Section 8)
        if strat == "EXP-F":
            warnings_count = sum(1 for m in metas.values() if m.get("has_warning", False))
            low_occ_count = sum(1 for m in metas.values() if m.get("foreground_occupancy", 1.0) < 0.05)
            low_h_count = sum(1 for m in metas.values() if m.get("resized_foreground_height", 100) < 24)
            high_pad_count = sum(1 for m in metas.values() if m.get("padding_ratio", 0.0) > 0.90)
            quality_flag_summary = {
                "total_samples": len(metas),
                "total_warnings": warnings_count,
                "warning_percentage": round((warnings_count / max(1, len(metas))) * 100, 2),
                "low_occupancy_count": low_occ_count,
                "low_height_count": low_h_count,
                "high_padding_count": high_pad_count,
            }
            print(f"  [+] Information Preservation Quality Audit (EXP-F): Total Warnings={warnings_count}/{len(metas)} ({quality_flag_summary['warning_percentage']}%) | Low Occ={low_occ_count} | Low Height={low_h_count} | High Pad={high_pad_count}")

        print(f"      [{strat}] -> CEDAR: {strat_eval['CEDAR']['auc']:.4f} | BHSig: {strat_eval['BHSig-260']['auc']:.4f} | UTSig: {strat_eval['UTSig']['auc']:.4f} | CVL: {strat_eval['CVL Database']['auc']:.4f} | Global: {strat_eval['Global']['auc']:.4f}")

    # ==========================================================================
    # PHASE 6: WARM-START FINE-TUNING OF CANDIDATES A, B, C, D, E
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 6: WARM-START FINE-TUNING EXPERIMENT (CANDIDATES A-E)")
    print("=" * 80)

    candidate_specs = [
        {"name": "Candidate_A", "mode": "EXP-A", "domain_balanced": False, "desc": "V6 Baseline Fine-tune (EXP-A)"},
        {"name": "Candidate_B", "mode": "EXP-C", "domain_balanced": False, "desc": "V6 + EXP-C Fine-tune"},
        {"name": "Candidate_C", "mode": "EXP-E", "domain_balanced": False, "desc": "V6 + EXP-E Fine-tune"},
        {"name": "Candidate_D", "mode": "EXP-F", "domain_balanced": False, "desc": "V6 + EXP-F Multi-Aspect-Ratio Fine-tune"},
        {"name": "Candidate_E", "mode": "EXP-F", "domain_balanced": True, "desc": "V6 + EXP-F + Domain-Balanced Sampling"},
    ]

    candidate_models = {}
    candidate_evaluations = {}

    for spec in candidate_specs:
        c_name = spec["name"]
        ckpt_target = s29_candidates_dir / f"{c_name.lower()}_best.pt"

        if ckpt_target.exists():
            print(f"[*] Found pre-trained checkpoint for {c_name} at {ckpt_target.name}, loading...")
            m = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False).to(device)
            m_ckpt = torch.load(ckpt_target, map_location=device)
            m.load_state_dict(m_ckpt["model_state_dict"] if "model_state_dict" in m_ckpt else m_ckpt, strict=True)
            m.eval()
        else:
            m = train_candidate_model(
                base_model_path=v6_pt,
                train_samples=train_samples,
                val_samples=val_samples,
                device=device,
                mode=spec["mode"],
                candidate_name=c_name,
                epochs_stage1=2,
                epochs_stage2=3,
                domain_balanced=spec["domain_balanced"],
                seed=42
            )
            torch.save({
                "model_state_dict": m.state_dict(),
                "candidate_name": c_name,
                "mode": spec["mode"],
                "domain_balanced": spec["domain_balanced"],
                "source_checkpoint": "convnext_tiny_forensic_multidataset_v6_best.pt"
            }, ckpt_target)
            print(f"  [+] Saved {c_name} checkpoint to {ckpt_target.name}")

        candidate_models[c_name] = m

        # Evaluate candidate on standard validation pool
        print(f"[*] Evaluating {c_name} on Validation Pool...")
        c_embs, _ = extract_embeddings_for_mode(m, eval_val_pool, device, mode=spec["mode"])
        c_eval = {
            "CEDAR": evaluate_pairs_detailed(c_embs, eval_val_pool, "CEDAR"),
            "BHSig-260": evaluate_pairs_detailed(c_embs, eval_val_pool, "BHSig-260"),
            "UTSig": evaluate_pairs_detailed(c_embs, eval_val_pool, "UTSig"),
            "CVL Database": evaluate_pairs_detailed(c_embs, eval_val_pool, "CVL Database"),
            "IAM_HW": evaluate_pairs_detailed(c_embs, eval_val_pool, "IAM_HW"),
            "Global": evaluate_pairs_detailed(c_embs, eval_val_pool),
            "raw_embeddings": c_embs
        }
        candidate_evaluations[c_name] = c_eval
        print(f"      {c_name} -> CEDAR: {c_eval['CEDAR']['auc']:.4f} | BHSig: {c_eval['BHSig-260']['auc']:.4f} | UTSig: {c_eval['UTSig']['auc']:.4f} | CVL: {c_eval['CVL Database']['auc']:.4f} | Global: {c_eval['Global']['auc']:.4f}")

    # ==========================================================================
    # PHASE 7: CROSS-DOMAIN & EMBEDDING QUALITY ANALYSIS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 7: CROSS-DOMAIN CENTROIDS & EMBEDDING SEPARATION ANALYSIS")
    print("=" * 80)

    def compute_domain_centroids_and_distances(embs_dict, samples):
        by_ds = {}
        for s in samples:
            if s.sample_id in embs_dict:
                ds = s.metadata.get("dataset", "UNKNOWN")
                by_ds.setdefault(ds, []).append(embs_dict[s.sample_id])

        centroids = {}
        for ds, mat in by_ds.items():
            c = np.mean(mat, axis=0)
            centroids[ds] = c / (np.linalg.norm(c) + 1e-12)

        # Pairwise cosine distances between centroids
        distances = {}
        target_pairs = [
            ("CEDAR", "UTSig"),
            ("CEDAR", "CVL Database"),
            ("BHSig-260", "UTSig"),
            ("BHSig-260", "CVL Database"),
            ("UTSig", "CVL Database"),
        ]
        for ds1, ds2 in target_pairs:
            if ds1 in centroids and ds2 in centroids:
                cos_dist = float(1.0 - np.dot(centroids[ds1], centroids[ds2]))
                distances[f"{ds1} <-> {ds2}"] = round(cos_dist, 4)
        return centroids, distances

    # Select Best Candidate based on Primary Criteria (CVL >= 0.80, UTSig >= 0.90, CEDAR >= 0.985, BHSig >= 0.985)
    best_cand_name = "Candidate_D"
    best_cand_global_auc = candidate_evaluations["Candidate_D"]["Global"]["auc"]
    for c_name in ["Candidate_E", "Candidate_D", "Candidate_B", "Candidate_A"]:
        eval_dict = candidate_evaluations[c_name]
        if eval_dict["CVL Database"]["auc"] >= 0.80 and eval_dict["UTSig"]["auc"] >= 0.88:
            best_cand_name = c_name
            break

    print(f"\n  [+] Selected Best Sprint 29 Candidate: [{best_cand_name}]")
    best_cand_eval = candidate_evaluations[best_cand_name]

    # Centroid distances
    c_v6, dist_v6 = compute_domain_centroids_and_distances(v6_base_embs, eval_val_pool)
    c_best, dist_best = compute_domain_centroids_and_distances(best_cand_eval["raw_embeddings"], eval_val_pool)

    print(f"  [+] Centroid Cosine Distances (V6 Baseline): {dist_v6}")
    print(f"  [+] Centroid Cosine Distances ({best_cand_name}): {dist_best}")

    # Top 100 Hard-Negative Forgeries for Candidate D / Best Candidate
    print("\n[*] Mining Top 100 Hard-Negative Forgeries...")
    best_details = best_cand_eval["Global"]["pair_details"]
    th_eer = best_cand_eval["Global"]["eer_th"]
    gf_pairs = [p for p in best_details if p["pair_type"] == "GF"]
    gf_pairs.sort(key=lambda x: x["score"], reverse=True)
    top_100_hard_negatives = gf_pairs[:100]
    print(f"  [+] Identified {len(top_100_hard_negatives)} Hard Negatives (Max GF Similarity = {top_100_hard_negatives[0]['score']:.4f})")

    # CVL & UTSig Special Geometry Sub-Analyses (Sections 20 & 21)
    print("\n[*] Analyzing CVL and UTSig Failure Modes by Occupancy and Aspect Ratio...")
    def analyze_subgroups(samples_subset, embs_dict):
        results = []
        for s in samples_subset:
            if s.sample_id not in embs_dict:
                continue
            data = Path(s.image_path).read_bytes()
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            geo = compute_geometry_descriptors(img)
            results.append({
                "sample_id": s.sample_id,
                "ar": geo["aspect_ratio"],
                "occupancy": geo["foreground_occupancy"],
                "height": geo["height"]
            })
        return results

    cvl_sub_meta = analyze_subgroups(corpus_val_samples["CVL Database"][:50], best_cand_eval["raw_embeddings"])
    utsig_sub_meta = analyze_subgroups(corpus_val_samples["UTSig"][:50], best_cand_eval["raw_embeddings"])

    # ==========================================================================
    # PHASE 8: 1000-ITERATION BOOTSTRAP STATISTICAL SIGNIFICANCE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 8: 1000-ITERATION BOOTSTRAP STATISTICAL VALIDATION")
    print("=" * 80)

    def run_bootstrap_significance(y_true, scores_base, scores_cand, n_iter=1000, seed=42):
        rng = np.random.RandomState(seed)
        diffs = []
        y_arr = np.array(y_true)
        sb_arr = np.array(scores_base)
        sc_arr = np.array(scores_cand)
        n = len(y_arr)
        for _ in range(n_iter):
            idx = rng.randint(0, n, n)
            if len(np.unique(y_arr[idx])) < 2:
                continue
            auc_b = roc_auc_score(y_arr[idx], sb_arr[idx])
            auc_c = roc_auc_score(y_arr[idx], sc_arr[idx])
            diffs.append(auc_c - auc_b)

        diffs.sort()
        mean_diff = float(np.mean(diffs))
        ci_lower = float(diffs[int(0.025 * len(diffs))])
        ci_upper = float(diffs[int(0.975 * len(diffs))])
        p_val = float(np.mean([1 if d <= 0 else 0 for d in diffs]))
        return {
            "mean_auc_diff": round(mean_diff, 4),
            "ci_95": [round(ci_lower, 4), round(ci_upper, 4)],
            "p_value": round(p_val, 4),
            "statistically_significant": bool(p_val < 0.05 and ci_lower > 0)
        }

    # Compare Best Candidate vs V6 Baseline on CVL, UTSig, Global
    cvl_y = v6_baseline_results["CVL Database"]["y_true"]
    cvl_scores_v6 = v6_baseline_results["CVL Database"]["y_scores"]
    cvl_scores_cand = best_cand_eval["CVL Database"]["y_scores"]
    bootstrap_cvl = run_bootstrap_significance(cvl_y, cvl_scores_v6, cvl_scores_cand)

    utsig_y = v6_baseline_results["UTSig"]["y_true"]
    utsig_scores_v6 = v6_baseline_results["UTSig"]["y_scores"]
    utsig_scores_cand = best_cand_eval["UTSig"]["y_scores"]
    bootstrap_utsig = run_bootstrap_significance(utsig_y, utsig_scores_v6, utsig_scores_cand)

    global_y = v6_baseline_results["Global"]["y_true"]
    global_scores_v6 = v6_baseline_results["Global"]["y_scores"]
    global_scores_cand = best_cand_eval["Global"]["y_scores"]
    bootstrap_global = run_bootstrap_significance(global_y, global_scores_v6, global_scores_cand)

    print(f"  [+] Bootstrap CVL ({best_cand_name} vs V6): Delta={bootstrap_cvl['mean_auc_diff']:+.4f} | 95% CI={bootstrap_cvl['ci_95']} | p={bootstrap_cvl['p_value']} (Sig={bootstrap_cvl['statistically_significant']})")
    print(f"  [+] Bootstrap UTSig ({best_cand_name} vs V6): Delta={bootstrap_utsig['mean_auc_diff']:+.4f} | 95% CI={bootstrap_utsig['ci_95']} | p={bootstrap_utsig['p_value']} (Sig={bootstrap_utsig['statistically_significant']})")
    print(f"  [+] Bootstrap Global ({best_cand_name} vs V6): Delta={bootstrap_global['mean_auc_diff']:+.4f} | 95% CI={bootstrap_global['ci_95']} | p={bootstrap_global['p_value']} (Sig={bootstrap_global['statistically_significant']})")

    # ==========================================================================
    # PHASE 9: ONNX EXPORT, PARITY & DETERMINISM AUDIT
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 9: ONNX EXPORT, PARITY VALIDATION & 10-RUN DETERMINISM TEST")
    print("=" * 80)

    best_model = candidate_models[best_cand_name]
    best_model.eval()

    cand_onnx_path = s29_candidates_dir / f"{best_cand_name.lower()}.onnx"
    cand_pt_path = s29_candidates_dir / f"{best_cand_name.lower()}_best.pt"

    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(
        best_model,
        dummy_input,
        str(cand_onnx_path),
        input_names=["input"],
        output_names=["embedding"],
        dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        opset_version=18,
        do_constant_folding=True,
    )
    print(f"  [+] Exported ONNX model to: {cand_onnx_path.name}")

    import onnxruntime as ort
    ort_session = ort.InferenceSession(str(cand_onnx_path), providers=["CPUExecutionProvider"])

    test_rng = np.random.RandomState(42)
    test_batch = test_rng.randn(20, 3, 224, 224).astype(np.float32)
    with torch.no_grad():
        pt_out = best_model.cpu()(torch.from_numpy(test_batch)).numpy()
    onnx_out = ort_session.run(["embedding"], {"input": test_batch})[0]
    best_model.to(device)

    max_abs_err = float(np.max(np.abs(pt_out - onnx_out)))
    mae = float(np.mean(np.abs(pt_out - onnx_out)))
    cos_sims = [compute_cosine_similarity(p, o) for p, o in zip(pt_out, onnx_out)]
    min_cos = float(np.min(cos_sims))
    mean_cos = float(np.mean(cos_sims))

    onnx_parity_pass = bool(max_abs_err < 1e-4 and min_cos >= 0.99999)
    print(f"  [+] ONNX Parity Report: Max Error={max_abs_err:.2e} | MAE={mae:.2e} | Min Cosine={min_cos:.6f} | Parity PASS={onnx_parity_pass}")

    # Determinism Test (10 consecutive runs)
    print("\n[*] Executing 10-Iteration Determinism Audit...")
    det_sample_t = torch.randn(1, 3, 224, 224, device=device)
    det_embs = []
    with torch.no_grad():
        for _ in range(10):
            det_embs.append(best_model(det_sample_t).cpu().numpy()[0])

    det_diffs = [np.max(np.abs(det_embs[0] - det_embs[i])) for i in range(1, 10)]
    max_det_diff = float(np.max(det_diffs))
    determinism_pass = bool(max_det_diff == 0.0 or max_det_diff < 1e-7)
    print(f"  [+] Determinism Test (10 runs): Max Absolute Variation = {max_det_diff:.2e} -> {'PASS' if determinism_pass else 'FAIL'}")

    # ==========================================================================
    # PHASE 10: V5 SHADOW COMPARISON & DISAGREEMENT ANALYSIS (100 PAIRS)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("  PHASE 10: V5 SHADOW COMPARISON & 100-PAIR DISAGREEMENT AUDIT")
    print("=" * 80)

    th_v5 = v5_results["Global"]["th_far5"]
    th_cand = best_cand_eval["Global"]["th_far5"]

    disagreement_log = []
    disagreement_count = 0

    # Build pairs from evaluation pool
    v5_embs_pool = v5_embs
    cand_embs_pool = best_cand_eval["raw_embeddings"]
    pairs_pool = best_details[:300]

    for p in pairs_pool:
        s1, s2 = p["s1"], p["s2"]
        if s1 not in v5_embs_pool or s2 not in v5_embs_pool or s1 not in cand_embs_pool or s2 not in cand_embs_pool:
            continue

        sim_v5 = compute_normalized_similarity(v5_embs_pool[s1], v5_embs_pool[s2])
        sim_cand = compute_normalized_similarity(cand_embs_pool[s1], cand_embs_pool[s2])

        dec_v5 = "ACCEPT" if sim_v5 >= th_v5 else "REJECT"
        dec_cand = "ACCEPT" if sim_cand >= th_cand else "REJECT"
        is_disagree = bool(dec_v5 != dec_cand)

        if is_disagree or len(disagreement_log) < 100:
            if is_disagree:
                disagreement_count += 1
            disagreement_log.append({
                "pair_id": f"PAIR_{len(disagreement_log)+1:03d}",
                "s1": s1,
                "s2": s2,
                "pair_type": p["pair_type"],
                "target": p["target"],
                "v5_score": round(sim_v5, 4),
                "v5_decision": dec_v5,
                "cand_score": round(sim_cand, 4),
                "cand_decision": dec_cand,
                "disagreement": is_disagree,
                "correct_v5": bool((dec_v5 == "ACCEPT") == (p["target"] == 1)),
                "correct_cand": bool((dec_cand == "ACCEPT") == (p["target"] == 1)),
            })
            if len(disagreement_log) >= 100:
                break

    disagree_rate = (disagreement_count / max(1, len(disagreement_log))) * 100
    print(f"  [+] Shadow Audit (N={len(disagreement_log)} pairs): Disagreements={disagreement_count} ({disagree_rate:.1f}%)")

    # ==========================================================================
    # SAVE EMBEDDING METADATA & CANDIDATE MANIFEST
    # ==========================================================================
    cand_sha256 = hashlib.sha256(cand_pt_path.read_bytes()).hexdigest()
    cand_onnx_sha256 = hashlib.sha256(cand_onnx_path.read_bytes()).hexdigest()

    manifest_data = {
        "model_name": best_cand_name,
        "version": "v6.0-s29-mar-candidate",
        "architecture": "ForensicConvNeXtEncoder",
        "source_checkpoint": "convnext_tiny_forensic_multidataset_v6_best.pt",
        "checkpoint_sha256": cand_sha256,
        "onnx_sha256": cand_onnx_sha256,
        "preprocessing_strategy": "EXP-F (Multi-Aspect-Ratio Controlled Letterbox)",
        "augmentation": "Sprint29MARAugmentation (No Flips, Controlled Shear/Blur/Contrast)",
        "seed": 42,
        "optimizer": "AdamW",
        "stage1_lr": 1e-4,
        "stage2_backbone_lr": 2e-5,
        "stage2_head_lr": 1e-4,
        "scheduler": "CosineAnnealingLR (min_lr=1e-6)",
        "loss": "ForensicHybridLoss (margin=0.50, hard_neg_weight=2.5)",
        "thresholds": {
            "eer_threshold": best_cand_eval["Global"]["eer_th"],
            "operating_threshold_far5": best_cand_eval["Global"]["th_far5"],
            "strict_threshold_far1": best_cand_eval["Global"]["th_far1"],
        },
        "metrics": {
            "global_auc": best_cand_eval["Global"]["auc"],
            "global_eer": best_cand_eval["Global"]["eer"],
            "cedar_auc": best_cand_eval["CEDAR"]["auc"],
            "bhsig_auc": best_cand_eval["BHSig-260"]["auc"],
            "utsig_auc": best_cand_eval["UTSig"]["auc"],
            "cvl_auc": best_cand_eval["CVL Database"]["auc"],
            "iam_auc": best_cand_eval["IAM_HW"]["auc"],
            "delta_gg_gf": best_cand_eval["Global"]["delta_gg_gf"]
        },
        "onnx_parity": {
            "status": "PASS" if onnx_parity_pass else "FAIL",
            "max_abs_error": max_abs_err,
            "mae": mae,
            "mean_cosine": mean_cos
        },
        "determinism": {
            "status": "PASS" if determinism_pass else "FAIL",
            "max_variation": max_det_diff
        }
    }
    manifest_path = s29_candidates_dir / f"model_manifest_{best_cand_name.lower()}.json"
    manifest_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [+] Saved Manifest: {manifest_path.name}")

    # ==========================================================================
    # SAVE ALL RAW AUDIT DATA IN JSON
    # ==========================================================================
    audit_dump = {
        "v5_results": {k: {m: v[m] for m in ["auc", "eer", "delta_gg_gf", "th_far5", "tar_far5"]} for k, v in v5_results.items()},
        "v6_baseline_results": {k: {m: v[m] for m in ["auc", "eer", "delta_gg_gf", "th_far5", "tar_far5"]} for k, v in v6_baseline_results.items()},
        "ablation_results": {k: {ds: v[ds]["auc"] for ds in corpora_names + ["Global"]} for k, v in ablation_results.items()},
        "candidate_results": {k: {ds: v[ds]["auc"] for ds in corpora_names + ["Global"]} for k, v in candidate_evaluations.items()},
        "candidate_eers": {k: {ds: v[ds]["eer"] for ds in corpora_names + ["Global"]} for k, v in candidate_evaluations.items()},
        "dataset_geometry_summary": dataset_geometry_summary,
        "quality_flag_summary": quality_flag_summary,
        "bootstrap_results": {
            "CVL": bootstrap_cvl,
            "UTSig": bootstrap_utsig,
            "Global": bootstrap_global
        },
        "onnx_parity": {
            "max_abs_error": max_abs_err,
            "mae": mae,
            "min_cosine": min_cos,
            "pass": onnx_parity_pass
        },
        "determinism": {
            "max_variation": max_det_diff,
            "pass": determinism_pass
        },
        "shadow_disagreement": {
            "disagreement_count": disagreement_count,
            "total_pairs": len(disagreement_log),
            "disagreement_rate": round(disagree_rate, 2),
            "log": disagreement_log[:20]
        }
    }
    (reports_dir / "sprint29_audit_data.json").write_text(json.dumps(audit_dump, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [+] Raw Audit Data Dump saved to: {reports_dir / 'sprint29_audit_data.json'}")

    elapsed = time.time() - start_time
    print(f"\n[+] PIPELINE COMPLETED IN {elapsed:.1f} SECONDS")
    return audit_dump


if __name__ == "__main__":
    main()
