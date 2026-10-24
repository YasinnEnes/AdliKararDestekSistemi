"""
FHDSS SPRINT 30.2 — Low-Similarity Root-Cause Audit & Inference Robustness Validation Runner
Executes Controlled Experiment Matrices A through G on CEDAR genuine samples.
Saves comprehensive results to reports/v6_s30_2/sprint30_2_experiment_results.json
and hashes to reports/v6_s30_2/sprint30_2_artifact_hashes.json.
"""

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.preprocessing.pipeline import ForensicImagePreprocessor
from cv.features.extractors import ForensicFeatureExtractor
from ml.models.comparison import ForensicComparisonEngine
from ml.models.fusion import HybridForensicFusion
from ml.models.siamese import ForensicONNXRuntimeEncoder, compute_normalized_similarity, compute_cosine_similarity
from ml.preprocessing.contract import preprocess_for_inference

REPORTS_DIR = PROJECT_ROOT / "reports" / "v6_s30_2"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def imread_unicode(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")
    data = path.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image at {path}")
    return img


def sha256_of_file(path: Path) -> str:
    if not path.exists():
        return "NOT_FOUND"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_ui_band(score_pct: float) -> Dict[str, str]:
    if score_pct >= 98.0:
        return {"band": "SUPERIMPOSITION_ANOMALY", "label": "Tetabuk / Dolandırıcılık Şüphesi"}
    elif score_pct >= 85.0:
        return {"band": "NATURAL_VARIATION_MATCH", "label": "Doğal Varyasyon (Orijinal)"}
    elif score_pct >= 70.0:
        return {"band": "SKILLED_SIMULATION_ZONE", "label": "Kritik Ayrım (Nitelikli Taklit Şüphesi)"}
    elif score_pct >= 50.0:
        return {"band": "INCONCLUSIVE_DISGUISE_ZONE", "label": "Kararsız / İstekli Bozma"}
    elif score_pct >= 25.0:
        return {"band": "LOW_SIMILARITY_NON_SOURCE", "label": "Düşük Benzerlik (Farklı Kaynak)"}
    else:
        return {"band": "COMPLETE_DISPARITY", "label": "Tam Uyumsuzluk"}


class ExperimentPipeline:
    def __init__(self):
        self.preprocessor = ForensicImagePreprocessor()
        self.feature_extractor = ForensicFeatureExtractor()
        self.comparison_engine = ForensicComparisonEngine()
        self.fusion = HybridForensicFusion()
        
        self.onnx_path = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5.onnx"
        self.encoder = ForensicONNXRuntimeEncoder(onnx_model_path=str(self.onnx_path))

    def evaluate_pair(
        self,
        q_img: np.ndarray,
        ref_imgs: List[np.ndarray],
        experiment_id: str,
        category: str,
        description: str,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        # 1. Preprocessing
        art_q = self.preprocessor.process(q_img)
        ref_arts = [self.preprocessor.process(r) for r in ref_imgs]

        # 2. Morphological features
        feat_q = self.feature_extractor.extract_all(art_q.skeleton, art_q.binarized)
        ref_feats = [self.feature_extractor.extract_all(r_art.skeleton, r_art.binarized) for r_art in ref_arts]

        # 3. Morphological comparison
        comp_report = self.comparison_engine.compare(feat_q, ref_feats)

        # 4. Deep embeddings
        doc_q = art_q.isolated_document if getattr(art_q, "isolated_document", None) is not None else q_img
        t_q = preprocess_for_inference(doc_q)
        e_q = self.encoder(t_q).numpy()[0]

        ref_sims = []
        ref_raw_cosines = []
        for idx, r_img in enumerate(ref_imgs):
            r_art = ref_arts[idx]
            doc_r = r_art.isolated_document if getattr(r_art, "isolated_document", None) is not None else r_img
            t_r = preprocess_for_inference(doc_r)
            e_r = self.encoder(t_r).numpy()[0]
            ref_sims.append(float(compute_normalized_similarity(e_q, e_r)))
            ref_raw_cosines.append(float(compute_cosine_similarity(e_q, e_r)))

        deep_sim = float(np.mean(ref_sims))
        raw_cos = float(np.mean(ref_raw_cosines))

        # 5. Hybrid Fusion (with gating)
        fusion_res = self.fusion.fuse(comp_report, deep_sim, raw_cosine=raw_cos)

        # 6. Gating Ablation (calculate pre-gating baseline score)
        s_morph = float(comp_report.similarity_score)
        pre_gating_score = round(0.45 * s_morph + 0.55 * deep_sim, 4)
        gating_delta = round(pre_gating_score - fusion_res.hybrid_similarity_score, 4)

        # 7. UI Percentage and Band
        final_pct = round(fusion_res.hybrid_similarity_score * 100.0, 1)
        ui_band = compute_ui_band(final_pct)

        # Extract top feature deviations
        sig_diffs_summary = [
            {
                "feature": d.feature_name_tr,
                "q_val": round(float(d.questioned_value), 3),
                "ref_mean": round(float(d.reference_mean), 3),
                "z_score": round(float(d.z_score), 2),
                "abs_diff": round(float(d.absolute_difference), 3),
            }
            for d in comp_report.significant_differences
        ]

        return {
            "experiment_id": experiment_id,
            "category": category,
            "description": description,
            "parameters": params,
            "layers": {
                "layer_1_input_shapes": {
                    "questioned_shape": list(q_img.shape),
                    "reference_shapes": [list(r.shape) for r in ref_imgs],
                },
                "layer_2_preprocessing": {
                    "q_isolated": art_q.metadata.get("document_isolation", {}).get("isolated", False),
                    "q_dark_border_ratio": art_q.metadata.get("document_isolation", {}).get("dark_border_ratio", 0.0),
                    "q_deskew_angle": art_q.metadata.get("deskew_angle_deg", 0.0),
                },
                "layer_3_morphological_features": {
                    "slant_deg": round(float(feat_q.structural.slant_angle_deg), 2),
                    "stroke_length_px": round(float(feat_q.structural.avg_stroke_length_px), 2),
                    "inter_stroke_spacing_px": round(float(feat_q.structural.inter_stroke_spacing_px), 2),
                    "baseline_dev_px": round(float(feat_q.structural.baseline_deviation_px), 2),
                    "loop_count": int(feat_q.structural.loop_count),
                    "crossing_count": int(feat_q.structural.crossing_count),
                    "terminal_count": int(feat_q.structural.terminal_count),
                    "tremor_index": round(float(feat_q.line_quality.tremor_index_psd), 4),
                },
                "layer_4_deep_metric": {
                    "raw_cosine": round(raw_cos, 4),
                    "normalized_similarity": round(deep_sim, 4),
                },
                "layer_5_mahalanobis": {
                    "mahalanobis_distance": round(float(comp_report.mahalanobis_distance), 4),
                    "morphological_similarity": round(s_morph, 4),
                    "significant_differences_count": len(comp_report.significant_differences),
                    "significant_differences": sig_diffs_summary,
                },
                "layer_6_pre_gating": {
                    "pre_gating_hybrid_score": pre_gating_score,
                },
                "layer_7_gating": {
                    "gating_triggered": fusion_res.gating_triggered,
                    "gating_penalty": round(float(fusion_res.gating_penalty), 4),
                    "gating_weight": round(float(fusion_res.gating_weight), 4),
                    "structural_contradictions": fusion_res.structural_contradictions,
                    "penalty_details": fusion_res.penalty_details,
                    "gating_score_delta": gating_delta,
                },
                "layer_8_final_decision": {
                    "final_hybrid_score": round(float(fusion_res.hybrid_similarity_score), 4),
                    "ui_percentage": final_pct,
                    "ui_band": ui_band["band"],
                    "ui_band_label": ui_band["label"],
                },
            },
        }


def main():
    print("=" * 80)
    print("  FHDSS SPRINT 30.2: LOW-SIMILARITY ROOT-CAUSE AUDIT EXPERIMENTS")
    print("=" * 80)

    # 1. Model Hashes Verification
    v5_pt = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5_best.pt"
    v5_onnx = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5.onnx"
    v5_manifest = PROJECT_ROOT / "artifacts" / "weights" / "model_manifest.json"

    pt_hash = sha256_of_file(v5_pt)
    onnx_hash = sha256_of_file(v5_onnx)
    manifest_hash = sha256_of_file(v5_manifest)

    expected_pt_hash = "0fe6963acaba48c26817202662907d1d2e52f57115d807d03497fdc33641938f"
    expected_onnx_hash = "ddb89311ad2681fb0e4f88e6ecfe3a7ef3850855d692bc91749a53a7fb52d066"
    expected_manifest_hash = "8ebd6f0c59cd4136474d1f1e770345190a50b84f40f04f425065aafbe221370d"

    print(f"[*] V5 PT Hash:       {pt_hash[:16]}... (Intact: {pt_hash == expected_pt_hash})")
    print(f"[*] V5 ONNX Hash:     {onnx_hash[:16]}... (Intact: {onnx_hash == expected_onnx_hash})")
    print(f"[*] V5 Manifest Hash: {manifest_hash[:16]}... (Intact: {manifest_hash == expected_manifest_hash})")

    artifact_hashes = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "v5_production_pt": {
            "path": str(v5_pt),
            "expected_sha256": expected_pt_hash,
            "actual_sha256": pt_hash,
            "verified": pt_hash == expected_pt_hash,
        },
        "v5_production_onnx": {
            "path": str(v5_onnx),
            "expected_sha256": expected_onnx_hash,
            "actual_sha256": onnx_hash,
            "verified": onnx_hash == expected_onnx_hash,
        },
        "v5_production_manifest": {
            "path": str(v5_manifest),
            "expected_sha256": expected_manifest_hash,
            "actual_sha256": manifest_hash,
            "verified": manifest_hash == expected_manifest_hash,
        },
    }

    with open(REPORTS_DIR / "sprint30_2_artifact_hashes.json", "w", encoding="utf-8") as f:
        json.dump(artifact_hashes, f, indent=2)

    # 2. Load Genuine & Forgery Samples from CEDAR Writer 1
    cedar_root = PROJECT_ROOT / "datasets" / "DS_Downloaded" / "CEDAR" / "CEDAR1" / "signatures"
    org_dir = cedar_root / "full_org"
    forg_dir = cedar_root / "full_forg"

    base_path = org_dir / "original_1_1.png"
    assert base_path.exists(), f"Missing base image: {base_path}"
    img_base = imread_unicode(base_path)
    h_base, w_base = img_base.shape[:2]

    pipeline = ExperimentPipeline()
    all_results = []

    # ==========================================================================
    # MATRIX A: Identity & Determinism
    # ==========================================================================
    print("\n[*] Running Matrix A: Identity & Determinism...")
    for run_idx in range(1, 4):
        res = pipeline.evaluate_pair(
            q_img=img_base,
            ref_imgs=[img_base],
            experiment_id=f"EXP_A_IDENTITY_RUN_{run_idx}",
            category="A_IDENTITY",
            description=f"Original 1_1 compared against itself (Run {run_idx})",
            params={"run": run_idx},
        )
        all_results.append(res)
        print(f"    Run {run_idx}: Deep={res['layers']['layer_4_deep_metric']['normalized_similarity']:.4f}, "
              f"Final={res['layers']['layer_8_final_decision']['final_hybrid_score']:.4f}")

    # ==========================================================================
    # MATRIX B: Scale & Resolution
    # ==========================================================================
    print("\n[*] Running Matrix B: Scale Variations (0.4x, 0.6x, 1.0x, 1.5x, 2.5x)...")
    scales = [0.4, 0.6, 1.0, 1.5, 2.5]
    for s in scales:
        new_w, new_h = max(10, int(w_base * s)), max(10, int(h_base * s))
        interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
        img_scaled = cv2.resize(img_base, (new_w, new_h), interpolation=interp)
        
        res = pipeline.evaluate_pair(
            q_img=img_scaled,
            ref_imgs=[img_base],
            experiment_id=f"EXP_B_SCALE_{s}X",
            category="B_SCALE",
            description=f"Original scaled to {s}x ({new_w}x{new_h}) compared with 1.0x reference",
            params={"scale": s, "interp": "AREA" if s < 1.0 else "CUBIC", "target_size": [new_w, new_h]},
        )
        all_results.append(res)
        print(f"    Scale {s}x: Deep={res['layers']['layer_4_deep_metric']['normalized_similarity']:.3f}, "
              f"Morph={res['layers']['layer_5_mahalanobis']['morphological_similarity']:.3f}, "
              f"Gating={res['layers']['layer_7_gating']['gating_triggered']}, "
              f"Final={res['layers']['layer_8_final_decision']['final_hybrid_score']:.3f} "
              f"({res['layers']['layer_8_final_decision']['ui_percentage']}%)")

    # ==========================================================================
    # MATRIX C: ROI & Whitespace
    # ==========================================================================
    print("\n[*] Running Matrix C: ROI & Canvas Whitespace...")
    # C1: Tight Crop (Foreground ink bounding box only)
    gray = cv2.cvtColor(img_base, cv2.COLOR_BGR2GRAY)
    _, bin_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(bin_otsu > 0)
    if len(xs) > 0 and len(ys) > 0:
        x1, y1, x2, y2 = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
        img_tight = img_base[y1:y2+1, x1:x2+1]
    else:
        img_tight = img_base.copy()

    # C2: Medium Padding (+60px all sides)
    img_medium_pad = cv2.copyMakeBorder(img_base, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    # C3: Full Document Simulation (1600x1200 white canvas)
    canvas_full = np.full((1600, 1200, 3), 255, dtype=np.uint8)
    canvas_full[900:900+h_base, 300:300+w_base] = img_base

    roi_cases = [
        ("TIGHT_CROP", img_tight, "Strict foreground ink crop"),
        ("ORIGINAL_CROP", img_base, "Original CEDAR standard crop"),
        ("MEDIUM_PAD", img_medium_pad, "Signature with +60px white padding"),
        ("FULL_A4_PAGE", canvas_full, "Signature inside full 1600x1200 white page canvas"),
    ]

    for c_id, img_c, c_desc in roi_cases:
        res = pipeline.evaluate_pair(
            q_img=img_c,
            ref_imgs=[img_base],
            experiment_id=f"EXP_C_ROI_{c_id}",
            category="C_ROI_WHITESPACE",
            description=f"{c_desc} compared against original crop",
            params={"case": c_id, "shape": list(img_c.shape)},
        )
        all_results.append(res)
        print(f"    ROI {c_id}: Deep={res['layers']['layer_4_deep_metric']['normalized_similarity']:.3f}, "
              f"Morph={res['layers']['layer_5_mahalanobis']['morphological_similarity']:.3f}, "
              f"Final={res['layers']['layer_8_final_decision']['final_hybrid_score']:.3f} "
              f"({res['layers']['layer_8_final_decision']['ui_percentage']}%)")

    # ==========================================================================
    # MATRIX D: Rotation & Skew
    # ==========================================================================
    print("\n[*] Running Matrix D: Rotation Angle Variations (0°, ±3°, ±6°, ±12°)...")
    angles = [0.0, 3.0, -3.0, 6.0, -6.0, 12.0, -12.0]
    for ang in angles:
        # Rotation with canvas enlargement to prevent clipping
        M = cv2.getRotationMatrix2D((w_base / 2, h_base / 2), ang, 1.0)
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_w = int((h_base * sin_a) + (w_base * cos_a))
        new_h = int((h_base * cos_a) + (w_base * sin_a))
        M[0, 2] += (new_w / 2) - (w_base / 2)
        M[1, 2] += (new_h / 2) - (h_base / 2)
        img_rotated = cv2.warpAffine(img_base, M, (new_w, new_h), borderValue=(255, 255, 255))

        res = pipeline.evaluate_pair(
            q_img=img_rotated,
            ref_imgs=[img_base],
            experiment_id=f"EXP_D_ROT_{int(ang)}DEG",
            category="D_ROTATION",
            description=f"Original rotated by {ang}° (unclipped canvas)",
            params={"angle_deg": ang, "rotated_shape": list(img_rotated.shape)},
        )
        all_results.append(res)
        print(f"    Rot {ang:+.0f}°: Slant={res['layers']['layer_3_morphological_features']['slant_deg']}°, "
              f"Deep={res['layers']['layer_4_deep_metric']['normalized_similarity']:.3f}, "
              f"Gating={res['layers']['layer_7_gating']['gating_triggered']}, "
              f"Final={res['layers']['layer_8_final_decision']['final_hybrid_score']:.3f}")

    # ==========================================================================
    # MATRIX E: Stroke Thickness & Line Quality Artifacts
    # ==========================================================================
    print("\n[*] Running Matrix E: Stroke Thickness & Line Quality...")
    # E1: Dilation (Thick pen / bleed)
    kernel_3 = np.ones((3, 3), np.uint8)
    img_dilated = cv2.erode(img_base, kernel_3, iterations=1) # Erode on BGR (black ink darkens/expands)

    # E2: Erosion (Faint pen / thin stroke)
    img_eroded = cv2.dilate(img_base, kernel_3, iterations=1) # Dilate on BGR (white expands, ink thins)

    # E3: Gaussian Noise (Scanner noise)
    noise = np.random.normal(0, 15, img_base.shape).astype(np.float32)
    img_noisy = np.clip(img_base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # E4: Low Contrast (Faint scan)
    img_low_contrast = (img_base.astype(np.float32) * 0.4 + 150).clip(0, 255).astype(np.uint8)

    # E5: Paper Shadow / Non-uniform Lighting (Gradient)
    grad = np.linspace(0.65, 1.0, w_base, dtype=np.float32)
    grad_2d = np.tile(grad, (h_base, 1))
    img_shadow = (img_base.astype(np.float32) * np.expand_dims(grad_2d, axis=-1)).clip(0, 255).astype(np.uint8)

    quality_cases = [
        ("THICK_STROKE_DILATE", img_dilated, "Thick marker stroke / ink bleeding"),
        ("THIN_STROKE_ERODE", img_eroded, "Thin / faint pen stroke"),
        ("SCANNER_NOISE", img_noisy, "Gaussian scanner sensor noise (sigma=15)"),
        ("LOW_CONTRAST", img_low_contrast, "Faint scan low contrast"),
        ("PAPER_SHADOW", img_shadow, "Non-uniform paper lighting gradient"),
    ]

    for q_id, img_q, q_desc in quality_cases:
        res = pipeline.evaluate_pair(
            q_img=img_q,
            ref_imgs=[img_base],
            experiment_id=f"EXP_E_QUALITY_{q_id}",
            category="E_LINE_QUALITY",
            description=f"{q_desc} compared against clean original",
            params={"case": q_id},
        )
        all_results.append(res)
        print(f"    Quality {q_id}: Deep={res['layers']['layer_4_deep_metric']['normalized_similarity']:.3f}, "
              f"Morph={res['layers']['layer_5_mahalanobis']['morphological_similarity']:.3f}, "
              f"Terminals={res['layers']['layer_3_morphological_features']['terminal_count']}, "
              f"Final={res['layers']['layer_8_final_decision']['final_hybrid_score']:.3f}")

    # ==========================================================================
    # MATRIX F: Reference Count Multiplicity (N = 1, 2, 3, 5)
    # ==========================================================================
    print("\n[*] Running Matrix F: Reference Count Multiplicity...")
    # Load Writer 1 genuine samples 2 to 6
    refs_pool = [imread_unicode(org_dir / f"original_1_{i}.png") for i in range(2, 7)]
    forg_sample = imread_unicode(forg_dir / "forgeries_1_1.png")

    ref_configs = [
        ("N1_SINGLE", [refs_pool[0]], "1 genuine reference sample (Single Reference)"),
        ("N2_DUAL", refs_pool[:2], "2 genuine reference samples"),
        ("N3_TRIAD", refs_pool[:3], "3 genuine reference samples (Forensic Minimum Recommended)"),
        ("N5_MULTI", refs_pool[:5], "5 genuine reference samples (Forensic Standard Recommended)"),
    ]

    for r_id, r_list, r_desc in ref_configs:
        # F1. Genuine-Genuine Comparison
        res_gg = pipeline.evaluate_pair(
            q_img=img_base,
            ref_imgs=r_list,
            experiment_id=f"EXP_F_REF_{r_id}_GENUINE",
            category="F_REFERENCE_COUNT",
            description=f"Questioned Genuine 1_1 vs {r_desc}",
            params={"ref_count": len(r_list), "pair_type": "GENUINE_GENUINE"},
        )
        all_results.append(res_gg)

        # F2. Genuine-Skilled Forgery Comparison
        res_gf = pipeline.evaluate_pair(
            q_img=forg_sample,
            ref_imgs=r_list,
            experiment_id=f"EXP_F_REF_{r_id}_FORGERY",
            category="F_REFERENCE_COUNT",
            description=f"Questioned Skilled Forgery 1_1 vs {r_desc}",
            params={"ref_count": len(r_list), "pair_type": "SKILLED_FORGERY"},
        )
        all_results.append(res_gf)

        print(f"    {r_id} (N={len(r_list)}): "
              f"GG Final={res_gg['layers']['layer_8_final_decision']['final_hybrid_score']:.3f} (D_M={res_gg['layers']['layer_5_mahalanobis']['mahalanobis_distance']:.2f}) | "
              f"GF Final={res_gf['layers']['layer_8_final_decision']['final_hybrid_score']:.3f} (D_M={res_gf['layers']['layer_5_mahalanobis']['mahalanobis_distance']:.2f})")

    # ==========================================================================
    # MATRIX G: Gating Ablation Summary
    # ==========================================================================
    print("\n[*] Compiling Matrix G: Gating Ablation Impact across all experiments...")
    gating_triggered_count = sum(1 for r in all_results if r["layers"]["layer_7_gating"]["gating_triggered"])
    max_suppression = max(r["layers"]["layer_7_gating"]["gating_score_delta"] for r in all_results)
    avg_suppression = np.mean([r["layers"]["layer_7_gating"]["gating_score_delta"] for r in all_results if r["layers"]["layer_7_gating"]["gating_triggered"]])

    print(f"    Total Experiments: {len(all_results)}")
    print(f"    Gating Triggered:  {gating_triggered_count}/{len(all_results)} cases")
    print(f"    Max Score Suppression: {max_suppression:.4f}")
    print(f"    Avg Suppression (when active): {avg_suppression:.4f}")

    # Save to JSON
    output_json = {
        "metadata": {
            "audit_sprint": "SPRINT_30_2",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_tested": "convnext_tiny_forensic_multidataset_v5.onnx",
            "model_sha256": onnx_hash,
            "total_experiments": len(all_results),
            "gating_triggered_count": gating_triggered_count,
            "max_score_suppression_delta": round(float(max_suppression), 4),
            "avg_score_suppression_delta": round(float(avg_suppression), 4),
        },
        "experiments": all_results,
    }

    results_file = REPORTS_DIR / "sprint30_2_experiment_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(output_json, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Successfully saved {len(all_results)} experiment results to {results_file.name}")


if __name__ == "__main__":
    main()
