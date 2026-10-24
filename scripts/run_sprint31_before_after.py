"""
FHDSS SPRINT 31 — End-to-End Before & After Verification Runner
Executes Test Matrices A through H comparing:
- Baseline (Sprint 30.2 flawed pipeline)
- Corrected (Sprint 31 robust pipeline)
Saves results to reports/v6_s31/sprint31_before_after_metrics.json.
"""
from pathlib import Path
import hashlib
import json
import math
import sys
import time
from typing import Any, Dict, List

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
from ml.models.siamese import (
    ForensicConvNeXtEncoder,
    ForensicONNXRuntimeEncoder,
    compute_cosine_similarity,
    compute_normalized_similarity,
)
from ml.preprocessing.contract import preprocess_for_inference

REPORTS_DIR = PROJECT_ROOT / "reports" / "v6_s31"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def imread_unicode(path: Path) -> np.ndarray:
    data = path.read_bytes()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to decode image at {path}")
    return img


class BeforeAfterEvaluator:
    def __init__(self):
        self.preprocessor = ForensicImagePreprocessor(deskew_enabled=False)
        self.extractor_v1 = ForensicFeatureExtractor(scale_invariant=False)
        self.extractor_v2 = ForensicFeatureExtractor(scale_invariant=True)
        self.comp_engine = ForensicComparisonEngine()
        self.fusion = HybridForensicFusion()
        
        onnx_path = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5.onnx"
        self.encoder = ForensicONNXRuntimeEncoder(onnx_model_path=str(onnx_path))

    def evaluate_condition(
        self,
        q_img: np.ndarray,
        ref_imgs: List[np.ndarray],
        experiment_id: str,
        category: str,
        description: str,
    ) -> Dict[str, Any]:
        """Aynı girdi çifti üzerinde hem eski (Pre-fix) hem yeni (Post-fix) pipeline'ı çalıştırır."""
        # 1. Preprocessing
        art_q = self.preprocessor.process(q_img)
        arts_ref = [self.preprocessor.process(r) for r in ref_imgs]

        # ==========================================
        # A. PRE-FIX PIPELINE (Sprint 30.2 Davranışı)
        # ==========================================
        # Eski ROI: isolated_document (tam sayfa)
        q_doc_old = art_q.isolated_document if getattr(art_q, "isolated_document", None) is not None else q_img
        t_q_old = preprocess_for_inference(q_doc_old)
        e_q_old = self.encoder(t_q_old).numpy()[0]

        ref_sims_old = []
        ref_cos_old = []
        for idx, r_img in enumerate(ref_imgs):
            r_art = arts_ref[idx]
            r_doc_old = r_art.isolated_document if getattr(r_art, "isolated_document", None) is not None else r_img
            t_r_old = preprocess_for_inference(r_doc_old)
            e_r_old = self.encoder(t_r_old).numpy()[0]
            ref_sims_old.append(float(compute_normalized_similarity(e_q_old, e_r_old)))
            ref_cos_old.append(float(compute_cosine_similarity(e_q_old, e_r_old)))

        deep_sim_old = float(np.mean(ref_sims_old))
        raw_cos_old = float(np.mean(ref_cos_old))

        # Eski Morfoloji: v1 Raw Pixels
        feat_q_old = self.extractor_v1.extract_all(art_q.skeleton, art_q.binarized)
        feats_r_old = [self.extractor_v1.extract_all(r.skeleton, r.binarized) for r in arts_ref]
        
        # Eski N=2 Kovaryans Simülasyonu (Rank-1 hatası)
        if len(ref_imgs) == 2:
            ref_mat_old = np.array([r.feature_vector for r in feats_r_old], dtype=np.float64)
            cov_bad = np.cov(ref_mat_old, rowvar=False)
            for i, k in enumerate(self.comp_engine.FEATURE_KEYS):
                cov_bad[i, i] = max(float(cov_bad[i, i]), float(self.comp_engine.MIN_PRIOR_STD[k] ** 2))
            cov_bad += 1e-4 * np.eye(13)
            diff_old = np.array(feat_q_old.feature_vector) - np.mean(ref_mat_old, axis=0)
            d_m_sq = float(diff_old.T @ np.linalg.pinv(cov_bad) @ diff_old)
            d_m_old = math.sqrt(max(0.0, d_m_sq / 13.0))
            s_morph_old = float(math.exp(-0.25 * d_m_old))
            # Eski ComparisonReport benzerliği
            comp_old = self.comp_engine.compare(feat_q_old, feats_r_old)
            comp_old.mahalanobis_distance = round(d_m_old, 3)
            comp_old.similarity_score = round(s_morph_old, 4)
        else:
            comp_old = self.comp_engine.compare(feat_q_old, feats_r_old)

        fus_old = self.fusion.fuse(comp_old, deep_sim_old, raw_cosine=raw_cos_old)
        score_old = fus_old.hybrid_similarity_score

        # ==========================================
        # B. POST-FIX PIPELINE (Sprint 31 Düzeltilmiş)
        # ==========================================
        # Yeni ROI: normalized_crop (imza ROI'si)
        def _get_roi(art, fallback):
            if art is not None:
                crop = getattr(art, "normalized_crop", None)
                bbox_meta = getattr(art, "metadata", {}).get("bounding_box_normalization", {})
                if crop is not None and crop.size > 0 and bbox_meta.get("has_content", True):
                    return crop
                doc = getattr(art, "isolated_document", None)
                if doc is not None and doc.size > 0:
                    return doc
            return fallback

        q_target_new = _get_roi(art_q, q_img)
        t_q_new = preprocess_for_inference(q_target_new)
        e_q_new = self.encoder(t_q_new).numpy()[0]

        ref_sims_new = []
        ref_cos_new = []
        for idx, r_img in enumerate(ref_imgs):
            r_art = arts_ref[idx]
            r_target_new = _get_roi(r_art, r_img)
            t_r_new = preprocess_for_inference(r_target_new)
            e_r_new = self.encoder(t_r_new).numpy()[0]
            ref_sims_new.append(float(compute_normalized_similarity(e_q_new, e_r_new)))
            ref_cos_new.append(float(compute_cosine_similarity(e_q_new, e_r_new)))

        deep_sim_new = float(np.mean(ref_sims_new))
        raw_cos_new = float(np.mean(ref_cos_new))

        # Yeni Morfoloji: v2 Scale-Invariant Ratios
        feat_q_new = self.extractor_v2.extract_all(art_q.skeleton, art_q.binarized)
        feats_r_new = [self.extractor_v2.extract_all(r.skeleton, r.binarized) for r in arts_ref]
        comp_new = self.comp_engine.compare(feat_q_new, feats_r_new)

        fus_new = self.fusion.fuse(comp_new, deep_sim_new, raw_cosine=raw_cos_new)
        score_new = fus_new.hybrid_similarity_score

        return {
            "experiment_id": experiment_id,
            "category": category,
            "description": description,
            "reference_count": len(ref_imgs),
            "before_fix": {
                "deep_similarity": round(deep_sim_old, 4),
                "raw_cosine": round(raw_cos_old, 4),
                "morphological_similarity": round(float(comp_old.similarity_score), 4),
                "mahalanobis_distance": round(float(comp_old.mahalanobis_distance), 4),
                "gating_triggered": bool(fus_old.gating_triggered),
                "gating_weight": round(float(fus_old.gating_weight), 4),
                "gating_penalty": round(float(fus_old.gating_penalty), 4),
                "final_score": round(score_old, 4),
                "ui_percentage": round(score_old * 100.0, 1),
                "concordance": fus_old.concordance_level,
            },
            "after_fix": {
                "deep_similarity": round(deep_sim_new, 4),
                "raw_cosine": round(raw_cos_new, 4),
                "morphological_similarity": round(float(comp_new.similarity_score), 4),
                "mahalanobis_distance": round(float(comp_new.mahalanobis_distance), 4),
                "covariance_method": comp_new.covariance_method,
                "condition_number": comp_new.condition_number,
                "confidence_warning": comp_new.confidence_warning,
                "gating_triggered": bool(fus_new.gating_triggered),
                "gating_weight": round(float(fus_new.gating_weight), 4),
                "gating_penalty": round(float(fus_new.gating_penalty), 4),
                "final_score": round(score_new, 4),
                "ui_percentage": round(score_new * 100.0, 1),
                "concordance": fus_new.concordance_level,
            },
            "score_improvement_delta": round(score_new - score_old, 4),
        }


def run_benchmark():
    evaluator = BeforeAfterEvaluator()
    cedar_root = PROJECT_ROOT / "datasets" / "DS_Downloaded" / "CEDAR" / "CEDAR1" / "signatures"
    org_dir = cedar_root / "full_org"
    forg_dir = cedar_root / "full_forg"

    # Load authentic CEDAR Writer 1 samples
    img_w1_1 = imread_unicode(org_dir / "original_1_1.png")
    img_w1_2 = imread_unicode(org_dir / "original_1_2.png")
    img_w1_3 = imread_unicode(org_dir / "original_1_3.png")
    img_w1_4 = imread_unicode(org_dir / "original_1_4.png")
    img_w1_5 = imread_unicode(org_dir / "original_1_5.png")

    # Load skilled forgeries
    img_forg_1 = imread_unicode(forg_dir / "forgeries_1_1.png")
    img_forg_2 = imread_unicode(forg_dir / "forgeries_1_2.png")

    # Load different writer (Writer 2)
    img_w2_1 = imread_unicode(org_dir / "original_2_1.png")

    experiments = []

    # Category A: Identity
    print("Running Category A: Identity...")
    experiments.append(evaluator.evaluate_condition(
        img_w1_1, [img_w1_1],
        "EXP_A_IDENTITY", "A_IDENTITY", "Original 1_1 compared against itself"
    ))

    # Category B: Scale
    print("Running Category B: Scale variations...")
    h, w = img_w1_1.shape[:2]
    for scale in [0.4, 0.6, 1.5, 2.5]:
        scaled_q = cv2.resize(img_w1_1, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC)
        experiments.append(evaluator.evaluate_condition(
            scaled_q, [img_w1_1],
            f"EXP_B_SCALE_{scale}X", "B_SCALE", f"Original 1_1 scaled to {scale}x vs 1.0x"
        ))

    # Category C: ROI / Whitespace
    print("Running Category C: ROI and Whitespace...")
    page_wide = np.full((1400, 1000, 3), 255, dtype=np.uint8)
    page_wide[700:700+h, 300:300+w] = img_w1_1
    experiments.append(evaluator.evaluate_condition(
        page_wide, [img_w1_1],
        "EXP_C_WIDE_PAGE", "C_ROI_WHITESPACE", "Original 1_1 on large A4 white page vs tight crop"
    ))

    # Category D: Rotation & Thickness
    print("Running Category D: Rotation and Thickness...")
    for angle in [-12.0, -3.0, 3.0, 12.0]:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot_q = cv2.warpAffine(img_w1_1, M, (w, h), borderValue=(255, 255, 255))
        experiments.append(evaluator.evaluate_condition(
            rot_q, [img_w1_1],
            f"EXP_D_ROT_{angle}DEG", "D_ROTATION", f"Original 1_1 rotated by {angle} deg"
        ))
    kernel = np.ones((3, 3), np.uint8)
    dilated_q = cv2.erode(img_w1_1, kernel, iterations=1)  # ink becomes thicker
    experiments.append(evaluator.evaluate_condition(
        dilated_q, [img_w1_1],
        "EXP_D_THICKNESS", "D_THICKNESS", "Original 1_1 with thickened stroke line"
    ))

    # Category E: Image Quality
    print("Running Category E: Quality and Noise...")
    noise = np.random.RandomState(42).normal(0, 15, img_w1_1.shape).astype(np.float32)
    noisy_q = np.clip(img_w1_1.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    experiments.append(evaluator.evaluate_condition(
        noisy_q, [img_w1_1],
        "EXP_E_NOISE_STD15", "E_QUALITY", "Original 1_1 with Gaussian scanner noise sigma=15"
    ))

    # Category F: Reference Count
    print("Running Category F: Reference Counts...")
    experiments.append(evaluator.evaluate_condition(
        img_w1_1, [img_w1_2],
        "EXP_F_REF_N1", "F_REFERENCE_COUNT", "Genuine Writer 1 with N=1 reference (Sample 2)"
    ))
    experiments.append(evaluator.evaluate_condition(
        img_w1_1, [img_w1_2, img_w1_3],
        "EXP_F_REF_N2", "F_REFERENCE_COUNT", "Genuine Writer 1 with N=2 references (Samples 2, 3)"
    ))
    experiments.append(evaluator.evaluate_condition(
        img_w1_1, [img_w1_2, img_w1_3, img_w1_4],
        "EXP_F_REF_N3", "F_REFERENCE_COUNT", "Genuine Writer 1 with N=3 references (Samples 2, 3, 4)"
    ))
    experiments.append(evaluator.evaluate_condition(
        img_w1_1, [img_w1_2, img_w1_3, img_w1_4, img_w1_5],
        "EXP_F_REF_N5", "F_REFERENCE_COUNT", "Genuine Writer 1 with N=5 references (Samples 2, 3, 4, 5)"
    ))

    # Category G: Skilled Forgeries & Different Writer
    print("Running Category G: Discrimination...")
    experiments.append(evaluator.evaluate_condition(
        img_forg_1, [img_w1_1, img_w1_2, img_w1_3],
        "EXP_G_FORGERY_1", "G_FORGERY", "Skilled Forgery 1 vs Genuine Writer 1 (N=3)"
    ))
    experiments.append(evaluator.evaluate_condition(
        img_forg_2, [img_w1_1, img_w1_2, img_w1_3],
        "EXP_G_FORGERY_2", "G_FORGERY", "Skilled Forgery 2 vs Genuine Writer 1 (N=3)"
    ))
    experiments.append(evaluator.evaluate_condition(
        img_w2_1, [img_w1_1, img_w1_2, img_w1_3],
        "EXP_G_DIFFERENT_WRITER", "G_DIFFERENT_WRITER", "Different Writer (Writer 2 vs Writer 1, N=3)"
    ))

    # Category H: Blank & Corrupt Input
    print("Running Category H: Robustness & Fallback...")
    blank_page = np.full((500, 500, 3), 255, dtype=np.uint8)
    experiments.append(evaluator.evaluate_condition(
        blank_page, [img_w1_1],
        "EXP_H_BLANK_ROI", "H_ROBUSTNESS", "Blank white page with no signature content vs Genuine"
    ))

    # Summary Metrics
    total = len(experiments)
    deltas = [e["score_improvement_delta"] for e in experiments]
    avg_delta = float(np.mean(deltas))

    out_bundle = {
        "sprint": "SPRINT_31",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_verified": "convnext_tiny_forensic_multidataset_v5.onnx",
        "total_experiments": total,
        "average_score_improvement_delta": round(avg_delta, 4),
        "results": experiments,
    }

    out_path = REPORTS_DIR / "sprint31_before_after_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_bundle, f, indent=2, ensure_ascii=False)

    print(f"\n[+] Successfully saved {total} experiment evaluations to {out_path}")
    print(f"[+] Average score improvement delta: {avg_delta:+.4f}")


if __name__ == "__main__":
    run_benchmark()
