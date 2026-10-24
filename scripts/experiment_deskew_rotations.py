import math
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import cv2
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cv.preprocessing.pipeline import ForensicImagePreprocessor
from cv.features.extractors import ForensicFeatureExtractor
from ml.models.comparison import ForensicComparisonEngine
from ml.models.fusion import HybridForensicFusion
from ml.models.siamese import ForensicConvNeXtEncoder
from ml.preprocessing.contract import preprocess_for_inference
from scripts.debug_similarity import create_demo_img


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Görüntüyü merkezinden verilen açı kadar döndürür, arka planı beyaz (255) doldurur."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    rot_mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    rotated = cv2.warpAffine(
        image,
        rot_mat,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated


def run_deskew_experiment() -> Dict[str, Any]:
    print("\n" + "=" * 80)
    print("  FHDSS SPRINT 21: DESKEW & ROTATION STABILITY EXPERIMENT")
    print("=" * 80)

    # V2 modelini yükle
    weights_path = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_cedar_v2_best.pt"
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    if weights_path.exists():
        ckpt = torch.load(weights_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        encoder.load_state_dict(state_dict, strict=False)
    encoder.eval()

    preprocessor = ForensicImagePreprocessor()
    extractor = ForensicFeatureExtractor()
    engine = ForensicComparisonEngine()
    fusion = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)

    base_img = create_demo_img(slant_angle=14.5, with_tremor=False)
    test_rotations = [0.0, 0.5, -0.5, 1.0, -1.0, 3.0, -3.0]

    results_deskew_on: List[Dict[str, Any]] = []
    results_deskew_off: List[Dict[str, Any]] = []

    for deskew_mode in [True, False]:
        prep = preprocessor
        mode_label = "DESKEW AÇIK (Hough)" if deskew_mode else "DESKEW KAPALI"
        print(f"\n--- MOD: {mode_label} ---")
        print(f"{'Rotasyon':>10s} | {'Est. Skew':>10s} | {'Cosine':>8s} | {'Mahalanobis':>12s} | {'Morph Sim':>10s} | {'Gating Trig':>12s} | {'Final Score':>11s}")
        print("-" * 85)

        # Referans görsel
        art_ref = prep.process(base_img, deskew_enabled=deskew_mode)
        feat_ref = extractor.extract_all(art_ref.skeleton, art_ref.binarized)
        t_ref = preprocess_for_inference(base_img)
        with torch.no_grad():
            emb_ref = encoder(t_ref)

        for rot in test_rotations:
            rot_img = rotate_image(base_img, rot)
            art_q = prep.process(rot_img, deskew_enabled=deskew_mode)
            feat_q = extractor.extract_all(art_q.skeleton, art_q.binarized)

            t_q = preprocess_for_inference(rot_img)
            with torch.no_grad():
                emb_q = encoder(t_q)

            cos_sim = float(F.cosine_similarity(emb_q, emb_ref, dim=-1).item())
            norm_deep = (cos_sim + 1.0) / 2.0

            comp_rep = engine.compare(feat_q, [feat_ref])
            fus_res = fusion.fuse(comp_rep, norm_deep)

            est_angle = float(art_q.metadata.get("deskew_angle_deg", 0.0))

            row = {
                "rotation_deg": rot,
                "estimated_skew_deg": est_angle,
                "embedding_cosine": round(cos_sim, 4),
                "mahalanobis": round(comp_rep.mahalanobis_distance, 4),
                "morphological_score": round(comp_rep.similarity_score, 4),
                "gating_triggered": fus_res.gating_triggered,
                "final_score": round(fus_res.hybrid_similarity_score, 4),
            }
            if deskew_mode:
                results_deskew_on.append(row)
            else:
                results_deskew_off.append(row)

            print(f"{rot:>9.1f}° | {est_angle:>9.2f}° | {cos_sim:>8.4f} | {comp_rep.mahalanobis_distance:>12.3f} | {comp_rep.similarity_score:>10.4f} | {str(fus_res.gating_triggered):>12s} | {fus_res.hybrid_similarity_score:>11.4f}")

    return {
        "deskew_on": results_deskew_on,
        "deskew_off": results_deskew_off,
    }


if __name__ == "__main__":
    run_deskew_experiment()
