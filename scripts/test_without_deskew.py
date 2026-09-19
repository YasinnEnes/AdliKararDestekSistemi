import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from cv.preprocessing.pipeline import ForensicImagePreprocessor
from cv.features.extractors import ForensicFeatureExtractor
from ml.models.comparison import ForensicComparisonEngine
from ml.models.fusion import HybridForensicFusion
from scripts.debug_similarity import create_demo_img

# Without deskew
preprocessor = ForensicImagePreprocessor()
extractor = ForensicFeatureExtractor()
engine = ForensicComparisonEngine()
fusion = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)

img_q = create_demo_img(14.5, False)
img_r1 = create_demo_img(14.0, False)
img_r2 = create_demo_img(15.0, False)

art_q = preprocessor.process(img_q, deskew_enabled=False)
feat_q = extractor.extract_all(art_q.skeleton, art_q.binarized)

art_r1 = preprocessor.process(img_r1, deskew_enabled=False)
feat_r1 = extractor.extract_all(art_r1.skeleton, art_r1.binarized)

art_r2 = preprocessor.process(img_r2, deskew_enabled=False)
feat_r2 = extractor.extract_all(art_r2.skeleton, art_r2.binarized)

print("=== WITHOUT DESKEW ===")
for k in engine.FEATURE_KEYS:
    vq = getattr(feat_q.structural, k, getattr(feat_q.line_quality, k, None))
    vr1 = getattr(feat_r1.structural, k, getattr(feat_r1.line_quality, k, None))
    vr2 = getattr(feat_r2.structural, k, getattr(feat_r2.line_quality, k, None))
    print(f"{k:25s} | Q: {str(vq):10s} | R1: {str(vr1):10s} | R2: {str(vr2):10s}")

comp1 = engine.compare(feat_q, [feat_r1])
print(f"\nComp1 (1 ref) Sim: {comp1.similarity_score}, Mahalanobis: {comp1.mahalanobis_distance}")
print("Sig diffs 1 ref count:", len(comp1.significant_differences))
for d in comp1.significant_differences:
    print(f"  Sig diff: {d.feature_key} Q={d.questioned_value} RefMean={d.reference_mean} z={d.z_score}")

fus1 = fusion.fuse(comp1, 1.0)
print(f"Fusion1 Hybrid Score: {fus1.hybrid_similarity_score}, Gating: {fus1.gating_triggered}")
