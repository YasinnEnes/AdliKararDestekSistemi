import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
import torch
import math

from cv.preprocessing.pipeline import ForensicImagePreprocessor
from cv.features.extractors import ForensicFeatureExtractor
from ml.models.comparison import ForensicComparisonEngine
from ml.models.siamese import ForensicConvNeXtEncoder, compute_embedding_distance
from ml.models.fusion import HybridForensicFusion
from ml.evaluation.hypothesis import ForensicHypothesisEvaluator
from cv.preprocessing.suitability import SuitabilityAssessor

def create_demo_img(slant_angle=14.5, with_tremor=False):
    img = np.ones((400, 600, 3), dtype=np.uint8) * 245
    cv2.line(img, (50, 300), (550, 300), (200, 190, 180), 1)
    
    def bezier_curve(p0, p1, p2, p3):
        return [( (1-t)**3 * p0[0] + 3*(1-t)**2 * t * p1[0] + 3*(1-t)*t**2 * p2[0] + t**3 * p3[0],
                  (1-t)**3 * p0[1] + 3*(1-t)**2 * t * p1[1] + 3*(1-t)*t**2 * p2[1] + t**3 * p3[1] ) for t in np.linspace(0, 1, 50)]

    c1 = bezier_curve((-160, 30), (-130, -80), (-80, -70), (-60, 15))
    c2 = bezier_curve((-60, 15), (-40, 60), (-10, 70), (10, 10))
    c3 = bezier_curve((10, 10), (40, -100), (100, -80), (120, 20))
    c4 = bezier_curve((120, 20), (140, 60), (180, 70), (200, 40))
    all_pts = c1 + c2 + c3 + c4
    c5 = bezier_curve((-140, 65), (-40, 85), (90, 45), (180, 65))
    
    rad = math.radians(slant_angle)
    cos_r = math.cos(rad)
    sin_r = math.sin(rad)
    
    def transform(p):
        rx = p[0] * cos_r - p[1] * sin_r + 300
        ry = p[0] * sin_r + p[1] * cos_r + 200
        return (int(rx), int(ry))

    curve_pts = np.array([transform(p) for p in all_pts], dtype=np.int32)
    bottom_pts = np.array([transform(p) for p in c5], dtype=np.int32)

    cv2.polylines(img, [curve_pts], False, (26, 26, 46), 4, cv2.LINE_AA)
    cv2.polylines(img, [bottom_pts], False, (26, 26, 46), 4, cv2.LINE_AA)
    return img

preprocessor = ForensicImagePreprocessor()
extractor = ForensicFeatureExtractor()
engine = ForensicComparisonEngine()
encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
encoder.eval()
fusion = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)
hypothesis_eval = ForensicHypothesisEvaluator()

img_q = create_demo_img(14.5, False)
img_r1 = create_demo_img(14.0, False)
img_r2 = create_demo_img(15.0, False)

art_q = preprocessor.process(img_q)
feat_q = extractor.extract_all(art_q.skeleton, art_q.binarized)

art_r1 = preprocessor.process(img_r1)
feat_r1 = extractor.extract_all(art_r1.skeleton, art_r1.binarized)
art_r2 = preprocessor.process(img_r2)
feat_r2 = extractor.extract_all(art_r2.skeleton, art_r2.binarized)

print("\n=== FEATURE VALUES COMPARISON ===")
for k in engine.FEATURE_KEYS:
    vq = getattr(feat_q.structural, k, getattr(feat_q.line_quality, k, None))
    vr1 = getattr(feat_r1.structural, k, getattr(feat_r1.line_quality, k, None))
    vr2 = getattr(feat_r2.structural, k, getattr(feat_r2.line_quality, k, None))
    print(f"{k:25s} | Q: {str(vq):10s} | R1: {str(vr1):10s} | R2: {str(vr2):10s}")

# Deep Sim
q_t = torch.from_numpy(cv2.resize(img_q, (224, 224))).permute(2, 0, 1).float().unsqueeze(0) / 255.0
r1_t = torch.from_numpy(cv2.resize(img_r1, (224, 224))).permute(2, 0, 1).float().unsqueeze(0) / 255.0
r2_t = torch.from_numpy(cv2.resize(img_r2, (224, 224))).permute(2, 0, 1).float().unsqueeze(0) / 255.0

with torch.inference_mode():
    eq = encoder(q_t)
    er1 = encoder(r1_t)
    er2 = encoder(r2_t)
    _, _, sim1 = compute_embedding_distance(eq, er1)
    _, _, sim2 = compute_embedding_distance(eq, er2)
    deep_sim = float(np.mean([sim1, sim2]))

print(f"\nDeep sim r1: {sim1}, r2: {sim2}, avg: {deep_sim}")

# Single ref comparison
comp1 = engine.compare(feat_q, [feat_r1])
print(f"Comp1 Sim: {comp1.similarity_score}, Mahalanobis: {comp1.mahalanobis_distance}")
print("Sig diffs 1 ref count:", len(comp1.significant_differences))
for d in comp1.significant_differences:
    print(f"  Sig diff: {d.feature_key} Q={d.questioned_value} RefMean={d.reference_mean} z={d.z_score}")

fus1 = fusion.fuse(comp1, sim1)
print(f"Fusion1 Hybrid Score: {fus1.hybrid_similarity_score}, Gating: {fus1.gating_triggered}, Penalty: {fus1.gating_penalty}")
print(f"Contradictions: {fus1.structural_contradictions}")

# 2 refs comparison
comp2 = engine.compare(feat_q, [feat_r1, feat_r2])
print(f"\nComp2 Sim: {comp2.similarity_score}, Mahalanobis: {comp2.mahalanobis_distance}")
print("Sig diffs 2 refs count:", len(comp2.significant_differences))
for d in comp2.significant_differences:
    print(f"  Sig diff: {d.feature_key} Q={d.questioned_value} RefMean={d.reference_mean} z={d.z_score}")

fus2 = fusion.fuse(comp2, deep_sim)
print(f"Fusion2 Hybrid Score: {fus2.hybrid_similarity_score}, Gating: {fus2.gating_triggered}, Penalty: {fus2.gating_penalty}")
