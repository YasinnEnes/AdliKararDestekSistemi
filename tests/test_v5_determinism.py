"""
FHDSS SPRINT 25 — Phase 5 Test: Determinism Test
Verifies that 10 sequential passes of identical inputs through the V5 model,
cosine comparison, and fusion produce strictly identical outputs (zero std dev).
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import cv2

from ml.models.siamese import ForensicConvNeXtEncoder, compute_normalized_similarity
from ml.models.comparison import ComparisonReport
from ml.models.fusion import HybridForensicFusion
from ml.preprocessing.contract import preprocess_for_inference

def test_v5_inference_determinism():
    pt_path = Path("artifacts/weights/convnext_tiny_forensic_multidataset_v5_best.pt")
    assert pt_path.exists()
    
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    
    fusion = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)
    
    # Deterministic synthetic input
    rng = np.random.RandomState(42)
    img_a = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img_b = rng.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    
    t_a = preprocess_for_inference(img_a, is_bgr=True)
    t_b = preprocess_for_inference(img_b, is_bgr=True)
    
    dummy_morph = ComparisonReport(
        similarity_score=0.88,
        euclidean_distance=1.5,
        mahalanobis_distance=0.5,
        concordant_features=[],
        natural_variations=[],
        significant_diffs=[],
        preliminary_opinion="Concordant",
        reference_count=3,
        audit_hash_sha256="det_test",
    )
    
    embs_a = []
    embs_b = []
    scores = []
    
    for _ in range(10):
        with torch.no_grad():
            ea = model(t_a).cpu().numpy()[0]
            eb = model(t_b).cpu().numpy()[0]
            
        sim = compute_normalized_similarity(ea, eb)
        cos = float(np.dot(ea, eb))
        res = fusion.fuse(dummy_morph, deep_similarity_score=sim, raw_cosine=cos)
        
        embs_a.append(ea)
        embs_b.append(eb)
        scores.append(res.hybrid_similarity_score)
        
    std_ea = float(np.max(np.std(np.array(embs_a), axis=0)))
    std_eb = float(np.max(np.std(np.array(embs_b), axis=0)))
    std_scores = float(np.std(scores))
    
    assert std_ea < 1e-6, f"Embedding A non-deterministic: std={std_ea}"
    assert std_eb < 1e-6, f"Embedding B non-deterministic: std={std_eb}"
    assert std_scores < 1e-6, f"Fusion score non-deterministic: std={std_scores}"
