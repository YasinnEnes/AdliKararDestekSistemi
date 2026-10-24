"""
FHDSS SPRINT 25 — Phase 4 Test: Zero-Score & Gating Forensic Audit
Validates that low scores stem solely from genuine orthogonality/anti-parallelism
or forensic gating penalties, with typed ScoreProvenance tracking.
"""
import pytest
import numpy as np

from ml.models.siamese import (
    compute_normalized_similarity,
    compute_cosine_similarity,
)
from ml.models.comparison import ComparisonReport, FeatureDifferenceDetail, DifferenceType
from ml.models.fusion import (
    HybridForensicFusion,
    ScoreProvenance,
    ScoreStatus,
)

def test_cosine_normalization():
    # Identical vectors -> Cosine = 1.0 -> Normalized = 1.0
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sim_norm = compute_normalized_similarity(v1, v2)
    assert abs(sim_norm - 1.0) < 1e-5
    
    # Orthogonal vectors -> Cosine = 0.0 -> Normalized = 0.5
    v3 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sim_ortho = compute_normalized_similarity(v1, v3)
    assert abs(sim_ortho - 0.5) < 1e-5
    
    # Opposite vectors -> Cosine = -1.0 -> Normalized = 0.0
    v4 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
    sim_opp = compute_normalized_similarity(v1, v4)
    assert abs(sim_opp - 0.0) < 1e-5

def test_score_provenance_audit():
    fusion_engine = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)
    
    comp_normal = ComparisonReport(
        similarity_score=0.92,
        euclidean_distance=1.2,
        mahalanobis_distance=0.35,
        concordant_features=[],
        natural_variations=[],
        significant_diffs=[],
        preliminary_opinion="Concordant",
        reference_count=3,
        audit_hash_sha256="test_hash_a",
    )
    res_a = fusion_engine.fuse(comp_normal, deep_similarity_score=0.94, raw_cosine=0.88)
    assert res_a.score_provenance.status == ScoreStatus.VALID_SCORE
    assert res_a.hybrid_similarity_score > 0.90
    
    # Low similarity (both deep and morph low)
    comp_diff = ComparisonReport(
        similarity_score=0.15,
        euclidean_distance=15.0,
        mahalanobis_distance=7.5,
        concordant_features=[],
        natural_variations=[],
        significant_diffs=[],
        preliminary_opinion="Different",
        reference_count=3,
        audit_hash_sha256="test_hash_c",
    )
    res_c = fusion_engine.fuse(comp_diff, deep_similarity_score=0.12, raw_cosine=-0.76)
    assert res_c.score_provenance.status == ScoreStatus.REAL_LOW_SIMILARITY
    assert res_c.hybrid_similarity_score < 0.25
    
    # Invalid Embedding (NaN)
    res_d = fusion_engine.fuse(comp_normal, deep_similarity_score=float("nan"))
    assert res_d.score_provenance.status == ScoreStatus.INVALID_EMBEDDING
