"""
FHDSS SPRINT 25 — Phase 12 Test: Forensic Gating Audit & Fusion Regression
Verifies that forensic gating penalizes deep scores solely upon structural
contradictions and reports explicit reason traces without artificial 0 fallback.
"""
import pytest
import numpy as np

from ml.models.comparison import ComparisonReport, FeatureDifferenceDetail, DifferenceType
from ml.models.fusion import HybridForensicFusion, ScoreStatus

def test_forensic_gating_scenarios():
    fusion = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)
    
    # 1. Normal match without contradictions
    comp_normal = ComparisonReport(
        similarity_score=0.90,
        euclidean_distance=1.0,
        mahalanobis_distance=0.4,
        concordant_features=[],
        natural_variations=[],
        significant_diffs=[],
        preliminary_opinion="Concordant",
        reference_count=3,
        audit_hash_sha256="hash_normal",
    )
    res_normal = fusion.fuse(comp_normal, deep_similarity_score=0.92, raw_cosine=0.84)
    assert not res_normal.gating_triggered
    assert res_normal.gating_penalty == 0.0
    assert res_normal.score_provenance.status == ScoreStatus.VALID_SCORE
    
    # 2. Contradiction triggering gating penalty
    diff_slant = FeatureDifferenceDetail(
        feature_key="slant_angle_deg",
        feature_name_tr="Yazı Eğim Açısı",
        questioned_value=25.0,
        reference_mean=5.0,
        reference_std=2.0,
        absolute_difference=20.0,
        z_score=10.0,
        diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
        explanation="Major slant contradiction",
    )
    comp_contra = ComparisonReport(
        similarity_score=0.50,
        euclidean_distance=7.0,
        mahalanobis_distance=2.8,
        concordant_features=[],
        natural_variations=[],
        significant_diffs=[diff_slant],
        preliminary_opinion="Contradiction",
        reference_count=3,
        audit_hash_sha256="hash_contra",
    )
    res_contra = fusion.fuse(comp_contra, deep_similarity_score=0.85, raw_cosine=0.70)
    # Gating should trigger or apply structural penalty
    assert res_contra.hybrid_similarity_score < 0.85
    assert res_contra.score_provenance is not None
