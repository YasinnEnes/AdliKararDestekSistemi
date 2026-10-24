"""
FHDSS SPRINT 25 — Phase 11 Test: Mahalanobis & Reference Robustness
Verifies that Mahalanobis distance calculation is numerically stable
across 1, 3, and 5 references without LinAlgError or singular matrix crashes.
"""
import pytest
import numpy as np

from ml.models.comparison import ForensicComparisonEngine

def test_mahalanobis_reference_counts():
    engine = ForensicComparisonEngine()
    
    # 13 morphological features
    p = 13
    q_vec = np.random.RandomState(42).randn(p)
    
    # Test 1 reference
    ref_1 = np.random.RandomState(43).randn(1, p)
    dist_1 = engine.compute_stabilized_mahalanobis(q_vec, ref_1)
    assert not np.isnan(dist_1)
    assert not np.isinf(dist_1)
    assert dist_1 >= 0.0
    
    # Test 2 references
    ref_2 = np.random.RandomState(44).randn(2, p)
    dist_2 = engine.compute_stabilized_mahalanobis(q_vec, ref_2)
    assert not np.isnan(dist_2)
    assert not np.isinf(dist_2)
    assert dist_2 >= 0.0
    
    # Test 3 references (Ledoit-Wolf)
    ref_3 = np.random.RandomState(45).randn(3, p)
    dist_3 = engine.compute_stabilized_mahalanobis(q_vec, ref_3)
    assert not np.isnan(dist_3)
    assert not np.isinf(dist_3)
    assert dist_3 >= 0.0
    
    # Test 5 references (Ledoit-Wolf)
    ref_5 = np.random.RandomState(46).randn(5, p)
    dist_5 = engine.compute_stabilized_mahalanobis(q_vec, ref_5)
    assert not np.isnan(dist_5)
    assert not np.isinf(dist_5)
    assert dist_5 >= 0.0
    
    # Test identical references (covariance collapse test)
    ref_identical = np.tile(q_vec, (5, 1))
    dist_ident = engine.compute_stabilized_mahalanobis(q_vec, ref_identical)
    assert abs(dist_ident - 0.0) < 1e-4, f"Identical vectors distance: {dist_ident}"
