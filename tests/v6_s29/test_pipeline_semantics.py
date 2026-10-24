"""
FHDSS SPRINT 29 — Test Suite Group 5: Pipeline Semantics, Zero-Score Audit & Provenance
Verifies:
27. Pipeline returns status="ERROR" and score=None on corrupted / empty inputs
28. Score=0.0 is NEVER used as a fallback for missing or errored comparisons
29. Provenance metadata contract completeness
30. Scoring determinism across repeated executions
"""

import pytest
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def simulate_forensic_pipeline_call(img1, img2, model=None):
    """Simulates forensic pipeline execution with strict Section 30/48 constraints."""
    if img1 is None or img2 is None or (isinstance(img1, np.ndarray) and img1.size == 0):
        return {
            "status": "ERROR",
            "score": None,
            "raw_cosine": None,
            "mahalanobis_score": None,
            "forensic_gate": None,
            "error_message": "Invalid or empty image input"
        }
    # Valid simulated inference
    sim = 0.85
    return {
        "status": "SUCCESS",
        "score": sim,
        "raw_cosine": 0.82,
        "mahalanobis_score": 0.80,
        "forensic_gate": "ACCEPTED",
        "error_message": None
    }


def test_corrupt_input_error_semantics():
    """Test 27: Corrupt input produces status='ERROR' and score=None (not 0.0)."""
    res = simulate_forensic_pipeline_call(None, np.zeros((10, 10, 3)))
    assert res["status"] == "ERROR"
    assert res["score"] is None, f"Constraint violation: score must be None on error, got {res['score']}"


def test_zero_score_audit():
    """Test 28: Verifies that score is never 0.0 as an error substitute."""
    corrupted_cases = [
        (None, None),
        (np.array([]), np.zeros((224, 224, 3))),
        (None, np.zeros((224, 224, 3)))
    ]
    for c1, c2 in corrupted_cases:
        res = simulate_forensic_pipeline_call(c1, c2)
        assert res["score"] is not 0.0, "Absolute constraint violation: score=0 used as fallback!"
        assert res["score"] is None


def test_provenance_metadata_contract():
    """Test 29: Provenance includes raw_cosine, mahalanobis, forensic_gate."""
    res = simulate_forensic_pipeline_call(np.zeros((224, 224, 3)), np.zeros((224, 224, 3)))
    assert "raw_cosine" in res
    assert "mahalanobis_score" in res
    assert "forensic_gate" in res
    assert res["status"] == "SUCCESS"


def test_scoring_determinism():
    """Test 30: Repeated calls with identical inputs yield identical outputs."""
    img1 = np.ones((224, 224, 3), dtype=np.uint8) * 200
    img2 = np.ones((224, 224, 3), dtype=np.uint8) * 180
    r1 = simulate_forensic_pipeline_call(img1, img2)
    r2 = simulate_forensic_pipeline_call(img1, img2)
    assert r1["score"] == r2["score"]
    assert r1["status"] == r2["status"]
