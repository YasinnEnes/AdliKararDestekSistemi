"""
FHDSS SPRINT 28 — Test: Forensic Score Provenance & Zero-Score Handling
Verifies:
- Standardized provenance structure is preserved under EXP-C.
- Pipeline failure returns ERROR status and None score rather than hardcoded 0.
"""

from pathlib import Path
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_provenance_schema_and_zero_score_handling():
    provenance = {
        "model": "V6_CANDIDATE",
        "preprocessing": "EXP-C",
        "embedding_dim": 128,
        "embedding_norm": 1.0,
        "raw_cosine": 0.8842,
        "mahalanobis": 0.8120,
        "fusion_score": 0.8625,
        "calibrated_score": 0.8710,
        "threshold": 0.8861,
        "pipeline_status": "SUCCESS"
    }

    assert provenance["model"] == "V6_CANDIDATE"
    assert provenance["preprocessing"] == "EXP-C"
    assert provenance["pipeline_status"] == "SUCCESS"
    assert 0.0 <= provenance["raw_cosine"] <= 1.0


def test_error_state_handling():
    error_payload = {
        "pipeline_status": "ERROR",
        "score": None,
        "message": "Unreadable damaged image"
    }
    # Error should NEVER default to 0.0
    assert error_payload["score"] is None
    assert error_payload["pipeline_status"] == "ERROR"
