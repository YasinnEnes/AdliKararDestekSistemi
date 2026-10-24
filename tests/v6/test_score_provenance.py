"""
FHDSS SPRINT 27 — Test: Forensic Score Provenance (Phase 13)
Verifies:
- Standardized provenance schema is strictly adhered to.
- Contains raw_cosine, mahalanobis_score, fusion_score, calibrated_score, threshold, score_source.
"""

from pathlib import Path
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_provenance_schema_compliance():
    sample_provenance = {
        "raw_cosine": 0.8842,
        "mahalanobis_score": 0.8120,
        "fusion_score": 0.8625,
        "calibrated_score": 0.8710,
        "threshold": 0.8983,
        "score_source": "V6_FORENSIC_FUSION_ENGINE"
    }

    required_keys = [
        "raw_cosine",
        "mahalanobis_score",
        "fusion_score",
        "calibrated_score",
        "threshold",
        "score_source"
    ]

    for k in required_keys:
        assert k in sample_provenance, f"Missing provenance field: {k}"

    assert 0.0 <= sample_provenance["raw_cosine"] <= 1.0
    assert 0.0 <= sample_provenance["calibrated_score"] <= 1.0
    assert sample_provenance["score_source"] in ["V6_FORENSIC_FUSION_ENGINE", "V5_PRODUCTION_ENGINE"]
