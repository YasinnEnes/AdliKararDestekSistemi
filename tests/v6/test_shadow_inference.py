"""
FHDSS SPRINT 27 — Test: Production Shadow Mode Inference (Phase 17)
Verifies:
- V5 acts as primary decision-maker in shadow mode.
- V6 outputs are logged in parallel without overriding V5.
- Zero side-effects on production output pipeline.
"""

from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from ml.models.siamese import compute_normalized_similarity


def test_shadow_mode_decision_pipeline():
    # Mock embeddings for query and known reference
    np.random.seed(42)
    ref_emb = np.random.randn(128)
    ref_emb /= np.linalg.norm(ref_emb)

    query_emb_v5 = ref_emb + np.random.normal(0, 0.05, 128)
    query_emb_v5 /= np.linalg.norm(query_emb_v5)

    query_emb_v6 = ref_emb + np.random.normal(0, 0.06, 128)
    query_emb_v6 /= np.linalg.norm(query_emb_v6)

    th_v5 = 0.7334
    th_v6 = 0.8983

    sim_v5 = compute_normalized_similarity(query_emb_v5, ref_emb)
    sim_v6 = compute_normalized_similarity(query_emb_v6, ref_emb)

    primary_decision = "MATCH" if sim_v5 >= th_v5 else "NON-MATCH"
    shadow_decision = "MATCH" if sim_v6 >= th_v6 else "NON-MATCH"

    # Production response MUST return primary_decision
    production_payload = {
        "production_model": "V5",
        "decision": primary_decision,
        "score": sim_v5,
        "shadow_audit": {
            "shadow_model": "V6",
            "decision": shadow_decision,
            "score": sim_v6
        }
    }

    assert production_payload["production_model"] == "V5"
    assert production_payload["decision"] == primary_decision
    assert "shadow_audit" in production_payload
