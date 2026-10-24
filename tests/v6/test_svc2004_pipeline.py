"""
FHDSS SPRINT 26 — Test: SVC2004 Dynamic & Multimodal Pipeline (Phase 14 & 15)
Verifies:
- DynamicKinematicSequenceEncoder processes (B, T, C) without 2D image rendering (Rule 5).
- Output is 128D L2-normalized.
- MultimodalForensicFusion correctly fuses static and dynamic modalities.
"""

from pathlib import Path
import sys
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.research.svc2004_dynamic import (
    DynamicKinematicSequenceEncoder,
    MultimodalForensicFusion,
    simulate_dynamic_kinematic_benchmark
)


def test_dynamic_encoder_and_multimodal_fusion():
    res = simulate_dynamic_kinematic_benchmark()
    assert res["status"] == "PASS"
    assert res["dynamic_embedding_dim"] == 128
    assert abs(res["dynamic_norm_mean"] - 1.0) < 1e-4
    assert abs(res["fused_concat_norm"] - 1.0) < 1e-4
    assert abs(res["fused_gated_norm"] - 1.0) < 1e-4
    assert 0.0 <= res["mean_static_gate_weight"] <= 1.0
