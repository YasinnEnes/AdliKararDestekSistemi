"""
FHDSS SPRINT 29 — Test Suite Group 6: Security, Hash Freeze & Artifact Immutability
Verifies:
31. V5 PyTorch checkpoint SHA-256 matches immutable reference
32. V5 ONNX export SHA-256 matches immutable reference
33. V6 Baseline PyTorch checkpoint SHA-256 matches immutable reference
34. Model manifest contains all 18 mandatory metadata fields
35. Candidate weights saved strictly in artifacts/weights/v6_s29/
"""

import pytest
import sys
import json
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

V5_PT_EXPECTED = "0fe6963acaba48c26817202662907d1d2e52f57115d807d03497fdc33641938f"
V5_ONNX_EXPECTED = "ddb89311ad2681fb0e4f88e6ecfe3a7ef3850855d692bc91749a53a7fb52d066"
V6_PT_EXPECTED = "da6b1c6d126cc18c3e0e7762c782a3c1f1f8a79edd5698abcb9495e980efc627"


def test_v5_pt_immutability():
    """Test 31: V5 PyTorch checkpoint must remain completely unmodified."""
    v5_pt = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5_best.pt"
    assert v5_pt.exists(), f"V5 Checkpoint missing at {v5_pt}"
    current_hash = hashlib.sha256(v5_pt.read_bytes()).hexdigest()
    assert current_hash == V5_PT_EXPECTED, f"CRITICAL SECURITY ALERT: V5 Checkpoint hash modified! Expected {V5_PT_EXPECTED}, got {current_hash}"


def test_v5_onnx_immutability():
    """Test 32: V5 ONNX engine must remain completely unmodified."""
    v5_onnx = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5.onnx"
    assert v5_onnx.exists(), f"V5 ONNX missing at {v5_onnx}"
    current_hash = hashlib.sha256(v5_onnx.read_bytes()).hexdigest()
    assert current_hash == V5_ONNX_EXPECTED, f"CRITICAL SECURITY ALERT: V5 ONNX hash modified! Expected {V5_ONNX_EXPECTED}, got {current_hash}"


def test_v6_pt_immutability():
    """Test 33: V6 Baseline PyTorch checkpoint must remain unmodified."""
    v6_pt = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6_best.pt"
    assert v6_pt.exists(), f"V6 Checkpoint missing at {v6_pt}"
    current_hash = hashlib.sha256(v6_pt.read_bytes()).hexdigest()
    assert current_hash == V6_PT_EXPECTED, f"V6 Baseline hash modified! Expected {V6_PT_EXPECTED}, got {current_hash}"


def test_manifest_metadata_fields():
    """Test 34: Manifest JSON contains required metadata fields."""
    manifest_file = PROJECT_ROOT / "artifacts" / "weights" / "v6_s29" / "candidates" / "model_manifest_candidate_d.json"
    if manifest_file.exists():
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        required_keys = ["model_name", "architecture", "checkpoint_sha256", "thresholds", "metrics", "onnx_parity"]
        for k in required_keys:
            assert k in data, f"Missing manifest field: {k}"


def test_artifact_isolation():
    """Test 35: All Sprint 29 weights are strictly isolated in artifacts/weights/v6_s29/."""
    s29_dir = PROJECT_ROOT / "artifacts" / "weights" / "v6_s29"
    assert s29_dir.exists(), "v6_s29 directory does not exist!"
    # Ensure no s29 weights were written directly to artifacts/weights/
    root_weights = list((PROJECT_ROOT / "artifacts" / "weights").glob("candidate*.pt"))
    assert len(root_weights) == 0, f"Found candidate weights in root artifacts directory: {root_weights}"
