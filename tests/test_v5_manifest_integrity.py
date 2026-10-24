"""
FHDSS SPRINT 25 — Phase 14 & 16 Test: Manifest & Cryptographic Integrity
Verifies that model_manifest.json points to V5 as active_primary_model,
that the files exist on disk, and their calculated SHA-256 hashes match.
"""
import hashlib
import json
import pytest
from pathlib import Path

from ml.models.registry import model_registry

def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_manifest_v5_integrity():
    model_registry.reload()
    manifest = model_registry.load_manifest()
    
    assert manifest.get("active_primary_model") == "convnext_tiny_forensic_multidataset_v5"
    
    pt_path = model_registry.get_active_pt_path()
    onnx_path = model_registry.get_active_onnx_path()
    
    assert pt_path.exists(), f"PT file does not exist: {pt_path}"
    assert onnx_path.exists(), f"ONNX file does not exist: {onnx_path}"
    assert pt_path.name == "convnext_tiny_forensic_multidataset_v5_best.pt"
    assert onnx_path.name == "convnext_tiny_forensic_multidataset_v5.onnx"
    
    # Verify SHA-256
    expected_pt_sha = manifest["weights"]["pytorch_checkpoint"]["sha256"]
    expected_onnx_sha = manifest["weights"]["onnx_model"]["sha256"]
    
    actual_pt_sha = compute_sha256(pt_path)
    actual_onnx_sha = compute_sha256(onnx_path)
    
    assert actual_pt_sha.lower() == expected_pt_sha.lower()
    assert actual_onnx_sha.lower() == expected_onnx_sha.lower()
    
    # Verify V4 baseline is also preserved and intact
    v4_pt = pt_path.parent / "convnext_tiny_forensic_multidataset_v4_best.pt"
    assert v4_pt.exists()
    assert compute_sha256(v4_pt) == "3f0640047c170874cd0369f9c19451dd9fa857a24f8410d7c3f5c2393b5e2e5e"

def test_operational_thresholds_configured():
    manifest = model_registry.load_manifest()
    thresholds = manifest.get("operational_thresholds", {})
    assert "balanced_eer" in thresholds
    assert "youden_optimal" in thresholds
    assert "forensic_strict_far_1pct" in thresholds
    assert "forensic_standard_far_5pct" in thresholds
    assert manifest.get("active_inference_threshold") is not None
