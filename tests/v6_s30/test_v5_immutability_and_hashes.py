"""
Sprint 30 Tests: V5 Production Authority Immutability & Model Checkpoint Integrity
Verifies zero drift on frozen V5 and V6 baselines.
"""

import hashlib
import json
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_v5_production_pt_hash_exact():
    v5_pt = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5_best.pt"
    assert v5_pt.exists(), "V5 Production PT missing!"
    expected_hash = "0fe6963acaba48c26817202662907d1d2e52f57115d807d03497fdc33641938f"
    actual_hash = hashlib.sha256(v5_pt.read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f"V5 PT modified! {actual_hash} != {expected_hash}"


def test_v5_production_onnx_hash_exact():
    v5_onnx = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5.onnx"
    assert v5_onnx.exists(), "V5 Production ONNX missing!"
    expected_hash = "ddb89311ad2681fb0e4f88e6ecfe3a7ef3850855d692bc91749a53a7fb52d066"
    actual_hash = hashlib.sha256(v5_onnx.read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f"V5 ONNX modified! {actual_hash} != {expected_hash}"


def test_v5_production_manifest_hash_exact():
    v5_manifest = PROJECT_ROOT / "artifacts" / "weights" / "model_manifest.json"
    assert v5_manifest.exists(), "V5 Production manifest missing!"
    expected_hash = "8ebd6f0c59cd4136474d1f1e770345190a50b84f40f04f425065aafbe221370d"
    actual_hash = hashlib.sha256(v5_manifest.read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f"V5 Manifest modified! {actual_hash} != {expected_hash}"


def test_v6_baseline_pt_hash_exact():
    v6_pt = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6_best.pt"
    assert v6_pt.exists(), "V6 Baseline PT missing!"
    expected_hash = "da6b1c6d126cc18c3e0e7762c782a3c1f1f8a79edd5698abcb9495e980efc627"
    actual_hash = hashlib.sha256(v6_pt.read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f"V6 PT modified! {actual_hash} != {expected_hash}"


def test_v6_baseline_onnx_hash_exact():
    v6_onnx = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6.onnx"
    assert v6_onnx.exists(), "V6 Baseline ONNX missing!"
    expected_hash = "e33a97f933d8410743777726af83c86a3408bc48431ddaaa6f86f1c9c680a2e3"
    actual_hash = hashlib.sha256(v6_onnx.read_bytes()).hexdigest()
    assert actual_hash == expected_hash, f"V6 ONNX modified! {actual_hash} != {expected_hash}"


def test_all_candidate_checkpoints_exist():
    cand_dir = PROJECT_ROOT / "artifacts" / "weights" / "v6_s29" / "candidates"
    for c in ["a", "b", "c", "d", "e"]:
        best_pt = cand_dir / f"candidate_{c}_best.pt"
        assert best_pt.exists(), f"Candidate {c} best.pt missing!"
        assert best_pt.stat().st_size > 50_000_000, f"Candidate {c} best.pt truncated!"
