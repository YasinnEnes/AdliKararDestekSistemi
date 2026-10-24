"""
FHDSS SPRINT 26 — Test: V6 Manifest Integrity
Verifies:
- model_manifest_v6.json exists and is valid JSON.
- Contains all required fields (thresholds, metrics, hashes).
- Recorded checkpoint and ONNX SHA-256 hashes match physical files on disk.
"""

from pathlib import Path
import hashlib
import json
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_v6_manifest_fields_and_hashes():
    v6_weights = PROJECT_ROOT / "artifacts" / "weights" / "v6"
    manifest_path = v6_weights / "model_manifest_v6.json"
    v6_pt = v6_weights / "convnext_tiny_forensic_multidataset_v6_best.pt"
    v6_onnx = v6_weights / "convnext_tiny_forensic_multidataset_v6.onnx"

    if not manifest_path.exists():
        pytest.skip("V6 manifest not created yet")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Required top-level keys
    for key in ["model_version", "architecture", "checkpoint_sha256", "onnx_sha256", "thresholds", "metrics"]:
        assert key in manifest, f"Missing manifest key: {key}"

    # Verify SHA-256 matches actual files
    if v6_pt.exists():
        actual_pt_sha = hashlib.sha256(v6_pt.read_bytes()).hexdigest()
        assert manifest["checkpoint_sha256"] == actual_pt_sha, "Manifest checkpoint SHA-256 mismatch"

    if v6_onnx.exists():
        actual_onnx_sha = hashlib.sha256(v6_onnx.read_bytes()).hexdigest()
        assert manifest["onnx_sha256"] == actual_onnx_sha, "Manifest ONNX SHA-256 mismatch"

    # Verify threshold monotonicity
    th = manifest["thresholds"]
    assert th["eer_threshold"] <= th["standard_threshold_far5"] <= th["strict_threshold_far1"]
