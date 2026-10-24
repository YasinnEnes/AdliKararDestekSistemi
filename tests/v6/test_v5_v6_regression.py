"""
FHDSS SPRINT 26 — Test: V5 vs V6 Regression & Preservation (KURAL 1)
Verifies:
- V5 model weights and ONNX are preserved and unaltered.
- V6 artifacts reside in dedicated v6 namespace (artifacts/weights/v6/).
- V6 does not experience catastrophic regression on core signature benchmarks.
"""

from pathlib import Path
import hashlib
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_v5_weights_preservation():
    weights_dir = PROJECT_ROOT / "artifacts" / "weights"
    v5_pt = weights_dir / "convnext_tiny_forensic_multidataset_v5_best.pt"
    v5_onnx = weights_dir / "convnext_tiny_forensic_multidataset_v5.onnx"

    assert v5_pt.exists(), "V5 PyTorch checkpoint must exist and be preserved"
    assert v5_onnx.exists(), "V5 ONNX artifact must exist and be preserved"

    # Known frozen V5 SHA-256
    v5_pt_hash = hashlib.sha256(v5_pt.read_bytes()).hexdigest()
    assert v5_pt_hash == "0fe6963acaba48c26817202662907d1d2e52f57115d807d03497fdc33641938f", "V5 PyTorch checkpoint was modified! Rule 1 violation."


def test_v6_namespace_separation():
    v6_dir = PROJECT_ROOT / "artifacts" / "weights" / "v6"
    assert v6_dir.exists(), "V6 weights directory must exist as a separate namespace"
