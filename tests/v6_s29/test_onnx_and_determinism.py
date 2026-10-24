"""
FHDSS SPRINT 29 — Test Suite Group 7: ONNX Parity, Determinism & Export Integrity
Verifies:
36. PyTorch vs ONNX numerical parity (MAE < 1e-4, Max Abs Error < 1e-4)
37. ONNX vs PyTorch Cosine Similarity >= 0.99999
38. 10-Iteration Determinism Audit (Zero embedding drift on identical inputs)
39. Dynamic axes batch size flexibility in ONNX model
40. Zero NaN or Inf values in ONNX execution
"""

import pytest
import sys
import numpy as np
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.siamese import (
    ForensicConvNeXtEncoder,
    compute_cosine_similarity
)


@pytest.fixture(scope="module")
def v6_onnx_session():
    import onnxruntime as ort
    onnx_path = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6.onnx"
    if not onnx_path.exists():
        pytest.skip("V6 ONNX model not found")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session, onnx_path


@pytest.fixture(scope="module")
def v6_pt_model():
    v6_ckpt = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6_best.pt"
    if not v6_ckpt.exists():
        pytest.skip("V6 PT checkpoint not found")
    m = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    c = torch.load(v6_ckpt, map_location="cpu")
    m.load_state_dict(c["model_state_dict"] if "model_state_dict" in c else c, strict=True)
    m.eval()
    return m


def test_pt_onnx_parity(v6_pt_model, v6_onnx_session):
    """Test 36 & 37: PyTorch vs ONNX outputs match with MAE < 1e-4 and Cosine >= 0.99999."""
    session, _ = v6_onnx_session
    rng = np.random.RandomState(42)
    dummy_input = rng.randn(4, 3, 224, 224).astype(np.float32)

    with torch.no_grad():
        pt_out = v6_pt_model(torch.from_numpy(dummy_input)).numpy()

    onnx_out = session.run(["embedding"], {"input": dummy_input})[0]

    max_abs_err = np.max(np.abs(pt_out - onnx_out))
    mae = np.mean(np.abs(pt_out - onnx_out))
    cos_sims = [compute_cosine_similarity(p, o) for p, o in zip(pt_out, onnx_out)]
    min_cos = np.min(cos_sims)

    assert max_abs_err < 1e-4, f"Parity violation: max_abs_err = {max_abs_err:.2e} >= 1e-4"
    assert min_cos >= 0.99999, f"Cosine similarity violation: min_cos = {min_cos:.6f} < 0.99999"


def test_ten_iteration_determinism(v6_pt_model):
    """Test 38: 10 repeated inferences on the same input yield bitwise/exact identical embeddings."""
    rng = np.random.RandomState(1337)
    sample_input = torch.from_numpy(rng.randn(1, 3, 224, 224).astype(np.float32))

    outputs = []
    with torch.no_grad():
        for _ in range(10):
            outputs.append(v6_pt_model(sample_input).numpy()[0])

    for i in range(1, 10):
        diff = np.max(np.abs(outputs[0] - outputs[i]))
        assert diff < 1e-7, f"Determinism failure: iteration {i} drifted by {diff:.2e}"


def test_onnx_dynamic_batching(v6_onnx_session):
    """Test 39: ONNX model supports dynamic batch sizes (e.g. N=1, N=7)."""
    session, _ = v6_onnx_session
    for b in [1, 3, 7]:
        inp = np.random.randn(b, 3, 224, 224).astype(np.float32)
        out = session.run(["embedding"], {"input": inp})[0]
        assert out.shape == (b, 128), f"Expected shape ({b}, 128), got {out.shape}"


def test_onnx_no_nan_or_inf(v6_onnx_session):
    """Test 40: ONNX inferences produce exclusively finite values."""
    session, _ = v6_onnx_session
    inp = np.random.randn(5, 3, 224, 224).astype(np.float32)
    out = session.run(["embedding"], {"input": inp})[0]
    assert np.all(np.isfinite(out)), "Non-finite values found in ONNX output!"
