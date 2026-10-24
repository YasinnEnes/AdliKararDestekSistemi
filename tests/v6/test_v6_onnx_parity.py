"""
FHDSS SPRINT 26 — Test: V6 PyTorch vs ONNX Parity (Phase 29)
Verifies:
- Numerical parity between PyTorch checkpoint and exported ONNX model.
- Max Absolute Error < 1e-4.
- Embedding Cosine Similarity > 0.9999.
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

try:
    import onnxruntime as ort
    HAS_ORT = True
except ImportError:
    HAS_ORT = False

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.siamese import ForensicConvNeXtEncoder, compute_cosine_similarity


@pytest.mark.skipif(not HAS_ORT, reason="onnxruntime not installed")
def test_v6_pt_onnx_numerical_parity():
    v6_weights = PROJECT_ROOT / "artifacts" / "weights" / "v6"
    v6_pt = v6_weights / "convnext_tiny_forensic_multidataset_v6_best.pt"
    v6_onnx = v6_weights / "convnext_tiny_forensic_multidataset_v6.onnx"

    if not v6_pt.exists() or not v6_onnx.exists():
        pytest.skip("V6 weights or ONNX not generated yet")

    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(v6_pt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=True)
    model.eval()

    session = ort.InferenceSession(str(v6_onnx), providers=["CPUExecutionProvider"])

    rng = np.random.RandomState(42)
    test_batch = rng.randn(4, 3, 224, 224).astype(np.float32)

    with torch.no_grad():
        pt_out = model(torch.from_numpy(test_batch)).numpy()

    onnx_out = session.run(["embedding"], {"input": test_batch})[0]

    max_abs_err = np.max(np.abs(pt_out - onnx_out))
    assert max_abs_err < 1e-4, f"PT vs ONNX Parity Violation! Max Abs Error = {max_abs_err:.2e} >= 1e-4"

    for p, o in zip(pt_out, onnx_out):
        cos_sim = compute_cosine_similarity(p, o)
        assert cos_sim > 0.9999, f"Cosine similarity below parity tolerance: {cos_sim}"
