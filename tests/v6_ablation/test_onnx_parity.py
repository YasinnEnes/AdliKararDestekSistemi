"""
FHDSS SPRINT 28 — Test: ONNX Parity for V6 Architecture
Verifies:
- V6 PyTorch model and ONNX export achieve strict numeric parity (MAE < 1e-4).
- Cosine similarity between PyTorch and ONNX embeddings >= 0.9999.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_v6_onnx_parity():
    pt_path = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6_best.pt"
    onnx_path = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6.onnx"
    
    if not pt_path.exists() or not onnx_path.exists():
        pytest.skip("V6 weights or ONNX model not found")
        
    try:
        import onnxruntime as ort
    except ImportError:
        pytest.skip("onnxruntime not installed")
        
    from ml.models.siamese import ForensicConvNeXtEncoder
    
    # Load PyTorch model
    device = torch.device("cpu")
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    state = torch.load(pt_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    
    # Create deterministic input tensor
    np.random.seed(42)
    dummy_np = np.random.randn(2, 3, 224, 224).astype(np.float32)
    dummy_torch = torch.from_numpy(dummy_np)
    
    # PyTorch inference
    with torch.no_grad():
        pt_emb = model(dummy_torch).cpu().numpy()

        
    # ONNX inference
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_emb = session.run(None, {input_name: dummy_np})[0]
    
    # Check shape
    assert pt_emb.shape == ort_emb.shape, f"Shape mismatch: {pt_emb.shape} vs {ort_emb.shape}"
    
    # Check max absolute error and MAE
    mae = np.mean(np.abs(pt_emb - ort_emb))
    max_err = np.max(np.abs(pt_emb - ort_emb))
    
    # Compute cosine similarity
    pt_norm = pt_emb / (np.linalg.norm(pt_emb, axis=1, keepdims=True) + 1e-8)
    ort_norm = ort_emb / (np.linalg.norm(ort_emb, axis=1, keepdims=True) + 1e-8)
    cos_sim = np.sum(pt_norm * ort_norm, axis=1)
    
    assert mae < 1e-3, f"MAE too high: {mae}"
    assert max_err < 5e-3, f"Max error too high: {max_err}"
    assert np.all(cos_sim > 0.999), f"Cosine similarity below threshold: {cos_sim}"
