"""
FHDSS SPRINT 25 — Phase 2 Test: PyTorch vs ONNX Forensic Parity
Ensures V5 ONNX model output matches PyTorch model within strict floating point
tolerance (Max Absolute Error < 1e-4, MAE < 1e-5, Cosine > 0.99999).
"""
import pytest
import numpy as np
import torch
from pathlib import Path
import onnxruntime as ort

from ml.models.siamese import ForensicConvNeXtEncoder

def test_v5_pytorch_onnx_parity():
    pt_path = Path("artifacts/weights/convnext_tiny_forensic_multidataset_v5_best.pt")
    onnx_path = Path("artifacts/weights/convnext_tiny_forensic_multidataset_v5.onnx")
    
    assert pt_path.exists(), "V5 PyTorch checkpoint not found"
    assert onnx_path.exists(), "V5 ONNX model not found"
    
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    
    # 10 deterministic test inputs
    rng = np.random.RandomState(42)
    inputs_np = rng.randn(10, 3, 224, 224).astype(np.float32)
    inputs_pt = torch.from_numpy(inputs_np)
    
    with torch.no_grad():
        out_pt = model(inputs_pt).cpu().numpy()
        
    out_onnx = session.run(["embedding"], {"input": inputs_np})[0]
    
    assert out_pt.shape == (10, 128)
    assert out_onnx.shape == (10, 128)
    
    abs_diff = np.abs(out_pt - out_onnx)
    max_err = float(np.max(abs_diff))
    mae = float(np.mean(abs_diff))
    rmse = float(np.sqrt(np.mean(abs_diff ** 2)))
    
    cos_sims = np.sum(out_pt * out_onnx, axis=1)
    min_cos = float(np.min(cos_sims))
    
    assert max_err < 1e-4, f"Max error {max_err} exceeds 1e-4"
    assert mae < 1e-5, f"MAE {mae} exceeds 1e-5"
    assert rmse < 1e-5, f"RMSE {rmse} exceeds 1e-5"
    assert min_cos >= 0.99999, f"Min cosine similarity {min_cos} below 0.99999"
