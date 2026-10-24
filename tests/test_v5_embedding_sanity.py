"""
FHDSS SPRINT 25 — Phase 3 Test: Embedding Sanity Check
Validates that V5 embedding vectors are 128D, unit L2-normalized, free of NaN/Inf,
and not collapsed into constant vectors.
"""
import pytest
import numpy as np
import torch
from pathlib import Path

from ml.models.siamese import ForensicConvNeXtEncoder

def test_v5_embedding_properties():
    pt_path = Path("artifacts/weights/convnext_tiny_forensic_multidataset_v5_best.pt")
    assert pt_path.exists(), "V5 PyTorch checkpoint not found"
    
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(pt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    
    inputs = [
        torch.zeros((1, 3, 224, 224), dtype=torch.float32),
        torch.ones((1, 3, 224, 224), dtype=torch.float32),
        torch.randn((1, 3, 224, 224), dtype=torch.float32),
        torch.full((1, 3, 224, 224), 0.5, dtype=torch.float32),
    ]
    
    for x in inputs:
        with torch.no_grad():
            emb = model(x).cpu().numpy()[0]
            
        assert emb.shape == (128,), f"Expected 128D, got {emb.shape}"
        assert not np.isnan(emb).any(), "Embedding contains NaN"
        assert not np.isinf(emb).any(), "Embedding contains Inf"
        
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-4, f"Embedding norm {norm} is not unit L2"
        
        std = np.std(emb)
        assert std > 0.01, f"Embedding standard deviation {std} indicates constant vector"
