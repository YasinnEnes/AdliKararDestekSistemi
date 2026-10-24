"""
FHDSS SPRINT 28 — Test: Inference Determinism
Verifies:
- Running inference 10x on identical inputs yields bitwise / numerical identical embeddings (difference < 1e-7).
- Preprocessing EXP-C pipeline is strictly deterministic.
"""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from ml.preprocessing.ablation import get_preprocessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def test_preprocessing_determinism():
    preprocessor = get_preprocessor("EXP-C")
    img = Image.new("RGB", (400, 100), color=(255, 255, 255))
    
    # Draw simple ink
    for x in range(50, 350):
        img.putpixel((x, 50), (10, 20, 30))
        
    tensors = [preprocessor(img) for _ in range(10)]
    for i in range(1, 10):
        diff = torch.max(torch.abs(tensors[0] - tensors[i])).item()
        assert diff == 0.0, f"Non-deterministic preprocessing at repetition {i}: diff={diff}"


def test_model_inference_determinism():
    pt_path = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6_best.pt"
    if not pt_path.exists():
        pytest.skip("V6 weights not found")
        
    from ml.models.siamese import ForensicConvNeXtEncoder
    
    device = torch.device("cpu")
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    state = torch.load(pt_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224)
    
    embeddings = []
    with torch.no_grad():
        for _ in range(10):
            emb = model(dummy_input).numpy()
            embeddings.append(emb)
            
    for i in range(1, 10):
        diff = np.max(np.abs(embeddings[0] - embeddings[i]))
        assert diff < 1e-7, f"Non-deterministic model output at rep {i}: max diff={diff}"

