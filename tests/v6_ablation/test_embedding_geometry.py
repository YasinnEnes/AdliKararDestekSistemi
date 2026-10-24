"""
FHDSS SPRINT 28 — Test: Embedding Geometry
Verifies non-collapse across feature dimensions.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
from ml.models.siamese import ForensicConvNeXtEncoder
from ml.preprocessing.ablation import preprocess_exp_c_content_crop


def test_embedding_geometry_variance():
    v6_weights = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6_best.pt"
    if not v6_weights.exists():
        pytest.skip("V6 weights missing")

    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(v6_weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=True)
    model.eval()

    embs = []
    # Test on structurally varied patterns (lines, boxes, text strokes)
    for i in range(5):
        img = np.full((100, 300, 3), 255, dtype=np.uint8)
        if i == 0:
            img[40:60, 50:250] = 0  # horizontal bar
        elif i == 1:
            img[20:80, 140:160] = 0  # vertical bar
        elif i == 2:
            img[30:70, 80:220] = 20  # box
        elif i == 3:
            img[10:90, 50:250:10] = 10  # striped
        else:
            img[20:80:5, 40:260:5] = 0  # grid
            
        t = preprocess_exp_c_content_crop(img)
        with torch.no_grad():
            embs.append(model(t).numpy()[0])

    embs_arr = np.array(embs)
    dim_vars = np.var(embs_arr, axis=0)
    # Check that dimensions are active and variance is healthy
    assert np.mean(dim_vars) > 1e-4, f"Dimensional collapse detected: mean var={np.mean(dim_vars)}"
    assert not np.isnan(embs_arr).any(), "NaN found in embeddings"
    assert not np.isinf(embs_arr).any(), "Inf found in embeddings"

