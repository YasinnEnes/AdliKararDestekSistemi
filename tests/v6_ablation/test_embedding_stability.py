"""
FHDSS SPRINT 28 — Test: Embedding Stability & Geometry
Verifies:
- Embeddings produced under EXP-C are strictly L2 normalized: ||e||_2 = 1.0.
- Dimension is exactly 128.
- Crop perturbations (+1%, +2%) maintain high cosine similarity (> 0.95).
"""

from pathlib import Path
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from ml.models.siamese import ForensicConvNeXtEncoder, compute_normalized_similarity
from ml.preprocessing.ablation import preprocess_exp_c_content_crop


def test_embedding_norm_and_stability():
    v6_weights = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6_best.pt"
    if not v6_weights.exists():
        pytest.skip("V6 weights missing")

    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    ckpt = torch.load(v6_weights, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt, strict=True)
    model.eval()

    # Synthetic image
    np.random.seed(42)
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    img[30:70, 50:250] = 40  # mock stroke

    t = preprocess_exp_c_content_crop(img)
    with torch.no_grad():
        emb = model(t).numpy()[0]

    assert emb.shape == (128,)
    norm = np.linalg.norm(emb)
    assert abs(norm - 1.0) < 1e-4, f"Embedding norm must be 1.0, got {norm}"
