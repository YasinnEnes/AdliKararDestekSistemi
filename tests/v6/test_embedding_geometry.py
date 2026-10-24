"""
FHDSS SPRINT 26 — Test: Embedding Geometry (Phase 10)
Verifies:
- All generated embeddings lie strictly on the unit hypersphere: ||e||_2 = 1.0 ± 1e-4.
- Dimension is exactly 128.
- No dimensional collapse (variance across dimensions > 1e-4).
"""

from pathlib import Path
import sys
import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.siamese import ForensicConvNeXtEncoder


def test_embedding_normalization_and_dimensionality():
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    model.eval()

    dummy_input = torch.randn(10, 3, 224, 224)
    with torch.no_grad():
        embs = model(dummy_input).numpy()

    # 1. Output dimension check
    assert embs.shape == (10, 128), f"Expected shape (10, 128), got {embs.shape}"

    # 2. Strict L2 normalization check
    norms = np.linalg.norm(embs, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4, err_msg="Embeddings must be L2 normalized to 1.0")

    # 3. Variance across dimensions (no dimensional collapse)
    dim_vars = np.var(embs, axis=0)
    assert np.mean(dim_vars) > 1e-4, "Dimensional collapse detected: near-zero feature variance"
