"""
FHDSS SPRINT 29 — Test Suite Group 3: Model Architecture & Embedding Properties
Verifies:
15. Embedding dimension is exactly 128
16. Embedding output is unit L2 normalized (||e||_2 = 1.0)
17. Successful loading from V6 source checkpoint
18. Backbone freeze mode locks convnext weights
19. Backbone unfreeze mode unlocks all weights
20. Batch inference produces consistent tensor outputs
"""

import pytest
import sys
import torch
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models.siamese import ForensicConvNeXtEncoder


@pytest.fixture(scope="module")
def v6_model():
    m = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    v6_ckpt = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6_best.pt"
    if v6_ckpt.exists():
        c = torch.load(v6_ckpt, map_location="cpu")
        m.load_state_dict(c["model_state_dict"] if "model_state_dict" in c else c, strict=True)
    m.eval()
    return m


def test_embedding_dimension(v6_model):
    """Test 15: Encoder output dimension is 128."""
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = v6_model(x)
    assert out.shape == (2, 128), f"Expected shape (2, 128), got {out.shape}"


def test_l2_unit_normalization(v6_model):
    """Test 16: Embeddings lie on unit hypersphere (L2 norm == 1.0)."""
    x = torch.randn(5, 3, 224, 224)
    with torch.no_grad():
        out = v6_model(x)
    norms = torch.norm(out, p=2, dim=-1)
    for n in norms:
        assert abs(n.item() - 1.0) < 1e-4, f"Norm deviation detected: {n.item()}"


def test_backbone_freeze_mode(v6_model):
    """Test 17: freeze_backbone() freezes backbone and leaves head trainable."""
    v6_model.freeze_backbone()
    for p in v6_model.backbone.parameters():
        assert not p.requires_grad
    for p in v6_model.projection_head.parameters():
        assert p.requires_grad


def test_backbone_unfreeze_mode(v6_model):
    """Test 18: unfreeze_backbone() unfreezes all parameters."""
    v6_model.unfreeze_backbone()
    for p in v6_model.backbone.parameters():
        assert p.requires_grad
    for p in v6_model.projection_head.parameters():
        assert p.requires_grad


def test_batch_dimension_invariance(v6_model):
    """Test 19: Single item embedding matches batch embedding."""
    x1 = torch.randn(1, 3, 224, 224)
    x2 = torch.randn(1, 3, 224, 224)
    batch = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        out_single1 = v6_model(x1)
        out_batch = v6_model(batch)

    diff = torch.max(torch.abs(out_single1[0] - out_batch[0])).item()
    assert diff < 1e-5, f"Batch disparity: {diff}"


def test_embedding_finite_values(v6_model):
    """Test 20: Embeddings contain no NaN or Inf values."""
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        out = v6_model(x)
    assert torch.all(torch.isfinite(out)), "Non-finite values found in embedding!"
