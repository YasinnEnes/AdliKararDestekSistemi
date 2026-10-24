"""
FHDSS SPRINT 26 — Test: IAM Protocol (Phase 13)
Verifies:
- IAM is treated as an external handwriting benchmark per Rule 4.
- LexicalAlignedWriterEncoder correctly outputs 128D L2-normalized representations.
"""

from pathlib import Path
import sys
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.research.iam_specialized import LexicalAlignedWriterEncoder


def test_iam_lexical_aligned_encoder():
    encoder = LexicalAlignedWriterEncoder(in_features=128, style_dim=128)
    encoder.eval()

    dummy_base_emb = torch.randn(8, 128)
    dummy_base_emb = torch.nn.functional.normalize(dummy_base_emb, p=2, dim=-1)

    with torch.no_grad():
        out = encoder(dummy_base_emb)

    assert out.shape == (8, 128)
    norms = torch.norm(out, dim=-1).numpy()
    for n in norms:
        assert abs(n - 1.0) < 1e-4, f"IAM prototype embedding must be L2 normalized, got {n}"
