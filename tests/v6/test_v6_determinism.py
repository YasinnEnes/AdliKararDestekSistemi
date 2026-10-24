"""
FHDSS SPRINT 26 — Test: V6 Inference Determinism
Verifies:
- Consecutive inferences on identical input produce bit-exact identical outputs (difference == 0.0).
- Eval mode properly disables dropout and freezes batchnorm.
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


def test_v6_deterministic_forward_pass():
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    model.eval()

    torch.manual_seed(1337)
    test_tensor = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        out1 = model(test_tensor).numpy()
        out2 = model(test_tensor).numpy()

    max_diff = np.max(np.abs(out1 - out2))
    assert max_diff == 0.0, f"Non-deterministic inference detected! Max diff = {max_diff}"
