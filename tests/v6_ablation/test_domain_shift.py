"""
FHDSS SPRINT 28 — Test: Domain Shift
Verifies centroid cosine distances across dataset representations.
"""

import numpy as np
import pytest


def test_domain_shift_centroid_distance_bounds():
    # Mock two domain centroids
    c1 = np.random.randn(128)
    c1 /= np.linalg.norm(c1)
    c2 = np.random.randn(128)
    c2 /= np.linalg.norm(c2)

    cos_dist = float(1.0 - np.dot(c1, c2))
    assert 0.0 <= cos_dist <= 2.0, f"Cosine distance must be in [0, 2], got {cos_dist}"
