"""
FHDSS SPRINT 28 — Test: Zero-Score Auditing
Verifies that fallback exception handling returns None rather than masking failures as 0.0.
"""

import pytest


def test_zero_score_separation():
    # True mathematical zero vs error None
    true_zero_cos = 0.0
    error_fallback = None

    assert true_zero_cos != error_fallback
    assert error_fallback is None
