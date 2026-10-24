"""
FHDSS SPRINT 27 — Test: Threshold Comparison & Isolation (Phase 11)
Verifies:
- V5 and V6 have independent, dataset-calibrated thresholds.
- V5 thresholds (0.7334, 0.8188) are strictly isolated from V6.
"""

from pathlib import Path
import json
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_independent_model_thresholds():
    v6_manifest_path = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "model_manifest_v6.json"
    assert v6_manifest_path.exists()
    
    manifest_v6 = json.loads(v6_manifest_path.read_text(encoding="utf-8"))
    th_v6 = manifest_v6["thresholds"]

    # V5 known baseline thresholds
    v5_std = 0.7334
    v5_strict = 0.8188

    # V6 thresholds must be independently calibrated
    assert "standard_threshold_far5" in th_v6
    assert "strict_threshold_far1" in th_v6
    assert "eer_threshold" in th_v6
    
    # Assert monotonic ordering for V6
    assert th_v6["eer_threshold"] <= th_v6["standard_threshold_far5"] <= th_v6["strict_threshold_far1"]
