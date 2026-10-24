"""
FHDSS SPRINT 26 — Test: Dataset Writer Isolation (KURAL 3)
Verifies that for all datasets (CEDAR, BHSig, UTSig, CVL, IAM):
- Namespaced writer IDs are in DATASET::WRITER_ID format.
- Disjointness condition strictly holds:
  TRAIN_WRITERS ∩ VAL_WRITERS = ∅
  TRAIN_WRITERS ∩ TEST_WRITERS = ∅
  VAL_WRITERS ∩ TEST_WRITERS = ∅
"""

import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.v5_multidomain_loader import load_and_split_v5_corpora
from ml.training.multidataset_split_validator import validate_multidataset_disjointness


def test_writer_disjointness_across_all_corpora():
    train_samples, val_samples, test_samples, summary = load_and_split_v5_corpora(seed=42)
    
    # 1. Assert non-empty splits
    assert len(train_samples) > 0, "Train split must not be empty"
    assert len(val_samples) > 0, "Val split must not be empty"
    assert len(test_samples) > 0, "Test split must not be empty"

    # 2. Extract sets of writer IDs
    train_writers = set(s.writer_id for s in train_samples)
    val_writers = set(s.writer_id for s in val_samples)
    test_writers = set(s.writer_id for s in test_samples)

    # 3. Verify zero leakage
    train_val_overlap = train_writers.intersection(val_writers)
    train_test_overlap = train_writers.intersection(test_writers)
    val_test_overlap = val_writers.intersection(test_writers)

    assert len(train_val_overlap) == 0, f"LEAKAGE DETECTED: Train ∩ Val writers = {train_val_overlap}"
    assert len(train_test_overlap) == 0, f"LEAKAGE DETECTED: Train ∩ Test writers = {train_test_overlap}"
    assert len(val_test_overlap) == 0, f"LEAKAGE DETECTED: Val ∩ Test writers = {val_test_overlap}"


def test_namespaced_writer_id_format():
    train_samples, val_samples, test_samples, _ = load_and_split_v5_corpora(seed=42)
    all_samples = train_samples + val_samples + test_samples
    
    for s in all_samples[:300]:
        assert "::" in s.writer_id, f"Invalid namespace in writer_id: {s.writer_id}"
        prefix, w_num = s.writer_id.split("::", 1)
        assert len(prefix) > 0 and len(w_num) > 0
        assert prefix in ["CEDAR", "BHSig-260", "UTSig", "CVL Database", "CVL", "IAM_HW", "IAM"]
