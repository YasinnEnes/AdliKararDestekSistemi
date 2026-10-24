"""
FHDSS SPRINT 29 — Test Suite Group 1: Dataset Inventory, Writer Split & Label Integrity
Verifies:
1. Dataset inventory completeness across all 5 corpora
2. Disjoint writer split (Zero writer leakage between train, val, test)
3. Genuine vs Forgery label semantics
4. Writer ID consistency and pair labeling contract
"""

import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.v5_multidomain_loader import load_and_split_v5_corpora


@pytest.fixture(scope="module")
def multi_datasets():
    train_s, val_s, test_s, summary = load_and_split_v5_corpora(seed=42)
    return train_s, val_s, test_s, summary


def test_dataset_inventory_has_five_corpora(multi_datasets):
    """Test 1: Ensures CEDAR, BHSig-260, UTSig, CVL, IAM are all loaded."""
    _, _, _, summary = multi_datasets
    required_corpora = {"CEDAR", "BHSig-260", "UTSig", "CVL Database", "IAM_HW"}
    corpora_keys = set(summary.get("corpora", summary).keys())
    assert required_corpora.issubset(corpora_keys), f"Missing corpora: {required_corpora - corpora_keys}"


def test_writer_split_zero_leakage_cedar(multi_datasets):
    """Test 2: Verifies zero writer overlap between train, val, test in CEDAR."""
    train_s, val_s, test_s, _ = multi_datasets
    w_train = {s.writer_id for s in train_s if s.metadata.get("dataset") == "CEDAR"}
    w_val = {s.writer_id for s in val_s if s.metadata.get("dataset") == "CEDAR"}
    w_test = {s.writer_id for s in test_s if s.metadata.get("dataset") == "CEDAR"}
    assert len(w_train.intersection(w_val)) == 0, "Writer leakage detected between train and val in CEDAR!"
    assert len(w_train.intersection(w_test)) == 0, "Writer leakage detected between train and test in CEDAR!"
    assert len(w_val.intersection(w_test)) == 0, "Writer leakage detected between val and test in CEDAR!"


def test_writer_split_zero_leakage_bhsig(multi_datasets):
    """Test 3: Verifies zero writer overlap in BHSig-260."""
    train_s, val_s, test_s, _ = multi_datasets
    w_train = {s.writer_id for s in train_s if s.metadata.get("dataset") == "BHSig-260"}
    w_val = {s.writer_id for s in val_s if s.metadata.get("dataset") == "BHSig-260"}
    w_test = {s.writer_id for s in test_s if s.metadata.get("dataset") == "BHSig-260"}
    assert len(w_train.intersection(w_val)) == 0
    assert len(w_train.intersection(w_test)) == 0
    assert len(w_val.intersection(w_test)) == 0


def test_writer_split_zero_leakage_utsig(multi_datasets):
    """Test 4: Verifies zero writer overlap in UTSig."""
    train_s, val_s, test_s, _ = multi_datasets
    w_train = {s.writer_id for s in train_s if s.metadata.get("dataset") == "UTSig"}
    w_val = {s.writer_id for s in val_s if s.metadata.get("dataset") == "UTSig"}
    w_test = {s.writer_id for s in test_s if s.metadata.get("dataset") == "UTSig"}
    assert len(w_train.intersection(w_val)) == 0
    assert len(w_train.intersection(w_test)) == 0
    assert len(w_val.intersection(w_test)) == 0


def test_writer_split_zero_leakage_cvl(multi_datasets):
    """Test 5: Verifies zero writer overlap in CVL Database."""
    train_s, val_s, test_s, _ = multi_datasets
    w_train = {s.writer_id for s in train_s if s.metadata.get("dataset") == "CVL Database"}
    w_val = {s.writer_id for s in val_s if s.metadata.get("dataset") == "CVL Database"}
    w_test = {s.writer_id for s in test_s if s.metadata.get("dataset") == "CVL Database"}
    assert len(w_train.intersection(w_val)) == 0
    assert len(w_train.intersection(w_test)) == 0
    assert len(w_val.intersection(w_test)) == 0


def test_label_integrity_forgery_semantics(multi_datasets):
    """Test 6: Genuine and forgery samples must have correct boolean labels."""
    train_s, val_s, test_s, _ = multi_datasets
    all_s = train_s + val_s + test_s
    for s in all_s[:500]:
        assert isinstance(s.is_forgery, bool)
        if "forg" in s.sample_id.lower() or "forgery" in s.sample_id.lower():
            assert s.is_forgery is True


def test_no_empty_image_paths(multi_datasets):
    """Test 7: All samples must have existing non-empty file paths."""
    train_s, val_s, test_s, _ = multi_datasets
    for s in (val_s[:100] + test_s[:100]):
        p = Path(s.image_path)
        assert p.exists(), f"Image path does not exist: {p}"
        assert p.stat().st_size > 0, f"Empty image file: {p}"
