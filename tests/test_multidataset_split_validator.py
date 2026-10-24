"""
Tests for ml/training/multidataset_split_validator.py (Sprint 22)
"""

import pytest
from ml.datasets.forensic_dataset import ForensicSample
from ml.training.multidataset_split_validator import (
    detect_multidataset_leakage,
    extract_namespaced_writers,
    parse_namespaced_writer_id,
    validate_multidataset_disjointness,
)


def test_namespaced_writer_parsing_valid():
    ds, w = parse_namespaced_writer_id("CEDAR::writer_001")
    assert ds == "CEDAR"
    assert w == "writer_001"

    ds, w = parse_namespaced_writer_id("BHSig-260::bengali_w12")
    assert ds == "BHSig-260"
    assert w == "bengali_w12"

    ds, w = parse_namespaced_writer_id("UTSig::w05")
    assert ds == "UTSig"
    assert w == "w05"


def test_namespaced_writer_parsing_invalid():
    with pytest.raises(ValueError, match="namespace edilmelidir"):
        parse_namespaced_writer_id("writer_001")

    with pytest.raises(ValueError, match="Bilinmeyen veri kümesi"):
        parse_namespaced_writer_id("RANDOM_CORPUS::writer_001")


def test_multidataset_disjointness_clean():
    train_samples = [
        ForensicSample("s1", "CEDAR::w1", None, False),
        ForensicSample("s2", "CEDAR::w2", None, False),
        ForensicSample("s3", "BHSig-260::w1", None, False),
        ForensicSample("s4", "UTSig::w1", None, False),
    ]
    val_samples = [
        ForensicSample("s5", "CEDAR::w3", None, False),
        ForensicSample("s6", "BHSig-260::w2", None, False),
        ForensicSample("s7", "UTSig::w2", None, False),
    ]
    test_samples = [
        ForensicSample("s8", "CEDAR::w4", None, False),
        ForensicSample("s9", "BHSig-260::w3", None, False),
        ForensicSample("s10", "UTSig::w3", None, False),
    ]

    assert validate_multidataset_disjointness(train_samples, val_samples, test_samples) is True
    report = detect_multidataset_leakage(train_samples, val_samples, test_samples)
    assert report["is_disjoint"] is True
    assert report["total_unique_writers"] == 10
    assert report["dataset_breakdown"]["CEDAR"]["train"] == 2
    assert report["dataset_breakdown"]["CEDAR"]["val"] == 1
    assert report["dataset_breakdown"]["CEDAR"]["test"] == 1


def test_multidataset_disjointness_leakage_detected():
    train_samples = [
        ForensicSample("s1", "CEDAR::w1", None, False),
        ForensicSample("s2", "BHSig-260::w1", None, False),
    ]
    val_samples = [
        ForensicSample("s3", "CEDAR::w1", None, False),  # Leaked!
        ForensicSample("s4", "BHSig-260::w2", None, False),
    ]

    with pytest.raises(ValueError, match="Yazar Sızıntısı"):
        validate_multidataset_disjointness(train_samples, val_samples)
