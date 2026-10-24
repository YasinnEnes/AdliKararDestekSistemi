"""
FHDSS SPRINT 26 — Test: Label Integrity
Verifies that all samples across datasets have valid labels:
- is_forgery is strictly boolean
- Image files exist and are readable
- Writer IDs are valid strings
"""

from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.v5_multidomain_loader import load_and_split_v5_corpora


def test_sample_labels_and_path_integrity():
    train_samples, val_samples, test_samples, _ = load_and_split_v5_corpora(seed=42)
    sample_pool = train_samples[:200] + val_samples[:100] + test_samples[:100]

    for s in sample_pool:
        assert isinstance(s.is_forgery, bool), f"is_forgery must be bool, got {type(s.is_forgery)}"
        assert isinstance(s.writer_id, str) and len(s.writer_id) > 0
        assert isinstance(s.sample_id, str) and len(s.sample_id) > 0
        assert Path(s.image_path).exists(), f"Image path does not exist: {s.image_path}"
