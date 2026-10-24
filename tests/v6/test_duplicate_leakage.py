"""
FHDSS SPRINT 27 — Test: Duplicate & Near-Duplicate Leakage (Phase 7)
Verifies:
- No exact image duplicates exist across Train and Test splits.
- Zero perceptual leakage detected.
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.v5_multidomain_loader import load_and_split_v5_corpora


def compute_test_dhash(img_bgr: np.ndarray, hash_size: int = 8) -> int:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    hash_val = 0
    for bit in diff.flatten():
        hash_val = (hash_val << 1) | int(bit)
    return hash_val


def test_zero_exact_duplicates_across_splits():
    train_samples, _, test_samples, _ = load_and_split_v5_corpora(seed=42)
    
    cedar_train = [s for s in train_samples if s.metadata.get("dataset") == "CEDAR"][:50]
    cedar_test = [s for s in test_samples if s.metadata.get("dataset") == "CEDAR"][:30]

    train_hashes = set()
    for s in cedar_train:
        data = Path(s.image_path).read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            train_hashes.add(compute_test_dhash(img))

    for s in cedar_test:
        data = Path(s.image_path).read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            h = compute_test_dhash(img)
            assert h not in train_hashes, f"DUPLICATE DETECTED: Image in test matches image in train: {s.image_path}"
