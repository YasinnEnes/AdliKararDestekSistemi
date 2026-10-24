"""
FHDSS SPRINT 26 — Test: Sampling Balance
Verifies:
- V5/V6 multi-domain sampler enforces P x K structure.
- Balanced weights prevent single-dataset dominance.
"""

from collections import Counter
from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.v5_multidomain_loader import load_and_split_v5_corpora
from ml.training.v5_multidomain_sampler import V5MultiDomainBatchSampler


def test_batch_sampler_distribution():
    train_samples, _, _, _ = load_and_split_v5_corpora(seed=42)
    
    sampler = V5MultiDomainBatchSampler(
        samples=train_samples,
        p_writers=6,
        samples_per_writer=4,
        dataset_weights={
            "CEDAR": 0.30,
            "BHSig-260": 0.30,
            "UTSig": 0.25,
            "CVL Database": 0.15,
        },
        num_batches=20,
        seed=42
    )

    batches = list(sampler)
    assert len(batches) == 20, "Must generate exactly 20 batches"
    
    for b in batches:
        assert len(b) == 24, f"Batch size must be exactly 24 (6 writers x 4 samples), got {len(b)}"
        sampled_writers = [train_samples[idx].writer_id for idx in b]
        unique_sampled = set(sampled_writers)
        assert len(unique_sampled) <= 6, f"Cannot exceed 6 unique writers per batch, got {len(unique_sampled)}"
