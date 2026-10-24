"""
FHDSS — Sprint 24: V5 Multi-Domain Forensic Batch Sampler
Supports both Signature Corpora (Genuine + Forgery) and Handwriting Corpora (Genuine only).

Features:
- Configurable dataset weights (CEDAR: 0.28, BHSig: 0.28, UTSig: 0.24, CVL: 0.10, IAM: 0.10)
- Balanced P x K sampling: P writers per batch, 4 samples per writer
  - Signature writer: 2 Genuine + 2 Forgeries
  - Handwriting writer: 4 Genuine samples (no forgeries)
- Prevents any corpus from dominating training
- Deterministic epoch seeding
"""

import random
from typing import Any, Dict, Iterator, List, Optional
from torch.utils.data import Sampler

from ml.training.multidataset_split_validator import parse_namespaced_writer_id


class V5MultiDomainBatchSampler(Sampler[List[int]]):
    """
    V5 Çoklu Alan (İmza + El Yazısı) Yapılandırılmış Batch Örnekleyicisi.
    """

    def __init__(
        self,
        samples: List[Any],
        p_writers: int = 6,
        samples_per_writer: int = 4,
        dataset_weights: Optional[Dict[str, float]] = None,
        num_batches: int = 200,
        seed: int = 42,
    ):
        super().__init__()
        self.samples = samples
        self.p_writers = p_writers
        self.samples_per_writer = samples_per_writer
        self.batch_size = p_writers * samples_per_writer  # 6 x 4 = 24
        self.num_batches = num_batches
        self.seed = seed
        self.epoch = 0

        # İndeksleme: dataset -> writer -> {"gen": [], "forg": []}
        self.dataset_writers: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
        for idx, sample in enumerate(samples):
            w_id = sample.writer_id if hasattr(sample, "writer_id") else sample["writer_id"]
            is_forg = sample.is_forgery if hasattr(sample, "is_forgery") else sample["is_forgery"]
            ds_name, _ = parse_namespaced_writer_id(w_id)

            if ds_name not in self.dataset_writers:
                self.dataset_writers[ds_name] = {}
            if w_id not in self.dataset_writers[ds_name]:
                self.dataset_writers[ds_name][w_id] = {"gen": [], "forg": []}

            if is_forg:
                self.dataset_writers[ds_name][w_id]["forg"].append(idx)
            else:
                self.dataset_writers[ds_name][w_id]["gen"].append(idx)

        # Uygun yazarları belirle:
        # İmza için: en az 1 gen ve 1 forg
        # El yazısı için: en az 2 gen
        self.dataset_eligible_writers: Dict[str, List[str]] = {}
        for ds_name, writers in self.dataset_writers.items():
            eligible = []
            for w, pools in writers.items():
                if len(pools["forg"]) > 0:
                    # İmza korpusu
                    if len(pools["gen"]) >= 1 and len(pools["forg"]) >= 1:
                        eligible.append(w)
                else:
                    # El yazısı korpusu
                    if len(pools["gen"]) >= 2:
                        eligible.append(w)
            self.dataset_eligible_writers[ds_name] = sorted(eligible) if eligible else sorted(list(writers.keys()))

        self.available_datasets = sorted(list(self.dataset_eligible_writers.keys()))
        if not self.available_datasets:
            raise ValueError("Kullanılabilir veri kümesi bulunamadı!")

        # Ağırlıkları normalize et
        default_weights = {
            "CEDAR": 0.28,
            "BHSig-260": 0.28,
            "UTSig": 0.24,
            "CVL Database": 0.10,
            "IAM_HW": 0.10,
        }
        weights_to_use = dataset_weights or default_weights
        total_w = sum(weights_to_use.get(ds, 0.20) for ds in self.available_datasets)
        self.dataset_weights = {
            ds: weights_to_use.get(ds, 0.20) / total_w
            for ds in self.available_datasets
        }

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch * 10007)
        ds_list = list(self.dataset_weights.keys())
        ds_probs = [self.dataset_weights[ds] for ds in ds_list]

        for _ in range(self.num_batches):
            batch_indices: List[int] = []
            # Batch için P adet dataset seç
            chosen_datasets = rng.choices(ds_list, weights=ds_probs, k=self.p_writers)

            for ds in chosen_datasets:
                eligible_writers = self.dataset_eligible_writers[ds]
                chosen_writer = rng.choice(eligible_writers)
                pools = self.dataset_writers[ds][chosen_writer]

                gen_pool = pools["gen"]
                forg_pool = pools["forg"]

                if forg_pool:
                    # İmza: 2 genuine + 2 forgery
                    if len(gen_pool) >= 2:
                        chosen_gen = rng.sample(gen_pool, 2)
                    else:
                        chosen_gen = rng.choices(gen_pool, k=2)

                    if len(forg_pool) >= 2:
                        chosen_forg = rng.sample(forg_pool, 2)
                    else:
                        chosen_forg = rng.choices(forg_pool, k=2)

                    batch_indices.extend(chosen_gen)
                    batch_indices.extend(chosen_forg)
                else:
                    # El yazısı: 4 genuine
                    if len(gen_pool) >= self.samples_per_writer:
                        chosen_gen = rng.sample(gen_pool, self.samples_per_writer)
                    else:
                        chosen_gen = rng.choices(gen_pool, k=self.samples_per_writer)
                    batch_indices.extend(chosen_gen)

            yield batch_indices

    def get_sampling_report(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "p_writers": self.p_writers,
            "samples_per_writer": self.samples_per_writer,
            "weights": self.dataset_weights,
            "eligible_writers": {ds: len(w) for ds, w in self.dataset_eligible_writers.items()},
        }
