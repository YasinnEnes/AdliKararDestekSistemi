"""
FHDSS — Multi-Dataset Structured Batch Sampler (Sprint 22)

Çoklu veri kümesi (CEDAR, BHSig-260, UTSig) eğitimlerinde:
1. Veri kümesi dengesi (Dataset Weighting): Hiçbir veri kümesi sırf daha fazla
   örnek içerdiği için modeli domine etmez.
2. Adli Yapılandırılmış Batch (P x K): Her yazar için tam K_genuine ve K_forgery.
3. Deterministik epoch bazlı örnekleme.
"""

import random
from typing import Any, Dict, Iterator, List, Optional
from torch.utils.data import Sampler

from ml.training.multidataset_split_validator import parse_namespaced_writer_id


class MultiDatasetForensicBatchSampler(Sampler[List[int]]):
    """
    Çoklu veri kümesi adli yapılandırılmış batch örnekleyicisi.
    
    Özellikler:
    - P adet yazar seçilir (varsayılan: 4 veya 6 yazar).
    - Her batch içerisinde veri kümeleri (CEDAR, BHSig-260, UTSig) ağırlıklandırılmış
      olarak temsil edilir (örneğin eşit ağırlık: 1/3, 1/3, 1/3).
    - Her yazar için K_genuine ve K_forgery örneği seçilir.
    - Hiçbir veri kümesi büyüklüğünden ötürü baskın gelmez.
    """

    def __init__(
        self,
        samples: List[Any],
        p_writers: int = 4,
        genuines_per_writer: int = 2,
        forgeries_per_writer: int = 2,
        dataset_weights: Optional[Dict[str, float]] = None,
        num_batches: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.samples = samples
        self.p_writers = p_writers
        self.genuines_per_writer = genuines_per_writer
        self.forgeries_per_writer = forgeries_per_writer
        self.batch_size = p_writers * (genuines_per_writer + forgeries_per_writer)
        self.seed = seed
        self.epoch = 0

        # 1. Numuneleri veri kümesi ve yazar bazında indeksle
        # self.dataset_writers[dataset_name][writer_id] = {"gen": [indices], "forg": [indices]}
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

        # 2. Uygun yazarları filtrele (en az 1 genuine ve 1 forgery içerenler)
        self.dataset_eligible_writers: Dict[str, List[str]] = {}
        for ds_name, writers in self.dataset_writers.items():
            eligible = [
                w for w, pools in writers.items()
                if len(pools["gen"]) >= 1 and len(pools["forg"]) >= 1
            ]
            self.dataset_eligible_writers[ds_name] = sorted(eligible) if eligible else sorted(list(writers.keys()))

        self.available_datasets = sorted(list(self.dataset_eligible_writers.keys()))
        if not self.available_datasets:
            raise ValueError("Örnekleyicide kullanılabilir veri kümesi bulunamadı!")

        # 3. Ağırlıkları normalize et
        if dataset_weights is None:
            # Eşit temsil
            w_val = 1.0 / len(self.available_datasets)
            self.dataset_weights = {ds: w_val for ds in self.available_datasets}
        else:
            total_w = sum(dataset_weights.get(ds, 0.0) for ds in self.available_datasets)
            if total_w <= 0:
                raise ValueError(f"Geçersiz veri kümesi ağırlıkları toplamı: {total_w}")
            self.dataset_weights = {
                ds: dataset_weights.get(ds, 0.0) / total_w
                for ds in self.available_datasets
            }

        # 4. Batch sayısı hesabı
        if num_batches is not None:
            self.num_batches = num_batches
        else:
            total_gen = sum(
                len(pools["gen"])
                for ds in self.dataset_writers.values()
                for pools in ds.values()
            )
            self.num_batches = max(10, total_gen // (self.p_writers * self.genuines_per_writer))

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def get_sampling_report(self) -> Dict[str, Any]:
        """Ağırlıklar, yazar ve numune istatistiklerini raporlar."""
        breakdown = {}
        for ds in self.available_datasets:
            breakdown[ds] = {
                "weight": round(self.dataset_weights[ds], 4),
                "total_writers": len(self.dataset_writers[ds]),
                "eligible_writers": len(self.dataset_eligible_writers[ds]),
                "total_genuines": sum(len(p["gen"]) for p in self.dataset_writers[ds].values()),
                "total_forgeries": sum(len(p["forg"]) for p in self.dataset_writers[ds].values()),
            }
        return {
            "p_writers": self.p_writers,
            "genuines_per_writer": self.genuines_per_writer,
            "forgeries_per_writer": self.forgeries_per_writer,
            "batch_size": self.batch_size,
            "num_batches": self.num_batches,
            "dataset_weights": self.dataset_weights,
            "datasets": breakdown,
        }

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed + self.epoch * 1000)
        ds_list = self.available_datasets
        ds_probs = [self.dataset_weights[ds] for ds in ds_list]

        for _ in range(self.num_batches):
            batch_indices: List[int] = []

            # 1. Veri kümesi ağırlıklarına göre P adet yazarın hangi veri kümelerinden geleceğini seç
            chosen_datasets = rng.choices(ds_list, weights=ds_probs, k=self.p_writers)

            # 2. Her seçilen veri kümesinden bir yazar seç
            for ds in chosen_datasets:
                eligible = self.dataset_eligible_writers[ds]
                writer = rng.choice(eligible)
                pools = self.dataset_writers[ds][writer]
                gen_pool = pools["gen"]
                forg_pool = pools["forg"]

                # Genuines seç
                if len(gen_pool) >= self.genuines_per_writer:
                    sampled_gen = rng.sample(gen_pool, self.genuines_per_writer)
                else:
                    sampled_gen = rng.choices(gen_pool, k=self.genuines_per_writer)

                # Forgeries seç
                if len(forg_pool) >= self.forgeries_per_writer:
                    sampled_forg = rng.sample(forg_pool, self.forgeries_per_writer)
                elif len(forg_pool) > 0:
                    sampled_forg = rng.choices(forg_pool, k=self.forgeries_per_writer)
                else:
                    # Fallback: sahte yoksa ek genuine seç
                    sampled_forg = rng.choices(gen_pool, k=self.forgeries_per_writer)

                batch_indices.extend(sampled_gen)
                batch_indices.extend(sampled_forg)

            yield batch_indices
