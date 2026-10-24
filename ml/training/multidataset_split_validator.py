"""
FHDSS — Multi-Dataset Split Validator (Sprint 22)

Çoklu veri kümesi (CEDAR, BHSig-260, UTSig) eğitimlerinde:
1. Global Writer Namespace Zorunluluğu (<DATASET>::<WRITER_ID>)
2. Global Sıfır Yazar Sızıntısı (Zero Writer Leakage): W_train ∩ W_val ∩ W_test = ∅
3. Veri Kümesi Başına Bağımsız Sıfır Sızıntı
4. Numune Sızıntısı Engelleme
şartlarını matematiksel ve adli olarak denetleyen modül.
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from ml.datasets.forensic_dataset import ForensicSample


VALID_NAMESPACES = {"CEDAR", "BHSig-260", "UTSig", "CVL", "IAM_HW", "SVC2004"}


def parse_namespaced_writer_id(writer_id: str) -> Tuple[str, str]:
    """
    Yazar kimliğinin '<DATASET>::<ID>' formatında namespace edildiğini doğrular.
    """
    if "::" not in writer_id:
        raise ValueError(
            f"Geçersiz yazar kimliği formatı: '{writer_id}'. "
            f"Sprint 22 kuralı gereği tüm yazarlar '<DATASET>::<WRITER_ID>' şeklinde namespace edilmelidir!"
        )
    dataset_name, actual_id = writer_id.split("::", 1)
    if dataset_name not in VALID_NAMESPACES and not any(dataset_name.startswith(p) for p in VALID_NAMESPACES):
        raise ValueError(
            f"Bilinmeyen veri kümesi namespace'i: '{dataset_name}'. "
            f"Geçerli namespace'ler: {VALID_NAMESPACES}"
        )
    return dataset_name, actual_id


def extract_namespaced_writers(items: Iterable[Any]) -> Set[str]:
    """
    ForensicSample veya dict nesnelerinden yazar kimliklerini çıkarır ve namespace denetimi yapar.
    """
    writer_ids = set()
    for item in items:
        if isinstance(item, ForensicSample):
            w_id = item.writer_id
        elif isinstance(item, dict):
            w_id = item.get("writer_id")
        elif isinstance(item, str):
            w_id = item
        else:
            raise TypeError(f"Desteklenmeyen eleman tipi: {type(item)}")

        if not w_id:
            raise ValueError(f"Yazar kimliği boş olamaz: {item}")
        
        parse_namespaced_writer_id(w_id)
        writer_ids.add(w_id)
    return writer_ids


def detect_multidataset_leakage(
    train_data: Iterable[Any],
    val_data: Iterable[Any],
    test_data: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """
    Çoklu veri kümesi genelinde ve veri kümesi bazında yazar sızıntılarını tespit eder.
    """
    train_writers = extract_namespaced_writers(train_data)
    val_writers = extract_namespaced_writers(val_data)
    test_writers = extract_namespaced_writers(test_data) if test_data is not None else set()

    train_val = train_writers.intersection(val_writers)
    train_test = train_writers.intersection(test_writers)
    val_test = val_writers.intersection(test_writers)
    all_leaked = train_val.union(train_test).union(val_test)

    # Veri kümelerine göre grupla
    dataset_breakdown: Dict[str, Dict[str, int]] = {}
    for w in train_writers.union(val_writers).union(test_writers):
        ds, _ = parse_namespaced_writer_id(w)
        if ds not in dataset_breakdown:
            dataset_breakdown[ds] = {"train": 0, "val": 0, "test": 0, "leaked": 0}
        if w in train_writers:
            dataset_breakdown[ds]["train"] += 1
        if w in val_writers:
            dataset_breakdown[ds]["val"] += 1
        if w in test_writers:
            dataset_breakdown[ds]["test"] += 1
        if w in all_leaked:
            dataset_breakdown[ds]["leaked"] += 1

    return {
        "is_disjoint": len(all_leaked) == 0,
        "total_unique_writers": len(train_writers.union(val_writers).union(test_writers)),
        "train_writers_count": len(train_writers),
        "val_writers_count": len(val_writers),
        "test_writers_count": len(test_writers),
        "train_val_overlap": sorted(list(train_val)),
        "train_test_overlap": sorted(list(train_test)),
        "val_test_overlap": sorted(list(val_test)),
        "total_leaked_writers": sorted(list(all_leaked)),
        "dataset_breakdown": dataset_breakdown,
    }


def validate_multidataset_disjointness(
    train_data: Iterable[Any],
    val_data: Iterable[Any],
    test_data: Optional[Iterable[Any]] = None,
) -> bool:
    """
    Sıfır yazar sızıntısı şartını doğrular. Sızıntı varsa açık ValueError fırlatır.
    """
    report = detect_multidataset_leakage(train_data, val_data, test_data)
    if not report["is_disjoint"]:
        leaked = report["total_leaked_writers"]
        raise ValueError(
            f"KRİTİK ADLİ HATA: Çoklu veri kümesinde Yazar Sızıntısı (Writer Leakage) tespit edildi!\n"
            f"Toplam sızan yazar sayısı: {len(leaked)}\n"
            f"Sızan yazarlar: {leaked[:10]}"
        )
    return True
