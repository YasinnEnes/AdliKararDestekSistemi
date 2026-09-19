"""
FHDSS - Dataset Split & Writer Disjointness Validator (Sprint 18.4)

Train / Validation / Test kümeleri arasında adli yazar sızıntısını (writer leakage)
matematiksel ve teknik olarak denetleyen, Train ∩ Val = ∅ ve Train ∩ Test = ∅
şartını mutlak olarak garanti eden doğrulayıcı araç.
"""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union
from ml.datasets.forensic_dataset import ForensicSample


def _extract_writer_ids(items: Iterable[Any]) -> Set[str]:
    """
    ForensicSample, dict veya str formatındaki verilerden tekil writer_id kümesini çıkarır.
    """
    writer_ids = set()
    for item in items:
        if isinstance(item, str):
            writer_ids.add(item)
        elif isinstance(item, ForensicSample):
            writer_ids.add(item.writer_id)
        elif isinstance(item, dict):
            w_id = item.get("writer_id") or item.get("writer") or item.get("author")
            if w_id:
                writer_ids.add(str(w_id))
            else:
                raise ValueError(f"Geçersiz veri formatı: writer_id anahtarı bulunamadı: {item}")
        else:
            raise TypeError(f"Desteklenmeyen eleman türü: {type(item)}")
    return writer_ids


def detect_writer_overlap(
    train_data: Iterable[Any],
    val_data: Iterable[Any],
    test_data: Optional[Iterable[Any]] = None,
) -> Dict[str, Set[str]]:
    """
    Train, Validation ve Test kümeleri arasındaki yazar kesişimlerini (overlap) tespit eder.
    
    Returns:
        {
            "train_val_overlap": Set[str],
            "train_test_overlap": Set[str],
            "val_test_overlap": Set[str],
            "total_leaked_writers": Set[str]
        }
    """
    train_writers = _extract_writer_ids(train_data)
    val_writers = _extract_writer_ids(val_data)
    test_writers = _extract_writer_ids(test_data) if test_data is not None else set()

    train_val = train_writers.intersection(val_writers)
    train_test = train_writers.intersection(test_writers)
    val_test = val_writers.intersection(test_writers)
    all_leaked = train_val.union(train_test).union(val_test)

    return {
        "train_val_overlap": train_val,
        "train_test_overlap": train_test,
        "val_test_overlap": val_test,
        "total_leaked_writers": all_leaked,
    }


def validate_writer_disjointness(
    train_data: Iterable[Any],
    val_data: Iterable[Any],
    test_data: Optional[Iterable[Any]] = None,
) -> bool:
    """
    Kümeler arasında yazar sızıntısı olup olmadığını kesin olarak doğrular.
    Herhangi bir kesişim tespit edilirse ValueError fırlatır.
    
    Returns:
        True: Sızıntı yok, ayrık (disjoint) yapıda.
    Raises:
        ValueError: Sızıntı tespit edildiğinde örtüşen yazarları belirten hata fırlatır.
    """
    overlaps = detect_writer_overlap(train_data, val_data, test_data)
    leaked = overlaps["total_leaked_writers"]

    if leaked:
        details = []
        if overlaps["train_val_overlap"]:
            details.append(f"Train-Val kesişimi ({len(overlaps['train_val_overlap'])} yazar): {sorted(list(overlaps['train_val_overlap']))[:5]}")
        if overlaps["train_test_overlap"]:
            details.append(f"Train-Test kesişimi ({len(overlaps['train_test_overlap'])} yazar): {sorted(list(overlaps['train_test_overlap']))[:5]}")
        if overlaps["val_test_overlap"]:
            details.append(f"Val-Test kesişimi ({len(overlaps['val_test_overlap'])} yazar): {sorted(list(overlaps['val_test_overlap']))[:5]}")
        
        raise ValueError(
            f"KRİTİK HATA: Adli yazar sızıntısı (Writer Leakage) tespit edildi! "
            f"Toplam {len(leaked)} örtüşen yazar bulundu.\nDetaylar:\n" + "\n".join(details)
        )

    return True


def generate_split_report(
    train_data: Iterable[Any],
    val_data: Iterable[Any],
    test_data: Optional[Iterable[Any]] = None,
) -> Dict[str, Any]:
    """
    Kümelerin yazar ve numune istatistiklerini içeren detaylı ayrım raporu üretir.
    """
    train_writers = _extract_writer_ids(train_data)
    val_writers = _extract_writer_ids(val_data)
    test_writers = _extract_writer_ids(test_data) if test_data is not None else set()

    overlaps = detect_writer_overlap(train_data, val_data, test_data)
    is_disjoint = len(overlaps["total_leaked_writers"]) == 0

    train_count = len(list(train_data))
    val_count = len(list(val_data))
    test_count = len(list(test_data)) if test_data is not None else 0
    total_samples = train_count + val_count + test_count
    all_writers = train_writers.union(val_writers).union(test_writers)

    return {
        "status": "PASSED" if is_disjoint else "FAILED",
        "is_disjoint": is_disjoint,
        "total_unique_writers": len(all_writers),
        "total_samples": total_samples,
        "splits": {
            "train": {
                "writer_count": len(train_writers),
                "sample_count": train_count,
                "writer_ratio": round(len(train_writers) / max(1, len(all_writers)), 3),
            },
            "val": {
                "writer_count": len(val_writers),
                "sample_count": val_count,
                "writer_ratio": round(len(val_writers) / max(1, len(all_writers)), 3),
            },
            "test": {
                "writer_count": len(test_writers),
                "sample_count": test_count,
                "writer_ratio": round(len(test_writers) / max(1, len(all_writers)), 3),
            },
        },
        "overlaps": {
            "train_val_count": len(overlaps["train_val_overlap"]),
            "train_test_count": len(overlaps["train_test_overlap"]),
            "val_test_count": len(overlaps["val_test_overlap"]),
            "total_leaked_count": len(overlaps["total_leaked_writers"]),
            "leaked_writers": sorted(list(overlaps["total_leaked_writers"])),
        },
    }
