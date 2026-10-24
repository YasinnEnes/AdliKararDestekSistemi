"""
FHDSS — Multi-Dataset Loader & Splitter (Sprint 22)

CEDAR, BHSig-260 (Bengali & Hindi) ve UTSig (Farsça) veri kümelerini:
1. İsim alanlı (namespaced) global writer ID'ler ile yükler.
2. Her veri kümesi için yazar düzeyinde (writer-level) %70 Train, %15 Val, %15 Test bölütlemesi yapar.
3. Kümeler arası ve küme içi mutlak sıfır yazar sızıntısı (W_train ∩ W_val ∩ W_test = ∅) sağlar.
"""

from pathlib import Path
import random
from typing import Any, Dict, List, Optional, Tuple

from ml.datasets.forensic_dataset import ForensicSample
from ml.training.multidataset_split_validator import validate_multidataset_disjointness


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def load_cedar_samples(cedar_signatures_dir: Optional[Path] = None) -> List[ForensicSample]:
    """CEDAR veri kümesini isim alanlı (CEDAR::writer_xxx) olarak yükler."""
    root = cedar_signatures_dir or (PROJECT_ROOT / "datasets" / "DS_Downloaded" / "CEDAR" / "CEDAR1" / "signatures")
    samples: List[ForensicSample] = []

    full_org = root / "full_org"
    full_forg = root / "full_forg"

    if full_org.exists():
        for p in sorted(full_org.glob("*.png")):
            parts = p.stem.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                w_id = f"CEDAR::writer_{int(parts[1]):03d}"
                samples.append(ForensicSample(
                    sample_id=f"CEDAR_{p.stem}",
                    writer_id=w_id,
                    image_path=str(p),
                    is_forgery=False,
                    metadata={"dataset": "CEDAR", "filename": p.name, "language": "Latin/English"}
                ))

    if full_forg.exists():
        for p in sorted(full_forg.glob("*.png")):
            parts = p.stem.split("_")
            if len(parts) >= 3 and parts[1].isdigit():
                w_id = f"CEDAR::writer_{int(parts[1]):03d}"
                samples.append(ForensicSample(
                    sample_id=f"CEDAR_{p.stem}",
                    writer_id=w_id,
                    image_path=str(p),
                    is_forgery=True,
                    metadata={"dataset": "CEDAR", "filename": p.name, "language": "Latin/English", "forgery_type": "SKILLED"}
                ))

    return samples


def load_bhsig_samples(bhsig_dir: Optional[Path] = None) -> List[ForensicSample]:
    """BHSig-260 (Bengali + Hindi) veri kümesini isim alanlı olarak yükler."""
    root = bhsig_dir or (PROJECT_ROOT / "datasets" / "processed" / "BHSig-260")
    samples: List[ForensicSample] = []

    for p in sorted(root.rglob("*.tif")):
        fname = p.name
        # Formatlar: B-S-1-G-01.tif veya H-S-011-G-01.tif
        parts = fname.replace(".tif", "").split("-")
        if len(parts) >= 5:
            prefix = parts[0]  # 'B' or 'H'
            w_num = parts[2]
            mode = parts[3]    # 'G' or 'F'
            is_forg = (mode == "F")

            if prefix == "B":
                lang = "Bengali"
                w_id = f"BHSig-260::bengali_w{int(w_num):03d}"
            else:
                lang = "Hindi"
                w_id = f"BHSig-260::hindi_w{int(w_num):03d}"

            samples.append(ForensicSample(
                sample_id=f"BHSig_{p.stem}",
                writer_id=w_id,
                image_path=str(p),
                is_forgery=is_forg,
                metadata={"dataset": "BHSig-260", "filename": fname, "language": lang, "forgery_type": "SKILLED" if is_forg else "GENUINE"}
            ))

    return samples


def load_utsig_samples(utsig_dir: Optional[Path] = None) -> List[ForensicSample]:
    """UTSig (Farsça) veri kümesini isim alanlı ve sahte türü detaylarıyla yükler."""
    root = utsig_dir or (PROJECT_ROOT / "datasets" / "processed" / "UTSig")
    samples: List[ForensicSample] = []

    for p in sorted(root.rglob("*.tif")):
        rel = p.relative_to(root)
        parts = rel.parts
        # Örnekler:
        # UTSig / Genuine / {writer_id} / {idx}.tif
        # UTSig / Forgery / {subtype} / {writer_id} / {idx}.tif
        if "Genuine" in parts:
            gen_idx = parts.index("Genuine")
            if len(parts) > gen_idx + 1:
                w_num = parts[gen_idx + 1]
                w_id = f"UTSig::w{int(w_num):03d}"
                samples.append(ForensicSample(
                    sample_id=f"UTSig_gen_{w_num}_{p.stem}",
                    writer_id=w_id,
                    image_path=str(p),
                    is_forgery=False,
                    metadata={"dataset": "UTSig", "filename": p.name, "language": "Persian", "forgery_type": "GENUINE"}
                ))
        elif "Forgery" in parts:
            forg_idx = parts.index("Forgery")
            if len(parts) > forg_idx + 2:
                subtype = parts[forg_idx + 1]  # 'Skilled', 'Simple', 'Opposite Hand'
                w_num = parts[forg_idx + 2]
                w_id = f"UTSig::w{int(w_num):03d}"
                samples.append(ForensicSample(
                    sample_id=f"UTSig_forg_{subtype}_{w_num}_{p.stem}",
                    writer_id=w_id,
                    image_path=str(p),
                    is_forgery=True,
                    metadata={"dataset": "UTSig", "filename": p.name, "language": "Persian", "forgery_type": subtype}
                ))

    return samples


def split_corpus_by_disjoint_writers(
    samples: List[ForensicSample],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[ForensicSample], List[ForensicSample], List[ForensicSample]]:
    """Tek bir korpusun örneklerini yazar bazında ayrık kümelere böler."""
    writer_to_samples: Dict[str, List[ForensicSample]] = {}
    for s in samples:
        writer_to_samples.setdefault(s.writer_id, []).append(s)

    unique_writers = sorted(list(writer_to_samples.keys()))
    rng = random.Random(seed)
    rng.shuffle(unique_writers)

    n_writers = len(unique_writers)
    n_train = max(1, int(round(n_writers * train_ratio)))
    n_val = max(1, int(round(n_writers * val_ratio)))
    n_test = n_writers - n_train - n_val

    # Test yazarı 0 kalmasın
    if n_test <= 0:
        n_test = max(1, n_writers - n_train - 1)
        n_val = n_writers - n_train - n_test

    train_writers = set(unique_writers[:n_train])
    val_writers = set(unique_writers[n_train:n_train + n_val])
    test_writers = set(unique_writers[n_train + n_val:])

    train_samples = [s for w in train_writers for s in writer_to_samples[w]]
    val_samples = [s for w in val_writers for s in writer_to_samples[w]]
    test_samples = [s for w in test_writers for s in writer_to_samples[w]]

    return train_samples, val_samples, test_samples


def load_and_split_all_corpora(
    seed: int = 42,
) -> Tuple[List[ForensicSample], List[ForensicSample], List[ForensicSample], Dict[str, Any]]:
    """
    CEDAR, BHSig-260 ve UTSig'i yükler, her birini bağımsız yazar-ayrık
    bölütler ve birleştirir.
    
    Returns:
        (all_train_samples, all_val_samples, all_test_samples, summary_dict)
    """
    cedar_samples = load_cedar_samples()
    bhsig_samples = load_bhsig_samples()
    utsig_samples = load_utsig_samples()

    # CEDAR: 38 train / 8 val / 9 test (V2/V3 baseline ile parite)
    c_train, c_val, c_test = split_corpus_by_disjoint_writers(
        cedar_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )
    # BHSig-260
    b_train, b_val, b_test = split_corpus_by_disjoint_writers(
        bhsig_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )
    # UTSig
    u_train, u_val, u_test = split_corpus_by_disjoint_writers(
        utsig_samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )

    all_train = c_train + b_train + u_train
    all_val = c_val + b_val + u_val
    all_test = c_test + b_test + u_test

    # Sıfır yazar sızıntısı doğrulaması
    validate_multidataset_disjointness(all_train, all_val, all_test)

    summary = {
        "status": "ZERO_LEAKAGE_CONFIRMED",
        "datasets": {
            "CEDAR": {
                "total": len(cedar_samples),
                "train_samples": len(c_train),
                "val_samples": len(c_val),
                "test_samples": len(c_test),
                "train_writers": len(set(s.writer_id for s in c_train)),
                "val_writers": len(set(s.writer_id for s in c_val)),
                "test_writers": len(set(s.writer_id for s in c_test)),
            },
            "BHSig-260": {
                "total": len(bhsig_samples),
                "train_samples": len(b_train),
                "val_samples": len(b_val),
                "test_samples": len(b_test),
                "train_writers": len(set(s.writer_id for s in b_train)),
                "val_writers": len(set(s.writer_id for s in b_val)),
                "test_writers": len(set(s.writer_id for s in b_test)),
            },
            "UTSig": {
                "total": len(utsig_samples),
                "train_samples": len(u_train),
                "val_samples": len(u_val),
                "test_samples": len(u_test),
                "train_writers": len(set(s.writer_id for s in u_train)),
                "val_writers": len(set(s.writer_id for s in u_val)),
                "test_writers": len(set(s.writer_id for s in u_test)),
            },
        },
        "total_train_samples": len(all_train),
        "total_val_samples": len(all_val),
        "total_test_samples": len(all_test),
        "total_all_samples": len(all_train) + len(all_val) + len(all_test),
    }

    return all_train, all_val, all_test, summary
