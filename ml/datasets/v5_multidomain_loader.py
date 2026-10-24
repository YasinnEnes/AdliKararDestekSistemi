"""
FHDSS — Sprint 24: V5 Multi-Domain Forensic Dataset Loader & Splitter
Loads and splits 5 real corpora:
1. CEDAR (Latin / English Signatures: Genuine + Skilled Forgery)
2. BHSig-260 (Indic Bengali & Hindi Signatures: Genuine + Skilled Forgery)
3. UTSig (Perso-Arabic Persian Signatures: Genuine + Skilled/Disguise Forgery)
4. CVL Database (Multilingual English/German Handwriting Lines: Same-writer genuine)
5. IAM_HW (English Handwriting Words: Same-writer genuine)

Guarantees:
- Namespaced writer IDs: DATASET::WRITER_ID
- Disjoint writer split per corpus (70% Train, 15% Val, 15% Test)
- Zero writer leakage across all subsets (W_train ∩ W_val ∩ W_test = ∅)
- Test set is strictly frozen and held out
"""

import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.datasets.forensic_dataset import ForensicSample
from ml.training.multidataset_split_validator import validate_multidataset_disjointness


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
                    metadata={"dataset": "CEDAR", "filename": p.name, "language": "Latin/English", "task": "SIGNATURE_VERIFICATION"}
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
                    metadata={"dataset": "CEDAR", "filename": p.name, "language": "Latin/English", "forgery_type": "SKILLED", "task": "SIGNATURE_VERIFICATION"}
                ))

    return samples


def load_bhsig_samples(bhsig_dir: Optional[Path] = None) -> List[ForensicSample]:
    """BHSig-260 (Bengali + Hindi) veri kümesini isim alanlı olarak yükler."""
    root = bhsig_dir or (PROJECT_ROOT / "datasets" / "processed" / "BHSig-260")
    samples: List[ForensicSample] = []

    for p in sorted(root.rglob("*.tif")):
        fname = p.name
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
                metadata={"dataset": "BHSig-260", "filename": fname, "language": lang, "forgery_type": "SKILLED" if is_forg else "GENUINE", "task": "SIGNATURE_VERIFICATION"}
            ))

    return samples


def load_utsig_samples(utsig_dir: Optional[Path] = None) -> List[ForensicSample]:
    """UTSig (Farsça) veri kümesini isim alanlı ve sahte türü detaylarıyla yükler."""
    root = utsig_dir or (PROJECT_ROOT / "datasets" / "processed" / "UTSig")
    samples: List[ForensicSample] = []

    for p in sorted(root.rglob("*.tif")):
        rel = p.relative_to(root)
        parts = rel.parts
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
                    metadata={"dataset": "UTSig", "filename": p.name, "language": "Persian", "forgery_type": "GENUINE", "task": "SIGNATURE_VERIFICATION"}
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
                    metadata={"dataset": "UTSig", "filename": p.name, "language": "Persian", "forgery_type": subtype, "task": "SIGNATURE_VERIFICATION"}
                ))

    return samples


def load_cvl_samples(cvl_dir: Optional[Path] = None) -> List[ForensicSample]:
    """CVL Database (kırpılmış el yazısı satırları) isim alanlı ve is_forgery=False olarak yükler."""
    root = cvl_dir or (PROJECT_ROOT / "datasets" / "processed" / "CVL")
    samples: List[ForensicSample] = []

    if root.exists():
        for p in sorted(root.glob("*.jpg")):
            parts = p.stem.split("-")
            if parts and parts[0].isdigit():
                w_id = f"CVL::writer_{int(parts[0]):04d}"
                samples.append(ForensicSample(
                    sample_id=f"CVL_{p.stem}",
                    writer_id=w_id,
                    image_path=str(p),
                    is_forgery=False,
                    metadata={"dataset": "CVL Database", "filename": p.name, "language": "Multilingual", "task": "HANDWRITING_WRITER_VERIFICATION"}
                ))

    return samples


def load_iam_samples(iam_dir: Optional[Path] = None) -> List[ForensicSample]:
    """IAM_HW (kırpılmış el yazısı kelimeleri) isim alanlı ve is_forgery=False olarak yükler."""
    root = iam_dir or (PROJECT_ROOT / "datasets" / "processed" / "IAM_HW")
    samples: List[ForensicSample] = []

    if root.exists():
        for p in sorted(root.rglob("*.png")):
            parts = p.parts
            if "words" in parts:
                w_idx = parts.index("words") + 1
                if len(parts) > w_idx:
                    w_name = parts[w_idx]
                    w_id = f"IAM_HW::writer_{w_name}"
                    samples.append(ForensicSample(
                        sample_id=f"IAM_{p.stem}",
                        writer_id=w_id,
                        image_path=str(p),
                        is_forgery=False,
                        metadata={"dataset": "IAM_HW", "filename": p.name, "language": "English", "task": "HANDWRITING_WRITER_VERIFICATION"}
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


def load_and_split_v5_corpora(
    seed: int = 42,
) -> Tuple[List[ForensicSample], List[ForensicSample], List[ForensicSample], Dict[str, Any]]:
    """
    Tüm 5 korpusu (CEDAR, BHSig, UTSig, CVL, IAM) yükler, bağımsız yazar-ayrık
    bölütler ve sıfır sızıntıyı doğrular.
    """
    cedar_samples = load_cedar_samples()
    bhsig_samples = load_bhsig_samples()
    utsig_samples = load_utsig_samples()
    cvl_samples = load_cvl_samples()
    iam_samples = load_iam_samples()

    # Split each corpus independently with fixed seed
    c_train, c_val, c_test = split_corpus_by_disjoint_writers(cedar_samples, seed=seed)
    b_train, b_val, b_test = split_corpus_by_disjoint_writers(bhsig_samples, seed=seed)
    u_train, u_val, u_test = split_corpus_by_disjoint_writers(utsig_samples, seed=seed)
    cvl_train, cvl_val, cvl_test = split_corpus_by_disjoint_writers(cvl_samples, seed=seed)
    iam_train, iam_val, iam_test = split_corpus_by_disjoint_writers(iam_samples, seed=seed)

    all_train = c_train + b_train + u_train + cvl_train + iam_train
    all_val = c_val + b_val + u_val + cvl_val + iam_val
    all_test = c_test + b_test + u_test + cvl_test + iam_test

    # Doğrula: Mutlak sıfır yazar sızıntısı
    validate_multidataset_disjointness(all_train, all_val, all_test)

    summary = {
        "status": "ZERO_LEAKAGE_CONFIRMED",
        "random_seed": seed,
        "corpora": {
            "CEDAR": {
                "task": "SIGNATURE_VERIFICATION",
                "total": len(cedar_samples),
                "train_samples": len(c_train),
                "val_samples": len(c_val),
                "test_samples": len(c_test),
                "train_writers": len(set(s.writer_id for s in c_train)),
                "val_writers": len(set(s.writer_id for s in c_val)),
                "test_writers": len(set(s.writer_id for s in c_test)),
            },
            "BHSig-260": {
                "task": "SIGNATURE_VERIFICATION",
                "total": len(bhsig_samples),
                "train_samples": len(b_train),
                "val_samples": len(b_val),
                "test_samples": len(b_test),
                "train_writers": len(set(s.writer_id for s in b_train)),
                "val_writers": len(set(s.writer_id for s in b_val)),
                "test_writers": len(set(s.writer_id for s in b_test)),
            },
            "UTSig": {
                "task": "SIGNATURE_VERIFICATION",
                "total": len(utsig_samples),
                "train_samples": len(u_train),
                "val_samples": len(u_val),
                "test_samples": len(u_test),
                "train_writers": len(set(s.writer_id for s in u_train)),
                "val_writers": len(set(s.writer_id for s in u_val)),
                "test_writers": len(set(s.writer_id for s in u_test)),
            },
            "CVL Database": {
                "task": "HANDWRITING_WRITER_VERIFICATION",
                "total": len(cvl_samples),
                "train_samples": len(cvl_train),
                "val_samples": len(cvl_val),
                "test_samples": len(cvl_test),
                "train_writers": len(set(s.writer_id for s in cvl_train)),
                "val_writers": len(set(s.writer_id for s in cvl_val)),
                "test_writers": len(set(s.writer_id for s in cvl_test)),
            },
            "IAM_HW": {
                "task": "HANDWRITING_WRITER_VERIFICATION",
                "total": len(iam_samples),
                "train_samples": len(iam_train),
                "val_samples": len(iam_val),
                "test_samples": len(iam_test),
                "train_writers": len(set(s.writer_id for s in iam_train)),
                "val_writers": len(set(s.writer_id for s in iam_val)),
                "test_writers": len(set(s.writer_id for s in iam_test)),
            },
        },
        "total_train_samples": len(all_train),
        "total_val_samples": len(all_val),
        "total_test_samples": len(all_test),
        "total_all_samples": len(all_train) + len(all_val) + len(all_test),
        "total_unique_writers": len(set(s.writer_id for s in all_train + all_val + all_test)),
    }

    return all_train, all_val, all_test, summary


if __name__ == "__main__":
    train, val, test, summ = load_and_split_v5_corpora()
    print("V5 Corpora Loaded Successfully:")
    print(json.dumps(summ, indent=2))
