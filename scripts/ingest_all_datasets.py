"""
FHDSS - Arşiv Adli Veri Setleri ETL Boru Hattı (Dataset Ingestion & Disjoint-Writer Splitter)

datasets/DS_Downloaded/ altındaki BHSig-260, UTSig, IAM Handwriting vb. arşivleri tespit edip
datasets/raw/ dizinine açan ve yazar bazında sıfır sızıntılı (disjoint-writer) Train/Val/Test
bölütleri ve JSON manifestleri oluşturan boru hattı.
"""

import argparse
import json
import os
import random
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


class ForensicDatasetIngestionPipeline:
    """
    Tüm indirilen adli el yazısı ve imza veri setlerinin otomatik çıkarımı,
    indekslenmesi ve sıfır sızıntılı yazar bölütlemesini gerçekleştiren ETL yöneticisi.
    """

    SUPPORTED_DATASETS = {
        "CEDAR": {"zip_name": "CEDAR.zip", "type": "signature", "language": "english", "folder_name": "CEDAR"},
        "BHSig-260": {"zip_name": "BHSig-260.zip", "type": "signature", "language": "bengali_hindi"},
        "UTSig": {"zip_name": "UTSig.zip", "type": "signature", "language": "persian"},
        "IAM_HW": {"zip_name": "iam_handwriting.zip", "type": "handwriting", "language": "english"},
        "SVC2004": {"zip_name": "SVC2004_task1.zip", "type": "dynamic_trajectory", "language": "english_chinese"},
        "CVL Database": {"zip_name": "CVL Database.zip", "type": "handwriting", "language": "multilingual"},
    }

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.base_dir = base_dir or Path(__file__).resolve().parent.parent
        self.download_dir = self.base_dir / "datasets" / "DS_Downloaded"
        self.raw_dir = self.base_dir / "datasets" / "raw"
        self.splits_dir = self.base_dir / "datasets" / "splits"
        self.processed_dir = self.base_dir / "datasets" / "processed"

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.splits_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def extract_archive(
        self,
        zip_path: Path,
        target_dir: Path,
        dry_run: bool = False,
        max_files: Optional[int] = None,
    ) -> List[str]:
        """
        ZIP arşivindeki dosyaları hedefe açar.
        dry_run=True ise arşiv içeriğini disk alanı tüketmeden listeler.
        """
        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP arşivi bulunamadı: {zip_path}")

        extracted_members: List[str] = []
        target_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            namelist = [n for n in zf.namelist() if not n.endswith("/")]
            if max_files:
                namelist = namelist[:max_files]

            for member in namelist:
                extracted_members.append(member)
                if not dry_run:
                    zf.extract(member, target_dir)

        return extracted_members

    def discover_writers_and_samples(
        self,
        dataset_name: str,
        file_list: List[str],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Dosya adları ve hiyerarşisinden yazar kimliklerini (writer_id) ve
        numune türlerini (genuine vs forgery) veri setine özgü heuristiklerle ayrıştırır.
        """
        import re
        writers: Dict[str, List[Dict[str, Any]]] = {}

        for item in file_list:
            path_obj = Path(item)
            ext = path_obj.suffix.lower()
            if ext not in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".txt", ".dat"]:
                continue

            stem = path_obj.stem
            stem_lower = stem.lower()
            fname_lower = path_obj.name.lower()
            path_lower = str(path_obj).lower()

            # Sahtecilik tespiti
            is_forgery = any(k in fname_lower or k in path_lower for k in [
                "forg", "sahte", "taklit", "fake", "bad", "-f-", "_f_", "opposite hand", "forgery"
            ])

            # Veri setine göre yazar kimliği tespiti
            if dataset_name == "BHSig-260":
                m = re.search(r"([BH]-S-\d+)", stem)
                writer_id = f"BHSig_{m.group(1)}" if m else f"{dataset_name}_W_001"
            elif dataset_name == "CEDAR":
                parts = stem.split("_")
                if len(parts) >= 2 and parts[1].isdigit():
                    writer_id = f"CEDAR_W_{int(parts[1]):03d}"
                else:
                    writer_id = f"{dataset_name}_W_001"
            elif dataset_name == "UTSig":
                w_num = None
                for part in reversed(path_obj.parts[:-1]):
                    if part.isdigit():
                        w_num = int(part)
                        break
                writer_id = f"UTSig_W_{w_num:03d}" if w_num is not None else f"{dataset_name}_W_001"
            elif dataset_name == "CVL Database":
                parts = stem.split("-")
                writer_id = f"CVL_W_{int(parts[0]):04d}" if parts and parts[0].isdigit() else f"{dataset_name}_W_001"
            elif dataset_name == "IAM_HW":
                # iam_words/words/a01/a01-000u/a01-000u-00-00.png
                parts = path_obj.parts
                if len(parts) >= 4 and parts[1] == "words":
                    writer_id = f"IAM_W_{parts[2]}"
                else:
                    writer_id = f"{dataset_name}_W_001"
            elif dataset_name == "SVC2004":
                parts = path_obj.parts
                writer_id = f"SVC_W_{parts[0]}" if parts else f"{dataset_name}_W_001"
            else:
                # Genel fallback
                parent_name = path_obj.parent.name
                if parent_name and any(c.isdigit() for c in parent_name) and parent_name not in ["raw", "signatures"]:
                    writer_id = f"{dataset_name}_{parent_name}"
                else:
                    parts = stem.split("_")
                    if len(parts) >= 2 and any(p.isdigit() for p in parts):
                        num = next((p for p in parts if p.isdigit()), "001")
                        writer_id = f"{dataset_name}_W_{int(num):03d}"
                    else:
                        writer_id = f"{dataset_name}_W_001"

            if writer_id not in writers:
                writers[writer_id] = []

            sample_key = f"{writer_id}_{stem}" if not stem.startswith(writer_id) else stem
            writers[writer_id].append({
                "sample_id": sample_key,
                "relative_path": item,
                "is_forgery": is_forgery,
                "extension": ext,
            })

        return writers

    def split_disjoint_writers(
        self,
        writers: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Yazarları Train / Val / Test kümelerine kesin olarak ayrıştırır.
        Her yazar sadece tek bir kümede yer alabilir (Train ∩ Val ∩ Test = ∅).
        """
        random.seed(self.seed)
        writer_list = sorted(list(writers.keys()))
        random.shuffle(writer_list)

        n_total = len(writer_list)
        n_train = max(1, int(n_total * self.train_ratio))
        n_val = max(1, int(n_total * self.val_ratio)) if n_total >= 3 else 0

        train_writers = writer_list[:n_train]
        val_writers = writer_list[n_train:n_train + n_val]
        test_writers = writer_list[n_train + n_val:]

        if not test_writers and n_total >= 2:
            test_writers = [writer_list[-1]]
            if writer_list[-1] in train_writers:
                train_writers.remove(writer_list[-1])
            elif writer_list[-1] in val_writers:
                val_writers.remove(writer_list[-1])

        def gather_samples(w_list: List[str]) -> List[Dict[str, Any]]:
            samples = []
            for w in w_list:
                for s in writers.get(w, []):
                    s_copy = dict(s)
                    s_copy["writer_id"] = w
                    samples.append(s_copy)
            return samples

        return {
            "summary": {
                "total_writers": n_total,
                "train_writers_count": len(train_writers),
                "val_writers_count": len(val_writers),
                "test_writers_count": len(test_writers),
                "train_samples_count": len(gather_samples(train_writers)),
                "val_samples_count": len(gather_samples(val_writers)),
                "test_samples_count": len(gather_samples(test_writers)),
            },
            "writers": {
                "train": train_writers,
                "val": val_writers,
                "test": test_writers,
            },
            "splits": {
                "train": gather_samples(train_writers),
                "val": gather_samples(val_writers),
                "test": gather_samples(test_writers),
            },
        }

    def process_dataset(
        self,
        dataset_name: str,
        dry_run: bool = False,
        sample_limit: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """Tek bir veri setini arşivden okur, açar ve bölütleme manifestini kaydeder."""
        meta = self.SUPPORTED_DATASETS.get(dataset_name)
        if not meta:
            print(f"[-] Desteklenmeyen veri seti: {dataset_name}")
            return None

        # ZIP dosyasını veya mevcut açılmış klasörü ara
        zip_path = self.download_dir / dataset_name / meta["zip_name"]
        if not zip_path.exists():
            zip_path = self.download_dir / meta["zip_name"]

        target_raw = self.raw_dir / dataset_name

        if zip_path.exists():
            print(f"[+] {dataset_name} işleniyor... (Kaynak: {zip_path.name})")
            file_list = self.extract_archive(
                zip_path=zip_path,
                target_dir=target_raw,
                dry_run=dry_run,
                max_files=sample_limit,
            )
        elif (self.download_dir / dataset_name).is_dir():
            print(f"[+] {dataset_name} mevcut klasörden taranıyor: {self.download_dir / dataset_name}")
            source_dir = self.download_dir / dataset_name
            file_list = [str(p.relative_to(source_dir)) for p in source_dir.rglob("*.*") if p.is_file()]
            if sample_limit:
                file_list = file_list[:sample_limit]
        else:
            print(f"[-] {dataset_name} ZIP dosyası veya klasörü bulunamadı.")
            return None


        writers = self.discover_writers_and_samples(dataset_name, file_list)
        split_result = self.split_disjoint_writers(writers)

        # Manifest JSON kaydet
        manifest_path = self.splits_dir / f"{dataset_name.lower().replace(' ', '_')}_split.json"
        manifest_payload = {
            "dataset_name": dataset_name,
            "metadata": meta,
            "dry_run": dry_run,
            "summary": split_result["summary"],
            "writers": split_result["writers"],
            "splits": split_result["splits"],
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_payload, f, indent=2, ensure_ascii=False)

        print(f"[OK] {dataset_name} tamamlandı -> Manifest: {manifest_path.name}")
        print(f"    Yazarlar: Toplam {split_result['summary']['total_writers']} (Train: {len(split_result['writers']['train'])}, Val: {len(split_result['writers']['val'])}, Test: {len(split_result['writers']['test'])})")

        return manifest_payload

    def run_all(self, dry_run: bool = True, sample_limit: Optional[int] = 500) -> Dict[str, Any]:
        """Tüm desteklenen arşiv veri setlerini sırayla işler."""
        results = {}
        for ds_name in self.SUPPORTED_DATASETS:
            res = self.process_dataset(ds_name, dry_run=dry_run, sample_limit=sample_limit)
            if res:
                results[ds_name] = res["summary"]

        return results


def main():
    parser = argparse.ArgumentParser(description="FHDSS Adli Veri Setleri ETL Boru Hattı")
    parser.add_argument("--extract-all", action="store_true", help="Arşivleri gerçekten diske aç (varsayılan: dry-run)")
    parser.add_argument("--sample-limit", type=int, default=None, help="İşlenecek maksimum dosya adedi (hızlı test için)")
    parser.add_argument("--dataset", type=str, default=None, help="Yalnızca belirtilen veri setini işle (örn: UTSig, BHSig-260)")
    args = parser.parse_args()

    pipeline = ForensicDatasetIngestionPipeline()
    is_dry = not args.extract_all

    if args.dataset:
        pipeline.process_dataset(args.dataset, dry_run=is_dry, sample_limit=args.sample_limit)
    else:
        pipeline.run_all(dry_run=is_dry, sample_limit=args.sample_limit)


if __name__ == "__main__":
    main()
