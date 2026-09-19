import math
import os
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


class PairType(str, Enum):
    """Adli Numune Çift Tipleri."""
    GENUINE_GENUINE = "GENUINE_GENUINE"        # Aynı yazarın iki orijinal örneği (Etiket: 1)
    GENUINE_FORGERY = "GENUINE_FORGERY"        # Orijinal vs Yetenekli Taklit / Sahtecilik (Etiket: 0)
    DIFFERENT_WRITERS = "DIFFERENT_WRITERS"    # İki farklı yazarın orijinal örnekleri (Etiket: 0)


@dataclass
class ForensicSample:
    """Tekil Adli Belge / İmza Numunesi Veri Yapısı."""
    sample_id: str
    writer_id: str
    image_path: Optional[str] = None
    image_data: Optional[np.ndarray] = None
    is_forgery: bool = False
    metadata: Optional[Dict[str, Any]] = None


# ==============================================================================
# 1. Adli Görüntü Dönüşümleri ve Deterministik Veri Artırma
# ==============================================================================

from ml.preprocessing.contract import IMAGENET_MEAN, IMAGENET_STD, pad_to_square, to_rgb


class ForensicAugmentation:
    """
    Adli El Yazısı ve İmza için Biyomekanik Yapıyı Koruyan Veri Artırma Katmanı.
    Sprint 17.4: Ortak Preprocessing Sözleşmesi ile Tam Uyum (Aspect Ratio Preservation + ImageNet Norm).
    """
    def __init__(self, is_training: bool = True, target_size: Tuple[int, int] = (224, 224)):
        self.is_training = is_training
        self.target_size = target_size
        self.normalizer = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image: np.ndarray) -> torch.Tensor:
        """
        NumPy BGR/Grayscale görseli 3-kanallı PyTorch tensörüne `(3, H, W)` dönüştürür.
        """
        if image is None or image.size == 0:
            raise ValueError("Boş veya geçersiz görsel verisi.")

        # 1. Kesin RGB Kanal Dönüşümü
        img_rgb = to_rgb(image, is_bgr=True)

        h, w = img_rgb.shape[:2]

        if self.is_training:
            # Hafif rotasyon [-5, +5 derece]
            angle = random.uniform(-5.0, 5.0)
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img_rgb = cv2.warpAffine(
                img_rgb, M, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )

            # Hafif kontrast ve parlaklık dalgalanması
            alpha = random.uniform(0.90, 1.10)
            beta = random.uniform(-10, 10)
            img_rgb = np.clip(alpha * img_rgb + beta, 0, 255).astype(np.uint8)

        # 2. En-boy oranını koruyarak kareye tamamlama (Kısa kenarı beyaz dolgulama)
        img_padded = pad_to_square(img_rgb, background_value=(255, 255, 255))

        # 3. Hedef boyuta yeniden boyutlandır
        img_resized = cv2.resize(img_padded, self.target_size, interpolation=cv2.INTER_AREA)

        # 4. Normalizasyon: [0, 255] -> [0.0, 1.0] -> ImageNet standart ortalama/sapma
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        tensor = self.normalizer(tensor)

        return tensor



# ==============================================================================
# 2. PyTorch Tekil Numune Dataset (Metrik Öğrenme: ArcFace / SupCon)
# ==============================================================================

class ForensicWriterDataset(Dataset):
    """
    Yazar sınıflandırma ve metrik öğrenme (SupCon / ArcFace) için PyTorch Dataset.
    """
    def __init__(
        self,
        samples: List[ForensicSample],
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    ):
        self.samples = samples
        self.transform = transform or ForensicAugmentation(is_training=False)

        # Yazarları tamsayı etiketlere eşle
        unique_writers = sorted(list(set(s.writer_id for s in samples)))
        self.writer_to_label = {w_id: idx for idx, w_id in enumerate(unique_writers)}
        self.label_to_writer = {idx: w_id for w_id, idx in self.writer_to_label.items()}
        self.num_classes = len(unique_writers)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, sample: ForensicSample) -> np.ndarray:
        if sample.image_data is not None:
            return sample.image_data
        if sample.image_path and os.path.exists(sample.image_path):
            try:
                # Windows Unicode / Türkçe karakterli yol desteği
                data = Path(sample.image_path).read_bytes()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception:
                pass
            img = cv2.imread(sample.image_path)
            if img is not None:
                return img
        raise ValueError(f"Görsel yüklenemedi: {sample.sample_id}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        image_np = self._load_image(sample)
        tensor = self.transform(image_np)
        label = self.writer_to_label[sample.writer_id]
        return tensor, label, sample.sample_id


# ==============================================================================
# 3. PyTorch Adli Çift Dataset (Siamese / Triplet / Verification)
# ==============================================================================

class ForensicPairDataset(Dataset):
    """
    Adli karşılaştırmalı inceleme için Pozitif ve Negatif çiftler üreten PyTorch Dataset.
    """
    def __init__(
        self,
        pairs: List[Tuple[ForensicSample, ForensicSample, float, PairType]],
        transform: Optional[Callable[[np.ndarray], torch.Tensor]] = None,
    ):
        self.pairs = pairs
        self.transform = transform or ForensicAugmentation(is_training=False)

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_image(self, sample: ForensicSample) -> np.ndarray:
        if sample.image_data is not None:
            return sample.image_data
        if sample.image_path and os.path.exists(sample.image_path):
            try:
                data = Path(sample.image_path).read_bytes()
                img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
            except Exception:
                pass
            img = cv2.imread(sample.image_path)
            if img is not None:
                return img
        raise ValueError(f"Görsel yüklenemedi: {sample.sample_id}")

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        sample_a, sample_b, label, pair_type = self.pairs[idx]

        img_a = self._load_image(sample_a)
        img_b = self._load_image(sample_b)

        tensor_a = self.transform(img_a)
        tensor_b = self.transform(img_b)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return tensor_a, tensor_b, label_tensor, pair_type.value


# ==============================================================================
# 4. Yazar Bazlı Sıfır Sızıntılı (Disjoint-Writer) Bölütleme ve Çift Üretici
# ==============================================================================

class ForensicDataPipeline:
    """
    Adli Belge ve El Yazısı Veri Boru Hattı.
    
    Özellikler:
    - **Sıfır Veri Sızıntısı (Zero Data Leakage):** Yazarlar Train / Val / Test kümelerine
      kesinlikle bölünerek dağıtılır. Test yazarları eğitimde asla görülmez.
    - Dengeli pozitif (Genuine-Genuine) ve negatif (Genuine-Forgery, Different-Writers) çift üretimi.
    - Sentetik adli örnek üretici (dış veri seti bulunmadığında tam fonksiyonel doğrulama için).
    """

    @staticmethod
    def load_from_directory(data_dir: Union[str, Path]) -> List[ForensicSample]:
        """
        CEDAR, BHSig260, UTSig veya özel klasör yapısındaki adli imza/yazı veri setlerini otomatik yükler.
        Desteklenen yapılar:
        1. CEDAR Formatı: full_org/original_W_S.png ve full_forg/forgeries_W_S.png
        2. Yazar Klasörü Formatı: data_dir/writer_id/genuine_*.png veya forgery_*.png
        3. Düz Dosya Formatı: original_*.png, forgery_*.png
        """
        root = Path(data_dir)
        if not root.exists():
            raise FileNotFoundError(f"Veri dizini bulunamadı: {data_dir}")

        samples: List[ForensicSample] = []

        # 1. CEDAR Formatı Kontrolü (Doğrudan veya Özyinelemeli / Recursive)
        org_dirs = [root / "full_org"] if (root / "full_org").exists() else list(root.rglob("full_org"))
        forg_dirs = [root / "full_forg"] if (root / "full_forg").exists() else list(root.rglob("full_forg"))

        seen_paths = set()
        if org_dirs or forg_dirs:
            for org_dir in org_dirs:
                for p in org_dir.glob("*.*"):
                    if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"] and str(p.resolve()) not in seen_paths:
                        seen_paths.add(str(p.resolve()))
                        parts = p.stem.split("_")
                        w_id = f"YAZAR_{int(parts[1]):03d}" if len(parts) >= 2 and parts[1].isdigit() else p.stem
                        samples.append(ForensicSample(
                            sample_id=p.stem,
                            writer_id=w_id,
                            image_path=str(p),
                            is_forgery=False,
                            metadata={"dataset": "CEDAR", "filename": p.name},
                        ))
            for forg_dir in forg_dirs:
                for p in forg_dir.glob("*.*"):
                    if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"] and str(p.resolve()) not in seen_paths:
                        seen_paths.add(str(p.resolve()))
                        parts = p.stem.split("_")
                        w_id = f"YAZAR_{int(parts[1]):03d}" if len(parts) >= 2 and parts[1].isdigit() else p.stem
                        samples.append(ForensicSample(
                            sample_id=p.stem,
                            writer_id=w_id,
                            image_path=str(p),
                            is_forgery=True,
                            metadata={"dataset": "CEDAR", "filename": p.name, "forgery_type": "SKILLED"},
                        ))
            if samples:
                return samples

        # 2. Alt Klasörler (Yazar Başına Klasör)
        subdirs = [d for d in root.iterdir() if d.is_dir() and d.name not in ["__pycache__", ".git"]]
        if subdirs:
            for d in subdirs:
                w_id = d.name
                for p in d.glob("*.*"):
                    if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"] and str(p.resolve()) not in seen_paths:
                        seen_paths.add(str(p.resolve()))
                        fname_lower = p.name.lower()
                        is_forg = any(k in fname_lower for k in ["forg", "sahte", "taklit", "fake", "bad"])
                        samples.append(ForensicSample(
                            sample_id=p.stem,
                            writer_id=w_id,
                            image_path=str(p),
                            is_forgery=is_forg,
                            metadata={"filename": p.name},
                        ))
            if samples:
                return samples

        # 3. Düz Liste
        for p in root.glob("*.*"):
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"] and str(p.resolve()) not in seen_paths:
                seen_paths.add(str(p.resolve()))
                fname_lower = p.name.lower()
                is_forg = any(k in fname_lower for k in ["forg", "sahte", "taklit", "fake", "bad"])
                w_id = p.stem.split("_")[0] if "_" in p.stem else "YAZAR_001"
                samples.append(ForensicSample(
                    sample_id=p.stem,
                    writer_id=w_id,
                    image_path=str(p),
                    is_forgery=is_forg,
                    metadata={"filename": p.name},
                ))

        if not samples:
            raise FileNotFoundError(f"Belirtilen veri dizininde ('{data_dir}') desteklenen formatta adli numune bulunamadı.")

        return samples


    @staticmethod
    def generate_synthetic_forensic_corpus(
        num_writers: int = 16,
        samples_per_writer: int = 6,
        forgeries_per_writer: int = 2,
        seed: int = 42,
    ) -> List[ForensicSample]:
        """
        Deterministik ve gerçekçi biyomekanik sentetik adli imza/el yazısı korpusu üretir.
        """
        random.seed(seed)
        np.random.seed(seed)

        corpus: List[ForensicSample] = []

        for w_idx in range(1, num_writers + 1):
            writer_id = f"YAZAR_{w_idx:03d}"
            base_slant = random.uniform(-15.0, 20.0)
            base_loop_count = random.randint(1, 5)

            # 1. Orijinal Örnekler (Genuines)
            for s_idx in range(1, samples_per_writer + 1):
                sample_id = f"{writer_id}_GENUINE_{s_idx:02d}"
                # Sentetik görsel oluştur (300x300 beyaz zemin)
                img = np.full((300, 300), 255, dtype=np.uint8)
                
                # Yazarın karakteristik çizgileri
                slant_offset = base_slant + random.gauss(0.0, 2.0)
                pt1 = (40 + int(slant_offset), 80)
                pt2 = (220, 180)
                cv2.line(img, pt1, pt2, 0, 2, cv2.LINE_AA)
                cv2.putText(
                    img, f"Adli_{writer_id}", (30, 220),
                    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.8, 0, 2, cv2.LINE_AA
                )
                for l in range(base_loop_count):
                    cx = 80 + l * 30 + random.randint(-4, 4)
                    cy = 130 + random.randint(-4, 4)
                    cv2.circle(img, (cx, cy), 14, 0, 2, cv2.LINE_AA)

                corpus.append(ForensicSample(
                    sample_id=sample_id,
                    writer_id=writer_id,
                    image_data=img,
                    is_forgery=False,
                    metadata={"base_slant": base_slant, "loop_count": base_loop_count},
                ))

            # 2. Yetenekli Sahtecilik Örnekleri (Skilled Forgeries - Titreme ve farklı eğim)
            for f_idx in range(1, forgeries_per_writer + 1):
                sample_id = f"{writer_id}_FORGERY_{f_idx:02d}"
                img = np.full((300, 300), 255, dtype=np.uint8)
                # Taklitçi yazarın zorlanması: titrek hatlar ve açı sapması
                f_slant = base_slant + random.choice([-25.0, 25.0])
                pt1 = (40 + int(f_slant), 80)
                pt2 = (220, 180)
                cv2.line(img, pt1, pt2, 0, 2, cv2.LINE_AA)
                cv2.putText(
                    img, f"Adli_{writer_id}", (30, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA
                )
                # Titreme (tremor) simülasyonu
                for t in range(20, 260, 10):
                    offset_y = 150 + int(4.0 * math.sin(t * 0.8))
                    cv2.circle(img, (t, offset_y), 2, 0, -1)

                corpus.append(ForensicSample(
                    sample_id=sample_id,
                    writer_id=writer_id,
                    image_data=img,
                    is_forgery=True,
                    metadata={"target_writer": writer_id, "forgery_type": "SKILLED"},
                ))

        return corpus

    @staticmethod
    def split_by_disjoint_writers(
        samples: List[ForensicSample],
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        test_ratio: float = 0.20,
        seed: int = 42,
    ) -> Tuple[List[ForensicSample], List[ForensicSample], List[ForensicSample]]:
        """
        Yazarları Train, Val ve Test kümelerine sıfır sızıntıyla (disjoint) ayırır.
        """
        random.seed(seed)
        all_writers = sorted(list(set(s.writer_id for s in samples)))
        random.shuffle(all_writers)

        n_total = len(all_writers)
        n_train = max(1, int(n_total * train_ratio))
        n_val = max(1, int(n_total * val_ratio))

        train_writers = set(all_writers[:n_train])
        val_writers = set(all_writers[n_train:n_train + n_val])
        test_writers = set(all_writers[n_train + n_val:])

        # Eğer test kümesi boş kaldıysa en az 1 yazar aktar
        if len(test_writers) == 0 and len(train_writers) > 1:
            moved = train_writers.pop()
            test_writers.add(moved)

        train_samples = [s for s in samples if s.writer_id in train_writers]
        val_samples = [s for s in samples if s.writer_id in val_writers]
        test_samples = [s for s in samples if s.writer_id in test_writers]

        return train_samples, val_samples, test_samples

    @staticmethod
    def create_pairs(
        samples: List[ForensicSample],
        max_pairs_per_writer: int = 15,
        seed: int = 42,
    ) -> List[Tuple[ForensicSample, ForensicSample, float, PairType]]:
        """
        Verilen numunelerden dengeli pozitif ve negatif çiftler üretir.
        """
        random.seed(seed)
        writer_to_genuines: Dict[str, List[ForensicSample]] = {}
        writer_to_forgeries: Dict[str, List[ForensicSample]] = {}

        for s in samples:
            if s.is_forgery:
                writer_to_forgeries.setdefault(s.writer_id, []).append(s)
            else:
                writer_to_genuines.setdefault(s.writer_id, []).append(s)

        pairs: List[Tuple[ForensicSample, ForensicSample, float, PairType]] = []
        all_writers = list(writer_to_genuines.keys())

        for w_id, genuines in writer_to_genuines.items():
            # 1. Pozitif Çiftler (Genuine - Genuine) -> Etiket: 1.0
            pos_count = 0
            for i in range(len(genuines)):
                for j in range(i + 1, len(genuines)):
                    pairs.append((genuines[i], genuines[j], 1.0, PairType.GENUINE_GENUINE))
                    pos_count += 1
                    if pos_count >= max_pairs_per_writer:
                        break
                if pos_count >= max_pairs_per_writer:
                    break

            # 2. Negatif Çiftler - Yetenekli Sahtecilik (Genuine - Forgery) -> Etiket: 0.0
            forgeries = writer_to_forgeries.get(w_id, [])
            for f in forgeries:
                if genuines:
                    g = random.choice(genuines)
                    pairs.append((g, f, 0.0, PairType.GENUINE_FORGERY))

            # 3. Negatif Çiftler - Farklı Yazarlar (Genuine - Different Writer) -> Etiket: 0.0
            other_writers = [o for o in all_writers if o != w_id]
            for _ in range(max(1, pos_count)):
                if other_writers and genuines:
                    other_w = random.choice(other_writers)
                    other_genuines = writer_to_genuines.get(other_w, [])
                    if other_genuines:
                        pairs.append((
                            random.choice(genuines),
                            random.choice(other_genuines),
                            0.0,
                            PairType.DIFFERENT_WRITERS
                        ))

        random.shuffle(pairs)
        return pairs
