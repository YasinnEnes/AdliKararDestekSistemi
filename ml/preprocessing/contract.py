"""
FHDSS - Ortak Adli Görüntü Ön İşleme Sözleşmesi (Preprocessing Contract).
Sprint 17.4: Training Preprocessing == Live Inference Preprocessing Sözleşmesi.

Bu modül; model eğitimi, validasyon ve canlı API/servis çıkarımı arasında:
1. Renk formatı (RGB),
2. En-boy oranı korumalı kare dolgulama (aspect ratio preserving padding),
3. (224, 224) hedef boyutlandırma,
4. Standart ImageNet normalizasyonu (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
işlemlerinin tek bir kaynaktan yönetilmesini ve kesin uyumunu garanti eder.
"""

from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def to_rgb(image: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """
    Girdi görüntüsünü kesin olarak 3-kanallı RGB formatına dönüştürür.
    
    Args:
        image: 2D (H, W) gri seviye veya 3D (H, W, C) BGR/RGB/BGRA matris.
        is_bgr: 3 kanallı renkli görüntülerin BGR kaynaklı olup olmadığı (OpenCV default).
    
    Returns:
        3-kanallı RGB uint8 numpy matrisi (H, W, 3).
    """
    if image is None or image.size == 0:
        raise ValueError("Boş veya geçersiz görsel verisi.")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif channels == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if is_bgr else image
    raise ValueError(f"Desteklenmeyen görüntü boyutu/kanal sayısı: {image.shape}")


def pad_to_square(
    image: np.ndarray,
    background_value: Union[int, Tuple[int, int, int]] = 255,
) -> np.ndarray:
    """
    Görüntünün en-boy oranını (aspect ratio) bozmamak için kısa kenarı
    simetrik beyaz (veya verilen arka plan) dolgu ile kareye tamamlar.
    
    Args:
        image: (H, W) veya (H, W, C) formatında görüntü matrisi.
        background_value: Dolgu rengi (varsayılan 255 beyaz).
        
    Returns:
        Kare (max_dim, max_dim) veya (max_dim, max_dim, C) matris.
    """
    h, w = image.shape[:2]
    if h == w:
        return image

    max_dim = max(h, w)
    pad_top = (max_dim - h) // 2
    pad_bottom = max_dim - h - pad_top
    pad_left = (max_dim - w) // 2
    pad_right = max_dim - w - pad_left

    if image.ndim == 3 and image.shape[2] == 3 and isinstance(background_value, (int, float)):
        border_val = (int(background_value), int(background_value), int(background_value))
    elif image.ndim == 3 and image.shape[2] == 4 and isinstance(background_value, (int, float)):
        border_val = (int(background_value), int(background_value), int(background_value), 255)
    elif isinstance(background_value, (int, float)):
        border_val = background_value
    else:
        border_val = tuple(background_value)

    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=border_val,
    )
    return padded



class ForensicPreprocessingContract:
    """
    Eğitim ve Canlı Çıkarım için Ortak Deterministik Ön İşleme Yöneticisi.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        preserve_aspect_ratio: bool = True,
        mean: Optional[list] = None,
        std: Optional[list] = None,
    ):
        self.target_size = target_size
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.mean = mean or IMAGENET_MEAN
        self.std = std or IMAGENET_STD
        self.normalizer = T.Normalize(mean=self.mean, std=self.std)

    def preprocess_image(
        self,
        image: np.ndarray,
        is_bgr: bool = True,
    ) -> torch.Tensor:
        """
        Ham numpy görüntüsünü ImageNet standartlarında normalize edilmiş `(3, H, W)` PyTorch tensörüne dönüştürür.
        """
        # 1. RGB Kanal Dönüşümü
        rgb = to_rgb(image, is_bgr=is_bgr)

        # 2. En-Boy Oranı Koruma (Kare Dolgulama)
        if self.preserve_aspect_ratio:
            processed = pad_to_square(rgb, background_value=(255, 255, 255))
        else:
            processed = rgb

        # 3. Hedef Boyuta Boyutlandırma
        if processed.shape[:2] != self.target_size:
            processed = cv2.resize(processed, self.target_size, interpolation=cv2.INTER_AREA)

        # 4. [0, 255] -> [0.0, 1.0] Tensör Dönüşümü
        tensor = torch.from_numpy(processed).permute(2, 0, 1).float() / 255.0

        # 5. ImageNet Normalizasyonu
        normalized = self.normalizer(tensor)
        return normalized

    def __call__(self, image: np.ndarray, is_bgr: bool = True) -> torch.Tensor:
        return self.preprocess_image(image, is_bgr=is_bgr)


# Global tekil sözleşme örneği
_DEFAULT_CONTRACT = ForensicPreprocessingContract()


def preprocess_for_inference(
    image: Union[np.ndarray, torch.Tensor],
    is_bgr: bool = True,
) -> torch.Tensor:
    """
    Canlı API, arka plan servisleri ve testler için standart çıkarım tensörü üretici.
    
    Returns:
        (1, 3, 224, 224) float32 normalize PyTorch tensörü.
    """
    if isinstance(image, torch.Tensor):
        t = image
        if t.ndim == 3:
            t = t.unsqueeze(0)
        # Eğer zaten 3x224x224 ise ve ortalama sapma standart aralıkta ise doğrudan dön
        if t.shape[-2:] == (224, 224) and t.size(1) == 3:
            # Kontrol: eğer değerler sadece [0, 1] aralığındaysa ImageNet normalizasyonu uygula
            if t.min() >= 0.0 and t.max() <= 1.0:
                normalizer = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                t = normalizer(t[0]).unsqueeze(0)
            return t
        # Aksi halde tensörü numpy'a alıp standart pipeline'dan geçir
        np_img = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if np_img.max() <= 1.0:
            np_img = (np_img * 255.0).astype(np.uint8)
        else:
            np_img = np_img.astype(np.uint8)
        image = np_img

    tensor_3d = _DEFAULT_CONTRACT.preprocess_image(image, is_bgr=is_bgr)
    return tensor_3d.unsqueeze(0)


def inference_preprocess(
    image: Union[np.ndarray, torch.Tensor],
    is_bgr: bool = True,
) -> torch.Tensor:
    """
    Sprint 22 Sözleşmesi: Canlı çıkarım (inference) ön işleme fonksiyonu.
    preprocess_for_inference() ile birebir eşittir.
    Returns:
        (1, 3, 224, 224) float32 normalize PyTorch tensörü.
    """
    return preprocess_for_inference(image, is_bgr=is_bgr)


class V4ForensicAugmentation:
    """
    Sprint 22 V4 Adli Biyomekanik Veri Artırma ve Ön İşleme Katmanı.
    
    Kullanılan kontrollü artırmalar:
    - rotation: ±2°
    - small translation: ±4 piksel
    - mild contrast variation: [0.95, 1.05]
    - mild brightness variation: [-5, +5]
    - very small Gaussian noise: sigma <= 1.5
    
    Kesinlikle KULLANILMAYAN agresif dönüşümler:
    - Horizontal/vertical flip
    - Large rotation (>2°)
    - Elastic deformation
    - Strong perspective transformation
    - Aggressive crop / Random erasing
    """
    def __init__(self, is_training: bool = True, target_size: Tuple[int, int] = (224, 224)):
        self.is_training = is_training
        self.target_size = target_size
        self.normalizer = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __call__(self, image: np.ndarray, is_bgr: bool = True) -> torch.Tensor:
        if image is None or image.size == 0:
            raise ValueError("Boş veya geçersiz görsel verisi.")

        img_rgb = to_rgb(image, is_bgr=is_bgr)
        h, w = img_rgb.shape[:2]

        if self.is_training:
            import random
            # 1. Küçük rotasyon (±2 derece) ve hafif öteleme (±4 piksel)
            angle = random.uniform(-2.0, 2.0)
            tx = random.uniform(-4.0, 4.0)
            ty = random.uniform(-4.0, 4.0)
            center = (w / 2.0, h / 2.0)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty
            img_rgb = cv2.warpAffine(
                img_rgb, M, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )

            # 2. Hafif kontrast ve parlaklık
            alpha = random.uniform(0.95, 1.05)
            beta = random.uniform(-5.0, 5.0)
            img_rgb = np.clip(alpha * img_rgb.astype(np.float32) + beta, 0, 255).astype(np.uint8)

            # 3. Çok küçük Gauss gürültüsü
            if random.random() < 0.3:
                noise = np.random.normal(0, 1.0, img_rgb.shape).astype(np.float32)
                img_rgb = np.clip(img_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Pad to square
        img_padded = pad_to_square(img_rgb, background_value=(255, 255, 255))
        img_resized = cv2.resize(img_padded, self.target_size, interpolation=cv2.INTER_AREA)

        # Tensör ve normalizasyon
        tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
        return self.normalizer(tensor)


def training_preprocess(image: np.ndarray, is_training: bool = True, is_bgr: bool = True) -> torch.Tensor:
    """
    Sprint 22 Sözleşmesi: Model eğitimi için standart ön işleme fonksiyonu.
    is_training=False durumunda inference_preprocess ile tensör düzeyinde tam özdeştir.
    """
    aug = V4ForensicAugmentation(is_training=is_training)
    return aug(image, is_bgr=is_bgr)

