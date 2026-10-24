import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from skimage.morphology import skeletonize as skimage_skeletonize


@dataclass(frozen=True)
class PreprocessingArtifacts:
    """
    Adli görüntü ön işleme adımlarının çıktılarını ve meta verilerini saklayan
    değiştirilemez (immutable) veri taşıyıcı.
    
    Orijinal kanıt hiçbir zaman bozulmaz; her aşama bağımsız bir katman olarak saklanır.
    """
    original: np.ndarray
    grayscale: np.ndarray
    clahe: np.ndarray
    deskewed: np.ndarray
    binarized: np.ndarray
    skeleton: np.ndarray
    normalized_crop: np.ndarray
    isolated_document: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ForensicImagePreprocessor:
    """
    Adli El Yazısı Analizi için Deterministik ve İzlenebilir Görüntü Ön İşleme Hattı.
    
    Özellikler:
    - Orijinal görüntüyü koruma (non-destructive)
    - Deterministik işlem garantisi (aynı girdi -> aynı çıktı)
    - CLAHE ile adaptif aydınlatma ve kontrast normalizasyonu
    - Sauvola ve Otsu adaptif binarizasyon
    - Hough dönüşümü temelli hassas deskewing (eğiklik düzeltme)
    - Zhang-Suen morfolojik iskeletleştirme (skeletonization)
    - Otomatik kontur kırpma ve merkezleme (Bounding Box Normalization)
    - Zemin gürültüsü ve tarama sararması temizliği
    """

    def __init__(
        self,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (8, 8),
        binarization_method: str = "sauvola",
        sauvola_window_size: int = 25,
        sauvola_k: float = 0.2,
        max_skew_angle_deg: float = 45.0,
        denoise_enabled: bool = True,
        clean_speckle_noise: bool = True,
        deskew_enabled: bool = False,
        document_isolation_enabled: bool = True,
    ):
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.binarization_method = binarization_method.lower()
        self.sauvola_window_size = (
            sauvola_window_size if sauvola_window_size % 2 == 1 else sauvola_window_size + 1
        )
        self.sauvola_k = sauvola_k
        self.max_skew_angle_deg = max_skew_angle_deg
        self.denoise_enabled = denoise_enabled
        self.clean_speckle_noise = clean_speckle_noise
        self.deskew_enabled = deskew_enabled
        self.document_isolation_enabled = document_isolation_enabled

    def isolate_document_boundary(
        self,
        image: np.ndarray,
        dark_border_threshold: float = 0.15,
        min_document_area_ratio: float = 0.15,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Telefon kamerası çekimlerinde masa, çevre arka plan ve gölgeleri
        otomatik tespit ederek yalnızca asıl kağıt/belge alanını (ROI) izole eder.
        Eğer görüntü zaten kırpılmış temiz bir belge ise (örn. telefonun 'belge tara' modu),
        orijinal görüntüyü korur ve gereksiz kırpma yapmaz.
        """
        if image is None or image.size == 0:
            return image, {"isolated": False, "reason": "empty_image"}

        h, w = image.shape[:2]
        gray = self.to_grayscale(image)

        # Görüntü kenarlarındaki piksel parlaklığını incele (üst, alt, sol, sağ %3 marjin)
        margin_y = max(5, int(h * 0.03))
        margin_x = max(5, int(w * 0.03))
        top_edge = gray[:margin_y, :]
        bottom_edge = gray[-margin_y:, :]
        left_edge = gray[:, :margin_x]
        right_edge = gray[:, -margin_x:]
        border_pixels = np.concatenate([
            top_edge.ravel(),
            bottom_edge.ravel(),
            left_edge.ravel(),
            right_edge.ravel(),
        ])

        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        thresh_val, bin_otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_border_ratio = float(np.mean(border_pixels < thresh_val))

        # Eğer kenarlarda belirgin koyu zemin (masa, klavye, oda sahnesi) varsa
        if dark_border_ratio >= dark_border_threshold:
            contours, _ = cv2.findContours(bin_otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [
                c for c in contours if cv2.contourArea(c) >= min_document_area_ratio * h * w
            ]
            if valid_contours:
                largest = max(valid_contours, key=cv2.contourArea)
                bx, by, bw, bh = cv2.boundingRect(largest)

                # Güvenli iç kırpma koordinatları
                crop_x1 = max(0, bx)
                crop_y1 = max(0, by)
                crop_x2 = min(w, bx + bw)
                crop_y2 = min(h, by + bh)

                cropped_doc = image[crop_y1:crop_y2, crop_x1:crop_x2]
                metadata = {
                    "isolated": True,
                    "original_shape": (h, w),
                    "crop_box_xyxy": (crop_x1, crop_y1, crop_x2, crop_y2),
                    "dark_border_ratio": round(dark_border_ratio, 3),
                    "document_shape": cropped_doc.shape[:2],
                }
                return cropped_doc, metadata

        return image, {
            "isolated": False,
            "original_shape": (h, w),
            "dark_border_ratio": round(dark_border_ratio, 3),
        }

    def to_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü 8-bit tek kanallı Gri Seviye (Grayscale) matrise dönüştürür."""
        if image is None or image.size == 0:
            raise ValueError("Girdi görüntüsü boş veya geçersiz.")

        if image.ndim == 2:
            return image.astype(np.uint8, copy=True)
        elif image.ndim == 3:
            channels = image.shape[2]
            if channels == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif channels == 1:
                return image[:, :, 0].astype(np.uint8, copy=True)

        raise ValueError(f"Desteklenmeyen görüntü boyutu/kanal sayısı: {image.shape}")

    def apply_clahe(
        self,
        gray_image: np.ndarray,
        clip_limit: Optional[float] = None,
        tile_grid_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Kontrast Sınırlı Adaptif Histogram Eşitleme (CLAHE) uygular."""
        clip = clip_limit if clip_limit is not None else self.clahe_clip_limit
        grid = tile_grid_size if tile_grid_size is not None else self.clahe_tile_grid_size

        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
        return clahe.apply(gray_image)

    def estimate_skew_angle(self, gray_or_bin_image: np.ndarray) -> float:
        """
        Hough Çizgi Dönüşümü ve morfolojik gradyan ile el yazısı/satır eğiklik açısını (derece) tahmin eder.
        """
        # Kenarları veya ikili çizgileri belirginleştir
        edges = cv2.Canny(gray_or_bin_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=100,
            minLineLength=int(gray_or_bin_image.shape[1] * 0.15),
            maxLineGap=20,
        )

        if lines is None or len(lines) == 0:
            return 0.0

        angles = []
        for line in lines:
            coords = np.array(line).ravel()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            dx = x2 - x1
            dy = y2 - y1
            if dx == 0:
                continue
            angle_rad = math.atan2(dy, dx)
            angle_deg = math.degrees(angle_rad)

            # Sadece yatay eksene yakın satır çizgilerini hesaba kat
            if abs(angle_deg) <= self.max_skew_angle_deg:
                angles.append(angle_deg)

        if not angles:
            return 0.0

        # Medyan açı değeri outlier'lara karşı dayanıklıdır
        median_angle = float(np.median(angles))
        return median_angle

    def deskew(
        self,
        image: np.ndarray,
        angle_deg: Optional[float] = None,
        border_value: int = 255,
    ) -> Tuple[np.ndarray, float]:
        """
        Görüntüyü verilen veya hesaplanan eğiklik açısı kadar döndürerek düzeltir (Deskewing).
        """
        if angle_deg is None:
            angle_deg = self.estimate_skew_angle(image)

        if abs(angle_deg) < 0.05:
            return image.copy(), 0.0

        h, w = image.shape[:2]
        center = (w / 2.0, h / 2.0)
        rot_mat = cv2.getRotationMatrix2D(center, angle_deg, scale=1.0)

        # Beyaz/açık adli zemin rengiyle doldurarak döndür
        deskewed = cv2.warpAffine(
            image,
            rot_mat,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value,
        )
        return deskewed, angle_deg

    def remove_background_noise(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Tarama sararması, aydınlatma gradyanı ve kağıt dokusu gürültüsünü morfolojik zemin tahminlemesiyle dengeler.
        """
        # 1. Geniş çekirdekli morfolojik açılma ile zemin aydınlatmasını tahmin et
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 35))
        background = cv2.morphologyEx(gray_image, cv2.MORPH_DILATE, kernel)
        
        # 2. Aydınlatma farkını normalize et (division normalization)
        norm = cv2.divide(gray_image, background, scale=255.0)
        norm = np.clip(norm, 0, 255).astype(np.uint8)

        # 3. İsteğe bağlı kenar koruyucu bilateral filtre
        if self.denoise_enabled:
            norm = cv2.bilateralFilter(norm, d=5, sigmaColor=30, sigmaSpace=30)

        return norm

    def binarize(
        self,
        gray_image: np.ndarray,
        method: Optional[str] = None,
        window_size: Optional[int] = None,
        k: Optional[float] = None,
        invert_output: bool = False,
    ) -> np.ndarray:
        """
        Adaptif binarizasyon (Sauvola veya Otsu) ve zemin gürültü filtreleme.
        
        Args:
            gray_image: Gri seviye 8-bit görüntü.
            method: 'sauvola' veya 'otsu'.
            invert_output: True ise mürekkep pikselleri 255 (beyaz), zemin 0 (siyah) yapılır.
                           False ise standart belge görünümü: Zemin 255 (beyaz), mürekkep 0 (siyah).
        """
        chosen_method = (method or self.binarization_method).lower()
        w_size = window_size or self.sauvola_window_size
        k_val = k if k is not None else self.sauvola_k

        # Zemin aydınlatmasını düzelt
        cleaned_gray = self.remove_background_noise(gray_image)

        if chosen_method == "sauvola":
            thresh = threshold_sauvola(cleaned_gray, window_size=w_size, k=k_val)
            # zemin > thresh (255), mürekkep <= thresh (0)
            binary = (cleaned_gray > thresh).astype(np.uint8) * 255
        elif chosen_method == "otsu":
            _, binary = cv2.threshold(cleaned_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError(f"Bilinmeyen binarizasyon yöntemi: {chosen_method}")

        # İzole minik toz/nokta gürültülerini temizle (Speckle noise filtering)
        if self.clean_speckle_noise:
            # Mürekkep pikselleri (binary == 0) üzerinde bağlı bileşen analizi
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cv2.bitwise_not(binary), connectivity=8)
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if area <= 3:  # 3 pikselden küçük izole noktaları temizle
                    binary[labels == i] = 255

        if invert_output:
            binary = cv2.bitwise_not(binary)

        return binary

    def skeletonize(self, binary_image: np.ndarray, is_inverted: bool = False) -> np.ndarray:
        """
        Zhang-Suen morfolojik iskeletleştirme algoritması ile el yazısı hatlarını 1-piksel kalınlığa indirger.
        
        Args:
            binary_image: 8-bit ikili görüntü.
            is_inverted: Eğer mürekkep 255 ise True; zemin 255 (mürekkep 0) ise False verilir.
        """
        # İskelet çıkarma işlemi için ön planın (mürekkebin) True/1 olması gerekir
        if not is_inverted:
            foreground = (binary_image == 0)
        else:
            foreground = (binary_image > 0)

        skeleton_bool = skimage_skeletonize(foreground, method="zhang")
        skeleton_uint8 = (skeleton_bool.astype(np.uint8)) * 255
        return skeleton_uint8

    def bounding_box_crop_and_center(
        self,
        image: np.ndarray,
        binary_mask: Optional[np.ndarray] = None,
        target_size: Tuple[int, int] = (224, 224),
        padding_ratio: float = 0.08,
        background_value: int = 255,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        İmza ve el yazısı vuruşlarının etrafındaki aşırı boşlukları kırpar (tight bounding box),
        oransal güvenli kenar boşluğu (padding) ekler ve en boy oranını (aspect ratio) koruyarak
        hedef boyuttaki temiz beyaz arka plana tam merkezlenmiş olarak yerleştirir.
        
        Args:
            image: Gri seviye (H, W) veya BGR (H, W, 3) görüntü.
            binary_mask: İsteğe bağlı ikili maske (arka plan 255, mürekkep 0). Verilmezse otomatik hesaplanır.
            target_size: Çıkış boyutu (Genişlik, Yükseklik) - varsayılan (224, 224).
            padding_ratio: Kırpma etrafına eklenecek oransal marjin (varsayılan 0.08 = %8).
            background_value: Doldurulacak arka plan rengi (255 = beyaz).
            
        Returns:
            centered_image: Hedef boyutta tam merkezlenmiş görüntü.
            bbox_metadata: Bounding box koordinatları ve normalizasyon metrikleri.
        """
        if binary_mask is None:
            gray = self.to_grayscale(image)
            _, binary_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Mürekkep piksellerini bul (zemin 255, mürekkep < 128)
        ink_pixels = (binary_mask < 128)
        y_coords, x_coords = np.where(ink_pixels)

        h, w = image.shape[:2]
        target_w, target_h = target_size

        if len(x_coords) == 0 or len(y_coords) == 0:
            # Boş görüntü fallback
            if image.ndim == 3:
                canvas = np.full((target_h, target_w, image.shape[2]), background_value, dtype=np.uint8)
            else:
                canvas = np.full((target_h, target_w), background_value, dtype=np.uint8)
            return canvas, {"has_content": False, "bbox_xywh": (0, 0, w, h)}

        min_x, max_x = int(np.min(x_coords)), int(np.max(x_coords))
        min_y, max_y = int(np.min(y_coords)), int(np.max(y_coords))
        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1

        # Oransal kenar boşluğu (padding) ekle
        pad_x = max(2, int(bbox_w * padding_ratio))
        pad_y = max(2, int(bbox_h * padding_ratio))

        crop_x1 = max(0, min_x - pad_x)
        crop_y1 = max(0, min_y - pad_y)
        crop_x2 = min(w, max_x + pad_x + 1)
        crop_y2 = min(h, max_y + pad_y + 1)

        cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]
        crop_h_actual, crop_w_actual = cropped.shape[:2]

        # En boy oranını koruyarak hedef tuval boyutuna ölçekle
        avail_w = target_w * (1.0 - padding_ratio * 1.5)
        avail_h = target_h * (1.0 - padding_ratio * 1.5)
        scale = min(avail_w / max(1, crop_w_actual), avail_h / max(1, crop_h_actual))

        new_w = max(1, int(round(crop_w_actual * scale)))
        new_h = max(1, int(round(crop_h_actual * scale)))

        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        resized_crop = cv2.resize(cropped, (new_w, new_h), interpolation=interp)

        # Tuval üzerine merkeze yerleştir
        if image.ndim == 3:
            canvas = np.full((target_h, target_w, image.shape[2]), background_value, dtype=np.uint8)
        else:
            canvas = np.full((target_h, target_w), background_value, dtype=np.uint8)

        offset_x = max(0, (target_w - new_w) // 2)
        offset_y = max(0, (target_h - new_h) // 2)

        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized_crop

        bbox_metadata = {
            "has_content": True,
            "original_shape": (h, w),
            "bbox_xywh": (min_x, min_y, bbox_w, bbox_h),
            "crop_box_xyxy": (crop_x1, crop_y1, crop_x2, crop_y2),
            "scale_factor": round(float(scale), 4),
            "centering_offset": (offset_x, offset_y),
            "target_size": target_size,
        }

        return canvas, bbox_metadata

    def process(
        self,
        image: np.ndarray,
        binarization_method: Optional[str] = None,
        deskew_enabled: Optional[bool] = None,
    ) -> PreprocessingArtifacts:
        """
        Tüm ön işleme adımlarını deterministik ve adli kanıt zincirini bozmayacak şekilde sırayla yürütür.
        Sprint 21: El yazısı ve imzalarda Hough deskew hatasını önlemek için varsayılan olarak deskew_enabled=False.
        """
        original_copy = image.copy()
        active_deskew = self.deskew_enabled if deskew_enabled is None else deskew_enabled

        # 0. Belge Sınırı ve Masa İzolasyonu (Document Boundary & Clutter Removal)
        if self.document_isolation_enabled:
            doc_img, doc_meta = self.isolate_document_boundary(original_copy)
        else:
            doc_img, doc_meta = original_copy.copy(), {"isolated": False}

        # 1. Grayscale (İzole edilmiş temiz belge üzerinden)
        gray = self.to_grayscale(doc_img)

        # 2. CLAHE Contrast Normalization
        clahe_img = self.apply_clahe(gray)

        # 3. Deskewing
        skew_angle = 0.0
        if active_deskew:
            deskewed_img, skew_angle = self.deskew(clahe_img)
        else:
            deskewed_img = clahe_img.copy()

        # 4. Adaptive Binarization (Standard: arka plan 255, mürekkep 0)
        bin_img = self.binarize(deskewed_img, method=binarization_method)

        # 5. Zhang-Suen Skeletonization (mürekkep 0 -> iskelet 255)
        skeleton_img = self.skeletonize(bin_img, is_inverted=False)

        # 6. Bounding box cropping and centering normalization
        normalized_crop, bbox_meta = self.bounding_box_crop_and_center(
            image=deskewed_img,
            binary_mask=bin_img,
            target_size=(224, 224),
            padding_ratio=0.08,
            background_value=255,
        )

        metadata = {
            "original_shape": original_copy.shape,
            "document_shape": doc_img.shape,
            "document_isolation": doc_meta,
            "grayscale_shape": gray.shape,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_grid_size": self.clahe_tile_grid_size,
            "deskew_angle_deg": round(skew_angle, 3),
            "binarization_method": binarization_method or self.binarization_method,
            "sauvola_window_size": self.sauvola_window_size,
            "sauvola_k": self.sauvola_k,
            "skeleton_method": "zhang_suen",
            "bounding_box_normalization": bbox_meta,
        }

        return PreprocessingArtifacts(
            original=original_copy,
            grayscale=gray,
            clahe=clahe_img,
            deskewed=deskewed_img,
            binarized=bin_img,
            skeleton=skeleton_img,
            normalized_crop=normalized_crop,
            isolated_document=doc_img,
            metadata=metadata,
        )
