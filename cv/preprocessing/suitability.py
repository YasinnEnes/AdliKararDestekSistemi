from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field


class SuitabilityStatus(str, Enum):
    """Adli Belge / El Yazısı Uygunluk Karar Statüleri."""
    SUITABLE = "SUITABLE"                    # İncelemeye tam uygun
    PARTIALLY_SUITABLE = "PARTIALLY_SUITABLE"# Kısıtlı / Uyarılarla incelenebilir
    INSUFFICIENT = "INSUFFICIENT"            # Yetersiz kalite / İnceleme yapılamaz
    NOT_COMPARABLE = "NOT_COMPARABLE"        # Karşılaştırılamaz (Boş, tamamen siyah/beyaz veya bozuk veri)


class SuitabilityMetrics(BaseModel):
    """Görüntü kalite ve adli el yazısı uygunluk metrikleri."""
    blur_laplacian_var: float = Field(..., description="Laplacian varyansı bulanıklık skoru")
    rms_contrast: float = Field(..., description="Kök ortalama kare kontrast skoru (RMS)")
    michelson_contrast: float = Field(..., description="Michelson kontrast oranı [0.0, 1.0]")
    snr_db: float = Field(..., description="Sinyal-Gürültü Oranı (SNR) dB")
    stroke_density: float = Field(..., description="Mürekkep / yazı pikseli yoğunluk oranı")
    connected_components_count: int = Field(..., description="Tespit edilen bağlı yazı bileşenleri sayısı")
    width: int = Field(..., description="Görüntü genişliği (piksel)")
    height: int = Field(..., description="Görüntü yüksekliği (piksel)")


class SuitabilityResult(BaseModel):
    """Adli Uygunluk ve Kalite Değerlendirme Rapor Çıktısı."""
    status: SuitabilityStatus = Field(..., description="Nihai uygunluk kararı")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="0.0 - 1.0 arası genel kalite skoru")
    is_admissible: bool = Field(..., description="Adli karar destek sürecine dahil edilebilir mi?")
    metrics: SuitabilityMetrics = Field(..., description="Hesaplanan sayısal metrikler")
    reasons: List[str] = Field(default_factory=list, description="Karara ilişkin adli gerekçeler")
    recommendations: List[str] = Field(default_factory=list, description="İyileştirme veya yeniden tarama önerileri")


class SuitabilityAssessor:
    """
    Adli El Yazısı Uygunluk ve Kalite Değerlendirme Motoru (Suitability Assessment Engine).
    
    Belgelerin adli incelemeye ve derin öğrenme modellerine girmeden önce
    çözünürlük, bulanıklık, kontrast ve yazı varlığı açısından filtrelenmesini sağlar.
    """

    def __init__(
        self,
        min_width: int = 100,
        min_height: int = 100,
        blur_threshold_insufficient: float = 60.0,
        blur_threshold_suitable: float = 120.0,
        rms_contrast_min: float = 20.0,
        min_stroke_density: float = 0.003,    # En az %0.3 yazı pikseli
        max_stroke_density: float = 0.85,     # Aşırı siyah/leke tespiti
        min_connected_components: int = 2,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.blur_threshold_insufficient = blur_threshold_insufficient
        self.blur_threshold_suitable = blur_threshold_suitable
        self.rms_contrast_min = rms_contrast_min
        self.min_stroke_density = min_stroke_density
        self.max_stroke_density = max_stroke_density
        self.min_connected_components = min_connected_components

    def _ensure_grayscale(self, image: np.ndarray) -> np.ndarray:
        """Görüntüyü 8-bit tek kanallı gri seviyeye dönüştürür."""
        if image is None or image.size == 0:
            raise ValueError("Girdi görüntüsü geçersiz veya boş.")

        if image.ndim == 2:
            return image.astype(np.uint8, copy=True)
        elif image.ndim == 3:
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            elif image.shape[2] == 1:
                return image[:, :, 0].astype(np.uint8, copy=True)

        raise ValueError(f"Desteklenmeyen görüntü boyutu: {image.shape}")

    def compute_metrics(self, image: np.ndarray) -> SuitabilityMetrics:
        """Görüntünün tüm fiziksel ve adli kalite metriklerini hesaplar."""
        gray = self._ensure_grayscale(image)
        h, w = gray.shape

        # 1. Bulanıklık (Laplacian Varyansı)
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        # 2. RMS Kontrastı (Standart sapma)
        rms_contrast = float(np.std(gray))

        # 3. Michelson Kontrastı
        min_val = float(np.min(gray))
        max_val = float(np.max(gray))
        michelson = (max_val - min_val) / (max_val + min_val + 1e-6)

        # 4. Sinyal-Gürültü Oranı (SNR)
        mean_val = float(np.mean(gray))
        std_val = float(np.std(gray))
        if std_val > 1e-4:
            snr_db = float(20.0 * np.log10((mean_val + 1e-6) / (std_val + 1e-6)))
        else:
            snr_db = 0.0

        # 5. Yazı / Mürekkep Yoğunluğu ve Bağlı Bileşen Analizi
        # Otsu eşikleme ile arka planı ayır
        _, bin_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        total_pixels = h * w
        foreground_pixels = int(np.count_nonzero(bin_inv))
        stroke_density = foreground_pixels / max(total_pixels, 1)

        # Bağlı bileşen analizi (Connected Components)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            bin_inv, connectivity=8
        )
        # 0. etiket arka plan olduğundan hariç tut
        valid_components = max(0, num_labels - 1)

        return SuitabilityMetrics(
            blur_laplacian_var=round(laplacian_var, 3),
            rms_contrast=round(rms_contrast, 3),
            michelson_contrast=round(float(michelson), 4),
            snr_db=round(snr_db, 3),
            stroke_density=round(stroke_density, 5),
            connected_components_count=valid_components,
            width=w,
            height=h,
        )

    def assess(self, image: np.ndarray) -> SuitabilityResult:
        """
        Görüntüyü değerlendirir ve adli karar destek uygunluk sonucunu üretir.
        """
        metrics = self.compute_metrics(image)
        reasons: List[str] = []
        recommendations: List[str] = []

        is_not_comparable = False
        is_insufficient = False
        is_partially_suitable = False

        # --- Çözünürlük Kontrolü ---
        if metrics.width < self.min_width or metrics.height < self.min_height:
            is_not_comparable = True
            reasons.append(
                f"Görüntü çözünürlüğü yetersiz ({metrics.width}x{metrics.height} px, beklenen min: {self.min_width}x{self.min_height} px)."
            )
            recommendations.append("Belgeyi en az 300 DPI optik çözünürlükte yeniden tarayınız.")

        # --- Yazı / Mürekkep Yoğunluğu Kontrolü ---
        if metrics.stroke_density < self.min_stroke_density or metrics.connected_components_count < self.min_connected_components:
            is_not_comparable = True
            reasons.append(
                f"Görüntüde karşılaştırılabilecek düzeyde el yazısı/mürekkep izi bulunamadı (Yoğunluk: {metrics.stroke_density:.4f}, Bileşen: {metrics.connected_components_count})."
            )
            recommendations.append("Görüntünün doğru kırpıldığından ve el yazısı içerdiğinden emin olunuz.")

        elif metrics.stroke_density > self.max_stroke_density:
            is_not_comparable = True
            reasons.append("Görüntü aşırı doygun/lekelenmiş veya tamamen siyah zemin içeriyor.")
            recommendations.append("Belgenin ışıklandırmasını ve tarama ayarlarını kontrol ediniz.")

        # --- Bulanıklık (Blur) Kontrolü ---
        if metrics.blur_laplacian_var < self.blur_threshold_insufficient:
            is_insufficient = True
            reasons.append(
                f"Aşırı odak/hareket bulanıklığı tespit edildi (Laplacian varyansı: {metrics.blur_laplacian_var:.1f} < {self.blur_threshold_insufficient})."
            )
            recommendations.append("Belgenin net odaklı, sabitlenmiş ve titreşimsiz yeni bir görüntüsünü alınız.")
        elif metrics.blur_laplacian_var < self.blur_threshold_suitable:
            is_partially_suitable = True
            reasons.append(
                f"Hafif bulanıklık mevcut (Laplacian varyansı: {metrics.blur_laplacian_var:.1f}). İnceleme kısıtlı güvenilirlikle yapılabilir."
            )

        # --- Kontrast Kontrolü ---
        if metrics.rms_contrast < self.rms_contrast_min:
            if is_insufficient:
                reasons.append(f"Düşük kontrast seviyesi (RMS: {metrics.rms_contrast:.1f} < {self.rms_contrast_min}).")
            else:
                is_partially_suitable = True
                reasons.append(f"Zemin ile mürekkep kontrastı zayıf (RMS: {metrics.rms_contrast:.1f} < {self.rms_contrast_min}).")
            recommendations.append("Düzgün aydınlatma ve yüksek dinamik aralıklı tarayıcı kullanınız.")

        # --- Nihai Statü ve Skor Belirleme ---
        if is_not_comparable:
            status = SuitabilityStatus.NOT_COMPARABLE
            score = 0.0
            is_admissible = False
        elif is_insufficient:
            status = SuitabilityStatus.INSUFFICIENT
            score = min(0.35, max(0.05, metrics.blur_laplacian_var / (self.blur_threshold_suitable * 2.0)))
            is_admissible = False
        elif is_partially_suitable:
            status = SuitabilityStatus.PARTIALLY_SUITABLE
            score = min(0.79, max(0.40, metrics.blur_laplacian_var / (self.blur_threshold_suitable * 1.5)))
            is_admissible = True
        else:
            status = SuitabilityStatus.SUITABLE
            score = min(1.0, max(0.80, 0.80 + (metrics.blur_laplacian_var / 1000.0) * 0.20))
            is_admissible = True
            reasons.append("Görüntü adli inceleme için gerekli netlik, kontrast ve çözünürlük standartlarını sağlamaktadır.")

        return SuitabilityResult(
            status=status,
            overall_score=round(score, 3),
            is_admissible=is_admissible,
            metrics=metrics,
            reasons=reasons,
            recommendations=recommendations,
        )
