import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field
from scipy.ndimage import gaussian_filter1d


# ==============================================================================
# Pydantic Veri Modelleri
# ==============================================================================

class StructuralSpatialFeatures(BaseModel):
    """Adli El Yazısı Yapısal ve Uzamsal Öznitelikleri."""
    slant_angle_deg: float = Field(..., description="Yazı eğim açısı [-90, +90 derece]")
    baseline_angle_deg: float = Field(..., description="Temel yazı satırı eğim açısı (derece)")
    baseline_deviation_px: float = Field(..., description="Temel çizgi standart sapması / dalgalanma (piksel)")
    loop_count: int = Field(..., description="Kapalı döngü / ilmek sayısı")
    crossing_count: int = Field(..., description="Kesişim / çatallanma / kavşak noktası sayısı (degree >= 3)")
    terminal_count: int = Field(..., description="Uç nokta sayısı (degree == 1)")
    avg_stroke_length_px: float = Field(..., description="Ortalama vuruş / çizgi uzunluğu (piksel)")
    inter_stroke_spacing_px: float = Field(..., description="Vuruşlar / karakterler arası ortalama mesafe (piksel)")


class LineQualityFeatures(BaseModel):
    """Çizgi Kalitesi ve Kinematik Dinamik Proxy Öznitelikleri."""
    tremor_index_psd: float = Field(..., description="FFT yüksek frekanslı titreme spektral yoğunluk indeksi")
    smoothness_energy: float = Field(..., description="Eğrilik varyasyon / bükülme enerjisi (küçük = pürüzsüz)")
    mean_velocity_proxy: float = Field(..., description="2/3 Güç Yasası ortalama hız kestirimi")
    velocity_variance_proxy: float = Field(..., description="Hız varyansı kestirimi")
    acceleration_proxy: float = Field(..., description="Kinematik ivme dinamik proxy kestirimi")


class ForensicFeatureReport(BaseModel):
    """Kombine Adli Öznitelik Raporu ve Vektör Paketi."""
    structural: StructuralSpatialFeatures
    line_quality: LineQualityFeatures
    feature_vector: List[float] = Field(..., description="Sayısal karşılaştırma vektörü")
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# Gürültü ve Kağıt Dokusu Filtreleme Fonksiyonu
# ==============================================================================

def filter_paper_grain_and_noise(image: np.ndarray, min_component_size: int = 4) -> np.ndarray:
    """
    Kağıt dokusu lifleri, tarama parazitleri ve ayrık lekeleri temizleyen adaptif morfolojik filtre.
    """
    if image is None or image.size == 0 or np.count_nonzero(image) == 0:
        return image

    bin_img = (image > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, connectivity=8)

    clean_img = np.zeros_like(bin_img)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_component_size:
            clean_img[labels == i] = 255

    return clean_img


# ==============================================================================
# 1. Yapısal & Uzamsal Öznitelik Çıkarıcı (Structural & Spatial Extractor)
# ==============================================================================

class StructuralSpatialExtractor:
    """
    İskelet ve ikili görüntüler üzerinden morfolojik, topolojik ve uzamsal öznitelikleri çıkarır.
    
    Özellikler:
    - 8-komşuluk çekirdeği ile Uç Noktalar (Terminals) ve Kavşaklar (Crossings/Junctions)
    - Döngü / İlmek (Loop) analizi
    - Momentler ve doğrultu analizi ile Yazı Eğimi (Slant)
    - Taban çizgisi (Baseline) regresyonu ve dalgalanma sapması
    - Vuruş uzunlukları ve boşluk mesafeleri
    """

    def __init__(self):
        # 3x3 8-komşuluk konvolüsyon çekirdeği
        self.neighbor_kernel = np.array(
            [[1, 1, 1],
             [1, 0, 1],
             [1, 1, 1]], dtype=np.uint8
        )

    def extract_topological_points(self, skeleton: np.ndarray) -> Tuple[int, int]:
        """İskelet üzerindeki uç noktaları (degree=1) ve kesişim/kavşakları (degree>=3) sayar."""
        if skeleton is None or skeleton.size == 0:
            return 0, 0
        skel_clean = filter_paper_grain_and_noise(skeleton, min_component_size=3)
        skel_bin = (skel_clean > 0).astype(np.uint8)
        if np.count_nonzero(skel_bin) == 0:
            return 0, 0

        neighbor_map = cv2.filter2D(skel_bin, -1, self.neighbor_kernel) * skel_bin
        
        terminals = int(np.count_nonzero(neighbor_map == 1))
        crossings = int(np.count_nonzero(neighbor_map >= 3))
        return terminals, crossings

    def extract_loops(self, binary_image: np.ndarray) -> int:
        """Kapalı döngü (loop) / delik sayısını hiyerarşik kontur analizi ile hesaplar."""
        if binary_image is None or binary_image.size == 0:
            return 0

        # Mürekkep 255 olacak şekilde normalize et
        if np.mean(binary_image) > 127:
            bin_inv = (binary_image == 0).astype(np.uint8) * 255
        else:
            bin_inv = (binary_image > 0).astype(np.uint8) * 255

        contours, hierarchy = cv2.findContours(bin_inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        
        loop_count = 0
        if hierarchy is not None and len(hierarchy) > 0:
            for h in hierarchy[0]:
                # h[3] ebeveyn kontur indeksidir; >= 0 ise bu bir iç boşluk/döngüdür
                if h[3] >= 0:
                    loop_count += 1

        return loop_count

    def calculate_slant_angle(self, skeleton: np.ndarray) -> float:
        """Merkezi momentler ve PCA eksen doğrultusu ile yazı eğim açısını (derece) hesaplar."""
        y_indices, x_indices = np.nonzero(skeleton > 0)
        if len(x_indices) < 10:
            return 0.0

        x_mean = np.mean(x_indices)
        y_mean = np.mean(y_indices)

        mu20 = np.mean((x_indices - x_mean) ** 2)
        mu02 = np.mean((y_indices - y_mean) ** 2)
        mu11 = np.mean((x_indices - x_mean) * (y_indices - y_mean))

        # Ana eylemsizlik ekseni açısı atan2(2*mu11, mu20 - mu02)
        denom = mu20 - mu02
        if abs(denom) < 1e-9 and abs(mu11) < 1e-9:
            angle_rad = 0.0
        else:
            angle_rad = 0.5 * math.atan2(2.0 * mu11, denom)

        angle_deg = math.degrees(angle_rad)
        return float(round(angle_deg, 2))

    def calculate_baseline_features(self, binary_or_skel: np.ndarray) -> Tuple[float, float]:
        """Taban çizgisi regresyonu eğimi ve piksel sapmasını (deviation) hesaplar."""
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (binary_or_skel > 0).astype(np.uint8), connectivity=8
        )

        if num_labels <= 2:
            return 0.0, 0.0

        # Her bileşenin taban (en alt) noktasını bul: y + height
        base_points = []
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT] + (stats[i, cv2.CC_STAT_WIDTH] / 2.0)
            y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area > 10:
                base_points.append((x, y))

        if len(base_points) < 3:
            return 0.0, 0.0

        pts = np.array(base_points)
        x_vals = pts[:, 0]
        y_vals = pts[:, 1]

        # Doğrusal regresyon fit y = slope * x + intercept
        slope, intercept = np.polyfit(x_vals, y_vals, deg=1)
        fitted_y = slope * x_vals + intercept
        residuals = y_vals - fitted_y
        deviation = float(np.std(residuals))

        baseline_angle = float(math.degrees(math.atan(slope)))
        return round(baseline_angle, 2), round(deviation, 2)

    def extract(
        self,
        skeleton_image: np.ndarray,
        binary_image: Optional[np.ndarray] = None,
    ) -> StructuralSpatialFeatures:
        """Tüm yapısal ve uzamsal öznitelikleri çıkarır."""
        terminals, crossings = self.extract_topological_points(skeleton_image)
        loop_count = self.extract_loops(binary_image if binary_image is not None else skeleton_image)
        slant_angle = self.calculate_slant_angle(skeleton_image)
        baseline_angle, baseline_dev = self.calculate_baseline_features(skeleton_image)

        # Ortalama vuruş uzunluğu (İskelet pikseli / bileşen sayısı)
        skel_pixels = int(np.count_nonzero(skeleton_image > 0))
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            (skeleton_image > 0).astype(np.uint8), connectivity=8
        )
        comp_count = max(1, num_labels - 1)
        avg_stroke_length = skel_pixels / comp_count

        # Vuruşlar arası ortalama yatay mesafe
        if comp_count > 1:
            lefts = sorted([stats[i, cv2.CC_STAT_LEFT] for i in range(1, num_labels)])
            spacings = [lefts[i+1] - lefts[i] for i in range(len(lefts)-1)]
            avg_spacing = float(np.mean(spacings)) if spacings else 0.0
        else:
            avg_spacing = 0.0

        return StructuralSpatialFeatures(
            slant_angle_deg=slant_angle,
            baseline_angle_deg=baseline_angle,
            baseline_deviation_px=baseline_dev,
            loop_count=loop_count,
            crossing_count=crossings,
            terminal_count=terminals,
            avg_stroke_length_px=round(avg_stroke_length, 2),
            inter_stroke_spacing_px=round(avg_spacing, 2),
        )


# ==============================================================================
# 2. Çizgi Kalitesi & Kinematik Proxy Çıkarıcı (Line Quality & Kinematics)
# ==============================================================================

class LineQualityKinematicsExtractor:
    """
    İskelet kontur hatları üzerinden FFT tabanlı titreme (Tremor) ve 
    2/3 Güç Yasası (Two-Thirds Power Law) kinematik hız/ivme tahminlerini çıkarır.
    """

    def __init__(self, min_stroke_points: int = 12):
        self.min_stroke_points = min_stroke_points

    def _extract_stroke_contours(self, skeleton: np.ndarray) -> List[np.ndarray]:
        """İskeletten sürekli çizgi izlerini (stroke paths) ayıklar."""
        skel_bin = (skeleton > 0).astype(np.uint8)
        contours, _ = cv2.findContours(skel_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        valid_strokes = []
        for c in contours:
            pts = c.reshape(-1, 2)
            if len(pts) >= self.min_stroke_points:
                valid_strokes.append(pts)
        return valid_strokes

    def compute_tremor_psd(self, stroke_pts: np.ndarray) -> float:
        """
        FFT Spektral Güç Yoğunluğu ve yüksek frekanslı sapma enerjisi ile titreme (Tremor) indeksini hesaplar.
        Sahte, zorlanmış veya yaşlılık/hastalık kaynaklı el yazılarında titreme enerjisi belirgin artar.
        """
        dx = np.diff(stroke_pts[:, 0])
        dy = np.diff(stroke_pts[:, 1])
        angles = np.unwrap(np.arctan2(dy, dx))

        if len(angles) < 8:
            return 0.0

        # Düzgünleştirilmiş temel hareket eğrisi (low-pass)
        smoothed_angles = gaussian_filter1d(angles, sigma=2.0)
        residuals = angles - smoothed_angles

        # FFT Spektral Analizi
        fft_vals = np.fft.rfft(residuals)
        power_spectrum = np.abs(fft_vals) ** 2
        
        # Yüksek frekans spektral güç oranı ve rezidüel sapma standart sapması kombinasyonu
        half_idx = len(power_spectrum) // 2
        high_freq_power = float(np.sum(power_spectrum[half_idx:])) / max(len(power_spectrum), 1)
        residual_std = float(np.std(residuals))

        # Boyutsuz normalize tremor indeksi
        tremor_score = float(residual_std * (1.0 + np.sqrt(high_freq_power)))

        return float(round(tremor_score, 4))

    def compute_curvature_and_kinematics(
        self, stroke_pts: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Eğrilik enerjisi ve 2/3 Güç Yasası (v = k * kappa^(-1/3)) hız/ivme kestirimlerini üretir.
        """
        dx = np.gradient(stroke_pts[:, 0].astype(np.float64))
        dy = np.gradient(stroke_pts[:, 1].astype(np.float64))
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        # Eğrilik (Curvature) kappa = |dx*ddy - dy*ddx| / (dx^2 + dy^2)^(3/2)
        denom = (dx ** 2 + dy ** 2) ** 1.5
        denom[denom < 1e-4] = 1e-4
        kappa = np.abs(dx * ddy - dy * ddx) / denom

        # 1. Bükülme / Eğrilik Değişim Enerjisi (Smoothness energy)
        smoothness_energy = float(np.mean(kappa ** 2))

        # 2. 2/3 Güç Yasası Hız Kestirimi (Kinematic Velocity Proxy)
        # v(t) = k * (kappa(t) + eps)^(-1/3)
        velocity_proxy = (kappa + 0.05) ** (-1.0 / 3.0)
        mean_vel = float(np.mean(velocity_proxy))
        vel_var = float(np.var(velocity_proxy))

        # 3. İvme Kestirimi (Acceleration Proxy)
        acc_proxy = float(np.mean(np.abs(np.gradient(velocity_proxy))))

        return smoothness_energy, mean_vel, vel_var, acc_proxy

    def extract(self, skeleton_image: np.ndarray) -> LineQualityFeatures:
        """Tüm çizgi kalitesi ve kinematik öznitelikleri çıkarır."""
        strokes = self._extract_stroke_contours(skeleton_image)
        if not strokes:
            return LineQualityFeatures(
                tremor_index_psd=0.0,
                smoothness_energy=0.0,
                mean_velocity_proxy=0.0,
                velocity_variance_proxy=0.0,
                acceleration_proxy=0.0,
            )

        tremor_scores = []
        smoothness_scores = []
        mean_velocities = []
        vel_variances = []
        accelerations = []

        for stroke in strokes:
            tremor = self.compute_tremor_psd(stroke)
            smoothness, mean_v, var_v, acc = self.compute_curvature_and_kinematics(stroke)

            tremor_scores.append(tremor)
            smoothness_scores.append(smoothness)
            mean_velocities.append(mean_v)
            vel_variances.append(var_v)
            accelerations.append(acc)

        return LineQualityFeatures(
            tremor_index_psd=round(float(np.mean(tremor_scores)), 4),
            smoothness_energy=round(float(np.mean(smoothness_scores)), 4),
            mean_velocity_proxy=round(float(np.mean(mean_velocities)), 4),
            velocity_variance_proxy=round(float(np.mean(vel_variances)), 4),
            acceleration_proxy=round(float(np.mean(accelerations)), 4),
        )


# ==============================================================================
# 3. Birleşik Adli Öznitelik Çıkarıcı (Forensic Feature Extractor)
# ==============================================================================

class ForensicFeatureExtractor:
    """
    Yapısal, Uzamsal, Çizgi Kalitesi ve Kinematik Çıkarıcıları koordine eden ana motor.
    """

    def __init__(self):
        self.structural_extractor = StructuralSpatialExtractor()
        self.kinematics_extractor = LineQualityKinematicsExtractor()

    def extract_all(
        self,
        skeleton_image: np.ndarray,
        binary_image: Optional[np.ndarray] = None,
    ) -> ForensicFeatureReport:
        """Tam adli öznitelik vektörünü ve raporunu üretir."""
        structural = self.structural_extractor.extract(skeleton_image, binary_image)
        line_quality = self.kinematics_extractor.extract(skeleton_image)

        # Karşılaştırma motorları için normalize sayısal özellik vektörü
        feature_vector = [
            structural.slant_angle_deg,
            structural.baseline_angle_deg,
            structural.baseline_deviation_px,
            float(structural.loop_count),
            float(structural.crossing_count),
            float(structural.terminal_count),
            structural.avg_stroke_length_px,
            structural.inter_stroke_spacing_px,
            line_quality.tremor_index_psd,
            line_quality.smoothness_energy,
            line_quality.mean_velocity_proxy,
            line_quality.velocity_variance_proxy,
            line_quality.acceleration_proxy,
        ]

        metadata = {
            "total_feature_dimensions": len(feature_vector),
            "extractor_version": "1.0.0",
        }

        return ForensicFeatureReport(
            structural=structural,
            line_quality=line_quality,
            feature_vector=[round(v, 4) for v in feature_vector],
            metadata=metadata,
        )
