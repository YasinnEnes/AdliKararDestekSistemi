import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field


class ComplexityCategory(str, Enum):
    """Found & Rogers İmza Karmaşıklık Seviyesi."""
    LOW = "LOW"        # Düşük karmaşıklık (< 0.35) -> Taklit edilme riski yüksek
    MEDIUM = "MEDIUM"  # Orta karmaşıklık [0.35, 0.70)
    HIGH = "HIGH"      # Yüksek karmaşıklık (>= 0.70) -> Ayırt edicilik yüksek, taklidi zor


class ComplexityComponents(BaseModel):
    """İmza karmaşıklığını oluşturan 5 temel matematiksel bileşen."""
    turning_points: int = Field(..., description="Dönüm / yön değişim noktası sayısı (TP)")
    crossings: int = Field(..., description="Kesişim ve çatallanma sayısı (CR)")
    loops: int = Field(..., description="Kapalı döngü / ilmek sayısı (LP)")
    skeleton_length_px: int = Field(..., description="Toplam iskelet çizgi uzunluğu (SL)")
    spatial_entropy: float = Field(..., description="2B Uzamsal Shannon Entropisi [0.0, 1.0]")


class ComplexityResult(BaseModel):
    """Found & Rogers İmza Karmaşıklık Analizi Raporu."""
    overall_complexity_index: float = Field(..., ge=0.0, le=1.0, description="Normalize karmaşıklık indeksi [0.0, 1.0]")
    category: ComplexityCategory = Field(..., description="Karmaşıklık kategorisi (LOW, MEDIUM, HIGH)")
    components: ComplexityComponents = Field(..., description="Ham bileşen değerleri")
    feature_contributions: Dict[str, float] = Field(..., description="Her bileşenin skora yüzde katkısı")
    forgery_vulnerability: str = Field(..., description="Adli taklit edilebilirlik ve simülasyon risk değerlendirmesi")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analiz güvenilirlik skoru")


class SignatureComplexityScorer:
    """
    Found & Rogers (2007) / Dewhurst Metodolojisine Dayalı İmza Karmaşıklığı Değerlendirme Motoru.
    
    Bileşenler:
    1. TP (Turning Points): Çizgi rotasındaki açı değişimleri ve infleksiyon noktaları.
    2. CR (Crossings): İskelet üzerindeki 8-komşuluk kesişim/kavşak sayısı (degree >= 3).
    3. LP (Loops): Hiyerarşik kontur boşluk analizi ile kapalı ilmek sayısı.
    4. SL (Skeleton Length): Toplam piksel vuruş uzunluğu.
    5. H_spatial (2D Spatial Shannon Entropy): Vuruşların ızgara hücrelerine dağılım düzensizliği.
    """

    def __init__(
        self,
        weight_tp: float = 0.20,
        weight_cr: float = 0.20,
        weight_lp: float = 0.25,
        weight_sl: float = 0.15,
        weight_entropy: float = 0.20,
        grid_size: Tuple[int, int] = (8, 8),
    ):
        self.w_tp = weight_tp
        self.w_cr = weight_cr
        self.w_lp = weight_lp
        self.w_sl = weight_sl
        self.w_entropy = weight_entropy
        self.grid_size = grid_size

        # Ağırlıkların toplamının 1.0 olmasını sağla
        total_w = self.w_tp + self.w_cr + self.w_lp + self.w_sl + self.w_entropy
        self.w_tp /= total_w
        self.w_cr /= total_w
        self.w_lp /= total_w
        self.w_sl /= total_w
        self.w_entropy /= total_w

    def count_turning_points(self, skeleton: np.ndarray, angle_threshold_deg: float = 40.0) -> int:
        """İskelet hatlarındaki dönüm ve yön sapma noktalarını (Turning Points) hesaplar."""
        skel_bin = (skeleton > 0).astype(np.uint8)
        contours, _ = cv2.findContours(skel_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        turning_points_total = 0
        thresh_rad = math.radians(angle_threshold_deg)

        for c in contours:
            pts = c.reshape(-1, 2)
            if len(pts) < 6:
                continue

            # 3 adımlı pencere ile yön türevi
            dx = pts[2:, 0] - pts[:-2, 0]
            dy = pts[2:, 1] - pts[:-2, 1]
            angles = np.unwrap(np.arctan2(dy, dx))

            # Açısal ivme / ani yön değişimleri
            angle_diffs = np.abs(np.diff(angles))
            turning_points_total += int(np.count_nonzero(angle_diffs >= thresh_rad))

        return turning_points_total

    def count_crossings(self, skeleton: np.ndarray) -> int:
        """İskelet kesişim ve çatallanma (Junction) sayısını 8-komşuluk çekirdeği ile hesaplar."""
        skel_bin = (skeleton > 0).astype(np.uint8)
        if np.count_nonzero(skel_bin) == 0:
            return 0

        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
        neighbor_map = cv2.filter2D(skel_bin, -1, kernel) * skel_bin
        return int(np.count_nonzero(neighbor_map >= 3))

    def count_loops(self, binary_image: np.ndarray) -> int:
        """Kapalı döngü / ilmek sayısını hiyerarşik kontur delikleriyle tespit eder."""
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
                if h[3] >= 0:
                    loop_count += 1
        return loop_count

    def compute_spatial_entropy(self, skeleton: np.ndarray) -> float:
        """
        2B Uzamsal Shannon Entropisini hesaplar.
        İmzanın sınır kutusu (bounding box) grid_size x grid_size hücrelere bölünür ve
        vuruş piksellerinin hücrelere dağılım homojenliği ölçülür.
        """
        y_indices, x_indices = np.nonzero(skeleton > 0)
        if len(x_indices) < 10:
            return 0.0

        min_x, max_x = int(np.min(x_indices)), int(np.max(x_indices))
        min_y, max_y = int(np.min(y_indices)), int(np.max(y_indices))

        width = max(max_x - min_x + 1, self.grid_size[1])
        height = max(max_y - min_y + 1, self.grid_size[0])

        # Grid histogramı oluştur
        cell_h = height / self.grid_size[0]
        cell_w = width / self.grid_size[1]

        grid_counts = np.zeros(self.grid_size, dtype=np.float64)
        for y, x in zip(y_indices, x_indices):
            r = min(int((y - min_y) / cell_h), self.grid_size[0] - 1)
            c = min(int((x - min_x) / cell_w), self.grid_size[1] - 1)
            grid_counts[r, c] += 1

        total_pixels = np.sum(grid_counts)
        if total_pixels == 0:
            return 0.0

        probs = grid_counts / total_pixels
        non_zero_probs = probs[probs > 0]

        # Shannon Entropisi: H = - sum(p * log2(p))
        entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))

        # Maksimum teorik entropi ile normalize et: log2(grid_size[0] * grid_size[1])
        max_entropy = math.log2(self.grid_size[0] * self.grid_size[1])
        normalized_entropy = float(entropy / max_entropy) if max_entropy > 0 else 0.0

        return float(min(1.0, max(0.0, normalized_entropy)))

    def score(
        self,
        skeleton_image: np.ndarray,
        binary_image: Optional[np.ndarray] = None,
    ) -> ComplexityResult:
        """
        Görüntüden 5 bileşeni çıkarır ve Found & Rogers karmaşıklık indeksini ($CI$) üretir.
        Kağıt lifi ve parazit gürültüleri adaptif olarak temizlenir.
        """
        from cv.features.extractors import filter_paper_grain_and_noise
        skel_clean = filter_paper_grain_and_noise(skeleton_image, min_component_size=3)

        # 1. Bileşenleri Hesapla
        tp = self.count_turning_points(skel_clean)
        cr = self.count_crossings(skel_clean)
        lp = self.count_loops(binary_image if binary_image is not None else skel_clean)
        sl = int(np.count_nonzero(skel_clean > 0))
        entropy = self.compute_spatial_entropy(skel_clean)

        # 2. Doygunluk / Normalize Dönüşümleri (Saturating sigmoid / exponential scaling)
        # Her bir alt metrik [0.0, 1.0] aralığına normalize edilir
        norm_tp = 1.0 - math.exp(-tp / 20.0)
        norm_cr = 1.0 - math.exp(-cr / 15.0)
        norm_lp = 1.0 - math.exp(-lp / 4.0)
        norm_sl = 1.0 - math.exp(-sl / 1200.0)
        norm_entropy = entropy

        # 3. Ağırlıklı Bileşik Karmaşıklık İndeksi (CI)
        term_tp = self.w_tp * norm_tp
        term_cr = self.w_cr * norm_cr
        term_lp = self.w_lp * norm_lp
        term_sl = self.w_sl * norm_sl
        term_h = self.w_entropy * norm_entropy

        ci = term_tp + term_cr + term_lp + term_sl + term_h
        ci_clamped = round(float(min(1.0, max(0.0, ci))), 4)

        # 4. Yüzde Katkı Dağılımı
        total_terms = term_tp + term_cr + term_lp + term_sl + term_h + 1e-8
        contributions = {
            "turning_points_pct": round((term_tp / total_terms) * 100.0, 2),
            "crossings_pct": round((term_cr / total_terms) * 100.0, 2),
            "loops_pct": round((term_lp / total_terms) * 100.0, 2),
            "skeleton_length_pct": round((term_sl / total_terms) * 100.0, 2),
            "spatial_entropy_pct": round((term_h / total_terms) * 100.0, 2),
        }

        # 5. Kategori ve Risk Değerlendirmesi
        if ci_clamped < 0.35:
            category = ComplexityCategory.LOW
            vulnerability = (
                "YÜKSEK RİSK: İmza basit geometrik hatlardan oluşuyor. "
                "Taklit (simulation/forgery) girişimlerine karşı savunmasızdır; "
                "bilirkişi incelemesinde yüksek sayıda referans belge gereklidir."
            )
        elif ci_clamped < 0.70:
            category = ComplexityCategory.MEDIUM
            vulnerability = (
                "ORTA RİSK: Standart adli ayırt ediciliğe sahip imza yapısı. "
                "Doğal varyasyon aralığı içinde güvenle karşılaştırılabilir."
            )
        else:
            category = ComplexityCategory.HIGH
            vulnerability = (
                "DÜŞÜK RİSK: Yüksek karmaşıklıkta, çok sayıda dönüş/kesişim ve döngü içeren imza. "
                "Taklit edilmesi son derece zordur; adli delil ve teşhis değeri yüksektir."
            )

        # Güvenilirlik skoru: İskelet piksel sayısına bağlı
        confidence = round(float(min(1.0, max(0.5, sl / 300.0))), 2)

        return ComplexityResult(
            overall_complexity_index=ci_clamped,
            category=category,
            components=ComplexityComponents(
                turning_points=tp,
                crossings=cr,
                loops=lp,
                skeleton_length_px=sl,
                spatial_entropy=round(entropy, 4),
            ),
            feature_contributions=contributions,
            forgery_vulnerability=vulnerability,
            confidence=confidence,
        )
