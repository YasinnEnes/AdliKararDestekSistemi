"""
FHDSS - Bilgisayarlı Görü Belge Segmentasyon Modülü (Document Segmenter)

Tam sayfa taranmış adli belgelerden (A4) metin bloklarını, satırları ve
imza bölgelerini tespit edip kırpan projeksiyon profili ve morfolojik
bağlı bileşenler (connected components) algoritması.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class LineSegment:
    """Ayrıştırılmış tek bir metin satırı verisi."""
    line_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    image: np.ndarray
    mean_intensity: float
    stroke_pixel_count: int


@dataclass
class SignatureCandidate:
    """Tespit edilen imza adayı bölgesi verisi."""
    candidate_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    image: np.ndarray
    confidence_score: float
    aspect_ratio: float
    density: float
    reason: str


@dataclass
class DocumentSegmentationResult:
    """Tam sayfa adli belge segmentasyon sonucu."""
    original_shape: Tuple[int, int]
    lines: List[LineSegment]
    signatures: List[SignatureCandidate]
    num_lines_detected: int
    num_signatures_detected: int


class ForensicDocumentSegmenter:
    """
    Adli Doküman Segmentasyon Motoru.
    
    Yetenekler:
    - Belge ön işleme ve binarizasyon (preprocess_document_roi)
    - Yatay projeksiyon profili ile satır segmentasyonu (segment_lines)
    - Morfolojik bağlı bileşenler ile imza adayı tespiti (extract_signature_candidates)
    """

    def __init__(
        self,
        min_line_height: int = 12,
        max_line_height: int = 150,
        min_signature_area: int = 400,
        max_signature_area: Optional[int] = None,
    ):
        self.min_line_height = min_line_height
        self.max_line_height = max_line_height
        self.min_signature_area = min_signature_area
        self.max_signature_area = max_signature_area

    def preprocess_document_roi(self, image: np.ndarray) -> np.ndarray:
        """
        Belge görselini gri seviyeye dönüştürür, zemin gürültüsünü filtreler ve
        adaptif Otsu binarizasyonu uygulayarak ikili (binary) görüntü üretir.
        Mürekkep pikselleri = 255 (beyaz ön plan), Zemin = 0 (siyah arka plan).
        """
        if image is None or image.size == 0:
            raise ValueError("Geçersiz veya boş görsel verisi.")

        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Hafif Gaussian yumuşatma ile tarayıcı kumlanmasını gider
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Otsu binarizasyon (Mürekkep koyu ise ters çevir: mürekkep=255)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        return binary

    def segment_lines(
        self,
        image: np.ndarray,
        min_line_height: Optional[int] = None,
        valley_threshold_ratio: float = 0.05,
    ) -> List[LineSegment]:
        """
        Yatay projeksiyon profili (horizontal projection profile) kullanarak
        belgedeki bağımsız metin satırlarını tespit eder ve kırpar.
        """
        min_h = min_line_height or self.min_line_height

        # Ön işlemden geçir (mürekkep pikselleri = 255)
        binary = self.preprocess_document_roi(image)
        h, w = binary.shape[:2]

        # Yatay izdüşüm profili: her satırdaki mürekkep piksel sayısı
        horizontal_proj = np.sum(binary > 0, axis=1)
        max_proj = np.max(horizontal_proj) if np.max(horizontal_proj) > 0 else 1.0
        threshold = max_proj * valley_threshold_ratio

        # Satır aralıklarını belirle (vadiler ve tepeler)
        in_line = False
        start_y = 0
        line_spans: List[Tuple[int, int]] = []

        for y in range(h):
            if horizontal_proj[y] > threshold:
                if not in_line:
                    in_line = True
                    start_y = y
            else:
                if in_line:
                    in_line = False
                    end_y = y
                    if (end_y - start_y) >= min_h and (end_y - start_y) <= self.max_line_height:
                        line_spans.append((start_y, end_y))

        # Son satır belgenin dibinde bitiyorsa ekle
        if in_line and (h - start_y) >= min_h:
            line_spans.append((start_y, h))

        # Eğer bir span birden fazla satır içeriyorsa iç vadilerden böl
        refined_spans: List[Tuple[int, int]] = []
        for sy, ey in line_spans:
            span_h = ey - sy
            if span_h > 40:
                sub_proj = horizontal_proj[sy:ey]
                min_val = np.min(sub_proj)
                median_val = np.median(sub_proj)
                inner_valleys = [i for i in range(8, span_h - 8) if sub_proj[i] <= max(min_val + 2, median_val * 0.40)]
                if inner_valleys:
                    split_points = []
                    last_v = -99
                    for v in inner_valleys:
                        if v - last_v > 10:
                            split_points.append(sy + v)
                        last_v = v
                    prev_y = sy
                    for sp in split_points:
                        if sp - prev_y >= min_h:
                            refined_spans.append((prev_y, sp))
                            prev_y = sp
                    if ey - prev_y >= min_h:
                        refined_spans.append((prev_y, ey))
                    continue
            refined_spans.append((sy, ey))
        line_spans = refined_spans

        # Orijinal renkli/gri görüntüden satır kırpmalarını oluştur
        lines: List[LineSegment] = []
        margin_x = 5
        margin_y = 2

        for idx, (y1, y2) in enumerate(line_spans, start=1):
            y_start = max(0, y1 - margin_y)
            y_end = min(h, y2 + margin_y)
            line_bin = binary[y_start:y_end, :]

            # Satırdaki metnin sol ve sağ sınırlarını (x1, x2) dikey projeksiyonla bul
            vert_proj = np.sum(line_bin > 0, axis=0)
            ink_x = np.where(vert_proj > 0)[0]
            if len(ink_x) == 0:
                continue

            x_start = max(0, ink_x[0] - margin_x)
            x_end = min(w, ink_x[-1] + margin_x)

            line_w = x_end - x_start
            line_h = y_end - y_start
            if line_w <= 5 or line_h <= 5:
                continue

            crop_roi = image[y_start:y_end, x_start:x_end]
            stroke_cnt = int(np.sum(binary[y_start:y_end, x_start:x_end] > 0))
            mean_val = float(np.mean(crop_roi))

            lines.append(LineSegment(
                line_id=f"LINE_{idx:03d}",
                bbox=(x_start, y_start, line_w, line_h),
                image=crop_roi,
                mean_intensity=mean_val,
                stroke_pixel_count=stroke_cnt,
            ))

        return lines

    def extract_signature_candidates(
        self,
        image: np.ndarray,
        min_area: Optional[int] = None,
        max_area: Optional[int] = None,
    ):
        """
        Belgedeki imza alanlarını tespit eder.
        
        Adli İmza Kriterleri:
        1. Bağlantılı bileşen morfolojisi: Standart harflerden daha geniş ve sürekli vuruşlar.
        2. En-Boy Oranı: İmzalar genelde yatay yayılımlıdır (aspect_ratio: 1.2 - 6.0).
        3. Belge konumu: İmzalar sıklıkla belgenin alt yarısında (özellikle sağ alt veya orta alt) bulunur.
        4. Piksel karmaşıklığı: Düz çizgilerden veya tablolardan farklı kontur dallanması içerir.
        """
        min_a = min_area or self.min_signature_area
        h, w = image.shape[:2]
        max_a = max_area or int(h * w * 0.40)

        binary = self.preprocess_document_roi(image)

        # Yakın vuruşları birbirine bağlamak için hafif genişletme (dilation)
        morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 6))
        dilated = cv2.dilate(binary, morph_kernel, iterations=2)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated, connectivity=8)

        candidates: List[SignatureCandidate] = []

        for i in range(1, num_labels):
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            box_w = int(stats[i, cv2.CC_STAT_WIDTH])
            box_h = int(stats[i, cv2.CC_STAT_HEIGHT])
            area = int(stats[i, cv2.CC_STAT_AREA])

            # Boyut filtreleri
            if area < min_a or area > max_a:
                continue
            if box_w < 30 or box_h < 15:
                continue
            if box_w > (w * 0.95) and box_h < 10:  # Sayfa yatay çizgisi / ayırıcı
                continue

            aspect_ratio = float(box_w) / float(box_h)

            # Ham görselden ROI çıkar
            pad = 6
            y_start = max(0, y - pad)
            y_end = min(h, y + box_h + pad)
            x_start = max(0, x - pad)
            x_end = min(w, x + box_w + pad)

            crop_roi = image[y_start:y_end, x_start:x_end]
            crop_bin = binary[y_start:y_end, x_start:x_end]
            ink_pixels = int(np.sum(crop_bin > 0))
            density = ink_pixels / float((crop_roi.shape[0] * crop_roi.shape[1]) + 1e-6)

            # Adli İmza Skorlama Heuristiği (0.0 - 1.0)
            score = 0.50
            reasons = []

            # A. Konum puanı: Belgenin alt yarısında yer alma bonusu
            vertical_pos_ratio = (y + box_h / 2.0) / float(h)
            if vertical_pos_ratio > 0.45:
                score += 0.20
                reasons.append("Belgenin alt yarısında konumlu")
            if (x + box_w / 2.0) / float(w) > 0.40:
                score += 0.10
                reasons.append("Sağ/orta alt imza bloğu uyumu")

            # B. En-Boy oranı puanı: İmzalar tipik olarak geniştir
            if 1.2 <= aspect_ratio <= 6.0:
                score += 0.15
                reasons.append(f"İmza en-boy oranı uyumlu ({aspect_ratio:.2f})")
            elif aspect_ratio < 0.6:
                score -= 0.20

            # C. Yoğunluk kontrolü: Aşırı dolu (tablo/damga) veya aşırı boş değil
            if 0.03 <= density <= 0.40:
                score += 0.10
                reasons.append(f"Mürekkep dağılım yoğunluğu adli aralıkta ({density:.2%})")
            else:
                score -= 0.15

            confidence = float(np.clip(score, 0.05, 0.98))

            if confidence >= 0.40:
                candidates.append(SignatureCandidate(
                    candidate_id=f"SIG_CANDIDATE_{len(candidates) + 1:02d}",
                    bbox=(x_start, y_start, x_end - x_start, y_end - y_start),
                    image=crop_roi,
                    confidence_score=round(confidence, 3),
                    aspect_ratio=round(aspect_ratio, 2),
                    density=round(density, 4),
                    reason=", ".join(reasons),
                ))

        # Güven skoruna göre sırala (en olası imza ilk sırada)
        candidates.sort(key=lambda c: c.confidence_score, reverse=True)
        return candidates

    def segment_document(self, image: np.ndarray) -> DocumentSegmentationResult:
        """Belgedeki hem satırları hem de imza adaylarını tek seferde çıkarır."""
        lines = self.segment_lines(image)
        signatures = self.extract_signature_candidates(image)

        return DocumentSegmentationResult(
            original_shape=(image.shape[0], image.shape[1]),
            lines=lines,
            signatures=signatures,
            num_lines_detected=len(lines),
            num_signatures_detected=len(signatures),
        )

    # ==========================================================================
    # Sprint 16.11: Karmaşık Adli Doküman Katman Ayırma ve Artefakt Bastırma
    # ==========================================================================

    def detect_and_filter_seals_and_stamps(
        self,
        image_bgr: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kırmızı, mor ve mavi resmi mühür ve damgaları renk uzayı (HSV) yardımıyla tespit eder.
        Dönüş: (temizlenmis_yazi_maskesi, muhafazali_muhur_maskesi)
        """
        if image_bgr is None or image_bgr.ndim != 3:
            empty = np.zeros(image_bgr.shape[:2], dtype=np.uint8) if image_bgr is not None else np.zeros((1, 1), dtype=np.uint8)
            return empty, empty

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Kırmızı mürekkep maskesi (HSV çift aralık)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)

        # Mavi / Mor resmi kaşe maskesi
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([140, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        seal_mask = mask_red | mask_blue

        # Standart ikili mürekkep maskesi
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Mühürleri yazı maskesinden çıkar
        handwriting_clean = cv2.bitwise_and(binary, binary, mask=cv2.bitwise_not(seal_mask))

        return handwriting_clean, seal_mask

    def remove_printed_guidelines(
        self,
        binary_image: np.ndarray,
        min_line_length: int = 80,
    ) -> np.ndarray:
        """
        Resmi formlardaki yatay kılavuz çizgileri ve tablo kenarlıklarını
        morfolojik çizgi çekirdekleriyle tespit edip yazı vuruşlarına zarar vermeden filtreler.
        """
        if binary_image is None or binary_image.size == 0:
            return binary_image

        # Yatay çizgi tespiti
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1))
        detected_lines = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, horiz_kernel)

        # Çizgileri orijinal ikili görüntüden çıkar
        cleaned = cv2.subtract(binary_image, detected_lines)
        return cleaned

    def filter_document_artifacts(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Mühür, damga ve kılavuz çizgilerini kombine olarak filtreleyerek
        morfolojik öznitelik çıkarımı için arındırılmış adli mürekkep maskesi döner.
        
        NOT:
        # TODO / DATASET_REQUIRED: Derin Doküman Yerleşim Analizi (LayoutLM / Mask R-CNN)
        # için etiketli Türkçe resmi evrak veri seti temin edildiğinde derin model katmanı eklenecektir.
        """
        hw_clean, seal_mask = self.detect_and_filter_seals_and_stamps(image_bgr)
        hw_no_guidelines = self.remove_printed_guidelines(hw_clean)

        return {
            "clean_handwriting_mask": hw_no_guidelines,
            "detected_seal_mask": seal_mask,
            "seal_detected": bool(np.sum(seal_mask > 0) > 200),
            "method": "hsv_morphological_separation_v1",
        }
