import hashlib
import json
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from cv.features.extractors import ForensicFeatureReport
from ml.models.comparison import ComparisonReport


class IndicatorLevel(str, Enum):
    """Taklit / Gizleme Anomali Düzeyi."""
    NO_STRONG_EVIDENCE = "NO_STRONG_EVIDENCE"       # Belirgin anomali/taklit embaresi gözlenmedi
    POSSIBLE_INDICATORS = "POSSIBLE_INDICATORS"     # Olası duraksama ve motor kontrol düzensizliği
    STRONG_INDICATORS = "STRONG_INDICATORS"         # Güçlü taklit (simulation) ve kinematik ayrışma bulguları


class SimulationIndicators(BaseModel):
    """Taklit ve Gizleme Anomali Analiz Çıktısı."""
    indicator_level: IndicatorLevel = Field(..., description="Nihai taklit gösterge seviyesi")
    overall_anomaly_score: float = Field(..., ge=0.0, le=1.0, description="0.0 - 1.0 arası genel anomali skoru")
    tremor_anomaly_detected: bool = Field(..., description="Yüksek frekanslı mikroskobik titreme anomalisi")
    kinematic_decoupling_detected: bool = Field(..., description="2/3 Güç Yasası hız-eğrilik ayrışma anomalisi")
    structural_disguise_detected: bool = Field(..., description="Kasıtlı eğim/geometri gizleme anomalisi")
    anomaly_details: List[str] = Field(default_factory=list, description="Nesnel adli gözlem maddeleri")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Analiz güvenilirlik düzeyi")
    audit_hash_sha256: str = Field(..., description="Denetim izi bütünlük imzası")


class ForensicDisguiseDetector:
    """
    Adli El Yazısı ve İmza Taklit / Gizleme Anomali Motoru (Simulation & Disguise Detector).
    
    Adli Bilişim İlkeleri:
    - Orijinal el yazısı, balistik nöromusküler motor programlarla akıcı üretilir.
    - Taklit (Simulation) ve kopyalama girişimlerinde:
      1. Çizgiler hızlı atılmak yerine yavaş çizilir; bu durum yüksek frekanslı titremeye (Tremor) yol açar.
      2. 2/3 Güç Yasası (v = k * kappa^(-1/3)) ihlal edilir (Kinematik ayrışma).
      3. Kalem duraksamaları (Hesitations) ve yapay kıvrımlar oluşur.
    - Kasıtlı Gizleme (Disguise) durumunda:
      Yazar eğimi (slant) veya temel satırı radikal değiştirir fakat ince motor titreme özellikleri varlığını sürdürür.
    """

    def __init__(
        self,
        tremor_threshold_abs: float = 0.12,
        smoothness_threshold_abs: float = 0.10,
        slant_disguise_threshold_deg: float = 22.0,
    ):
        self.tremor_threshold_abs = tremor_threshold_abs
        self.smoothness_threshold_abs = smoothness_threshold_abs
        self.slant_disguise_threshold_deg = slant_disguise_threshold_deg

    def analyze(
        self,
        questioned_report: Union[ForensicFeatureReport, Dict[str, Any], List[float]],
        comparison_report: Optional[ComparisonReport] = None,
    ) -> SimulationIndicators:
        """
        Sorgulanan belgedeki kinematik ve yapısal öznitelikleri taklit/gizleme anomalileri açısından inceler.
        """
        # Öznitelik değerlerini normalize biçimde çek
        if isinstance(questioned_report, ForensicFeatureReport):
            q_tremor = questioned_report.line_quality.tremor_index_psd
            q_smoothness = questioned_report.line_quality.smoothness_energy
            q_mean_vel = questioned_report.line_quality.mean_velocity_proxy
            q_vel_var = questioned_report.line_quality.velocity_variance_proxy
            q_slant = questioned_report.structural.slant_angle_deg
        elif isinstance(questioned_report, dict):
            q_tremor = float(questioned_report.get("tremor_index_psd", 0.0))
            q_smoothness = float(questioned_report.get("smoothness_energy", 0.0))
            q_mean_vel = float(questioned_report.get("mean_velocity_proxy", 2.0))
            q_vel_var = float(questioned_report.get("velocity_variance_proxy", 0.3))
            q_slant = float(questioned_report.get("slant_angle_deg", 0.0))
        elif isinstance(questioned_report, (list, tuple)):
            # 13 boyutlu vektörden indeksler: 8: tremor, 9: smoothness, 10: mean_vel, 11: vel_var, 0: slant
            q_slant = float(questioned_report[0])
            q_tremor = float(questioned_report[8])
            q_smoothness = float(questioned_report[9])
            q_mean_vel = float(questioned_report[10])
            q_vel_var = float(questioned_report[11])
        else:
            raise ValueError(f"Geçersiz veri formatı: {type(questioned_report)}")

        details: List[str] = []
        anomaly_scores: List[float] = []

        # 1. Titreme (Tremor) Anomalisi
        tremor_anomaly = False
        if q_tremor >= self.tremor_threshold_abs:
            tremor_anomaly = True
            severity = min(1.0, (q_tremor - self.tremor_threshold_abs) / 0.20 + 0.5)
            anomaly_scores.append(severity)
            details.append(
                f"Yüksek Frekanslı Titreme (Tremor): Vuruş yörüngesinde doğal balistik akış yerine "
                f"yapay duraksama ve mikroskobik titreşim dalgaları tespit edildi (İndeks: {q_tremor:.3f})."
            )
        else:
            anomaly_scores.append(0.0)

        # 2. Kinematik Hız-Eğrilik Ayrışması (2/3 Güç Yasası & Yavaş Çizim)
        kinematic_anomaly = False
        if q_smoothness >= self.smoothness_threshold_abs or (q_mean_vel < 1.5 and q_vel_var > 0.45):
            kinematic_anomaly = True
            severity = min(1.0, q_smoothness / 0.25)
            anomaly_scores.append(severity)
            details.append(
                f"Kinematik Hız Sapması: Çizgilerin doğal el yazısı hızında atılmadığı, "
                f"yavaşlatılmış ve kontrollü taklit hareketi ile çizildiği yönünde kinematik sapma belirlendi."
            )
        else:
            anomaly_scores.append(0.0)

        # 3. Kasıtlı Gizleme / Yapısal Sapma (Eğer referans karşılaştırma raporu mevcutsa)
        structural_disguise = False
        if comparison_report is not None:
            # Eğim farkını kontrol et
            for diff in comparison_report.significant_differences:
                if diff.feature_key == "slant_angle_deg" and diff.absolute_difference >= self.slant_disguise_threshold_deg:
                    structural_disguise = True
                    details.append(
                        f"Kasıtlı Eğim Değişimi (Disguise): Yazı eğiminde referans alışkanlıklara kıyasla "
                        f"{diff.absolute_difference:.1f}° düzeyinde radikal yön değişimi gözlendi."
                    )
                    anomaly_scores.append(0.8)
                    break
            
            # Anlamlı fark sayısına göre anomali katkısı
            num_sig = len(comparison_report.significant_differences)
            if num_sig >= 3:
                anomaly_scores.append(min(1.0, num_sig * 0.25))

        # Genel Anomali Skoru
        overall_score = float(np.mean(anomaly_scores)) if anomaly_scores else 0.0
        overall_score = round(float(min(1.0, max(0.0, overall_score))), 4)

        # Seviye Belirleme
        if overall_score >= 0.60 or (tremor_anomaly and kinematic_anomaly):
            level = IndicatorLevel.STRONG_INDICATORS
            if not details:
                details.append("Yazı yapısında ve motor hareketlerinde belirgin taklit/zorlanma anomalileri mevcuttur.")
        elif overall_score >= 0.30 or tremor_anomaly or structural_disguise:
            level = IndicatorLevel.POSSIBLE_INDICATORS
            if not details:
                details.append("Hafif düzeyde motor duraksaması ve kısmi yapısal farklılık gözlenmiştir.")
        else:
            level = IndicatorLevel.NO_STRONG_EVIDENCE
            details.append("Yazı akışı doğal, balistik nöromusküler hareketlerle uyumludur; belirgin bir anomali saptanmamıştır.")

        confidence = round(0.85 if comparison_report is not None else 0.70, 2)

        # Rapor Bütünlük Karması
        hash_payload = {
            "indicator_level": level.value,
            "overall_anomaly_score": overall_score,
            "tremor_anomaly": tremor_anomaly,
            "kinematic_anomaly": kinematic_anomaly,
            "structural_disguise": structural_disguise,
            "num_details": len(details),
        }
        serialized = json.dumps(hash_payload, sort_keys=True)
        audit_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        return SimulationIndicators(
            indicator_level=level,
            overall_anomaly_score=overall_score,
            tremor_anomaly_detected=tremor_anomaly,
            kinematic_decoupling_detected=kinematic_anomaly,
            structural_disguise_detected=structural_disguise,
            anomaly_details=details,
            confidence=confidence,
            audit_hash_sha256=audit_hash,
        )
