import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from cv.features.extractors import ForensicFeatureReport
from ml.models.comparison import ComparisonReport, ForensicComparisonEngine, DifferenceType


class HybridFusionResult(BaseModel):
    """Adli Denetimli Hibrit Füzyon Karar Raporu."""
    hybrid_similarity_score: float = Field(..., ge=0.0, le=1.0, description="Nihai hibrit adli benzerlik puanı S_hybrid")
    morphological_similarity: float = Field(..., ge=0.0, le=1.0, description="Morfolojik & topolojik benzerlik puanı")
    deep_metric_similarity: float = Field(..., ge=0.0, le=1.0, description="ConvNeXt-Tiny derin embedding kosinüs benzerliği")
    gating_penalty: float = Field(..., ge=0.0, description="Uygulanan adli çelişki ceza kesintisi")
    gating_triggered: bool = Field(..., description="Derin skoru kısıtlayan yapısal çelişki tespit edildi mi?")
    structural_contradictions: List[str] = Field(default_factory=list, description="Tespit edilen adli yapısal çelişkiler")
    concordance_level: str = Field(..., description="Adli uyum ve karar statüsü")
    forensic_explanation: str = Field(..., description="Bilirkişi için hibrit karar gerekçelendirmesi")
    audit_hash_sha256: str = Field(..., description="Hibrit karar bütünlük imzası")
    raw_hybrid_score: Optional[float] = Field(default=None, description="Kalibrasyon öncesi ham birleşik skor")
    calibration_status: Optional[str] = Field(default="CALIBRATED_PROBABILISTIC", description="Uygulanan skor kalibrasyon metodu")
    penalty_details: List[str] = Field(default_factory=list, description="Kademeli ceza ve normalizasyon detayları")


class HybridForensicFusion:
    """
    Adli Denetimli Hibrit Füzyon Katmanı (Forensic Gated Hybrid Fusion).
    
    Özellikler:
    - Morfolojik / Çizge öznitelik farkları (`ComparisonReport` / `ForensicFeatureReport`) ile 
      ConvNeXt-Tiny derin metrik embedding mesafesini birleştirir.
    - **Kademeli Adli Gating Mekanizması (Gradual Penalty Curve):** Sert eşik kesintileri yerine
      sapma şiddetiyle orantılı yumuşak üstel/sigmoidal ceza fonksiyonu uygular.
    - **Olasılıksal Skor Kalibrasyonu:** Ham mesafeleri adli karar destek aralıklarına (%85-%97 doğal uyum,
      %70-%84 kritik ayrım, %50-%69 bozma, %98+ tetabuk) kalibre eder.
    """

    def __init__(
        self,
        weight_morphological: float = 0.45,
        weight_deep: float = 0.55,
        critical_difference_threshold: int = 2,
    ):
        self.w_morph = weight_morphological
        self.w_deep = weight_deep
        self.critical_diff_thresh = critical_difference_threshold

        # Ağırlıkları normalize et
        total_w = self.w_morph + self.w_deep
        self.w_morph /= total_w
        self.w_deep /= total_w

    def detect_structural_contradictions(
        self, morphological_report: ComparisonReport
    ) -> List[str]:
        """
        Adli incelemede sahtecilik veya farklı yazar emaresi olan majör yapısal çelişkileri tespit eder.
        """
        contradictions = []

        for diff in morphological_report.significant_differences:
            key = diff.feature_key
            z = abs(diff.z_score)

            if "slant" in key and (z > 3.0 or diff.absolute_difference > 15.0):
                contradictions.append(
                    f"Zıt Eğim Çelişkisi: {diff.feature_name_tr} (Sapma: {diff.z_score:+.2f} sigma, Fark: {diff.absolute_difference:.1f}°)"
                )
            elif any(k in key for k in ["loop", "crossing", "junction", "branch", "node"]) and z > 2.5:
                contradictions.append(
                    f"Kavşak/Döngü Topolojisi Uyumsuzluğu: {diff.feature_name_tr} (Sapma: {diff.z_score:+.2f} sigma)"
                )
            elif "tremor" in key and z > 2.5:
                contradictions.append(
                    f"Kinematik Titreme/Tremor Sapması: {diff.feature_name_tr} (Sapma: {diff.z_score:+.2f} sigma)"
                )
            elif "baseline" in key and z > 3.0:
                contradictions.append(
                    f"Taban Çizgisi Sapması: {diff.feature_name_tr} (Sapma: {diff.z_score:+.2f} sigma)"
                )
            elif z > 3.5:
                contradictions.append(
                    f"Kritik Morfolojik Sapma: {diff.feature_name_tr} (Sapma: {diff.z_score:+.2f} sigma)"
                )

        return contradictions

    def compute_gradual_gating(
        self,
        morphological_report: ComparisonReport,
        s_morph: float,
        s_deep: float,
    ) -> Tuple[float, float, bool, List[str], List[str]]:
        """
        Kademeli ceza eğrisi ve adaptif derin model sönümlemesi.
        """
        contradictions = self.detect_structural_contradictions(morphological_report)
        sig_diffs = morphological_report.significant_differences
        num_contradictions = len(contradictions)
        num_sig_diffs = len(sig_diffs)

        penalty_details: List[str] = []

        if num_contradictions == 0 and num_sig_diffs == 0:
            return s_deep, 0.0, False, [], ["Gating cezası uygulanmadı (Tam morfolojik uyum)"]

        # Sürekli sapma şiddeti hesabı
        total_severity = 0.0
        for diff in sig_diffs:
            z_excess = max(0.0, abs(diff.z_score) - 2.5)
            total_severity += z_excess
            if z_excess > 0.1:
                penalty_details.append(f"{diff.feature_name_tr}: Sapma {diff.z_score:+.2f}σ")

        contradiction_severity = num_contradictions * 3.0
        discrepancy = total_severity + contradiction_severity

        # 1. Kademeli Derin Model Sönümleme
        damping_factor = 1.0 - math.exp(-0.25 * discrepancy)
        max_deep_reduction = 0.65 if num_contradictions > 0 else 0.40
        s_deep_effective = s_deep * (1.0 - max_deep_reduction * damping_factor)

        # 2. Kademeli Ceza Kesintisi
        max_penalty = 0.38 if num_contradictions > 0 else 0.22
        gating_penalty = max_penalty * (1.0 - math.exp(-0.20 * discrepancy))
        gating_penalty = round(float(gating_penalty), 4)

        gating_triggered = gating_penalty > 0.01 or num_contradictions > 0

        if gating_triggered:
            penalty_details.append(f"Kademeli Ceza: -%{gating_penalty * 100:.1f} (Sapma Şiddeti: {discrepancy:.2f})")

        return s_deep_effective, gating_penalty, gating_triggered, contradictions, penalty_details

    def calibrate_similarity_score(
        self,
        raw_score: float,
        s_morph: float,
        s_deep: float,
        gating_triggered: bool,
    ) -> float:
        """
        Ham birleşik skoru adli standartlara ve doğal varyasyon olasılık aralıklarına kalibre eder.
        """
        # Tetabuk / İzdüşüm çakışması
        if raw_score >= 0.98:
            return round(float(min(1.0, raw_score)), 4)

        # Doğal varyasyonlu yüksek uyum bölgesi (gating yok ve iki model de uyumlu)
        if not gating_triggered and s_morph >= 0.80 and s_deep >= 0.80:
            # 0.85 - 0.97 aralığına yumuşak haritalama
            scaled = 0.85 + (raw_score - 0.80) * (0.12 / 0.18)
            calibrated = min(0.975, max(0.85, scaled))
            return round(float(calibrated), 4)

        return round(float(min(1.0, max(0.0, raw_score))), 4)

    def fuse(
        self,
        morphological_report: Union[ComparisonReport, ForensicFeatureReport, Dict[str, Any]],
        deep_similarity_score: float,
        reference_reports: Optional[List[Union[ForensicFeatureReport, Dict[str, Any]]]] = None,
    ) -> HybridFusionResult:
        """
        Morfolojik rapor ve derin öğrenme benzerlik skorunu kademeli adli gating ve kalibrasyonla birleştirir.
        """
        # Eğer doğrudan ForensicFeatureReport verilmişse ComparisonEngine ile rapor üret
        if not isinstance(morphological_report, ComparisonReport):
            engine = ForensicComparisonEngine()
            if reference_reports and len(reference_reports) > 0:
                morphological_report = engine.compare(morphological_report, reference_reports)
            else:
                # Tekil karşılaştırma fallback
                morphological_report = engine.compare(morphological_report, [morphological_report])

        s_morph = float(min(1.0, max(0.0, morphological_report.similarity_score)))
        if math.isnan(deep_similarity_score) or math.isinf(deep_similarity_score):
            s_deep = 0.5
        else:
            s_deep = float(min(1.0, max(0.0, deep_similarity_score)))

        num_sig_diffs = len(morphological_report.significant_differences)

        # Kademeli Gating Hesaplama
        (
            s_deep_effective,
            gating_penalty,
            gating_triggered,
            structural_contradictions,
            penalty_details,
        ) = self.compute_gradual_gating(morphological_report, s_morph, s_deep)

        # Ağırlıklı birleşim ve gating ceza kesintisi
        base_score = (self.w_morph * s_morph) + (self.w_deep * s_deep_effective)
        raw_final_score = base_score - gating_penalty
        raw_final_score = round(float(min(1.0, max(0.0, raw_final_score))), 4)

        # Olasılıksal Skor Kalibrasyonu
        final_score = self.calibrate_similarity_score(
            raw_score=raw_final_score,
            s_morph=s_morph,
            s_deep=s_deep,
            gating_triggered=gating_triggered,
        )

        # --- Adli Kanaat ve Karar Seviyesi ---
        if final_score >= 0.98:
            concordance = "⚠️ ANORMAL BİREBİR ÇAKIŞMA (TETABUK / DOLANDIRICILIK VE İZDÜŞÜM ŞÜPHESİ)"
            explanation = (
                f"Hibrit Analiz Sonucu: Benzerlik skoru (%{final_score*100:.1f}) insan biyomekaniği ve doğal varyasyon "
                "sınırlarını aşan derecede aşırı yüksektir. İki imzanın geometrik olarak %98+ birebir çakışması (tetabuk), "
                "samimi imza atımında imkansız olup orijinal bir belgeden ışıklı masa, fotokopi, asetat veya dijital "
                "montaj yoluyla aktarıldığına (İzdüşüm Sahteciliği ve Dolandırıcılık Şüphesi) işaret etmektedir."
            )
        elif final_score >= 0.85 and not gating_triggered:
            concordance = "KUVVETLİ ORİJİNAL DESTEĞİ (DOĞAL VARYASYON & SAMİMİ AİDİYET)"
            explanation = (
                f"Hibrit Analiz Sonucu: Hem morfolojik öznitelikler (%{s_morph*100:.1f}) hem de "
                f"ConvNeXt-Tiny derin metrik embedding'leri (%{s_deep*100:.1f}) referans yazarın "
                "doğal varyasyon sınırları içerisinde tam uyumludur. Balistik akış doğaldır."
            )
        elif final_score >= 0.70:
            concordance = "KRİTİK AYRIM (NİTELİKLİ TAKLİT ŞÜPHESİ / MİKRO ANALİZ ŞART)"
            explanation = (
                f"Hibrit Analiz Sonucu: Genel profil benzerlik göstermektedir (Skor: %{final_score*100:.1f}). "
                "Usta bir taklitçinin serbest taklidi veya deformasyon ayrımı için FFT titreme ve 2/3 Güç Yasası kontrolleri kritiktir."
            )
        elif final_score >= 0.50:
            concordance = "KARARSIZ / OLASI İSTEKLİ BOZMA (DISGUISE)"
            explanation = (
                f"Hibrit Analiz Sonucu: Morfolojik göstergeler ile derin model çıktısı arasında kısmi tutarsızlık mevcuttur (Skor: %{final_score*100:.1f}). "
                "Şahsın kasten değiştirmesi veya acemi taklit ihtimaline karşı ek mukayese belgeleri önerilir."
            )
        elif final_score >= 0.25:
            concordance = "KUVVETLİ SAHTECİLİK / FARKLI YAZAR DESTEĞİ (KABA TAKLİT)"
            explanation = (
                f"Hibrit Analiz Sonucu: Temel morfolojik yapı ve derin metrik mesafesi referans yazar profilinden uzaktır (Skor: %{final_score*100:.1f}). "
                "Farklı bir şahsın eli ürünü olduğu değerlendirilmektedir."
            )
        else:
            concordance = "TAM UYUMSUZLUK / RASTGELE SAHTECİLİK (KESİN FARKLI KAYNAK)"
            if gating_triggered:
                reasons_str = "; ".join(structural_contradictions[:2]) if structural_contradictions else f"{num_sig_diffs} adet kritik fark"
                explanation = (
                    f"Hibrit Analiz Sonucu: Derin metrik benzerliği (%{s_deep*100:.1f}) olsa dahi, "
                    f"tespit edilen yapısal çelişkiler ({reasons_str}) sebebiyle adli gating kalkanı devreye girmiş "
                    f"ve nihai skor %{final_score*100:.1f} seviyesine indirilmiştir. Belgenin sahte/taklit olma ihtimali kesindir."
                )
            else:
                explanation = (
                    f"Hibrit Analiz Sonucu: Hem morfolojik farklar hem de derin metrik mesafesi "
                    f"referans yazar profilinden tamamen ayrışmaktadır (Skor: %{final_score*100:.1f})."
                )

        # Rapor bütünlük imzası
        payload = {
            "hybrid_similarity_score": final_score,
            "morphological_similarity": s_morph,
            "deep_metric_similarity": s_deep,
            "gating_penalty": round(gating_penalty, 4),
            "gating_triggered": gating_triggered,
            "num_significant_diffs": num_sig_diffs,
            "structural_contradictions": structural_contradictions,
            "concordance_level": concordance,
        }
        serialized = json.dumps(payload, sort_keys=True)
        audit_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        return HybridFusionResult(
            hybrid_similarity_score=final_score,
            morphological_similarity=round(s_morph, 4),
            deep_metric_similarity=round(s_deep, 4),
            gating_penalty=round(gating_penalty, 4),
            gating_triggered=gating_triggered,
            structural_contradictions=structural_contradictions,
            concordance_level=concordance,
            forensic_explanation=explanation,
            audit_hash_sha256=audit_hash,
            raw_hybrid_score=raw_final_score,
            calibration_status="CALIBRATED_PROBABILISTIC",
            penalty_details=penalty_details,
        )
