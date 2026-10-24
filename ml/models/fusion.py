import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

from enum import Enum
from cv.features.extractors import ForensicFeatureReport
from ml.models.comparison import ComparisonReport, ForensicComparisonEngine, DifferenceType


class ScoreStatus(str, Enum):
    """Adli benzerlik skoru hesaplama ve güvenilirlik durumu (Sprint 21)."""
    VALID_SCORE = "VALID_SCORE"
    REAL_LOW_SIMILARITY = "REAL_LOW_SIMILARITY"
    CALCULATION_FAILURE = "CALCULATION_FAILURE"
    INSUFFICIENT_REFERENCE_DATA = "INSUFFICIENT_REFERENCE_DATA"
    PREPROCESSING_FAILURE = "PREPROCESSING_FAILURE"
    INVALID_EMBEDDING = "INVALID_EMBEDDING"
    GATING_REJECTION = "GATING_REJECTION"


class ScoreProvenance(BaseModel):
    """Her analiz için tam izlenebilir adli skor kökeni ve hesaplama bileşenleri (Sprint 22)."""
    raw_cosine: Optional[float] = Field(default=None, description="Ham kosinüs benzerliği [-1.0, +1.0]")
    normalized_similarity: Optional[float] = Field(default=None, description="Normalize derin benzerlik skoru [0.0, 1.0]")
    deep_similarity_raw: Optional[float] = Field(default=None, description="Geriye dönük uyumluluk: derin benzerlik")
    morphological_score: Optional[float] = Field(default=None, description="13 boyutlu morfolojik Mahalanobis benzerliği")
    morphological_similarity: Optional[float] = Field(default=None, description="Geriye dönük uyumluluk: morfolojik benzerlik")
    mahalanobis_distance: Optional[float] = Field(default=None, description="Mahalanobis varyans mesafesi")
    gating_weight: Optional[float] = Field(default=None, description="Uygulanan sürekli gating derin model ağırlığı")
    calibrated_score: Optional[float] = Field(default=None, description="Kalibre edilmiş nihai adli skor")
    final_score: Optional[float] = Field(default=None, description="Karara esas nihai benzerlik skoru")
    reference_count: Optional[int] = Field(default=1, description="İncelemede kullanılan mukayese referans sayısı")
    embedding_dimension: int = Field(default=128, description="Derin öznitelik vektör boyutu")
    embedding_norm: float = Field(default=1.0, description="L2 norm değeri (birim küre)")
    preprocessing: Dict[str, Any] = Field(
        default_factory=lambda: {"deskew": False, "size": [224, 224], "normalization": "ImageNet"},
        description="Ön işleme sözleşme parametreleri",
    )
    status: ScoreStatus = Field(default=ScoreStatus.VALID_SCORE, description="Skor hesaplama güvenilirlik durumu")
    reason: Optional[str] = Field(default=None, description="Durum gerekçesi veya başarısızlık detayı")



class HybridFusionResult(BaseModel):
    """Adli Denetimli Hibrit Füzyon Karar Raporu."""
    hybrid_similarity_score: float = Field(..., ge=0.0, le=1.0, description="Nihai hibrit adli benzerlik puanı S_hybrid")
    morphological_similarity: float = Field(..., ge=0.0, le=1.0, description="Morfolojik & topolojik benzerlik puanı")
    deep_metric_similarity: float = Field(..., ge=0.0, le=1.0, description="ConvNeXt-Tiny derin embedding kosinüs benzerliği")
    gating_penalty: float = Field(..., ge=0.0, description="Uygulanan adli çelişki ceza kesintisi")
    gating_weight: float = Field(default=0.55, ge=0.0, le=1.0, description="Dinamik sürekli derin model ağırlığı (Continuous Gating)")
    gating_triggered: bool = Field(..., description="Derin skoru kısıtlayan yapısal çelişki tespit edildi mi?")
    structural_contradictions: List[str] = Field(default_factory=list, description="Tespit edilen adli yapısal çelişkiler")
    concordance_level: str = Field(..., description="Adli uyum ve karar statüsü")
    forensic_explanation: str = Field(..., description="Bilirkişi için hibrit karar gerekçelendirmesi")
    audit_hash_sha256: str = Field(..., description="Hibrit karar bütünlük imzası")
    raw_hybrid_score: Optional[float] = Field(default=None, description="Kalibrasyon öncesi ham birleşik skor")
    penalty_details: List[str] = Field(default_factory=list, description="Kademeli ceza ve normalizasyon detayları")
    score_provenance: Optional[ScoreProvenance] = Field(default=None, description="Tam denetlenebilir adli skor provenance nesnesi")
    gating_audit_log: Optional[Dict[str, Any]] = Field(default=None, description="Adli gating tetiklenme ve ağırlık audit logu")


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
        Sprint 31 Düzeltmesi: Ölçek ve küçük örneklem (N<3) kaynaklı mikro farkların sahtecilik
        çelişkisi olarak etiketlenmesi engellenmiştir.
        """
        contradictions = []
        ref_count = getattr(morphological_report, "reference_count", 1)

        for diff in morphological_report.significant_differences:
            key = diff.feature_key
            z = abs(diff.z_score)

            # Eğim çelişkisi: hem istatistiksel z > 3.5 hem de açısal fark > 18° olmalıdır
            if "slant" in key and (z > 3.5 and diff.absolute_difference > 18.0):
                contradictions.append(
                    f"Zıt Eğim Çelişkisi: {diff.feature_name_tr} (Sapma: {diff.z_score:+.2f}σ, Fark: {diff.absolute_difference:.1f}°)"
                )
            elif "loop" in key and z > 3.0 and diff.absolute_difference >= 2.0:
                contradictions.append(
                    f"Döngü Topolojisi Uyumsuzluğu: {diff.feature_name_tr} (Sapma: {diff.z_score:+.2f}σ, Fark: {diff.absolute_difference:.0f} adet)"
                )
            elif "crossing" in key and z > 3.5 and diff.absolute_difference >= 5.0:
                contradictions.append(
                    f"Kavşak Topolojisi Uyumsuzluğu: {diff.feature_name_tr} (Sapma: {diff.z_score:+.2f}σ)"
                )
            elif "tremor" in key and z > 3.0:
                contradictions.append(
                    f"Kinematik Titreme/Tremor Sapması: {diff.feature_name_tr} (Sapma: {diff.z_score:+.2f}σ)"
                )
            elif ref_count >= 3 and z > 4.0:
                # Sadece referans sayısı yeterli (N>=3) iken piksel dışı genel z > 4.0 sapma majör çelişki sayılır
                contradictions.append(
                    f"Kritik Morfolojik Sapma: {diff.feature_name_tr} (Sapma: {diff.z_score:+.2f}σ)"
                )

        return contradictions

    def compute_gradual_gating(
        self,
        morphological_report: ComparisonReport,
        s_morph: float,
        s_deep: float,
    ) -> Tuple[float, float, float, bool, List[str], List[str]]:
        """
        Sürekli adli gating mekanizması (Continuous Gating).
        Asla derin skoru sıfırla çarpmaz veya skoru doğrudan ceza çıkarma ile sıfıra kilitlemez.
        Çelişki şiddetine göre derin ve morfolojik modeller arasındaki denge ağırlığını ayarlar.
        Döner:
            (s_deep_effective, gating_penalty, gating_weight, gating_triggered, contradictions, penalty_details)
        """
        contradictions = self.detect_structural_contradictions(morphological_report)
        sig_diffs = morphological_report.significant_differences
        num_contradictions = len(contradictions)
        num_sig_diffs = len(sig_diffs)

        penalty_details: List[str] = []

        if num_contradictions == 0 and num_sig_diffs == 0:
            return s_deep, 0.0, self.w_deep, False, [], ["Gating cezası uygulanmadı (Tam morfolojik uyum)"]

        # Sürekli sapma şiddeti hesabı
        total_severity = 0.0
        for diff in sig_diffs:
            z_excess = max(0.0, abs(diff.z_score) - 2.5)
            total_severity += z_excess
            if z_excess > 0.1:
                penalty_details.append(f"{diff.feature_name_tr}: Sapma {diff.z_score:+.2f}σ")

        contradiction_severity = num_contradictions * 3.0
        discrepancy = total_severity + contradiction_severity

        # Sprint 31: Yetersiz referansta (N<3) derin modeli aşırı cezalandırmayı önle (min w_deep = 0.35 - 0.40)
        ref_count = getattr(morphological_report, "reference_count", 1)
        min_w_deep = 0.40 if ref_count < 3 else 0.30

        decay = math.exp(-0.15 * discrepancy)
        gating_weight = max(min_w_deep, min(0.80, self.w_deep * (0.35 + 0.65 * decay)))
        gating_weight = round(float(gating_weight), 4)

        # 2. Gösterge Ceza Miktarı (Log ve Adli Açıklama İçin)
        max_penalty = 0.35 if num_contradictions > 0 else 0.20
        gating_penalty = max_penalty * (1.0 - math.exp(-0.20 * discrepancy))
        gating_penalty = round(float(gating_penalty), 4)

        gating_triggered = gating_penalty > 0.01 or num_contradictions > 0

        if gating_triggered:
            penalty_details.append(
                f"Sürekli Gating: Derin Ağırlığı %{gating_weight * 100:.1f}, "
                f"Morfolojik Ağırlık %{(1.0 - gating_weight) * 100:.1f} (Sapma Şiddeti: {discrepancy:.2f})"
            )

        return s_deep, gating_penalty, gating_weight, gating_triggered, contradictions, penalty_details

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
        if raw_score >= 0.985:
            return round(float(min(1.0, raw_score)), 4)

        # Doğal varyasyonlu yüksek uyum bölgesi (gating yok ve iki model de uyumlu)
        if not gating_triggered and s_morph >= 0.60 and s_deep >= 0.75:
            # 0.82 - 0.97 aralığına yumuşak haritalama
            scaled = 0.82 + (raw_score - 0.65) * (0.15 / 0.33)
            calibrated = min(0.975, max(0.82, scaled))
            return round(float(calibrated), 4)

        return round(float(min(1.0, max(0.0, raw_score))), 4)

    def fuse(
        self,
        morphological_report: Union[ComparisonReport, ForensicFeatureReport, Dict[str, Any]],
        deep_similarity_score: float,
        reference_reports: Optional[List[Union[ForensicFeatureReport, Dict[str, Any]]]] = None,
        raw_cosine: Optional[float] = None,
    ) -> HybridFusionResult:
        """
        Morfolojik rapor ve derin öğrenme benzerlik skorunu sürekli adli gating ve kalibrasyonla birleştirir.
        """
        # Eğer doğrudan ForensicFeatureReport verilmişse ComparisonEngine ile rapor üret
        if not isinstance(morphological_report, ComparisonReport):
            engine = ForensicComparisonEngine()
            if reference_reports and len(reference_reports) > 0:
                morphological_report = engine.compare(morphological_report, reference_reports)
            else:
                morphological_report = engine.compare(morphological_report, [morphological_report])

        s_morph = float(min(1.0, max(0.0, morphological_report.similarity_score)))
        if math.isnan(deep_similarity_score) or math.isinf(deep_similarity_score):
            s_deep = 0.5
            deep_is_valid = False
            raw_cos_val = None
        else:
            s_deep = float(min(1.0, max(0.0, deep_similarity_score)))
            deep_is_valid = True
            raw_cos_val = round(float(raw_cosine), 4) if raw_cosine is not None else round(float((s_deep * 2.0) - 1.0), 4)

        num_sig_diffs = len(morphological_report.significant_differences)

        # Sürekli Gating Hesaplama
        (
            s_deep_effective,
            gating_penalty,
            gating_weight,
            gating_triggered,
            structural_contradictions,
            penalty_details,
        ) = self.compute_gradual_gating(morphological_report, s_morph, s_deep)

        # Sürekli Ağırlıklı Füzyon: S_raw = w_g * S_deep + (1 - w_g) * S_morph
        raw_final_score = (gating_weight * s_deep) + ((1.0 - gating_weight) * s_morph)
        raw_final_score = round(float(min(1.0, max(0.0, raw_final_score))), 4)

        # Olasılıksal Skor Kalibrasyonu
        final_score = self.calibrate_similarity_score(
            raw_score=raw_final_score,
            s_morph=s_morph,
            s_deep=s_deep,
            gating_triggered=gating_triggered,
        )

        # Adli Kanaat ve Karar Seviyesi
        if final_score >= 0.985:
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
            if gating_triggered:
                explanation = (
                    f"Hibrit Analiz Sonucu: Yapısal çelişkiler tespit edilmiş ve adli gating kalkanı devreye girmiştir (Skor: %{final_score*100:.1f}). "
                    "Farklı bir şahsın eli ürünü olduğu değerlendirilmektedir."
                )
            else:
                explanation = (
                    f"Hibrit Analiz Sonucu: Temel morfolojik yapı ve derin metrik mesafesi referans yazar profilinden uzaktır (Skor: %{final_score*100:.1f}). "
                    "Farklı bir şahsın eli ürünü olduğu değerlendirilmektedir."
                )
        else:
            concordance = "TAM UYUMSUZLUK / RASTGELE SAHTECİLİK (KESİN FARKLI KAYNAK)"
            if gating_triggered:
                reasons_str = "; ".join(structural_contradictions[:2]) if structural_contradictions else f"{num_sig_diffs} adet kritik fark"
                explanation = (
                    f"Hibrit Analiz Sonucu: Yapısal çelişkiler ({reasons_str}) tespit edilmiş, adli gating kalkanı devreye girmiş "
                    f"ve derin model (%{s_deep*100:.1f}) ile morfolojik analiz (%{s_morph*100:.1f}) dengelenerek "
                    f"nihai skor %{final_score*100:.1f} seviyesinde hesaplanmıştır. Belgenin sahte/taklit olma ihtimali yüksektir."
                )
            else:
                explanation = (
                    f"Hibrit Analiz Sonucu: Hem morfolojik farklar hem de derin metrik mesafesi "
                    f"referans yazar profilinden tamamen ayrışmaktadır (Skor: %{final_score*100:.1f})."
                )

        # Score Provenance Durumu Belirleme (Sprint 21 Section 27-28)
        if not deep_is_valid:
            score_status = ScoreStatus.INVALID_EMBEDDING
            reason_str = "Derin model embedding çıktısı geçersiz (NaN/Inf)"
        elif morphological_report.reference_count == 1:
            score_status = ScoreStatus.VALID_SCORE
            reason_str = "Tek referanslı analiz tamamlandı (genişletilmiş adli prior devrede)"
        elif final_score < 0.25:
            score_status = ScoreStatus.REAL_LOW_SIMILARITY
            reason_str = "Gerçek düşük benzerlik: hem derin metrik hem morfolojik öznitelikler belirgin farklı"
        elif gating_triggered and final_score < 0.50:
            score_status = ScoreStatus.GATING_REJECTION
            reason_str = f"Yapısal çelişki kaynaklı gating sönümlemesi: {len(structural_contradictions)} çelişki"
        else:
            score_status = ScoreStatus.VALID_SCORE
            reason_str = "Doğrulanmış adli skor hesabı"

        ref_cnt = getattr(morphological_report, "reference_count", 1)
        score_provenance = ScoreProvenance(
            raw_cosine=raw_cos_val,
            normalized_similarity=round(s_deep, 4),
            deep_similarity_raw=round(s_deep, 4),
            morphological_score=round(s_morph, 4),
            morphological_similarity=round(s_morph, 4),
            mahalanobis_distance=round(morphological_report.mahalanobis_distance, 4),
            gating_weight=round(gating_weight, 4),
            calibrated_score=round(final_score, 4),
            final_score=round(final_score, 4),
            reference_count=ref_cnt,
            embedding_dimension=128,
            embedding_norm=1.0,
            preprocessing={"deskew": False, "size": [224, 224], "normalization": "ImageNet"},
            status=score_status,
            reason=reason_str,
        )

        # Sprint 31: Adli Gating Denetim ve İzlenebilirlik Logu
        gating_audit_log = {
            "gating_triggered": gating_triggered,
            "rules_triggered": structural_contradictions,
            "discrepancy_score": round(float(discrepancy), 2) if 'discrepancy' in locals() else 0.0,
            "pre_gating": {
                "s_deep": round(float(s_deep), 4),
                "s_morph": round(float(s_morph), 4),
                "initial_w_deep": self.w_deep,
                "initial_w_morph": self.w_morph,
            },
            "post_gating": {
                "effective_gating_weight": round(float(gating_weight), 4),
                "gating_penalty": round(float(gating_penalty), 4),
                "raw_fused_score": round(float(raw_final_score), 4),
            },
            "calibrated_final_score": round(float(final_score), 4),
            "ui_percentage": round(float(final_score * 100.0), 1),
            "reference_count": ref_cnt,
            "covariance_method": getattr(morphological_report, "covariance_method", "N/A"),
            "condition_number": getattr(morphological_report, "condition_number", None),
        }

        # Rapor bütünlük imzası
        payload = {
            "hybrid_similarity_score": final_score,
            "morphological_similarity": s_morph,
            "deep_metric_similarity": s_deep,
            "gating_penalty": round(gating_penalty, 4),
            "gating_weight": round(gating_weight, 4),
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
            gating_weight=round(gating_weight, 4),
            gating_triggered=gating_triggered,
            structural_contradictions=structural_contradictions,
            concordance_level=concordance,
            forensic_explanation=explanation,
            audit_hash_sha256=audit_hash,
            raw_hybrid_score=raw_final_score,
            calibration_status="CALIBRATED_PROBABILISTIC",
            penalty_details=penalty_details,
            score_provenance=score_provenance,
            gating_audit_log=gating_audit_log,
        )
