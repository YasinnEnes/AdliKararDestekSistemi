import hashlib
import json
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from cv.preprocessing.suitability import SuitabilityResult
from ml.models.fusion import HybridFusionResult


class ENFSIConclusionScale(str, Enum):
    """
    ENFSI (European Network of Forensic Science Institutes) 
    Bayesyen Kanıt Gücü ve Hipotez Destek Düzeyleri.
    """
    EXTREMELY_STRONG_SUPPORT_FOR_H1 = "EXTREMELY_STRONG_SUPPORT_FOR_H1" # log10(LR) >= 4.0  (LR >= 10,000)
    VERY_STRONG_SUPPORT_FOR_H1 = "VERY_STRONG_SUPPORT_FOR_H1"           # 3.0 <= log10(LR) < 4.0 (LR 1,000 - 10,000)
    STRONG_SUPPORT_FOR_H1 = "STRONG_SUPPORT_FOR_H1"                     # 2.0 <= log10(LR) < 3.0 (LR 100 - 1,000)
    MODERATE_SUPPORT_FOR_H1 = "MODERATE_SUPPORT_FOR_H1"                 # 1.0 <= log10(LR) < 2.0 (LR 10 - 100)
    WEAK_SUPPORT_FOR_H1 = "WEAK_SUPPORT_FOR_H1"                         # 0.5 <= log10(LR) < 1.0 (LR 3 - 10)
    INCONCLUSIVE = "INCONCLUSIVE"                                       # -0.5 < log10(LR) < 0.5 (LR ~ 1.0)
    WEAK_SUPPORT_FOR_H2 = "WEAK_SUPPORT_FOR_H2"                         # -1.0 < log10(LR) <= -0.5 (LR 0.1 - 0.3)
    MODERATE_SUPPORT_FOR_H2 = "MODERATE_SUPPORT_FOR_H2"                 # -2.0 < log10(LR) <= -1.0 (LR 0.01 - 0.1)
    STRONG_SUPPORT_FOR_H2 = "STRONG_SUPPORT_FOR_H2"                     # -3.0 < log10(LR) <= -2.0 (LR 0.001 - 0.01)
    VERY_STRONG_SUPPORT_FOR_H2 = "VERY_STRONG_SUPPORT_FOR_H2"           # log10(LR) <= -3.0 (LR <= 0.001)


class ForensicSimilarityBand(str, Enum):
    """
    Kriminalojik ve Adli Belge İnceleme Standartlarında 6 Kademeli Benzerlik Bandı.
    (Doğal Varyasyon vs. Tetabuk Paradoksu ve Taklit Analizi)
    """
    SUPERIMPOSITION_ANOMALY = "SUPERIMPOSITION_ANOMALY"       # %98 - %100: Tetabuk / İzdüşüm / Sahtecilik Riski
    NATURAL_VARIATION_MATCH = "NATURAL_VARIATION_MATCH"       # %85 - %97.9: Doğal Varyasyon & Orijinal Alan (H1)
    SKILLED_SIMULATION_ZONE = "SKILLED_SIMULATION_ZONE"       # %70 - %84.9: Kritik Ayrım (Nitelikli Taklit / Deforme Orijinal)
    INCONCLUSIVE_DISGUISE_ZONE = "INCONCLUSIVE_DISGUISE_ZONE" # %50 - %69.9: Kararsız (İstekli Bozma / Amatör Taklit)
    LOW_SIMILARITY_NON_SOURCE = "LOW_SIMILARITY_NON_SOURCE"   # %25 - %49.9: Düşük Benzerlik (Kaba Taklit / Farklı Şahıs - H2)
    COMPLETE_DISPARITY = "COMPLETE_DISPARITY"                 # %0 - %24.9: Tam Uyumsuzluk (Rastgele Sahtecilik - H2)


class ForensicSimilarityScaleEvaluation(BaseModel):
    """Kriminalojik Benzerlik ve Tetabuk Risk Değerlendirmesi."""
    band: ForensicSimilarityBand = Field(..., description="Kriminalojik benzerlik bandı")
    band_label_tr: str = Field(..., description="Türkçe adli benzerlik bant başlığı")
    similarity_percentage: float = Field(..., description="Değerlendirilen nihai hibrit benzerlik yüzdesi")
    min_pct: float = Field(..., description="Bandın asgari yüzde sınırı")
    max_pct: float = Field(..., description="Bandın azami yüzde sınırı")
    risk_category: str = Field(..., description="Adli risk kategorisi")
    forensic_interpretation_tr: str = Field(..., description="Kriminalojik ve adli tıp açıklaması")
    is_superimposition_anomaly: bool = Field(default=False, description="Birebir çakışma (tetabuk) anomalisi var mı?")
    requires_micro_inspection: bool = Field(default=False, description="Mikroskobik hat ve dinamik doğrulama zorunlu mu?")


class HypothesisEvaluationReport(BaseModel):
    """ENFSI Bayesyen Hipotez Değerlendirme ve Belirsizlik Raporu."""
    support_level: ENFSIConclusionScale = Field(..., description="ENFSI sözel kanıt gücü düzeyi")
    log_likelihood_ratio: float = Field(..., description="Log10(LR) kanıt gücü skoru")
    likelihood_ratio_estimate: float = Field(..., description="Likelihood Ratio kestirimi P(E|H1)/P(E|H2)")
    uncertainty_score: float = Field(..., ge=0.0, le=1.0, description="0.0 - 1.0 arası toplam belirsizlik skoru (U)")
    confidence_interval_log_lr: Tuple[float, float] = Field(..., description="Log10(LR) için %95 güven aralığı")
    propositions: Dict[str, str] = Field(..., description="Değerlendirilen H1 ve H2 adli önermeleri")
    verbal_conclusion_tr: str = Field(..., description="Mahkeme ve adli makamlar için Türkçe sözel kanaat ifadesi")
    forensic_limitations: List[str] = Field(default_factory=list, description="Adli kısıtlar ve ihtiyati şerhler")
    similarity_scale_evaluation: Optional[ForensicSimilarityScaleEvaluation] = Field(
        default=None, description="6 Kademeli Kriminalojik Benzerlik ve Tetabuk Analizi"
    )
    audit_hash_sha256: str = Field(..., description="Hipotez değerlendirme bütünlük imzası")


class ForensicHypothesisEvaluator:
    """
    ENFSI Standartlarında Bayesyen Hipotez Değerlendirme ve Belirsizlik Motoru.
    
    Önermeler:
    - H1 (Prosecution / Source Hypothesis): Sorgulanan yazıyı şüpheli/referans yazar X yazmıştır.
    - H2 (Defense / Alternative Hypothesis): Sorgulanan yazıyı X dışındaki meçhul bir şahıs veya taklitçi yazmıştır.
    
    Likelihood Ratio (Olasılık Oranı):
    LR = P(E | H1) / P(E | H2)
    
    Belirsizlik Modellemesi:
    Numune adedi (N_ref), yazar içi varyans, görüntü kalitesi ve yapısal çelişki cezalarını 
    birleştirerek [0.0, 1.0] aralığında belirsizlik puanı (Uncertainty Score) ve güven aralığı hesaplar.
    """

    def __init__(
        self,
        nominal_scaling_k: float = 9.0,
        min_reliable_ref_count: int = 3,
    ):
        self.k = nominal_scaling_k
        self.min_ref_count = min_reliable_ref_count

    def evaluate(
        self,
        hybrid_result: HybridFusionResult,
        reference_count: int,
        suitability_result: Optional[SuitabilityResult] = None,
        suspect_name: str = "Şüpheli / Referans Şahıs",
    ) -> HypothesisEvaluationReport:
        """
        Hibrit benzerlik puanı, referans sayısı ve görüntü kalitesini birleştirerek Bayesyen rapor üretir.
        """
        s_hybrid = hybrid_result.hybrid_similarity_score
        limitations: List[str] = []

        # ----------------------------------------------------------------------
        # 1. Belirsizlik (Uncertainty) Modellemesi
        # ----------------------------------------------------------------------
        # A. Numune Adedi Belirsizliği (N < 3 ise yüksek belirsizlik)
        if reference_count <= 1:
            u_sample = 0.40
            limitations.append(
                f"KRİTİK KISIT: Yalnızca {reference_count} adet referans numune incelenmiştir (Beklenen min: {self.min_ref_count}). "
                "Yazarın doğal varyasyon sınırları tam temsil edilemediğinden sonuçlar ihtiyati kısıtla değerlendirilmelidir."
            )
        elif reference_count == 2:
            u_sample = 0.22
            limitations.append(
                f"KISIT: 2 adet referans numune incelenmiştir. Güvenilirliği artırmak için ek mukayese belgeleri önerilir."
            )
        elif reference_count < 5:
            u_sample = 0.10
        else:
            u_sample = 0.04

        # B. Görüntü Kalite Belirsizliği
        if suitability_result is not None:
            q_score = suitability_result.overall_score
            u_quality = max(0.0, (1.0 - q_score) * 0.25)
            if q_score < 0.70:
                limitations.append(
                    f"Görüntü kalitesi / netlik puanı (%{q_score*100:.1f}) sebebiyle mikroskobik hat analizinde belirsizlik artmıştır."
                )
        else:
            u_quality = 0.08

        # C. Adli Gating & Çelişki Belirsizliği
        u_gating = 0.15 if hybrid_result.gating_triggered else 0.0
        if hybrid_result.gating_triggered:
            limitations.append(
                "Morfolojik göstergeler ile derin model çıktısı arasında yapısal çelişki saptanmış ve adli kalkan uygulanmıştır."
            )

        total_uncertainty = round(float(min(0.95, max(0.05, u_sample + u_quality + u_gating))), 4)

        # ----------------------------------------------------------------------
        # 2. Kriminalojik Benzerlik ve Tetabuk (Birebir Çakışma) Analizi
        # ----------------------------------------------------------------------
        pct = round(s_hybrid * 100, 1)
        is_tetabuk = False
        req_micro = False

        if s_hybrid >= 0.98:
            is_tetabuk = True
            req_micro = True
            band = ForensicSimilarityBand.SUPERIMPOSITION_ANOMALY
            band_label = "⚠️ Anormal Çakışma (Tetabuk / İzdüşüm)"
            risk_cat = "KRİTİK_İZDÜŞÜM_ŞÜPHESİ"
            interp = "İki imza birebir çakışmaktadır. İnsan eli aynı imzayı iki kez %100 aynı atamayacağı için kopya / izdüşüm şüphesi taşır."
            limitations.append(
                f"⚠️ İki imza %{pct} oranında birebir çakışmaktadır (tetabuk). İnsan eli aynı imzayı iki kez %100 aynı atamayacağı için kopya / izdüşüm şüphesi taşır."
            )
        elif s_hybrid >= 0.85:
            band = ForensicSimilarityBand.NATURAL_VARIATION_MATCH
            band_label = " Doğal Varyasyon (Orijinal İmza)"
            risk_cat = "GÜVENLİ_AİDİYET"
            interp = "İmza akışı ve alışkanlıkları yazarın doğal varyasyon sınırları içerisindedir (Orijinal)."
        elif s_hybrid >= 0.70:
            req_micro = True
            band = ForensicSimilarityBand.SKILLED_SIMULATION_ZONE
            band_label = "🔍 Kritik Ayrım (Nitelikli Taklit Şüphesi)"
            risk_cat = "NİTELİKLİ_TAKLİT_RİSKİ"
            interp = "Genel şekil benzer; ancak taklit şüphesine karşı mikro titreme analizi gerekir."
        elif s_hybrid >= 0.50:
            req_micro = True
            band = ForensicSimilarityBand.INCONCLUSIVE_DISGUISE_ZONE
            band_label = "⚖️ Kararsız (Olası İstekli Bozma)"
            risk_cat = "BELİRSİZ_KANIT"
            interp = "Kısmi benzerlik ve sapmalar var; kasıtlı imza bozma veya amatör taklit şüphesidir."
        elif s_hybrid >= 0.25:
            band = ForensicSimilarityBand.LOW_SIMILARITY_NON_SOURCE
            band_label = "❌ Düşük Benzerlik (Kaba Taklit)"
            risk_cat = "SAHTECİLİK_DESTEĞİ"
            interp = "Temel yapı referans yazarla uyuşmamaktadır (Farklı Kişi / Kaba Taklit)."
        else:
            band = ForensicSimilarityBand.COMPLETE_DISPARITY
            band_label = "🚫 Tam Uyumsuzluk (Farklı İmza)"
            risk_cat = "KESİN_FARKLI_YAZAR"
            interp = "İki imza arasında hiçbir yapısal benzerlik yoktur (Farklı İmza)."

        # ----------------------------------------------------------------------
        # 3. Log-Likelihood Ratio (Log10(LR)) ve ENFSI Sözel Kanaat Hesabı
        # ----------------------------------------------------------------------
        if is_tetabuk:
            # Tetabuk durumunda kanıt H1'i değil, H2 Sahtecilik/İzdüşüm/Dolandırıcılık hipotezini destekler!
            log_lr = -3.80
            clamped_log_lr = -3.80
            lr_estimate = 0.0002
            scale = ENFSIConclusionScale.VERY_STRONG_SUPPORT_FOR_H2
            verbal = (
                f"İki imza %{pct} oranında birebir çakışmaktadır (tetabuk). İnsan eli aynı imzayı iki kez birebir atamayacağından, "
                f"orijinal bir belgeden kopyalama (İzdüşüm Sahteciliği - H2) yapıldığı değerlendirilmektedir."
            )
            ci_lower = -4.50
            ci_upper = -3.10
        else:
            # S_hybrid in [0.0, 0.979]. Merkez = 0.50.
            uncertainty_attenuation = 1.0 - (0.65 * total_uncertainty)
            raw_log_lr = self.k * (s_hybrid - 0.50) * uncertainty_attenuation
            log_lr = round(float(raw_log_lr), 3)

            # LR Kestirimi (Sayısal taşmayı önlemek için clamp [-6.0, +6.0])
            clamped_log_lr = max(-6.0, min(6.0, log_lr))
            lr_estimate = round(float(math.pow(10.0, clamped_log_lr)), 4)

            # %95 Güven Aralığı (+/- 1.96 * sigma_lr)
            ci_margin = round(float(1.96 * total_uncertainty * 1.5), 3)
            ci_lower = round(log_lr - ci_margin, 3)
            ci_upper = round(log_lr + ci_margin, 3)

            if log_lr >= 4.0:
                scale = ENFSIConclusionScale.EXTREMELY_STRONG_SUPPORT_FOR_H1
                verbal = f"Bulgular, imzanın '{suspect_name}' eli ürünü olduğunu SON DERECE KUVVETLE DESTEKLEMEKTEDİR (Orijinal - H1)."
            elif log_lr >= 3.0:
                scale = ENFSIConclusionScale.VERY_STRONG_SUPPORT_FOR_H1
                verbal = f"Bulgular, imzanın '{suspect_name}' eli ürünü olduğunu ÇOK KUVVETLE DESTEKLEMEKTEDİR (Orijinal - H1)."
            elif log_lr >= 2.0:
                scale = ENFSIConclusionScale.STRONG_SUPPORT_FOR_H1
                verbal = f"Bulgular, imzanın '{suspect_name}' eli ürünü olduğunu KUVVETLE DESTEKLEMEKTEDİR (Orijinal - H1)."
            elif log_lr >= 1.0:
                scale = ENFSIConclusionScale.MODERATE_SUPPORT_FOR_H1
                verbal = f"Bulgular, imzanın '{suspect_name}' eli ürünü olduğunu ORTA DÜZEYDE DESTEKLEMEKTEDİR (H1)."
            elif log_lr >= 0.5:
                scale = ENFSIConclusionScale.WEAK_SUPPORT_FOR_H1
                verbal = f"Bulgular, imzanın '{suspect_name}' eli ürünü olduğunu ZAYIF DÜZEYDE DESTEKLEMEKTEDİR (H1)."
            elif log_lr > -0.5:
                scale = ENFSIConclusionScale.INCONCLUSIVE
                verbal = "Elde edilen adli veriler kesin kanaat oluşturmak için yetersizdir (Kararsız / Ek Mukayese Gerekli)."
            elif log_lr > -1.0:
                scale = ENFSIConclusionScale.WEAK_SUPPORT_FOR_H2
                verbal = f"Bulgular, imzanın '{suspect_name}' DIŞINDA BİR ŞAHSA ait olduğunu ZAYIF DÜZEYDE DESTEKLEMEKTEDİR (H2)."
            elif log_lr > -2.0:
                scale = ENFSIConclusionScale.MODERATE_SUPPORT_FOR_H2
                verbal = f"Bulgular, imzanın '{suspect_name}' DIŞINDA BİR ŞAHSA ait olduğunu ORTA DÜZEYDE DESTEKLEMEKTEDİR (H2)."
            elif log_lr > -3.0:
                scale = ENFSIConclusionScale.STRONG_SUPPORT_FOR_H2
                verbal = f"Bulgular, imzanın '{suspect_name}' DIŞINDA BİR ŞAHSA ait olduğunu KUVVETLE DESTEKLEMEKTEDİR (Sahtecilik - H2)."
            else:
                scale = ENFSIConclusionScale.VERY_STRONG_SUPPORT_FOR_H2
                verbal = f"Bulgular, imzanın '{suspect_name}' DIŞINDA BİR ŞAHSA ait olduğunu ÇOK KUVVETLE DESTEKLEMEKTEDİR (Sahtecilik - H2)."

        propositions = {
            "H1 (Savcılık / Kaynak Hipotezi)": f"Sorgulanan el yazısı/imza, {suspect_name} tarafından yazılmıştır.",
            "H2 (Savunma / Alternatif Hipotez)": f"Sorgulanan el yazısı/imza, {suspect_name} dışındaki başka bir şahıs tarafından yazılmış / taklit edilmiştir (İzdüşüm/Sahtecilik).",
        }

        sim_evaluation = ForensicSimilarityScaleEvaluation(
            band=band,
            band_label_tr=band_label,
            similarity_percentage=pct,
            min_pct=98.0 if band == ForensicSimilarityBand.SUPERIMPOSITION_ANOMALY else (85.0 if band == ForensicSimilarityBand.NATURAL_VARIATION_MATCH else (70.0 if band == ForensicSimilarityBand.SKILLED_SIMULATION_ZONE else (50.0 if band == ForensicSimilarityBand.INCONCLUSIVE_DISGUISE_ZONE else (25.0 if band == ForensicSimilarityBand.LOW_SIMILARITY_NON_SOURCE else 0.0)))),
            max_pct=100.0 if band == ForensicSimilarityBand.SUPERIMPOSITION_ANOMALY else (97.9 if band == ForensicSimilarityBand.NATURAL_VARIATION_MATCH else (84.9 if band == ForensicSimilarityBand.SKILLED_SIMULATION_ZONE else (69.9 if band == ForensicSimilarityBand.INCONCLUSIVE_DISGUISE_ZONE else (49.9 if band == ForensicSimilarityBand.LOW_SIMILARITY_NON_SOURCE else 24.9)))),
            risk_category=risk_cat,
            forensic_interpretation_tr=interp,
            is_superimposition_anomaly=is_tetabuk,
            requires_micro_inspection=req_micro,
        )

        # Rapor Bütünlük İmzası
        payload = {
            "support_level": scale.value,
            "log_likelihood_ratio": log_lr,
            "uncertainty_score": total_uncertainty,
            "reference_count": reference_count,
            "similarity_band": band.value,
        }
        serialized = json.dumps(payload, sort_keys=True)
        audit_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        return HypothesisEvaluationReport(
            support_level=scale,
            log_likelihood_ratio=log_lr,
            likelihood_ratio_estimate=lr_estimate,
            uncertainty_score=total_uncertainty,
            confidence_interval_log_lr=(ci_lower, ci_upper),
            propositions=propositions,
            verbal_conclusion_tr=verbal,
            forensic_limitations=limitations,
            similarity_scale_evaluation=sim_evaluation,
            audit_hash_sha256=audit_hash,
        )
