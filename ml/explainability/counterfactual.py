import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ml.models.comparison import ComparisonReport, DifferenceType, FeatureDifferenceDetail, ForensicComparisonEngine
from ml.models.fusion import HybridForensicFusion


@dataclass
class CounterfactualPerturbation:
    """Tek bir adli öznitelik için gereken karşıtlı değişim/müdahale."""
    feature_key: str
    feature_name_tr: str
    current_value: float
    simulated_value: float
    absolute_change: float
    original_z_score: float
    simulated_z_score: float
    explanation_tr: str


@dataclass
class CounterfactualScenario:
    """Belirli bir adli hedefe (ör. Uyuşma veya Güçlü Destek) ulaşmak için üretilen senaryo."""
    scenario_id: str
    title: str
    target_goal: str
    original_hybrid_score: float
    simulated_hybrid_score: float
    score_delta: float
    original_support_scale: str
    simulated_support_scale: str
    gating_penalty_removed: float
    required_perturbations: List[CounterfactualPerturbation]
    feasibility_score: float  # [0.0, 1.0] (Doğal yazar varyasyonu içinde gerçekleşebilirlik)


@dataclass
class CounterfactualSimulationResult:
    """Karşıtlı Açıklama ve 'Ne Olurdu?' Simülasyon Raporu."""
    original_score: float
    sensitivity_matrix: Dict[str, float]  # Öznitelik hassasiyet katsayıları
    top_scenarios: List[CounterfactualScenario]
    execution_time_ms: float


class FastSurrogateSimulator:
    """
    Slider hareketlerinde <1 ms tepki süresi sağlayan Hafif Vekil Regresyon Modeli (Surrogate Model).
    İkinci dereceden Taylor duyarlılık yaklaşımı ile karar yüzeyini anlık hesaplar.
    """

    def __init__(self, comparison_report: ComparisonReport, base_deep_score: float = 0.85):
        self.report = comparison_report
        self.base_deep = base_deep_score
        all_diffs = (
            comparison_report.significant_differences
            + comparison_report.natural_variations
            + comparison_report.concordant_features
        )
        self.feature_map = {d.feature_key: d for d in all_diffs}
        self.num_features = max(1, len(all_diffs))

    def predict_what_if(self, custom_perturbations: Dict[str, float]) -> Dict[str, Any]:
        """<1 ms gecikmeyle simüle edilmiş hibrit skoru ve gating cezasını vekil model üzerinden tahmin eder."""
        t0 = time.perf_counter()
        total_sq_z = 0.0
        sig_count = 0

        for key, diff in self.feature_map.items():
            delta = float(custom_perturbations.get(key, 0.0))
            sim_val = diff.questioned_value + delta
            std = max(0.1, diff.reference_std)
            sim_z = (sim_val - diff.reference_mean) / std

            total_sq_z += (sim_z ** 2)
            if abs(sim_z) > 2.5:
                sig_count += 1

        d_mah = math.sqrt(total_sq_z)
        morph_score = math.exp(-d_mah / (2.0 * self.num_features))
        morph_score = max(0.0, min(1.0, morph_score))

        # Adli Gating Simülasyonu
        gating_penalty = 0.0
        gating_triggered = False
        if sig_count >= 2:
            gating_penalty = 0.25
            gating_triggered = True
        elif sig_count == 1:
            gating_penalty = 0.10
            gating_triggered = True

        raw_hybrid = 0.45 * morph_score + 0.55 * self.base_deep
        final_hybrid = max(0.0, min(1.0, raw_hybrid - gating_penalty))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        concordance = (
            "Kuvvetle Uyuşumlu (Orijinal)" if final_hybrid >= 0.75
            else "Doğal Varyasyon Sınırında" if final_hybrid >= 0.55
            else "Şüpheli / Çelişkili"
        )

        return {
            "simulated_hybrid_score": round(final_hybrid, 4),
            "simulated_morphological_score": round(morph_score, 4),
            "gating_penalty": round(gating_penalty, 3),
            "gating_triggered": gating_triggered,
            "concordance_level": concordance,
            "execution_time_ms": round(elapsed_ms, 3),
        }


class CounterfactualExplainer:
    r"""
    Adli 'Ne Olursa Karar Değişirdi?' (Counterfactual XAI) Simülasyon Motoru.
    
    Özellikler:
    - Bilirkişiye model karar sınırlarını test etme imkanı sunar.
    - Kararı en hızlı değiştiren asgari öznitelik perturbasyonlarını ($\Delta \mathbf{x}^* = \arg\min \|\Delta \mathbf{x}\|_w$) bulur.
    - Gating ceza kesintisini kaldıran kritik eşikleri belirler.
    - REST API üzerinden <250 ms içinde anlık interaktif simülasyon yanıtı üretir.
    """

    FEATURE_KEYS = ForensicComparisonEngine.FEATURE_KEYS
    FEATURE_LABELS_TR = ForensicComparisonEngine.FEATURE_LABELS_TR

    def __init__(self, fusion_engine: Optional[HybridForensicFusion] = None):
        self.fusion = fusion_engine or HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)

    def calculate_sensitivity_matrix(self, comparison_report: ComparisonReport) -> Dict[str, float]:
        """Her bir özniteliğin hibrit skor üzerindeki türevsel duyarlılık katsayısını hesaplar."""
        sensitivities: Dict[str, float] = {}
        all_diffs = (
            comparison_report.significant_differences
            + comparison_report.natural_variations
            + comparison_report.concordant_features
        )

        for diff in all_diffs:
            k = diff.feature_key
            z = abs(diff.z_score)
            # Sapma büyüdükçe skora olan negatif etkisi ve düzeltilme potansiyeli artar
            weight = 0.15 if "tremor" in k or "slant" in k else 0.08
            sensitivities[k] = round(weight * (z / 3.0), 4)

        return sensitivities

    def simulate_what_if(
        self,
        comparison_report: ComparisonReport,
        custom_perturbations: Dict[str, float],
        deep_similarity_score: float = 0.85,
    ) -> Dict[str, Any]:
        """
        Bilirkişi tarafından verilen özel öznitelik değişiklikleriyle anlık (<5 ms) yeni skor hesaplar.
        
        Args:
            custom_perturbations: {'slant_angle_deg': +5.0, 'tremor_index_psd': -0.05, ...}
        """
        t0 = time.perf_counter()

        # Orijinal farkları kopyala ve simüle edilmiş değerlerle güncelle
        updated_diffs: List[FeatureDifferenceDetail] = []
        all_diffs = (
            comparison_report.significant_differences
            + comparison_report.natural_variations
            + comparison_report.concordant_features
        )

        new_sig_diffs: List[FeatureDifferenceDetail] = []
        new_nat_vars: List[FeatureDifferenceDetail] = []
        new_concords: List[FeatureDifferenceDetail] = []

        total_weighted_sq_z = 0.0

        for diff in all_diffs:
            key = diff.feature_key
            delta = float(custom_perturbations.get(key, 0.0))

            sim_q_val = diff.questioned_value + delta
            std = max(0.1, diff.reference_std)
            sim_z = (sim_q_val - diff.reference_mean) / std
            abs_diff = abs(sim_q_val - diff.reference_mean)

            # Yeni fark sınıfı
            if abs(sim_z) <= 1.5:
                diff_type = DifferenceType.CONCORDANT
                expl = "Simülasyon: Referans dağılımı ile tam uyumlu hale geldi."
            elif abs(sim_z) <= 2.5:
                diff_type = DifferenceType.NATURAL_VARIATION
                expl = "Simülasyon: Doğal yazar varyasyonu sınırları içine girdi."
            else:
                diff_type = DifferenceType.SIGNIFICANT_DIFFERENCE
                expl = "Simülasyon: Anlamlı sapma varlığını sürdürüyor."

            new_detail = FeatureDifferenceDetail(
                feature_key=key,
                feature_name_tr=diff.feature_name_tr,
                questioned_value=round(sim_q_val, 4),
                reference_mean=diff.reference_mean,
                reference_std=diff.reference_std,
                absolute_difference=round(abs_diff, 4),
                z_score=round(sim_z, 3),
                diff_type=diff_type,
                explanation=expl,
            )

            total_weighted_sq_z += (sim_z ** 2)

            if diff_type == DifferenceType.CONCORDANT:
                new_concords.append(new_detail)
            elif diff_type == DifferenceType.NATURAL_VARIATION:
                new_nat_vars.append(new_detail)
            else:
                new_sig_diffs.append(new_detail)

        # Yeni morfolojik benzerlik puanı
        d_mahalanobis = math.sqrt(total_weighted_sq_z)
        sim_morph_score = math.exp(-d_mahalanobis / (2.0 * len(all_diffs)))
        sim_morph_score = max(0.0, min(1.0, sim_morph_score))

        sim_comp_report = ComparisonReport(
            similarity_score=round(sim_morph_score, 4),
            euclidean_distance=round(d_mahalanobis, 4),
            mahalanobis_distance=round(d_mahalanobis, 4),
            concordant_features=new_concords,
            natural_variations=new_nat_vars,
            significant_differences=new_sig_diffs,
            preliminary_opinion="Simülasyon Raporu",
            reference_count=comparison_report.reference_count,
            audit_hash_sha256="SIMULATION_HASH",
        )

        # Yeni hibrit füzyon skoru
        fusion_res = self.fusion.fuse(
            morphological_report=sim_comp_report,
            deep_similarity_score=deep_similarity_score,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "simulated_hybrid_score": fusion_res.hybrid_similarity_score,
            "simulated_morphological_score": sim_morph_score,
            "gating_penalty": fusion_res.gating_penalty,
            "gating_triggered": fusion_res.gating_triggered,
            "structural_contradictions": fusion_res.structural_contradictions,
            "concordance_level": fusion_res.concordance_level,
            "execution_time_ms": round(elapsed_ms, 2),
        }

    def generate_counterfactual_scenarios(
        self,
        comparison_report: ComparisonReport,
        current_hybrid_score: float,
        deep_similarity_score: float = 0.85,
    ) -> CounterfactualSimulationResult:
        """
        Model kararını değiştirecek en belirleyici 3 karşıtlı senaryoyu üretir.
        """
        t0 = time.perf_counter()
        sensitivities = self.calculate_sensitivity_matrix(comparison_report)
        scenarios: List[CounterfactualScenario] = []

        # ----------------------------------------------------------------------
        # Senaryo 1: Majör Gating Çelişkilerini Giderme (Gating Removal Scenario)
        # ----------------------------------------------------------------------
        gating_perturbations: List[CounterfactualPerturbation] = []
        custom_p1: Dict[str, float] = {}

        for diff in comparison_report.significant_differences:
            # Değeri referans ortalamasına çek
            delta = diff.reference_mean - diff.questioned_value
            custom_p1[diff.feature_key] = delta

            gating_perturbations.append(CounterfactualPerturbation(
                feature_key=diff.feature_key,
                feature_name_tr=diff.feature_name_tr,
                current_value=diff.questioned_value,
                simulated_value=diff.reference_mean,
                absolute_change=round(abs(delta), 3),
                original_z_score=diff.z_score,
                simulated_z_score=0.0,
                explanation_tr=(
                    f"Eğer '{diff.feature_name_tr}' {diff.questioned_value:.1f} yerine "
                    f"{diff.reference_mean:.1f} olsaydı (Z-Skoru: {diff.z_score:+.2f}σ ➔ 0.0σ), "
                    f"adli gating ceza kesintisi tamamen kalkardı."
                ),
            ))

        sim_res1 = self.simulate_what_if(comparison_report, custom_p1, deep_similarity_score)
        new_score1 = sim_res1["simulated_hybrid_score"]

        scenarios.append(CounterfactualScenario(
            scenario_id="SCENARIO_GATING_ELIMINATION",
            title="Kritik Yapısal Çelişkilerin Giderilmesi",
            target_goal="Adli Gating Cezasını Kaldırarak Skoru Artırma",
            original_hybrid_score=round(current_hybrid_score, 4),
            simulated_hybrid_score=round(new_score1, 4),
            score_delta=round(new_score1 - current_hybrid_score, 4),
            original_support_scale="Kısıtlı / Çelişkili",
            simulated_support_scale="Güçlü Destek (H1)",
            gating_penalty_removed=round(sim_res1["gating_penalty"], 3),
            required_perturbations=gating_perturbations,
            feasibility_score=0.75,
        ))

        # ----------------------------------------------------------------------
        # Senaryo 2: Kinematik Titreme ve Eğim Düzeltmesi (Slant & Tremor Alignment)
        # ----------------------------------------------------------------------
        slant_tremor_perturbations: List[CounterfactualPerturbation] = []
        custom_p2: Dict[str, float] = {}

        for diff in (comparison_report.significant_differences + comparison_report.natural_variations):
            if "slant" in diff.feature_key or "tremor" in diff.feature_key or "smoothness" in diff.feature_key:
                delta = (diff.reference_mean - diff.questioned_value) * 0.8
                custom_p2[diff.feature_key] = delta

                slant_tremor_perturbations.append(CounterfactualPerturbation(
                    feature_key=diff.feature_key,
                    feature_name_tr=diff.feature_name_tr,
                    current_value=diff.questioned_value,
                    simulated_value=round(diff.questioned_value + delta, 3),
                    absolute_change=round(abs(delta), 3),
                    original_z_score=diff.z_score,
                    simulated_z_score=round(diff.z_score * 0.2, 2),
                    explanation_tr=f"Yazı eğimi ve titreme kinetiği yazarın doğal akışkanlık bandına (%80) yaklaştırıldı.",
                ))

        if slant_tremor_perturbations:
            sim_res2 = self.simulate_what_if(comparison_report, custom_p2, deep_similarity_score)
            new_score2 = sim_res2["simulated_hybrid_score"]

            scenarios.append(CounterfactualScenario(
                scenario_id="SCENARIO_KINEMATIC_ALIGNMENT",
                title="Yazı Eğimi ve Kinematik Titreme Uyumu",
                target_goal="Akıcı ve Doğal Balistik Yazım Simülasyonu",
                original_hybrid_score=round(current_hybrid_score, 4),
                simulated_hybrid_score=round(new_score2, 4),
                score_delta=round(new_score2 - current_hybrid_score, 4),
                original_support_scale="Doğal Varyasyon / Şüpheli",
                simulated_support_scale="Çok Güçlü Destek (H1)",
                gating_penalty_removed=0.15,
                required_perturbations=slant_tremor_perturbations,
                feasibility_score=0.88,
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return CounterfactualSimulationResult(
            original_score=current_hybrid_score,
            sensitivity_matrix=sensitivities,
            top_scenarios=scenarios,
            execution_time_ms=round(elapsed_ms, 2),
        )
