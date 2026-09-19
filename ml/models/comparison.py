import hashlib
import json
import math
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from cv.features.extractors import ForensicFeatureReport


class DifferenceType(str, Enum):
    """Adli Karşılaştırma Öznitelik Uyum / Fark Statüsü."""
    CONCORDANT = "CONCORDANT"                      # Uyuşan (z <= 1.5)
    NATURAL_VARIATION = "NATURAL_VARIATION"        # Doğal Varyasyon (1.5 < z <= 2.5)
    SIGNIFICANT_DIFFERENCE = "SIGNIFICANT_DIFFERENCE" # Anlamlı / Şüpheli Fark (z > 2.5)


class FeatureDifferenceDetail(BaseModel):
    """Her bir adli özniteliğin detaylı karşılaştırma ve sapma analizi."""
    feature_key: str = Field(..., description="Öznitelik teknik adı")
    feature_name_tr: str = Field(..., description="Öznitelik Türkçe bilirkişi adı")
    questioned_value: float = Field(..., description="Sorgulanan belgedeki değer")
    reference_mean: float = Field(..., description="Referans belgeler ortalaması")
    reference_std: float = Field(..., description="Referans belgeler standart sapması")
    absolute_difference: float = Field(..., description="Mutlak fark |Q - R_mean|")
    z_score: float = Field(..., description="Standartlaştırılmış Z-Skoru sapması")
    diff_type: DifferenceType = Field(..., description="Fark sınıfı")
    explanation: str = Field(..., description="Adli yorum ve açıklama")


class ComparisonReport(BaseModel):
    """Temel Adli Karşılaştırma ve Benzerlik Raporu."""
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="0.0 - 1.0 arası genel benzerlik skoru")
    euclidean_distance: float = Field(..., description="Öklid mesafe metriği")
    mahalanobis_distance: float = Field(..., description="Standartlaştırılmış Mahalanobis varyans mesafesi")
    concordant_features: List[FeatureDifferenceDetail] = Field(default_factory=list, description="Uyuşan özellikler")
    natural_variations: List[FeatureDifferenceDetail] = Field(default_factory=list, description="Doğal varyasyon özellikler")
    significant_differences: List[FeatureDifferenceDetail] = Field(default_factory=list, description="Anlamlı / şüpheli farklar")
    preliminary_opinion: str = Field(..., description="Ön teknik kanaat özeti")
    reference_count: int = Field(..., description="Kullanılan referans belge sayısı")
    audit_hash_sha256: str = Field(..., description="Rapor bütünlük imzası")


class ForensicComparisonEngine:
    """
    Sorgulanan (Questioned) ve Referans (Reference) Belgeleri Karşılaştıran Adli Motor.
    
    Özellikler:
    - Referans popülasyonunun ortalama (mu_R) ve varyans (sigma_R) analizi.
    - Z-skoru tabanlı nesnel kural motoru.
    - Doğal varyasyon (Natural Variation) ile sahtecilik/farklı yazar göstergesi olan 
      anlamlı farkların (Significant Differences) ayrıştırılması.
    - Mahalanobis / Standardize Öklid mesafe hesabı.
    """

    FEATURE_KEYS = [
        "slant_angle_deg",
        "baseline_angle_deg",
        "baseline_deviation_px",
        "loop_count",
        "crossing_count",
        "terminal_count",
        "avg_stroke_length_px",
        "inter_stroke_spacing_px",
        "tremor_index_psd",
        "smoothness_energy",
        "mean_velocity_proxy",
        "velocity_variance_proxy",
        "acceleration_proxy",
    ]

    FEATURE_LABELS_TR = {
        "slant_angle_deg": "Yazı Eğim Açısı",
        "baseline_angle_deg": "Taban Satır Eğimi",
        "baseline_deviation_px": "Taban Çizgisi Dalgalanması",
        "loop_count": "Kapalı Döngü / İlmek Sayısı",
        "crossing_count": "Kesişim / Kavşak Sayısı",
        "terminal_count": "Uç Nokta Sayısı",
        "avg_stroke_length_px": "Ortalama Vuruş Uzunluğu",
        "inter_stroke_spacing_px": "Vuruşlar Arası Mesafe",
        "tremor_index_psd": "Titreme / Tremor İndeksi",
        "smoothness_energy": "Eğrilik & Pürüzsüzlük Enerjisi",
        "mean_velocity_proxy": "Kinematik Ortalama Hız",
        "velocity_variance_proxy": "Kinematik Hız Değişkenliği",
        "acceleration_proxy": "Kinematik İvme Dinamiği",
    }

    # Varsayılan minimum doğal tolerans standart sapmaları (tek referans veya sıfır varyans durumları için)
    MIN_PRIOR_STD = {
        "slant_angle_deg": 3.0,
        "baseline_angle_deg": 2.5,
        "baseline_deviation_px": 2.0,
        "loop_count": 0.8,
        "crossing_count": 2.0,
        "terminal_count": 2.0,
        "avg_stroke_length_px": 10.0,
        "inter_stroke_spacing_px": 8.0,
        "tremor_index_psd": 0.10,
        "smoothness_energy": 0.12,
        "mean_velocity_proxy": 0.20,
        "velocity_variance_proxy": 0.15,
        "acceleration_proxy": 0.12,
    }

    def _to_vector(self, item: Union[ForensicFeatureReport, List[float], np.ndarray, Dict[str, Any]]) -> np.ndarray:
        """Girdi formatını normalize numpy vektörüne dönüştürür."""
        if isinstance(item, ForensicFeatureReport):
            return np.array(item.feature_vector, dtype=np.float64)
        elif isinstance(item, (list, tuple)):
            return np.array(item, dtype=np.float64)
        elif isinstance(item, np.ndarray):
            return item.astype(np.float64)
        elif isinstance(item, dict):
            vec = []
            for k in self.FEATURE_KEYS:
                vec.append(float(item.get(k, 0.0)))
            return np.array(vec, dtype=np.float64)
        raise ValueError(f"Desteklenmeyen öznitelik veri formatı: {type(item)}")

    def compute_stabilized_mahalanobis(
        self,
        q_vec: np.ndarray,
        ref_matrix: np.ndarray,
        regularization_lambda: Optional[float] = None,
    ) -> float:
        """
        Sprint 16.8: Ledoit-Wolf ve Tikhonov Düzenlileştirmeli Kararlı Mahalanobis Mesafesi.
        2-3 numune, tekil numune veya özdeş numunelerde tekil matris (singular matrix)
        oluşmasını ve LinAlgError istisnalarını kesin olarak önler.
        """
        p = q_vec.shape[0]
        n_samples = ref_matrix.shape[0]
        ref_mean = np.mean(ref_matrix, axis=0)
        diff = q_vec - ref_mean

        if n_samples >= 2:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf(assume_centered=False, store_precision=True)
                lw.fit(ref_matrix)
                cov = lw.covariance_
            except Exception:
                cov = np.cov(ref_matrix, rowvar=False)
        else:
            # Tekil numunede prior varyans matrisi
            ref_std = np.array([self.MIN_PRIOR_STD.get(k, 1.0) for k in self.FEATURE_KEYS], dtype=np.float64)
            cov = np.diag(ref_std ** 2)

        # Düzenlileştirme katsayısı (Tikhonov / Shrinkage lambda)
        if regularization_lambda is not None:
            lam = max(1e-6, float(regularization_lambda))
        else:
            tr = float(np.trace(cov))
            lam = max(1e-4, (tr / p) * 1e-3) if tr > 0 else 1e-3

        cov_reg = cov + lam * np.eye(p)

        try:
            cov_inv = np.linalg.pinv(cov_reg)
            dist_sq = float(diff.T @ cov_inv @ diff)
            norm_dist = math.sqrt(max(0.0, dist_sq / p))
            return float(norm_dist)
        except Exception:
            ref_std = np.sqrt(np.diag(cov_reg))
            z = diff / np.maximum(ref_std, 1e-6)
            return float(np.sqrt(np.mean(z ** 2)))

    def compare(
        self,
        questioned: Union[ForensicFeatureReport, List[float], np.ndarray, Dict[str, Any]],
        references: List[Union[ForensicFeatureReport, List[float], np.ndarray, Dict[str, Any]]],
    ) -> ComparisonReport:
        """
        Sorgulanan belge ile referans belgeler kümesini adli kurallarla karşılaştırır.
        """
        if not references:
            raise ValueError("Karşılaştırma için en az 1 referans belge gereklidir.")

        q_vec = self._to_vector(questioned)
        ref_matrix = np.array([self._to_vector(r) for r in references], dtype=np.float64)

        num_features = len(self.FEATURE_KEYS)
        if q_vec.shape[0] != num_features or ref_matrix.shape[1] != num_features:
            raise ValueError(
                f"Öznitelik vektör boyutu uyumsuz: Beklenen {num_features}, "
                f"Gelen Q={q_vec.shape[0]}, Ref={ref_matrix.shape[1]}"
            )

        # 1. Referans İstatistikleri (Ortalama ve Standart Sapma)
        ref_mean = np.mean(ref_matrix, axis=0)
        ref_std_raw = np.std(ref_matrix, axis=0)

        # Doğal yazar içi varyasyonu garantiye almak için minimum prior std ve göreli tolerans ile harmanla
        ref_std = np.zeros(num_features, dtype=np.float64)
        for i, key in enumerate(self.FEATURE_KEYS):
            min_std = self.MIN_PRIOR_STD.get(key, 1.0)
            scale_std = abs(float(ref_mean[i])) * 0.10
            ref_std[i] = max(float(ref_std_raw[i]), min_std, scale_std)

        # 2. Mutlak Farklar ve Z-Skorları
        abs_diff = np.abs(q_vec - ref_mean)
        z_scores = abs_diff / ref_std

        # 3. Mesafe ve Benzerlik Skorları (Sprint 16.8: Stabilize Mahalanobis Mesafesi)
        mahalanobis_dist = self.compute_stabilized_mahalanobis(q_vec, ref_matrix)
        euclidean_dist = float(np.linalg.norm(abs_diff))

        # Benzerlik Skoru (0.0 - 1.0): S = exp(-0.25 * D_M)
        similarity_score = float(np.exp(-0.25 * mahalanobis_dist))
        similarity_score = round(min(1.0, max(0.0, similarity_score)), 4)

        # 4. Öznitelik Bazlı Ayrıştırma
        concordant: List[FeatureDifferenceDetail] = []
        natural_vars: List[FeatureDifferenceDetail] = []
        significant_diffs: List[FeatureDifferenceDetail] = []

        for i, key in enumerate(self.FEATURE_KEYS):
            name_tr = self.FEATURE_LABELS_TR[key]
            q_val = round(float(q_vec[i]), 3)
            r_m = round(float(ref_mean[i]), 3)
            r_s = round(float(ref_std[i]), 3)
            diff = round(float(abs_diff[i]), 3)
            z = round(float(z_scores[i]), 2)

            if z <= 1.5:
                diff_type = DifferenceType.CONCORDANT
                expl = f"{name_tr} değeri referans alışkanlıklarıyla tam uyum içindedir (|Δ|={diff}, z={z})."
                detail = FeatureDifferenceDetail(
                    feature_key=key,
                    feature_name_tr=name_tr,
                    questioned_value=q_val,
                    reference_mean=r_m,
                    reference_std=r_s,
                    absolute_difference=diff,
                    z_score=z,
                    diff_type=diff_type,
                    explanation=expl,
                )
                concordant.append(detail)
            elif z <= 2.5:
                diff_type = DifferenceType.NATURAL_VARIATION
                expl = f"{name_tr} değeri yazarın olağan doğal varyasyon sınırları dahilindedir (|Δ|={diff}, z={z})."
                detail = FeatureDifferenceDetail(
                    feature_key=key,
                    feature_name_tr=name_tr,
                    questioned_value=q_val,
                    reference_mean=r_m,
                    reference_std=r_s,
                    absolute_difference=diff,
                    z_score=z,
                    diff_type=diff_type,
                    explanation=expl,
                )
                natural_vars.append(detail)
            else:
                diff_type = DifferenceType.SIGNIFICANT_DIFFERENCE
                expl = f"DİKKAT: {name_tr} değerinde referans profilinden belirgin sapma tespit edildi (|Δ|={diff}, z={z} > 2.5)!"
                detail = FeatureDifferenceDetail(
                    feature_key=key,
                    feature_name_tr=name_tr,
                    questioned_value=q_val,
                    reference_mean=r_m,
                    reference_std=r_s,
                    absolute_difference=diff,
                    z_score=z,
                    diff_type=diff_type,
                    explanation=expl,
                )
                significant_diffs.append(detail)

        # 5. Ön Teknik Kanaat Oluşturma
        num_sig = len(significant_diffs)
        if similarity_score >= 0.80 and num_sig == 0:
            opinion = (
                "KUVVETLİ ORİJİNAL DESTEĞİ: Sorgulanan el yazısı/imza, referans belgelerin tüm karakteristik "
                "alışkanlıkları ve doğal varyasyon sınırları ile yüksek derecede örtüşmektedir. "
                "Anlamlı veya şüpheli hiçbir yapısal sapma gözlenmemiştir."
            )
        elif similarity_score >= 0.65 and num_sig <= 1:
            opinion = (
                "ORTA DÜZEYDE ORİJİNAL DESTEĞİ: Sorgulanan örnek genel hatlarıyla referans alışkanlıklarıyla uyumludur. "
                "Tespit edilen hafif farklılıklar doğal varyasyon ile açıklanabilir niteliktedir."
            )
        elif similarity_score >= 0.45 and num_sig <= 2:
            opinion = (
                "KARARSIZ / EK İNCELEME GEREKLİ: Sorgulanan örnekte hem uyumlu hem de farklılık gösteren "
                "karakteristikler bulunmaktadır. Kesin kanaat için daha fazla sayıda güncel referans belge incelenmelidir."
            )
        else:
            opinion = (
                "KUVVETLİ SAHTECİLİK / FARKLI YAZAR DESTEĞİ: Sorgulanan örnekte referans alışkanlıklarıyla bağdaşmayan "
                f"{num_sig} adet kritik yapısal/kinematik anlamlı sapma tespit edilmiştir. "
                "Yazının farklı bir şahıs eli ürünü veya taklit olma olasılığı yüksektir."
            )

        # 6. Rapor Bütünlük Karması
        hash_payload = {
            "similarity_score": similarity_score,
            "euclidean_distance": round(euclidean_dist, 3),
            "mahalanobis_distance": round(mahalanobis_dist, 3),
            "num_concordant": len(concordant),
            "num_natural_vars": len(natural_vars),
            "num_significant_diffs": len(significant_diffs),
            "reference_count": len(references),
        }
        serialized = json.dumps(hash_payload, sort_keys=True)
        audit_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        return ComparisonReport(
            similarity_score=similarity_score,
            euclidean_distance=round(euclidean_dist, 3),
            mahalanobis_distance=round(mahalanobis_dist, 3),
            concordant_features=concordant,
            natural_variations=natural_vars,
            significant_differences=significant_diffs,
            preliminary_opinion=opinion,
            reference_count=len(references),
            audit_hash_sha256=audit_hash,
        )
