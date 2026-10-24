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
    covariance_method: Optional[str] = Field(default=None, description="Kovaryans kestirim yöntemi")
    condition_number: Optional[float] = Field(default=None, description="Kovaryans matrisi koşul sayısı")
    confidence_warning: Optional[str] = Field(default=None, description="Örneklem yetersizliği veya kovaryans belirsizlik uyarısı")


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

    # Sprint 31: Boyutsuz normalize oranlar için tolerans standart sapmaları
    MIN_PRIOR_STD_NORM = {
        "slant_angle_deg": 3.0,
        "baseline_angle_deg": 2.5,
        "baseline_deviation_px": 0.040,   # %4.0 yazı yüksekliği toleransı
        "loop_count": 0.8,
        "crossing_count": 2.0,
        "terminal_count": 2.0,
        "avg_stroke_length_px": 0.120,    # %12.0 köşegen uzunluk toleransı
        "inter_stroke_spacing_px": 0.060, # %6.0 genişlik aralık toleransı
        "tremor_index_psd": 0.10,
        "smoothness_energy": 0.12,
        "mean_velocity_proxy": 0.20,
        "velocity_variance_proxy": 0.15,
        "acceleration_proxy": 0.12,
    }

    def _get_min_prior_std(self, key: str, mean_val: float) -> float:
        """Piksel veya normalize oran rejimine göre doğru asgari standart sapmayı döner."""
        if key == "baseline_deviation_px" and abs(mean_val) < 1.0:
            return self.MIN_PRIOR_STD_NORM.get(key, 0.040)
        elif key == "avg_stroke_length_px" and abs(mean_val) < 2.0:
            return self.MIN_PRIOR_STD_NORM.get(key, 0.120)
        elif key == "inter_stroke_spacing_px" and abs(mean_val) < 1.0:
            return self.MIN_PRIOR_STD_NORM.get(key, 0.060)
        return self.MIN_PRIOR_STD.get(key, 1.0)

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
        Sprint 31 P0 Düzeltmesi: N=1, N=2 ve N>=3 için Kararlı, Düzenlileştirilmiş Mahalanobis Mesafesi.
        Rank-1 tekil matris patlamasını (N=2), tek örnek belirsizliğini (N=1) ve aşırı mesafeleri
        kesin olarak önler.
        """
        p = q_vec.shape[0]
        n_samples = ref_matrix.shape[0]
        ref_mean = np.mean(ref_matrix, axis=0)
        diff = q_vec - ref_mean

        # 1. Temel Adli Prior Diyagonal Matrisi (Sigma_0)
        ref_std_prior = np.zeros(p, dtype=np.float64)
        for i, k in enumerate(self.FEATURE_KEYS):
            min_std = self._get_min_prior_std(k, float(ref_mean[i]))
            if n_samples == 1:
                min_std *= 1.5
                scale_std = abs(float(ref_mean[i])) * 0.20
            elif n_samples == 2:
                min_std *= 1.25
                scale_std = abs(float(ref_mean[i])) * 0.15
            else:
                scale_std = abs(float(ref_mean[i])) * 0.10
            ref_std_prior[i] = max(min_std, scale_std)
        cov_prior = np.diag(ref_std_prior ** 2)

        cov_method = "DIAGONAL_PRIOR"
        if n_samples >= 3:
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf(assume_centered=False, store_precision=True)
                lw.fit(ref_matrix)
                cov = lw.covariance_
                cov_method = "LEDOIT_WOLF"
            except Exception:
                cov = np.cov(ref_matrix, rowvar=False)
                cov_method = "SAMPLE_COVARIANCE"
            # Köşegen varyansını minimum prior ile güvenceye al
            for i in range(p):
                cov[i, i] = max(float(cov[i, i]), float(cov_prior[i, i]))
        elif n_samples == 2:
            # Sprint 31 P0 Düzeltmesi: N=2 durumunda sample cov rank-1'dir.
            # Diyagonal adli prior (cov_prior) hedefli kontrollü shrinkage (%70 prior, %30 sample):
            s_raw = np.cov(ref_matrix, rowvar=False)
            gamma = 0.30  # sample covariance ağırlığı
            cov = (1.0 - gamma) * cov_prior + gamma * s_raw

            # Korelasyon sınırlandırması (Kovaryans korelasyon katsayılarını [-0.5, +0.5] aralığına kırp)
            stds = np.sqrt(np.maximum(np.diag(cov), 1e-8))
            for i in range(p):
                cov[i, i] = max(float(cov[i, i]), float(cov_prior[i, i]))
                for j in range(p):
                    if i != j:
                        r_ij = cov[i, j] / (stds[i] * stds[j])
                        r_clamped = max(-0.50, min(0.50, r_ij))
                        cov[i, j] = r_clamped * stds[i] * stds[j]
            cov_method = "TARGET_SHRINKAGE_N2"
        else:
            # N=1 tekil referans
            cov = cov_prior
            cov_method = "SINGLE_REF_PRIOR"

        # Tikhonov düzenlileştirmesi ve koşul sayısı kontrolü
        tr = float(np.trace(cov))
        if regularization_lambda is not None:
            lam = max(1e-5, float(regularization_lambda))
        else:
            lam = max(1e-4, (tr / p) * 1e-3) if tr > 0 else 1e-3

        cov_reg = cov + lam * np.eye(p)

        # Koşul sayısı kontrolü (ill-conditioned durumunda lambda'yı artır)
        try:
            cond = float(np.linalg.cond(cov_reg))
            if cond > 1e4:
                cov_reg += (tr / p) * 0.05 * np.eye(p)
                cond = float(np.linalg.cond(cov_reg))
        except Exception:
            cond = 1e6

        self._last_cov_method = cov_method
        self._last_cond_number = round(cond, 2)

        try:
            cov_inv = np.linalg.pinv(cov_reg)
            dist_sq = float(diff.T @ cov_inv @ diff)
            norm_dist = math.sqrt(max(0.0, dist_sq / p))
            # Aşırı mesafe patlamasını mantıksal üst sınırla sınırla (maksimum 12.0)
            return float(min(12.0, max(0.0, norm_dist)))
        except Exception:
            ref_std = np.sqrt(np.diag(cov_reg))
            z = diff / np.maximum(ref_std, 1e-6)
            fallback_dist = float(np.sqrt(np.mean(z ** 2)))
            return float(min(12.0, max(0.0, fallback_dist)))

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
        is_single_ref = len(references) == 1
        is_two_ref = len(references) == 2

        # Doğal yazar içi varyasyonu garantiye almak için minimum prior std ve göreli tolerans ile harmanla
        ref_std = np.zeros(num_features, dtype=np.float64)
        for i, key in enumerate(self.FEATURE_KEYS):
            min_std = self._get_min_prior_std(key, float(ref_mean[i]))
            if is_single_ref:
                min_std *= 1.5
                scale_std = abs(float(ref_mean[i])) * 0.20
            elif is_two_ref:
                min_std *= 1.25
                scale_std = abs(float(ref_mean[i])) * 0.15
            else:
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

        # 6. Referans Sayısı Güvenilirlik & Belirsizlik Uyarısı
        cov_method = getattr(self, "_last_cov_method", "DIAGONAL_PRIOR")
        cond_num = getattr(self, "_last_cond_number", None)
        if len(references) == 1:
            conf_warning = (
                "UYARI: Tek referans numune (N=1) ile analiz yapılmıştır. Yazar içi doğal varyasyon "
                "bilinemediğinden genişletilmiş adli prior kullanılmıştır. Kesin adli kanaat için en az 3 referans önerilir."
            )
        elif len(references) == 2:
            conf_warning = (
                "BİLGİ: 2 referans numune (N=2) ile analiz yapılmıştır. Kovaryans rank-1 sınırlaması "
                "nedeniyle adli prior ile düzenlileştirilmiştir (Target Shrinkage). Tavsiye edilen referans sayısı >= 3'tür."
            )
        else:
            conf_warning = None

        # 7. Rapor Bütünlük Karması
        hash_payload = {
            "similarity_score": similarity_score,
            "euclidean_distance": round(euclidean_dist, 3),
            "mahalanobis_distance": round(mahalanobis_dist, 3),
            "num_concordant": len(concordant),
            "num_natural_vars": len(natural_vars),
            "num_significant_diffs": len(significant_diffs),
            "reference_count": len(references),
            "covariance_method": cov_method,
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
            covariance_method=cov_method,
            condition_number=cond_num,
            confidence_warning=conf_warning,
        )
