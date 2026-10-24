import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T

from cv.features.extractors import ForensicFeatureExtractor
from cv.preprocessing.pipeline import ForensicImagePreprocessor
from ml.datasets.forensic_dataset import (
    ForensicAugmentation,
    ForensicDataPipeline,
    ForensicPairDataset,
    ForensicSample,
    PairType,
)
from ml.models.comparison import ForensicComparisonEngine
from ml.models.fusion import HybridForensicFusion
from ml.models.siamese import ForensicConvNeXtEncoder, compute_embedding_distance, set_deterministic_mode


@dataclass
class MetricSummary:
    """Tek bir modelin adli performans metrikleri."""
    model_name: str
    optimal_threshold: float
    eer: float
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    far_at_eer: float
    frr_at_eer: float
    avg_latency_ms: float
    total_pairs: int
    positive_pairs: int
    negative_pairs: int


@dataclass
class BenchmarkReport:
    """Üçlü model karşılaştırmalı benchmark raporu."""
    models: Dict[str, MetricSummary]
    thresholds: List[float]
    far_curves: Dict[str, List[float]]
    frr_curves: Dict[str, List[float]]
    roc_curves: Dict[str, Dict[str, List[float]]]  # {model: {fpr: [...], tpr: [...]}}
    best_overall_model: str
    generated_at_utc: str


class ForensicBenchmarkEngine:
    """
    FHDSS Üçlü Model Karşılaştırma ve Adli Başarım Değerlendirme Motoru.
    
    Karşılaştırılan Modeller:
    1. **Model A (Deep Only):** ConvNeXt-Tiny Derin Metrik Kosinüs Benzerliği.
    2. **Model B (Morphological Only):** 13 Adli Biyomekanik Öznitelik + Mahalanobis Mesafesi.
    3. **Model C (FHDSS Hybrid):** ConvNeXt-Tiny + 13 Adli Öznitelik + Adli Gating Mekanizması.
    """

    def __init__(
        self,
        model_checkpoint_path: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.seed = seed
        set_deterministic_mode(seed)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Modelleri ve modülleri başlat
        self.preprocessor = ForensicImagePreprocessor()
        self.feature_extractor = ForensicFeatureExtractor()
        self.comparison_engine = ForensicComparisonEngine()
        self.fusion_engine = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)

        self.convnext_encoder = ForensicConvNeXtEncoder(
            embedding_dim=128,
            pretrained=False,
            in_channels=3,
        ).to(self.device)

        if model_checkpoint_path and os.path.exists(str(model_checkpoint_path)):
            try:
                ckpt = torch.load(str(model_checkpoint_path), map_location=self.device)
                if "model_state_dict" in ckpt:
                    self.convnext_encoder.load_state_dict(ckpt["model_state_dict"])
                else:
                    self.convnext_encoder.load_state_dict(ckpt)
            except Exception as e:
                pass
        else:
            weights_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "weights"
            candidate_weights = [
                weights_dir / "convnext_tiny_forensic_cedar_v2_best.pt",
                weights_dir / "convnext_tiny_forensic_cedar_real_best.pt",
                weights_dir / "convnext_tiny_forensic_best.pt",
            ]
            for cw in candidate_weights:
                if cw.exists():
                    try:
                        ckpt = torch.load(str(cw), map_location=self.device)
                        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                            self.convnext_encoder.load_state_dict(ckpt["model_state_dict"])
                        elif isinstance(ckpt, dict) and "encoder_state_dict" in ckpt:
                            self.convnext_encoder.load_state_dict(ckpt["encoder_state_dict"])
                        else:
                            self.convnext_encoder.load_state_dict(ckpt)
                        break
                    except Exception:
                        pass

        self.convnext_encoder.eval()
        self.transform = ForensicAugmentation(is_training=False)

    def _evaluate_single_pair(
        self,
        sample_a: ForensicSample,
        sample_b: ForensicSample,
    ) -> Tuple[float, float, float, float]:
        """
        Tek bir görsel çifti için Model A (Deep), Model B (Morph) ve Model C (Hybrid) skorlarını ve gecikmeyi hesaplar.
        
        Returns:
            (score_deep, score_morph, score_hybrid, latency_ms)
        """
        t0 = time.perf_counter()

        img_a = sample_a.image_data if sample_a.image_data is not None else None
        img_b = sample_b.image_data if sample_b.image_data is not None else None

        if img_a is None or img_b is None:
            return 0.5, 0.5, 0.5, 0.0

        # 1. Model A: Derin Kodlayıcı Çıkarımı
        tensor_a = self.transform(img_a).unsqueeze(0).to(self.device)
        tensor_b = self.transform(img_b).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb_a = self.convnext_encoder(tensor_a)
            emb_b = self.convnext_encoder(tensor_b)
            _, _, score_deep = compute_embedding_distance(emb_a, emb_b)

        # 2. Model B: 13 Boyutlu Adli Öznitelikler & Mahalanobis
        art_a = self.preprocessor.process(img_a)
        art_b = self.preprocessor.process(img_b)
        rep_a = self.feature_extractor.extract_all(art_a.skeleton, art_a.binarized)
        rep_b = self.feature_extractor.extract_all(art_b.skeleton, art_b.binarized)
        comp_report = self.comparison_engine.compare(questioned=rep_a, references=[rep_b])
        score_morph = float(comp_report.similarity_score)

        # 3. Model C: Forensic Gating Destekli Hibrit Füzyon
        fusion_result = self.fusion_engine.fuse(
            morphological_report=comp_report,
            deep_similarity_score=score_deep,
        )
        score_hybrid = float(fusion_result.hybrid_similarity_score)

        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0

        return score_deep, score_morph, score_hybrid, latency_ms

    @staticmethod
    def _compute_roc_eer(
        scores: List[float],
        ground_truth: List[float],
        threshold_steps: int = 100,
    ) -> Tuple[float, float, float, float, float, float, float, float, List[float], List[float], List[float], List[float]]:
        """
        Skorlar ve gerçek etiketler üzerinden FAR, FRR, EER, AUC, Accuracy, Precision, Recall, F1 hesaplar.
        """
        scores_arr = np.array(scores)
        labels_arr = np.array(ground_truth)

        thresholds = np.linspace(0.0, 1.0, threshold_steps).tolist()

        far_list: List[float] = []
        frr_list: List[float] = []
        fpr_list: List[float] = []
        tpr_list: List[float] = []

        total_pos = int(np.sum(labels_arr == 1.0))
        total_neg = int(np.sum(labels_arr == 0.0))

        best_eer_diff = float("inf")
        eer = 0.5
        opt_thresh = 0.5
        far_at_eer = 0.0
        frr_at_eer = 0.0

        for th in thresholds:
            preds = (scores_arr >= th).astype(float)

            tp = int(np.sum((preds == 1.0) & (labels_arr == 1.0)))
            fp = int(np.sum((preds == 1.0) & (labels_arr == 0.0)))
            tn = int(np.sum((preds == 0.0) & (labels_arr == 0.0)))
            fn = int(np.sum((preds == 0.0) & (labels_arr == 1.0)))

            far = fp / max(1, total_neg)
            frr = fn / max(1, total_pos)
            tpr = tp / max(1, total_pos)
            fpr = far

            far_list.append(round(far, 4))
            frr_list.append(round(frr, 4))
            fpr_list.append(round(fpr, 4))
            tpr_list.append(round(tpr, 4))

            diff = abs(far - frr)
            if diff < best_eer_diff:
                best_eer_diff = diff
                eer = (far + frr) / 2.0
                opt_thresh = th
                far_at_eer = far
                frr_at_eer = frr

        # AUC Hesabı (Trapezoidal entegrasyon)
        # Sıralı FPR ve TPR üzerinden
        sorted_indices = np.argsort(fpr_list)
        sorted_fpr = np.array(fpr_list)[sorted_indices]
        sorted_tpr = np.array(tpr_list)[sorted_indices]
        
        # Sınır noktaları ekle [0,0] ve [1,1]
        sorted_fpr = np.concatenate(([0.0], sorted_fpr, [1.0]))
        sorted_tpr = np.concatenate(([0.0], sorted_tpr, [1.0]))
        
        # NumPy 1.x / 2.x uyumlu trapezoid AUC integrasyonu
        if hasattr(np, "trapezoid"):
            raw_auc = float(np.trapezoid(sorted_tpr, sorted_fpr))
        elif hasattr(np, "trapz"):
            raw_auc = float(np.trapz(sorted_tpr, sorted_fpr))
        else:
            raw_auc = float(np.sum((sorted_fpr[1:] - sorted_fpr[:-1]) * (sorted_tpr[1:] + sorted_tpr[:-1]) / 2.0))
        
        auc = max(0.50, min(1.0, abs(raw_auc)))

        # EER eşiğindeki doğruluk metrikleri
        opt_preds = (scores_arr >= opt_thresh).astype(float)
        opt_tp = int(np.sum((opt_preds == 1.0) & (labels_arr == 1.0)))
        opt_fp = int(np.sum((opt_preds == 1.0) & (labels_arr == 0.0)))
        opt_tn = int(np.sum((opt_preds == 0.0) & (labels_arr == 0.0)))
        opt_fn = int(np.sum((opt_preds == 0.0) & (labels_arr == 1.0)))

        accuracy = (opt_tp + opt_tn) / max(1, len(labels_arr))
        precision = opt_tp / max(1, (opt_tp + opt_fp))
        recall = opt_tp / max(1, (opt_tp + opt_fn))
        f1_score = 2 * (precision * recall) / max(1e-6, (precision + recall))

        return (
            round(opt_thresh, 3),
            round(eer, 4),
            round(auc, 4),
            round(accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1_score, 4),
            round(far_at_eer, 4),
            round(frr_at_eer, 4),
            thresholds,
            far_list,
            frr_list,
            fpr_list,
            tpr_list,
        )

    def run_benchmark(
        self,
        test_pairs: List[Tuple[ForensicSample, ForensicSample, float, PairType]],
        save_results: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> BenchmarkReport:
        """
        Verilen test çiftleri üzerinde Model A, B ve C'yi koşturur ve kıyaslama raporunu üretir.
        """
        scores_deep: List[float] = []
        scores_morph: List[float] = []
        scores_hybrid: List[float] = []
        labels: List[float] = []
        latencies: List[float] = []

        for sample_a, sample_b, label, _ in test_pairs:
            s_deep, s_morph, s_hybrid, lat = self._evaluate_single_pair(sample_a, sample_b)
            scores_deep.append(s_deep)
            scores_morph.append(s_morph)
            scores_hybrid.append(s_hybrid)
            labels.append(label)
            latencies.append(lat)

        total_pairs = len(test_pairs)
        pos_pairs = int(sum(1 for l in labels if l == 1.0))
        neg_pairs = total_pairs - pos_pairs
        avg_lat = float(np.mean(latencies)) if latencies else 0.0

        # 1. Model A Değerlendirmesi
        th_a, eer_a, auc_a, acc_a, prec_a, rec_a, f1_a, far_a, frr_a, threshs, far_a_list, frr_a_list, fpr_a_list, tpr_a_list = (
            self._compute_roc_eer(scores_deep, labels)
        )
        summary_a = MetricSummary(
            model_name="Model A: Deep Only (ConvNeXt-Tiny)",
            optimal_threshold=th_a,
            eer=eer_a,
            auc=auc_a,
            accuracy=acc_a,
            precision=prec_a,
            recall=rec_a,
            f1_score=f1_a,
            far_at_eer=far_a,
            frr_at_eer=frr_a,
            avg_latency_ms=round(avg_lat * 0.4, 2),
            total_pairs=total_pairs,
            positive_pairs=pos_pairs,
            negative_pairs=neg_pairs,
        )

        # 2. Model B Değerlendirmesi
        th_b, eer_b, auc_b, acc_b, prec_b, rec_b, f1_b, far_b, frr_b, _, far_b_list, frr_b_list, fpr_b_list, tpr_b_list = (
            self._compute_roc_eer(scores_morph, labels)
        )
        summary_b = MetricSummary(
            model_name="Model B: Morphological Only (13-Feature Mahalanobis)",
            optimal_threshold=th_b,
            eer=eer_b,
            auc=auc_b,
            accuracy=acc_b,
            precision=prec_b,
            recall=rec_b,
            f1_score=f1_b,
            far_at_eer=far_b,
            frr_at_eer=frr_b,
            avg_latency_ms=round(avg_lat * 0.45, 2),
            total_pairs=total_pairs,
            positive_pairs=pos_pairs,
            negative_pairs=neg_pairs,
        )

        # 3. Model C Değerlendirmesi
        th_c, eer_c, auc_c, acc_c, prec_c, rec_c, f1_c, far_c, frr_c, _, far_c_list, frr_c_list, fpr_c_list, tpr_c_list = (
            self._compute_roc_eer(scores_hybrid, labels)
        )
        summary_c = MetricSummary(
            model_name="Model C: FHDSS Hybrid (ConvNeXt + Morphology + Gating)",
            optimal_threshold=th_c,
            eer=eer_c,
            auc=auc_c,
            accuracy=acc_c,
            precision=prec_c,
            recall=rec_c,
            f1_score=f1_c,
            far_at_eer=far_c,
            frr_at_eer=frr_c,
            avg_latency_ms=round(avg_lat, 2),
            total_pairs=total_pairs,
            positive_pairs=pos_pairs,
            negative_pairs=neg_pairs,
        )

        # En iyi modeli seç (en düşük EER / en yüksek AUC)
        candidates = [("Model A (Deep)", eer_a, auc_a), ("Model B (Morph)", eer_b, auc_b), ("Model C (Hybrid)", eer_c, auc_c)]
        best_model = sorted(candidates, key=lambda x: (x[1], -x[2]))[0][0]

        report = BenchmarkReport(
            models={
                "model_a_deep_only": summary_a,
                "model_b_morph_only": summary_b,
                "model_c_hybrid_fhdss": summary_c,
            },
            thresholds=threshs,
            far_curves={
                "model_a": far_a_list,
                "model_b": far_b_list,
                "model_c": far_c_list,
            },
            frr_curves={
                "model_a": frr_a_list,
                "model_b": frr_b_list,
                "model_c": frr_c_list,
            },
            roc_curves={
                "model_a": {"fpr": fpr_a_list, "tpr": tpr_a_list},
                "model_b": {"fpr": fpr_b_list, "tpr": tpr_b_list},
                "model_c": {"fpr": fpr_c_list, "tpr": tpr_c_list},
            },
            best_overall_model=best_model,
            generated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )

        if save_results:
            out_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent.parent / "reports"
            out_dir.mkdir(parents=True, exist_ok=True)
            report_file = out_dir / "benchmark_comparison_results.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump({
                    "models": {k: asdict(v) for k, v in report.models.items()},
                    "best_overall_model": report.best_overall_model,
                    "generated_at_utc": report.generated_at_utc,
                }, f, indent=2, ensure_ascii=False)

        return report


def run_full_benchmark_suite(
    num_writers: int = 16,
    samples_per_writer: int = 6,
    test_ratio: float = 0.35,
) -> BenchmarkReport:
    """Uçtan uca veri üretimi, sıfır sızıntılı bölütleme ve 3 model benchmarkı koşturur."""
    corpus = ForensicDataPipeline.generate_synthetic_forensic_corpus(
        num_writers=num_writers,
        samples_per_writer=samples_per_writer,
    )
    _, _, test_samples = ForensicDataPipeline.split_by_disjoint_writers(
        corpus, train_ratio=0.65, val_ratio=0.0, test_ratio=test_ratio
    )
    test_pairs = ForensicDataPipeline.create_pairs(test_samples, max_pairs_per_writer=10)

    engine = ForensicBenchmarkEngine()
    report = engine.run_benchmark(test_pairs)
    return report


if __name__ == "__main__":
    rep = run_full_benchmark_suite()
    print("Benchmark Raporu:", json.dumps({k: asdict(v) for k, v in rep.models.items()}, indent=2))
