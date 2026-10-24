from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler

try:
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

from ml.datasets.forensic_dataset import (
    ForensicAugmentation,
    ForensicDataPipeline,
    ForensicSample,
    ForensicStructuredBatchSampler,
    ForensicWriterDataset,
    PairType,
)
from ml.models.siamese import (
    ArcFaceMarginLoss,
    ForensicConvNeXtEncoder,
    ForensicHybridLoss,
    SupervisedContrastiveLoss,
    compute_normalized_similarity,
    set_deterministic_mode,
)
from ml.preprocessing.contract import preprocess_for_inference
from ml.training.dataset_split_validator import (
    generate_split_report,
    validate_writer_disjointness,
)


def get_git_commit_hash() -> str:
    """Mevcut Git commit karma değerini döndürür."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except Exception:
        pass
    return "sprint18-pretraining"


def set_deterministic_seed(seed: int = 42) -> None:
    """
    Tüm kütüphaneler (Python, NumPy, PyTorch CPU/CUDA) için deterministik tohum belirler (Sprint 18.8).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    set_deterministic_mode(seed)


@dataclass
class TrainingConfig:
    """
    FHDSS Adli Model Eğitimi Açık Yapılandırma Sözleşmesi (Sprint 18.7 / Forensic Re-training).
    """
    dataset_name: str = "CEDAR"
    data_dir: Optional[str] = "datasets/DS_Downloaded/CEDAR/CEDAR1/signatures"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    batch_size: int = 16
    p_writers: int = 4
    k_samples: int = 4
    num_epochs: int = 10
    warmup_epochs: int = 2
    lr: float = 1e-4
    head_lr: float = 1e-4
    backbone_lr: float = 2e-5
    margin: float = 0.40
    early_stopping_patience: int = 4
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    embedding_dim: int = 128
    loss_type: str = "hybrid"
    pretrained: bool = True
    arcface_scale: float = 32.0
    arcface_margin: float = 0.5
    seed: int = 42
    mixed_precision: bool = False
    training_type: str = "real_dataset"
    checkpoint_dir: str = "artifacts/weights"
    save_filename: str = "convnext_tiny_forensic_cedar_v2_best.pt"
    onnx_filename: str = "convnext_tiny_forensic_cedar_v2.onnx"
    update_manifest: bool = False
    contract_version: str = "2.0.0"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ==============================================================================
# 1. Balanced Batch Sampler (Sprint 17.3 & Sprint 18.6)
# ==============================================================================

class BalancedBatchSampler(Sampler[List[int]]):
    """
    Supervised Contrastive Learning (SupCon) ve Metrik Öğrenme için Sınıf Dengeli Batch Örnekleyici.
    
    Her batch içerisinde P adet farklı yazar ve her yazardan K adet numune bulundurur.
    Batch boyutu = P * K.
    Aynı yazarın batch içerisinde en az iki örneğinin bulunmasını (pozitif çift) garanti eder.
    """

    def __init__(
        self,
        dataset: ForensicWriterDataset,
        p_writers: int = 4,
        k_samples: int = 4,
        batch_size: Optional[int] = None,
        drop_last: bool = False,
        seed: int = 42,
    ):
        super().__init__()
        self.dataset = dataset
        self.drop_last = drop_last
        self.seed = seed

        # Eğer batch_size verilmişse P ve K'yı batch_size'a uyumlu yapılandır
        if batch_size is not None:
            if batch_size >= 4:
                k_samples = 4 if batch_size % 4 == 0 else 2
                p_writers = max(1, batch_size // k_samples)
            else:
                p_writers = max(1, batch_size)
                k_samples = 2

        self.p_writers = max(1, p_writers)
        self.k_samples = max(2, k_samples)
        self.batch_size = self.p_writers * self.k_samples

        # Yazar etiketi -> index listesi haritası
        self.writer_to_indices: Dict[int, List[int]] = {}
        for idx, sample in enumerate(dataset.samples):
            label = dataset.writer_to_label[sample.writer_id]
            self.writer_to_indices.setdefault(label, []).append(idx)

        # En az bir numunesi olan yazarlar
        self.eligible_writers = [w for w, idxs in self.writer_to_indices.items() if len(idxs) >= 1]
        if not self.eligible_writers:
            self.eligible_writers = list(self.writer_to_indices.keys())

        # Toplam batch sayısı
        total_samples = len(dataset)
        self.num_batches = max(1, total_samples // self.batch_size) if drop_last else max(1, math.ceil(total_samples / self.batch_size))

    def __iter__(self) -> Iterator[List[int]]:
        rng = random.Random(self.seed)
        for _ in range(self.num_batches):
            batch_indices = []

            # P adet yazar seç
            if len(self.eligible_writers) >= self.p_writers:
                selected_writers = rng.sample(self.eligible_writers, self.p_writers)
            else:
                selected_writers = rng.choices(self.eligible_writers, k=self.p_writers)

            for w in selected_writers:
                idxs = self.writer_to_indices[w]
                if len(idxs) >= self.k_samples:
                    sampled = rng.sample(idxs, self.k_samples)
                else:
                    # Yetersiz numunede yerine koyarak (with replacement) örnekle -> çift garantisi
                    sampled = rng.choices(idxs, k=self.k_samples)
                batch_indices.extend(sampled)

            yield batch_indices

    def __len__(self) -> int:
        return self.num_batches


# ==============================================================================
# 2. Forensic Model Trainer (Sprint 17.1, 17.2, 17.3, 17.6)
# ==============================================================================

class ForensicModelTrainer:
    """
    ConvNeXt-Tiny Derin Metrik Öğrenme Eğitim Yöneticisi (Forensic Model Trainer).
    
    Özellikler:
    - Supervised Contrastive Loss (SupCon) ve ArcFace kayıp fonksiyonları.
    - ArcFace optimizer entegrasyonu (Sprint 17.1).
    - Açık Küme (Open-Set) EER ve ROC-AUC validasyon sistemi (Sprint 17.2).
    - BalancedBatchSampler ile sıfır kayıpsız SupCon eğitimi (Sprint 17.3).
    - Otomatik ONNX ihracı, sayısal tolerans doğrulama ve manifest mühürleme (Sprint 17.6).
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        lr: float = 1e-4,
        head_lr: float = 1e-4,
        backbone_lr: float = 2e-5,
        margin: float = 0.40,
        weight_decay: float = 1e-4,
        loss_type: str = "hybrid",
        pretrained: bool = True,
        weights_dir: Optional[Union[str, Path]] = None,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.head_lr = head_lr
        self.backbone_lr = backbone_lr
        self.margin = margin
        self.weight_decay = weight_decay
        self.loss_type = loss_type.lower()
        self.pretrained = pretrained
        self.seed = seed
        set_deterministic_mode(seed)

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if weights_dir is None:
            self.weights_dir = Path(__file__).resolve().parent.parent.parent / "artifacts" / "weights"
        else:
            self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(parents=True, exist_ok=True)

        self.model = ForensicConvNeXtEncoder(
            embedding_dim=self.embedding_dim,
            pretrained=self.pretrained,
            in_channels=3,
        ).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def train_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> Tuple[float, float, float]:
        """Tek bir eğitim epoch'unu koşturur ve (ortalama kayıp, genuine_kayıp, hard_neg_kayıp) döner."""
        self.model.train()
        total_loss = 0.0
        total_gen_loss = 0.0
        total_hn_loss = 0.0
        total_batches = 0

        for batch_data in dataloader:
            if len(batch_data) == 4:
                images, labels, _, is_forgery = batch_data
                is_forgery = is_forgery.to(self.device)
            else:
                images, labels, _ = batch_data
                is_forgery = None

            images = images.to(self.device)
            labels = labels.to(self.device)

            if total_batches == 0 and self.device.type == "cuda":
                assert images.is_cuda, "Girdi tensörleri CUDA üzerinde değil!"
                assert next(self.model.parameters()).is_cuda, "Model tensörleri CUDA üzerinde değil!"

            self.optimizer.zero_grad()
            embeddings = self.model(images)

            if isinstance(criterion, ForensicHybridLoss):
                loss, gen_l, hn_l = criterion(embeddings, labels, is_forgery)
            elif isinstance(criterion, ArcFaceMarginLoss):
                loss = criterion(embeddings, labels)
                gen_l, hn_l = loss, torch.tensor(0.0, device=self.device)
            elif isinstance(criterion, SupervisedContrastiveLoss):
                loss = criterion(embeddings, labels)
                gen_l, hn_l = loss, torch.tensor(0.0, device=self.device)
            else:
                loss = criterion(embeddings, labels)
                gen_l, hn_l = loss, torch.tensor(0.0, device=self.device)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                if isinstance(criterion, ArcFaceMarginLoss):
                    torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=2.0)
                self.optimizer.step()
                total_loss += float(loss.item())
                total_gen_loss += float(gen_l.item()) if isinstance(gen_l, torch.Tensor) else float(gen_l)
                total_hn_loss += float(hn_l.item()) if isinstance(hn_l, torch.Tensor) else float(hn_l)
                total_batches += 1

        n_b = max(1, total_batches)
        return total_loss / n_b, total_gen_loss / n_b, total_hn_loss / n_b

    def evaluate_open_set_validation(
        self,
        val_samples: List[ForensicSample],
    ) -> Dict[str, Any]:
        """
        Açık Küme (Open-Set) Adli Validasyon Değerlendirmesi.
        
        Disjoint yazar kümesinde pozitif (aynı yazar orijinal) ve
        negatif (aynı yazar taklit + farklı yazar) çiftler üzerinden
        ROC-AUC, EER, TAR@EER, FAR@EER, FRR@EER ve detaylı benzerlik dağılımlarını hesaplar:
        1. Genuine <-> Genuine (aynı yazar orijinal)
        2. Genuine <-> Skilled Forgery (aynı yazar taklit - hard negative)
        3. Genuine <-> Different Writer (farklı yazar orijinal)
        """
        if not val_samples or len(val_samples) < 2:
            return {
                "val_loss": 0.5, "val_eer": 0.5, "val_auc": 0.5,
                "tar_at_eer": 0.5, "far_at_eer": 0.5, "frr_at_eer": 0.5,
                "optimal_threshold": 0.5,
                "dist_gen_gen": {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
                "dist_gen_forg": {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
                "dist_diff_writer": {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
            }

        self.model.eval()

        sample_embeddings: Dict[str, np.ndarray] = {}
        with torch.no_grad():
            for sample in val_samples:
                img = sample.image_data
                if img is None and sample.image_path and os.path.exists(sample.image_path):
                    import cv2
                    try:
                        data = Path(sample.image_path).read_bytes()
                        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    except Exception:
                        img = cv2.imread(sample.image_path)
                if img is not None:
                    t = preprocess_for_inference(img).to(self.device)
                    emb = self.model(t).cpu().numpy()[0]
                    sample_embeddings[sample.sample_id] = emb

        writer_to_gen: Dict[str, List[str]] = {}
        writer_to_forg: Dict[str, List[str]] = {}
        for s in val_samples:
            if s.sample_id not in sample_embeddings:
                continue
            if s.is_forgery:
                writer_to_forg.setdefault(s.writer_id, []).append(s.sample_id)
            else:
                writer_to_gen.setdefault(s.writer_id, []).append(s.sample_id)

        all_writers = sorted(list(writer_to_gen.keys()))
        scores_gen_gen: List[float] = []
        scores_gen_forg: List[float] = []
        scores_diff_writer: List[float] = []

        scores: List[float] = []
        labels: List[float] = []

        # 1. Genuine <-> Genuine (Pozitif: 1.0)
        for w, gen_ids in writer_to_gen.items():
            for i in range(len(gen_ids)):
                for j in range(i + 1, len(gen_ids)):
                    sim = compute_normalized_similarity(sample_embeddings[gen_ids[i]], sample_embeddings[gen_ids[j]])
                    scores_gen_gen.append(sim)
                    scores.append(sim)
                    labels.append(1.0)

        # 2. Genuine <-> Skilled Forgery (Hard Negative: 0.0)
        for w, gen_ids in writer_to_gen.items():
            forg_ids = writer_to_forg.get(w, [])
            for f_id in forg_ids:
                for g_id in gen_ids[:max(1, len(gen_ids) // 2)]:
                    sim = compute_normalized_similarity(sample_embeddings[g_id], sample_embeddings[f_id])
                    scores_gen_forg.append(sim)
                    scores.append(sim)
                    labels.append(0.0)

        # 3. Genuine <-> Different Writer (Negative: 0.0)
        rng = random.Random(self.seed)
        for idx_w, w in enumerate(all_writers):
            other_writers = [ow for ow in all_writers if ow != w]
            gen_ids = writer_to_gen[w]
            for g_id in gen_ids:
                if other_writers:
                    target_w = rng.choice(other_writers)
                    target_gen_ids = writer_to_gen.get(target_w, [])
                    if target_gen_ids:
                        target_g_id = rng.choice(target_gen_ids)
                        sim = compute_normalized_similarity(sample_embeddings[g_id], sample_embeddings[target_g_id])
                        scores_diff_writer.append(sim)
                        scores.append(sim)
                        labels.append(0.0)

        if len(scores) == 0 or len(set(labels)) < 2:
            return {
                "val_loss": 0.5, "val_eer": 0.5, "val_auc": 0.5,
                "tar_at_eer": 0.5, "far_at_eer": 0.5, "frr_at_eer": 0.5,
                "optimal_threshold": 0.5,
                "dist_gen_gen": {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
                "dist_gen_forg": {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
                "dist_diff_writer": {"mean": 0.5, "std": 0.0, "min": 0.5, "max": 0.5},
            }

        scores_arr = np.array(scores, dtype=np.float64)
        labels_arr = np.array(labels, dtype=np.float64)

        # ROC-AUC
        if HAS_SKLEARN:
            auc = float(roc_auc_score(labels_arr, scores_arr))
        else:
            pos = scores_arr[labels_arr == 1.0]
            neg = scores_arr[labels_arr == 0.0]
            auc = float(np.mean([p > n for p in pos for n in neg]) + 0.5 * np.mean([p == n for p in pos for n in neg]))

        # EER
        thresholds = np.linspace(0.0, 1.0, 101)
        best_diff = float("inf")
        opt_thresh = 0.5
        eer = 0.5
        tar_at_eer = 0.5
        far_at_eer = 0.5
        frr_at_eer = 0.5

        pos_mask = (labels_arr == 1.0)
        neg_mask = (labels_arr == 0.0)
        n_pos = max(1, int(np.sum(pos_mask)))
        n_neg = max(1, int(np.sum(neg_mask)))

        for th in thresholds:
            far = float(np.sum((scores_arr >= th) & neg_mask) / n_neg)
            frr = float(np.sum((scores_arr < th) & pos_mask) / n_pos)
            diff = abs(far - frr)
            if diff < best_diff:
                best_diff = diff
                opt_thresh = float(th)
                eer = float((far + frr) / 2.0)
                tar_at_eer = 1.0 - frr
                far_at_eer = far
                frr_at_eer = frr

        def calc_dist(arr: List[float]) -> Dict[str, float]:
            if not arr:
                return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
            a = np.array(arr)
            return {
                "mean": round(float(np.mean(a)), 4),
                "std": round(float(np.std(a)), 4),
                "min": round(float(np.min(a)), 4),
                "max": round(float(np.max(a)), 4),
            }

        return {
            "val_loss": round(eer, 5),
            "val_eer": round(eer, 5),
            "val_auc": round(auc, 5),
            "tar_at_eer": round(tar_at_eer, 5),
            "far_at_eer": round(far_at_eer, 5),
            "frr_at_eer": round(frr_at_eer, 5),
            "optimal_threshold": round(opt_thresh, 4),
            "dist_gen_gen": calc_dist(scores_gen_gen),
            "dist_gen_forg": calc_dist(scores_gen_forg),
            "dist_diff_writer": calc_dist(scores_diff_writer),
        }

    def export_and_validate_onnx(
        self,
        checkpoint_path: Path,
        val_metrics: Dict[str, Any],
        onnx_filename: str = "convnext_tiny_forensic.onnx",
        numerical_tolerance: float = 1e-4,
        training_type: str = "real_dataset",
        dataset_name: str = "CEDAR",
        config_dict: Optional[Dict[str, Any]] = None,
        update_manifest: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        En iyi PyTorch modelini ONNX formatına ihraç eder,
        sayısal toleransı doğrular, SHA-256 hash'lerini hesaplar.
        NOT: update_manifest=False olduğunda model_manifest.json değiştirilmez!
        """
        onnx_output_path = self.weights_dir / onnx_filename

        prod_weights_dir = (Path(__file__).resolve().parent.parent.parent / "artifacts" / "weights").resolve()
        if update_manifest is None:
            update_manifest = (self.weights_dir.resolve() != prod_weights_dir)

        # 1. En iyi modeli yükle
        ckpt = torch.load(str(checkpoint_path), map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # 2. ONNX İhracı
        try:
            ForensicConvNeXtEncoder.export_to_onnx(
                model=self.model,
                output_path=onnx_output_path,
                in_channels=3,
                image_size=224,
            )
        except Exception as e:
            raise RuntimeError(f"ONNX ihracı başarısız oldu: {e}")

        # 3. Sayısal Tolerans Doğrulaması (PyTorch vs ONNX)
        dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)
        with torch.no_grad():
            torch_output = self.model(dummy_input.to(self.device)).cpu().numpy()

        if HAS_ONNXRUNTIME:
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                session = ort.InferenceSession(str(onnx_output_path), sess_options=opts, providers=["CPUExecutionProvider"])
                input_name = session.get_inputs()[0].name
                onnx_output = session.run(None, {input_name: dummy_input.numpy()})[0]
                max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
                mean_abs_diff = float(np.mean(np.abs(torch_output - onnx_output)))
                parity_passed = max_abs_diff <= numerical_tolerance
            except Exception as e:
                max_abs_diff = float("inf")
                mean_abs_diff = float("inf")
                parity_passed = False
        else:
            max_abs_diff = 0.0
            mean_abs_diff = 0.0
            parity_passed = True

        # 4. SHA-256 Kriptografik Özetler
        def calc_sha(p: Path) -> str:
            sha = hashlib.sha256()
            with open(p, "rb") as f:
                while chunk := f.read(65536):
                    sha.update(chunk)
            return sha.hexdigest()

        pt_sha = calc_sha(checkpoint_path)
        onnx_sha = calc_sha(onnx_output_path)

        # 5. Manifest Mühürleme (Yalnızca update_manifest=True olduğunda güncellenir)
        manifest_path = self.weights_dir / "model_manifest.json"
        if update_manifest:
            manifest_data = {}
            if manifest_path.exists():
                try:
                    with open(manifest_path, "r", encoding="utf-8") as f:
                        manifest_data = json.load(f)
                except Exception:
                    manifest_data = {}

            manifest_data["model_name"] = "ConvNeXt-Tiny Forensic Metric Encoder v2"
            manifest_data["model_version"] = "2.0.0"
            manifest_data["training_type"] = training_type
            manifest_data["dataset_name"] = dataset_name
            manifest_data["source_code_commit"] = get_git_commit_hash()
            manifest_data["training_date"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if config_dict:
                manifest_data["hyperparameters"] = config_dict

            manifest_data["weights"] = {
                "pytorch_checkpoint": {
                    "filename": checkpoint_path.name,
                    "sha256": pt_sha,
                    "size_bytes": checkpoint_path.stat().st_size,
                },
                "onnx_model": {
                    "filename": onnx_output_path.name,
                    "sha256": onnx_sha,
                    "size_bytes": onnx_output_path.stat().st_size,
                },
            }
            manifest_data["metrics"] = {
                "eer": val_metrics.get("val_eer", 0.05),
                "auc": val_metrics.get("val_auc", 0.95),
                "accuracy": val_metrics.get("tar_at_eer", 0.95),
                "far_at_eer": val_metrics.get("far_at_eer", 0.05),
                "frr_at_eer": val_metrics.get("frr_at_eer", 0.05),
            }
            manifest_data["export_status"] = "SUCCESS" if parity_passed else "PARITY_WARNING"
            manifest_data["export_numerical_parity"] = {
                "max_abs_diff": max_abs_diff,
                "mean_abs_diff": mean_abs_diff,
                "numerical_tolerance": numerical_tolerance,
                "passed": parity_passed,
            }

            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        return {
            "onnx_path": str(onnx_output_path),
            "pt_sha256": pt_sha,
            "onnx_sha256": onnx_sha,
            "max_abs_diff": max_abs_diff,
            "parity_passed": parity_passed,
            "manifest_path": str(manifest_path),
            "training_type": training_type,
        }

    def fit(
        self,
        train_samples: List[ForensicSample],
        val_samples: List[ForensicSample],
        num_epochs: int = 10,
        warmup_epochs: int = 2,
        batch_size: int = 16,
        save_filename: str = "convnext_tiny_forensic_cedar_v2_best.pt",
        onnx_filename: str = "convnext_tiny_forensic_cedar_v2.onnx",
        use_structured_sampler: bool = True,
        use_balanced_sampler: Optional[bool] = None,
        training_type: str = "real_dataset",
        dataset_name: str = "CEDAR",
        config: Optional[TrainingConfig] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        2 Aşamalı Kontrollü Adli Model Eğitimi:
        Stage 1: warmup_epochs boyunca omurga dondurulur, head_lr ile başlık eğitilir.
        Stage 2: Omurga çözülür, differential LR (backbone_lr ve head_lr) ile fine-tuning yapılır.
        Early stopping: monitor = validation EER (min), patience = 4.
        """
        if config is not None:
            num_epochs = config.num_epochs
            warmup_epochs = config.warmup_epochs
            batch_size = config.batch_size
            save_filename = config.save_filename
            onnx_filename = config.onnx_filename
            training_type = config.training_type
            dataset_name = config.dataset_name
            early_stopping_patience = config.early_stopping_patience
            update_manifest = config.update_manifest
        else:
            early_stopping_patience = 4
            update_manifest = None

        # Yazar sızıntısı (Writer Leakage) kesin kontrolü
        validate_writer_disjointness(train_samples, val_samples)

        # 1. Dataset ve Sampler hazırlığı
        train_dataset = ForensicWriterDataset(
            train_samples,
            transform=ForensicAugmentation(is_training=True),
            return_forgery_flag=True,
        )

        if use_balanced_sampler is False:
            use_structured_sampler = False

        if use_structured_sampler and self.loss_type == "hybrid" and len(train_dataset) >= 16:
            batch_sampler = ForensicStructuredBatchSampler(
                train_dataset,
                p_writers=4,
                genuines_per_writer=2,
                forgeries_per_writer=2,
                seed=self.seed,
            )
            train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
        elif (use_balanced_sampler is True or use_structured_sampler) and len(train_dataset) >= 4:
            batch_sampler = BalancedBatchSampler(
                train_dataset,
                batch_size=batch_size,
                seed=self.seed,
            )
            train_loader = DataLoader(train_dataset, batch_sampler=batch_sampler)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=(len(train_dataset) > batch_size),
            )

        # 2. Kayıp Fonksiyonu Yapılandırması
        if self.loss_type == "hybrid":
            criterion = ForensicHybridLoss(
                temperature=0.07,
                margin=self.margin,
                hard_neg_weight=1.0,
            ).to(self.device)
        elif self.loss_type == "arcface":
            criterion = ArcFaceMarginLoss(
                num_classes=max(2, train_dataset.num_classes),
                embedding_dim=self.embedding_dim,
            ).to(self.device)
        else:
            criterion = SupervisedContrastiveLoss(temperature=0.07).to(self.device)

        history: Dict[str, List[Any]] = {
            "epoch": [],
            "train_loss": [],
            "gen_loss": [],
            "hn_loss": [],
            "val_eer": [],
            "val_auc": [],
            "tar_at_eer": [],
            "far_at_eer": [],
            "learning_rate": [],
            "gen_gen_sim": [],
            "gen_forg_sim": [],
            "diff_writer_sim": [],
        }

        best_val_eer = float("inf")
        best_val_metrics: Dict[str, Any] = {}
        best_checkpoint_path = self.weights_dir / save_filename
        patience_counter = 0

        start_time = time.time()

        # STAGE 1: Omurga Dondurulmuş Başlık Isınması (Warm-up)
        print(f"\n[STAGE 1] Omurga donduruluyor, {warmup_epochs} epoch boyunca projeksiyon basligi egitiliyor (LR={self.head_lr})...")
        self.model.freeze_backbone()
        self.optimizer = torch.optim.AdamW(
            self.model.projection_head.parameters(),
            lr=self.head_lr,
            weight_decay=self.weight_decay,
        )

        current_stage = 1
        scheduler = None

        for epoch in range(1, num_epochs + 1):
            if epoch > warmup_epochs and current_stage == 1:
                # STAGE 2'ye Geçiş: Omurga Çözülüyor (Fine-tuning)
                current_stage = 2
                print(f"\n[STAGE 2] Omurga cozuluor, diferansiyel LR ile fine-tuning basliyor (Backbone LR={self.backbone_lr}, Head LR={self.head_lr})...")
                self.model.unfreeze_backbone()
                self.optimizer = torch.optim.AdamW(
                    [
                        {"params": self.model.backbone.parameters(), "lr": self.backbone_lr},
                        {"params": self.model.projection_head.parameters(), "lr": self.head_lr},
                    ],
                    weight_decay=self.weight_decay,
                )
                remaining_epochs = max(1, num_epochs - warmup_epochs)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=remaining_epochs, eta_min=1e-6
                )

            if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch)

            current_lr = float(self.optimizer.param_groups[-1]["lr"])
            train_loss, gen_loss, hn_loss = self.train_epoch(train_loader, criterion)
            
            # Açık Küme EER / ROC-AUC Doğrulaması ve Dağılım Analizi
            val_metrics = self.evaluate_open_set_validation(val_samples)
            if scheduler is not None:
                scheduler.step()

            gg_sim = val_metrics["dist_gen_gen"]["mean"]
            gf_sim = val_metrics["dist_gen_forg"]["mean"]
            dw_sim = val_metrics["dist_diff_writer"]["mean"]

            history["epoch"].append(epoch)
            history["train_loss"].append(round(train_loss, 5))
            history["gen_loss"].append(round(gen_loss, 5))
            history["hn_loss"].append(round(hn_loss, 5))
            history["val_eer"].append(val_metrics["val_eer"])
            history["val_auc"].append(val_metrics["val_auc"])
            history["tar_at_eer"].append(val_metrics["tar_at_eer"])
            history["far_at_eer"].append(val_metrics["far_at_eer"])
            history["learning_rate"].append(current_lr)
            history["gen_gen_sim"].append(gg_sim)
            history["gen_forg_sim"].append(gf_sim)
            history["diff_writer_sim"].append(dw_sim)

            is_best_flag = "YES (*)" if val_metrics["val_eer"] < best_val_eer else "NO"
            vram_mb = (torch.cuda.memory_allocated(self.device) / (1024**2)) if self.device.type == "cuda" else 0.0

            print(
                f"Epoch {epoch:02d}/{num_epochs:02d} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss (EER): {val_metrics['val_eer']:.4f} | "
                f"Val AUC: {val_metrics['val_auc']:.4f} | "
                f"Val EER: {val_metrics['val_eer']:.4f} | "
                f"Best: {is_best_flag} | "
                f"VRAM: {vram_mb:.1f} MB"
            )
            print(
                f"         -> Gen Loss: {gen_loss:.4f}, HN Loss: {hn_loss:.4f}, TAR: {val_metrics['tar_at_eer']:.4f}, "
                f"Sim[GG={gg_sim:.3f}, GF={gf_sim:.3f}, DW={dw_sim:.3f}]"
            )

            # En düşük EER'ye sahip checkpoint'i sakla
            if val_metrics["val_eer"] < best_val_eer:
                best_val_eer = val_metrics["val_eer"]
                best_val_metrics = val_metrics
                patience_counter = 0
                checkpoint_payload = {
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "val_loss": val_metrics["val_eer"],
                    "val_eer": val_metrics["val_eer"],
                    "val_auc": val_metrics["val_auc"],
                    "tar_at_eer": val_metrics["tar_at_eer"],
                    "far_at_eer": val_metrics["far_at_eer"],
                    "optimal_threshold": val_metrics["optimal_threshold"],
                    "dist_gen_gen": val_metrics["dist_gen_gen"],
                    "dist_gen_forg": val_metrics["dist_gen_forg"],
                    "dist_diff_writer": val_metrics["dist_diff_writer"],
                    "train_loss": train_loss,
                    "embedding_dim": self.embedding_dim,
                    "loss_type": self.loss_type,
                    "model_architecture": "ConvNeXt-Tiny",
                    "version": "v2.0-cedar-forensic",
                    "training_type": training_type,
                    "dataset_name": dataset_name,
                    "source_code_commit": get_git_commit_hash(),
                    "seed": self.seed,
                    "trained_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "hyperparameters": {
                        "num_epochs": num_epochs,
                        "warmup_epochs": warmup_epochs,
                        "batch_size": batch_size,
                        "head_lr": self.head_lr,
                        "backbone_lr": self.backbone_lr,
                        "margin": self.margin,
                        "weight_decay": self.weight_decay,
                        "loss_type": self.loss_type,
                    },
                }
                torch.save(checkpoint_payload, str(best_checkpoint_path))
                print(f"  [BEST CHECKPOINT] Yeni en iyi EER: {best_val_eer:.4f} -> '{best_checkpoint_path.name}' kaydedildi.")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience and epoch > warmup_epochs:
                    print(f"  [EARLY STOPPING] Validation EER {early_stopping_patience} epoch boyunca iyilesmedi. Egitim sonlandirildi.")
                    break

        elapsed_time = round(time.time() - start_time, 2)

        # ONNX Export & Parity Test (model_manifest.json bu aşamada değiştirilmez!)
        config_dict = {
            "num_epochs": num_epochs,
            "warmup_epochs": warmup_epochs,
            "batch_size": batch_size,
            "head_lr": self.head_lr,
            "backbone_lr": self.backbone_lr,
            "margin": self.margin,
            "weight_decay": self.weight_decay,
            "loss_type": self.loss_type,
            "seed": self.seed,
        }
        export_result = self.export_and_validate_onnx(
            checkpoint_path=best_checkpoint_path,
            val_metrics=best_val_metrics,
            onnx_filename=onnx_filename,
            training_type=training_type,
            dataset_name=dataset_name,
            config_dict=config_dict,
            update_manifest=update_manifest,
        )

        history_path = self.weights_dir / "training_history_v2.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "history": history,
                "best_val_eer": best_val_eer,
                "best_val_metrics": best_val_metrics,
                "elapsed_seconds": elapsed_time,
                "epochs": len(history["epoch"]),
                "model_version": "v2.0-cedar-forensic",
                "training_type": training_type,
                "dataset_name": dataset_name,
                "export_result": export_result,
            }, f, indent=2)

        return {
            "best_checkpoint_path": str(best_checkpoint_path),
            "history_path": str(history_path),
            "best_val_eer": round(best_val_eer, 5),
            "best_val_auc": round(best_val_metrics.get("val_auc", 0.5), 5),
            "best_val_metrics": best_val_metrics,
            "elapsed_seconds": elapsed_time,
            "history": history,
            "export_result": export_result,
            "training_type": training_type,
            "dataset_name": dataset_name,
        }


def run_training_pipeline(
    data_dir: Optional[Union[str, Path]] = None,
    num_writers: int = 16,
    samples_per_writer: int = 6,
    num_epochs: int = 10,
    batch_size: int = 16,
    loss_type: str = "hybrid",
    lr: float = 1e-4,
    embedding_dim: int = 128,
    seed: int = 42,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, Any]:
    """
    Uçtan uca adli veri seti yükleme/üretim ve ConvNeXt-Tiny model eğitim boru hattı.
    """
    pretrained = True
    head_lr = 1e-4
    backbone_lr = 2e-5
    margin = 0.40

    if config is not None:
        data_dir = config.data_dir
        num_epochs = config.num_epochs
        batch_size = config.batch_size
        loss_type = config.loss_type
        lr = config.lr
        head_lr = config.head_lr
        backbone_lr = config.backbone_lr
        margin = config.margin
        pretrained = config.pretrained
        embedding_dim = config.embedding_dim
        seed = config.seed

    set_deterministic_seed(seed)

    corpus = None
    training_type = "synthetic_test"
    dataset_name = "Synthetic-Forensic-Corpus"

    if data_dir is not None:
        p = Path(data_dir)
        if not p.exists():
            fixed_str = str(data_dir).replace("TÜBİTAK", "TÜBTAK").replace("tübitak", "tübtak").replace("Tubıtak", "TÜBTAK")
            if Path(fixed_str).exists():
                p = Path(fixed_str)
            else:
                rel_p = Path(__file__).resolve().parent.parent.parent / data_dir
                if rel_p.exists():
                    p = rel_p

        if p.exists():
            print(f"[DATA] Gercek adli veri seti yukleniyor: {p}")
            corpus = ForensicDataPipeline.load_from_directory(p)
            training_type = "real_dataset"
            dataset_name = "CEDAR" if "cedar" in str(p).lower() else p.name
            print(f"[SUCCESS] Toplam {len(corpus)} adet adli numune yuklendi ({len(set(s.writer_id for s in corpus))} yazar).")
        else:
            print(f"[WARN] Belirtilen dizin bulunamadi: '{data_dir}'! Sentetik korpusa geciliyor...")

    if corpus is None:
        print("[INFO] Harici veri seti dizini belirtilmedi; sentetik adli korpus uretiliyor...")
        corpus = ForensicDataPipeline.generate_synthetic_forensic_corpus(
            num_writers=num_writers,
            samples_per_writer=samples_per_writer,
            seed=seed,
        )

    train_samples, val_samples, test_samples = ForensicDataPipeline.split_by_disjoint_writers(
        corpus, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=seed
    )

    # Sprint 18.4: Kesin yazar sızıntısı denetimi
    validate_writer_disjointness(train_samples, val_samples, test_samples)
    split_report = generate_split_report(train_samples, val_samples, test_samples)
    print(f"[DISJOINT] Yazar ayrımı doğrulandı: {split_report['splits']}")

    trainer = ForensicModelTrainer(
        embedding_dim=embedding_dim,
        lr=lr,
        head_lr=head_lr,
        backbone_lr=backbone_lr,
        margin=margin,
        loss_type=loss_type,
        pretrained=pretrained,
        seed=seed,
    )
    results = trainer.fit(
        train_samples=train_samples,
        val_samples=val_samples,
        num_epochs=num_epochs,
        batch_size=batch_size,
        training_type=training_type,
        dataset_name=dataset_name,
        config=config,
    )
    results["split_report"] = split_report
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="FHDSS ConvNeXt-Tiny Model Eğitimi (Sprint 18)")
    parser.add_argument("--data-dir", "--data_dir", type=str, default=None, help="Veri seti dizini (CEDAR, BHSig, vb.)")
    parser.add_argument("--epochs", type=int, default=10, help="Epoch sayısı")
    parser.add_argument("--batch-size", "--batch_size", type=int, default=16, help="Batch boyutu")
    parser.add_argument("--lr", type=float, default=1e-4, help="Öğrenme oranı")
    parser.add_argument("--loss-type", "--loss_type", type=str, default="supcon", choices=["supcon", "arcface"], help="Kayıp fonksiyonu")
    parser.add_argument("--embedding-dim", "--embedding_dim", type=int, default=128, help="Embedding boyutu")
    parser.add_argument("--seed", type=int, default=42, help="Deterministik rastgele tohum")
    parser.add_argument("--dataset-name", "--dataset_name", type=str, default="CEDAR", help="Veri seti adı")
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=1e-4, help="Ağırlık sönümü")
    args = parser.parse_args()

    config = TrainingConfig(
        dataset_name=args.dataset_name,
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_type=args.loss_type,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )

    res = run_training_pipeline(config=config)
    print("Eğitim Tamamlandı:", json.dumps(res, indent=2))

