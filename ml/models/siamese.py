import math
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False

try:
    from pytorch_metric_learning import losses as pml_losses
    HAS_PML = True
except ImportError:
    HAS_PML = False


def set_deterministic_mode(seed: int = 42) -> None:
    """
    Model çıkarım ve eğitiminin CPU ve CUDA üzerinde tam deterministik ve tekrarlanabilir olmasını sağlar.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Bazı CPU kernel operasyonları strict deterministic flag desteklemeyebilir
        pass


class ForensicConvNeXtEncoder(nn.Module):
    """
    Adli Belge ve El Yazısı İncelemesi için ConvNeXt-Tiny Tabanlı Derin Metrik Kodlayıcı.
    
    Özellikler:
    - ConvNeXt-Tiny omurgası (timm veya torchvision fallback).
    - 128 boyutlu L2-normalize birim küre gömme (embedding) projeksiyonu.
    - Tek kanal (Grayscale) ve 3-kanal (RGB) adli görsel desteği.
    - CPU ve CUDA üzerinde tam deterministik çıkarım.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = False,
        in_channels: int = 3,
        dropout_p: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.in_channels = in_channels
        self.is_pretrained = pretrained

        # 1. ConvNeXt-Tiny Omurga Yapılandırması
        if pretrained:
            try:
                weights = models.ConvNeXt_Tiny_Weights.DEFAULT
                convnext = models.convnext_tiny(weights=weights)
                in_features = convnext.classifier[2].in_features
                convnext.classifier = nn.Identity()
                self.backbone = convnext
            except Exception as e:
                raise RuntimeError(
                    f"KRİTİK HATA: ConvNeXt_Tiny_Weights.DEFAULT ağırlıkları yüklenemedi: {e}. "
                    "Pretrained=False fallback reddedildi!"
                )
        else:
            if HAS_TIMM:
                try:
                    self.backbone = timm.create_model(
                        "convnext_tiny",
                        pretrained=False,
                        num_classes=0,
                        in_chans=in_channels,
                    )
                    in_features = getattr(self.backbone, "num_features", 768)
                except Exception:
                    convnext = models.convnext_tiny(weights=None)
                    in_features = convnext.classifier[2].in_features
                    convnext.classifier = nn.Identity()
                    self.backbone = convnext
            else:
                convnext = models.convnext_tiny(weights=None)
                in_features = convnext.classifier[2].in_features
                convnext.classifier = nn.Identity()
                self.backbone = convnext

        # 2. 128 Boyutlu L2-Normalize Projeksiyon Başlığı
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, embedding_dim),
        )

    def freeze_backbone(self) -> None:
        """Omurga (backbone) ağırlıklarını dondurur; yalnızca projeksiyon başlığı güncellenir."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Omurga (backbone) ağırlıklarını çözer; uçtan uca ince ayar (fine-tuning) yapılır."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Girdi tensörünü `(B, C, H, W)` 128 boyutlu L2-normalize embedding vektörüne `(B, 128)` dönüştürür.
        
        Args:
            x: `(B, C, H, W)` veya `(C, H, W)` biçiminde görsel tensörü.
        
        Returns:
            normalized_embeddings: `(B, 128)` boyutunda ||e||_2 = 1.0 normalize embeddingler.
        """
        # Boyut hizalama
        if x.dim() == 3:
            x = x.unsqueeze(0)

        # Tek kanal gri görsel gelirse 3 kanala çoğalt
        if x.size(1) == 1 and self.in_channels == 3:
            x = x.repeat(1, 3, 1, 1)

        features = self.backbone(x)
        if features.dim() > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)

        raw_embeddings = self.projection_head(features)
        
        # Birim küre üzerinde L2 Normalizasyonu: ||emb||_2 = 1.0
        normalized_embeddings = F.normalize(raw_embeddings, p=2, dim=-1)
        return normalized_embeddings

    def extract_embedding(self, image: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Tekil görsel veya tensörden L2-normalize embedding çıkarır.
        Sprint 17.4: Ortak preprocessing sözleşmesi (RGB + ImageNet Norm + Padding) uygulanır.
        """
        self.eval()
        with torch.inference_mode():
            if isinstance(image, np.ndarray):
                from ml.preprocessing.contract import preprocess_for_inference
                device = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
                tensor = preprocess_for_inference(image).to(device)
            elif isinstance(image, torch.Tensor):
                tensor = image
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
            else:
                raise TypeError(f"Desteklenmeyen girdi türü: {type(image)}")

            return self.forward(tensor)

    @classmethod
    def export_to_onnx(
        cls,
        model: "ForensicConvNeXtEncoder",
        output_path: Union[str, Path],
        in_channels: int = 3,
        image_size: int = 224,
    ) -> str:
        """PyTorch ForensicConvNeXtEncoder modelini optimize edilmiş ONNX formatına dönüştürür."""
        model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else torch.device("cpu")
        dummy_input = torch.randn(1, in_channels, image_size, image_size, device=device)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(out_p),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["embedding"],
                dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy_input,
                str(out_p),
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["embedding"],
                dynamic_axes={"input": {0: "batch_size"}, "embedding": {0: "batch_size"}},
            )
        return str(out_p)




class ArcFaceMarginLoss(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) Kayıp Fonksiyonu.
    
    Adli yazar doğrulama (Writer Verification) görevinde sınıf içi mesafeyi daraltıp
    sınıflar arası açısal ayrımı maksimuma çıkarır:
    L = -log( e^(s * cos(theta_y + m)) / (e^(s * cos(theta_y + m)) + sum_{j!=y} e^(s * cos(theta_j))) )
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 128,
        scale: float = 64.0,
        margin: float = 0.50,
        easy_margin: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin

        # Sınıf ağırlık matrisi W (num_classes x embedding_dim)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalize gömme vektörleri `(B, embedding_dim)`
            labels: Gerçek yazar sınıf etiketleri `(B,)`
        """
        w_norm = F.normalize(self.weight, p=2, dim=1)
        cosine = F.linear(embeddings, w_norm)
        sine = torch.sqrt(torch.clamp(1.0 - torch.pow(cosine, 2), 1e-7, 1.0))
        
        # cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale

        loss = F.cross_entropy(logits, labels)
        return loss


class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Learning (SupCon) Kayıp Fonksiyonu.
    
    Aynı yazarın farklı imza/yazı örneklerini (pozitif çiftler) birim kürede birbirine yaklaştırırken,
    farklı yazarların örneklerini (negatif çiftler) birbirinden uzaklaştırır.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: L2-normalize gömme vektörleri `(B, embedding_dim)`
            labels: Yazar etiketleri `(B,)`
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Benzerlik matrisi (B x B)
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Öz-benzerlik maskesi (diagonal)
        diag_mask = torch.eye(batch_size, dtype=torch.bool, device=device)

        # Pozitif etiket maskesi: labels[i] == labels[j]
        labels = labels.contiguous().view(-1, 1)
        label_mask = torch.eq(labels, labels.T).to(device)
        positive_mask = label_mask & ~diag_mask

        # Sayısal kararlılık için max çıkarımı
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        exp_logits = torch.exp(logits) * (~diag_mask)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-7)

        # Pozitif çiftlerin ortalama log olasılığı
        pos_counts = positive_mask.sum(1)
        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / torch.clamp(pos_counts, min=1.0)

        # Yalnızca en az bir pozitif örneği olanlar üzerinden ortalama al
        valid_mask = pos_counts > 0
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = -mean_log_prob_pos[valid_mask].mean()
        return loss


class ForensicHybridLoss(nn.Module):
    """
    Adli Bilişim İçin Hibrit Kayıp Fonksiyonu (Forensic Hybrid Loss).
    
    Adli imza/el yazısı incelemesinde label collision (etiket çatışması) sorununu kesin olarak önler:
    - Orijinal (genuine) imzalar yazar kimliğine (writer identity) göre pozitif çift olarak kümelenir.
    - Yetenekli taklitler / sahte örnekler (skilled forgeries) asla yazar sınıfına pozitif dahil edilmez;
      aksine 'HARD NEGATIVE' olarak kayba iletilir ve marjin cezasıyla bastırılır.
    
    Bileşenler:
    1. L_genuine: Yalnızca orijinal numuneler arasındaki Supervised Contrastive Loss.
       sim(Genuine_A, Genuine_B) -> 1.0 (arttırılır).
    2. L_hard_neg: Aynı yazarın orijinali ile taklidi arasındaki marjin cezası.
       sim(Genuine_A, Forgery_A) -> 0.0 (azaltılır, en azından pos_sim - margin altına itilir).
    """
    def __init__(
        self,
        temperature: float = 0.07,
        margin: float = 0.40,
        hard_neg_weight: float = 1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.margin = margin
        self.hard_neg_weight = hard_neg_weight
        self.supcon = SupervisedContrastiveLoss(temperature=temperature)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        is_forgery: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            embeddings: (B, D) L2-normalize tensör
            labels: (B,) Yazar etiketleri
            is_forgery: (B,) Boole tensörü (True: Sahte, False: Orijinal)
        Returns:
            (total_loss, genuine_loss, hard_neg_loss)
        """
        device = embeddings.device
        batch_size = embeddings.shape[0]

        if is_forgery is None:
            loss = self.supcon(embeddings, labels)
            return loss, loss, torch.tensor(0.0, device=device)

        is_forgery_bool = is_forgery.bool()
        gen_mask = ~is_forgery_bool
        forg_mask = is_forgery_bool

        # 1. Component A: Genuine SupCon Loss (Yalnızca Orijinal Örnekler)
        gen_embeddings = embeddings[gen_mask]
        gen_labels = labels[gen_mask]

        if gen_embeddings.shape[0] >= 2 and len(torch.unique(gen_labels)) >= 1:
            gen_loss = self.supcon(gen_embeddings, gen_labels)
        else:
            gen_loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 2. Component B: Hard Negative Triplet / Margin Loss (Orijinal vs Taklit)
        hard_neg_losses = []
        unique_writers = torch.unique(labels)

        for w in unique_writers:
            w_gen_idx = (labels == w) & gen_mask
            w_forg_idx = (labels == w) & forg_mask

            w_gen_emb = embeddings[w_gen_idx]
            w_forg_emb = embeddings[w_forg_idx]

            n_gen = w_gen_emb.shape[0]
            n_forg = w_forg_emb.shape[0]

            if n_gen >= 1 and n_forg >= 1:
                # Orijinal-orijinal pozitif referans benzerliği
                if n_gen >= 2:
                    sim_gg = torch.matmul(w_gen_emb, w_gen_emb.T)
                    diag_mask = torch.eye(n_gen, dtype=torch.bool, device=device)
                    pos_sim = sim_gg[~diag_mask].mean()
                else:
                    pos_sim = torch.tensor(1.0, device=device)

                # Orijinal vs Sahte benzerlikleri: (n_gen, n_forg)
                sim_gf = torch.matmul(w_gen_emb, w_forg_emb.T)

                # sim(G, F) - sim(G, G) + margin <= 0 kuralı
                triplet_violations = F.relu(sim_gf - pos_sim + self.margin)
                
                # Doğrudan sahte benzerliğini baskılama
                push_down = F.relu(sim_gf - 0.20)

                loss_w = triplet_violations.mean() + 0.5 * push_down.mean()
                hard_neg_losses.append(loss_w)

        if hard_neg_losses:
            hard_neg_loss = torch.stack(hard_neg_losses).mean()
        else:
            hard_neg_loss = torch.tensor(0.0, device=device, requires_grad=True)

        total_loss = gen_loss + self.hard_neg_weight * hard_neg_loss
        return total_loss, gen_loss, hard_neg_loss


def compute_cosine_similarity(
    emb1: Union[torch.Tensor, np.ndarray],
    emb2: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    İki L2-normalize vektör arasındaki ham kosinüs benzerliğini [-1.0, 1.0] aralığında hesaplar.
    """
    if isinstance(emb1, np.ndarray):
        emb1 = torch.from_numpy(emb1).float()
    if isinstance(emb2, np.ndarray):
        emb2 = torch.from_numpy(emb2).float()
    if emb1.dim() == 1:
        emb1 = emb1.unsqueeze(0)
    if emb2.dim() == 1:
        emb2 = emb2.unsqueeze(0)
    emb1 = F.normalize(emb1, p=2, dim=-1)
    emb2 = F.normalize(emb2, p=2, dim=-1)
    cos_sim = float(torch.sum(emb1 * emb2, dim=-1).item())
    return max(-1.0, min(1.0, cos_sim))


def compute_normalized_similarity(
    emb1: Union[torch.Tensor, np.ndarray],
    emb2: Union[torch.Tensor, np.ndarray],
) -> float:
    """
    Sprint 17.5: İki embedding arasındaki normalize adli benzerlik skorunu kesin olarak [0.0, 1.0] aralığında hesaplar.
    Formül: (cos_sim + 1.0) / 2.0
    cos = -1.0 -> 0.0
    cos =  0.0 -> 0.5
    cos = +1.0 -> 1.0
    """
    cos_sim = compute_cosine_similarity(emb1, emb2)
    norm_sim = (cos_sim + 1.0) / 2.0
    return max(0.0, min(1.0, float(norm_sim)))


def compute_embedding_distance(
    emb1: torch.Tensor, emb2: torch.Tensor
) -> Tuple[float, float, float]:
    """
    İki L2-normalize embedding arasındaki kosinüs benzerliği, açısal mesafe ve [0.0, 1.0] benzerlik skorunu hesaplar.
    
    Returns:
        (cosine_similarity [-1.0, 1.0], angular_distance [0.0, 1.0], normalized_similarity [0.0, 1.0])
    """
    cos_sim = compute_cosine_similarity(emb1, emb2)
    theta = math.acos(cos_sim)
    angular_dist = float(theta / math.pi)
    norm_sim = compute_normalized_similarity(emb1, emb2)
    return round(cos_sim, 4), round(angular_dist, 4), round(norm_sim, 4)


class ForensicONNXRuntimeEncoder:
    """
    Yüksek Performanslı ve Düşük Gecikmeli ONNX Runtime Çıkarım Motoru (Forensic ONNX Runtime Engine).
    
    Özellikler:
    - ONNX Runtime C++ omurgası ile PyTorch CPU çıkarımını 2-4x hızlandırır.
    - ONNX modeli veya runtime bulunamadığında şeffaf şekilde PyTorch PyTorch modeline fallback yapar.
    - $L_2$-normalize 128 boyutlu embedding garantisi verir.
    """

    def __init__(
        self,
        onnx_model_path: Optional[Union[str, Path]] = None,
        fallback_pytorch_model: Optional[ForensicConvNeXtEncoder] = None,
        embedding_dim: int = 128,
    ):
        self.onnx_model_path = Path(onnx_model_path) if onnx_model_path else None
        self.embedding_dim = embedding_dim
        self.session = None
        self.fallback_model = fallback_pytorch_model or ForensicConvNeXtEncoder(embedding_dim=embedding_dim)
        self.fallback_model.eval()

        if self.onnx_model_path and self.onnx_model_path.exists():
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.session = ort.InferenceSession(str(self.onnx_model_path), sess_options=opts, providers=["CPUExecutionProvider"])
                self.input_name = self.session.get_inputs()[0].name
            except Exception:
                self.session = None

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Embedding çıkarımını ONNX veya PyTorch fallback ile gerçekleştirir."""
        if self.session is not None:
            try:
                if isinstance(x, torch.Tensor):
                    x_np = x.detach().cpu().numpy().astype(np.float32)
                else:
                    x_np = x.astype(np.float32)

                if x_np.ndim == 3:
                    x_np = np.expand_dims(x_np, axis=0)

                outputs = self.session.run(None, {self.input_name: x_np})
                emb_np = outputs[0]
                # L2 normalize
                norm = np.linalg.norm(emb_np, axis=-1, keepdims=True) + 1e-12
                emb_norm = emb_np / norm
                return torch.from_numpy(emb_norm).float()
            except Exception:
                pass

        # PyTorch Fallback
        if isinstance(x, np.ndarray):
            x_t = torch.from_numpy(x).float()
            if x_t.dim() == 3:
                x_t = x_t.unsqueeze(0)
        else:
            x_t = x

        with torch.no_grad():
            return self.fallback_model(x_t)

    @classmethod
    def export_to_onnx(
        cls,
        model: ForensicConvNeXtEncoder,
        output_path: Union[str, Path],
        in_channels: int = 3,
        image_size: int = 224,
    ) -> str:
        """PyTorch ForensicConvNeXtEncoder modelini optimize edilmiş ONNX formatına dönüştürür."""
        model.eval()
        dummy_input = torch.randn(1, in_channels, image_size, image_size)
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        torch.onnx.export(
            model,
            dummy_input,
            str(out_p),
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["input_image"],
            output_names=["embedding"],
            dynamic_axes={"input_image": {0: "batch_size"}, "embedding": {0: "batch_size"}},
        )
        return str(out_p)

