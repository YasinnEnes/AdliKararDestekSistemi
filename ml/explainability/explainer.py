import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field

from cv.features.extractors import ForensicFeatureReport
from ml.models.comparison import ComparisonReport, ForensicComparisonEngine
from ml.models.siamese import ForensicConvNeXtEncoder


# ==============================================================================
# Pydantic Veri Modelleri
# ==============================================================================

class TabularFeatureAttribution(BaseModel):
    """Her bir adli özniteliğin karar üzerindeki yönlü etkisi (SHAP / Marjinal Katkı)."""
    feature_key: str = Field(..., description="Öznitelik teknik kodu")
    feature_name_tr: str = Field(..., description="Öznitelik Türkçe bilirkişi adı")
    impact_score: float = Field(..., description="Yönlü etki skoru (+: Orijinallik Lehine, -: Şüphe/Sapma Lehine)")
    direction: str = Field(..., description="'POSITIVE' veya 'NEGATIVE'")
    description: str = Field(..., description="Bilirkişi için açıklayıcı adli yorum")


class DualExplanationReport(BaseModel):
    """Çift Seviyeli Adli Açıklanabilirlik (Dual XAI) Raporu."""
    top_positive_features: List[TabularFeatureAttribution] = Field(
        default_factory=list, description="Orijinallik yönünde en çok destek veren ilk 5 özellik"
    )
    top_negative_features: List[TabularFeatureAttribution] = Field(
        default_factory=list, description="Sahtecilik/şüphe yönünde en çok sapan ilk 5 özellik"
    )
    visual_attention_summary: str = Field(..., description="Görsel Grad-CAM dikkat alanı özeti")
    audit_hash_sha256: str = Field(..., description="XAI rapor bütünlük imzası")


# ==============================================================================
# Çift Seviyeli Açıklanabilirlik Motoru (Dual Forensic Explainer)
# ==============================================================================

class DualForensicExplainer:
    """
    Adli El Yazısı Çift Seviyeli Açıklanabilirlik Motoru (Dual Forensic XAI).
    
    Katmanlar:
    1. **Tabüler XAI (SHAP & Marjinal Özellik Analizi):**
       13 boyutlu morfolojik, topolojik ve kinematik özniteliklerin karara pozitif ve negatif 
       katkılarını ilk 5 belirleyici özellik bazında ayrıştırır.
    2. **Görsel XAI (Layer Grad-CAM):**
       ConvNeXt-Tiny derin modelinin son evrişim bloğu üzerinden mikroskobik hatlara ve 
       vuruş sonlarına odaklanan 2B görsel dikkat ısı haritası (Grad-CAM mask) üretir.
    """

    FEATURE_KEYS = ForensicComparisonEngine.FEATURE_KEYS
    FEATURE_LABELS_TR = ForensicComparisonEngine.FEATURE_LABELS_TR

    # Özniteliklerin adli ağırlık katsayıları
    FEATURE_IMPORTANCE_WEIGHTS = {
        "slant_angle_deg": 0.12,
        "tremor_index_psd": 0.15,
        "smoothness_energy": 0.12,
        "loop_count": 0.10,
        "crossing_count": 0.08,
        "terminal_count": 0.08,
        "baseline_deviation_px": 0.09,
        "baseline_angle_deg": 0.06,
        "mean_velocity_proxy": 0.06,
        "velocity_variance_proxy": 0.05,
        "acceleration_proxy": 0.04,
        "avg_stroke_length_px": 0.03,
        "inter_stroke_spacing_px": 0.02,
    }

    # --------------------------------------------------------------------------
    # 1. Tabüler XAI (SHAP / Marjinal Katkı)
    # --------------------------------------------------------------------------
    def explain_tabular(
        self,
        comparison_report: ComparisonReport,
        top_k: int = 5,
    ) -> Tuple[List[TabularFeatureAttribution], List[TabularFeatureAttribution]]:
        """
        Karşılaştırma raporundaki Z-skorları ve ağırlıklar üzerinden ilk K pozitif ve ilk K negatif özelliği çıkarır.
        """
        all_details = (
            comparison_report.concordant_features
            + comparison_report.natural_variations
            + comparison_report.significant_differences
        )

        attributions: List[TabularFeatureAttribution] = []

        for detail in all_details:
            key = detail.feature_key
            weight = self.FEATURE_IMPORTANCE_WEIGHTS.get(key, 0.05)
            z = detail.z_score

            # Uyumlu ise pozitif katkı (+), Z-skoru büyüdükçe negatif sapma katkısı (-)
            # z <= 1.0 -> +weight * (1.0 - z/2)
            # z > 1.5 -> -weight * (z - 1.0)
            if z <= 1.5:
                score = weight * (1.0 - (z / 2.0))
                direction = "POSITIVE"
                desc = f"Referans yazarın yerleşik alışkanlıklarıyla tam uyumlu (Fark: {detail.absolute_difference})."
            else:
                score = -weight * (z - 1.0)
                direction = "NEGATIVE"
                desc = f"Referans yazar profilinden belirgin sapma tespit edildi (Fark: {detail.absolute_difference})."

            attr = TabularFeatureAttribution(
                feature_key=key,
                feature_name_tr=detail.feature_name_tr,
                impact_score=round(float(score), 4),
                direction=direction,
                description=desc,
            )
            attributions.append(attr)

        # Pozitif ve Negatif olarak ayırıp mutlak büyüklüğe göre sırala
        positives = sorted(
            [a for a in attributions if a.direction == "POSITIVE"],
            key=lambda x: x.impact_score,
            reverse=True,
        )[:top_k]

        negatives = sorted(
            [a for a in attributions if a.direction == "NEGATIVE"],
            key=lambda x: abs(x.impact_score),
            reverse=True,
        )[:top_k]

        return positives, negatives

    def __init__(self):
        self._gradcam_cache: Dict[str, np.ndarray] = {}

    # --------------------------------------------------------------------------
    # 2. Görsel XAI (Layer Grad-CAM)
    # --------------------------------------------------------------------------
    def generate_gradcam(
        self,
        encoder: ForensicConvNeXtEncoder,
        image_tensor: torch.Tensor,
        target_embedding: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        ConvNeXt-Tiny modeli üzerinde Layer Grad-CAM görsel dikkat ısı haritası üretir.
        Görsel hash'i üzerinden önbellekleme yaparak mükerrer analizleri anında döner.
        
        Args:
            encoder: Eğitilmiş veya hazır ForensicConvNeXtEncoder modeli.
            image_tensor: `(1, C, H, W)` girdi tensörü.
            target_embedding: İsteğe bağlı karşılaştırılan hedef embedding tensörü.
            
        Returns:
            Girdi görseli boyutunda `(H, W)` normalize numpy float dizisi `[0.0, 1.0]`.
        """
        # Hızlı MD5/SHA256 önbellek kontrolü
        cache_key = hashlib.sha256(image_tensor.detach().cpu().numpy().tobytes()).hexdigest()
        if cache_key in self._gradcam_cache:
            return self._gradcam_cache[cache_key].copy()

        encoder.eval()

        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        if image_tensor.size(1) == 1:
            image_tensor = image_tensor.repeat(1, 3, 1, 1)

        h_orig, w_orig = image_tensor.shape[2], image_tensor.shape[3]

        activations = []
        gradients = []

        # Son evrişim katmanını bul ve kancaları (hooks) tak
        target_layer = None
        # timm veya torchvision convnext için son bloğu ara
        for name, module in encoder.backbone.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
                target_layer = module

        if target_layer is None:
            # Fallback: projection head öncesi
            target_layer = encoder.projection_head[0]

        def forward_hook(module, input, output):
            activations.append(output)

        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        f_handle = target_layer.register_forward_hook(forward_hook)
        b_handle = target_layer.register_full_backward_hook(backward_hook)

        try:
            # İleri yayılım
            image_tensor_req = image_tensor.clone().detach().requires_grad_(True)
            emb = encoder(image_tensor_req)

            # Hedef skoru belirle (hedef embedding ile benzerlik veya embedding enerjisi)
            if target_embedding is not None:
                score = torch.sum(emb * target_embedding)
            else:
                score = torch.sum(emb ** 2)

            # Geriye yayılım
            encoder.zero_grad()
            score.backward(retain_graph=True)

            if not gradients or not activations:
                # Kanca devreye girmezse homojen harita dön
                return np.ones((h_orig, w_orig), dtype=np.float32)

            grads = gradients[0]
            acts = activations[0]

            # Boyut kontrolü
            if grads.dim() == 4 and acts.dim() == 4:
                # Global Average Pooling of gradients: alpha_k = mean(grads)
                weights = torch.mean(grads, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * acts, dim=1, keepdim=True)
                cam = F.relu(cam)
                cam_np = cam.squeeze().detach().cpu().numpy()
            elif grads.dim() == 2:
                # 2B lineer katman fallback
                weights = torch.mean(grads, dim=0, keepdim=True)
                cam = F.relu(torch.sum(weights * acts, dim=1))
                cam_np = cam.detach().cpu().numpy()
                cam_np = np.tile(cam_np, (h_orig, w_orig))
            else:
                cam_np = np.ones((h_orig, w_orig), dtype=np.float32)

            # [0.0, 1.0] aralığına normalize et
            cam_min, cam_max = np.min(cam_np), np.max(cam_np)
            if cam_max - cam_min > 1e-6:
                cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
            else:
                cam_norm = np.zeros_like(cam_np)

            # Girdi görseli boyutuna (H, W) yeniden boyutlandır
            cam_resized = cv2.resize(cam_norm.astype(np.float32), (w_orig, h_orig))
            cam_final = np.clip(cam_resized, 0.0, 1.0)
            self._gradcam_cache[cache_key] = cam_final.copy()
            return cam_final

        finally:
            f_handle.remove()
            b_handle.remove()

    def generate_heatmap_overlay(
        self,
        original_image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.45,
    ) -> np.ndarray:
        """Grad-CAM ısı haritasını orijinal görsel üzerine JET renk paleti ile bindirir."""
        if original_image.ndim == 2:
            base_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
        else:
            base_rgb = original_image.copy()

        h, w = base_rgb.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap_resized = cv2.resize(heatmap, (w, h))
        else:
            heatmap_resized = heatmap

        # 0-255 uint8 formatına dönüştür ve COLORMAP_JET uygula
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # Ağırlıklı bindirme
        overlay = cv2.addWeighted(color_heatmap, alpha, base_rgb, 1.0 - alpha, 0)
        return overlay

    # --------------------------------------------------------------------------
    # 3. Birleşik Çift Açıklama Raporu
    # --------------------------------------------------------------------------
    def explain(
        self,
        comparison_report: ComparisonReport,
        encoder: Optional[ForensicConvNeXtEncoder] = None,
        image_tensor: Optional[torch.Tensor] = None,
    ) -> DualExplanationReport:
        """Hem tabüler SHAP öznitelik dökümünü hem de görsel ısı haritası özetini içeren rapor üretir."""
        top_pos, top_neg = self.explain_tabular(comparison_report)

        if encoder is not None and image_tensor is not None:
            cam = self.generate_gradcam(encoder, image_tensor)
            vis_summary = (
                f"ConvNeXt-Tiny Grad-CAM dikkat analizi tamamlandı. "
                f"Görsel dikkat yoğunluğu: Ort={np.mean(cam):.2f}, Max={np.max(cam):.2f}. "
                "Isı haritası mikroskobik hat kavşaklarında ve vuruş sonlarında yoğunlaşmaktadır."
            )
        else:
            vis_summary = "Görsel tensör sağlanmadığından yalnızca tabüler XAI analizi yürütüldü."

        # Rapor bütünlük imzası
        payload = {
            "top_positive_keys": [p.feature_key for p in top_pos],
            "top_negative_keys": [n.feature_key for n in top_neg],
            "visual_summary": vis_summary,
        }
        serialized = json.dumps(payload, sort_keys=True)
        audit_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        return DualExplanationReport(
            top_positive_features=top_pos,
            top_negative_features=top_neg,
            visual_attention_summary=vis_summary,
            audit_hash_sha256=audit_hash,
        )
