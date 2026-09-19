"""
Forensic ONNX Inference Engine with Seamless PyTorch CPU Fallback.
Adli Belge İncelemesi için ConvNeXt-Tiny ONNX Runtime Çıkarım Motoru ve Hata Güvenceli PyTorch Fallback Mekanizması.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch

from backend.app.core.config import settings
from backend.app.core.logger import logger

from ml.models.registry import model_registry

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


class ForensicONNXInferenceEngine:
    """
    ConvNeXt derin metrik modeli için CPU üzerinde yüksek başarımlı ONNX Runtime çıkarım motoru.
    Model dosyası bulunamazsa, ONNX Runtime kurulu değilse veya herhangi bir runtime hatası oluşursa
    sistemin çalışmasını aksatmadan doğrudan PyTorch CPU çıkarımına geçer ve adli denetim izi bırakır.
    """

    def __init__(
        self,
        pytorch_model: Optional[torch.nn.Module] = None,
        onnx_model_path: Optional[Union[str, Path]] = None,
    ):
        self.pytorch_model = pytorch_model
        if self.pytorch_model is not None:
            self.pytorch_model.eval()

        if onnx_model_path:
            self.onnx_model_path = Path(onnx_model_path)
        else:
            try:
                self.onnx_model_path = model_registry.get_active_onnx_path()
            except Exception:
                self.onnx_model_path = Path(settings.BASE_DIR / "artifacts" / "weights" / "convnext_tiny_forensic_cedar_real.onnx")

        self.ort_session: Optional[Any] = None
        self.fallback_active: bool = False
        self.last_fallback_reason: Optional[str] = None
        self.input_name: str = "input"
        self.output_name: str = "embedding"

        self._initialize_onnx()

    def _initialize_onnx(self) -> None:
        """ONNX oturumunu başlatmaya çalışır; başarısız olursa PyTorch fallback modunu etkinleştirir."""
        if not HAS_ONNXRUNTIME:
            self.fallback_active = True
            self.last_fallback_reason = "onnxruntime kütüphanesi ortamda yüklü değil."
            logger.warning(f"ONNX Motoru: {self.last_fallback_reason} PyTorch CPU fallback devrede.")
            return

        if not self.onnx_model_path.exists():
            # Model dosyası henüz yoksa PyTorch modelinden otomatik dönüştürmeyi dene
            if self.pytorch_model is not None:
                exported = self.export_from_pytorch(self.pytorch_model, self.onnx_model_path)
                if not exported:
                    self.fallback_active = True
                    self.last_fallback_reason = f"ONNX model dosyası mevcut değil: {self.onnx_model_path}"
                    return
            else:
                self.fallback_active = True
                self.last_fallback_reason = f"ONNX model dosyası bulunamadı ve PyTorch referansı verilmedi: {self.onnx_model_path}"
                return

        try:
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 2
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.ort_session = ort.InferenceSession(
                str(self.onnx_model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )
            self.input_name = self.ort_session.get_inputs()[0].name
            self.output_name = self.ort_session.get_outputs()[0].name
            self.fallback_active = False
            logger.info(f"ConvNeXt ONNX CPU Çıkarım Motoru başlatıldı: {self.onnx_model_path.name}")
        except Exception as e:
            self.fallback_active = True
            self.last_fallback_reason = f"ONNX InferenceSession başlatılırken hata: {str(e)}"
            logger.warning(f"{self.last_fallback_reason}. PyTorch fallback devrede.")

    def export_from_pytorch(
        self,
        pytorch_model: torch.nn.Module,
        output_path: Path,
        input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
    ) -> bool:
        """PyTorch ConvNeXt modelini ONNX formatına ihraç eder."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_input = torch.randn(*input_shape, dtype=torch.float32)
            pytorch_model.eval()

            # Legacy stable exporter without dynamo unicode log issues
            try:
                torch.onnx.export(
                    pytorch_model,
                    dummy_input,
                    str(output_path),
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["embedding"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "embedding": {0: "batch_size"},
                    },
                    dynamo=False,
                )
            except TypeError:
                torch.onnx.export(
                    pytorch_model,
                    dummy_input,
                    str(output_path),
                    export_params=True,
                    opset_version=14,
                    do_constant_folding=True,
                    input_names=["input"],
                    output_names=["embedding"],
                    dynamic_axes={
                        "input": {0: "batch_size"},
                        "embedding": {0: "batch_size"},
                    },
                )
            logger.info(f"ConvNeXt modeli başarıyla ONNX formatına dönüştürüldü: {output_path}")
            return True
        except Exception as e:
            logger.warning(f"PyTorch -> ONNX dönüştürme başarısız oldu: {e}")
            return False

    def encode(self, input_tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Girdi tensörünü adli 128 boyutlu L2-normalize birim küre embedding vektörüne dönüştürür.
        ONNX hatası veya eksikliği durumunda otomatik olarak PyTorch fallback kullanır.
        
        Args:
            input_tensor: (B, 3, H, W) veya (3, H, W) formatında normalleştirilmiş tensör/dizi.
            
        Returns:
            (B, 128) boyutunda numpy float32 embedding dizisi.
        """
        # Standartlaştırılmış girdi formatı: (B, 3, 224, 224)
        if isinstance(input_tensor, torch.Tensor):
            t = input_tensor.detach().cpu()
            if t.ndim == 3:
                t = t.unsqueeze(0)
            if t.shape[-2:] != (224, 224):
                t = torch.nn.functional.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            np_input = t.numpy().astype(np.float32)
            torch_input = t
        else:
            arr = np.asarray(input_tensor, dtype=np.float32)
            if arr.ndim == 3:
                arr = np.expand_dims(arr, 0)
            # Resize if necessary
            if arr.shape[-2:] != (224, 224):
                t_tmp = torch.from_numpy(arr)
                t_tmp = torch.nn.functional.interpolate(t_tmp, size=(224, 224), mode="bilinear", align_corners=False)
                np_input = t_tmp.numpy().astype(np.float32)
                torch_input = t_tmp
            else:
                np_input = arr
                torch_input = torch.from_numpy(arr)

        # 1. ONNX Çıkarımı Denemesi
        if not self.fallback_active and self.ort_session is not None:
            try:
                ort_inputs = {self.input_name: np_input}
                ort_outs = self.ort_session.run([self.output_name], ort_inputs)
                raw_emb = ort_outs[0]
                # L2-normalizasyon kontrolü
                norms = np.linalg.norm(raw_emb, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                return raw_emb / norms
            except Exception as e:
                self.fallback_active = True
                self.last_fallback_reason = f"ONNX çıkarımı sırasında çalışma zamanı hatası: {str(e)}"
                logger.warning(f"{self.last_fallback_reason}. PyTorch fallback işletiliyor.")

        # 2. PyTorch CPU Fallback
        if self.pytorch_model is not None:
            with torch.inference_mode():
                out = self.pytorch_model(torch_input)
                if isinstance(out, torch.Tensor):
                    out_np = out.detach().cpu().numpy()
                    norms = np.linalg.norm(out_np, axis=1, keepdims=True)
                    norms = np.where(norms == 0, 1.0, norms)
                    return out_np / norms

        raise RuntimeError("Ne ONNX motoru ne de PyTorch fallback modeli çıkarım için kullanılamıyor.")

    def get_status(self) -> Dict[str, Any]:
        """Çıkarım motorunun mevcut çalışma modu ve adli durumunu raporlar."""
        return {
            "active_engine": "PYTORCH_FALLBACK" if self.fallback_active else "ONNX_RUNTIME_CPU",
            "onnx_available": HAS_ONNXRUNTIME,
            "model_path": str(self.onnx_model_path),
            "fallback_active": self.fallback_active,
            "fallback_reason": self.last_fallback_reason,
            "device": "CPU",
        }
