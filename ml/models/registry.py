"""
FHDSS - Adli Model Sicili ve Bütünlük Doğrulayıcı (Model Registry & Manifest Verifier).
Sprint 16.10: Ağırlık dosyalarının SHA-256 kriptografik denetimi ve tahrifat tespiti.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend.app.core.config import settings
from backend.app.core.logger import logger


class ModelIntegrityError(Exception):
    """Model ağırlıklarında SHA-256 uyuşmazlığı veya tahrifat hatası."""
    pass


class ForensicModelRegistry:
    """
    Adli Model Manifestosu ve Bütünlük Denetim Yöneticisi.
    """

    def __init__(self, manifest_path: Optional[Path] = None):
        if manifest_path is None:
            self.manifest_path = settings.BASE_DIR / "artifacts" / "weights" / "model_manifest.json"
        else:
            self.manifest_path = Path(manifest_path)
        self._manifest_cache: Optional[Dict[str, Any]] = None

    def load_manifest(self) -> Dict[str, Any]:
        """Manifest dosyasını JSON olarak yükler."""
        if self._manifest_cache is not None:
            return self._manifest_cache

        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Model manifestosu bulunamadı: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self._manifest_cache = json.load(f)
        return self._manifest_cache

    def reload(self) -> Dict[str, Any]:
        """Manifest önbelleğini temizler ve yeniden yükler."""
        self._manifest_cache = None
        return self.load_manifest()

    def compute_file_sha256(self, file_path: Path) -> str:
        """Dosyanın SHA-256 kriptografik özetini hesaplar."""
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha.update(chunk)
        return sha.hexdigest()

    def verify_weights_integrity(
        self,
        weights_path: Path,
        raise_on_mismatch: bool = False,
    ) -> Tuple[bool, str, Optional[str]]:
        """
        Belirtilen ağırlık dosyasının manifestodaki SHA-256 ile uyuşup uyuşmadığını denetler.
        Dönüş: (is_valid: bool, actual_sha256: str, expected_sha256: Optional[str])
        """
        if not weights_path.exists():
            msg = f"Ağırlık dosyası diskte bulunamadı: {weights_path}"
            if raise_on_mismatch:
                raise FileNotFoundError(msg)
            return False, "", None

        actual_sha = self.compute_file_sha256(weights_path)
        filename = weights_path.name

        manifest = self.load_manifest()
        weights_dict = manifest.get("weights", {})

        expected_sha = None
        for key, entry in weights_dict.items():
            if entry.get("filename") == filename:
                expected_sha = entry.get("sha256")
                break

        if not expected_sha:
            logger.warning(f"Manifestoda '{filename}' için kayıt bulunamadı.")
            return True, actual_sha, None

        if actual_sha.lower() != expected_sha.lower():
            err_msg = (
                f"KRİTİK MODEL GÜVENLİK İHLALİ: '{filename}' için SHA-256 karma uyuşmazlığı! "
                f"Beklenen: {expected_sha}, Gerçek: {actual_sha}. Dosya tahrif edilmiş olabilir."
            )
            logger.error(err_msg)
            if raise_on_mismatch:
                raise ModelIntegrityError(err_msg)
            return False, actual_sha, expected_sha

        logger.info(f"Model doğrulaması başarılı: [{filename}] SHA-256 eşleşti.")
        if self.is_synthetic_checkpoint():
            logger.warning("UYARI: Model sicilindeki ağırlıklar SENTETİK TEST ağırlıklarıdır. Gerçek adli delil kararlarında kullanılmamalıdır.")
        return True, actual_sha, expected_sha

    def get_active_model_id(self) -> str:
        """Manifestoda birincil aktif olarak işaretlenen model kimliğini döner."""
        manifest = self.load_manifest()
        return manifest.get("active_primary_model", "convnext_tiny_forensic_cedar_real")

    def get_active_pt_path(self) -> Path:
        """Aktif birincil PyTorch model ağırlık dosyasının mutlak yolunu döner."""
        manifest = self.load_manifest()
        weights_dict = manifest.get("weights", {})
        pt_entry = weights_dict.get("pytorch_checkpoint", {})
        fn = pt_entry.get("filename", "convnext_tiny_forensic_cedar_real_best.pt")
        pt_path = self.manifest_path.parent / fn
        if not pt_path.exists():
            # Fallback: Eğer CEDAR ağırlığı henüz bulunamadıysa (test ortamında)
            legacy_pt = self.manifest_path.parent / "convnext_tiny_forensic_best.pt"
            if legacy_pt.exists():
                logger.warning(f"CEDAR modeli bulunamadı ({pt_path.name}), baseline modele başvuruluyor: {legacy_pt.name}")
                return legacy_pt
        return pt_path

    def get_active_onnx_path(self) -> Path:
        """Aktif birincil ONNX model dosyasının mutlak yolunu döner."""
        manifest = self.load_manifest()
        weights_dict = manifest.get("weights", {})
        onnx_entry = weights_dict.get("onnx_model", {})
        fn = onnx_entry.get("filename", "convnext_tiny_forensic_cedar_real.onnx")
        onnx_path = self.manifest_path.parent / fn
        if not onnx_path.exists():
            legacy_onnx = self.manifest_path.parent / "convnext_tiny_forensic.onnx"
            if legacy_onnx.exists():
                logger.warning(f"CEDAR ONNX bulunamadı ({onnx_path.name}), baseline ONNX modeline başvuruluyor: {legacy_onnx.name}")
                return legacy_onnx
        return onnx_path

    def get_active_model_metadata(self) -> Dict[str, Any]:
        """Aktif modele ait sürüm, karma ve adli metrik bilgilerini döner."""
        manifest = self.load_manifest()
        active_id = self.get_active_model_id()
        pt_path = self.get_active_pt_path()
        onnx_path = self.get_active_onnx_path()
        pt_entry = manifest.get("weights", {}).get("pytorch_checkpoint", {})
        
        pt_sha = pt_entry.get("sha256")
        if not pt_sha and pt_path.exists():
            pt_sha = self.compute_file_sha256(pt_path)

        return {
            "model_id": active_id,
            "manifest_version": manifest.get("manifest_version", "2.0.0"),
            "architecture": manifest.get("model_architecture", "ConvNeXt-Tiny"),
            "embedding_dim": manifest.get("embedding_dimension", 128),
            "training_type": manifest.get("training_type", "real_dataset"),
            "dataset_name": manifest.get("dataset_name", "CEDAR"),
            "pt_filename": pt_path.name,
            "onnx_filename": onnx_path.name,
            "model_sha256": pt_sha,
            "metrics": manifest.get("test_metrics", manifest.get("val_metrics", {})),
        }

    def get_model_training_type(self, filename: Optional[str] = None) -> str:
        """Modelin eğitim türünü döner: 'synthetic_test' veya 'real_dataset'."""
        manifest = self.load_manifest()
        if filename:
            weights_dict = manifest.get("weights", {})
            for key, entry in weights_dict.items():
                if isinstance(entry, dict) and entry.get("filename") == filename:
                    return entry.get("training_type", manifest.get("training_type", "synthetic_test"))
            for m in manifest.get("registered_models", []):
                if isinstance(m, dict) and (m.get("filename") == filename or m.get("pt_filename") == filename):
                    return m.get("training_type", "real_dataset")
            return manifest.get("training_type", "synthetic_test")

        if "synthetic_baseline" in manifest:
            return "synthetic_test"
        return manifest.get("training_type", "synthetic_test")

    def is_synthetic_checkpoint(self, filename: Optional[str] = None) -> bool:
        """Modelin sentetik/test modeli olup olmadığını kontrol eder."""
        return self.get_model_training_type(filename) == "synthetic_test"


# Singleton Registry
model_registry = ForensicModelRegistry()


