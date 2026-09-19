"""
FHDSS - Yüksek Çözünürlüklü Adli Görüntüler için Yamalı Çıkarım Motoru (Tiled / Patch-based Inference).
Sprint 16.9: 600-1200 DPI adli taramalarda mikroskobik titreme ve vuruş detaylarını koruyan derin model çıkarımı.
"""

import math
from typing import Any, Callable, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from ml.models.siamese import ForensicConvNeXtEncoder


class ForensicTiledInference:
    """
    Yüksek çözünürlüklü adli belgeleri bellek taşması olmadan (OOM-safe)
    örten yamalar (overlapping tiles) halinde bölen, çıkarım yapan ve öznitelikleri toplayan motor.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tile_size: int = 512,
        overlap: float = 0.20,
        batch_size: int = 16,
        device: Optional[torch.device] = None,
        target_input_size: Tuple[int, int] = (224, 224),
    ):
        self.tile_size = tile_size
        self.overlap = max(0.0, min(0.5, overlap))
        self.batch_size = batch_size
        self.target_input_size = target_input_size

        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model
        if self.model is not None:
            self.model.to(self.device)
            self.model.eval()

        self._normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def generate_tiles(self, image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Görseli örtüşen yamalara (tiles) ayırır.
        Dönüş: [(tile_img, (y1, y2, x1, x2)), ...]
        """
        if image is None or image.size == 0:
            raise ValueError("Boş veya geçersiz görsel.")

        h, w = image.shape[:2]
        step = int(self.tile_size * (1.0 - self.overlap))
        step = max(1, step)

        # Eğer görsel zaten tile boyutundan küçük veya eşitse doğrudan tek parça dön
        if h <= self.tile_size and w <= self.tile_size:
            return [(image, (0, h, 0, w))]

        tiles = []
        y_starts = list(range(0, max(1, h - self.tile_size + 1), step))
        if y_starts[-1] + self.tile_size < h:
            y_starts.append(h - self.tile_size)

        x_starts = list(range(0, max(1, w - self.tile_size + 1), step))
        if x_starts[-1] + self.tile_size < w:
            x_starts.append(w - self.tile_size)

        for y in set(y_starts):
            y_end = min(h, y + self.tile_size)
            y_start = max(0, y_end - self.tile_size)
            for x in set(x_starts):
                x_end = min(w, x + self.tile_size)
                x_start = max(0, x_end - self.tile_size)
                tile = image[y_start:y_end, x_start:x_end]
                tiles.append((tile, (y_start, y_end, x_start, x_end)))

        return tiles

    def _prepare_tensor(self, img_bgr: np.ndarray) -> torch.Tensor:
        """NumPy BGR görseli ImageNet normalize 3x224x224 tensöre dönüştürür."""
        if img_bgr.ndim == 2:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        elif img_bgr.shape[2] == 4:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2RGB)
        else:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(rgb, self.target_input_size, interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        return self._normalize(t)

    @torch.no_grad()
    def extract_tiled_embedding(
        self,
        image: np.ndarray,
        global_weight: float = 0.50,
    ) -> np.ndarray:
        """
        Görselin hem global hem de yamalı (tiled) derin özniteliklerini çıkarıp L2-normalize global vektör döner.
        """
        if self.model is None:
            raise RuntimeError("ForensicTiledInference model nesnesi yüklenmemiş.")

        if image is None or image.size == 0:
            raise ValueError("Boş görsel ile çıkarım yapılamaz.")

        # 1. Global Embedding (Tüm görselin genel kompozisyonu)
        global_tensor = self._prepare_tensor(image).unsqueeze(0).to(self.device)
        global_emb = self.model(global_tensor).cpu().numpy().flatten()
        global_emb = global_emb / (np.linalg.norm(global_emb) + 1e-12)

        # 2. Tiled Patches Extraction
        tiles = self.generate_tiles(image)
        if len(tiles) <= 1:
            return global_emb

        tile_tensors = [self._prepare_tensor(t[0]) for t in tiles]
        patch_embeddings = []

        # Bellek taşmasını engellemek için batch işleme
        for i in range(0, len(tile_tensors), self.batch_size):
            batch = torch.stack(tile_tensors[i : i + self.batch_size]).to(self.device)
            emb = self.model(batch).cpu().numpy()
            patch_embeddings.append(emb)

        patch_embeddings = np.vstack(patch_embeddings)  # (N_tiles, D)

        # 3. Aggregation (Yamaların dikkat ve varyans ağırlıklı birleşimi)
        # Çok boş veya bembeyaz/saydam yamaların etkisini azaltmak için L2 norm ile normalize et
        patch_norms = np.linalg.norm(patch_embeddings, axis=1, keepdims=True) + 1e-12
        normed_patches = patch_embeddings / patch_norms
        aggregated_patches = np.mean(normed_patches, axis=0)
        aggregated_patches = aggregated_patches / (np.linalg.norm(aggregated_patches) + 1e-12)

        # 4. Global ve Tiled Füzyonu
        tile_weight = 1.0 - global_weight
        final_embedding = (global_weight * global_emb) + (tile_weight * aggregated_patches)
        final_embedding = final_embedding / (np.linalg.norm(final_embedding) + 1e-12)

        return final_embedding.astype(np.float32)
