"""
Çoklu Numune Analizi ve Embedding Kümeleme Servisi.
Forensic Multi-Sample Embedding Clustering, PCA 2D Projection, and Intra-Writer Variability Modeling.
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.decomposition import PCA

from backend.app.core.logger import logger


class MultiSampleClusteringService:
    """
    Çoklu referans numuneler ile sorgulanan imzanın derin embedding uzayındaki dağılımını,
    yazar içi doğal varyasyonunu (intra-writer variability) ve tolerans elipsini hesaplar.
    """

    def analyze_clusters(
        self,
        questioned_sample_id: str,
        questioned_embedding: np.ndarray,
        reference_samples: List[Dict[str, Any]],  # List of {"sample_id": ..., "embedding": np.ndarray}
    ) -> Dict[str, Any]:
        """
        Sorgulanan ve referans embedding vektörlerini PCA ile 2 boyuta indirger,
        referansların tolerans elipsini ve yazar içi tutarlılık metriklerini çıkarır.
        """
        ref_count = len(reference_samples)
        if ref_count == 0:
            raise ValueError("Kümeleme analizi için en az bir adet referans numune gereklidir.")

        # Questioned embedding: (128,)
        q_emb = np.asarray(questioned_embedding, dtype=np.float32).flatten()
        ref_embs = [np.asarray(s["embedding"], dtype=np.float32).flatten() for s in reference_samples]
        ref_matrix = np.stack(ref_embs, axis=0)  # (N, 128)

        # 1. Yazar İçi Varyasyon (Intra-Writer Variability) İstatistikleri
        ref_centroid_128 = np.mean(ref_matrix, axis=0)
        ref_distances_to_centroid = [float(np.linalg.norm(r - ref_centroid_128)) for r in ref_matrix]
        mean_intra_dist = float(np.mean(ref_distances_to_centroid))
        max_intra_dist = float(np.max(ref_distances_to_centroid)) if ref_distances_to_centroid else 0.0
        std_intra_dist = float(np.std(ref_distances_to_centroid)) if len(ref_distances_to_centroid) > 1 else 0.0

        q_distance_to_centroid = float(np.linalg.norm(q_emb - ref_centroid_128))

        # 2. PCA 2D İndirgeme
        all_embeddings = np.vstack([ref_matrix, q_emb.reshape(1, -1)])  # (N+1, 128)
        
        # En az 2 bileşen için PCA uygula
        n_samples = all_embeddings.shape[0]
        n_components = 2 if n_samples >= 2 else 1
        pca = PCA(n_components=n_components, random_state=42)
        projected = pca.fit_transform(all_embeddings)  # (N+1, 2)

        if n_components == 1:
            # 1 boyut varsa 2. boyutu 0 yap
            projected = np.column_stack([projected, np.zeros(n_samples)])

        ref_coords_2d = projected[:-1]  # (N, 2)
        q_coord_2d = projected[-1]      # (2,)

        # 2D Centroid ve Kovaryans Elipsi Hesabı
        center_2d = np.mean(ref_coords_2d, axis=0)
        center_x = float(center_2d[0])
        center_y = float(center_2d[1])

        if ref_count >= 3:
            cov = np.cov(ref_coords_2d, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # Sırala (büyükten küçüğe)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

            # 95% güven aralığı için chi-square faktörü (2 serbestlik derecesi: ~5.991, karekökü ~2.447)
            chi_scale = 2.447
            radius_x = float(chi_scale * np.sqrt(max(eigenvalues[0], 1e-4)))
            radius_y = float(chi_scale * np.sqrt(max(eigenvalues[1], 1e-4)))
            angle_rad = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            angle_deg = float(np.degrees(angle_rad))
        else:
            # 1 veya 2 numune için varsayılan tolerans dairesi
            radius_x = float(max(max_intra_dist, 0.25) * 1.5)
            radius_y = radius_x * 0.8
            angle_deg = 0.0

        # Sorgulanan numunenin elips içinde olup olmadığını test et
        dx = q_coord_2d[0] - center_x
        dy = q_coord_2d[1] - center_y
        cos_a = math.cos(math.radians(angle_deg))
        sin_a = math.sin(math.radians(angle_deg))
        x_rot = cos_a * dx + sin_a * dy
        y_rot = -sin_a * dx + cos_a * dy
        normalized_distance = ((x_rot / max(radius_x, 1e-5)) ** 2) + ((y_rot / max(radius_y, 1e-5)) ** 2)
        is_inside_tolerance = bool(normalized_distance <= 1.0)

        # Tutarlılık Derecesi (Consistency Rating)
        if mean_intra_dist < 0.20:
            consistency_rating = "YÜKSEK (Düşük Yazar İçi Değişkenlik / Kararlı İmza)"
        elif mean_intra_dist < 0.45:
            consistency_rating = "ORTA (Doğal Yazar İçi Varyasyon)"
        else:
            consistency_rating = "GENİŞ (Yüksek Yazar İçi Değişkenlik / Değişken İmza)"

        references_result = []
        for i, ref in enumerate(reference_samples):
            references_result.append({
                "sample_id": ref["sample_id"],
                "pca_x": round(float(ref_coords_2d[i][0]), 4),
                "pca_y": round(float(ref_coords_2d[i][1]), 4),
                "distance_to_centroid": round(ref_distances_to_centroid[i], 4),
            })

        return {
            "questioned": {
                "sample_id": questioned_sample_id,
                "pca_x": round(float(q_coord_2d[0]), 4),
                "pca_y": round(float(q_coord_2d[1]), 4),
                "distance_to_centroid": round(q_distance_to_centroid, 4),
                "normalized_ellipse_distance": round(float(normalized_distance), 4),
                "is_inside_tolerance_ellipse": is_inside_tolerance,
                "forensic_status": "DOĞAL_VARYASYON_İÇİNDE" if is_inside_tolerance else "VARYASYON_DIŞINDA_ANOMALİ",
            },
            "references": references_result,
            "variability_ellipse": {
                "center_x": round(center_x, 4),
                "center_y": round(center_y, 4),
                "radius_x": round(radius_x, 4),
                "radius_y": round(radius_y, 4),
                "angle_deg": round(angle_deg, 2),
                "confidence_level": 0.95,
            },
            "intra_writer_variability": {
                "mean_distance": round(mean_intra_dist, 4),
                "max_distance": round(max_intra_dist, 4),
                "std_dev": round(std_intra_dist, 4),
                "consistency_rating": consistency_rating,
                "reference_count": ref_count,
            },
            "pca_explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        }


multi_sample_clustering_service = MultiSampleClusteringService()
