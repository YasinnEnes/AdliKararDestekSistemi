from typing import Any, Dict, List, Optional, Union
from ml.dynamic.timeseries import (
    BiometricComparisonResult,
    BiometricFeatureProfile,
    BiometricPoint,
    BiometricTrajectory,
    DynamicTimeSeriesAnalyzer,
)


class BiometricService:
    """
    Adli Biyometrik İmza ve Dinamik Zaman Serisi Servis Katmanı.
    """

    def __init__(self):
        self.analyzer = DynamicTimeSeriesAnalyzer()

    def parse_points(
        self,
        raw_points: List[Dict[str, Any]],
        sample_id: str = "SAMPLE_Q",
        writer_id: str = "UNKNOWN",
    ) -> BiometricTrajectory:
        """JSON sözlük listesinden BiometricTrajectory nesnesi oluşturur."""
        points: List[BiometricPoint] = []
        for p in raw_points:
            points.append(BiometricPoint(
                x=float(p.get("x", 0.0)),
                y=float(p.get("y", 0.0)),
                t=float(p.get("t", 0.0)),
                pressure=float(p.get("pressure", 0.5)),
                azimuth_deg=float(p.get("azimuth_deg", 0.0)),
                altitude_deg=float(p.get("altitude_deg", 90.0)),
                is_pen_down=bool(p.get("is_pen_down", True)),
            ))
        return BiometricTrajectory(sample_id=sample_id, writer_id=writer_id, points=points)

    def extract_features(self, trajectory_or_points: Union[BiometricTrajectory, List[Dict[str, Any]]]) -> BiometricFeatureProfile:
        """Yörünge veya nokta listesinden biyomekanik öznitelik profilini çıkarır."""
        if isinstance(trajectory_or_points, list):
            traj = self.parse_points(trajectory_or_points)
        else:
            traj = trajectory_or_points
        return self.analyzer.extract_profile(traj)

    def compare_signatures(
        self,
        questioned_data: Union[BiometricTrajectory, List[Dict[str, Any]]],
        reference_data: Union[BiometricTrajectory, List[Dict[str, Any]]],
    ) -> BiometricComparisonResult:
        """Sorgulanan ve referans dinamik imza yörüngelerini karşılaştırır."""
        traj_q = questioned_data if isinstance(questioned_data, BiometricTrajectory) else self.parse_points(questioned_data, sample_id="Q")
        traj_r = reference_data if isinstance(reference_data, BiometricTrajectory) else self.parse_points(reference_data, sample_id="REF")
        return self.analyzer.compare(traj_q, traj_r)
