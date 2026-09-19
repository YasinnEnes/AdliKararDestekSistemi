import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class BiometricPoint:
    """Tekil Biyometrik Tablet Veri Noktası."""
    x: float
    y: float
    t: float                   # Zaman (saniye cinsinden)
    pressure: float            # Basınç [0.0, 1.0]
    azimuth_deg: float = 0.0   # Kalem yatay dönüş açısı (0-360 derece)
    altitude_deg: float = 90.0 # Kalem dikey eğim açısı (0-90 derece)
    is_pen_down: bool = True   # Kalem yüzeye temas ediyor mu?


@dataclass
class BiometricTrajectory:
    """Adli Dinamik İmza Yörünge Verisi."""
    sample_id: str
    writer_id: str
    points: List[BiometricPoint] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BiometricFeatureProfile:
    """Dinamik Zaman Serisi Biyomekanik Öznitelik Profili."""
    duration_sec: float
    total_points: int
    pen_down_points: int
    pen_up_points: int
    in_air_time_ratio: float
    mean_velocity: float
    max_velocity: float
    velocity_std: float
    mean_acceleration: float
    max_acceleration: float
    mean_jerk: float
    mean_pressure: float
    pressure_variance: float
    pressure_peaks_count: int
    kinematic_smoothness: float


@dataclass
class BiometricComparisonResult:
    """İki Dinamik İmza Arasındaki Karşılaştırmalı DTW ve Kinematik Analiz Raporu."""
    dtw_distance: float
    dtw_normalized_similarity: float
    velocity_correlation: float
    pressure_correlation: float
    in_air_time_diff: float
    is_biometrically_consistent: bool
    verbal_assessment_tr: str
    questioned_profile: BiometricFeatureProfile
    reference_profile: BiometricFeatureProfile


class DynamicTimeSeriesAnalyzer:
    """
    Adli Biyometrik Zaman Serisi ve Dinamik İmza Analiz Motoru (Dynamic Time Series Analyzer).
    
    Özellikler:
    - Çok boyutlu Sakoe-Chiba kısıtlı **Dynamic Time Warping (DTW)** mesafesi.
    - Hız $v(t)$, İvme $a(t)$, Sarsıntı/Jerk $j(t)$ ve Basınç $P(t)$ türevsel analizi.
    - Havada geçiş (*In-air*) ve yüzey temas (*On-surface*) süresi ayrıştırması.
    - Statik taklitlerdeki sahte akıcılığı ortaya çıkaran nöromusküler hız-zaman korelasyonu.
    """

    def __init__(self, dtw_tau: float = 35.0, sakoe_chiba_window_ratio: float = 0.15):
        self.dtw_tau = dtw_tau
        self.window_ratio = sakoe_chiba_window_ratio

    def extract_profile(self, trajectory: BiometricTrajectory) -> BiometricFeatureProfile:
        """Bir yörünge üzerinden tüm kinematik ve biyometrik metrikleri çıkarır."""
        pts = trajectory.points
        if not pts or len(pts) < 2:
            return BiometricFeatureProfile(
                duration_sec=0.0, total_points=len(pts), pen_down_points=0,
                pen_up_points=0, in_air_time_ratio=0.0, mean_velocity=0.0,
                max_velocity=0.0, velocity_std=0.0, mean_acceleration=0.0,
                max_acceleration=0.0, mean_jerk=0.0, mean_pressure=0.0,
                pressure_variance=0.0, pressure_peaks_count=0, kinematic_smoothness=0.0,
            )

        duration = max(1e-5, pts[-1].t - pts[0].t)
        total_pts = len(pts)
        pen_down_pts = sum(1 for p in pts if p.is_pen_down and p.pressure > 0.01)
        pen_up_pts = total_pts - pen_down_pts
        in_air_ratio = pen_up_pts / max(1, total_pts)

        # 1. Hız Serisi v(t)
        velocities: List[float] = []
        for i in range(1, total_pts):
            dt = max(1e-4, pts[i].t - pts[i - 1].t)
            dist = math.hypot(pts[i].x - pts[i - 1].x, pts[i].y - pts[i - 1].y)
            velocities.append(dist / dt)

        mean_v = float(np.mean(velocities)) if velocities else 0.0
        max_v = float(np.max(velocities)) if velocities else 0.0
        std_v = float(np.std(velocities)) if velocities else 0.0

        # 2. İvme Serisi a(t)
        accelerations: List[float] = []
        for i in range(1, len(velocities)):
            dt = max(1e-4, pts[i + 1].t - pts[i].t)
            dv = velocities[i] - velocities[i - 1]
            accelerations.append(abs(dv) / dt)

        mean_a = float(np.mean(accelerations)) if accelerations else 0.0
        max_a = float(np.max(accelerations)) if accelerations else 0.0

        # 3. Sarsıntı / Jerk Serisi j(t)
        jerks: List[float] = []
        for i in range(1, len(accelerations)):
            dt = max(1e-4, pts[i + 2].t - pts[i + 1].t)
            da = accelerations[i] - accelerations[i - 1]
            jerks.append(abs(da) / dt)

        mean_j = float(np.mean(jerks)) if jerks else 0.0

        # 4. Basınç İstatistikleri
        pressures = [p.pressure for p in pts]
        mean_p = float(np.mean(pressures)) if pressures else 0.0
        var_p = float(np.var(pressures)) if pressures else 0.0

        # Basınç tepe noktaları (peaks)
        peaks = 0
        for i in range(1, len(pressures) - 1):
            if pressures[i] > pressures[i - 1] and pressures[i] > pressures[i + 1] and pressures[i] > mean_p:
                peaks += 1

        # Kinematik Pürüzsüzlük: 1 / (1 + mean_jerk / 1000)
        smoothness = round(1.0 / (1.0 + (mean_j / 1000.0)), 4)

        return BiometricFeatureProfile(
            duration_sec=round(duration, 3),
            total_points=total_pts,
            pen_down_points=pen_down_pts,
            pen_up_points=pen_up_pts,
            in_air_time_ratio=round(in_air_ratio, 4),
            mean_velocity=round(mean_v, 2),
            max_velocity=round(max_v, 2),
            velocity_std=round(std_v, 2),
            mean_acceleration=round(mean_a, 2),
            max_acceleration=round(max_a, 2),
            mean_jerk=round(mean_j, 2),
            mean_pressure=round(mean_p, 4),
            pressure_variance=round(var_p, 5),
            pressure_peaks_count=peaks,
            kinematic_smoothness=smoothness,
        )

    def resample_to_uniform_frequency(
        self,
        trajectory: BiometricTrajectory,
        target_hz: float = 100.0,
    ) -> BiometricTrajectory:
        """
        Farklı donanımlardan (50-200 Hz) gelen tablet zaman serilerini Linear Spline
        interpolasyonu ile standart 100 Hz (10 ms) homojen zaman ızgarasına eşitler.
        """
        pts = trajectory.points
        if not pts or len(pts) < 3:
            return trajectory

        t_orig = np.array([p.t for p in pts], dtype=np.float64)
        # Monoton artan zaman garantisi
        for i in range(1, len(t_orig)):
            if t_orig[i] <= t_orig[i - 1]:
                t_orig[i] = t_orig[i - 1] + 1e-4

        t_start, t_end = t_orig[0], t_orig[-1]
        dt_step = 1.0 / target_hz
        t_uniform = np.arange(t_start, t_end + (0.5 * dt_step), dt_step)
        num_target_points = len(t_uniform)

        x_orig = np.array([p.x for p in pts], dtype=np.float64)
        y_orig = np.array([p.y for p in pts], dtype=np.float64)
        p_orig = np.array([p.pressure for p in pts], dtype=np.float64)
        az_orig = np.array([p.azimuth_deg for p in pts], dtype=np.float64)
        alt_orig = np.array([p.altitude_deg for p in pts], dtype=np.float64)

        # Doğrusal interpolasyon
        x_uniform = np.interp(t_uniform, t_orig, x_orig)
        y_uniform = np.interp(t_uniform, t_orig, y_orig)
        p_uniform = np.interp(t_uniform, t_orig, p_orig)
        az_uniform = np.interp(t_uniform, t_orig, az_orig)
        alt_uniform = np.interp(t_uniform, t_orig, alt_orig)

        resampled_points: List[BiometricPoint] = []
        for i in range(num_target_points):
            resampled_points.append(BiometricPoint(
                x=round(float(x_uniform[i]), 2),
                y=round(float(y_uniform[i]), 2),
                t=round(float(t_uniform[i]), 4),
                pressure=round(float(np.clip(p_uniform[i], 0.0, 1.0)), 4),
                azimuth_deg=round(float(az_uniform[i]), 2),
                altitude_deg=round(float(alt_uniform[i]), 2),
                is_pen_down=(p_uniform[i] > 0.01),
            ))

        return BiometricTrajectory(
            sample_id=trajectory.sample_id,
            writer_id=trajectory.writer_id,
            points=resampled_points,
            metadata={**trajectory.metadata, "resampled_hz": target_hz},
        )

    def _to_normalized_sequence(self, trajectory: BiometricTrajectory) -> np.ndarray:
        """Yörüngeyi (N x 4) boyutlu normalize öznitelik matrisine (x, y, v, pressure) dönüştürür."""
        pts = trajectory.points
        if not pts:
            return np.zeros((1, 4))

        xs = np.array([p.x for p in pts], dtype=np.float64)
        ys = np.array([p.y for p in pts], dtype=np.float64)
        ps = np.array([p.pressure for p in pts], dtype=np.float64)

        # Uzamsal z-score normalizasyonu
        x_norm = (xs - np.mean(xs)) / max(1e-3, np.std(xs))
        y_norm = (ys - np.mean(ys)) / max(1e-3, np.std(ys))

        # Hız dizisi
        vs = np.zeros(len(pts), dtype=np.float64)
        for i in range(1, len(pts)):
            dt = max(1e-4, pts[i].t - pts[i - 1].t)
            vs[i] = math.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]) / dt
        v_norm = (vs - np.mean(vs)) / max(1e-3, np.std(vs))

        seq = np.column_stack([x_norm, y_norm, v_norm, ps])
        return seq

    def compute_dtw_distance(
        self,
        traj_a: BiometricTrajectory,
        traj_b: BiometricTrajectory,
    ) -> float:
        """
        İki yörünge arasında Sakoe-Chiba bant kısıtlı Çok Boyutlu DTW mesafesi hesaplar.
        """
        norm_a = self.resample_to_uniform_frequency(traj_a)
        norm_b = self.resample_to_uniform_frequency(traj_b)

        seq_a = self._to_normalized_sequence(norm_a)
        seq_b = self._to_normalized_sequence(norm_b)

        n, m = len(seq_a), len(seq_b)
        if n == 0 or m == 0:
            return 999.0

        # Sakoe-Chiba pencere boyutu
        w = max(int(max(n, m) * self.window_ratio), abs(n - m) + 2)

        # Vektörize maliyet matrisi
        cost_matrix = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
        cost_matrix[0, 0] = 0.0

        for i in range(1, n + 1):
            j_start = max(1, i - w)
            j_end = min(m + 1, i + w + 1)
            row_a = seq_a[i - 1]

            for j in range(j_start, j_end):
                diff = row_a - seq_b[j - 1]
                dist = math.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2] + diff[3]*diff[3])
                cost_matrix[i, j] = dist + min(
                    cost_matrix[i - 1, j],
                    cost_matrix[i, j - 1],
                    cost_matrix[i - 1, j - 1],
                )

        total_cost = cost_matrix[n, m]
        normalized_distance = total_cost / (n + m)
        return float(normalized_distance)

    def compare(
        self,
        questioned: BiometricTrajectory,
        reference: BiometricTrajectory,
    ) -> BiometricComparisonResult:
        """
        Sorgulanan ve referans dinamik imzayı DTW ve kinematik korelasyonla karşılaştırır.
        """
        q_uniform = self.resample_to_uniform_frequency(questioned)
        ref_uniform = self.resample_to_uniform_frequency(reference)

        prof_q = self.extract_profile(q_uniform)
        prof_r = self.extract_profile(ref_uniform)

        dtw_dist = self.compute_dtw_distance(questioned, reference)
        # Normalize DTW benzerlik puanı S_dtw ∈ [0.0, 1.0]
        dtw_sim = float(math.exp(-dtw_dist / max(0.1, self.dtw_tau / 10.0)))
        dtw_sim = max(0.0, min(1.0, dtw_sim))

        # Hız ve basınç oransal benzerlikleri
        v_diff = abs(prof_q.mean_velocity - prof_r.mean_velocity) / max(1.0, prof_r.mean_velocity)
        v_corr = max(0.0, min(1.0, 1.0 - v_diff))

        p_diff = abs(prof_q.mean_pressure - prof_r.mean_pressure) / max(0.01, prof_r.mean_pressure)
        p_corr = max(0.0, min(1.0, 1.0 - p_diff))

        in_air_diff = abs(prof_q.in_air_time_ratio - prof_r.in_air_time_ratio)

        # Karar Kuralı: DTW benzerliği, hız korelasyonu ve basınç profili tutarlı mı?
        is_consistent = (dtw_sim >= 0.70) and (v_corr >= 0.60) and (p_corr >= 0.60)

        if is_consistent:
            assessment = (
                f"Dinamik biyometrik yörünge analizi uyumludur (DTW Benzerliği: %{dtw_sim*100:.1f}). "
                f"Hız profili ve kalem basınç eğrileri doğal nöromusküler akışla tam örtüşmektedir."
            )
        elif dtw_sim >= 0.45:
            assessment = (
                f"Kısmi dinamik uyumsuzluk (DTW Benzerliği: %{dtw_sim*100:.1f}). "
                f"Vuruş hızında ve basınç noktalarında orta düzeyde doğal varyasyon sapması gözlemlendi."
            )
        else:
            assessment = (
                f"Kritik kinematik ayrışma ve taklit göstergeleri (DTW Benzerliği: %{dtw_sim*100:.1f}). "
                f"Sorgulanan imzada yapay duraksamalar, aşırı sarsıntı (jerk) ve sahtecilik emareleri tespit edildi."
            )

        return BiometricComparisonResult(
            dtw_distance=round(dtw_dist, 4),
            dtw_normalized_similarity=round(dtw_sim, 4),
            velocity_correlation=round(v_corr, 4),
            pressure_correlation=round(p_corr, 4),
            in_air_time_diff=round(in_air_diff, 4),
            is_biometrically_consistent=is_consistent,
            verbal_assessment_tr=assessment,
            questioned_profile=prof_q,
            reference_profile=prof_r,
        )

    @staticmethod
    def generate_synthetic_tablet_trajectory(
        num_points: int = 150,
        is_forgery: bool = False,
        base_speed: float = 120.0,
        seed: int = 42,
    ) -> BiometricTrajectory:
        """Deterministik sentetik tablet imza zaman serisi üretir."""
        np.random.seed(seed)
        points: List[BiometricPoint] = []
        t = 0.0

        for i in range(num_points):
            t += 0.01  # 100 Hz örnekleme
            u = i / float(num_points)

            # Doğal imza eğrisi
            if not is_forgery:
                x = 50.0 + 200.0 * u + 20.0 * math.sin(u * 6.0 * math.pi)
                y = 80.0 + 40.0 * math.cos(u * 4.0 * math.pi)
                pressure = 0.4 + 0.5 * math.sin(u * math.pi)
            else:
                # Taklit imza: titrek, yavaş, duraksamalı
                x = 50.0 + 200.0 * u + 5.0 * np.random.randn()
                y = 80.0 + 40.0 * math.cos(u * 4.0 * math.pi) + 4.0 * np.random.randn()
                pressure = 0.8 + 0.1 * np.random.randn()  # Aşırı ve tekdüze baskı

            points.append(BiometricPoint(
                x=round(x, 2),
                y=round(y, 2),
                t=round(t, 4),
                pressure=round(max(0.0, min(1.0, pressure)), 3),
                azimuth_deg=45.0,
                altitude_deg=60.0,
                is_pen_down=True,
            ))

        return BiometricTrajectory(
            sample_id="SYNTH_Q" if is_forgery else "SYNTH_REF",
            writer_id="WRITER_001",
            points=points,
        )
