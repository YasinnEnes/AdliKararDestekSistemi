"""
FHDSS - Dinamik Biyometrik İmza Öznitelik Çıkarıcı (Dynamic Kinematic Features)
Tablet / Stylus zaman serisi yörüngelerinden adli biyometrik öznitelikleri çıkarır.
"""

from typing import Any, Dict, List
import math
import numpy as np
from pydantic import BaseModel, Field


class DynamicSignatureFeatures(BaseModel):
    """Canlı imza verisinden türetilen dinamik kinematik öznitelikler modeli."""
    total_duration_sec: float = Field(..., description="Toplam imza atım süresi (saniye)")
    stroke_count: int = Field(default=1, description="Toplam vuruş / kalem darbesi sayısı")
    stroke_length_px: float = Field(..., description="Toplam yörünge piksel uzunluğu")
    mean_velocity: float = Field(..., description="Ortalama imza hızı (piksel/saniye)")
    max_velocity: float = Field(..., description="Maksimum pik anlık hızı")
    velocity_std: float = Field(..., description="Hız dalgalanması standart sapması")
    mean_pressure: float = Field(..., description="Ortalama kalem basıncı (0.0 - 1.0)")
    pressure_variance: float = Field(..., description="Basınç varyansı")
    pressure_peaks_count: int = Field(..., description="Basınç zirvesi sayısı")
    pen_down_ratio: float = Field(..., description="Kalemin kağıtla temas süresi oranı")
    pen_up_ratio: float = Field(..., description="Kalemin havada kaldığı süre oranı (In-air)")
    turning_points_count: int = Field(..., description="Ani yön değişimi sayısı")
    kinematic_smoothness: float = Field(..., description="Kinematik akıcılık ve harmonik pürüzsüzlük")
    pressure_available: bool = Field(default=True, description="Basınç sensörünün aktif olup olmadığı")


def extract_dynamic_features(points: List[Dict[str, Any]]) -> DynamicSignatureFeatures:
    """
    Ham zaman serisi noktalarından (x, y, time, pressure, is_pen_down)
    adli dinamik biyometrik öznitelikleri hesaplar.
    """
    if len(points) < 2:
        return DynamicSignatureFeatures(
            total_duration_sec=0.0,
            stroke_count=1,
            stroke_length_px=0.0,
            mean_velocity=0.0,
            max_velocity=0.0,
            velocity_std=0.0,
            mean_pressure=0.5,
            pressure_variance=0.0,
            pressure_peaks_count=0,
            pen_down_ratio=1.0,
            pen_up_ratio=0.0,
            turning_points_count=0,
            kinematic_smoothness=1.0,
            pressure_available=False,
        )

    times = [p.get("time", 0.0) for p in points]
    xs = [p.get("x", 0.0) for p in points]
    ys = [p.get("y", 0.0) for p in points]
    pressures = [p.get("pressure", 0.5) for p in points]
    has_real_pressure = any(p.get("pressure_available", False) for p in points) or (len(set(pressures)) > 3)

    total_duration = max(0.01, times[-1] - times[0])

    # Hız ve mesafe hesaplamaları
    velocities: List[float] = []
    total_length = 0.0
    turning_points = 0
    prev_angle = None

    for i in range(1, len(points)):
        dt = max(0.001, times[i] - times[i - 1])
        dx = xs[i] - xs[i - 1]
        dy = ys[i] - ys[i - 1]
        dist = math.hypot(dx, dy)
        total_length += dist
        v = dist / dt
        velocities.append(v)

        # Açısal yön değişimi
        current_angle = math.atan2(dy, dx)
        if prev_angle is not None:
            angle_diff = abs(current_angle - prev_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            if angle_diff > math.radians(60):
                turning_points += 1
        prev_angle = current_angle

    mean_v = float(np.mean(velocities)) if velocities else 0.0
    max_v = float(np.max(velocities)) if velocities else 0.0
    std_v = float(np.std(velocities)) if velocities else 0.0

    mean_p = float(np.mean(pressures)) if pressures else 0.5
    var_p = float(np.var(pressures)) if pressures else 0.0

    # Basınç tepe noktaları tespiti
    pressure_peaks = 0
    if len(pressures) > 4:
        for i in range(1, len(pressures) - 1):
            if pressures[i] > pressures[i - 1] and pressures[i] > pressures[i + 1] and pressures[i] > mean_p:
                pressure_peaks += 1

    # Kinematik akıcılık (Jerk türevi metriği)
    accelerations = []
    for i in range(1, len(velocities)):
        dt = max(0.001, times[i + 1] - times[i])
        a = (velocities[i] - velocities[i - 1]) / dt
        accelerations.append(abs(a))
    mean_accel = float(np.mean(accelerations)) if accelerations else 100.0
    smoothness = float(1.0 / (1.0 + mean_accel / 1000.0))

    # Kalem darbesi (stroke) ayrımı
    stroke_count = 1
    pen_down_time = 0.0
    pen_up_time = 0.0

    for i in range(len(points)):
        p = points[i]
        is_down = p.get("is_pen_down", True)
        dt = (times[i] - times[i - 1]) if i > 0 else 0.005
        if is_down:
            pen_down_time += dt
        else:
            pen_up_time += dt
            if i > 0 and points[i - 1].get("is_pen_down", True):
                stroke_count += 1

    total_time_sum = max(0.01, pen_down_time + pen_up_time)
    pen_down_ratio = min(1.0, max(0.0, pen_down_time / total_time_sum))
    pen_up_ratio = min(1.0, max(0.0, pen_up_time / total_time_sum))

    return DynamicSignatureFeatures(
        total_duration_sec=round(total_duration, 3),
        stroke_count=max(1, stroke_count),
        stroke_length_px=round(total_length, 2),
        mean_velocity=round(mean_v, 2),
        max_velocity=round(max_v, 2),
        velocity_std=round(std_v, 2),
        mean_pressure=round(mean_p, 3),
        pressure_variance=round(var_p, 4),
        pressure_peaks_count=pressure_peaks,
        pen_down_ratio=round(pen_down_ratio, 3),
        pen_up_ratio=round(pen_up_ratio, 3),
        turning_points_count=turning_points,
        kinematic_smoothness=round(smoothness, 3),
        pressure_available=has_real_pressure,
    )
