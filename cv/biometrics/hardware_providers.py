"""
FHDSS - Donanım Soyutlama Katmanı (Hardware Abstraction Layer - HAL)
Stylus, Tablet ve Tarayıcı Pointer Events imza girdi sağlayıcıları.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
import math


class ISignatureInputProvider(ABC):
    """Canlı imza ve biyometrik veri toplama arayüzü (Hardware Abstraction)."""

    @abstractmethod
    def capture_sample(self, raw_data: Any) -> List[Dict[str, Any]]:
        """Ham donanım verisini standart zaman serisi yörüngesine dönüştürür."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Donanımın desteklediği biyometrik yetenekleri (basınç, eğim, frekans vb.) döner."""
        pass

    @abstractmethod
    def validate_device(self) -> bool:
        """Cihazın adli standartlara uygunluğunu doğrular."""
        pass


class BrowserPointerProvider(ISignatureInputProvider):
    """
    HTML5 Pointer Events API sağlayıcısı.
    Mouse, Touch, Stylus ve Pen cihazlarından gelen standart tarayıcı olaylarını işler.
    """

    def __init__(self, device_metadata: Optional[Dict[str, Any]] = None):
        self.device_metadata = device_metadata or {}

    def capture_sample(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Gelen pointer event dizisini (x, y, pressure, timeStamp, pointerType, tiltX, tiltY)
        adli zaman serisi formatına dönüştürür.
        """
        normalized_points: List[Dict[str, Any]] = []

        if not raw_data:
            return normalized_points

        base_time = raw_data[0].get("timeStamp", 0)

        for p in raw_data:
            x = float(p.get("x", 0.0))
            y = float(p.get("y", 0.0))
            raw_p = p.get("pressure", None)

            # Basınç desteği kontrolü (0 veya None gelirse basınç desteklenmiyor olarak işaretlenir)
            pressure = float(raw_p) if raw_p is not None and raw_p > 0 else 0.5
            has_real_pressure = raw_p is not None and raw_p > 0

            time_delta_ms = float(p.get("timeStamp", 0) - base_time)
            time_sec = max(0.0, time_delta_ms / 1000.0)

            normalized_points.append({
                "x": round(x, 2),
                "y": round(y, 2),
                "time": round(time_sec, 4),
                "pressure": round(pressure, 3),
                "pressure_available": has_real_pressure,
                "pointer_type": p.get("pointerType", "mouse"),
                "tilt_x": p.get("tiltX", 0),
                "tilt_y": p.get("tiltY", 0),
            })

        return normalized_points

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider": "BrowserPointerProvider",
            "supports_pressure": True,
            "supports_tilt": True,
            "supports_touch": True,
            "supports_pen": True,
            "sampling_frequency_hz": "Variable (60-240Hz Browser Engine)",
        }

    def validate_device(self) -> bool:
        return True


class StylusProvider(ISignatureInputProvider):
    """Wacom / Surface / Apple Pencil gibi hassas basınçlı kalem cihazları sağlayıcısı."""

    def __init__(self, sample_rate_hz: int = 200, pressure_levels: int = 4096):
        self.sample_rate_hz = sample_rate_hz
        self.pressure_levels = pressure_levels

    def capture_sample(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        for p in raw_data:
            points.append({
                "x": float(p.get("x", 0.0)),
                "y": float(p.get("y", 0.0)),
                "time": float(p.get("time", 0.0)),
                "pressure": min(1.0, max(0.0, float(p.get("pressure", 0.5)))),
                "pressure_available": True,
                "pointer_type": "pen",
                "tilt_x": float(p.get("tilt_x", 0)),
                "tilt_y": float(p.get("tilt_y", 0)),
            })
        return points

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider": "StylusProvider",
            "supports_pressure": True,
            "pressure_levels": self.pressure_levels,
            "supports_tilt": True,
            "sampling_frequency_hz": f"{self.sample_rate_hz}Hz",
        }

    def validate_device(self) -> bool:
        return self.pressure_levels >= 1024


class MockSignatureProvider(ISignatureInputProvider):
    """Geliştirme ve birim testleri için sentetik biyometrik imza üreticisi."""

    def capture_sample(self, raw_data: Any = None) -> List[Dict[str, Any]]:
        points: List[Dict[str, Any]] = []
        num_points = 80
        total_time = 1.2
        for i in range(num_points):
            t = (i / num_points) * total_time
            x = 100.0 + i * 4.0 + math.sin(i * 0.3) * 15.0
            y = 150.0 + math.sin(i * 0.4) * 30.0 + math.cos(i * 0.2) * 10.0
            pressure = 0.3 + 0.5 * math.sin(i * 0.1) ** 2
            points.append({
                "x": round(x, 2),
                "y": round(y, 2),
                "time": round(t, 3),
                "pressure": round(pressure, 3),
                "pressure_available": True,
                "pointer_type": "mock_pen",
                "tilt_x": 0.0,
                "tilt_y": 0.0,
                "azimuth": 0.0,
            })
        return points

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider": "MockSignatureProvider",
            "supports_pressure": True,
            "supports_tilt": False,
            "supports_azimuth": False,
            "is_mock": True,
            "environment": "TEST_DEVELOPMENT_ONLY",
        }

    def validate_device(self) -> bool:
        return True


class WacomSignatureProvider(ISignatureInputProvider):
    """
    Wacom SDK / WILL (Wacom Ink Layer Language) Donanım Sağlayıcı Soyutlaması.
    Wacom STU ve Cintiq serisi profesyonel adli imza pedleri için arayüz.
    """

    def __init__(self, sample_rate_hz: int = 200, pressure_levels: int = 8192):
        self.sample_rate_hz = sample_rate_hz
        self.pressure_levels = pressure_levels

    def capture_sample(self, raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Native Wacom sürücüsü bağlı olmadığında PointerEvent normalize biçimini kabul eder
        points = []
        for p in (raw_data or []):
            points.append({
                "x": float(p.get("x", 0.0)),
                "y": float(p.get("y", 0.0)),
                "time": float(p.get("time", 0.0)),
                "pressure": min(1.0, max(0.0, float(p.get("pressure", 0.5)))),
                "pressure_available": True,
                "pointer_type": "wacom_pen",
                "tilt_x": float(p.get("tilt_x", 0.0)),
                "tilt_y": float(p.get("tilt_y", 0.0)),
                "azimuth": float(p.get("azimuth", 0.0)),
            })
        return points

    def get_capabilities(self) -> Dict[str, Any]:
        return {
            "provider": "WacomSignatureProvider",
            "supports_pressure": True,
            "pressure_levels": self.pressure_levels,
            "supports_tilt": True,
            "supports_azimuth": True,
            "sampling_frequency_hz": f"{self.sample_rate_hz}Hz",
            "native_sdk_required": "Wacom WILL / STU SDK",
            "status": "ABSTRACTION_READY",
        }

    def validate_device(self) -> bool:
        return self.pressure_levels >= 2048


# Geriye dönük uyumluluk ve Sprint 16 isimlendirme eşlemeleri
PointerEventSignatureProvider = BrowserPointerProvider
NativeStylusSignatureProvider = StylusProvider
