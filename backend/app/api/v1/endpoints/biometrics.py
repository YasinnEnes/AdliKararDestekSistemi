from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from backend.app.schemas.api_contracts import (
    BiometricCompareRequest,
    BiometricCompareResponse,
    BiometricPointPayload,
)
from backend.app.services.biometric_service import BiometricService
from cv.biometrics.dynamic_features import DynamicSignatureFeatures
from ml.dynamic.timeseries import DynamicTimeSeriesAnalyzer

router = APIRouter()
_BIOMETRIC_SERVICE = BiometricService()


@router.post(
    "/compare",
    response_model=BiometricCompareResponse,
    summary="Dinamik Biyometrik İmza Karşılaştırma (DTW & Kinematik)",
)
async def compare_biometric_signatures(payload: BiometricCompareRequest):
    """
    Tablet/Wacom cihazlarından gelen iki zaman serisi yörüngesini DTW ve kinematik profillerle karşılaştırır.
    """
    if len(payload.questioned_points) < 2 or len(payload.reference_points) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Biyometrik karşılaştırma için en az 2 nokta içeren yörünge gereklidir.",
        )

    q_points = [p.model_dump() for p in payload.questioned_points]
    ref_points = [p.model_dump() for p in payload.reference_points]

    result = _BIOMETRIC_SERVICE.compare_signatures(q_points, ref_points)

    return BiometricCompareResponse(
        dtw_distance=result.dtw_distance,
        dtw_normalized_similarity=result.dtw_normalized_similarity,
        velocity_correlation=result.velocity_correlation,
        pressure_correlation=result.pressure_correlation,
        in_air_time_diff=result.in_air_time_diff,
        is_biometrically_consistent=result.is_biometrically_consistent,
        verbal_assessment_tr=result.verbal_assessment_tr,
        questioned_profile=result.questioned_profile.__dict__,
        reference_profile=result.reference_profile.__dict__,
    )


@router.get(
    "/synthetic-demo",
    response_model=BiometricCompareResponse,
    summary="Demo Biyometrik Karşılaştırma Üret",
)
async def get_synthetic_biometric_demo():
    """Test ve demonstrasyon amaçlı sentetik biyometrik imza karşılaştırması döner."""
    traj_ref = DynamicTimeSeriesAnalyzer.generate_synthetic_tablet_trajectory(num_points=120, is_forgery=False)
    traj_q = DynamicTimeSeriesAnalyzer.generate_synthetic_tablet_trajectory(num_points=120, is_forgery=True)

    analyzer = DynamicTimeSeriesAnalyzer()
    res = analyzer.compare(traj_q, traj_ref)

    return BiometricCompareResponse(
        dtw_distance=res.dtw_distance,
        dtw_normalized_similarity=res.dtw_normalized_similarity,
        velocity_correlation=res.velocity_correlation,
        pressure_correlation=res.pressure_correlation,
        in_air_time_diff=res.in_air_time_diff,
        is_biometrically_consistent=res.is_biometrically_consistent,
        verbal_assessment_tr=res.verbal_assessment_tr,
        questioned_profile=res.questioned_profile.__dict__,
        reference_profile=res.reference_profile.__dict__,
    )


class LiveSignatureCaptureRequest(BaseModel):
    points: List[Dict[str, Any]]
    provider: str = "browser"  # "browser", "stylus", "mock"
    device_name: Optional[str] = "Standard Pointer / Stylus"


class LiveSignatureCaptureResponse(BaseModel):
    point_count: int
    normalized_points: List[Dict[str, Any]]
    features: DynamicSignatureFeatures
    provider_capabilities: Dict[str, Any]


@router.post(
    "/capture-live",
    response_model=LiveSignatureCaptureResponse,
    summary="Canlı İmza Pedi Verisini Yakala ve Kinematik Öznitelikleri Çıkar",
)
async def capture_live_signature(payload: LiveSignatureCaptureRequest):
    """
    HTML5 Pointer Events / Stylus üzerinden canlı yakalanan imza koordinatlarını işler,
    donanım soyutlaması üzerinden filtreler ve adli dinamik öznitelikleri üretir.
    """
    from cv.biometrics.hardware_providers import (
        BrowserPointerProvider,
        StylusProvider,
        MockSignatureProvider,
    )
    from cv.biometrics.dynamic_features import extract_dynamic_features

    provider_name = payload.provider.lower()
    if provider_name == "mock":
        provider = MockSignatureProvider()
    elif provider_name == "stylus":
        provider = StylusProvider()
    else:
        provider = BrowserPointerProvider(device_metadata={"name": payload.device_name})

    normalized = provider.capture_sample(payload.points)
    features = extract_dynamic_features(normalized)

    return LiveSignatureCaptureResponse(
        point_count=len(normalized),
        normalized_points=normalized,
        features=features,
        provider_capabilities=provider.get_capabilities(),
    )
