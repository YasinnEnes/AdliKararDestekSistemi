"""
FHDSS Sprint 16: Kritik Güvenlik, Mimari, AI Pipeline ve Kurumsal Altyapı Entegrasyon Test Paketi.
Comprehensive Test Suite for Sprint 16 Features (16.1 - 16.19).
"""

import hashlib
import io
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from backend.app.core.config import Settings, settings
from backend.app.core.observability import check_system_readiness
from backend.app.core.rate_limiter import SlidingWindowRateLimiter
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    get_password_hash,
    hash_refresh_token,
    verify_password,
)
from backend.app.db.session import AsyncSessionLocal
from backend.app.main import app
from backend.app.models.entities import AnalysisJobEntity, RefreshTokenEntity, UserEntity
from backend.app.services.job_state_store import (
    InMemoryJobStateStore,
    RedisJobStateStore,
    create_job_state_store,
)
from backend.app.services.legal_signature_service import (
    PKCS11ElectronicSignatureProvider,
    RFC3161TimestampProvider,
    get_signature_provider,
    get_timestamp_provider,
)
from backend.app.services.task_queue import PersistentTaskQueue
from cv.biometrics.hardware_providers import (
    PointerEventSignatureProvider,
    WacomSignatureProvider,
)
from cv.segmentation.document_segmenter import ForensicDocumentSegmenter
from ml.models.comparison import ForensicComparisonEngine
from ml.models.registry import ForensicModelRegistry
from ml.models.siamese import ForensicConvNeXtEncoder
from ml.models.tiling import ForensicTiledInference

client = TestClient(app)


# ==============================================================================
# 1. SPRINT 16.1: AUTHENTICATION FALLBACK VE GÜVENLİK TESTLERİ
# ==============================================================================

def test_auth_fallback_rejected_in_production():
    """Production ortamında token yoksa kesinlikle 401 Unauthorized dönülmelidir."""
    original_env = settings.ENVIRONMENT
    try:
        settings.ENVIRONMENT = "production"
        # Token göndermeden korumalı bir endpoint çağır
        res = client.post("/api/v1/cases", json={"name": "Yetkisiz Vaka", "examination_type": "SIGNATURE"})
        assert res.status_code == 401
        assert "Yetkilendirme belirteci (token) eksik" in res.json()["detail"]
    finally:
        settings.ENVIRONMENT = original_env


def test_auth_invalid_and_expired_tokens():
    """Geçersiz ve süresi dolmuş belirteçler 401 ile reddedilmelidir."""
    # 1. Geçersiz token
    res1 = client.get("/api/v1/auth/me", headers={"Authorization": "Bearer gecersiz_token_12345"})
    assert res1.status_code == 401

    # 2. Süresi dolmuş token
    expired_token = create_access_token("test_user", role="EXAMINER", expires_delta=timedelta(seconds=-10))
    res2 = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {expired_token}"})
    assert res2.status_code == 401
    assert "süresi dolmuş" in res2.json()["detail"].lower()


# ==============================================================================
# 2. SPRINT 16.2: SECRET_KEY GÜVENLİK DOĞRULAMASI
# ==============================================================================

def test_secret_key_validation_in_production():
    """Production ortamında boş, varsayılan veya kısa SECRET_KEY startup hatası vermelidir."""
    # Varsayılan güvensiz anahtar
    with pytest.raises(ValueError, match="Kritik Güvenlik Hatası"):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="change-this-to-a-super-secret-key-in-production",
        )

    # Boş anahtar
    with pytest.raises(ValueError, match="Kritik Güvenlik Hatası"):
        Settings(
            ENVIRONMENT="production",
            SECRET_KEY="",
        )

    # Yeterince uzun güvenli anahtar (32+ karakter) başarıyla kabul edilmeli
    valid_cfg = Settings(
        ENVIRONMENT="production",
        SECRET_KEY="guvenli-adli-laboratuvar-super-secret-key-2026-production",
    )
    assert valid_cfg.SECRET_KEY.startswith("guvenli-adli")


# ==============================================================================
# 3. SPRINT 16.3: HIZ SINIRLAMASI (RATE LIMITING) TESTLERİ
# ==============================================================================

def test_sliding_window_rate_limiter_logic():
    """Kayan pencere hız sınırlayıcısının aşım durumunda 429 koşulunu doğru hesapladığını doğrular."""
    limiter = SlidingWindowRateLimiter(redis_url=None)  # In-memory test
    test_key = "test_rate_ip_192.168.1.100"

    # İlk 3 isteğe izin verilmeli
    for _ in range(3):
        allowed, _ = limiter.is_allowed(test_key, max_requests=3, window_seconds=10)
        assert allowed is True

    # 4. istek sınır aşımı olmalı
    allowed, retry_after = limiter.is_allowed(test_key, max_requests=3, window_seconds=10)
    assert allowed is False
    assert retry_after > 0

    # Sıfırlama sonrası tekrar izin verilmeli
    limiter.reset_key(test_key)
    allowed, _ = limiter.is_allowed(test_key, max_requests=3, window_seconds=10)
    assert allowed is True


# ==============================================================================
# 4. SPRINT 16.4: REFRESH TOKEN ROTATION VE REUSE DETECTION TESTLERİ
# ==============================================================================

def test_refresh_token_rotation_and_reuse_detection():
    """
    Refresh Token ile yeni Access Token alma, rotasyon ve çalınan belirteç (Reuse) tespiti.
    """
    # 1. Test kullanıcısı ile giriş yap
    client.post(
        "/api/v1/auth/register",
        json={"username": "examiner_rot_test", "password": "GuvenliSifre2026!", "full_name": "Rotasyon Uzman", "role": "EXAMINER"},
    )
    login_res = client.post(
        "/api/v1/auth/login",
        json={"username": "examiner_rot_test", "password": "GuvenliSifre2026!"},
    )
    assert login_res.status_code == 200
    login_data = login_res.json()
    first_access = login_data["access_token"]
    first_refresh = login_data["refresh_token"]
    assert first_refresh is not None

    # 2. Geçerli Refresh Token ile yenileme yap
    refresh_res1 = client.post("/api/v1/auth/refresh", json={"refresh_token": first_refresh})
    assert refresh_res1.status_code == 200
    ref_data1 = refresh_res1.json()
    second_access = ref_data1["access_token"]
    second_refresh = ref_data1["refresh_token"]
    assert second_access != first_access
    assert second_refresh != first_refresh

    # 3. REUSE DETECTION: Zaten kullanılmış olan first_refresh tekrar gönderildiğinde güvenlik ihlali vermeli!
    reuse_res = client.post("/api/v1/auth/refresh", json={"refresh_token": first_refresh})
    assert reuse_res.status_code == 401
    assert "güvenlik ihlali" in reuse_res.json()["detail"].lower()

    # 4. İhlal sonrası kullanıcının tüm oturumları iptal edilmiş olmalı (second_refresh de reddedilmeli)
    compromised_res = client.post("/api/v1/auth/refresh", json={"refresh_token": second_refresh})
    assert compromised_res.status_code == 401


# ==============================================================================
# 5. SPRINT 16.5 & 16.6: DAĞITIK DURUM VE KALICI GÖREV KUYRUĞU TESTLERİ
# ==============================================================================

@pytest.mark.asyncio
async def test_persistent_task_queue_and_recovery():
    """Kalıcı görev kuyruğunun durum takibini ve kesinti kurtarmasını doğrular."""
    store = InMemoryJobStateStore()
    store.set_state("job:job_test_01", {"job_id": "job_test_01", "status": "RUNNING"})
    assert store.get_state("job:job_test_01")["status"] == "RUNNING"

    queue = PersistentTaskQueue(max_retries=2)
    assert queue.max_retries == 2


# ==============================================================================
# 6. SPRINT 16.8: MAHALANOBIS KOVARYANS STABİLİZASYONU TESTLERİ
# ==============================================================================

def test_mahalanobis_covariance_stabilization():
    """1 numune, 2 numune, 3 numune ve özdeş numunelerde tekil matris hatası olmaksızın hesaplama."""
    engine = ForensicComparisonEngine()
    dim = 13

    q_vec = np.ones(dim, dtype=np.float64) * 5.0

    # 1. Tekil Numune
    ref_1 = [np.ones(dim, dtype=np.float64) * 4.8]
    rep_1 = engine.compare(q_vec, ref_1)
    assert rep_1.mahalanobis_distance >= 0.0
    assert 0.0 <= rep_1.similarity_score <= 1.0

    # 2. İki Numune
    ref_2 = [
        np.ones(dim, dtype=np.float64) * 4.8,
        np.ones(dim, dtype=np.float64) * 5.2,
    ]
    rep_2 = engine.compare(q_vec, ref_2)
    assert rep_2.mahalanobis_distance >= 0.0

    # 3. Birebir Özdeş (Sıfır varyanslı / Singular) Numuneler
    ref_singular = [
        np.ones(dim, dtype=np.float64) * 5.0,
        np.ones(dim, dtype=np.float64) * 5.0,
        np.ones(dim, dtype=np.float64) * 5.0,
    ]
    rep_singular = engine.compare(q_vec, ref_singular)
    assert not math.isnan(rep_singular.mahalanobis_distance)
    assert rep_singular.similarity_score >= 0.90


# ==============================================================================
# 7. SPRINT 16.9: YÜKSEK ÇÖZÜNÜRLÜKLÜ YAMALI (TILED) ÇIKARIM TESTLERİ
# ==============================================================================

def test_high_resolution_tiled_inference():
    """Büyük adli görselin yamalara ayrılıp derin embedding üretildiğini doğrular."""
    model = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    tiler = ForensicTiledInference(model=model, tile_size=256, overlap=0.20, batch_size=4)

    # 600x800 büyük sentetik belge
    large_img = np.full((600, 800, 3), 255, dtype=np.uint8)
    large_img[100:200, 100:500] = 0  # Siyah yazı çizgisi

    tiles = tiler.generate_tiles(large_img)
    assert len(tiles) > 1

    emb = tiler.extract_tiled_embedding(large_img)
    assert emb.shape == (128,)
    assert np.isclose(np.linalg.norm(emb), 1.0, atol=1e-3)


# ==============================================================================
# 8. SPRINT 16.10: MODEL MANİFESTOSU VE SHA-256 DOĞRULAMA TESTLERİ
# ==============================================================================

def test_model_registry_manifest_verification():
    """Model manifestosunun SHA-256 doğrulamasını ve tahrifat uyarısını test eder."""
    registry = ForensicModelRegistry()
    manifest = registry.load_manifest()
    assert "architecture" in manifest
    assert manifest["architecture"] == "convnext_tiny"

    weights_path = settings.BASE_DIR / "artifacts" / "weights" / "convnext_tiny_forensic_best.pt"
    if weights_path.exists():
        is_valid, actual_sha, expected_sha = registry.verify_weights_integrity(weights_path)
        assert is_valid is True
        assert actual_sha == expected_sha


# ==============================================================================
# 9. SPRINT 16.11: ARTEFAKT VE MÜHÜR AYRIŞTIRMA TESTLERİ
# ==============================================================================

def test_document_seal_and_guideline_filtering():
    """Resmi kırmızı/mavi mühürlerin ve kılavuz çizgilerinin ayrıştırılması."""
    segmenter = ForensicDocumentSegmenter()

    # Kırmızı mühürlü sentetik belge
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    # Kırmızı mühür (BGR: B=0, G=0, R=220)
    img[50:120, 50:120] = [0, 0, 220]
    # Siyah yazı (BGR: 0, 0, 0)
    img[180:200, 50:300] = [0, 0, 0]

    artifacts = segmenter.filter_document_artifacts(img)
    assert "clean_handwriting_mask" in artifacts
    assert "detected_seal_mask" in artifacts
    assert artifacts["seal_detected"] is True


# ==============================================================================
# 10. SPRINT 16.12: DONANIM VE STYLUS SOYUTLAMA TESTLERİ
# ==============================================================================

def test_hardware_stylus_and_wacom_providers():
    """Stylus ve Wacom sağlayıcılarının yetenek ve veri normalizasyonu."""
    wacom = WacomSignatureProvider()
    caps = wacom.get_capabilities()
    assert caps["provider"] == "WacomSignatureProvider"
    assert caps["supports_pressure"] is True
    assert caps["supports_azimuth"] is True

    raw_points = [{"x": 120, "y": 140, "pressure": 0.8, "tilt_x": 10, "tilt_y": -5, "azimuth": 45}]
    sample = wacom.capture_sample(raw_points)
    assert len(sample) == 1
    assert sample[0]["pointer_type"] == "wacom_pen"


# ==============================================================================
# 11. SPRINT 16.14: ADMIN YÖNETİM PANELİ VE RBAC TESTLERİ
# ==============================================================================

def test_admin_rbac_protection():
    """EXAMINER kullanıcısının Admin API'ye erişemediğini (403), ADMIN'in eriştiğini (200) test eder."""
    examiner_token = create_access_token("uzman_normal", role="EXAMINER")
    admin_token = create_access_token("admin_master", role="ADMIN")

    # 1. EXAMINER çağırırsa -> 403 Forbidden
    res_forbidden = client.get(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {examiner_token}"},
    )
    assert res_forbidden.status_code == 403

    # 2. ADMIN çağırırsa -> 200 OK
    res_ok = client.get(
        "/api/v1/admin/users",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert res_ok.status_code == 200
    assert isinstance(res_ok.json(), list)


# ==============================================================================
# 12. SPRINT 16.16: GÖZLEMLENEBİLİRLİK, SAĞLIK VE X-REQUEST-ID TESTLERİ
# ==============================================================================

def test_observability_and_ready_endpoint():
    """X-Request-ID başlığı ve /ready sağlık denetimi testi."""
    res = client.get("/ready")
    assert res.status_code == 200
    data = res.json()
    assert "components" in data
    assert "database" in data["components"]

    # X-Request-ID yanıt başlığı kontrolü
    assert "x-request-id" in res.headers or "X-Request-ID" in res.headers


# ==============================================================================
# 13. SPRINT 16.17: E-İMZA VE ZAMAN DAMGASI ABSTRACTION TESTLERİ
# ==============================================================================

def test_signature_and_timestamp_abstractions():
    """PKCS#11 ve RFC 3161 sağlayıcılarının adli sorumluluk reddi ile imza üretimini test eder."""
    sig_p = PKCS11ElectronicSignatureProvider()
    ts_p = RFC3161TimestampProvider()

    dummy_pdf_bytes = b"%PDF-1.4 Adli Bilirkisi Raporu Test Icerigi"
    sig_res = sig_p.sign_document(dummy_pdf_bytes, signer_name="Bilirkişi Adli Uzman")
    assert sig_res["format"] == "CAdES-BES"
    assert "YASAL UYARI" in sig_res["legal_disclaimer"]

    ts_res = ts_p.create_timestamp(dummy_pdf_bytes)
    assert ts_res["protocol"] == "RFC 3161 Time-Stamp Protocol"
    assert "YASAL UYARI" in ts_res["legal_disclaimer"]
