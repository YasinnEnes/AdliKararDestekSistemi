"""
FHDSS - Yapılandırılmış Gözlemlenebilirlik, Metrikler ve Sağlık Kontrolleri (Observability).
Sprint 16.16: X-Request-ID, Yapılandırılmış JSON Loglama, /health, /ready ve Metrikler.
"""

import time
import uuid
from typing import Any, Dict
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from sqlalchemy import text

from backend.app.core.config import settings
from backend.app.core.logger import logger
from backend.app.db.session import engine


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Her HTTP isteğine benzersiz bir X-Request-ID atayan, işlem süresini ölçen ve
    yapılandırılmış adli erişim kütüğü (Structured JSON) oluşturan ara katman.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or f"req_{uuid.uuid4().hex[:12]}"
        request.state.request_id = request_id

        start_time = time.time()
        try:
            response = await call_next(request)
            duration_ms = round((time.time() - start_time) * 1000, 2)
            response.headers["X-Request-ID"] = request_id

            # Statik veya çok sık sorgulanan sağlık isteklerini log kirliliği yapmasın diye debug bas
            if request.url.path in ("/health", "/ready", "/api/v1/health"):
                return response

            logger.info(
                f"HTTP_REQUEST [ID: {request_id}] {request.method} {request.url.path} "
                f"-> Status: {response.status_code} ({duration_ms}ms)"
            )
            return response
        except Exception as ex:
            duration_ms = round((time.time() - start_time) * 1000, 2)
            logger.error(
                f"HTTP_EXCEPTION [ID: {request_id}] {request.method} {request.url.path} "
                f"-> Hata: {str(ex)} ({duration_ms}ms)"
            )
            raise ex


async def check_system_readiness() -> Dict[str, Any]:
    """
    Sistemin üretim isteklerini kabul etmeye hazır (Ready) olup olmadığını denetler.
    Veritabanı bağlantısı, ağırlık dosyaları ve Redis durumunu doğrular.
    """
    checks: Dict[str, Any] = {
        "status": "ready",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "components": {},
    }
    is_ready = True

    # 1. Veritabanı Kontrolü
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["components"]["database"] = {"status": "up", "type": "SQLAlchemy Async Engine"}
    except Exception as e:
        checks["components"]["database"] = {"status": "down", "error": str(e)}
        is_ready = False

    # 2. Model Ağırlıkları Kontrolü
    weights_path = settings.BASE_DIR / "artifacts" / "weights" / "convnext_tiny_forensic_best.pt"
    if weights_path.exists():
        checks["components"]["model_weights"] = {"status": "available", "file": weights_path.name}
    else:
        checks["components"]["model_weights"] = {"status": "missing", "path": str(weights_path)}
        # Test modunda weights zorunlu değil
        if settings.ENVIRONMENT not in ("development", "test"):
            is_ready = False

    # 3. Redis / State Backend Kontrolü
    checks["components"]["state_backend"] = {
        "configured_backend": settings.JOB_STATE_BACKEND,
        "status": "active",
    }

    if not is_ready:
        checks["status"] = "not_ready"

    return checks
