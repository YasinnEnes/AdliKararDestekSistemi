from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.v1.api import api_router
from backend.app.core.config import settings
from backend.app.core.logger import log_audit_event, logger, setup_logging
from backend.app.database import Base, engine
from backend.app.db.init_db import init_db
import backend.app.models  # Modelleri Base'e kaydetmek için


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Uygulama yaşam döngüsü başlatma ve kapatma olayları."""
    setup_logging()
    logger.info(f"{settings.PROJECT_NAME} v1.0.0 başlatılıyor... (Ortam: {settings.ENVIRONMENT})")
    
    # Sprint 14: Asenkron tekil veritabanı tablolarını (fhdss.db / PostgreSQL) hazırla
    await init_db()

    # Sprint 16.6: Kesintiye uğrayan arka plan görevlerini kurtar
    from backend.app.services.task_queue import task_queue
    await task_queue.recover_interrupted_jobs()

    log_audit_event(
        event_type="SYSTEM_STARTUP",
        action="START_APPLICATION_SERVER",
        details={"environment": settings.ENVIRONMENT, "debug": settings.DEBUG},
    )
    yield
    logger.info(f"{settings.PROJECT_NAME} kapatılıyor.")
    log_audit_event(
        event_type="SYSTEM_SHUTDOWN",
        action="STOP_APPLICATION_SERVER",
    )


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI Tabanlı, Modüler, Açıklanabilir ve Denetlenebilir Adli El Yazısı Karar Destek Sistemi API",
    version="1.0.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Yapılandırması
if settings.BACKEND_CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Sprint 16.16: Yapılandırılmış İstek Loglama ve X-Request-ID Ara Katmanı
from backend.app.core.observability import StructuredLoggingMiddleware, check_system_readiness
app.add_middleware(StructuredLoggingMiddleware)

# Sprint 20: Global Hata Yakalayıcı (Merkezi Exception Handler)
from fastapi import Request, status, HTTPException
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

@app.exception_handler(Exception)
async def global_unhandled_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, (HTTPException, StarletteHTTPException)):
        raise exc
    request_id = getattr(request.state, "request_id", "req-unknown")
    logger.error(
        f"[GlobalExceptionHandler] Unhandled Exception [Request: {request_id}] {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": {
                "message": "Sunucu tarafında beklenmeyen bir işlem hatası meydana geldi.",
                "error_code": "INTERNAL_SERVER_ERROR",
                "request_id": request_id,
            }
        },
    )

# Ana Router Entegrasyonu
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/health", tags=["Sistem Durumu"])
@app.get(f"{settings.API_V1_STR}/health", tags=["Sistem Durumu"])
def health_check():
    """Sistem ve API servis sağlık (Liveness) kontrolü."""
    return {
        "status": "healthy",
        "project_name": settings.PROJECT_NAME,
        "environment": settings.ENVIRONMENT,
        "version": "1.0.0",
        "audit_enabled": settings.AUDIT_LOG_ENABLED,
    }


@app.get("/ready", tags=["Sistem Durumu"])
@app.get(f"{settings.API_V1_STR}/ready", tags=["Sistem Durumu"])
async def readiness_check():
    """Sistem hazır olma (Readiness: DB, Ağırlıklar, Durum Deposu) kontrolü."""
    return await check_system_readiness()
