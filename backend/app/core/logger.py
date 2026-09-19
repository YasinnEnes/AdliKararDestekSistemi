import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from loguru import logger
except ImportError:
    # Henüz paketler yüklenmemişse geçici fallback
    import logging
    logger = logging.getLogger("FHDSS")

from backend.app.core.config import settings


def _format_console_record(record: dict) -> str:
    """Konsol için renkli ve okunabilir log formatı."""
    return (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>\n"
    )


def setup_logging() -> None:
    """
    Sistem ve Adli Bilişim Denetim (Audit) Loglama Altyapısını Başlatır.
    """
    # Log dizinini oluştur
    logs_dir = settings.LOGS_PATH
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not hasattr(logger, "remove"):
        return

    # Varsayılan Loguru handler'larını temizle
    logger.remove()

    # 1. Konsol Çıktısı
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format=_format_console_record,
        colorize=True,
        backtrace=True,
        diagnose=settings.DEBUG,
    )

    # 2. Genel Uygulama Log Dosyası (app.log)
    app_log_file = logs_dir / "app.log"
    logger.add(
        str(app_log_file),
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="20 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
    )

    # 3. Hata Log Dosyası (error.log)
    error_log_file = logs_dir / "error.log"
    logger.add(
        str(error_log_file),
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        rotation="10 MB",
        retention="60 days",
        compression="zip",
        encoding="utf-8",
        enqueue=True,
    )

    logger.info("FHDSS Logging altyapısı başarıyla başlatıldı.")


def log_audit_event(
    event_type: str,
    action: str,
    examiner_id: Optional[str] = "SYSTEM",
    document_hash: Optional[str] = None,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
    confidence_score: Optional[float] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Adli Delil Zinciri (Chain of Custody) ve Karar Destek Denetlenebilirlik Kaydı (Audit Record).
    
    Her analiz adımı, girdi belgesi hash'i, model çıktısı ve bilirkişi işlem kaydını 
    değiştirilemez biçimde JSONL formatında arşivler.
    """
    if not settings.AUDIT_LOG_ENABLED:
        return {}

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_data = {
        "timestamp_utc": timestamp,
        "event_type": event_type,
        "action": action,
        "examiner_id": examiner_id,
        "document_hash": document_hash,
        "model_name": model_name,
        "model_version": model_version,
        "confidence_score": confidence_score,
        "details": details or {},
    }

    # Kayıt bütünlüğü için SHA-256 imzası üret
    serialized = json.dumps(audit_data, sort_keys=True, default=str)
    record_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    audit_data["record_checksum_sha256"] = record_hash

    # Audit dosyasına append et
    audit_file = settings.AUDIT_LOG_PATH
    audit_file.parent.mkdir(parents=True, exist_ok=True)

    with open(audit_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_data, default=str) + "\n")

    if hasattr(logger, "bind"):
        logger.bind(audit=True).info(
            f"[AUDIT] Event={event_type} | Action={action} | DocHash={document_hash} | Checksum={record_hash[:8]}..."
        )

    return audit_data


# Modül yüklendiğinde logging'i yapılandır
setup_logging()
