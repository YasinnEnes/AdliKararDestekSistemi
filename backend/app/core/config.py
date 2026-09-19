import json
import os
from pathlib import Path
from typing import List, Union, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Forensic Handwriting Decision Support System (FHDSS)
    Tip Güvenli Çekirdek Yapılandırma Sınıfı
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # --------------------------------------------------------------------------
    # Genel Uygulama Ayarları
    # --------------------------------------------------------------------------
    PROJECT_NAME: str = "Forensic Handwriting Decision Support System"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "development-super-secret-key-for-testing-fhdss-2026"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 1 gün
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7  # 7 gün refresh token süresi

    # --------------------------------------------------------------------------
    # Hız Sınırlaması (Rate Limiting) & Dağıtık Altyapı
    # --------------------------------------------------------------------------
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_LOGIN_MAX_ATTEMPTS: int = 5
    RATE_LIMIT_LOGIN_WINDOW_SEC: int = 60
    RATE_LIMIT_PIPELINE_MAX_REQUESTS: int = 10
    RATE_LIMIT_PIPELINE_WINDOW_SEC: int = 60
    REDIS_URL: str = "redis://localhost:6379/0"
    JOB_STATE_BACKEND: str = "memory"  # "memory" veya "redis"
    TASK_QUEUE_MAX_RETRIES: int = 3
    SIGNATURE_PROVIDER: str = "mock"   # "mock" veya "pkcs11"
    TIMESTAMP_PROVIDER: str = "mock"   # "mock" veya "rfc3161"

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: Optional[str], info) -> str:
        env = info.data.get("ENVIRONMENT", "development").lower()
        if env in ("production", "staging"):
            if not v or not v.strip() or v == "change-this-to-a-super-secret-key-in-production" or len(v) < 32:
                raise ValueError(
                    "Kritik Güvenlik Hatası: Production veya Staging ortamında SECRET_KEY boş, "
                    "varsayılan ('change-this-to-a-super-secret-key-in-production') veya 32 karakterden kısa olamaz!"
                )
        if not v or not v.strip():
            return "development-super-secret-key-for-testing-fhdss-2026"
        return v

    # --------------------------------------------------------------------------
    # Veritabanı Yapılandırması (PostgreSQL)
    # --------------------------------------------------------------------------
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "forensic_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    DATABASE_URL: Optional[str] = None

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_connection(cls, v: Optional[str], info) -> str:
        if isinstance(v, str) and v.strip():
            return v
        values = info.data
        user = values.get("POSTGRES_USER", "postgres")
        password = values.get("POSTGRES_PASSWORD", "postgres")
        server = values.get("POSTGRES_SERVER", "localhost")
        port = values.get("POSTGRES_PORT", 5432)
        db = values.get("POSTGRES_DB", "forensic_db")
        return f"postgresql+asyncpg://{user}:{password}@{server}:{port}/{db}"

    # --------------------------------------------------------------------------
    # Güvenlik & CORS
    # --------------------------------------------------------------------------
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            v_stripped = v.strip()
            if v_stripped.startswith("[") and v_stripped.endswith("]"):
                try:
                    return json.loads(v_stripped)
                except Exception:
                    pass
            return [origin.strip() for origin in v_stripped.split(",") if origin.strip()]
        elif isinstance(v, list):
            return v
        return ["http://localhost:3000", "http://127.0.0.1:3000"]

    # --------------------------------------------------------------------------
    # Dosya ve Dizin Yolları
    # --------------------------------------------------------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent.parent
    DATASET_ROOT_PATH: Path = BASE_DIR / "datasets"
    MODEL_STORAGE_PATH: Path = BASE_DIR / "artifacts" / "models"
    REPORTS_PATH: Path = BASE_DIR / "reports"
    LOGS_PATH: Path = BASE_DIR / "artifacts" / "logs"

    # --------------------------------------------------------------------------
    # Donanım & Çıkarım (Inference) Parametreleri
    # --------------------------------------------------------------------------
    DEVICE: str = "cuda"  # "cuda" veya "cpu"
    BATCH_SIZE: int = 32

    # --------------------------------------------------------------------------
    # Adli Denetim (Audit) & Logging
    # --------------------------------------------------------------------------
    AUDIT_LOG_ENABLED: bool = True
    AUDIT_LOG_PATH: Path = BASE_DIR / "artifacts" / "logs" / "audit.jsonl"
    LOG_LEVEL: str = "INFO"


settings = Settings()
