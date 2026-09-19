"""
FHDSS - Veritabanı Yapılandırması ve Oturumu (Sprint 14 Tekilleştirilmiş Mimari)
"""

from backend.app.db.base import Base
from backend.app.db.session import (
    DATABASE_URL,
    AsyncSessionLocal,
    engine,
    get_db,
)

# Geriye dönük kod ve test uyumluluğu için aliaslar
SessionLocal = AsyncSessionLocal
SQLALCHEMY_DATABASE_URL = DATABASE_URL
