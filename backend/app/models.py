"""
FHDSS - SQLAlchemy Veritabanı Modelleri (Sprint 14 Tekilleştirilmiş Mimari)
"""

from backend.app.models.entities import (
    AnalysisResultEntity,
    AuditLogEntity,
    CaseEntity,
    SampleEntity,
    UserEntity,
)

# Geriye dönük tam uyumluluk: ForensicCase -> CaseEntity
ForensicCase = CaseEntity

__all__ = [
    "ForensicCase",
    "CaseEntity",
    "SampleEntity",
    "AnalysisResultEntity",
    "AuditLogEntity",
    "UserEntity",
]
