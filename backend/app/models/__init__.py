from backend.app.models.entities import (
    AnalysisResultEntity,
    AuditLogEntity,
    CaseEntity,
    SampleEntity,
    UserEntity,
)

# Geriye dönük uyumluluk
ForensicCase = CaseEntity

__all__ = [
    "CaseEntity",
    "SampleEntity",
    "AnalysisResultEntity",
    "AuditLogEntity",
    "UserEntity",
    "ForensicCase",
]
