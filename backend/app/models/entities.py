import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship, synonym

from backend.app.db.base import Base


class CaseEntity(Base):
    """Adli Vaka / İnceleme Dosyası Veritabanı Varlığı."""
    __tablename__ = "cases"

    case_id = Column(String(64), primary_key=True, index=True)
    name = Column(String(255), nullable=True, default="Adli Vaka", index=True)
    description = Column(Text, nullable=True)
    examination_type = Column(String(64), nullable=False, default="SIGNATURE")
    reference_number = Column(String(128), nullable=True)
    authority = Column(String(255), nullable=True, default="Cumhuriyet Başsavcılığı / Mahkeme")
    examiner_id = Column(String(128), nullable=False, default="BILIRKISI_UZMAN_01")
    notes = Column(Text, nullable=True, default="")
    status = Column(String(64), nullable=False, default="IN_PROGRESS")
    current_stage = Column(String(64), nullable=False, default="CASE_CREATED")
    state = Column(String(64), nullable=False, default="PHASE_A_INIT")
    questioned_sample_id = Column(String(128), nullable=True)
    suspect_name = Column(String(128), nullable=True)
    similarity_score = Column(Float, nullable=True)
    verbal_scale = Column(String(128), nullable=True)
    audit_hash = Column(String(64), nullable=True)
    merkle_root_hash = Column(String(64), nullable=True)
    created_at_utc = Column(String(64), nullable=False)
    updated_at_utc = Column(String(64), nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at_utc = Column(String(64), nullable=True)
    deleted_by = Column(String(64), nullable=True)

    # İlişkiler
    samples = relationship("SampleEntity", back_populates="case", cascade="all, delete-orphan")
    analysis_results = relationship("AnalysisResultEntity", back_populates="case", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLogEntity", back_populates="case", cascade="all, delete-orphan")
    jobs = relationship("AnalysisJobEntity", back_populates="case", cascade="all, delete-orphan")


class UserEntity(Base):
    """Adli Bilirkişi ve Kullanıcı Varlığı (RBAC & Oturum)."""
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(64), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(128), nullable=True)
    role = Column(String(32), default="EXAMINER", nullable=False)  # EXAMINER, SENIOR_REVIEWER, ADMIN
    is_active = Column(Boolean, default=True, nullable=False)
    created_at_utc = Column(String(64), nullable=False)


class SampleEntity(Base):
    """Adli Belge / Numune Görseli Veritabanı Varlığı."""
    __tablename__ = "samples"

    sample_id = Column(String(128), primary_key=True, index=True)
    case_id = Column(String(64), ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    sample_type = Column(String(32), nullable=False)  # "questioned" veya "reference"
    sha256_hash = Column(String(64), nullable=False)
    file_size_bytes = Column(Integer, nullable=False, default=0)
    file_path = Column(String(512), nullable=True)
    created_at_utc = Column(String(64), nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False, index=True)
    deleted_at_utc = Column(String(64), nullable=True)
    deleted_by = Column(String(64), nullable=True)

    case = relationship("CaseEntity", back_populates="samples")


class AnalysisResultEntity(Base):
    """Uçtan Uca Adli Karar Destek Analiz Sonucu Veritabanı Varlığı."""
    __tablename__ = "analysis_results"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(64), ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    suitability_json = Column(JSON, nullable=True)
    complexity_json = Column(JSON, nullable=True)
    comparison_json = Column(JSON, nullable=True)
    disguise_json = Column(JSON, nullable=True)
    fusion_json = Column(JSON, nullable=True)
    xai_json = Column(JSON, nullable=True)
    hypothesis_json = Column(JSON, nullable=True)
    audit_checksum_sha256 = Column(String(64), nullable=False)
    merkle_root_hash = Column(String(64), nullable=True)
    html_report_path = Column(String(512), nullable=True)
    pdf_report_path = Column(String(512), nullable=True)
    created_at_utc = Column(String(64), nullable=False)

    case = relationship("CaseEntity", back_populates="analysis_results")


class AuditLogEntity(Base):
    """Adli Kriptografik Denetim Günlüğü Varlığı."""
    __tablename__ = "audit_logs"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(64), ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=True, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    action = Column(String(128), nullable=False)
    examiner_id = Column(String(128), nullable=True)
    sha256_hash = Column(String(64), nullable=False)
    previous_hash = Column(String(64), nullable=True)  # Hash-chain bütünlük bağlantısı
    sequence_number = Column(Integer, nullable=True)   # Adli kayıt sıra numarası
    details_json = Column(JSON, nullable=True)
    created_at_utc = Column(String(64), nullable=False)

    case = relationship("CaseEntity", back_populates="audit_logs")


class AnalysisJobEntity(Base):
    """Asenkron Adli Analiz Görev / İş Kuyruğu Varlığı (Job Queue)."""
    __tablename__ = "analysis_jobs"

    job_id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    case_id = Column(String(64), ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(String(64), nullable=True)
    status = Column(String(32), default="QUEUED", nullable=False, index=True)  # QUEUED, PROCESSING, COMPLETED, FAILED, CANCELLED
    progress = Column(Integer, default=0, nullable=False)  # 0 - 100
    step_label = Column(String(255), default="Sıraya Alındı", nullable=False)
    current_step = synonym("step_label")
    created_at_utc = Column(String(64), nullable=False)
    started_at_utc = Column(String(64), nullable=True)
    completed_at_utc = Column(String(64), nullable=True)
    error_message = Column(Text, nullable=True)
    result_json = Column(JSON, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)

    case = relationship("CaseEntity", back_populates="jobs")


class RefreshTokenEntity(Base):
    """Güvenli Yenileme Belirteci (Refresh Token) Varlığı (Rotasyon ve İptal Takibi)."""
    __tablename__ = "refresh_tokens"

    id = Column(String(64), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    token_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA-256
    expires_at_utc = Column(String(64), nullable=False)
    is_revoked = Column(Boolean, default=False, nullable=False, index=True)
    created_at_utc = Column(String(64), nullable=False)
    replaced_by_token_hash = Column(String(64), nullable=True)  # Yeniden kullanım tespiti (Reuse detection)
