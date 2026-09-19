"""initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-09-15 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Cases Tablosu
    op.create_table(
        "cases",
        sa.Column("case_id", sa.String(length=64), primary_key=True),
        sa.Column("name", sa.String(length=255), nullable=True, default="Adli Vaka"),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("examination_type", sa.String(length=64), nullable=False, default="SIGNATURE"),
        sa.Column("reference_number", sa.String(length=128), nullable=True),
        sa.Column("authority", sa.String(length=255), nullable=True, default="Cumhuriyet Başsavcılığı / Mahkeme"),
        sa.Column("examiner_id", sa.String(length=128), nullable=False, default="BILIRKISI_UZMAN_01"),
        sa.Column("notes", sa.Text(), nullable=True, default=""),
        sa.Column("status", sa.String(length=64), nullable=False, default="IN_PROGRESS"),
        sa.Column("current_stage", sa.String(length=64), nullable=False, default="CASE_CREATED"),
        sa.Column("state", sa.String(length=64), nullable=False, default="PHASE_A_INIT"),
        sa.Column("questioned_sample_id", sa.String(length=128), nullable=True),
        sa.Column("suspect_name", sa.String(length=128), nullable=True),
        sa.Column("similarity_score", sa.Float(), nullable=True),
        sa.Column("verbal_scale", sa.String(length=128), nullable=True),
        sa.Column("audit_hash", sa.String(length=64), nullable=True),
        sa.Column("merkle_root_hash", sa.String(length=64), nullable=True),
        sa.Column("created_at_utc", sa.String(length=64), nullable=False),
        sa.Column("updated_at_utc", sa.String(length=64), nullable=False),
        sa.Column("is_deleted", sa.Boolean(), default=False, nullable=False),
        sa.Column("deleted_at_utc", sa.String(length=64), nullable=True),
        sa.Column("deleted_by", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_cases_case_id", "cases", ["case_id"])
    op.create_index("ix_cases_name", "cases", ["name"])
    op.create_index("ix_cases_is_deleted", "cases", ["is_deleted"])

    # 2. Users Tablosu
    op.create_table(
        "users",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("username", sa.String(length=64), unique=True, nullable=False),
        sa.Column("hashed_password", sa.String(length=255), nullable=False),
        sa.Column("full_name", sa.String(length=128), nullable=True),
        sa.Column("role", sa.String(length=32), default="EXAMINER", nullable=False),
        sa.Column("is_active", sa.Boolean(), default=True, nullable=False),
        sa.Column("created_at_utc", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_users_username", "users", ["username"])

    # 3. Samples Tablosu
    op.create_table(
        "samples",
        sa.Column("sample_id", sa.String(length=128), primary_key=True),
        sa.Column("case_id", sa.String(length=64), sa.ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("sample_type", sa.String(length=32), nullable=False),
        sa.Column("sha256_hash", sa.String(length=64), nullable=False),
        sa.Column("file_size_bytes", sa.Integer(), nullable=False, default=0),
        sa.Column("file_path", sa.String(length=512), nullable=True),
        sa.Column("created_at_utc", sa.String(length=64), nullable=False),
        sa.Column("is_deleted", sa.Boolean(), default=False, nullable=False),
        sa.Column("deleted_at_utc", sa.String(length=64), nullable=True),
        sa.Column("deleted_by", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_samples_sample_id", "samples", ["sample_id"])
    op.create_index("ix_samples_case_id", "samples", ["case_id"])
    op.create_index("ix_samples_is_deleted", "samples", ["is_deleted"])

    # 4. Analysis Results Tablosu
    op.create_table(
        "analysis_results",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("case_id", sa.String(length=64), sa.ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False),
        sa.Column("suitability_json", sa.JSON(), nullable=True),
        sa.Column("complexity_json", sa.JSON(), nullable=True),
        sa.Column("comparison_json", sa.JSON(), nullable=True),
        sa.Column("disguise_json", sa.JSON(), nullable=True),
        sa.Column("fusion_json", sa.JSON(), nullable=True),
        sa.Column("xai_json", sa.JSON(), nullable=True),
        sa.Column("hypothesis_json", sa.JSON(), nullable=True),
        sa.Column("audit_checksum_sha256", sa.String(length=64), nullable=False),
        sa.Column("merkle_root_hash", sa.String(length=64), nullable=True),
        sa.Column("html_report_path", sa.String(length=512), nullable=True),
        sa.Column("pdf_report_path", sa.String(length=512), nullable=True),
        sa.Column("created_at_utc", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_analysis_results_case_id", "analysis_results", ["case_id"])

    # 5. Audit Logs Tablosu
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("case_id", sa.String(length=64), sa.ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("action", sa.String(length=128), nullable=False),
        sa.Column("examiner_id", sa.String(length=128), nullable=True),
        sa.Column("sha256_hash", sa.String(length=64), nullable=False),
        sa.Column("previous_hash", sa.String(length=64), nullable=True),
        sa.Column("sequence_number", sa.Integer(), nullable=True),
        sa.Column("details_json", sa.JSON(), nullable=True),
        sa.Column("created_at_utc", sa.String(length=64), nullable=False),
    )
    op.create_index("ix_audit_logs_case_id", "audit_logs", ["case_id"])
    op.create_index("ix_audit_logs_event_type", "audit_logs", ["event_type"])

    # 6. Analysis Jobs Tablosu
    op.create_table(
        "analysis_jobs",
        sa.Column("job_id", sa.String(length=64), primary_key=True),
        sa.Column("case_id", sa.String(length=64), sa.ForeignKey("cases.case_id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), default="QUEUED", nullable=False),
        sa.Column("progress", sa.Integer(), default=0, nullable=False),
        sa.Column("step_label", sa.String(length=255), default="Sıraya Alındı", nullable=False),
        sa.Column("created_at_utc", sa.String(length=64), nullable=False),
        sa.Column("started_at_utc", sa.String(length=64), nullable=True),
        sa.Column("completed_at_utc", sa.String(length=64), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.Column("retry_count", sa.Integer(), default=0, nullable=False),
        sa.Column("max_retries", sa.Integer(), default=3, nullable=False),
    )
    op.create_index("ix_analysis_jobs_case_id", "analysis_jobs", ["case_id"])
    op.create_index("ix_analysis_jobs_status", "analysis_jobs", ["status"])

    # 7. Refresh Tokens Tablosu
    op.create_table(
        "refresh_tokens",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("user_id", sa.String(length=64), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("token_hash", sa.String(length=64), unique=True, nullable=False),
        sa.Column("expires_at_utc", sa.String(length=64), nullable=False),
        sa.Column("is_revoked", sa.Boolean(), default=False, nullable=False),
        sa.Column("created_at_utc", sa.String(length=64), nullable=False),
        sa.Column("replaced_by_token_hash", sa.String(length=64), nullable=True),
    )
    op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    op.create_index("ix_refresh_tokens_token_hash", "refresh_tokens", ["token_hash"])
    op.create_index("ix_refresh_tokens_is_revoked", "refresh_tokens", ["is_revoked"])


def downgrade() -> None:
    op.drop_table("refresh_tokens")
    op.drop_table("analysis_jobs")
    op.drop_table("audit_logs")
    op.drop_table("analysis_results")
    op.drop_table("samples")
    op.drop_table("users")
    op.drop_table("cases")
