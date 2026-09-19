"""
Adli Denetim İzi ve Kriptografik Hash Zinciri API Uç Noktaları.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.security import get_current_active_user, require_roles
from backend.app.db.session import get_db
from backend.app.models.entities import AuditLogEntity, UserEntity
from backend.app.services.audit_service import verify_audit_hash_chain

router = APIRouter()


@router.get(
    "/verify-chain",
    summary="Kriptografik Denetim İzi Zincirini Doğrula",
)
async def verify_chain(
    case_id: Optional[str] = Query(None, description="Opsiyonel vaka filtresi"),
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Tüm sistemdeki (veya belirli bir vakadaki) denetim kayıtlarını sırayla inceleyip
    önceki ve mevcut SHA-256 hash zincirini doğrular.
    Herhangi bir veri manipülasyonu varsa tespit eder.
    """
    return await verify_audit_hash_chain(db, case_id=case_id)


@router.get(
    "/case/{case_id}",
    summary="Vakaya Ait Adli Denetim Kayıtlarını Listele",
)
async def get_case_audit_logs(
    case_id: str,
    limit: int = Query(100, ge=1, le=500),
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Belirtilen vakaya ait denetim izini sıralı olarak döner."""
    stmt = (
        select(AuditLogEntity)
        .where(AuditLogEntity.case_id == case_id)
        .order_by(AuditLogEntity.sequence_number.asc())
        .limit(limit)
    )
    res = await db.execute(stmt)
    records = res.scalars().all()

    return {
        "case_id": case_id,
        "count": len(records),
        "audit_trail": [
            {
                "id": r.id,
                "sequence_number": r.sequence_number,
                "action": r.action,
                "user_id": r.user_id,
                "timestamp_utc": r.timestamp_utc,
                "previous_hash": r.previous_hash,
                "current_hash": r.current_hash,
                "details": r.details,
            }
            for r in records
        ],
    }
