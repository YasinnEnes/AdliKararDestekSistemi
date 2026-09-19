"""
FHDSS - Sistem Yöneticisi (Admin) ve Rol Yönetimi REST API Uç Noktaları.
Sprint 16.14: Yalnızca ADMIN rolünün erişebildiği kullanıcı yönetimi, rol atama ve denetim logu filtreleme.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.logger import log_audit_event, logger
from backend.app.core.security import get_current_active_user, get_password_hash, require_roles
from backend.app.db.session import get_db
from backend.app.models.entities import AuditLogEntity, UserEntity

router = APIRouter(dependencies=[Depends(require_roles(["ADMIN"]))])


class AdminUserCreateRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="Kullanıcı adı")
    password: str = Field(..., min_length=6, description="Geçici şifre")
    full_name: str = Field(..., description="Ad Soyad ve Ünvan")
    role: str = Field("EXAMINER", description="ADMIN, SENIOR_REVIEWER, EXAMINER")


class AdminUserRoleUpdateRequest(BaseModel):
    role: str = Field(..., description="Yeni rol: ADMIN, SENIOR_REVIEWER, EXAMINER")


class AdminUserStatusUpdateRequest(BaseModel):
    is_active: bool = Field(..., description="Kullanıcı aktiflik durumu")


class AdminPasswordResetRequest(BaseModel):
    new_password: str = Field(..., min_length=6, description="Yeni şifre")


class AdminUserResponse(BaseModel):
    id: str
    username: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at_utc: str


class AdminAuditLogItem(BaseModel):
    id: str
    case_id: Optional[str] = None
    event_type: str
    action: str
    examiner_id: Optional[str] = None
    sha256_hash: str
    details_json: Optional[Dict[str, Any]] = None
    created_at_utc: str


@router.get(
    "/users",
    response_model=List[AdminUserResponse],
    summary="Tüm Adli Kullanıcıları Listele (Admin)",
)
async def list_all_users(
    role: Optional[str] = Query(None, description="Role göre filtrele"),
    is_active: Optional[bool] = Query(None, description="Aktifliğe göre filtrele"),
    db: AsyncSession = Depends(get_db),
):
    """Sistemdeki kayıtlı tüm bilirkişi ve yönetici hesaplarını listeler."""
    stmt = select(UserEntity).order_by(desc(UserEntity.created_at_utc))
    if role:
        stmt = stmt.where(UserEntity.role == role.upper())
    if is_active is not None:
        stmt = stmt.where(UserEntity.is_active == is_active)

    res = await db.execute(stmt)
    users = res.scalars().all()

    return [
        AdminUserResponse(
            id=u.id,
            username=u.username,
            full_name=u.full_name,
            role=u.role,
            is_active=u.is_active,
            created_at_utc=u.created_at_utc,
        )
        for u in users
    ]


@router.post(
    "/users",
    response_model=AdminUserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Yeni Kullanıcı / Bilirkişi Oluştur (Admin)",
)
async def create_user(
    payload: AdminUserCreateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Yeni bir bilirkişi veya yönetici hesabı tanımlar."""
    check_stmt = select(UserEntity).where(UserEntity.username == payload.username)
    res = await db.execute(check_stmt)
    if res.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{payload.username}' kullanıcı adı zaten mevcut.",
        )

    valid_roles = ["ADMIN", "SENIOR_REVIEWER", "EXAMINER"]
    role = payload.role.upper() if payload.role.upper() in valid_roles else "EXAMINER"

    now_iso = datetime.now(timezone.utc).isoformat()
    new_u = UserEntity(
        username=payload.username,
        hashed_password=get_password_hash(payload.password),
        full_name=payload.full_name,
        role=role,
        is_active=True,
        created_at_utc=now_iso,
    )
    db.add(new_u)
    await db.commit()
    await db.refresh(new_u)

    log_audit_event(
        event_type="ADMIN_USER_CREATED",
        action="CREATE_USER_BY_ADMIN",
        examiner_id=current_user.username,
        details={"created_user_id": new_u.id, "created_username": new_u.username, "role": new_u.role},
    )

    return AdminUserResponse(
        id=new_u.id,
        username=new_u.username,
        full_name=new_u.full_name,
        role=new_u.role,
        is_active=new_u.is_active,
        created_at_utc=new_u.created_at_utc,
    )


@router.put(
    "/users/{user_id}/role",
    response_model=AdminUserResponse,
    summary="Kullanıcı Rolünü Güncelle (Admin)",
)
async def update_user_role(
    user_id: str,
    payload: AdminUserRoleUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Kullanıcının yetki seviyesini (Rolünü) günceller."""
    valid_roles = ["ADMIN", "SENIOR_REVIEWER", "EXAMINER"]
    new_role = payload.role.upper()
    if new_role not in valid_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Geçersiz rol. İzin verilen roller: {', '.join(valid_roles)}",
        )

    stmt = select(UserEntity).where(UserEntity.id == user_id)
    res = await db.execute(stmt)
    u = res.scalar_one_or_none()
    if not u:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kullanıcı bulunamadı.")

    u.role = new_role
    await db.commit()
    await db.refresh(u)

    log_audit_event(
        event_type="ADMIN_USER_ROLE_UPDATED",
        action="UPDATE_USER_ROLE",
        examiner_id=current_user.username,
        details={"target_user_id": u.id, "new_role": new_role},
    )

    return AdminUserResponse(
        id=u.id,
        username=u.username,
        full_name=u.full_name,
        role=u.role,
        is_active=u.is_active,
        created_at_utc=u.created_at_utc,
    )


@router.put(
    "/users/{user_id}/status",
    response_model=AdminUserResponse,
    summary="Kullanıcıyı Aktifleştir / Pasifleştir (Admin)",
)
async def update_user_status(
    user_id: str,
    payload: AdminUserStatusUpdateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Kullanıcı hesabını dondurur veya yeniden aktifleştirir."""
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kendi yönetici hesabınızı pasifleştiremezsiniz.",
        )

    stmt = select(UserEntity).where(UserEntity.id == user_id)
    res = await db.execute(stmt)
    u = res.scalar_one_or_none()
    if not u:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kullanıcı bulunamadı.")

    u.is_active = payload.is_active
    await db.commit()
    await db.refresh(u)

    log_audit_event(
        event_type="ADMIN_USER_STATUS_UPDATED",
        action="DEACTIVATE_USER" if not payload.is_active else "ACTIVATE_USER",
        examiner_id=current_user.username,
        details={"target_user_id": u.id, "is_active": u.is_active},
    )

    return AdminUserResponse(
        id=u.id,
        username=u.username,
        full_name=u.full_name,
        role=u.role,
        is_active=u.is_active,
        created_at_utc=u.created_at_utc,
    )


@router.post(
    "/users/{user_id}/reset-password",
    summary="Kullanıcı Şifresini Sıfırla (Admin)",
)
async def reset_user_password(
    user_id: str,
    payload: AdminPasswordResetRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Yönetici tarafından kullanıcının şifresini güvenle sıfırlar."""
    stmt = select(UserEntity).where(UserEntity.id == user_id)
    res = await db.execute(stmt)
    u = res.scalar_one_or_none()
    if not u:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Kullanıcı bulunamadı.")

    u.hashed_password = get_password_hash(payload.new_password)
    await db.commit()

    log_audit_event(
        event_type="ADMIN_PASSWORD_RESET",
        action="RESET_PASSWORD_BY_ADMIN",
        examiner_id=current_user.username,
        details={"target_user_id": u.id},
    )

    return {"status": "success", "message": f"'{u.username}' kullanıcısının şifresi başarıyla sıfırlandı."}


@router.get(
    "/audit-logs",
    response_model=List[AdminAuditLogItem],
    summary="Tüm Denetim İzlerini Filtrele ve Listele (Admin)",
)
async def query_audit_logs(
    case_id: Optional[str] = Query(None, description="Vaka ID'ye göre filtrele"),
    examiner_id: Optional[str] = Query(None, description="Bilirkişi / Kullanıcıya göre filtrele"),
    event_type: Optional[str] = Query(None, description="Olay türüne göre filtrele"),
    limit: int = Query(50, ge=1, le=500, description="Kayıt limiti"),
    offset: int = Query(0, ge=0, description="Sayfalama ötelemesi"),
    db: AsyncSession = Depends(get_db),
):
    """Sistemdeki tüm adli denetim ve işlem kütüklerini arar ve listeler."""
    stmt = select(AuditLogEntity).order_by(desc(AuditLogEntity.created_at_utc))
    if case_id:
        stmt = stmt.where(AuditLogEntity.case_id == case_id)
    if examiner_id:
        stmt = stmt.where(AuditLogEntity.examiner_id == examiner_id)
    if event_type:
        stmt = stmt.where(AuditLogEntity.event_type == event_type)

    stmt = stmt.limit(limit).offset(offset)
    res = await db.execute(stmt)
    logs = res.scalars().all()

    return [
        AdminAuditLogItem(
            id=log.id,
            case_id=log.case_id,
            event_type=log.event_type,
            action=log.action,
            examiner_id=log.examiner_id,
            sha256_hash=log.sha256_hash,
            details_json=log.details_json,
            created_at_utc=log.created_at_utc,
        )
        for log in logs
    ]
