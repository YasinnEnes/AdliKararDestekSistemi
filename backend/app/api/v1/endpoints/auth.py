"""
FHDSS - Kimlik Doğrulama ve Kullanıcı Yönetimi REST API Uç Noktaları
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.logger import log_audit_event, logger
from backend.app.core.rate_limiter import check_login_rate_limit, get_client_ip, limiter
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    get_current_active_user,
    get_password_hash,
    hash_refresh_token,
    verify_password,
)
from backend.app.db.session import get_db
from backend.app.models.entities import RefreshTokenEntity, UserEntity

router = APIRouter()


class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=64, description="Kullanıcı adı")
    password: str = Field(..., min_length=6, description="Şifre (en az 6 karakter)")
    full_name: Optional[str] = Field(None, description="Ad Soyad ve Ünvan")
    role: str = Field("EXAMINER", description="Rol: EXAMINER, SENIOR_REVIEWER, ADMIN")


class UserLoginRequest(BaseModel):
    username: str
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., description="Dönen yenileme belirteci (Refresh Token)")


class UserResponse(BaseModel):
    id: str
    username: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at_utc: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    refresh_token: Optional[str] = None
    user: UserResponse


class RefreshTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Bilirkişi / Kullanıcı Girişi (JWT ve Refresh Belirteci Üretir)",
)
async def login(
    request: Request,
    payload: Optional[UserLoginRequest] = None,
    form_data: Optional[OAuth2PasswordRequestForm] = Depends(lambda: None),
    db: AsyncSession = Depends(get_db),
):
    """
    Kullanıcı adı ve şifre ile güvenli oturum açar, imzalı JWT ve döner Refresh Token üretir.
    Sprint 16.3: IP bazlı brute-force hız sınırlaması (5 deneme/dakika).
    Sprint 16.4: Kriptografik hash'li kalıcı Refresh Token desteği.
    """
    # 1. Hız Sınırlaması Kontrolü
    await check_login_rate_limit(request)

    username = None
    password = None

    if payload:
        username = payload.username
        password = payload.password
    elif form_data:
        username = form_data.username
        password = form_data.password

    if not username or not password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Kullanıcı adı ve şifre zorunludur.",
        )

    stmt = select(UserEntity).where(UserEntity.username == username)
    res = await db.execute(stmt)
    user = res.scalar_one_or_none()

    # GÜVENLİK (Sprint 20 - P0): Varsayılan test hesabı kontrolü yalnızca TEST ortamında geçerlidir
    if user is None and username in ["admin", "bilirkisi_uzman"] and settings.ENVIRONMENT.lower() == "test":
        default_pwd = "password123"
        if password == default_pwd:
            user = UserEntity(
                username=username,
                hashed_password=get_password_hash(default_pwd),
                full_name="Baş Bilirkişi / Sistem Yöneticisi" if username == "admin" else "Bilirkişi Adli Uzman",
                role="ADMIN" if username == "admin" else "EXAMINER",
                is_active=True,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)

    if user is None or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Hatalı kullanıcı adı veya şifre.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Kullanıcı hesabı aktif değil.",
        )

    # Başarılı girişte IP hız sınırını sıfırla
    limiter.reset_key(f"login:{get_client_ip(request)}")

    token = create_access_token(
        subject=user.username,
        role=user.role,
        extra_claims={"name": user.full_name or user.username, "uid": user.id},
    )

    # Sprint 16.4: Kriptografik güvenli Refresh Token üret ve hash olarak DB'ye yaz
    raw_refresh_token, token_hash, expire_dt = create_refresh_token(subject=user.username)
    now_iso = datetime.now(timezone.utc).isoformat()
    rf_entity = RefreshTokenEntity(
        user_id=user.id,
        token_hash=token_hash,
        expires_at_utc=expire_dt.isoformat(),
        is_revoked=False,
        created_at_utc=now_iso,
    )
    db.add(rf_entity)
    await db.commit()

    log_audit_event(
        event_type="AUTH_LOGIN_SUCCESS",
        action="USER_AUTHENTICATED",
        examiner_id=user.username,
        details={"user_id": user.id, "role": user.role},
    )

    return TokenResponse(
        access_token=token,
        token_type="bearer",
        refresh_token=raw_refresh_token,
        user=UserResponse(
            id=user.id,
            username=user.username,
            full_name=user.full_name,
            role=user.role,
            is_active=user.is_active,
            created_at_utc=user.created_at_utc,
        ),
    )


@router.post(
    "/refresh",
    response_model=RefreshTokenResponse,
    summary="Erişim Belirtecini Yenile (Refresh Token Rotation & Reuse Detection)",
)
async def refresh_access_token(
    payload: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Kullanıcıya ait geçerli Refresh Token ile yeni bir Access Token ve yeni bir Refresh Token üretir.
    GÜVENLİK (Sprint 16.4):
    - Token Rotation: Her refresh işleminde eski token iptal edilir, yeni token verilir.
    - Reuse Detection: Zaten kullanılmış veya iptal edilmiş bir token tekrar kullanılırsa,
      olası çalınma şüphesiyle kullanıcının tüm oturumları geçersiz kılınır.
    """
    raw_token = payload.refresh_token.strip()
    target_hash = hash_refresh_token(raw_token)

    stmt = select(RefreshTokenEntity).where(RefreshTokenEntity.token_hash == target_hash)
    res = await db.execute(stmt)
    token_record = res.scalar_one_or_none()

    if not token_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz yetkilendirme yenileme belirteci.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 1. Reuse Detection (Yeniden Kullanım Tespiti)
    if token_record.is_revoked or token_record.replaced_by_token_hash:
        # İptal edilmiş bir token tekrar gönderildiyse güvenlik ihlali: kullanıcının tüm tokenlarını yak
        await db.execute(
            update(RefreshTokenEntity)
            .where(RefreshTokenEntity.user_id == token_record.user_id)
            .values(is_revoked=True)
        )
        await db.commit()
        log_audit_event(
            event_type="AUTH_SECURITY_ALERT",
            action="REFRESH_TOKEN_REUSE_DETECTED",
            details={"user_id": token_record.user_id, "compromised_token_hash": target_hash},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Oturum güvenlik ihlali tespit edildi: Daha önce kullanılmış veya iptal edilmiş belirteç gönderildi. Lütfen tekrar giriş yapın.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2. Süre Kontrolü
    now = datetime.now(timezone.utc)
    try:
        expire_dt = datetime.fromisoformat(token_record.expires_at_utc)
        if expire_dt.tzinfo is None:
            expire_dt = expire_dt.replace(tzinfo=timezone.utc)
        if now > expire_dt:
            token_record.is_revoked = True
            await db.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Yenileme belirtecinin süresi dolmuş. Lütfen tekrar giriş yapın.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    except ValueError:
        pass

    # 3. Kullanıcı Durum Kontrolü
    u_stmt = select(UserEntity).where(UserEntity.id == token_record.user_id)
    u_res = await db.execute(u_stmt)
    user = u_res.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Kullanıcı hesabı askıya alınmış veya mevcut değil.",
        )

    # 4. Token Rotation: Yeni Access Token ve Yeni Refresh Token Üret
    new_access_token = create_access_token(
        subject=user.username,
        role=user.role,
        extra_claims={"name": user.full_name or user.username, "uid": user.id},
    )
    new_raw_refresh, new_token_hash, new_expire_dt = create_refresh_token(subject=user.username)

    # Eski token'ı iptal et ve yenisiyle bağla
    token_record.is_revoked = True
    token_record.replaced_by_token_hash = new_token_hash

    # Yeni token kaydı oluştur
    new_rf_entity = RefreshTokenEntity(
        user_id=user.id,
        token_hash=new_token_hash,
        expires_at_utc=new_expire_dt.isoformat(),
        is_revoked=False,
        created_at_utc=now.isoformat(),
    )
    db.add(new_rf_entity)
    await db.commit()

    return RefreshTokenResponse(
        access_token=new_access_token,
        refresh_token=new_raw_refresh,
        token_type="bearer",
    )


@router.post(
    "/logout",
    summary="Oturumu Güvenli Şekilde Kapat ve Refresh Belirtecini İptal Et",
)
async def logout(
    payload: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Kullanıcının aktif Refresh Token'ını iptal ederek oturumu güvenli kapatır."""
    raw_token = payload.refresh_token.strip()
    target_hash = hash_refresh_token(raw_token)

    stmt = (
        update(RefreshTokenEntity)
        .where(
            RefreshTokenEntity.token_hash == target_hash,
            RefreshTokenEntity.user_id == current_user.id,
        )
        .values(is_revoked=True)
    )
    await db.execute(stmt)
    await db.commit()

    log_audit_event(
        event_type="AUTH_LOGOUT",
        action="USER_LOGGED_OUT",
        examiner_id=current_user.username,
        details={"user_id": current_user.id},
    )

    return {"status": "success", "message": "Oturum başarıyla sonlandırıldı ve yenileme belirteci iptal edildi."}


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Aktif Oturum Açmış Bilirkişi Profil Bilgileri",
)
async def get_current_user_profile(
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Mevcut yetkilendirilmiş kullanıcının profilini döner."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        full_name=current_user.full_name,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at_utc=current_user.created_at_utc,
    )


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Yeni Adli Bilirkişi / İncelemeci Kullanıcısı Kaydet",
)
async def register_user(
    payload: UserRegisterRequest,
    db: AsyncSession = Depends(get_db),
):
    """Sisteme yeni bir adli bilirkişi veya uzman kaydeder."""
    stmt = select(UserEntity).where(UserEntity.username == payload.username)
    res = await db.execute(stmt)
    existing = res.scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{payload.username}' kullanıcı adı zaten mevcut.",
        )

    # GÜVENLİK (Sprint 20 - P0): Genel kayıt olan kullanıcılar için ADMIN yetki yükseltmesi engellendi.
    # Public kayıt olan kullanıcılar yalnızca 'EXAMINER' veya 'SENIOR_REVIEWER' olabilir.
    # 'ADMIN' rolü yalnızca mevcut yetkili yönetici tarafından atanabilir.
    valid_roles = ["EXAMINER", "SENIOR_REVIEWER"]
    assigned_role = payload.role.upper() if payload.role.upper() in valid_roles else "EXAMINER"

    new_user = UserEntity(
        username=payload.username,
        hashed_password=get_password_hash(payload.password),
        full_name=payload.full_name,
        role=assigned_role,
        is_active=True,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)

    log_audit_event(
        event_type="USER_REGISTERED",
        action="REGISTER_FORENSIC_EXAMINER",
        examiner_id=new_user.username,
        details={"user_id": new_user.id, "role": new_user.role},
    )

    return UserResponse(
        id=new_user.id,
        username=new_user.username,
        full_name=new_user.full_name,
        role=new_user.role,
        is_active=new_user.is_active,
        created_at_utc=new_user.created_at_utc,
    )
