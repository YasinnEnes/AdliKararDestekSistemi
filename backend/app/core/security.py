"""
FHDSS - Güvenlik, Kimlik Doğrulama ve Rol Tabanlı Yetkilendirme (JWT & RBAC)
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.logger import logger
from backend.app.db.session import get_db
from backend.app.models.entities import UserEntity

# OAuth2 Şema Tanımı
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login",
    auto_error=False,
)

ALGORITHM = "HS256"


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Düz metin şifrenin bcrypt karmasıyla eşleştiğini doğrular."""
    try:
        pwd_bytes = plain_password.encode("utf-8")[:72]
        hash_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception:
        return False


def get_password_hash(password: str) -> str:
    """Şifreyi güvenli bcrypt algoritmasıyla karmalar."""
    pwd_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode("utf-8")


def create_access_token(
    subject: Union[str, Any],
    role: str = "EXAMINER",
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[Dict[str, Any]] = None,
) -> str:
    """Kullanıcı için süreli ve imzalı JWT erişim belirteci üretir."""
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode: Dict[str, Any] = {
        "sub": str(subject),
        "role": role,
        "iat": now,
        "exp": expire,
        "jti": secrets.token_hex(16),
    }
    if extra_claims:
        to_encode.update(extra_claims)

    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """JWT erişim belirtecini çözer ve imzayı doğrular."""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Oturum belirtecinin (token) süresi dolmuş. Lütfen tekrar giriş yapın.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz yetkilendirme belirteci.",
            headers={"WWW-Authenticate": "Bearer"},
        )


import hashlib
import secrets

def hash_refresh_token(token: str) -> str:
    """Yenileme belirtecini (Refresh Token) veritabanında saklamak için SHA-256 ile özetler."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def create_refresh_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> Tuple[str, str, datetime]:
    """
    Kullanıcı için rastgele ve kriptografik olarak güvenli bir Refresh Token üretir.
    Dönüş: (plain_token, token_hash, expire_dt)
    """
    raw_token = f"fhdss_rf_{secrets.token_urlsafe(48)}"
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire_dt = now + expires_delta
    else:
        expire_dt = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)

    token_hash = hash_refresh_token(raw_token)
    return raw_token, token_hash, expire_dt


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> UserEntity:
    """
    İstek başlığındaki JWT belirtecini doğrular ve ilgili kullanıcıyı döner.
    GÜVENLİK KURALI (Sprint 16.1):
    Production veya Staging ortamında token yoksa doğrudan 401 Unauthorized döner.
    Yalnızca ENVIRONMENT == 'test' durumunda geriye dönük test uyumluluğu için varsayılan kullanıcıya izin verilir.
    """
    if not token:
        if settings.ENVIRONMENT.lower() == "test":
            default_user = UserEntity(
                id="SYS_EXAMINER_01",
                username="bilirkisi_uzman",
                full_name="Bilirkişi Adli Uzman",
                role="EXAMINER",
                is_active=True,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            return default_user

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Yetkilendirme belirteci (token) eksik. Lütfen oturum açın.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(token)
    username: Optional[str] = payload.get("sub")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Belirteçte kullanıcı kimliği bulunamadı.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    stmt = select(UserEntity).where(UserEntity.username == username)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None:
        if settings.ENVIRONMENT.lower() == "test":
            # Test ortamında geriye dönük test paketi uyumluluğu için kullanıcıyı DB'ye kaydet ve dön
            user = UserEntity(
                id=f"USER_{username}",
                username=username,
                hashed_password="test_hashed_password",
                full_name=payload.get("name", username),
                role=payload.get("role", "EXAMINER"),
                is_active=True,
                created_at_utc=datetime.now(timezone.utc).isoformat(),
            )
            db.add(user)
            await db.commit()
            return user

        # GÜVENLİK (Sprint 20 - P0): Ghost User açığı kapatıldı.
        # Production ve Development ortamında DB'de bulunmayan veya hesabı silinmiş kullanıcılar için kesin 401 dönülür.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Oturum açan kullanıcı sistem veritabanında bulunamadı veya hesabı silinmiş.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_active_user(
    current_user: UserEntity = Depends(get_current_user),
) -> UserEntity:
    """Aktif kullanıcı kontrolü yapar."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Kullanıcı hesabı askıya alınmış veya pasif durumda.",
        )
    return current_user


def require_roles(allowed_roles: List[str]):
    """Belirli rollere sahip kullanıcıların erişimini denetleyen bağımlılık fabrikası."""
    async def role_checker(
        current_user: UserEntity = Depends(get_current_active_user),
    ) -> UserEntity:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Bu işlem için yetkiniz yetersiz. Gerekli rol(ler): {', '.join(allowed_roles)} (Mevcut rol: {current_user.role})",
            )
        return current_user

    return role_checker
