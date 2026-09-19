"""
FHDSS - Dağıtık ve Bellek İçi Hız Sınırlayıcı (Rate Limiter) Modülü.
Sprint 16.3: Brute-force ve DoS saldırılarına karşı IP ve Kullanıcı bazlı hız sınırlama.
"""

import time
from typing import Dict, List, Optional
from fastapi import HTTPException, Request, status

from backend.app.core.config import settings
from backend.app.core.logger import logger

try:
    import redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


class SlidingWindowRateLimiter:
    """
    Kayan Pencere (Sliding Window) Hız Sınırlayıcısı.
    Redis mevcutsa Redis Sorted Set kullanır; aksi halde güvenli bellek içi (in-memory) veri yapısı işletir.
    """

    def __init__(self, redis_url: Optional[str] = None):
        self.redis_client = None
        if _HAS_REDIS and redis_url and settings.JOB_STATE_BACKEND == "redis":
            try:
                client = redis.Redis.from_url(redis_url, decode_responses=True)
                client.ping()
                self.redis_client = client
                logger.info(f"RateLimiter: Redis bağlantısı aktif ({redis_url})")
            except Exception as e:
                logger.warning(f"RateLimiter: Redis'e bağlanılamadı, bellek içi fallback devrede: {e}")
                self.redis_client = None

        # Bellek içi fallback deposu: { key: [(timestamp_1), (timestamp_2), ...] }
        self._in_memory_store: Dict[str, List[float]] = {}

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> tuple[bool, int]:
        """
        İsteğe izin verilip verilmeyeceğini ve aşıldıysa kalan bekleme süresini (saniye) döner.
        Dönüş: (is_allowed: bool, retry_after: int)
        """
        if not settings.RATE_LIMIT_ENABLED:
            return True, 0

        now = time.time()
        window_start = now - window_seconds

        # 1. Redis Katmanı
        if self.redis_client:
            try:
                pipe = self.redis_client.pipeline()
                redis_key = f"rate_limit:{key}"
                pipe.zremrangebyscore(redis_key, 0, window_start)
                pipe.zcard(redis_key)
                pipe.zadd(redis_key, {str(now): now})
                pipe.expire(redis_key, window_seconds + 5)
                results = pipe.execute()
                current_count = results[1]

                if current_count >= max_requests:
                    return False, int(window_seconds)
                return True, 0
            except Exception as ex:
                logger.warning(f"RateLimiter: Redis işlemi başarısız, in-memory'e düşülüyor: {ex}")

        # 2. Bellek İçi Katman (Fallback)
        timestamps = self._in_memory_store.get(key, [])
        # Pencere dışı kalan eski zaman damgalarını temizle
        timestamps = [ts for ts in timestamps if ts > window_start]

        if len(timestamps) >= max_requests:
            earliest = timestamps[0]
            retry_after = max(1, int(window_seconds - (now - earliest)))
            self._in_memory_store[key] = timestamps
            return False, retry_after

        timestamps.append(now)
        self._in_memory_store[key] = timestamps
        return True, 0

    def reset_key(self, key: str):
        """Başarılı işlemlerde (ör. geçerli login) sayacı sıfırlar."""
        if self.redis_client:
            try:
                self.redis_client.delete(f"rate_limit:{key}")
            except Exception:
                pass
        if key in self._in_memory_store:
            del self._in_memory_store[key]


# Global tekil rate limiter örneği
limiter = SlidingWindowRateLimiter(redis_url=settings.REDIS_URL)


def get_client_ip(request: Request) -> str:
    """İstemci gerçek IP adresini tespit eder (Proxy/Forwarded başlıkları dahil)."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "127.0.0.1"


async def check_login_rate_limit(request: Request):
    """Giriş endpoint'i için IP bazlı brute-force koruma bağımlılığı."""
    client_ip = get_client_ip(request)
    key = f"login:{client_ip}"
    allowed, retry_after = limiter.is_allowed(
        key=key,
        max_requests=settings.RATE_LIMIT_LOGIN_MAX_ATTEMPTS,
        window_seconds=settings.RATE_LIMIT_LOGIN_WINDOW_SEC,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Çok fazla başarısız giriş denemesi yapıldı. Lütfen {retry_after} saniye sonra tekrar deneyin.",
            headers={"Retry-After": str(retry_after)},
        )


def check_pipeline_rate_limit(user_identifier: str):
    """Pipeline uç noktası için kullanıcı bazlı DoS koruma fonksiyonu."""
    key = f"pipeline:{user_identifier}"
    allowed, retry_after = limiter.is_allowed(
        key=key,
        max_requests=settings.RATE_LIMIT_PIPELINE_MAX_REQUESTS,
        window_seconds=settings.RATE_LIMIT_PIPELINE_WINDOW_SEC,
    )
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Analiz boru hattı istek sınırı aşıldı. Lütfen {retry_after} saniye sonra tekrar deneyin.",
            headers={"Retry-After": str(retry_after)},
        )
