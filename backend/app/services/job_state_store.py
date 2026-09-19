"""
FHDSS - Dağıtık ve Bellek İçi İş / Durum Deposu (Distributed & In-Memory Job State Store).
Sprint 16.5: Çoklu worker desteği için Redis ve In-Memory durum & olay yöneticisi.
"""

import asyncio
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings
from backend.app.core.logger import logger

try:
    import redis
    _HAS_REDIS = True
except ImportError:
    _HAS_REDIS = False


class IJobStateStore(ABC):
    """İş ve Orkestratör Durum Deposu Soyut Arayüzü."""

    @abstractmethod
    def set_state(self, key: str, data: Dict[str, Any], expire_seconds: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    def delete_state(self, key: str) -> None:
        pass


class InMemoryJobStateStore(IJobStateStore):
    """Tekil worker veya test ortamı için bellek içi (In-Memory) durum deposu."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def set_state(self, key: str, data: Dict[str, Any], expire_seconds: Optional[int] = None) -> None:
        self._store[key] = dict(data)

    def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        val = self._store.get(key)
        return dict(val) if val is not None else None

    def delete_state(self, key: str) -> None:
        self._store.pop(key, None)


class RedisJobStateStore(IJobStateStore):
    """Çoklu worker (Uvicorn workers > 1) veya üretim ortamı için Redis tabanlı durum deposu."""

    def __init__(self, redis_url: str):
        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def set_state(self, key: str, data: Dict[str, Any], expire_seconds: Optional[int] = None) -> None:
        serialized = json.dumps(data, ensure_ascii=False)
        if expire_seconds:
            self.client.setex(f"fhdss_state:{key}", expire_seconds, serialized)
        else:
            self.client.set(f"fhdss_state:{key}", serialized)

    def get_state(self, key: str) -> Optional[Dict[str, Any]]:
        raw = self.client.get(f"fhdss_state:{key}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def delete_state(self, key: str) -> None:
        self.client.delete(f"fhdss_state:{key}")


class IJobEventPublisher(ABC):
    """Gerçek Zamanlı SSE / WebSocket Olay Yayıncısı Soyut Arayüzü."""

    @abstractmethod
    async def publish(self, channel_id: str, data: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def subscribe(self, channel_id: str) -> asyncio.Queue:
        pass

    @abstractmethod
    def unsubscribe(self, channel_id: str, queue: asyncio.Queue) -> None:
        pass


class InMemoryJobEventPublisher(IJobEventPublisher):
    """Süreç içi bellek tabanlı SSE abone yöneticisi."""

    def __init__(self):
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}

    def subscribe(self, channel_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        if channel_id not in self._subscribers:
            self._subscribers[channel_id] = []
        self._subscribers[channel_id].append(q)
        return q

    def unsubscribe(self, channel_id: str, queue: asyncio.Queue) -> None:
        if channel_id in self._subscribers:
            if queue in self._subscribers[channel_id]:
                self._subscribers[channel_id].remove(queue)
            if not self._subscribers[channel_id]:
                del self._subscribers[channel_id]

    async def publish(self, channel_id: str, data: Dict[str, Any]) -> None:
        if channel_id in self._subscribers:
            for q in list(self._subscribers[channel_id]):
                try:
                    await q.put(data)
                except Exception:
                    pass


# Fabrika Fonksiyonları (Factory)
def create_job_state_store() -> IJobStateStore:
    if settings.JOB_STATE_BACKEND == "redis" and _HAS_REDIS:
        try:
            store = RedisJobStateStore(settings.REDIS_URL)
            store.client.ping()
            logger.info("JobStateStore: RedisBackend başarıyla başlatıldı.")
            return store
        except Exception as e:
            logger.warning(f"JobStateStore: Redis'e bağlanılamadı ({e}), In-Memory kullanılacak.")
    return InMemoryJobStateStore()


def create_job_event_publisher() -> IJobEventPublisher:
    return InMemoryJobEventPublisher()


# Global Singleton Depolar
job_state_store: IJobStateStore = create_job_state_store()
job_event_publisher: IJobEventPublisher = create_job_event_publisher()
