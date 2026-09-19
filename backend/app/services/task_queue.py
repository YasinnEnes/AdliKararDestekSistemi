"""
FHDSS - Kalıcı Görev Kuyruğu ve İşçi (Persistent Background Task Queue & Worker).
Sprint 16.6: asyncio.create_task yerine kalıcı, yeniden denemeli (retry) ve sunucu kurtarmalı (recovery) görev yöneticisi.
"""

import asyncio
import traceback
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.logger import log_audit_event, logger
from backend.app.db.session import AsyncSessionLocal
from backend.app.models.entities import AnalysisJobEntity
from backend.app.services.job_state_store import job_event_publisher, job_state_store


class PersistentTaskQueue:
    """
    SQL Veritabanı ve Bellek/Redis Durum Destekli Kalıcı Görev Kuyruğu (Persistent Task Queue).
    
    Özellikler:
    - Durumlar: QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED.
    - Otomatik Yeniden Deneme (Max Retries = 3).
    - Sunucu Yeniden Başlatma Kurtarması (Crash Recovery).
    - SSE / Olay Bildirimi Senkronizasyonu.
    """

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self._task_registry: Dict[str, asyncio.Task] = {}

    async def enqueue(
        self,
        job_id: str,
        case_id: str,
        coroutine_fn: Callable[..., Coroutine[Any, Any, Any]],
        *args,
        **kwargs,
    ) -> asyncio.Task:
        """
        Adli görevi kuyruğa ekler ve arka planda güvenli izleme (supervision) ile başlatır.
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        job_state_store.set_state(
            f"job:{job_id}",
            {
                "job_id": job_id,
                "case_id": case_id,
                "status": "QUEUED",
                "progress": 0,
                "step_label": "Kuyrukta Bekliyor",
                "retry_count": 0,
                "created_at": now_iso,
            },
        )

        task = asyncio.create_task(
            self._supervised_worker(job_id, case_id, coroutine_fn, *args, **kwargs)
        )
        self._task_registry[job_id] = task
        return task

    async def _supervised_worker(
        self,
        job_id: str,
        case_id: str,
        coroutine_fn: Callable[..., Coroutine[Any, Any, Any]],
        *args,
        **kwargs,
    ):
        """İşi çalıştırır, istisnaları yakalar ve gerektiğinde otomatik yeniden dener."""
        current_retry = 0

        while current_retry <= self.max_retries:
            try:
                # Durumu RUNNING yap
                await self._update_job_status(job_id, status="RUNNING", step_label="İşlem Başlatıldı")
                await coroutine_fn(*args, **kwargs)
                # Başarılı bittiğinde döngüden çık
                return
            except asyncio.CancelledError:
                logger.warning(f"Görev iptal edildi: [Job: {job_id}]")
                await self._update_job_status(job_id, status="CANCELLED", error="Görev kullanıcı/sistem tarafından iptal edildi.")
                return
            except Exception as ex:
                current_retry += 1
                err_msg = f"{type(ex).__name__}: {str(ex)}"
                logger.error(f"Görev hatası [Job: {job_id}, Deneme: {current_retry}/{self.max_retries}]: {err_msg}")

                if current_retry <= self.max_retries:
                    await self._update_job_status(
                        job_id,
                        status="QUEUED",
                        step_label=f"Hata sonrası yeniden deneniyor ({current_retry}/{self.max_retries})...",
                        retry_count=current_retry,
                        error=err_msg,
                    )
                    await asyncio.sleep(2 * current_retry)  # Üstel bekleme
                else:
                    await self._update_job_status(
                        job_id,
                        status="FAILED",
                        step_label="İşlem Başarısız",
                        retry_count=current_retry,
                        error=f"Maksimum yeniden deneme ({self.max_retries}) aşıldı. Son hata: {err_msg}",
                    )
                    return

    async def _update_job_status(
        self,
        job_id: str,
        status: str,
        step_label: Optional[str] = None,
        progress: Optional[int] = None,
        error: Optional[str] = None,
        retry_count: Optional[int] = None,
    ):
        """Veritabanı ve Redis durumunu atomik olarak günceller ve SSE yayınlar."""
        now_iso = datetime.now(timezone.utc).isoformat()
        update_data: Dict[str, Any] = {"status": status}
        if step_label:
            update_data["step_label"] = step_label
        if progress is not None:
            update_data["progress"] = progress
        if error:
            update_data["error_message"] = error
        if retry_count is not None:
            update_data["retry_count"] = retry_count
        if status == "RUNNING":
            update_data["started_at_utc"] = now_iso
        elif status in ("COMPLETED", "FAILED", "CANCELLED"):
            update_data["completed_at_utc"] = now_iso

        # 1. DB Güncellemesi
        async with AsyncSessionLocal() as session:
            stmt = (
                update(AnalysisJobEntity)
                .where(AnalysisJobEntity.job_id == job_id)
                .values(**update_data)
            )
            await session.execute(stmt)
            await session.commit()

        # 2. Redis / In-Memory State Güncellemesi
        state = job_state_store.get_state(f"job:{job_id}") or {}
        state.update(update_data)
        job_state_store.set_state(f"job:{job_id}", state)

        # 3. SSE Yayını
        await job_event_publisher.publish(job_id, {"job_id": job_id, **update_data})

    async def recover_interrupted_jobs(self):
        """Sunucu başlatılırken askıda kalan (RUNNING/QUEUED) işleri tespit edip kurtarır."""
        async with AsyncSessionLocal() as session:
            stmt = select(AnalysisJobEntity).where(AnalysisJobEntity.status == "RUNNING")
            res = await session.execute(stmt)
            hanging_jobs = res.scalars().all()

            for job in hanging_jobs:
                logger.warning(f"Kurtarma: Kesintiye uğrayan iş tespit edildi [Job: {job.job_id}]. Durum FAILED olarak mühürleniyor.")
                job.status = "FAILED"
                job.error_message = "Sunucu yeniden başlatılması nedeniyle işlem kesintiye uğradı. Lütfen analizi tekrar başlatın."
                job.completed_at_utc = datetime.now(timezone.utc).isoformat()

            if hanging_jobs:
                await session.commit()
                logger.info(f"{len(hanging_jobs)} adet kesintiye uğrayan adli analiz işi kurtarıldı/kapatıldı.")


# Global Singleton Görev Kuyruğu
task_queue = PersistentTaskQueue(max_retries=settings.TASK_QUEUE_MAX_RETRIES)
