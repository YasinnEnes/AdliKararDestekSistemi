"""
Asenkron Adli Analiz İş Kuyruğu ve SSE (Server-Sent Events) API Uç Noktaları.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.logger import logger
from backend.app.core.security import get_current_active_user
from backend.app.db.session import get_db
from backend.app.models.entities import AnalysisJobEntity, UserEntity
from backend.app.schemas.api_contracts import FullPipelineRequest
from backend.app.services.job_service import broadcaster, run_pipeline_job

router = APIRouter()


@router.post(
    "/submit",
    summary="Asenkron Adli İnceleme İşini Başlat",
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_analysis_job(
    payload: FullPipelineRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Uçtan uca adli karar destek analizini arka plan işçisi olarak kuyruğa alır.
    İstemciye hemen `job_id` döner ve SSE akışı ile takip imkanı sunar.
    """
    job_id = f"job_{uuid.uuid4().hex[:12]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    job_entity = AnalysisJobEntity(
        job_id=job_id,
        case_id=payload.case_id,
        status="PENDING",
        progress=0,
        current_step="İşlem Sırasına Alındı",
        created_at_utc=now_iso,
    )
    db.add(job_entity)
    await db.flush()

    # Sprint 16.6: Kalıcı ve denetimli görev kuyruğuna ekle (Persistent Task Queue)
    from backend.app.services.task_queue import task_queue
    await task_queue.enqueue(
        job_id,
        payload.case_id,
        run_pipeline_job,
        job_id,
        payload.case_id,
        payload.questioned_sample_id,
        payload.reference_sample_ids,
        payload.suspect_name,
        current_user.username,
    )

    logger.info(f"Adli analiz işi başlatıldı: [Job: {job_id}, Vaka: {payload.case_id}, Kullanıcı: {current_user.username}]")

    return {
        "job_id": job_id,
        "case_id": payload.case_id,
        "status": "PENDING",
        "message": "Adli analiz işi başarıyla kuyruğa alındı.",
        "stream_url": f"/api/v1/jobs/{job_id}/stream",
    }


@router.get(
    "/{job_id}",
    summary="Analiz İşi Durum Sorgusu",
)
async def get_job_status(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Belirtilen adli analiz işinin güncel durumunu döndürür."""
    stmt = select(AnalysisJobEntity).where(AnalysisJobEntity.job_id == job_id)
    res = await db.execute(stmt)
    job = res.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{job_id}' adli işi bulunamadı.")

    return {
        "job_id": job.job_id,
        "case_id": job.case_id,
        "status": job.status,
        "progress": job.progress,
        "current_step": job.current_step,
        "error_message": job.error_message,
        "result": job.result_json,
        "created_at_utc": job.created_at_utc,
        "completed_at_utc": job.completed_at_utc,
    }


@router.get(
    "/case/{case_id}/latest",
    summary="Vakaya Ait Son Analiz İşini Getir",
)
async def get_latest_job_for_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Bir vakaya ait en son başlatılmış analiz işini döndürür."""
    stmt = (
        select(AnalysisJobEntity)
        .where(AnalysisJobEntity.case_id == case_id)
        .order_by(AnalysisJobEntity.id.desc())
        .limit(1)
    )
    res = await db.execute(stmt)
    job = res.scalar_one_or_none()

    if not job:
        return {"job": None}

    return {
        "job": {
            "job_id": job.job_id,
            "case_id": job.case_id,
            "status": job.status,
            "progress": job.progress,
            "current_step": job.current_step,
            "error_message": job.error_message,
            "result": job.result_json,
            "created_at_utc": job.created_at_utc,
            "completed_at_utc": job.completed_at_utc,
        }
    }


@router.get(
    "/{job_id}/stream",
    summary="SSE Gerçek Zamanlı İlerleme Akışı",
)
async def stream_job_progress(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Server-Sent Events (SSE) protokolü üzerinden analizin ilerleme yüzdesini,
    mevcut adli adımını ve tamamlanma/hata durumunu anlık olarak istemciye iletir.
    """
    # Mevcut durumu al
    stmt = select(AnalysisJobEntity).where(AnalysisJobEntity.job_id == job_id)
    res = await db.execute(stmt)
    job = res.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{job_id}' adli işi bulunamadı.")

    async def event_generator():
        # İlk mevcut durumu hemen ilet
        initial_event = {
            "job_id": job.job_id,
            "case_id": job.case_id,
            "status": job.status,
            "progress": job.progress,
            "step": job.current_step,
            "error": job.error_message,
            "result": job.result_json,
        }
        yield f"data: {json.dumps(initial_event)}\n\n"

        if job.status in ("COMPLETED", "FAILED"):
            return

        queue = broadcaster.subscribe(job_id)
        try:
            while True:
                # 30 saniye boyunca olay gelmezse ping gönder (keep-alive)
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("status") in ("COMPLETED", "FAILED"):
                        break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
        finally:
            broadcaster.unsubscribe(job_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
