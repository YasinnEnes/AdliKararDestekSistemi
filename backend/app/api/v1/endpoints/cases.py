"""
FHDSS - Vaka ve Numune Yönetimi REST API Uç Noktaları (Sprint 14: Tekilleştirilmiş Asenkron Mimari)
"""

import hashlib
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.context_isolation import (
    ContextIsolationOrchestrator,
    IsolationState,
    IsolationViolationError,
)
from backend.app.core.guard import PsychologicalInferenceGuard
from backend.app.core.logger import log_audit_event, logger
from backend.app.core.security import get_current_active_user, require_roles
from backend.app.db.session import get_db
from backend.app.models.entities import CaseEntity, SampleEntity, AuditLogEntity, UserEntity
from backend.app.schemas.api_contracts import (
    CaseCreate,
    CaseResponse,
    CaseUpdate,
    CaseStatus,
    CaseStage,
    ExaminationType,
    SampleUploadResponse,
)

router = APIRouter()

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_MAGIC_BYTES = [
    (b"\xFF\xD8\xFF", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"II*\x00", "image/tiff"),
    (b"MM\x00*", "image/tiff"),
]

# Global orkestratör önbelleği (durum makinesini canlı tutmak için)
_ORCHESTRATORS: Dict[str, ContextIsolationOrchestrator] = {}


def get_case_orchestrator(
    case_id: str,
    examiner_id: str = "BILIRKISI_UZMAN_01",
    current_state: str = "PHASE_A_INIT",
    questioned_sample_id: Optional[str] = None,
) -> ContextIsolationOrchestrator:
    """Belirtilen vaka için Bilişsel İzolasyon Orkestratörünü getirir veya başlatır."""
    if case_id not in _ORCHESTRATORS:
        orch = ContextIsolationOrchestrator(
            case_id=case_id,
            examiner_id=examiner_id,
        )
        try:
            orch.state = IsolationState(current_state)
        except Exception:
            orch.state = IsolationState.PHASE_A_INIT
        if questioned_sample_id:
            orch.questioned_sample_id = questioned_sample_id
        _ORCHESTRATORS[case_id] = orch
    else:
        orch = _ORCHESTRATORS[case_id]
        if questioned_sample_id and not orch.questioned_sample_id:
            orch.questioned_sample_id = questioned_sample_id
    return _ORCHESTRATORS[case_id]


async def _build_case_response(
    entity: CaseEntity,
    db: AsyncSession,
) -> CaseResponse:
    """CaseEntity ve ilişkili numune verilerini birleştirerek CaseResponse döner."""
    ref_stmt = select(SampleEntity.sample_id).where(
        SampleEntity.case_id == entity.case_id,
        SampleEntity.sample_type == "reference",
        SampleEntity.is_deleted == False,
    )
    ref_res = await db.execute(ref_stmt)
    ref_sample_ids = [row[0] for row in ref_res.all()]

    # Soru konusu numunenin silinip silinmediğini kontrol et
    q_sample_id = entity.questioned_sample_id
    if q_sample_id:
        q_chk = await db.execute(select(SampleEntity.sample_id).where(SampleEntity.sample_id == q_sample_id, SampleEntity.is_deleted == False))
        if not q_chk.scalar_one_or_none():
            q_sample_id = None

    # Examination type parse
    exam_type = ExaminationType.SIGNATURE
    if entity.examination_type in [e.value for e in ExaminationType]:
        exam_type = ExaminationType(entity.examination_type)

    # Status parse
    parsed_status = CaseStatus.IN_PROGRESS
    if entity.status in [s.value for s in CaseStatus]:
        parsed_status = CaseStatus(entity.status)
    elif entity.status:
        parsed_status = entity.status

    # Stage parse
    current_stage_val = CaseStage.CASE_CREATED
    if entity.current_stage in [s.value for s in CaseStage]:
        current_stage_val = CaseStage(entity.current_stage)

    # Isolation state parse
    state_val = IsolationState.PHASE_A_INIT
    if entity.state in [s.value for s in IsolationState]:
        state_val = IsolationState(entity.state)

    return CaseResponse(
        case_id=entity.case_id,
        case_number=entity.case_id,
        name=entity.name,
        description=entity.description or "",
        examination_type=exam_type,
        reference_number=entity.reference_number,
        authority=entity.authority,
        examiner_id=entity.examiner_id,
        examiner_name=entity.examiner_id,
        suspect_name=entity.suspect_name,
        notes=entity.notes or "",
        status=parsed_status,
        current_stage=current_stage_val,
        state=state_val,
        questioned_sample_id=q_sample_id,
        reference_sample_ids=ref_sample_ids,
        similarity_score=entity.similarity_score,
        verbal_scale=entity.verbal_scale,
        audit_hash=entity.audit_hash,
        created_at_utc=entity.created_at_utc,
        updated_at_utc=entity.updated_at_utc,
    )


# ------------------------------------------------------------------------------
# REST API Uç Noktaları (Tekil Asenkron Mimari)
# ------------------------------------------------------------------------------

@router.get(
    "",
    response_model=List[CaseResponse],
    summary="Tüm Kayıtlı Adli Vakaları Listele",
)
async def list_cases(
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Veritabanına kayıtlı tüm adli vakaları son güncellenme sırasına göre listeler."""
    try:
        stmt = select(CaseEntity).where(CaseEntity.is_deleted == False).order_by(CaseEntity.updated_at_utc.desc())
        res = await db.execute(stmt)
        entities = res.scalars().all()

        responses: List[CaseResponse] = []
        for ent in entities:
            resp = await _build_case_response(ent, db)
            responses.append(resp)

        return responses
    except Exception as err:
        logger.error(f"Vaka listeleme beklenmeyen hata: {err}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "message": f"Vaka listeleme işlemi sırasında hata oluştu: {str(err)}",
                "error_code": "CASE_LIST_FAILED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@router.post(
    "",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Yeni Adli Vaka Oluştur (fhdss.db)",
)
async def create_case(
    payload: CaseCreate,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Yeni bir adli vaka kaydı oluşturur ve tekil asenkron veritabanına kalıcı olarak yazar.
    """
    # 1. Bilimsel Kapsam ve Etik Doğrulama
    PsychologicalInferenceGuard.validate_request(
        user_query=f"{payload.name} {payload.description or ''} {payload.notes or ''}",
        endpoint_name="/cases [POST]",
    )

    now = datetime.now(timezone.utc).isoformat()

    # 2. Vaka Numarası Belirle
    case_number = payload.case_id
    if not case_number:
        count_stmt = select(func.count(CaseEntity.case_id))
        count_res = await db.execute(count_stmt)
        count = count_res.scalar() or 0
        year = datetime.now(timezone.utc).year
        case_number = f"CASE-{year}-{count + 1:03d}"

    # Çakışma kontrolü
    existing_stmt = select(CaseEntity).where(CaseEntity.case_id == case_number)
    existing_res = await db.execute(existing_stmt)
    if existing_res.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": f"'{case_number}' numaralı vaka zaten mevcut.",
                "error_code": "CASE_ALREADY_EXISTS",
                "case_id": case_number,
            },
        )

    examiner = current_user.full_name or current_user.username or payload.examiner_id or "BILIRKISI_UZMAN_01"
    exam_type_str = payload.examination_type.value if hasattr(payload.examination_type, "value") else str(payload.examination_type)
    status_str = payload.status.value if hasattr(payload.status, "value") else str(payload.status)
    stage_str = payload.current_stage.value if hasattr(payload.current_stage, "value") else str(payload.current_stage)

    # 3. CaseEntity Oluştur
    new_entity = CaseEntity(
        case_id=case_number,
        name=payload.name or f"Adli Vaka {case_number}",
        description=payload.description or "",
        examination_type=exam_type_str,
        reference_number=payload.reference_number,
        authority=payload.authority or "Cumhuriyet Başsavcılığı / Mahkeme",
        examiner_id=examiner,
        notes=payload.notes or "",
        status=status_str,
        current_stage=stage_str,
        state=IsolationState.PHASE_A_INIT.value,
        questioned_sample_id=None,
        created_at_utc=now,
        updated_at_utc=now,
    )

    db.add(new_entity)
    await db.commit()
    await db.refresh(new_entity)

    # 4. Orkestratör Başlat
    orch = get_case_orchestrator(case_number, examiner, IsolationState.PHASE_A_INIT.value)
    orch.reset()

    # 5. Audit Log
    log_audit_event(
        event_type="CASE_CREATED",
        action="CREATE_FORENSIC_CASE",
        examiner_id=examiner,
        details={
            "case_number": case_number,
            "name": payload.name,
            "examination_type": exam_type_str,
        },
    )

    return await _build_case_response(new_entity, db)


@router.get(
    "/{case_id}",
    response_model=CaseResponse,
    summary="Belirtilen Adli Vaka Detaylarını Getir",
)
async def get_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Vaka numarası ile vakayı sorgular ve tüm adli durumunu döner."""
    stmt = select(CaseEntity).where(CaseEntity.case_id == case_id, CaseEntity.is_deleted == False)
    res = await db.execute(stmt)
    entity = res.scalar_one_or_none()

    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": f"'{case_id}' numaralı adli vaka bulunamadı veya silinmiş.",
                "error_code": "CASE_NOT_FOUND",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    return await _build_case_response(entity, db)


@router.put(
    "/{case_id}",
    response_model=CaseResponse,
    summary="Adli Vaka Bilgilerini Güncelle",
)
async def update_case(
    case_id: str,
    payload: CaseUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Mevcut bir adli vakanın meta verilerini günceller."""
    stmt = select(CaseEntity).where(CaseEntity.case_id == case_id, CaseEntity.is_deleted == False)
    res = await db.execute(stmt)
    entity = res.scalar_one_or_none()

    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{case_id}' numaralı vaka güncellenmek üzere bulunamadı veya silinmiş.",
        )

    if payload.name is not None:
        entity.name = payload.name
    if payload.description is not None:
        entity.description = payload.description
    if payload.examination_type is not None:
        entity.examination_type = payload.examination_type.value if hasattr(payload.examination_type, "value") else str(payload.examination_type)
    if payload.reference_number is not None:
        entity.reference_number = payload.reference_number
    if payload.authority is not None:
        entity.authority = payload.authority
    if payload.notes is not None:
        entity.notes = payload.notes
    if payload.status is not None:
        entity.status = payload.status.value if hasattr(payload.status, "value") else str(payload.status)
    if payload.current_stage is not None:
        entity.current_stage = payload.current_stage.value if hasattr(payload.current_stage, "value") else str(payload.current_stage)

    now = datetime.now(timezone.utc).isoformat()
    entity.updated_at_utc = now
    await db.commit()
    await db.refresh(entity)

    log_audit_event(
        event_type="CASE_UPDATED",
        action="UPDATE_FORENSIC_CASE",
        examiner_id=entity.examiner_id,
        details={
            "case_number": case_id,
            "status": entity.status,
            "current_stage": entity.current_stage,
        },
    )

    return await _build_case_response(entity, db)


@router.delete(
    "/{case_id}",
    status_code=status.HTTP_200_OK,
    summary="Adli Vakayı ve İlişkili Tüm Verileri Sil (Soft-Delete & RBAC)",
)
async def delete_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(require_roles(["ADMIN", "SENIOR_REVIEWER"])),
):
    """
    Adli delil zincirini ve ISO 17025 izlenebilirliğini korumak için vakayı
    ve bağlı tüm delilleri kalıcı silmek yerine 'soft-delete' olarak arşivler.
    Yalnızca ADMIN ve SENIOR_REVIEWER rollerine izin verilir.
    """
    stmt = select(CaseEntity).where(CaseEntity.case_id == case_id, CaseEntity.is_deleted == False)
    res = await db.execute(stmt)
    entity = res.scalar_one_or_none()

    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "message": f"'{case_id}' numaralı vaka bulunamadı veya daha önce silinmiş.",
                "error_code": "CASE_NOT_FOUND",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    now_utc = datetime.now(timezone.utc).isoformat()
    entity.is_deleted = True
    entity.deleted_at_utc = now_utc
    entity.deleted_by = current_user.username
    entity.status = "DELETED"

    # Bağlı numuneleri de soft-delete olarak işaretle
    sample_stmt = select(SampleEntity).where(SampleEntity.case_id == case_id, SampleEntity.is_deleted == False)
    sample_res = await db.execute(sample_stmt)
    for s in sample_res.scalars().all():
        s.is_deleted = True
        s.deleted_at_utc = now_utc
        s.deleted_by = current_user.username

    await db.commit()

    # Orkestratör önbelleğinden sil
    if case_id in _ORCHESTRATORS:
        del _ORCHESTRATORS[case_id]

    log_audit_event(
        event_type="CASE_SOFT_DELETED",
        action="SOFT_DELETE_FORENSIC_CASE",
        examiner_id=current_user.username,
        details={"case_id": case_id, "deleted_by": current_user.username},
    )

    return {
        "status": "success",
        "message": f"'{case_id}' numaralı vaka ve ilişkili delilleri adli bütünlük korunarak başarıyla arşivlendi (soft-deleted).",
        "case_id": case_id,
    }


@router.post(
    "/{case_id}/samples",
    response_model=SampleUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Vakaya Adli Belge / Görsel Yükle (Disk & DB)",
)
async def upload_sample(
    case_id: str,
    file: UploadFile = File(...),
    sample_type: str = Form("questioned", description="'questioned' veya 'reference'"),
    sample_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Vakaya ait sorgulanan veya referans belgeyi yükler, SHA-256 dijital karmasını hesaplar ve kaydeder.
    """
    stmt = select(CaseEntity).where(CaseEntity.case_id == case_id)
    res = await db.execute(stmt)
    c_entity = res.scalar_one_or_none()

    contents = await file.read()
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Yüklenen dosya içeriği boş.",
        )

    # 1. Dosya Boyutu Kontrolü (Maksimum 50 MB)
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Dosya boyutu çok büyük ({len(contents) / (1024 * 1024):.1f} MB). Maksimum izin verilen boyut: 50 MB.",
        )

    # 2. Magic Byte ve MIME Doğrulaması (JPEG, PNG, TIFF)
    is_valid_magic = any(contents.startswith(magic) for magic, _ in ALLOWED_MAGIC_BYTES)
    if not is_valid_magic:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Güvenlik Koruması: Dosya içeriği geçerli bir adli görsel formatı (JPEG, PNG, TIFF) değil. Zararlı veya desteklenmeyen dosya türleri reddedildi.",
        )

    # SHA-256 Bütünlük Karması
    file_hash = hashlib.sha256(contents).hexdigest()
    file_size = len(contents)

    # OpenCV ile görüntü matrisini doğrula
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Yüklenen dosya geçerli bir görsel formatı değil (PNG, JPG vb. olmalıdır).",
        )

    # Path traversal engelleme ve dosya adı sanitizasyonu
    clean_filename = Path(file.filename or "sample.png").name
    clean_ext = Path(clean_filename).suffix.lower()
    if clean_ext not in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:
        clean_ext = ".png"

    safe_sample_prefix = "".join(c for c in (sample_id or f"{sample_type.upper()}_{hashlib.md5(contents).hexdigest()[:8]}") if c.isalnum() or c in ("-", "_"))
    assigned_sample_id = safe_sample_prefix or f"SAMPLE_{hashlib.md5(contents).hexdigest()[:8]}"

    # Görseli diske kaydet
    samples_dir = settings.BASE_DIR / "artifacts" / "samples" / case_id
    samples_dir.mkdir(parents=True, exist_ok=True)
    disk_path = samples_dir / f"{assigned_sample_id}{clean_ext}"
    with open(disk_path, "wb") as f:
        f.write(contents)

    now = datetime.now(timezone.utc).isoformat()
    examiner_name = c_entity.examiner_id if c_entity else current_user.username

    # CaseEntity yoksa otomatik oluştur
    if not c_entity:
        c_entity = CaseEntity(
            case_id=case_id,
            name=f"Adli Vaka {case_id}",
            description="Otomatik oluşturulan vaka",
            examination_type="SIGNATURE",
            authority="Cumhuriyet Başsavcılığı / Mahkeme",
            examiner_id=examiner_name,
            notes="",
            status="PHASE_A",
            current_stage="CASE_CREATED",
            state=IsolationState.PHASE_A_INIT.value,
            questioned_sample_id=None,
            created_at_utc=now,
            updated_at_utc=now,
        )
        db.add(c_entity)
        await db.flush()

    # SampleEntity ekle veya güncelle
    sample_stmt = select(SampleEntity).where(SampleEntity.sample_id == assigned_sample_id)
    sample_res = await db.execute(sample_stmt)
    existing_sample = sample_res.scalar_one_or_none()

    if existing_sample:
        existing_sample.filename = file.filename or assigned_sample_id
        existing_sample.sample_type = sample_type.lower()
        existing_sample.sha256_hash = file_hash
        existing_sample.file_size_bytes = file_size
        existing_sample.file_path = str(disk_path)
    else:
        new_sample = SampleEntity(
            sample_id=assigned_sample_id,
            case_id=case_id,
            filename=file.filename or assigned_sample_id,
            sample_type=sample_type.lower(),
            sha256_hash=file_hash,
            file_size_bytes=file_size,
            file_path=str(disk_path),
            created_at_utc=now,
        )
        db.add(new_sample)

    # Bilişsel İzolasyon Durumunu Güncelle
    orchestrator = get_case_orchestrator(case_id, examiner_name, c_entity.state)

    if sample_type.lower() == "questioned":
        c_entity.questioned_sample_id = assigned_sample_id
        try:
            orchestrator.start_phase_a(
                questioned_sample_id=assigned_sample_id,
                document_hash=file_hash,
                force_reset=True,
            )
            c_entity.state = orchestrator.state.value
            c_entity.current_stage = "EVALUATION"
        except IsolationViolationError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    else:
        if c_entity.questioned_sample_id:
            orchestrator.state = IsolationState.PHASE_A_FROZEN
            c_entity.state = IsolationState.PHASE_A_FROZEN.value
        else:
            orchestrator.state = IsolationState.PHASE_A_INIT
            c_entity.state = IsolationState.PHASE_A_INIT.value

    c_entity.updated_at_utc = now
    await db.commit()

    log_audit_event(
        event_type="SAMPLE_UPLOADED",
        action="UPLOAD_DOCUMENT",
        examiner_id=examiner_name,
        document_hash=file_hash,
        details={
            "case_id": case_id,
            "sample_id": assigned_sample_id,
            "sample_type": sample_type,
            "filename": file.filename,
            "size_bytes": file_size,
            "file_path": str(disk_path),
        },
    )

    return SampleUploadResponse(
        sample_id=assigned_sample_id,
        case_id=case_id,
        filename=file.filename or assigned_sample_id,
        sample_type=sample_type.lower(),
        sha256_hash=file_hash,
        file_size_bytes=file_size,
    )


@router.get(
    "/{case_id}/samples/{sample_id}/image",
    summary="Numune Görselini İndir / Önizle",
)
async def get_sample_image(
    case_id: str,
    sample_id: str,
    db: AsyncSession = Depends(get_db),
):
    """Kayıtlı adli numunenin görsel dosyasını streaming yanıt olarak döner."""
    stmt = select(SampleEntity).where(
        SampleEntity.case_id == case_id,
        SampleEntity.sample_id == sample_id,
        SampleEntity.is_deleted == False,
    )
    res = await db.execute(stmt)
    sample = res.scalar_one_or_none()
    if not sample or not sample.file_path or not os.path.exists(sample.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{sample_id}' numaralı numunenin görsel dosyası bulunamadı.",
        )

    media_type = "image/png"
    lower_path = sample.file_path.lower()
    if lower_path.endswith((".jpg", ".jpeg")):
        media_type = "image/jpeg"
    elif lower_path.endswith((".tif", ".tiff")):
        media_type = "image/tiff"

    return FileResponse(sample.file_path, media_type=media_type)


@router.delete(
    "/{case_id}/samples/{sample_id}",
    status_code=status.HTTP_200_OK,
    summary="Vakaya Ait Adli Belgeyi / Numuneyi Sil (Soft-Delete & RBAC)",
)
async def delete_sample(
    case_id: str,
    sample_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(require_roles(["ADMIN", "SENIOR_REVIEWER"])),
):
    """
    Belirtilen numuneyi soft-delete olarak işaretler, delil zinciri bütünlüğünü korur.
    Yalnızca ADMIN ve SENIOR_REVIEWER rollerine izin verilir.
    """
    case_stmt = select(CaseEntity).where(CaseEntity.case_id == case_id, CaseEntity.is_deleted == False)
    case_res = await db.execute(case_stmt)
    case_entity = case_res.scalar_one_or_none()

    sample_stmt = select(SampleEntity).where(
        SampleEntity.case_id == case_id,
        SampleEntity.sample_id == sample_id,
        SampleEntity.is_deleted == False,
    )
    sample_res = await db.execute(sample_stmt)
    sample_entity = sample_res.scalar_one_or_none()

    if not sample_entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{sample_id}' numaralı numune '{case_id}' vakasında bulunamadı veya daha önce silinmiş.",
        )

    now_utc = datetime.now(timezone.utc).isoformat()
    sample_entity.is_deleted = True
    sample_entity.deleted_at_utc = now_utc
    sample_entity.deleted_by = current_user.username

    sample_type = sample_entity.sample_type

    if case_entity:
        orchestrator = get_case_orchestrator(case_id, case_entity.examiner_id, case_entity.state)
        if sample_type == "questioned":
            case_entity.questioned_sample_id = None
            case_entity.state = IsolationState.PHASE_A_INIT.value
            case_entity.current_stage = "CASE_CREATED"
            orchestrator.reset()
        else:
            if case_entity.questioned_sample_id:
                orchestrator.state = IsolationState.PHASE_A_FROZEN
                case_entity.state = IsolationState.PHASE_A_FROZEN.value
            else:
                orchestrator.reset()
                case_entity.state = IsolationState.PHASE_A_INIT.value

        case_entity.updated_at_utc = now_utc

    await db.commit()

    log_audit_event(
        event_type="SAMPLE_SOFT_DELETED",
        action="SOFT_DELETE_DOCUMENT",
        examiner_id=current_user.username,
        details={
            "case_id": case_id,
            "sample_id": sample_id,
            "sample_type": sample_type,
            "deleted_by": current_user.username,
        },
    )

    return {
        "status": "success",
        "message": f"'{sample_id}' numaralı numune adli bütünlük korunarak arşivlendi (soft-deleted).",
        "case_id": case_id,
        "sample_id": sample_id,
    }
