"""
Adli Denetim İzi ve Değiştirilemez Kriptografik Hash Zinciri (Hash-Chained Audit Trail) Servisi.
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.logger import log_audit_event, logger
from backend.app.models.entities import AuditLogEntity

GENESIS_HASH = "0" * 64


def compute_audit_hash(
    sequence_number: int,
    previous_hash: str,
    action: str,
    case_id: Optional[str],
    user_id: Optional[str],
    timestamp_utc: str,
    details_str: str,
) -> str:
    """Tekil bir denetim kaydının kriptografik SHA-256 hash'ini hesaplar."""
    raw = f"{sequence_number}|{previous_hash}|{action}|{case_id or ''}|{user_id or ''}|{timestamp_utc}|{details_str}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def record_audit_trail(
    db: AsyncSession,
    action: str,
    case_id: Optional[str] = None,
    user_id: Optional[str] = "SYSTEM",
    details: Optional[Dict[str, Any]] = None,
) -> AuditLogEntity:
    """
    Kriptografik olarak önceki kaydın hash'ine bağlanan (blockchain-benzeri)
    değiştirilemez yeni bir denetim kaydı oluşturur ve kaydeder.
    """
    # En son kaydı bul
    stmt = select(AuditLogEntity).order_by(AuditLogEntity.sequence_number.desc()).limit(1)
    res = await db.execute(stmt)
    last_record = res.scalar_one_or_none()

    if last_record and last_record.sequence_number is not None:
        seq_num = last_record.sequence_number + 1
        prev_hash = last_record.current_hash or GENESIS_HASH
    else:
        seq_num = 1
        prev_hash = GENESIS_HASH

    now_iso = datetime.now(timezone.utc).isoformat()
    details_clean = details or {}
    details_json = json.dumps(details_clean, sort_keys=True, default=str)

    curr_hash = compute_audit_hash(
        sequence_number=seq_num,
        previous_hash=prev_hash,
        action=action,
        case_id=case_id,
        user_id=user_id,
        timestamp_utc=now_iso,
        details_str=details_json,
    )

    audit_entity = AuditLogEntity(
        sequence_number=seq_num,
        case_id=case_id,
        event_type="AUDIT_CHAIN_RECORD",
        examiner_id=user_id,
        action=action,
        details_json=details_clean,
        created_at_utc=now_iso,
        previous_hash=prev_hash,
        sha256_hash=curr_hash,
    )
    db.add(audit_entity)
    await db.flush()

    # Dosya tabanlı JSONL denetim günlüğüne de paralel yaz
    log_audit_event(
        event_type="AUDIT_CHAIN_BLOCK",
        action=action,
        examiner_id=user_id,
        details={
            "sequence_number": seq_num,
            "previous_hash": prev_hash,
            "current_hash": curr_hash,
            "case_id": case_id,
            "inner_details": details_clean,
        },
    )

    logger.info(f"Adli denetim zincirine blok eklendi: [Seq: {seq_num}, Eylem: {action}, Hash: {curr_hash[:10]}...]")
    return audit_entity


async def verify_audit_hash_chain(
    db: AsyncSession,
    case_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Veritabanındaki tüm denetim zincirinin (veya belirli bir vakaya ait olanların)
    kriptografik bütünlüğünü baştan sona yeniden hesaplayarak doğrular.
    Herhangi bir satır değiştirilmiş veya silinmişse zincir kırılır ve tespit edilir.
    """
    query = select(AuditLogEntity).order_by(AuditLogEntity.sequence_number.asc())
    if case_id:
        query = query.where(AuditLogEntity.case_id == case_id)

    res = await db.execute(query)
    records = res.scalars().all()

    if not records:
        return {
            "is_valid": True,
            "total_records": 0,
            "verified_records": 0,
            "message": "Doğrulanacak denetim kaydı bulunamadı.",
            "broken_sequence": None,
        }

    expected_prev = GENESIS_HASH
    verified_count = 0

    for idx, rec in enumerate(records):
        # 1. Önceki hash bağıntısı kontrolü (Sadece tüm zincir kontrol ediliyorsa veya ilk kayıt için)
        if not case_id and rec.previous_hash != expected_prev:
            return {
                "is_valid": False,
                "total_records": len(records),
                "verified_records": verified_count,
                "broken_sequence": rec.sequence_number,
                "record_id": rec.id,
                "message": (
                    f"Denetim Zinciri Kırıldı! Sıra #{rec.sequence_number} kaydının 'previous_hash' değeri "
                    f"({rec.previous_hash[:12]}...) beklenen ({expected_prev[:12]}...) ile eşleşmiyor."
                ),
            }

        # 2. Mevcut hash özetini yeniden hesapla
        details_val = json.dumps(rec.details_json or {}, sort_keys=True, default=str)
        recalculated_hash = compute_audit_hash(
            sequence_number=rec.sequence_number,
            previous_hash=rec.previous_hash,
            action=rec.action,
            case_id=rec.case_id,
            user_id=rec.examiner_id,
            timestamp_utc=rec.created_at_utc,
            details_str=details_val,
        )

        if rec.sha256_hash != recalculated_hash:
            return {
                "is_valid": False,
                "total_records": len(records),
                "verified_records": verified_count,
                "broken_sequence": rec.sequence_number,
                "record_id": rec.id,
                "message": (
                    f"Veri Bütünlüğü İhlali (Tampering Detected)! Sıra #{rec.sequence_number} "
                    f"kaydı değiştirilmiş. Kayıtlı Hash: {rec.sha256_hash[:12]}..., Hesaplanan: {recalculated_hash[:12]}..."
                ),
            }

        expected_prev = rec.sha256_hash
        verified_count += 1

    return {
        "is_valid": True,
        "total_records": len(records),
        "verified_records": verified_count,
        "message": f"Kriptografik Hash Zinciri Tam ve Sağlam! Toplam {verified_count} adli kayıt doğrulandı.",
        "latest_chain_hash": expected_prev,
    }
