"""
Güvenli Adli Arşiv Paketi (.ZIP) Dışa Aktarma ve Kriptografik Bütünlük Doğrulama Servisi.
Forensic Case Archive (.ZIP) Export, Manifest Generation & Tamper Detection.
"""

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.logger import logger
from backend.app.models.entities import AnalysisResultEntity, AuditLogEntity, CaseEntity, SampleEntity
from backend.app.services.legal_signature_service import default_signature_provider, default_timestamp_provider


class ForensicArchiveService:
    """Adli vaka dosyalarını kriptografik manifest ve e-imza mührü ile paketleyen ve doğrulayan servis."""

    def __init__(self):
        self.archives_dir = settings.BASE_DIR / "artifacts" / "archives"
        self.archives_dir.mkdir(parents=True, exist_ok=True)

    async def export_case_archive(
        self,
        db: AsyncSession,
        case_id: str,
        examiner_username: str = "EXAMINER",
    ) -> Path:
        """
        Vakaya ait tüm delilleri, raporları, denetim izini ve analiz kayıtlarını
        manifest.json içeren mühürlü bir ZIP arşivine dönüştürür.
        """
        stmt_case = select(CaseEntity).where(CaseEntity.case_id == case_id)
        res_c = await db.execute(stmt_case)
        case = res_c.scalar_one_or_none()
        if not case:
            raise ValueError(f"'{case_id}' vakası bulunamadı.")

        # Numuneler
        stmt_s = select(SampleEntity).where(SampleEntity.sample_id.in_(
            select(SampleEntity.sample_id).where(SampleEntity.case_id == case_id)
        ))
        res_s = await db.execute(stmt_s)
        samples = res_s.scalars().all()

        # Analiz Sonuçları
        stmt_a = select(AnalysisResultEntity).where(AnalysisResultEntity.case_id == case_id)
        res_a = await db.execute(stmt_a)
        analyses = res_a.scalars().all()

        # Denetim İzi
        stmt_l = select(AuditLogEntity).where(AuditLogEntity.case_id == case_id).order_by(AuditLogEntity.sequence_number.asc())
        res_l = await db.execute(stmt_l)
        audit_logs = res_l.scalars().all()

        now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        sanitized_case_id = case_id.replace("/", "_").replace("\\", "_")
        zip_filename = f"forensic_case_{sanitized_case_id}_{now_str}.zip"
        zip_path = self.archives_dir / zip_filename

        file_manifest: List[Dict[str, str]] = []

        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # 1. /case/case_metadata.json
            case_dict = {
                "case_id": case.case_id,
                "name": case.name,
                "description": case.description,
                "authority": case.authority,
                "examiner_id": case.examiner_id,
                "status": case.status,
                "state": case.state,
                "merkle_root_hash": case.merkle_root_hash,
                "similarity_score": case.similarity_score,
                "verbal_scale": case.verbal_scale,
                "created_at_utc": case.created_at_utc,
            }
            case_bytes = json.dumps(case_dict, indent=2, default=str).encode("utf-8")
            zf.writestr("case/case_metadata.json", case_bytes)
            file_manifest.append({
                "path": "case/case_metadata.json",
                "sha256": hashlib.sha256(case_bytes).hexdigest(),
                "size_bytes": len(case_bytes),
            })

            # 2. /evidence/
            for s in samples:
                p = Path(s.file_path)
                if not p.is_absolute():
                    p = settings.BASE_DIR / p
                if p.exists():
                    img_bytes = p.read_bytes()
                    arcname = f"evidence/{s.sample_type.lower()}_{s.sample_id}_{p.name}"
                    zf.writestr(arcname, img_bytes)
                    file_manifest.append({
                        "path": arcname,
                        "sha256": hashlib.sha256(img_bytes).hexdigest(),
                        "size_bytes": len(img_bytes),
                    })

            # 3. /analysis/
            analysis_data = [
                {
                    "id": a.id,
                    "created_at_utc": a.created_at_utc,
                    "audit_checksum_sha256": a.audit_checksum_sha256,
                    "merkle_root_hash": a.merkle_root_hash,
                    "suitability": a.suitability_json,
                    "complexity": a.complexity_json,
                    "comparison": a.comparison_json,
                    "fusion": a.fusion_json,
                    "hypothesis": a.hypothesis_json,
                }
                for a in analyses
            ]
            analysis_bytes = json.dumps(analysis_data, indent=2, default=str).encode("utf-8")
            zf.writestr("analysis/analysis_results.json", analysis_bytes)
            file_manifest.append({
                "path": "analysis/analysis_results.json",
                "sha256": hashlib.sha256(analysis_bytes).hexdigest(),
                "size_bytes": len(analysis_bytes),
            })

            # 4. /audit/
            audit_data = [
                {
                    "sequence_number": log.sequence_number,
                    "action": log.action,
                    "user_id": log.user_id,
                    "timestamp_utc": log.timestamp_utc,
                    "previous_hash": log.previous_hash,
                    "current_hash": log.current_hash,
                    "details": log.details,
                }
                for log in audit_logs
            ]
            audit_bytes = json.dumps(audit_data, indent=2, default=str).encode("utf-8")
            zf.writestr("audit/audit_chain.json", audit_bytes)
            file_manifest.append({
                "path": "audit/audit_chain.json",
                "sha256": hashlib.sha256(audit_bytes).hexdigest(),
                "size_bytes": len(audit_bytes),
            })

            # 5. /reports/
            reports_dir = settings.BASE_DIR / "artifacts" / "reports"
            if reports_dir.exists():
                for rep in reports_dir.glob(f"*{sanitized_case_id}*"):
                    rep_bytes = rep.read_bytes()
                    arcname = f"reports/{rep.name}"
                    zf.writestr(arcname, rep_bytes)
                    file_manifest.append({
                        "path": arcname,
                        "sha256": hashlib.sha256(rep_bytes).hexdigest(),
                        "size_bytes": len(rep_bytes),
                    })

            # 6. Manifest Oluşturma ve E-İmza / Zaman Damgası ile Mühürleme
            raw_manifest_data = {
                "archive_format_version": "FHDSS-FORENSIC-ZIP-V1",
                "case_id": case.case_id,
                "case_name": case.name,
                "authority": case.authority or "Cumhuriyet Başsavcılığı",
                "merkle_root_hash": case.merkle_root_hash or "",
                "exported_by": examiner_username,
                "exported_at_utc": datetime.now(timezone.utc).isoformat(),
                "files": file_manifest,
            }
            manifest_json_str = json.dumps(raw_manifest_data, indent=2, sort_keys=True)
            manifest_bytes = manifest_json_str.encode("utf-8")

            # Mühürleme
            sig = default_signature_provider.sign_document(
                data_bytes=manifest_bytes,
                signer_name=examiner_username,
            )
            ts = default_timestamp_provider.create_timestamp(manifest_bytes)

            complete_manifest = {
                **raw_manifest_data,
                "electronic_signature": sig,
                "timestamp_authority": ts,
            }

            zf.writestr("manifest.json", json.dumps(complete_manifest, indent=2).encode("utf-8"))

        logger.info(f"Adli arşiv paketi başarıyla üretildi: {zip_path.name}")
        return zip_path

    def verify_case_archive(self, zip_source: Union[Path, bytes]) -> Dict[str, Any]:
        """
        Yüklenen veya belirtilen bir adli arşiv ZIP dosyasını manifest.json üzerinden
        kriptografik olarak denetler, değiştirilmiş veya eksik dosya olup olmadığını doğrular.
        """
        try:
            if isinstance(zip_source, (str, Path)):
                zf = zipfile.ZipFile(zip_source, "r")
            else:
                zf = zipfile.ZipFile(io.BytesIO(zip_source), "r")

            with zf:
                # 1. Zip Bomb ve Path Traversal Güvenlik Kontrolleri
                MAX_FILE_COUNT = 500
                MAX_UNCOMPRESSED_SIZE = 200 * 1024 * 1024  # 200 MB

                infolist = zf.infolist()
                if len(infolist) > MAX_FILE_COUNT:
                    return {
                        "is_valid": False,
                        "error": f"Güvenlik İhlali: Arşiv içerisindeki dosya sayısı ({len(infolist)}) izin verilen sınırı ({MAX_FILE_COUNT}) aşıyor.",
                    }

                total_uncompressed = 0
                for info in infolist:
                    fname = info.filename
                    if ".." in fname or fname.startswith("/") or fname.startswith("\\") or (len(fname) > 1 and fname[1] == ":"):
                        return {
                            "is_valid": False,
                            "error": f"Güvenlik İhlali (Path Traversal): Şüpheli dosya yolu tespit edildi: '{fname}'",
                        }
                    total_uncompressed += info.file_size
                    if total_uncompressed > MAX_UNCOMPRESSED_SIZE:
                        return {
                            "is_valid": False,
                            "error": f"Güvenlik İhlali (Decompression Bomb): Açılmış toplam içerik boyutu 200 MB sınırını aşıyor.",
                        }

                if "manifest.json" not in zf.namelist():
                    return {
                        "is_valid": False,
                        "error": "Geçersiz adli paket: 'manifest.json' kök dizinde bulunamadı.",
                    }

                manifest_bytes = zf.read("manifest.json")
                manifest = json.loads(manifest_bytes.decode("utf-8"))

                files_list = manifest.get("files", [])
                tampered_files = []
                verified_files = []

                for item in files_list:
                    file_path = item.get("path")
                    expected_sha = item.get("sha256")

                    if file_path not in zf.namelist():
                        tampered_files.append({
                            "path": file_path,
                            "error": "Dosya arşiv içerisinde eksik!",
                        })
                        continue

                    actual_bytes = zf.read(file_path)
                    actual_sha = hashlib.sha256(actual_bytes).hexdigest()

                    if actual_sha != expected_sha:
                        tampered_files.append({
                            "path": file_path,
                            "error": f"SHA-256 uyuşmazlığı! Beklenen: {expected_sha[:12]}..., Bulunan: {actual_sha[:12]}...",
                        })
                    else:
                        verified_files.append(file_path)

                is_valid = len(tampered_files) == 0

                return {
                    "is_valid": is_valid,
                    "case_id": manifest.get("case_id"),
                    "case_name": manifest.get("case_name"),
                    "merkle_root_hash": manifest.get("merkle_root_hash"),
                    "exported_by": manifest.get("exported_by"),
                    "exported_at_utc": manifest.get("exported_at_utc"),
                    "total_files": len(files_list),
                    "verified_files_count": len(verified_files),
                    "tampered_files": tampered_files,
                    "electronic_signature_present": "electronic_signature" in manifest,
                    "timestamp_present": "timestamp_authority" in manifest,
                    "message": (
                        "Adli arşiv paketi tam ve bozulmamış olarak doğrulandı."
                        if is_valid
                        else f"Bütünlük ihlali! {len(tampered_files)} dosyada manipülasyon tespit edildi."
                    ),
                }

        except Exception as e:
            return {
                "is_valid": False,
                "error": f"Arşiv açılırken veya işlenirken hata oluştu: {str(e)}",
            }


forensic_archive_service = ForensicArchiveService()
