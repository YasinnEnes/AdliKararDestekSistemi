"""
Adli Arşiv Paketi (.ZIP) API Uç Noktaları.
"""

from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import settings
from backend.app.core.security import get_current_active_user
from backend.app.db.session import get_db
from backend.app.models.entities import UserEntity
from backend.app.services.archive_service import forensic_archive_service

router = APIRouter()


@router.post(
    "/export/{case_id}",
    summary="Vaka İçin Güvenli Adli Arşiv Paketi (.ZIP) Üret",
)
async def export_case_archive(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Vakaya ait tüm delil görsellerini, analiz verilerini, mühürlü raporları
    ve hash-chained denetim izini manifest.json ile mühürleyip ZIP arşivi olarak sunar.
    """
    try:
        zip_path = await forensic_archive_service.export_case_archive(
            db=db,
            case_id=case_id,
            examiner_username=current_user.username,
        )
        return {
            "case_id": case_id,
            "filename": zip_path.name,
            "download_url": f"/api/v1/archive/download/{zip_path.name}",
            "size_bytes": zip_path.stat().st_size,
            "message": "Adli arşiv paketi başarıyla üretildi.",
        }
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Arşiv paketi üretilirken hata: {str(e)}")


@router.get(
    "/download/{filename}",
    summary="Üretilen Adli Arşiv Paketini İndir",
)
def download_archive(
    filename: str,
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Üretilmiş olan ZIP arşiv dosyasını güvenli ve yetkili şekilde indirir."""
    # Path traversal engelleme
    safe_name = Path(filename).name
    target_file = settings.BASE_DIR / "artifacts" / "archives" / safe_name

    if not target_file.exists() or not target_file.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Arşiv dosyası bulunamadı.")

    return FileResponse(
        path=str(target_file),
        media_type="application/zip",
        filename=safe_name,
    )


@router.post(
    "/verify",
    summary="Yüklenen Adli Arşiv Paketinin (.ZIP) Bütünlüğünü Doğrula",
)
async def verify_uploaded_archive(
    file: UploadFile = File(...),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Dışarıdan veya mahkemeden gelen bir .fhdss.zip veya .zip adli arşiv paketini açar,
    manifest.json içindeki hash özetlerini ve e-imzayı kontrol eder.
    Herhangi bir delil tahrifatı varsa raporlar.
    """
    if not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Yalnızca .zip formatındaki adli arşiv paketleri kabul edilir.")

    contents = await file.read()
    report = forensic_archive_service.verify_case_archive(contents)
    return report
