from datetime import datetime, timezone
from sqlalchemy import func, select
from backend.app.db.base import Base
from backend.app.db.session import engine, AsyncSessionLocal
from backend.app.core.logger import logger
from backend.app.models.entities import CaseEntity, UserEntity
import backend.app.models.entities  # Modelleri Base'e kaydetmek için import et


async def init_db():
    """Tüm veritabanı tablolarını asenkron olarak oluşturur ve başlangıç kullanıcılarını tohumlar."""
    from backend.app.core.security import get_password_hash

    logger.info("Veritabanı tabloları başlatılıyor...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

        def _migrate_columns(sync_conn):
            from sqlalchemy import text
            try:
                res = sync_conn.execute(text("PRAGMA table_info(audit_logs)")).fetchall()
                existing_cols = [r[1] for r in res]
                if "previous_hash" not in existing_cols:
                    sync_conn.execute(text("ALTER TABLE audit_logs ADD COLUMN previous_hash VARCHAR(64)"))
                if "sequence_number" not in existing_cols:
                    sync_conn.execute(text("ALTER TABLE audit_logs ADD COLUMN sequence_number INTEGER"))
            except Exception as e:
                logger.warning(f"Migrasyon kontrolü uyarısı: {e}")

        await conn.run_sync(_migrate_columns)
    logger.info("Veritabanı tabloları başarıyla hazırlandı.")

    # Başlangıç Adli Uzman ve Yönetici Kullanıcılarını Tohumla
    try:
        async with AsyncSessionLocal() as session:
            stmt = select(UserEntity).where(UserEntity.username == "admin")
            res = await session.execute(stmt)
            admin_user = res.scalar_one_or_none()
            if not admin_user:
                admin_user = UserEntity(
                    username="admin",
                    hashed_password=get_password_hash("password123"),
                    full_name="Baş Bilirkişi / Sistem Yöneticisi",
                    role="ADMIN",
                    is_active=True,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                )
                session.add(admin_user)

            stmt = select(UserEntity).where(UserEntity.username == "bilirkisi_uzman")
            res = await session.execute(stmt)
            examiner_user = res.scalar_one_or_none()
            if not examiner_user:
                examiner_user = UserEntity(
                    username="bilirkisi_uzman",
                    hashed_password=get_password_hash("password123"),
                    full_name="Bilirkişi Adli Uzman",
                    role="EXAMINER",
                    is_active=True,
                    created_at_utc=datetime.now(timezone.utc).isoformat(),
                )
                session.add(examiner_user)

            await session.commit()
            logger.info("Başlangıç kullanıcıları (admin, bilirkisi_uzman) kontrol edildi / tohumlandı.")

            # Başlangıç Örnek Adli Vakalarını Tohumla (Eğer veritabanı boşsa)
            case_count_stmt = select(func.count()).select_from(CaseEntity)
            case_count_res = await session.execute(case_count_stmt)
            case_count = case_count_res.scalar() or 0

            if case_count == 0:
                now_str = datetime.now(timezone.utc).isoformat()
                demo_case_1 = CaseEntity(
                    case_id="VAKA-2026-001",
                    name="[ÖRNEK VAKA 1] Ankara 4. Ağır Ceza Mahkemesi - Çek ve İmza Sahteciliği İncelemesi",
                    description="[TEST/ÖRNEK DOSYA] Müşteki şirketin keşide ettiği iddia edilen 250.000 TL bedelli çek üzerindeki borçlu imzasının incelenmesi ve aidiyet tespiti.",
                    examination_type="SIGNATURE",
                    reference_number="2026/142-E",
                    authority="Ankara 4. Ağır Ceza Mahkemesi",
                    examiner_id="bilirkisi_uzman",
                    suspect_name="Ahmet Yılmaz",
                    notes="Bu dosya eğitim ve demonstrasyon amaçlı örnek vakadır. İnceleme başlatmak için lütfen numune görseli yükleyiniz.",
                    status="IN_PROGRESS",
                    current_stage="CASE_CREATED",
                    state="PHASE_A_INIT",
                    created_at_utc=now_str,
                    updated_at_utc=now_str,
                )
                demo_case_2 = CaseEntity(
                    case_id="VAKA-2026-002",
                    name="[ÖRNEK VAKA 2] İstanbul 9. Asliye Hukuk Mahkemesi - Vasiyetname El Yazısı İncelemesi",
                    description="[TEST/ÖRNEK DOSYA] El yazısı ile tanzim edilmiş vasiyetname metninin müteveffa şahsın mukayese yazı ve rakamları ile mikroskobik morfolojik karşılaştırması.",
                    examination_type="HANDWRITING",
                    reference_number="2026/88-E",
                    authority="İstanbul 9. Asliye Hukuk Mahkemesi",
                    examiner_id="bilirkisi_uzman",
                    suspect_name="Mehmet Demir",
                    notes="Bu dosya eğitim ve demonstrasyon amaçlı örnek vakadır. İnceleme başlatmak için lütfen numune görseli yükleyiniz.",
                    status="IN_PROGRESS",
                    current_stage="CASE_CREATED",
                    state="PHASE_A_INIT",
                    created_at_utc=now_str,
                    updated_at_utc=now_str,
                )
                session.add_all([demo_case_1, demo_case_2])
                await session.commit()
                logger.info("Başlangıç örnek adli vakaları ([ÖRNEK VAKA 1], [ÖRNEK VAKA 2]) tohumlandı.")
    except Exception as e:
        logger.warning(f"Kullanıcı / Vaka tohumlama sırasında uyarı: {e}")
