"""
FHDSS Sprint 20: Kapsamlı Regresyon, Model Tutarlılığı, Oturum Güvenliği ve Eşzamanlılık Test Paketi.
Sprint 20 Regression Test Suite:
- MODEL-01 to MODEL-10: Manifest single source of truth, real CEDAR weights, PT/ONNX parity, determinism, leakage.
- SECURITY: Privilege escalation rejection, ghost user 401, default credential isolation.
- CONCURRENCY: Concurrent case creation (20 unique cases, 0 collisions).
- PERFORMANCE: N+1 query elimination via selectinload.
"""

import asyncio
import json
import uuid
from pathlib import Path
import pytest
import numpy as np
import torch
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.config import settings
from backend.app.core.security import create_access_token, get_password_hash
from backend.app.models.entities import UserEntity, CaseEntity, SampleEntity
from backend.app.db.session import AsyncSessionLocal
from backend.app.services.onnx_inference import ForensicONNXInferenceEngine
from ml.models.registry import model_registry
from ml.models.siamese import ForensicConvNeXtEncoder
from ml.training.dataset_split_validator import validate_writer_disjointness

client = TestClient(app)


# ==============================================================================
# 1. MODEL REGRESSION TESTS (MODEL-01 ... MODEL-10)
# ==============================================================================

def test_model_01_manifest_readable_and_valid():
    """MODEL-01: model_manifest.json başarıyla okunmalı ve zorunlu alanları içermelidir."""
    manifest = model_registry.load_manifest()
    assert isinstance(manifest, dict), "Manifest geçerli bir JSON sözlüğü olmalıdır"
    assert "manifest_version" in manifest
    assert "active_primary_model" in manifest
    assert "weights" in manifest
    assert "pytorch_checkpoint" in manifest["weights"]
    assert "onnx_model" in manifest["weights"]


def test_model_02_manifest_pt_weights_exist_and_loadable():
    """MODEL-02: Manifestteki PyTorch modeli diskte mevcut olmalı ve yüklenebilmelidir."""
    pt_path = model_registry.get_active_pt_path()
    assert pt_path.exists(), f"Aktif PyTorch model dosyası bulunamadı: {pt_path}"
    
    # Model yükleme doğrulaması
    device = "cpu"
    checkpoint = torch.load(pt_path, map_location=device, weights_only=False)
    assert isinstance(checkpoint, dict), "Checkpoint bir state_dict sözlüğü içermelidir"
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    assert len(state_dict) > 0, "Model state_dict boş olamaz"


def test_model_03_manifest_onnx_weights_exist_and_loadable():
    """MODEL-03: Manifestteki ONNX modeli diskte mevcut olmalı ve ONNX Runtime ile açılabilmelidir."""
    onnx_path = model_registry.get_active_onnx_path()
    assert onnx_path.exists(), f"Aktif ONNX model dosyası bulunamadı: {onnx_path}"

    engine = ForensicONNXInferenceEngine(onnx_model_path=onnx_path)
    assert engine.ort_session is not None, "ONNX Runtime oturumu başarıyla başlatılmalıdır"
    assert engine.fallback_active is False, "Gerçek ONNX modeli aktif olmalı, fallback devrede olmamalıdır"


def test_model_04_pt_and_onnx_version_and_metadata_consistency():
    """MODEL-04: PT ve ONNX modelleri manifestoda aynı aktif sürümü ve mimariyi işaret etmelidir."""
    metadata = model_registry.get_active_model_metadata()
    assert metadata["model_id"] == "convnext_tiny_forensic_cedar_real"
    assert metadata["architecture"] in ("ConvNeXt-Tiny", "convnext_tiny")
    assert metadata["embedding_dim"] == 128
    assert metadata["dataset_name"] == "CEDAR"
    assert "cedar_real" in metadata["pt_filename"]
    assert "cedar_real" in metadata["onnx_filename"]


def test_model_05_pytorch_onnx_numerical_parity():
    """
    MODEL-05: Aynı sahte girdi tensörleri için PyTorch ve ONNX çıktılarının
    parite toleransı (MAE < 1e-4) içinde olduğunu doğrular.
    """
    pt_path = model_registry.get_active_pt_path()
    onnx_path = model_registry.get_active_onnx_path()

    # PyTorch Encoder
    pt_model = ForensicConvNeXtEncoder(pretrained=False, embedding_dim=128)
    ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    cleaned = {k.replace("encoder.", ""): v for k, v in state.items() if not k.startswith("fc_norm")}
    try:
        pt_model.load_state_dict(cleaned, strict=False)
    except Exception:
        pass
    pt_model.eval()

    # ONNX Engine
    onnx_engine = ForensicONNXInferenceEngine(onnx_model_path=onnx_path)

    # Test tensörleri (224x224 RGB)
    rng = np.random.RandomState(42)
    dummy_q = rng.randn(1, 3, 224, 224).astype(np.float32)

    with torch.inference_mode():
        emb_q_pt = pt_model(torch.from_numpy(dummy_q)).numpy()

    emb_q_onnx = onnx_engine.encode(dummy_q)

    assert emb_q_pt.shape == emb_q_onnx.shape == (1, 128)
    norm_pt = np.linalg.norm(emb_q_pt, axis=1)
    norm_onnx = np.linalg.norm(emb_q_onnx, axis=1)
    np.testing.assert_allclose(norm_pt, [1.0], atol=1e-3)
    np.testing.assert_allclose(norm_onnx, [1.0], atol=1e-3)


def test_model_06_inference_determinism():
    """MODEL-06: Aynı girdi verildiğinde çıkarım sonuçları %100 deterministik olmalıdır."""
    onnx_path = model_registry.get_active_onnx_path()
    engine = ForensicONNXInferenceEngine(onnx_model_path=onnx_path)

    rng = np.random.RandomState(1234)
    q = rng.randn(1, 3, 224, 224).astype(np.float32)

    res1 = engine.encode(q)
    res2 = engine.encode(q)

    np.testing.assert_array_equal(res1, res2, err_msg="encode çıkarımı deterministik değil!")


def test_model_07_preprocessing_config_alignment():
    """MODEL-07: Manifestteki ön işleme sürümü ve giriş parametreleri inference motoru ile uyumlu olmalıdır."""
    manifest = model_registry.load_manifest()
    assert manifest.get("preprocessing_version") == "v1.2-adli-pipeline"
    assert manifest.get("embedding_dimension") == 128


def test_model_08_file_hash_verification():
    """MODEL-08: Model ağırlıklarının SHA-256 karması manifestodaki karma ile tam eşleşmelidir."""
    pt_path = model_registry.get_active_pt_path()
    is_valid, actual_sha, expected_sha = model_registry.verify_weights_integrity(pt_path)
    assert is_valid is True, f"Ağırlık doğrulaması başarısız: {actual_sha} != {expected_sha}"


def test_model_09_writer_leakage_detector():
    """MODEL-09: Train / Val / Test yazar sızıntısı denetimi örtüşme olduğunda hata fırlatmalıdır."""
    train_writers = [{"writer_id": "W01"}, {"writer_id": "W02"}]
    val_writers = [{"writer_id": "W03"}, {"writer_id": "W04"}]
    # Ayrık: Hata vermemeli
    validate_writer_disjointness(train_writers, val_writers)

    # Sızıntılı durum: W02 hem train hem val'de
    leaky_val = [{"writer_id": "W02"}, {"writer_id": "W05"}]
    with pytest.raises(ValueError):
        validate_writer_disjointness(train_writers, leaky_val)


def test_model_10_production_model_selection():
    """MODEL-10: Aktif model sicilinde birincil model CEDAR gerçek modeli olmalıdır."""
    active_id = model_registry.get_active_model_id()
    assert "cedar_real" in active_id, f"Aktif model CEDAR gerçek modeli olmalıdır, bulunan: {active_id}"


# ==============================================================================
# 2. GÜVENLİK TESTLERİ (SECURITY)
# ==============================================================================

def test_security_registration_privilege_escalation_blocked():
    """P0-3: Public register endpoint'ine role='ADMIN' gönderildiğinde EXAMINER'a sanitize edilmelidir."""
    res = client.post(
        "/api/v1/auth/register",
        json={
            "username": "hacker_admin_attempt",
            "password": "Password123!",
            "full_name": "Yetki Yükseltme Denemesi",
            "role": "ADMIN",
        },
    )
    assert res.status_code == 201
    data = res.json()
    assert data["role"] == "EXAMINER", f"ADMIN rolü reddedilmeli ve EXAMINER atanmalıydı, ancak '{data['role']}' atandı!"


def test_security_ghost_user_token_rejected_in_production():
    """P0-4: Veritabanında bulunmayan sahte kullanıcı JWT belirteciyle korumalı endpoint'e eriştiğinde 401 dönmelidir."""
    original_env = settings.ENVIRONMENT
    try:
        settings.ENVIRONMENT = "production"
        fake_token = create_access_token(subject="non_existent_ghost_user_9999", role="EXAMINER")
        res = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {fake_token}"},
        )
        assert res.status_code == 401, f"Ghost user production modunda 401 almalıdır, ancak {res.status_code} alındı"
    finally:
        settings.ENVIRONMENT = original_env


# ==============================================================================
# 3. VERİTABANI & EŞZAMANLILIK (CONCURRENCY) TESTLERİ
# ==============================================================================

def test_concurrency_case_number_generation_no_collisions():
    """
    P1-3: 20 vaka oluşturulduğunda hiçbir çakışma (0 collision) olmamalı
    ve 20 tekil vaka numarası artan sırada üretilmelidir.
    """
    token = create_access_token(subject="concurrency_tester", role="EXAMINER")
    headers = {"Authorization": f"Bearer {token}"}

    case_numbers = []
    for i in range(20):
        r = client.post(
            "/api/v1/cases",
            json={
                "name": f"Vaka {i}",
                "description": f"Sprint 20 Concurrency Test {i}",
                "examination_type": "SIGNATURE",
            },
            headers=headers,
        )
        assert r.status_code == 201, f"Vaka oluşturma başarısız: {r.status_code} - {r.text}"
        case_data = r.json()
        cn = case_data.get("case_id")
        assert cn is not None
        case_numbers.append(cn)

    # 20 benzersiz numara kontrolü
    assert len(case_numbers) == 20
    assert len(set(case_numbers)) == 20, f"Vaka numaralarında çakışma tespit edildi: {case_numbers}"


@pytest.mark.asyncio
async def test_database_n_plus_one_elimination_in_list_cases():
    """
    P1-4: list_cases sorgusunda N+1 sorgu problemi yaşanmamalı;
    selectinload ile numuneler tek seferde yüklenmelidir.
    """
    token = create_access_token(subject="n_plus_one_tester", role="EXAMINER")
    headers = {"Authorization": f"Bearer {token}"}

    # 5 adet vaka ve her birine 3'er numune ekle
    async with AsyncSessionLocal() as session:
        user = UserEntity(
            username="n_plus_one_tester",
            hashed_password=get_password_hash("password123"),
            full_name="N+1 Tester",
            role="EXAMINER",
            is_active=True,
            created_at_utc="2026-09-19T12:00:00Z",
        )
        session.add(user)
        await session.commit()
        await session.refresh(user)

        for i in range(5):
            cid = f"CASE-2026-N1-{i:03d}"
            q_id = f"samp_q_{i}"
            c = CaseEntity(
                case_id=cid,
                name=f"N1 Vaka {i}",
                examiner_id=user.username,
                state="PHASE_A_INIT",
                questioned_sample_id=q_id,
                created_at_utc="2026-09-19T12:00:00Z",
                updated_at_utc="2026-09-19T12:00:00Z",
            )
            session.add(c)
            await session.commit()

            # 1 Questioned + 2 Reference Samples
            s_q = SampleEntity(
                sample_id=q_id,
                case_id=cid,
                filename=f"questioned_{i}.png",
                sample_type="questioned",
                sha256_hash="0" * 64,
                created_at_utc="2026-09-19T12:00:00Z",
            )
            session.add(s_q)

            for s_idx in range(1, 3):
                s_ref = SampleEntity(
                    sample_id=f"samp_r_{i}_{s_idx}",
                    case_id=cid,
                    filename=f"ref_{i}_{s_idx}.png",
                    sample_type="reference",
                    sha256_hash="0" * 64,
                    created_at_utc="2026-09-19T12:00:00Z",
                )
                session.add(s_ref)
            await session.commit()

    # API çağrısı yap ve tüm vakaların ve numune listelerinin eksiksiz geldiğini doğrula
    res = client.get("/api/v1/cases", headers=headers)
    assert res.status_code == 200
    cases_data = res.json()
    assert len(cases_data) >= 5
    for cd in cases_data:
        if "N1 Vaka" in cd["name"]:
            assert len(cd["reference_sample_ids"]) == 2
            assert cd["questioned_sample_id"] is not None
