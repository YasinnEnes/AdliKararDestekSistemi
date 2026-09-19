"""
FHDSS (Adli Belge ve El Yazısı Karar Destek Sistemi)
Sprint 14 Doğrulama Test Paketi:
1. Tekilleştirilmiş Asenkron Veritabanı ve CRUD
2. Kimlik Doğrulama, JWT ve RBAC Rolleri
3. Tam Sayfa Belge ve İmza Segmentasyonu (cv/segmentation/)
4. Sağlam PDF Raporlama (ReportLab Fallback)
5. Arşiv Veri Setleri ETL ve Sıfır Sızıntılı Bölütleme
"""

import io
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.core.security import create_access_token, decode_access_token, get_password_hash, verify_password
from cv.preprocessing.suitability import SuitabilityAssessor
from cv.features.complexity import SignatureComplexityScorer
from cv.features.extractors import ForensicFeatureExtractor
from cv.preprocessing.pipeline import ForensicImagePreprocessor
from cv.segmentation.document_segmenter import ForensicDocumentSegmenter
from ml.evaluation.hypothesis import ForensicHypothesisEvaluator
from ml.explainability.explainer import DualForensicExplainer
from ml.models.comparison import ForensicComparisonEngine
from ml.models.disguise import ForensicDisguiseDetector
from ml.models.fusion import HybridForensicFusion
from backend.app.services.report_service import ForensicReportService
from scripts.ingest_all_datasets import ForensicDatasetIngestionPipeline

client = TestClient(app)


def create_synthetic_document_page() -> np.ndarray:
    """Başlık, 3 satır metin ve sağ altta adli imza içeren sentetik A4 adli belge üretir."""
    page = np.full((650, 450), 255, dtype=np.uint8)

    # 1. Başlık ve Metin Satırları (Net satır aralıkları)
    cv2.putText(page, "T.C. ANKARA CUMHURIYET BASSAVCILIGI", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 2)
    cv2.putText(page, "Sorusturma No: 2026/1042 Esas", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 2)
    cv2.putText(page, "Supheli sahsin borc senedi uzerindeki imzasi incelenmistir.", (40, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 2)
    cv2.putText(page, "Senet metnindeki vuruslar ve imza ornekleri mukayese edilmistir.", (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 2)

    # 2. Sağ Altta Karakteristik İmza Deseni (Aspect Ratio ~ 2.5, karmaşık vuruşlar)
    sig_center_x, sig_center_y = 310, 520
    cv2.line(page, (sig_center_x - 60, sig_center_y + 10), (sig_center_x + 80, sig_center_y - 20), 0, 2)
    cv2.circle(page, (sig_center_x - 20, sig_center_y - 10), 18, 0, 2)
    cv2.circle(page, (sig_center_x + 20, sig_center_y - 5), 14, 0, 2)
    cv2.putText(page, "AdliImza", (sig_center_x - 40, sig_center_y + 25), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 0.75, 0, 2)

    return page


# ==============================================================================
# 1. Kimlik Doğrulama, JWT ve RBAC Testleri
# ==============================================================================

def test_auth_jwt_token_generation_and_decoding():
    """create_access_token ve decode_access_token işlevlerini test eder."""
    token = create_access_token(
        subject="adli_uzman_test",
        role="SENIOR_REVIEWER",
        extra_claims={"name": "Kıdemli Bilirkişi"},
    )
    assert isinstance(token, str)
    assert len(token) > 20

    payload = decode_access_token(token)
    assert payload["sub"] == "adli_uzman_test"
    assert payload["role"] == "SENIOR_REVIEWER"
    assert payload["name"] == "Kıdemli Bilirkişi"


def test_auth_login_me_and_registration_endpoints():
    """POST /auth/login, GET /auth/me ve POST /auth/register uç noktalarını test eder."""
    # 1. Varsayılan admin ile giriş yap
    login_res = client.post("/api/v1/auth/login", json={"username": "admin", "password": "password123"})
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    token = token_data["access_token"]
    assert token_data["user"]["username"] == "admin"
    assert token_data["user"]["role"] == "ADMIN"

    # 2. GET /auth/me
    headers = {"Authorization": f"Bearer {token}"}
    me_res = client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["username"] == "admin"

    # 3. Yeni Bilirkişi Kaydet
    reg_payload = {
        "username": "uzman_ayse_sprint14",
        "password": "secure_password_2026",
        "full_name": "Uzman Dr. Ayşe Yılmaz",
        "role": "EXAMINER",
    }
    reg_res = client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_res.status_code == 201
    assert reg_res.json()["username"] == "uzman_ayse_sprint14"
    assert reg_res.json()["role"] == "EXAMINER"

    # 4. Yeni kullanıcı ile giriş yap
    new_login_res = client.post("/api/v1/auth/login", json={
        "username": "uzman_ayse_sprint14",
        "password": "secure_password_2026",
    })
    assert new_login_res.status_code == 200
    assert new_login_res.json()["user"]["full_name"] == "Uzman Dr. Ayşe Yılmaz"


# ==============================================================================
# 2. Tekilleştirilmiş Asenkron Veritabanı ve Vaka Yönetimi Testleri
# ==============================================================================

def test_unified_database_case_lifecycle():
    """Tekil asenkron veritabanında vaka açma, listeleme ve durum sorgulama testi."""
    case_id = "VAKA_SPRINT14_UNIFIED_01"
    create_payload = {
        "case_id": case_id,
        "name": "Sprint 14 Tekilleştirilmiş Mimari Vakası",
        "description": "Tekil SQLite fhdss.db üzerinde uçtan uca test",
        "examination_type": "SIGNATURE",
        "authority": "İstanbul 1. Ağır Ceza Mahkemesi",
        "examiner_id": "UZMAN_SPRINT14",
    }

    # 1. Oluştur
    res = client.post("/api/v1/cases", json=create_payload)
    assert res.status_code == 201
    data = res.json()
    assert data["case_id"] == case_id
    assert data["name"] == "Sprint 14 Tekilleştirilmiş Mimari Vakası"
    assert data["state"] == "PHASE_A_INIT"

    # 2. Listele
    list_res = client.get("/api/v1/cases")
    assert list_res.status_code == 200
    cases = list_res.json()
    assert any(c["case_id"] == case_id for c in cases)

    # 3. Güncelle
    update_res = client.put(f"/api/v1/cases/{case_id}", json={"notes": "Adli ön inceleme yapıldı."})
    assert update_res.status_code == 200
    assert update_res.json()["notes"] == "Adli ön inceleme yapıldı."

    # 4. Sil (RBAC: ADMIN yetkisi ile)
    admin_token = create_access_token("admin_sprint14", role="ADMIN")
    del_res = client.delete(f"/api/v1/cases/{case_id}", headers={"Authorization": f"Bearer {admin_token}"})
    assert del_res.status_code == 200


# ==============================================================================
# 3. Belge Segmentasyon Modülü Testleri (cv/segmentation/)
# ==============================================================================

def test_document_segmenter_lines_and_signatures():
    """Tam sayfa belgeden satır ve imza bölgelerinin ayrıştırılmasını test eder."""
    doc_page = create_synthetic_document_page()
    segmenter = ForensicDocumentSegmenter(min_line_height=10)

    # 1. Ön İşleme
    binary = segmenter.preprocess_document_roi(doc_page)
    assert binary.shape == doc_page.shape
    assert np.sum(binary > 0) > 500  # Mürekkep pikselleri mevcut

    # 2. Satır Segmentasyonu
    lines = segmenter.segment_lines(doc_page)
    assert len(lines) >= 3, f"En az 3 metin satırı tespit edilmelidir, bulunan: {len(lines)}"
    assert lines[0].bbox[3] >= 10  # Satır yüksekliği makul olmalı
    assert lines[0].image.size > 0

    # 3. İmza Adayı Tespiti
    signatures = segmenter.extract_signature_candidates(doc_page)
    assert len(signatures) >= 1, "Belgedeki adli imza tespit edilmelidir."
    top_candidate = signatures[0]
    assert top_candidate.confidence_score >= 0.50
    assert top_candidate.aspect_ratio >= 1.0
    assert top_candidate.image.shape[0] > 15
    assert top_candidate.image.shape[1] > 30

    # 4. Bütünleşik Doküman Segmentasyonu
    full_result = segmenter.segment_document(doc_page)
    assert full_result.num_lines_detected == len(lines)
    assert full_result.num_signatures_detected == len(signatures)


# ==============================================================================
# 4. Sağlam PDF Raporlama Testleri (ReportLab Fallback)
# ==============================================================================

def test_reportlab_pdf_fallback_generation(tmp_path):
    """ReportLab ile geçerli ve mühürlü adli A4 PDF raporu oluşumunu test eder."""
    report_service = ForensicReportService(output_dir=tmp_path)

    # Mock adli girdi verileri
    img = np.full((120, 120), 255, dtype=np.uint8)
    cv2.line(img, (20, 60), (100, 60), 0, 2)

    suitability = SuitabilityAssessor().assess(img)
    preprocessor = ForensicImagePreprocessor()
    artifacts = preprocessor.process(img)
    extractor = ForensicFeatureExtractor()
    features = extractor.extract_all(artifacts.skeleton, artifacts.binarized)
    complexity = SignatureComplexityScorer().score(img)
    comparison = ForensicComparisonEngine().compare(features, [features])
    disguise = ForensicDisguiseDetector().analyze(features, comparison)
    fusion = HybridForensicFusion().fuse(comparison, deep_similarity_score=0.88)
    hypothesis = ForensicHypothesisEvaluator().evaluate(fusion, reference_count=1, suitability_result=suitability)
    xai = DualForensicExplainer().explain(comparison)

    case_info = {
        "case_id": "VAKA_PDF_TEST_01",
        "examiner_id": "UZMAN_DR_REPORTLAB",
        "authority": "Bakırköy 4. Ağır Ceza Mahkemesi",
        "examination_type": "IMZA AIDİYETI",
    }

    # PDF üret (senkron)
    result = report_service.generate_report(
        case_info=case_info,
        suitability=suitability,
        comparison=comparison,
        complexity=complexity,
        disguise=disguise,
        fusion=fusion,
        xai=xai,
        hypothesis=hypothesis,
    )

    assert result.is_pdf_generated is True
    assert result.pdf_path is not None
    pdf_file = Path(result.pdf_path)
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 1000  # En az 1 KB boyutunda

    # PDF sihirli baytları (%PDF) kontrolü
    with open(pdf_file, "rb") as f:
        header = f.read(5)
    assert header == b"%PDF-", "Üretilen dosya geçerli bir PDF dosyası olmalıdır."


# ==============================================================================
# 5. Arşiv Veri Setleri ETL Boru Hattı Testleri
# ==============================================================================

def test_dataset_ingestion_splitter_logic():
    """Disjoint-writer bölütleme algoritmasının sıfır veri sızıntısı kuralını test eder."""
    pipeline = ForensicDatasetIngestionPipeline()

    # 10 farklı yazar ve her yazar için 4 numunelik sentetik katalog
    dummy_writers = {}
    for w in range(1, 11):
        w_id = f"WRITER_{w:02d}"
        dummy_writers[w_id] = [
            {"sample_id": f"{w_id}_S1", "is_forgery": False, "relative_path": f"{w_id}/s1.png"},
            {"sample_id": f"{w_id}_S2", "is_forgery": False, "relative_path": f"{w_id}/s2.png"},
            {"sample_id": f"{w_id}_F1", "is_forgery": True, "relative_path": f"{w_id}/f1.png"},
            {"sample_id": f"{w_id}_F2", "is_forgery": True, "relative_path": f"{w_id}/f2.png"},
        ]

    split_res = pipeline.split_disjoint_writers(dummy_writers)

    train_w = set(split_res["writers"]["train"])
    val_w = set(split_res["writers"]["val"])
    test_w = set(split_res["writers"]["test"])

    # Sıfır Sızıntı (Disjointness) Doğrulaması: Kümeler kesişemez
    assert len(train_w.intersection(val_w)) == 0, "Train ve Val kümelerinde yazar sızıntısı tespit edildi!"
    assert len(train_w.intersection(test_w)) == 0, "Train ve Test kümelerinde yazar sızıntısı tespit edildi!"
    assert len(val_w.intersection(test_w)) == 0, "Val ve Test kümelerinde yazar sızıntısı tespit edildi!"

    assert len(train_w) + len(val_w) + len(test_w) == 10
    assert split_res["summary"]["total_writers"] == 10
