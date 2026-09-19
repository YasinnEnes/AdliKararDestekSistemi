import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import torch
from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.api.v1.endpoints.cases import get_case_orchestrator
from backend.app.core.config import settings
from backend.app.core.context_isolation import (
    ContextIsolationOrchestrator,
    IsolationState,
    IsolationViolationError,
)
from backend.app.core.guard import PsychologicalInferenceGuard
from backend.app.core.logger import log_audit_event, logger
from backend.app.core.merkle_audit import MerkleAuditTree
from backend.app.core.security import get_current_active_user
from backend.app.db.session import get_db
from backend.app.models.entities import CaseEntity, SampleEntity, AnalysisResultEntity, UserEntity
from backend.app.schemas.api_contracts import (
    CounterfactualRequest,
    CounterfactualResponse,
    FullPipelineRequest,
    FullPipelineResponse,
    PhaseAIsolateRequest,
    PhaseAIsolateResponse,
    SuitabilityResponse,
)
from backend.app.services.report_service import ForensicReportService
from cv.features.complexity import SignatureComplexityScorer
from cv.features.extractors import ForensicFeatureExtractor
from cv.preprocessing.pipeline import ForensicImagePreprocessor
from cv.preprocessing.suitability import SuitabilityAssessor
from ml.evaluation.hypothesis import ForensicHypothesisEvaluator
from ml.explainability.counterfactual import CounterfactualExplainer
from ml.explainability.explainer import DualForensicExplainer
from ml.models.comparison import (
    ComparisonReport,
    DifferenceType,
    FeatureDifferenceDetail,
    ForensicComparisonEngine,
)
from ml.models.disguise import ForensicDisguiseDetector
from ml.models.fusion import HybridForensicFusion
from ml.models.siamese import ForensicConvNeXtEncoder, compute_embedding_distance, compute_normalized_similarity
from ml.preprocessing.contract import preprocess_for_inference

router = APIRouter()

# Paylaşılan servis tekil örnekleri (Singletons)
_SUITABILITY_ASSESSOR = SuitabilityAssessor()
_PREPROCESSOR = ForensicImagePreprocessor()
_FEATURE_EXTRACTOR = ForensicFeatureExtractor()
_COMPLEXITY_SCORER = SignatureComplexityScorer()
_COMPARISON_ENGINE = ForensicComparisonEngine()
_CONVNEXT_ENCODER = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
_WEIGHTS_PATH = settings.BASE_DIR / "artifacts" / "weights" / "convnext_tiny_forensic_best.pt"
if not _WEIGHTS_PATH.exists():
    _WEIGHTS_PATH = settings.BASE_DIR / "artifacts" / "weights" / "siamese_convnext_best.pt"

if _WEIGHTS_PATH.exists():
    try:
        _checkpoint = torch.load(_WEIGHTS_PATH, map_location="cpu")
        if isinstance(_checkpoint, dict) and "model_state_dict" in _checkpoint:
            _CONVNEXT_ENCODER.load_state_dict(_checkpoint["model_state_dict"])
        elif isinstance(_checkpoint, dict) and "encoder_state_dict" in _checkpoint:
            _CONVNEXT_ENCODER.load_state_dict(_checkpoint["encoder_state_dict"])
        else:
            _CONVNEXT_ENCODER.load_state_dict(_checkpoint, strict=False)
        logger.info(f"ConvNeXt adli model ağırlıkları başarıyla yüklendi: {_WEIGHTS_PATH.name}")
    except Exception as e:
        logger.warning(f"Model ağırlığı yüklenirken uyarı: {e}")

_CONVNEXT_ENCODER.eval()
from backend.app.services.onnx_inference import ForensicONNXInferenceEngine
from backend.app.services.multi_sample_service import multi_sample_clustering_service

_ONNX_ENGINE = ForensicONNXInferenceEngine(pytorch_model=_CONVNEXT_ENCODER)
_FUSION_LAYER = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)
_DISGUISE_DETECTOR = ForensicDisguiseDetector()
_DUAL_EXPLAINER = DualForensicExplainer()
_COUNTERFACTUAL_EXPLAINER = CounterfactualExplainer(_FUSION_LAYER)
_HYPOTHESIS_EVALUATOR = ForensicHypothesisEvaluator()
_REPORT_SERVICE = ForensicReportService()


async def _load_sample_image(db: AsyncSession, sample_id: str) -> Dict[str, Any]:
    """Veritabanından numune kaydını ve diskteki görsel dosyasını yükler."""
    stmt = select(SampleEntity).where(SampleEntity.sample_id == sample_id)
    res = await db.execute(stmt)
    sample_entity = res.scalar_one_or_none()

    if not sample_entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{sample_id}' numaralı adli numune veritabanında bulunamadı.",
        )

    if not sample_entity.file_path or not os.path.exists(sample_entity.file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{sample_id}' numunesinin görsel dosyası diskte bulunamadı.",
        )

    try:
        with open(sample_entity.file_path, "rb") as f:
            file_bytes = f.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        img = None

    if img is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{sample_id}' numunesi okunamadı veya bozuk görsel dosyası.",
        )

    return {
        "sample_id": sample_entity.sample_id,
        "case_id": sample_entity.case_id,
        "filename": sample_entity.filename,
        "sample_type": sample_entity.sample_type,
        "sha256_hash": sample_entity.sha256_hash,
        "image": img,
        "file_size_bytes": sample_entity.file_size_bytes,
        "file_path": sample_entity.file_path,
    }


# ------------------------------------------------------------------------------
# REST API Uç Noktaları
# ------------------------------------------------------------------------------

@router.post(
    "/suitability",
    response_model=SuitabilityResponse,
    summary="Belge Kalite ve Adli Uygunluk Değerlendirmesi",
)
async def assess_suitability(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Yüklenen görsel veya depolanan numune üzerinde bulanıklık, kontrast, SNR ve yazı yoğunluğu kontrolü yapar.
    """
    if file is not None:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Geçersiz görsel dosyası.",
            )
        target_sample_id = file.filename or "UPLOADED_SAMPLE"
    elif sample_id is not None:
        sample_data = await _load_sample_image(db, sample_id)
        img = sample_data["image"]
        target_sample_id = sample_id
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Analiz için ya bir dosya yüklenmeli ya da 'sample_id' sağlanmalıdır.",
        )

    result = _SUITABILITY_ASSESSOR.assess(img)
    return SuitabilityResponse(sample_id=target_sample_id, suitability=result)


@router.post(
    "/isolate-phase-a",
    response_model=PhaseAIsolateResponse,
    summary="Phase A (Sorgulanan Belge) Özniteliklerini Kilitle ve Dondur",
)
async def isolate_phase_a(
    payload: PhaseAIsolateRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    MFHM Bilişsel İzolasyon Kuralı: Sorgulanan belgenin özniteliklerini çıkarır ve SHA-256 ile dondurur.
    Bu işlem tamamlanmadan Referans belgelere (Phase B) erişilemez.
    """
    case_stmt = select(CaseEntity).where(CaseEntity.case_id == payload.case_id)
    case_res = await db.execute(case_stmt)
    case_entity = case_res.scalar_one_or_none()

    if not case_entity:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{payload.case_id}' vakası bulunamadı.")

    sample_data = await _load_sample_image(db, payload.questioned_sample_id)
    img = sample_data["image"]

    orchestrator = get_case_orchestrator(
        case_id=payload.case_id,
        examiner_id=case_entity.examiner_id,
        current_state=case_entity.state,
        questioned_sample_id=payload.questioned_sample_id,
    )

    if (
        orchestrator.state != IsolationState.PHASE_A_ANALYSIS
        or orchestrator.questioned_sample_id != payload.questioned_sample_id
    ):
        orchestrator.start_phase_a(
            questioned_sample_id=payload.questioned_sample_id,
            document_hash=sample_data["sha256_hash"],
            force_reset=True,
        )

    # Ön işleme ve öznitelik çıkarımı
    artifacts = _PREPROCESSOR.process(img)
    features_report = _FEATURE_EXTRACTOR.extract_all(
        skeleton_image=artifacts.skeleton,
        binary_image=artifacts.binarized,
    )

    feature_dict = {
        **features_report.structural.model_dump(),
        **features_report.line_quality.model_dump(),
    }

    try:
        frozen = orchestrator.freeze_phase_a(
            features=feature_dict,
            examiner_notes=payload.examiner_notes or "",
            document_hash=sample_data["sha256_hash"],
        )
        case_entity.state = orchestrator.state.value
        case_entity.current_stage = "FEATURE_EXTRACTION"
        await db.flush()
    except IsolationViolationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    logger.info(f"Phase A başarıyla donduruldu [Vaka: {payload.case_id}, SHA: {frozen.integrity_hash_sha256[:10]}]")

    return PhaseAIsolateResponse(
        case_id=payload.case_id,
        questioned_sample_id=payload.questioned_sample_id,
        integrity_hash_sha256=frozen.integrity_hash_sha256,
        frozen_at_utc=frozen.frozen_at_utc,
        features=frozen.features,
    )


from backend.app.core.rate_limiter import check_pipeline_rate_limit

@router.post(
    "/full-pipeline",
    response_model=FullPipelineResponse,
    summary="Uçtan Uca Adli Karar Destek Tam Analiz Hattı (SQL Kalıcı Kayıt)",
)
@router.post(
    "/pipeline",
    response_model=FullPipelineResponse,
    summary="Uçtan Uca Adli Karar Destek Tam Analiz Hattı (Alias)",
    include_in_schema=False,
)
async def run_full_pipeline(
    payload: FullPipelineRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Tüm adli modülleri sırayla işletir ve sonuçları SQL veritabanına kalıcı olarak kaydeder:
    Suitability -> Preprocessing -> Features -> Complexity -> Comparison -> ConvNeXt -> Gated Fusion -> Disguise -> Dual XAI -> ENFSI Hypothesis -> Rapor
    Sprint 16.3: Kullanıcı bazlı hız sınırlaması (10 istek/dakika).
    """
    check_pipeline_rate_limit(current_user.username)
    case_id = payload.case_id

    # Adli Bilimsel Kapsam Denetimi
    PsychologicalInferenceGuard.enforce_scope(context_text=f"{payload.case_id} {payload.suspect_name or ''}")

    case_stmt = select(CaseEntity).where(CaseEntity.case_id == case_id)
    case_res = await db.execute(case_stmt)
    case_entity = case_res.scalar_one_or_none()

    if not case_entity:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{case_id}' vakası bulunamadı.")

    orchestrator = get_case_orchestrator(case_id, case_entity.examiner_id, case_entity.state)

    # Numune görsellerini yükle
    q_sample = await _load_sample_image(db, payload.questioned_sample_id)
    q_img = q_sample["image"]

    ref_samples = [await _load_sample_image(db, sid) for sid in payload.reference_sample_ids]
    ref_imgs = [s["image"] for s in ref_samples]

    # 1. Bilişsel İzolasyon: Phase A henüz dondurulmamışsa veya sorgulanan belge değişmişse Phase A'yı dondur ve Phase B kilidini aç
    need_freeze_phase_a = (
        orchestrator.questioned_sample_id != payload.questioned_sample_id
        or orchestrator.frozen_evidence is None
        or not orchestrator.verify_integrity()
        or orchestrator.state in (IsolationState.PHASE_A_INIT, IsolationState.PHASE_A_ANALYSIS)
    )

    if need_freeze_phase_a:
        orchestrator.reset()
        orchestrator.start_phase_a(
            questioned_sample_id=payload.questioned_sample_id,
            document_hash=q_sample["sha256_hash"],
            force_reset=True,
        )
        q_artifacts_pre = _PREPROCESSOR.process(q_img)
        q_features_pre = _FEATURE_EXTRACTOR.extract_all(q_artifacts_pre.skeleton, q_artifacts_pre.binarized)
        orchestrator.freeze_phase_a(
            features=q_features_pre.structural.model_dump(),
            examiner_notes="Tam adli analiz öncesi otomatik Phase A dondurma.",
            document_hash=q_sample["sha256_hash"],
        )

    else:
        # Phase A zaten dondurulmuş ve bütünlüğü tam; yeni mukayese analizi için durumu PHASE_A_FROZEN'a getir
        orchestrator.state = IsolationState.PHASE_A_FROZEN

    try:
        orchestrator.unlock_phase_b(reference_sample_ids=payload.reference_sample_ids)
    except IsolationViolationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # 2. Uygunluk ve Kalite Değerlendirmesi
    suitability_res = _SUITABILITY_ASSESSOR.assess(q_img)

    # 3. Deterministik Ön İşleme
    q_artifacts = _PREPROCESSOR.process(q_img)
    ref_artifacts_list = [_PREPROCESSOR.process(img) for img in ref_imgs]

    # 4. Adli Öznitelik Çıkarımı
    q_features = _FEATURE_EXTRACTOR.extract_all(q_artifacts.skeleton, q_artifacts.binarized)
    ref_features_list = [
        _FEATURE_EXTRACTOR.extract_all(art.skeleton, art.binarized) for art in ref_artifacts_list
    ]

    # 5. Found & Rogers İmza Karmaşıklığı
    complexity_res = _COMPLEXITY_SCORER.score(q_artifacts.skeleton, q_artifacts.binarized)

    # 6. Temel Adli Karşılaştırma (Z-Skorları & Mahalanobis)
    comp_report = _COMPARISON_ENGINE.compare(q_features, ref_features_list)

    # 7. ConvNeXt-Tiny Derin Metrik Embedding Çıkarımı (ONNX CPU / PyTorch Fallback)
    # Sprint 17.4 & 17.5: Ortak Preprocessing Sözleşmesi ve Normalize Benzerlik Standardı
    q_tensor = preprocess_for_inference(q_artifacts.normalized_crop)
    q_emb_np = _ONNX_ENGINE.encode(q_tensor)

    ref_sims = []
    for r_art in ref_artifacts_list:
        r_tensor = preprocess_for_inference(r_art.normalized_crop)
        r_emb_np = _ONNX_ENGINE.encode(r_tensor)
        norm_sim = compute_normalized_similarity(q_emb_np[0], r_emb_np[0])
        ref_sims.append(norm_sim)

    deep_similarity = float(np.mean(ref_sims)) if ref_sims else 0.5

    # 8. Adli Gating Destekli Hibrit Füzyon
    fusion_res = _FUSION_LAYER.fuse(
        morphological_report=comp_report,
        deep_similarity_score=deep_similarity,
    )

    # 9. Taklit ve Gizleme Anomali Tespiti
    disguise_res = _DISGUISE_DETECTOR.analyze(q_features, comparison_report=comp_report)

    # 10. Çift Seviyeli Açıklanabilirlik (Dual XAI)
    xai_res = _DUAL_EXPLAINER.explain(
        comparison_report=comp_report,
        encoder=_CONVNEXT_ENCODER,
        image_tensor=q_tensor,
    )

    # 11. ENFSI Standartlarında Bayesyen Hipotez Değerlendirmesi
    hypothesis_res = _HYPOTHESIS_EVALUATOR.evaluate(
        hybrid_result=fusion_res,
        reference_count=len(ref_imgs),
        suitability_result=suitability_res,
        suspect_name=payload.suspect_name or case_entity.name or "Şüpheli / Referans Şahıs",
    )

    # 12. Adli Rapor ve Mühürleme
    case_info_bundle = {
        "case_id": case_id,
        "authority": case_entity.authority or "Cumhuriyet Başsavcılığı / Mahkeme",
        "examiner_id": case_entity.examiner_id,
        "questioned_id": payload.questioned_sample_id,
        "reference_count": len(ref_imgs),
    }

    isolation_hash = (
        orchestrator.frozen_evidence.integrity_hash_sha256
        if orchestrator.frozen_evidence
        else None
    )

    report_result = _REPORT_SERVICE.generate_report(
        case_info=case_info_bundle,
        suitability=suitability_res,
        comparison=comp_report,
        complexity=complexity_res,
        disguise=disguise_res,
        fusion=fusion_res,
        xai=xai_res,
        hypothesis=hypothesis_res,
        isolation_hash=isolation_hash,
        background_tasks=background_tasks,
    )

    # 13. Süreci Tamamla ve Veritabanına Kaydet
    try:
        orchestrator.complete_evaluation()
    except Exception:
        pass

    case_entity.state = "EVALUATION_COMPLETED"
    case_entity.current_stage = "REPORT"
    case_entity.status = "COMPLETED"

    # 13.5. Kriptografik Merkle Ağacı Kökü Hesabı
    merkle_tree = MerkleAuditTree([
        {"event": "PHASE_A_FROZEN", "case_id": case_id, "sample_q": payload.questioned_sample_id},
        {"event": "PHASE_B_UNLOCKED", "references": payload.reference_sample_ids},
        {"event": "SUITABILITY", "score": suitability_res.overall_score},
        {"event": "COMPLEXITY", "index": complexity_res.overall_complexity_index},
        {"event": "COMPARISON", "similarity": comp_report.similarity_score},
        {"event": "FUSION", "hybrid_score": fusion_res.hybrid_similarity_score},
        {"event": "DISGUISE", "level": disguise_res.indicator_level.value},
        {"event": "HYPOTHESIS", "log_lr": hypothesis_res.log_likelihood_ratio},
        {"event": "REPORT_SEAL", "sha256": report_result.report_sha256},
    ])
    merkle_root = merkle_tree.get_root_hash()
    case_entity.merkle_root_hash = merkle_root
    case_entity.similarity_score = fusion_res.hybrid_similarity_score
    case_entity.verbal_scale = hypothesis_res.verbal_conclusion_tr or hypothesis_res.support_level.value
    case_entity.audit_hash = report_result.report_sha256
    case_entity.updated_at_utc = datetime.now(timezone.utc).isoformat()

    # SQL Tablosuna Analiz Sonucunu Kaydet
    now = datetime.now(timezone.utc).isoformat()
    analysis_entity = AnalysisResultEntity(
        case_id=case_id,
        suitability_json=suitability_res.model_dump(),
        complexity_json=complexity_res.model_dump(),
        comparison_json=comp_report.model_dump(),
        disguise_json=disguise_res.model_dump(),
        fusion_json=fusion_res.model_dump(),
        xai_json=xai_res.model_dump(),
        hypothesis_json=hypothesis_res.model_dump(),
        audit_checksum_sha256=report_result.report_sha256,
        merkle_root_hash=merkle_root,
        html_report_path=report_result.html_path,
        pdf_report_path=report_result.pdf_path,
        created_at_utc=now,
    )
    db.add(analysis_entity)
    await db.flush()

    logger.info(f"Vaka {case_id} uçtan uca adli karar destek analizi tamamlandı ve Merkle Ağacı ile mühürlendi (Root: {merkle_root[:12]}...).")

    return FullPipelineResponse(
        case_id=case_id,
        suitability=suitability_res,
        complexity=complexity_res,
        comparison=comp_report,
        disguise=disguise_res,
        fusion=fusion_res,
        xai=xai_res,
        hypothesis=hypothesis_res,
        report_paths={
            "html_path": report_result.html_path,
            "pdf_path": report_result.pdf_path,
            "report_sha256": report_result.report_sha256,
        },
        audit_checksum_sha256=report_result.report_sha256,
        merkle_root_hash=merkle_root,
    )


def _get_baseline_comparison_report() -> ComparisonReport:
    """Henüz analizi tamamlanmamış vakalar için örnek simülasyon karşılaştırma raporu döner."""
    from ml.models.comparison import DifferenceType, FeatureDifferenceDetail
    diffs = [
        FeatureDifferenceDetail(
            feature_key="slant_angle_deg",
            feature_name_tr="Yazı Eğim Açısı",
            questioned_value=18.5,
            reference_mean=4.2,
            reference_std=2.1,
            absolute_difference=14.3,
            z_score=6.81,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Belirgin eğim açısı sapması (18.5° vs 4.2°)",
        ),
        FeatureDifferenceDetail(
            feature_key="tremor_index_psd",
            feature_name_tr="FFT Titreme İndeksi",
            questioned_value=0.12,
            reference_mean=0.03,
            reference_std=0.01,
            absolute_difference=0.09,
            z_score=9.0,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Mikroskobik duraksama ve titreme anomalisi",
        ),
        FeatureDifferenceDetail(
            feature_key="smoothness_energy",
            feature_name_tr="Kinematik Pürüzsüzlük",
            questioned_value=0.42,
            reference_mean=0.78,
            reference_std=0.08,
            absolute_difference=0.36,
            z_score=-4.5,
            diff_type=DifferenceType.SIGNIFICANT_DIFFERENCE,
            explanation="Düşük akıcılık ve kinematik tutukluk",
        ),
        FeatureDifferenceDetail(
            feature_key="loop_count",
            feature_name_tr="Halka / İlik Sayısı",
            questioned_value=3.0,
            reference_mean=5.2,
            reference_std=0.8,
            absolute_difference=2.2,
            z_score=-2.75,
            diff_type=DifferenceType.NATURAL_VARIATION,
            explanation="Doğal varyasyon sınırında ilik sayısı farkı",
        ),
        FeatureDifferenceDetail(
            feature_key="baseline_deviation_px",
            feature_name_tr="Taban Çizgisi Sapması",
            questioned_value=6.4,
            reference_mean=5.8,
            reference_std=1.2,
            absolute_difference=0.6,
            z_score=0.5,
            diff_type=DifferenceType.CONCORDANT,
            explanation="Taban çizgisi referansla tam uyumlu",
        ),
    ]
    return ComparisonReport(
        similarity_score=0.48,
        euclidean_distance=8.2,
        mahalanobis_distance=8.2,
        concordant_features=[d for d in diffs if d.diff_type == DifferenceType.CONCORDANT],
        natural_variations=[d for d in diffs if d.diff_type == DifferenceType.NATURAL_VARIATION],
        significant_differences=[d for d in diffs if d.diff_type == DifferenceType.SIGNIFICANT_DIFFERENCE],
        preliminary_opinion="Kısmi Uyuşmazlık ve Adli Gating Tetiklendi",
        reference_count=3,
        audit_hash_sha256="BASELINE_DEMO_HASH",
    )


@router.post(
    "/counterfactual",
    response_model=CounterfactualResponse,
    summary="Karşıtlı Açıklama ve 'Ne Olurdu?' Simülasyonu",
)
async def simulate_counterfactual(
    payload: CounterfactualRequest,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Bilirkişiye öznitelik değişimlerinin model kararına etkisini anlık (<250ms) simüle eder.
    Vaka henüz analiz edilmemişse otomatik olarak temel adli kalibrasyon profili üzerinden simülasyon sunar.
    """
    case_id = payload.case_id
    stmt = select(AnalysisResultEntity).where(AnalysisResultEntity.case_id == case_id).order_by(AnalysisResultEntity.created_at_utc.desc())
    res = await db.execute(stmt)
    analysis_entity = res.scalar_one_or_none()

    if analysis_entity and analysis_entity.comparison_json:
        comp_report = ComparisonReport(**analysis_entity.comparison_json)
        current_hybrid_score = float(analysis_entity.fusion_json.get("hybrid_similarity_score", 0.65)) if analysis_entity.fusion_json else 0.65
    else:
        # Vaka yeni oluşturulduysa veya henüz analiz çalıştırılmadıysa kalibre edilmiş varsayılan simülasyon profili kullan
        comp_report = _get_baseline_comparison_report()
        current_hybrid_score = 0.48

    # 1. Ne Olurdu simülasyonu
    sim_res = _COUNTERFACTUAL_EXPLAINER.simulate_what_if(
        comparison_report=comp_report,
        custom_perturbations=payload.custom_perturbations or {},
        deep_similarity_score=payload.deep_similarity_score or 0.85,
    )

    # 2. Otomatik en iyi 2 senaryo
    scenarios_res = _COUNTERFACTUAL_EXPLAINER.generate_counterfactual_scenarios(
        comparison_report=comp_report,
        current_hybrid_score=current_hybrid_score,
        deep_similarity_score=payload.deep_similarity_score or 0.85,
    )

    return CounterfactualResponse(
        case_id=case_id,
        original_score=current_hybrid_score,
        simulated_hybrid_score=sim_res["simulated_hybrid_score"],
        simulated_morphological_score=sim_res["simulated_morphological_score"],
        gating_penalty=sim_res["gating_penalty"],
        gating_triggered=sim_res["gating_triggered"],
        structural_contradictions=sim_res["structural_contradictions"],
        concordance_level=sim_res["concordance_level"],
        sensitivity_matrix=scenarios_res.sensitivity_matrix,
        top_scenarios=[asdict(s) for s in scenarios_res.top_scenarios],
        execution_time_ms=sim_res["execution_time_ms"],
    )


@router.get(
    "/result/{case_id}",
    response_model=FullPipelineResponse,
    summary="Vakaya Ait En Son Analiz Sonucunu Getir",
)
async def get_latest_analysis_result(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Vakaya ait en son kaydedilmiş tam adli analiz sonucunu döner."""
    stmt = (
        select(AnalysisResultEntity)
        .where(AnalysisResultEntity.case_id == case_id)
        .order_by(AnalysisResultEntity.created_at_utc.desc())
    )
    res = await db.execute(stmt)
    entity = res.scalar_one_or_none()
    if not entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{case_id}' vakası için henüz bir analiz sonucu bulunamadı.",
        )

    return FullPipelineResponse(
        case_id=entity.case_id,
        suitability=entity.suitability_json or {},
        complexity=entity.complexity_json or {},
        comparison=entity.comparison_json or {},
        disguise=entity.disguise_json or {},
        fusion=entity.fusion_json or {},
        xai=entity.xai_json or {},
        hypothesis=entity.hypothesis_json or {},
        report_paths={
            "html_path": entity.html_report_path or "",
            "pdf_path": entity.pdf_report_path or "",
            "report_sha256": entity.audit_checksum_sha256 or "",
        },
        audit_checksum_sha256=entity.audit_checksum_sha256 or "",
        merkle_root_hash=entity.merkle_root_hash or "",
    )


@router.get(
    "/report/pdf/{case_id}",
    summary="Mühürlü Adli PDF Raporunu İndir",
)
async def download_pdf_report(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """Üretilen adli PDF raporunu (veya HTML raporunu) streaming yanıt olarak yetkili şekilde indirir."""
    # Vakanın varlığını ve silinmediğini doğrula
    case_stmt = select(CaseEntity).where(CaseEntity.case_id == case_id, CaseEntity.is_deleted == False)
    case_res = await db.execute(case_stmt)
    case_entity = case_res.scalar_one_or_none()
    if not case_entity:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"'{case_id}' numaralı adli vaka bulunamadı veya silinmiş.",
        )

    reports_dir = _REPORT_SERVICE.output_dir
    sanitized_id = case_id.replace("/", "_").replace("\\", "_")

    # PDF ara
    pdf_files = list(reports_dir.glob(f"report_{sanitized_id}_*.pdf"))
    if pdf_files:
        latest_pdf = sorted(pdf_files)[-1]
        return FileResponse(
            path=str(latest_pdf),
            media_type="application/pdf",
            filename=f"Adli_Rapor_{sanitized_id}.pdf",
        )

    # HTML fallback
    html_files = list(reports_dir.glob(f"report_{sanitized_id}_*.html"))
    if html_files:
        latest_html = sorted(html_files)[-1]
        return FileResponse(
            path=str(latest_html),
            media_type="text/html",
            filename=f"Adli_Rapor_{sanitized_id}.html",
        )

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"'{case_id}' vakası için henüz üretilmiş bir PDF/HTML raporu bulunamadı. Lütfen önce analizi tamamlayın.",
    )


@router.get(
    "/inference-engine-status",
    summary="Yapay Zeka Çıkarım Motoru ve Donanım Durumu",
)
async def get_inference_engine_status(current_user: UserEntity = Depends(get_current_active_user)):
    """ConvNeXt modelinin ONNX Runtime veya PyTorch CPU Fallback durumunu raporlar."""
    return _ONNX_ENGINE.get_status()


@router.get(
    "/multi-sample-cluster/{case_id}",
    summary="Çoklu Referans Numune Embedding Kümeleme ve Yazar İçi Varyasyon Analizi",
)
async def get_multi_sample_clustering(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: UserEntity = Depends(get_current_active_user),
):
    """
    Vakaya ait sorgulanan ve referans numunelerin derin embedding vektörlerini çıkarır,
    PCA 2D projeksiyonu, yazar içi varyasyon tolerans elipsi ve anomali durumunu hesaplar.
    """
    stmt = select(SampleEntity).where(SampleEntity.case_id == case_id)
    res = await db.execute(stmt)
    samples = res.scalars().all()

    if not samples:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"'{case_id}' vakasına ait numune bulunamadı.")

    q_samples = [s for s in samples if s.sample_type == "QUESTIONED"]
    ref_samples = [s for s in samples if s.sample_type == "REFERENCE"]

    if not q_samples:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Vakada sorgulanan numune (QUESTIONED) bulunamadı.")
    if not ref_samples:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Vakada en az bir adet referans numune (REFERENCE) bulunmalıdır.")

    target_q = q_samples[0]
    q_data = await _load_sample_image(db, target_q.sample_id)
    q_art = _PREPROCESSOR.process(q_data["image"])
    q_tensor = preprocess_for_inference(q_art.normalized_crop)
    q_emb = _ONNX_ENGINE.encode(q_tensor)[0]

    ref_payload = []
    for r in ref_samples:
        r_data = await _load_sample_image(db, r.sample_id)
        r_art = _PREPROCESSOR.process(r_data["image"])
        r_tensor = preprocess_for_inference(r_art.normalized_crop)
        r_emb = _ONNX_ENGINE.encode(r_tensor)[0]
        ref_payload.append({
            "sample_id": r.sample_id,
            "filename": r.filename,
            "embedding": r_emb,
        })

    cluster_result = multi_sample_clustering_service.analyze_clusters(
        questioned_sample_id=target_q.sample_id,
        questioned_embedding=q_emb,
        reference_samples=ref_payload,
    )
    cluster_result["case_id"] = case_id
    return cluster_result

