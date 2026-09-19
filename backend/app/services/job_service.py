"""
Adli Karar Destek Asenkron Analiz İş Kuyruğu ve SSE (Server-Sent Events) Servisi.
Forensic Background Job Execution and Real-time Progress Broadcasting.
"""

import asyncio
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional
import cv2
import numpy as np
import torch
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.core.context_isolation import IsolationState, IsolationViolationError
from backend.app.core.guard import PsychologicalInferenceGuard
from backend.app.core.logger import log_audit_event, logger
from backend.app.core.merkle_audit import MerkleAuditTree
from backend.app.db.session import AsyncSessionLocal
from backend.app.models.entities import AnalysisJobEntity, AnalysisResultEntity, CaseEntity, SampleEntity
from backend.app.services.onnx_inference import ForensicONNXInferenceEngine
from backend.app.services.report_service import ForensicReportService
from cv.features.complexity import SignatureComplexityScorer
from cv.features.extractors import ForensicFeatureExtractor
from cv.preprocessing.pipeline import ForensicImagePreprocessor
from cv.preprocessing.suitability import SuitabilityAssessor
from ml.evaluation.hypothesis import ForensicHypothesisEvaluator
from ml.explainability.explainer import DualForensicExplainer
from ml.models.comparison import ForensicComparisonEngine
from ml.models.disguise import ForensicDisguiseDetector
from ml.models.fusion import HybridForensicFusion
from ml.models.siamese import ForensicConvNeXtEncoder, compute_embedding_distance, compute_normalized_similarity
from ml.preprocessing.contract import preprocess_for_inference

# Singletons for analysis pipeline
_SUITABILITY_ASSESSOR = SuitabilityAssessor()
_PREPROCESSOR = ForensicImagePreprocessor()
_FEATURE_EXTRACTOR = ForensicFeatureExtractor()
_COMPLEXITY_SCORER = SignatureComplexityScorer()
_COMPARISON_ENGINE = ForensicComparisonEngine()
_CONVNEXT_ENCODER = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
_ONNX_ENGINE: Optional[ForensicONNXInferenceEngine] = None

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
    except Exception as e:
        logger.warning(f"Job Service model ağırlığı yüklenirken uyarı: {e}")

_CONVNEXT_ENCODER.eval()
_ONNX_ENGINE = ForensicONNXInferenceEngine(pytorch_model=_CONVNEXT_ENCODER)

_FUSION_LAYER = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)
_DISGUISE_DETECTOR = ForensicDisguiseDetector()
_DUAL_EXPLAINER = DualForensicExplainer()
_HYPOTHESIS_EVALUATOR = ForensicHypothesisEvaluator()
_REPORT_SERVICE = ForensicReportService()


class JobEventBroadcaster:
    """Gerçek zamanlı SSE (Server-Sent Events) istemci abonelik yöneticisi."""

    def __init__(self):
        self._subscribers: Dict[str, List[asyncio.Queue]] = {}

    def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        if job_id not in self._subscribers:
            self._subscribers[job_id] = []
        self._subscribers[job_id].append(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue):
        if job_id in self._subscribers:
            if queue in self._subscribers[job_id]:
                self._subscribers[job_id].remove(queue)
            if not self._subscribers[job_id]:
                del self._subscribers[job_id]

    async def publish(self, job_id: str, data: Dict[str, Any]):
        if job_id in self._subscribers:
            for q in list(self._subscribers[job_id]):
                try:
                    await q.put(data)
                except Exception:
                    pass


broadcaster = JobEventBroadcaster()


async def _load_and_verify_sample(session, sample_id: str) -> Dict[str, Any]:
    """Numuneyi veritabanından çeker ve diskteki SHA-256 delil bütünlüğünü doğrular."""
    stmt = select(SampleEntity).where(SampleEntity.sample_id == sample_id)
    res = await session.execute(stmt)
    sample_entity = res.scalar_one_or_none()
    if not sample_entity:
        raise ValueError(f"'{sample_id}' numaralı adli numune veritabanında bulunamadı.")

    p = Path(sample_entity.file_path)
    if not p.is_absolute():
        p = settings.BASE_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Numune görseli diskte bulunamadı: {p}")

    file_bytes = p.read_bytes()
    current_sha = hashlib.sha256(file_bytes).hexdigest()
    if sample_entity.sha256_hash and current_sha != sample_entity.sha256_hash:
        raise ValueError(
            f"Adli Delil İhlali! Numune '{sample_id}' SHA-256 özeti eşleşmiyor. "
            f"Kayıtlı: {sample_entity.sha256_hash[:12]}..., Hesaplanan: {current_sha[:12]}..."
        )

    # Windows Unicode / Türkçe karakterli dosya yolları için imdecode kullanımı
    img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        img = cv2.imread(str(p))
    if img is None:
        raise ValueError(f"Numune dosyası geçerli bir görsel olarak açılamadı: {p}")

    return {
        "sample_id": sample_entity.sample_id,
        "filename": sample_entity.filename,
        "sample_type": sample_entity.sample_type,
        "sha256_hash": current_sha,
        "image": img,
        "file_path": str(p),
    }


async def run_pipeline_job(
    job_id: str,
    case_id: str,
    questioned_sample_id: str,
    reference_sample_ids: List[str],
    suspect_name: Optional[str] = None,
    examiner_username: str = "system_examiner",
) -> None:
    """Arka planda asenkron olarak çalışan ve adli boru hattı aşamalarını güncelleyen işçi."""

    async def update_progress(progress: int, step: str, status: str = "RUNNING", result_data: Optional[Dict] = None, err_msg: Optional[str] = None):
        async with AsyncSessionLocal() as session:
            stmt = select(AnalysisJobEntity).where(AnalysisJobEntity.job_id == job_id)
            res = await session.execute(stmt)
            job = res.scalar_one_or_none()
            if job:
                job.progress = progress
                job.current_step = step
                job.status = status
                if err_msg:
                    job.error_message = err_msg
                if result_data:
                    job.result_json = result_data
                if status in ("COMPLETED", "FAILED"):
                    job.completed_at_utc = datetime.now(timezone.utc).isoformat()
                await session.commit()

        event_payload = {
            "job_id": job_id,
            "case_id": case_id,
            "status": status,
            "progress": progress,
            "step": step,
            "error": err_msg,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if result_data and status == "COMPLETED":
            event_payload["result"] = result_data
        await broadcaster.publish(job_id, event_payload)

    try:
        await update_progress(5, "Adli Bilimsel Kapsam ve Yetkilendirme Doğrulanıyor")
        PsychologicalInferenceGuard.enforce_scope(context_text=f"{case_id} {suspect_name or ''}")

        # 1. Delil Bütünlüğü ve SHA-256 Doğrulama
        await update_progress(15, "Adli Delil Bütünlüğü ve SHA-256 Kriptografik Özeti Doğrulanıyor")
        async with AsyncSessionLocal() as session:
            q_sample = await _load_and_verify_sample(session, questioned_sample_id)
            ref_samples = [await _load_and_verify_sample(session, sid) for sid in reference_sample_ids]
            
            case_stmt = select(CaseEntity).where(CaseEntity.case_id == case_id)
            c_res = await session.execute(case_stmt)
            case_entity = c_res.scalar_one_or_none()
            if not case_entity:
                raise ValueError(f"'{case_id}' vakası bulunamadı.")
            case_authority = case_entity.authority or "Cumhuriyet Başsavcılığı / Mahkeme"
            case_name = case_entity.name
            examiner_id = case_entity.examiner_id or examiner_username

        # 2. Bilişsel İzolasyon (Phase A Freeze & Phase B Unlock)
        await update_progress(25, "Bilişsel İzolasyon: Phase A (Sorgulanan Belge) Donduruluyor")
        from backend.app.api.v1.endpoints.cases import get_case_orchestrator
        orchestrator = get_case_orchestrator(case_id, examiner_id, "INIT")
        orchestrator.reset()
        orchestrator.start_phase_a(
            questioned_sample_id=questioned_sample_id,
            document_hash=q_sample["sha256_hash"],
            force_reset=True,
        )

        q_img = q_sample["image"]
        ref_imgs = [s["image"] for s in ref_samples]

        # 3. Uygunluk ve Kalite Değerlendirmesi
        await update_progress(40, "Belge Kalite ve Adli Uygunluk İncelemesi Yapılıyor")
        suitability_res = _SUITABILITY_ASSESSOR.assess(q_img)

        # 4. Ön İşleme ve Öznitelik Çıkarımı
        await update_progress(55, "Deterministik Ön İşleme ve Morfolojik Öznitelikler Çıkarılıyor")
        q_artifacts = _PREPROCESSOR.process(q_img)
        ref_artifacts_list = [_PREPROCESSOR.process(img) for img in ref_imgs]

        q_features = _FEATURE_EXTRACTOR.extract_all(q_artifacts.skeleton, q_artifacts.binarized)
        ref_features_list = [
            _FEATURE_EXTRACTOR.extract_all(art.skeleton, art.binarized) for art in ref_artifacts_list
        ]
        complexity_res = _COMPLEXITY_SCORER.score(q_artifacts.skeleton, q_artifacts.binarized)

        # Phase A freeze
        orchestrator.freeze_phase_a(
            features=q_features.structural.model_dump(),
            examiner_notes="Asenkron adli analiz kapsamında otomatik Phase A dondurma.",
            document_hash=q_sample["sha256_hash"],
        )
        orchestrator.unlock_phase_b(reference_sample_ids=reference_sample_ids)

        # 5. Adli Karşılaştırma & Derin Metrik (ONNX CPU / PyTorch Fallback)
        await update_progress(70, "ONNX CPU Hızlandırmalı Derin Metrik ve Vektör Karşılaştırması")
        comp_report = _COMPARISON_ENGINE.compare(q_features, ref_features_list)

        # ONNX veya PyTorch Fallback ile embedding çıkarımı
        # Sprint 17.4 & 17.5: Ortak Preprocessing Sözleşmesi ve Normalize Benzerlik Standardı
        q_tensor = preprocess_for_inference(q_artifacts.normalized_crop)
        q_emb_np = _ONNX_ENGINE.encode(q_tensor)

        ref_sims = []
        for r_art in ref_artifacts_list:
            r_tensor = preprocess_for_inference(r_art.normalized_crop)
            r_emb_np = _ONNX_ENGINE.encode(r_tensor)
            # cosine similarity on normalized vectors -> [0.0, 1.0]
            sim = compute_normalized_similarity(q_emb_np[0], r_emb_np[0])
            ref_sims.append(sim)

        deep_similarity = float(np.mean(ref_sims)) if ref_sims else 0.5

        # 6. Hibrit Füzyon, Taklit Tespiti, Bayesyen Hipotez
        await update_progress(85, "Hibrit Füzyon, Taklit Analizi ve ENFSI Bayesyen Değerlendirme")
        fusion_res = _FUSION_LAYER.fuse(
            morphological_report=comp_report,
            deep_similarity_score=deep_similarity,
        )
        disguise_res = _DISGUISE_DETECTOR.analyze(q_features, comparison_report=comp_report)
        xai_res = _DUAL_EXPLAINER.explain(
            comparison_report=comp_report,
            encoder=_CONVNEXT_ENCODER,
            image_tensor=q_tensor,
        )
        hypothesis_res = _HYPOTHESIS_EVALUATOR.evaluate(
            hybrid_result=fusion_res,
            reference_count=len(ref_imgs),
            suitability_result=suitability_res,
            suspect_name=suspect_name or case_name or "Şüpheli / Mukayese Şahıs",
        )

        # 7. Adli Rapor ve Kriptografik Mühürleme
        await update_progress(95, "Adli Rapor Oluşturuluyor ve Merkle Ağacı ile Mühürleniyor")
        case_info_bundle = {
            "case_id": case_id,
            "authority": case_authority,
            "examiner_id": examiner_id,
            "questioned_id": questioned_sample_id,
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
        )

        # Merkle Tree Hesabı
        merkle_tree = MerkleAuditTree([
            {"event": "PHASE_A_FROZEN", "case_id": case_id, "sample_q": questioned_sample_id},
            {"event": "PHASE_B_UNLOCKED", "references": reference_sample_ids},
            {"event": "SUITABILITY", "score": suitability_res.overall_score},
            {"event": "COMPLEXITY", "index": complexity_res.overall_complexity_index},
            {"event": "COMPARISON", "similarity": comp_report.similarity_score},
            {"event": "FUSION", "hybrid_score": fusion_res.hybrid_similarity_score},
            {"event": "DISGUISE", "level": disguise_res.indicator_level.value},
            {"event": "HYPOTHESIS", "log_lr": hypothesis_res.log_likelihood_ratio},
            {"event": "REPORT_SEAL", "sha256": report_result.report_sha256},
        ])
        merkle_root = merkle_tree.get_root_hash()

        # Veritabanına Kalıcı Kayıt
        async with AsyncSessionLocal() as session:
            case_stmt = select(CaseEntity).where(CaseEntity.case_id == case_id)
            c_res = await session.execute(case_stmt)
            c_entity = c_res.scalar_one_or_none()
            if c_entity:
                c_entity.state = "EVALUATION_COMPLETED"
                c_entity.current_stage = "REPORT"
                c_entity.status = "COMPLETED"
                c_entity.merkle_root_hash = merkle_root
                c_entity.similarity_score = fusion_res.hybrid_similarity_score
                c_entity.verbal_scale = hypothesis_res.verbal_conclusion_tr or hypothesis_res.support_level.value
                c_entity.audit_hash = report_result.report_sha256
                c_entity.updated_at_utc = datetime.now(timezone.utc).isoformat()

            now_iso = datetime.now(timezone.utc).isoformat()
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
                created_at_utc=now_iso,
            )
            session.add(analysis_entity)
            await session.commit()

        final_result = {
            "case_id": case_id,
            "similarity_score": fusion_res.hybrid_similarity_score,
            "verbal_conclusion": hypothesis_res.verbal_conclusion_tr,
            "merkle_root_hash": merkle_root,
            "report_sha256": report_result.report_sha256,
            "html_report_path": report_result.html_path,
            "pdf_report_path": report_result.pdf_path,
            "suitability": suitability_res.model_dump(),
            "complexity": complexity_res.model_dump(),
            "comparison": comp_report.model_dump(),
            "disguise": disguise_res.model_dump(),
            "fusion": fusion_res.model_dump(),
            "xai": xai_res.model_dump(),
            "hypothesis": hypothesis_res.model_dump(),
            "report_paths": {
                "html_path": report_result.html_path,
                "pdf_path": report_result.pdf_path,
                "report_sha256": report_result.report_sha256,
            },
            "audit_checksum_sha256": report_result.report_sha256,
        }

        await update_progress(100, "Adli Karar Destek Analizi Başarıyla Tamamlandı", status="COMPLETED", result_data=final_result)
        logger.info(f"Asenkron adli analiz işi tamamlandı: [Job: {job_id}, Vaka: {case_id}, Merkle: {merkle_root[:10]}]")

    except Exception as e:
        err = f"{type(e).__name__}: {str(e)}"
        logger.error(f"Asenkron analiz işi başarısız: [Job: {job_id}] {err}\n{traceback.format_exc()}")
        await update_progress(100, "Analiz Sırasında Adli Güvenlik / İşlem Hatası Oluştu", status="FAILED", err_msg=err)
