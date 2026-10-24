"""
FHDSS SPRINT 25 — Phase 13 & 14 Test: Real End-to-End Production Case Integration
Executes full forensic pipeline from raw image files to PDF/HTML report:
Input -> Suitability -> Feature Extraction -> V5 ONNX -> 128D Embedding ->
Cosine -> Mahalanobis -> Forensic Fusion -> Calibration -> ENFSI LR -> XAI -> Report
"""
import pytest
import numpy as np
from pathlib import Path
import cv2

from backend.app.services.onnx_inference import ForensicONNXInferenceEngine
from backend.app.services.report_service import ForensicReportService
from cv.preprocessing.suitability import SuitabilityAssessor
from cv.features.extractors import ForensicFeatureExtractor
from cv.features.complexity import SignatureComplexityScorer
from ml.evaluation.hypothesis import ForensicHypothesisEvaluator
from ml.explainability.explainer import DualForensicExplainer
from ml.models.comparison import ForensicComparisonEngine
from ml.models.disguise import ForensicDisguiseDetector
from ml.models.fusion import HybridForensicFusion
from ml.models.registry import model_registry
from ml.models.siamese import ForensicConvNeXtEncoder, compute_normalized_similarity, compute_cosine_similarity
from ml.preprocessing.contract import preprocess_for_inference

def test_v5_end_to_end_pipeline():
    # 1. Load real images (CEDAR sample)
    cedar_root = Path("datasets/DS_Downloaded/CEDAR/CEDAR1/signatures/full_org")
    q_path = cedar_root / "original_1_1.png"
    r_paths = [cedar_root / "original_1_2.png", cedar_root / "original_1_3.png", cedar_root / "original_1_4.png"]
    
    assert q_path.exists(), "Questioned image not found"
    for r_p in r_paths:
        assert r_p.exists(), f"Reference image not found: {r_p}"
        
    q_img = cv2.imdecode(np.frombuffer(q_path.read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    ref_imgs = [cv2.imdecode(np.frombuffer(p.read_bytes(), np.uint8), cv2.IMREAD_COLOR) for p in r_paths]
    
    # 2. Suitability check
    suit_engine = SuitabilityAssessor()
    suit_res = suit_engine.assess(q_img)
    assert suit_res.status is not None
    
    # 3. Deterministic Preprocessing & Morphological Feature Extraction
    from cv.preprocessing.pipeline import ForensicImagePreprocessor
    preprocessor = ForensicImagePreprocessor()
    q_artifacts = preprocessor.process(q_img)
    ref_artifacts_list = [preprocessor.process(img) for img in ref_imgs]
    
    feat_extractor = ForensicFeatureExtractor()
    q_features = feat_extractor.extract_all(q_artifacts.skeleton, q_artifacts.binarized)
    ref_features_list = [
        feat_extractor.extract_all(art.skeleton, art.binarized) for art in ref_artifacts_list
    ]
        
    # 4. Complexity Scoring
    comp_scorer = SignatureComplexityScorer()
    complexity_res = comp_scorer.score(q_artifacts.skeleton, q_artifacts.binarized)
    assert complexity_res.overall_complexity_index > 0.0
    
    # 5. Morphological Comparison
    comp_engine = ForensicComparisonEngine()
    comp_res = comp_engine.compare(q_features, ref_features_list)
    assert comp_res.similarity_score > 0.0
    
    # 6. ONNX Model Inference (V5)
    encoder = ForensicConvNeXtEncoder(embedding_dim=128, pretrained=False)
    onnx_engine = ForensicONNXInferenceEngine(pytorch_model=encoder)
    assert onnx_engine.onnx_model_path.name == "convnext_tiny_forensic_multidataset_v5.onnx"
    
    q_tensor = preprocess_for_inference(q_img, is_bgr=True)
    q_emb_np = onnx_engine.encode(q_tensor)
    assert q_emb_np.shape == (1, 128)
    
    ref_sims = []
    ref_raw_cosines = []
    for r_img in ref_imgs:
        r_tensor = preprocess_for_inference(r_img, is_bgr=True)
        r_emb_np = onnx_engine.encode(r_tensor)
        norm_sim = compute_normalized_similarity(q_emb_np[0], r_emb_np[0])
        raw_cos = compute_cosine_similarity(q_emb_np[0], r_emb_np[0])
        ref_sims.append(norm_sim)
        ref_raw_cosines.append(raw_cos)
        
    deep_sim = float(np.mean(ref_sims))
    raw_cos_avg = float(np.mean(ref_raw_cosines))
    
    # 7. Forensic Hybrid Fusion
    fusion_engine = HybridForensicFusion(weight_morphological=0.45, weight_deep=0.55)
    fusion_res = fusion_engine.fuse(comp_res, deep_similarity_score=deep_sim, raw_cosine=raw_cos_avg)
    assert fusion_res.hybrid_similarity_score > 0.0
    assert fusion_res.score_provenance is not None
    
    # 8. Disguise & Simulation Detection
    disguise_detector = ForensicDisguiseDetector()
    disguise_res = disguise_detector.analyze(q_features, comparison_report=comp_res)
    assert disguise_res is not None
    
    # 9. Dual XAI Explainer
    xai_engine = DualForensicExplainer()
    xai_res = xai_engine.explain(
        comparison_report=comp_res,
        encoder=encoder,
        image_tensor=q_tensor,
    )
    assert xai_res is not None
    
    # 10. ENFSI Bayesian Hypothesis
    hyp_eval = ForensicHypothesisEvaluator()
    hyp_res = hyp_eval.evaluate(
        hybrid_result=fusion_res,
        reference_count=len(ref_imgs),
        suitability_result=suit_res,
        suspect_name="Test Şüpheli",
    )
    assert hyp_res.likelihood_ratio_estimate > 0.0
    
    # 11. Report Generation
    active_meta = model_registry.get_active_model_metadata()
    assert active_meta["model_id"] == "convnext_tiny_forensic_multidataset_v5"
    
    report_service = ForensicReportService()
    case_info = {
        "case_id": "TEST_CASE_V5_E2E",
        "examiner_id": "BILIRKISI_UZMAN_01",
        "model_id": active_meta["model_id"],
        "model_sha256": active_meta["model_sha256"],
        "model_version": active_meta["manifest_version"],
    }
    
    rep_res = report_service.generate_report(
        case_info=case_info,
        suitability=suit_res,
        comparison=comp_res,
        complexity=complexity_res,
        disguise=disguise_res,
        fusion=fusion_res,
        xai=xai_res,
        hypothesis=hyp_res,
    )
    assert rep_res.html_path is not None
    assert Path(rep_res.html_path).exists()
    assert rep_res.report_sha256 is not None
