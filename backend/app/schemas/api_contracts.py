from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.core.context_isolation import IsolationState
from cv.preprocessing.suitability import SuitabilityResult
from cv.features.complexity import ComplexityResult
from ml.models.comparison import ComparisonReport
from ml.models.disguise import SimulationIndicators
from ml.models.fusion import HybridFusionResult
from ml.explainability.explainer import DualExplanationReport
from ml.evaluation.hypothesis import HypothesisEvaluationReport


# ==============================================================================
# Vaka (Case) Yaşam Döngüsü ve Tipleri
# ==============================================================================

class ExaminationType(str, Enum):
    """Adli İnceleme Türü"""
    HANDWRITING = "HANDWRITING"                       # El Yazısı İncelemesi
    SIGNATURE = "SIGNATURE"                           # İmza İncelemesi
    HANDWRITING_AND_SIGNATURE = "HANDWRITING_AND_SIGNATURE" # El Yazısı + İmza
    OTHER = "OTHER"                                   # Diğer


class CaseStatus(str, Enum):
    """Adli Vaka Durumu"""
    DRAFT = "DRAFT"               # Taslak
    IN_PROGRESS = "IN_PROGRESS"   # Analiz Devam Ediyor
    COMPLETED = "COMPLETED"       # Tamamlandı
    ARCHIVED = "ARCHIVED"         # Arşivlendi


class CaseStage(str, Enum):
    """Adli Analiz Aşaması"""
    CASE_CREATED = "CASE_CREATED"                     # Vaka Oluşturuldu
    SUITABILITY = "SUITABILITY"                       # Görüntü Kalite ve Uygunluk
    PREPROCESSING = "PREPROCESSING"                   # Ön İşleme ve Normalizasyon
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"         # Öznitelik Çıkarımı
    COMPARISON = "COMPARISON"                         # Karşılaştırmalı İnceleme
    COMPLEXITY_ANALYSIS = "COMPLEXITY_ANALYSIS"       # Karmaşıklık Analizi
    SIMULATION_ANALYSIS = "SIMULATION_ANALYSIS"       # Taklit / Simülasyon Tespiti
    EVALUATION = "EVALUATION"                         # Bayesyen Değerlendirme
    REPORT = "REPORT"                                 # Bilirkişi Raporu
    COMPLETED = "COMPLETED"                           # İnceleme Tamamlandı


# ==============================================================================
# Vaka (Case) Yönetim Sözleşmeleri
# ==============================================================================

class CaseCreate(BaseModel):
    """Yeni adli vaka oluşturma isteği."""
    name: Optional[str] = Field(None, description="Vaka Adı (Boşsa otomatik üretilir)")
    description: Optional[str] = Field(None, description="Vaka Açıklaması (Boşsa varsayılan açıklama atanır)")
    examination_type: Optional[ExaminationType] = Field(default=ExaminationType.SIGNATURE, description="İnceleme Türü")
    case_id: Optional[str] = Field(None, description="Benzersiz adli dosya/vaka numarası (boşsa otomatik üretilir)")
    case_number: Optional[str] = Field(None, description="Vaka Numarası (case_number)")
    examiner_name: Optional[str] = Field(None, description="Bilirkişi Adı")
    suspect_name: Optional[str] = Field(None, description="Şüpheli / Sanık Adı")
    similarity_score: Optional[float] = Field(None, description="Hibrit benzerlik skoru")
    verbal_scale: Optional[str] = Field(None, description="Sözel değerlendirme ölçeği")
    audit_hash: Optional[str] = Field(None, description="Denetim SHA-256 karması")
    reference_number: Optional[str] = Field(None, description="Mahkeme / Savcılık Esas Numarası")
    authority: Optional[str] = Field("Cumhuriyet Başsavcılığı / Mahkeme", description="İncelemeyi talep eden adli makam")
    examiner_id: Optional[str] = Field("BILIRKISI_UZMAN_01", description="İncelemeyi yürüten uzman bilirkişi")
    notes: Optional[str] = Field("", description="Vaka hakkında başlangıç notları")
    status: Optional[Any] = Field(default=CaseStatus.IN_PROGRESS, description="Vaka Durumu")
    current_stage: Optional[CaseStage] = Field(default=CaseStage.CASE_CREATED, description="Mevcut Analiz Aşaması")


class CaseUpdate(BaseModel):
    """Vaka güncelleme isteği."""
    name: Optional[str] = None
    description: Optional[str] = None
    examination_type: Optional[ExaminationType] = None
    case_number: Optional[str] = None
    examiner_name: Optional[str] = None
    suspect_name: Optional[str] = None
    similarity_score: Optional[float] = None
    verbal_scale: Optional[str] = None
    audit_hash: Optional[str] = None
    reference_number: Optional[str] = None
    authority: Optional[str] = None
    examiner_id: Optional[str] = None
    notes: Optional[str] = None
    status: Optional[Any] = None
    current_stage: Optional[CaseStage] = None


class CaseResponse(BaseModel):
    """Vaka durum ve künye yanıtı."""
    case_id: str
    case_number: Optional[str] = None
    name: str
    description: str
    examination_type: ExaminationType
    reference_number: Optional[str] = None
    authority: Optional[str] = None
    examiner_id: str
    examiner_name: Optional[str] = None
    suspect_name: Optional[str] = None
    similarity_score: Optional[float] = None
    verbal_scale: Optional[str] = None
    audit_hash: Optional[str] = None
    notes: Optional[str] = ""
    status: Any = CaseStatus.IN_PROGRESS
    current_stage: CaseStage = CaseStage.CASE_CREATED
    state: IsolationState = IsolationState.PHASE_A_INIT
    questioned_sample_id: Optional[str] = None
    reference_sample_ids: List[str] = Field(default_factory=list)
    created_at_utc: str
    updated_at_utc: str


class SampleUploadResponse(BaseModel):
    """Adli belge/örnek yükleme yanıtı."""
    sample_id: str
    case_id: str
    filename: str
    sample_type: str = Field(..., description="'questioned' veya 'reference'")
    sha256_hash: str
    file_size_bytes: int


# ==============================================================================
# Analiz ve İzolasyon Sözleşmeleri
# ==============================================================================

class SuitabilityResponse(BaseModel):
    """Görüntü kalite ve adli uygunluk değerlendirme yanıtı."""
    sample_id: Optional[str] = None
    suitability: SuitabilityResult


class PhaseAIsolateRequest(BaseModel):
    """Phase A (Sorgulanan Belge) öznitelik kilitleme ve dondurma isteği."""
    case_id: str
    questioned_sample_id: str
    examiner_notes: Optional[str] = ""


class PhaseAIsolateResponse(BaseModel):
    """Phase A dondurulmuş kanıt mühür yanıtı."""
    case_id: str
    questioned_sample_id: str
    integrity_hash_sha256: str
    frozen_at_utc: str
    features: Dict[str, Any]


class FullPipelineRequest(BaseModel):
    """Uçtan uca adli karar destek analiz istek gövdesi."""
    case_id: str
    questioned_sample_id: str
    reference_sample_ids: List[str] = Field(..., min_length=1)
    suspect_name: Optional[str] = "Şüpheli / Referans Şahıs"
    generate_pdf: Optional[bool] = True


class FullPipelineResponse(BaseModel):
    """Uçtan uca adli karar destek tam analiz raporu."""
    case_id: str
    suitability: SuitabilityResult
    complexity: ComplexityResult
    comparison: ComparisonReport
    disguise: SimulationIndicators
    fusion: HybridFusionResult
    xai: DualExplanationReport
    hypothesis: HypothesisEvaluationReport
    report_paths: Dict[str, Optional[str]]
    audit_checksum_sha256: str
    merkle_root_hash: Optional[str] = None
    model_metadata: Optional[Dict[str, Any]] = None


# ==============================================================================
# Karşıtlı Açıklama (Counterfactual XAI) Sözleşmeleri
# ==============================================================================

class CounterfactualRequest(BaseModel):
    """Karşıtlı açıklama ve 'Ne Olurdu?' simülasyon istek gövdesi."""
    case_id: str
    custom_perturbations: Optional[Dict[str, float]] = Field(default_factory=dict, description="Öznitelik değişim değerleri")
    deep_similarity_score: Optional[float] = Field(default=0.85, description="Baz derin benzerlik skoru")


class CounterfactualResponse(BaseModel):
    """Karşıtlı açıklama simülasyon yanıtı."""
    case_id: str
    original_score: float
    simulated_hybrid_score: float
    simulated_morphological_score: float
    gating_penalty: float
    gating_triggered: bool
    structural_contradictions: List[str]
    concordance_level: str
    sensitivity_matrix: Dict[str, float]
    top_scenarios: List[Dict[str, Any]]
    execution_time_ms: float


# ==============================================================================
# Dinamik Biyometrik Zaman Serisi Sözleşmeleri
# ==============================================================================

class BiometricPointPayload(BaseModel):
    """Tekil tablet noktası."""
    x: float
    y: float
    t: float
    pressure: float = 0.5
    azimuth_deg: float = 0.0
    altitude_deg: float = 90.0
    is_pen_down: bool = True


class BiometricCompareRequest(BaseModel):
    """Dinamik imza karşılaştırma isteği."""
    case_id: Optional[str] = "DYNAMIC_CASE"
    questioned_points: List[BiometricPointPayload]
    reference_points: List[BiometricPointPayload]


class BiometricCompareResponse(BaseModel):
    """Dinamik imza karşılaştırma yanıtı."""
    dtw_distance: float
    dtw_normalized_similarity: float
    velocity_correlation: float
    pressure_correlation: float
    in_air_time_diff: float
    is_biometrically_consistent: bool
    verbal_assessment_tr: str
    questioned_profile: Dict[str, Any]
    reference_profile: Dict[str, Any]

