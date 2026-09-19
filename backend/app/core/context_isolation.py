import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.app.core.logger import log_audit_event, logger


class IsolationState(str, Enum):
    """Bilişsel İzolasyon Durum Makinesi Statüleri."""
    PHASE_A_INIT = "PHASE_A_INIT"          # Phase A başlatıldı, veri bekliyor
    PHASE_A_ANALYSIS = "PHASE_A_ANALYSIS"  # Sorgulanan belge inceleniyor
    PHASE_A_FROZEN = "PHASE_A_FROZEN"      # Phase A öznitelikleri donduruldu (Kilitli)
    PHASE_B_ACTIVE = "PHASE_B_ACTIVE"      # Referans belgeler açıldı ve karşılaştırma aktif
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED" # Değerlendirme tamamlandı


class IsolationViolationError(Exception):
    """
    Bilişsel İzolasyon ve Adli Prosedür İhlali Hatası.
    
    Phase A analizi tamamlanıp dondurulmadan Phase B verilerinin açılması veya
    dondurulmuş Phase A verilerinin değiştirilmesi durumunda fırlatılır.
    """
    pass


class FrozenEvidence(BaseModel):
    """Dondurulmuş Phase A Adli Kanıt ve Öznitelik Paketi."""
    case_id: str
    questioned_sample_id: str
    features: Dict[str, Any]
    examiner_notes: Optional[str] = ""
    frozen_at_utc: str
    integrity_hash_sha256: str


class ContextIsolationOrchestrator:
    """
    Adli El Yazısı Bilişsel İzolasyon Durum Makinesi (Context Isolation Orchestrator).
    
    Adli bilişimde 'Cognitive Bias' (Bilişsel Yanlılık) riskini bertaraf etmek için:
    1. Sorgulanan belge (Questioned Sample - Phase A) bağımsız olarak analiz edilir.
    2. Phase A öznitelikleri SHA-256 imzası ile dondurulur (freeze).
    3. Phase A kilitlenmeden Referans Belgelere (Reference Samples - Phase B) erişim engellenir.
    4. Dondurulmuş kanıtların sonradan değiştirilemezliği (tamper-proof) denetlenir.
    """

    def __init__(self, case_id: str, examiner_id: str = "SYSTEM"):
        self.case_id = case_id
        self.examiner_id = examiner_id
        self.state = IsolationState.PHASE_A_INIT
        self.questioned_sample_id: Optional[str] = None
        self.reference_sample_ids: List[str] = []
        self.frozen_evidence: Optional[FrozenEvidence] = None

        log_audit_event(
            event_type="CONTEXT_ISOLATION_INIT",
            action="INITIALIZE_CASE",
            examiner_id=self.examiner_id,
            details={"case_id": self.case_id, "initial_state": self.state.value},
        )

    def reset(self) -> None:
        """Yeni bir inceleme veya numune değişimi için durum makinesini sıfırlar."""
        self.state = IsolationState.PHASE_A_INIT
        self.questioned_sample_id = None
        self.frozen_evidence = None

        log_audit_event(
            event_type="CONTEXT_ISOLATION_RESET",
            action="RESET_CASE_ISOLATION",
            examiner_id=self.examiner_id,
            details={"case_id": self.case_id, "new_state": self.state.value},
        )

    def start_phase_a(self, questioned_sample_id: str, document_hash: str, force_reset: bool = False) -> None:
        """Sorgulanan belgenin (Phase A) analiz sürecini başlatır."""
        if self.state not in (IsolationState.PHASE_A_INIT, IsolationState.PHASE_A_ANALYSIS):
            if force_reset:
                self.reset()
            else:
                raise IsolationViolationError(
                    f"Geçersiz durum geçişi: Mevcut durum '{self.state.value}' iken Phase A başlatılamaz."
                )

        self.questioned_sample_id = questioned_sample_id
        self.state = IsolationState.PHASE_A_ANALYSIS

        log_audit_event(
            event_type="PHASE_A_START",
            action="START_QUESTIONED_ANALYSIS",
            examiner_id=self.examiner_id,
            document_hash=document_hash,
            details={"case_id": self.case_id, "questioned_sample_id": questioned_sample_id},
        )

    def freeze_phase_a(
        self,
        features: Dict[str, Any],
        examiner_notes: str = "",
        document_hash: Optional[str] = None,
    ) -> FrozenEvidence:
        """
        Phase A özniteliklerini ve analiz bulgularını SHA-256 ile dondurur.
        Bu adımdan sonra Phase A öznitelikleri değiştirilemez.
        """
        if self.state != IsolationState.PHASE_A_ANALYSIS:
            raise IsolationViolationError(
                f"Phase A dondurulamadı: Durum '{self.state.value}' (Beklenen: {IsolationState.PHASE_A_ANALYSIS.value})."
            )

        if not self.questioned_sample_id:
            raise IsolationViolationError("Sorgulanan belge kimliği (questioned_sample_id) tanımlı değil.")

        timestamp_utc = datetime.now(timezone.utc).isoformat()
        
        # Bütünlük karmasını hesapla
        payload = {
            "case_id": self.case_id,
            "questioned_sample_id": self.questioned_sample_id,
            "features": features,
            "examiner_notes": examiner_notes,
            "frozen_at_utc": timestamp_utc,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        integrity_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        frozen = FrozenEvidence(
            case_id=self.case_id,
            questioned_sample_id=self.questioned_sample_id,
            features=features,
            examiner_notes=examiner_notes,
            frozen_at_utc=timestamp_utc,
            integrity_hash_sha256=integrity_hash,
        )

        self.frozen_evidence = frozen
        self.state = IsolationState.PHASE_A_FROZEN

        log_audit_event(
            event_type="PHASE_A_FROZEN",
            action="FREEZE_QUESTIONED_FEATURES",
            examiner_id=self.examiner_id,
            document_hash=document_hash,
            details={
                "case_id": self.case_id,
                "integrity_hash": integrity_hash,
                "feature_count": len(features),
            },
        )

        return frozen

    def unlock_phase_b(self, reference_sample_ids: List[str]) -> bool:
        """
        Phase B'yi (Referans Belgeler) kullanıma açar veya referans listesini günceller.
        
        Güvenlik Kuralı: Phase A dondurulmamışsa (PHASE_A_INIT / PHASE_A_ANALYSIS) kesinlikle izin vermez.
        """
        valid_states = (
            IsolationState.PHASE_A_FROZEN,
            IsolationState.PHASE_B_ACTIVE,
            IsolationState.EVALUATION_COMPLETED,
        )
        if self.state not in valid_states:
            log_audit_event(
                event_type="ISOLATION_VIOLATION_ATTEMPT",
                action="UNAUTHORIZED_PHASE_B_ACCESS",
                examiner_id=self.examiner_id,
                details={
                    "case_id": self.case_id,
                    "attempted_state": self.state.value,
                    "reason": "Phase A dondurulmadan Phase B referansları açılamaz.",
                },
            )
            raise IsolationViolationError(
                f"Bilişsel İzolasyon İhlali: Phase A dondurulmadan ({IsolationState.PHASE_A_FROZEN.value}) "
                f"Phase B referans belgelerine erişilemez! Mevcut durum: '{self.state.value}'."
            )

        if not reference_sample_ids:
            raise ValueError("En az bir referans belge kimliği (reference_sample_id) sağlanmalıdır.")

        # Phase A verisinin bozulmadığını son kez doğrula
        if not self.verify_integrity():
            raise IsolationViolationError("Phase A veri bütünlük doğrulaması başarısız oldu! Kanıt değiştirilmiş olabilir.")

        self.reference_sample_ids = list(reference_sample_ids)
        self.state = IsolationState.PHASE_B_ACTIVE

        log_audit_event(
            event_type="PHASE_B_UNLOCKED",
            action="UNLOCK_REFERENCE_COMPARISON",
            examiner_id=self.examiner_id,
            details={
                "case_id": self.case_id,
                "reference_count": len(self.reference_sample_ids),
                "phase_a_hash": self.frozen_evidence.integrity_hash_sha256 if self.frozen_evidence else None,
            },
        )

        return True

    def complete_evaluation(self) -> None:
        """Tüm karşılaştırma ve değerlendirme sürecini tamamlar."""
        if self.state != IsolationState.PHASE_B_ACTIVE:
            raise IsolationViolationError(
                f"Değerlendirme tamamlanamadı: Durum '{self.state.value}' (Beklenen: {IsolationState.PHASE_B_ACTIVE.value})."
            )

        self.state = IsolationState.EVALUATION_COMPLETED

        log_audit_event(
            event_type="EVALUATION_COMPLETED",
            action="FINALIZE_CASE",
            examiner_id=self.examiner_id,
            details={"case_id": self.case_id},
        )

    def verify_integrity(self) -> bool:
        """Dondurulmuş Phase A kanıtının SHA-256 karması ile bütünlüğünü doğrular."""
        if not self.frozen_evidence:
            return False

        payload = {
            "case_id": self.frozen_evidence.case_id,
            "questioned_sample_id": self.frozen_evidence.questioned_sample_id,
            "features": self.frozen_evidence.features,
            "examiner_notes": self.frozen_evidence.examiner_notes,
            "frozen_at_utc": self.frozen_evidence.frozen_at_utc,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        recalculated_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

        return recalculated_hash == self.frozen_evidence.integrity_hash_sha256
