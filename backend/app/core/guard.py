from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
from fastapi import HTTPException, status
from pydantic import BaseModel, Field


class AllowedForensicScope(str, Enum):
    """FHDSS Tarafından Desteklenen Bilimsel ve Yasal Adli İnceleme Kapsamları."""
    AUTHORSHIP_COMPARISON = "authorship_comparison"         # Yazar aidiyeti karşılaştırması
    FORENSIC_COMPARISON = "forensic_comparison"             # Adli karşılaştırmalı inceleme
    MORPHOLOGICAL_ANALYSIS = "morphological_analysis"       # Morfolojik ve topolojik analiz
    SIGNATURE_COMPARISON = "signature_comparison"           # İmza aidiyeti ve simülasyon analizi
    DOCUMENT_ANALYSIS = "document_analysis"                 # Adli belge ve yazı analizi
    HANDWRITING = "HANDWRITING"                             # El yazısı incelemesi
    SIGNATURE = "SIGNATURE"                                 # İmza incelemesi
    HANDWRITING_AND_SIGNATURE = "HANDWRITING_AND_SIGNATURE" # El yazısı ve imza incelemesi
    OTHER = "OTHER"                                         # Diğer standart adli belge incelemeleri


class PsychologicalInferenceViolationError(Exception):
    """Adli Bilimsel Kapsam Dışı Analiz (Psikanaliz/Kişilik Çıkarımı) İhlali Hatası."""
    def __init__(self, detected_term: str, message: str):
        super().__init__(message)
        self.detected_term = detected_term
        self.message = message


class PsychologicalInferenceGuard:
    """
    FHDSS Adli Kapsam ve Bilimsel Sınır Koruma Motoru (Scope Guard).
    
    Adli Bilimsel Prensipler:
    - FHDSS; yalnızca Adli Belge İnceleme (Forensic Document Examination - FDE) disiplinine uygun,
      biyomekanik, nöromusküler ve morfolojik aidiyet ve sahtecilik analizleri yapar.
    - El yazısından kişilik analizi, karakter tahlili, ruhsal durum veya zekâ/dürüstlük çıkarımı
      yapma iddiaları (Grafoloji) sözdebilim (pseudoscience) olup Daubert / ENFSI standartlarına aykırıdır.
    - Bu koruma katmanı, backend, API ve model seviyesinde adli kapsam dışı istekleri kesin olarak engeller.
    """

    # Kesinlikle Yasaklanan Kapsam ve Terim Kümeleri
    FORBIDDEN_TERMS: Set[str] = {
        "personality",
        "character",
        "psychological_state",
        "mental_state",
        "personality_traits",
        "graphology_claims",
        "mood",
        "intelligence",
        "temperament",
        "honesty",
        "psikanaliz",
        "kisilik",
        "kişilik",
        "karakter",
        "ruhsal_durum",
        "ruh_hali",
        "psikolojik_durum",
        "psikoloji",
        "kisilik_analizi",
        "kişilik_analizi",
        "karakter_analizi",
        "grafoloji",
        "graphology",
    }

    # İzin Verilen Standart Adli Kapsamlar
    ALLOWED_SCOPES: Set[str] = {scope.value for scope in AllowedForensicScope}

    SCIENTIFIC_REJECTION_MESSAGE = (
        "Bu analiz türü FHDSS'in adli belge inceleme kapsamı dışındadır. "
        "Sistem kişilik, karakter veya psikolojik durum çıkarımı yapmaz."
    )

    SCIENTIFIC_RATIONALE = (
        "El yazısından kişilik veya ruhsal durum tahlili yapma iddiaları (Grafoloji), "
        "uluslararası adli bilim standartları (ENFSI / ASTM E2290 / Daubert) uyarınca "
        "sözdebilim (pseudoscience) kabul edilir ve mahkemelerde hiçbir delil geçerliliği yoktur. "
        "FHDSS karar destek sistemi yalnızca ölçülebilir biyomekanik, kinematik ve "
        "morfolojik aidiyet/sahtecilik analizleri gerçekleştirmektedir."
    )

    @classmethod
    def validate_text(cls, text: Optional[str]) -> Tuple[bool, Optional[str]]:
        """
        Metin içerisinde yasaklı psikanaliz/karakter analizi anahtar kelimelerini denetler.
        """
        if not text:
            return True, None

        normalized_text = text.lower().replace("-", " ").replace("_", " ")
        words = set(normalized_text.split())

        sorted_forbidden = sorted(cls.FORBIDDEN_TERMS, key=len, reverse=True)
        for forbidden in sorted_forbidden:
            forbidden_clean = forbidden.replace("_", " ")
            if forbidden_clean in normalized_text or forbidden in words:
                return False, forbidden

        return True, None

    @classmethod
    def enforce_scope(
        cls,
        examination_type: Optional[str] = None,
        context_text: Optional[str] = None,
    ) -> None:
        """
        Talebin adli bilimsel kapsamda olup olmadığını doğrular; ihlal varsa HTTP 400 fırlatır.
        """
        # 1. İnceleme Türü Kontrolü
        if examination_type:
            is_valid_text, forbidden_term = cls.validate_text(examination_type)
            if not is_valid_text:
                cls._raise_http_violation(forbidden_term)

        # 2. Vaka Notları ve Açıklama Metni Kontrolü
        if context_text:
            is_valid_text, forbidden_term = cls.validate_text(context_text)
            if not is_valid_text:
                cls._raise_http_violation(forbidden_term)

    @classmethod
    def validate_request(cls, user_query: Optional[str] = None, endpoint_name: str = "") -> None:
        """Kullanıcı sorgusunda psikolojik çıkarım/grafoloji olup olmadığını denetler."""
        cls.enforce_scope(context_text=user_query)

    @classmethod
    def _raise_http_violation(cls, detected_term: Optional[str] = None) -> None:
        """Kapsam dışı istek için yapılandırılmış HTTP 400 Bad Request fırlatır."""
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "OUT_OF_FORENSIC_SCOPE",
                "message": cls.SCIENTIFIC_REJECTION_MESSAGE,
                "detected_forbidden_term": detected_term or "Bilinmeyen Kapsam Dışı Terim",
                "scientific_rationale": cls.SCIENTIFIC_RATIONALE,
                "admissible_scopes": list(cls.ALLOWED_SCOPES),
            },
        )
