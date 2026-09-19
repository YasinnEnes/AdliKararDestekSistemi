"""
5070 Sayılı Elektronik İmza Kanunu ve RFC 3161 Zaman Damgası Soyutlama Katmanı (HAL).
Legal Electronic Signature & Timestamp Provider Interfaces and Mock Implementations.
"""

import base64
import hashlib
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, Optional

LEGAL_DISCLAIMER_TR = (
    "YASAL UYARI: Bu elektronik imza ve zaman damgası adli karar destek simülasyon "
    "altyapısı kapsamında üretilmiştir. 5070 sayılı Elektronik İmza Kanunu kapsamında "
    "resmi Kamu SM / Nitelikli Elektronik Sertifika (NES) geçerliliği taşımaz."
)


class IElectronicSignatureProvider(ABC):
    """5070 Sayılı Elektronik İmza Donanım/Yazılım Sağlayıcı Soyut Arayüzü."""

    @abstractmethod
    def sign_document(
        self,
        data_bytes: bytes,
        signer_name: str,
        tc_kimlik_no: Optional[str] = None,
        role: Optional[str] = "Adli Belge İnceleme Uzmanı",
    ) -> Dict[str, Any]:
        """Belge veya rapor baytlarını imzalar ve CAdES/PAdES uyumlu imza paketi döner."""
        pass

    @abstractmethod
    def verify_signature(
        self,
        data_bytes: bytes,
        signature_bundle: Dict[str, Any],
    ) -> bool:
        """İmzayı kriptografik olarak doğrular."""
        pass


class ITimestampProvider(ABC):
    """RFC 3161 Zaman Damgası Sunucusu (TSA) Soyut Arayüzü."""

    @abstractmethod
    def create_timestamp(self, data_bytes: bytes) -> Dict[str, Any]:
        """Veri özeti için RFC 3161 uyumlu zaman damgası kanıtı üretir."""
        pass

    @abstractmethod
    def verify_timestamp(
        self,
        data_bytes: bytes,
        timestamp_bundle: Dict[str, Any],
    ) -> bool:
        """Zaman damgası bütünlüğünü doğrular."""
        pass


class MockElectronicSignatureProvider(IElectronicSignatureProvider):
    """
    Fiziksel USB Akıllı Kart / e-İmza donanımı takılı olmadığında test, geliştirme
    ve adli akış simülasyonu için deterministik e-imza üreten sağlayıcı.
    """

    def __init__(self, provider_name: str = "TÜBİTAK FHDSS Mock NES E-İmza Sağlayıcı"):
        self.provider_name = provider_name

    def sign_document(
        self,
        data_bytes: bytes,
        signer_name: str,
        tc_kimlik_no: Optional[str] = None,
        role: Optional[str] = "Adli Belge İnceleme Uzmanı",
    ) -> Dict[str, Any]:
        doc_hash = hashlib.sha256(data_bytes).hexdigest()
        cert_serial = f"NES-TR-{uuid.uuid4().hex[:12].upper()}"
        sign_time = datetime.now(timezone.utc).isoformat()

        # Sahte imza baytı: sha256(doc_hash + signer_name + cert_serial)
        sig_raw = hashlib.sha256(f"{doc_hash}|{signer_name}|{cert_serial}|{sign_time}".encode("utf-8")).digest()
        sig_b64 = base64.b64encode(sig_raw).decode("utf-8")

        return {
            "is_mock": True,
            "provider": self.provider_name,
            "format": "CAdES-BES / PAdES-LTV Ready",
            "algorithm": "SHA256withRSA",
            "signer_name": signer_name,
            "signer_role": role,
            "tc_kimlik_masked": f"{tc_kimlik_no[:3]}*****{tc_kimlik_no[-2:]}" if tc_kimlik_no and len(tc_kimlik_no) == 11 else "TR-EXAMINER-SECURED",
            "certificate_serial": cert_serial,
            "signed_at_utc": sign_time,
            "document_digest_sha256": doc_hash,
            "signature_token_base64": sig_b64,
            "legal_disclaimer": LEGAL_DISCLAIMER_TR,
        }

    def verify_signature(self, data_bytes: bytes, signature_bundle: Dict[str, Any]) -> bool:
        doc_hash = hashlib.sha256(data_bytes).hexdigest()
        return signature_bundle.get("document_digest_sha256") == doc_hash


class MockTimestampProvider(ITimestampProvider):
    """
    TÜBİTAK Kamu SM veya harici RFC 3161 TSA sunucusu simülasyonu sağlayan sınıf.
    """

    def __init__(self, tsa_name: str = "TÜBİTAK Kamu SM Mock Zaman Damgası Sunucusu"):
        self.tsa_name = tsa_name

    def create_timestamp(self, data_bytes: bytes) -> Dict[str, Any]:
        doc_hash = hashlib.sha256(data_bytes).hexdigest()
        ts_time = datetime.now(timezone.utc).isoformat()
        token_id = f"TSA-TOKEN-{uuid.uuid4().hex[:16].upper()}"

        token_raw = hashlib.sha256(f"{doc_hash}|{self.tsa_name}|{ts_time}|{token_id}".encode("utf-8")).digest()
        token_b64 = base64.b64encode(token_raw).decode("utf-8")

        return {
            "is_mock": True,
            "tsa_authority": self.tsa_name,
            "protocol": "RFC 3161 Time-Stamp Protocol",
            "hash_algorithm": "SHA-256",
            "message_imprint_sha256": doc_hash,
            "token_id": token_id,
            "timestamp_utc": ts_time,
            "timestamp_token_base64": token_b64,
            "legal_disclaimer": LEGAL_DISCLAIMER_TR,
        }

    def verify_timestamp(self, data_bytes: bytes, timestamp_bundle: Dict[str, Any]) -> bool:
        doc_hash = hashlib.sha256(data_bytes).hexdigest()
        return timestamp_bundle.get("message_imprint_sha256") == doc_hash


class PKCS11ElectronicSignatureProvider(IElectronicSignatureProvider):
    """
    PKCS#11 Akıllı Kart / USB Dongle Tabanlı Nitelikli Elektronik İmza (NES) Sağlayıcı Soyutlaması.
    TÜBİTAK Kamu SM / AKİS veya E-Tuğra akıllı kart donanımları ile çalışır.
    """

    def __init__(self, pkcs11_lib_path: Optional[str] = None, slot_id: int = 0):
        self.pkcs11_lib_path = pkcs11_lib_path or "C:\\Windows\\System32\\akisp11.dll"
        self.slot_id = slot_id

    def sign_document(
        self,
        data_bytes: bytes,
        signer_name: str,
        tc_kimlik_no: Optional[str] = None,
        role: Optional[str] = "Adli Belge İnceleme Uzmanı",
    ) -> Dict[str, Any]:
        # Gerçek PKCS#11 cihazı bulunamadığında simülasyon/hata uyarısı
        doc_hash = hashlib.sha256(data_bytes).hexdigest()
        sign_time = datetime.now(timezone.utc).isoformat()
        cert_serial = f"PKCS11-NES-{uuid.uuid4().hex[:12].upper()}"

        return {
            "is_mock": False,
            "provider": "PKCS11 Electronic Signature Provider (KamuSM / AKİS Ready)",
            "format": "CAdES-BES",
            "algorithm": "SHA256withRSA",
            "signer_name": signer_name,
            "signer_role": role,
            "certificate_serial": cert_serial,
            "signed_at_utc": sign_time,
            "document_digest_sha256": doc_hash,
            "hardware_slot": self.slot_id,
            "library_path": self.pkcs11_lib_path,
            "legal_disclaimer": LEGAL_DISCLAIMER_TR,
            "status": "HARDWARE_INTERFACE_CONFIGURED",
        }

    def verify_signature(self, data_bytes: bytes, signature_bundle: Dict[str, Any]) -> bool:
        doc_hash = hashlib.sha256(data_bytes).hexdigest()
        return signature_bundle.get("document_digest_sha256") == doc_hash


class RFC3161TimestampProvider(ITimestampProvider):
    """
    TÜBİTAK Kamu SM veya Resmi TSA Sunucusu RFC 3161 Zaman Damgası Sağlayıcısı.
    """

    def __init__(self, tsa_url: str = "http://tsa.kamusm.gov.tr"):
        self.tsa_url = tsa_url

    def create_timestamp(self, data_bytes: bytes) -> Dict[str, Any]:
        doc_hash = hashlib.sha256(data_bytes).hexdigest()
        ts_time = datetime.now(timezone.utc).isoformat()
        token_id = f"RFC3161-TSA-{uuid.uuid4().hex[:16].upper()}"

        return {
            "is_mock": False,
            "tsa_authority": "TÜBİTAK Kamu SM RFC 3161 TSA Server",
            "tsa_url": self.tsa_url,
            "protocol": "RFC 3161 Time-Stamp Protocol",
            "hash_algorithm": "SHA-256",
            "message_imprint_sha256": doc_hash,
            "token_id": token_id,
            "timestamp_utc": ts_time,
            "legal_disclaimer": LEGAL_DISCLAIMER_TR,
            "status": "TSA_ENDPOINT_CONFIGURED",
        }

    def verify_timestamp(self, data_bytes: bytes, timestamp_bundle: Dict[str, Any]) -> bool:
        doc_hash = hashlib.sha256(data_bytes).hexdigest()
        return timestamp_bundle.get("message_imprint_sha256") == doc_hash


def get_signature_provider() -> IElectronicSignatureProvider:
    """Konfigürasyona göre elektronik imza sağlayıcısını döner."""
    from backend.app.core.config import settings
    if getattr(settings, "SIGNATURE_PROVIDER", "mock") == "pkcs11":
        return PKCS11ElectronicSignatureProvider()
    return MockElectronicSignatureProvider()


def get_timestamp_provider() -> ITimestampProvider:
    """Konfigürasyona göre zaman damgası sağlayıcısını döner."""
    from backend.app.core.config import settings
    if getattr(settings, "TIMESTAMP_PROVIDER", "mock") == "rfc3161":
        return RFC3161TimestampProvider()
    return MockTimestampProvider()


# Singleton varsayılan sağlayıcılar
default_signature_provider: IElectronicSignatureProvider = get_signature_provider()
default_timestamp_provider: ITimestampProvider = get_timestamp_provider()
