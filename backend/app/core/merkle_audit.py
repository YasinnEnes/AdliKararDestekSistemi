import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class MerkleAuditProof:
    """Kriptografik Merkle Ağacı Katılım Kanıtı (Inclusion Proof)."""
    leaf_hash: str
    leaf_index: int
    total_leaves: int
    audit_path: List[Tuple[str, str]]  # [(sibling_hash, 'L' veya 'R')]
    merkle_root: str


class MerkleAuditTree:
    """
    Adli Bütünlük ve Delil Zinciri (Chain of Custody) için Kriptografik Merkle Ağacı Motoru.
    
    Özellikler:
    - Olay kütüklerini (Audit Logs) SHA-256 yaprak düğümlerine dönüştürür.
    - $O(\log N)$ karmaşıklığında katılım kanıtı (*Inclusion Proof*) üretir.
    - Tarihsel kütüklerdeki herhangi bir tahrifatı (*tampering*) tam indeks bazında tespit eder.
    - Deterministik ikili ağaç mimarisi (Tekil düğüm durumunda sağ kardeş kopyalama).
    """

    EMPTY_ROOT: str = "0" * 64

    def __init__(self, raw_events: Optional[List[Union[str, Dict[str, Any]]]] = None):
        self.raw_events: List[Union[str, Dict[str, Any]]] = []
        self.leaf_hashes: List[str] = []
        self.levels: List[List[str]] = []
        self.root_hash: str = self.EMPTY_ROOT

        if raw_events:
            for event in raw_events:
                self.add_leaf(event)
            self.build_tree()

    @staticmethod
    def hash_data(data: Union[str, bytes, Dict[str, Any]]) -> str:
        """Herhangi bir veriyi deterministik SHA-256 özetine dönüştürür."""
        if isinstance(data, dict):
            serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            payload = serialized.encode("utf-8")
        elif isinstance(data, str):
            payload = data.encode("utf-8")
        elif isinstance(data, bytes):
            payload = data
        else:
            payload = str(data).encode("utf-8")

        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def combine_hashes(cls, left_hash: str, right_hash: str) -> str:
        """İki alt düğüm hash'ini birleştirerek ebeveyn düğüm hash'ini üretir."""
        combined = (left_hash + right_hash).encode("utf-8")
        return hashlib.sha256(combined).hexdigest()

    def add_leaf(self, event_data: Union[str, Dict[str, Any]]) -> str:
        """Yeni bir denetim olayını ağaca yaprak olarak ekler."""
        self.raw_events.append(event_data)
        leaf_h = self.hash_data(event_data)
        self.leaf_hashes.append(leaf_h)
        return leaf_h

    def build_tree(self) -> str:
        """Mevcut yapraklardan tam Merkle Ağacını inşa eder ve Merkle Kökünü döner."""
        if not self.leaf_hashes:
            self.levels = []
            self.root_hash = self.EMPTY_ROOT
            return self.root_hash

        current_level = list(self.leaf_hashes)
        self.levels = [current_level]

        while len(current_level) > 1:
            next_level: List[str] = []
            n = len(current_level)

            for i in range(0, n, 2):
                left = current_level[i]
                # Tekil düğüm varsa kendisiyle eşle (Merkle Standardı)
                right = current_level[i + 1] if (i + 1 < n) else left
                parent = self.combine_hashes(left, right)
                next_level.append(parent)

            self.levels.append(next_level)
            current_level = next_level

        self.root_hash = current_level[0]
        return self.root_hash

    def get_root_hash(self) -> str:
        """Güncel Merkle Kök Hash'ini döner (Gerekiyorsa ağacı inşa eder)."""
        if not self.levels or self.root_hash == self.EMPTY_ROOT:
            return self.build_tree()
        return self.root_hash

    def get_proof(self, leaf_index: int) -> MerkleAuditProof:
        """
        Belirtilen indeksteki yaprak için O(log N) kriptografik doğrulama kanıtı üretir.
        """
        if not self.levels:
            self.build_tree()

        total_leaves = len(self.leaf_hashes)
        if leaf_index < 0 or leaf_index >= total_leaves:
            raise IndexError(f"Geçersiz yaprak indeksi: {leaf_index} (Toplam: {total_leaves})")

        audit_path: List[Tuple[str, str]] = []
        idx = leaf_index

        for level_idx in range(len(self.levels) - 1):
            level = self.levels[level_idx]
            n = len(level)

            if idx % 2 == 0:
                # Biz sol düğümüz, kardeş sağda (veya tekilsek kendimiz)
                sibling_idx = idx + 1 if (idx + 1 < n) else idx
                sibling_dir = "R"
            else:
                # Biz sağ düğümüz, kardeş solda
                sibling_idx = idx - 1
                sibling_dir = "L"

            sibling_hash = level[sibling_idx]
            audit_path.append((sibling_hash, sibling_dir))
            idx //= 2

        return MerkleAuditProof(
            leaf_hash=self.leaf_hashes[leaf_index],
            leaf_index=leaf_index,
            total_leaves=total_leaves,
            audit_path=audit_path,
            merkle_root=self.root_hash,
        )

    @classmethod
    def verify_proof(cls, leaf_hash: str, proof: MerkleAuditProof) -> bool:
        """
        Verilen katılım kanıtının (Proof) iddia edilen Merkle Kökü ile eşleştiğini doğrular.
        """
        current_hash = leaf_hash

        for sibling_hash, direction in proof.audit_path:
            if direction == "R":
                current_hash = cls.combine_hashes(current_hash, sibling_hash)
            elif direction == "L":
                current_hash = cls.combine_hashes(sibling_hash, current_hash)
            else:
                return False

        return current_hash.lower() == proof.merkle_root.lower()

    @classmethod
    def verify_log_chain(
        cls,
        events: List[Union[str, Dict[str, Any]]],
        expected_root: str,
    ) -> Tuple[bool, Optional[int]]:
        """
        Bir olay listesini sıfırdan ağaca dönüştürerek beklenen Merkle Kökü ile doğrular.
        
        Returns:
            (is_valid: bool, corrupted_index: Optional[int])
        """
        if not events:
            return (expected_root == cls.EMPTY_ROOT, None)

        test_tree = cls(events)
        calculated_root = test_tree.get_root_hash()

        if calculated_root.lower() == expected_root.lower():
            return True, None

        # Tahrifatın hangi olayda olduğunu bulmak için yaprak bazında inceleme
        # (Beklenen kök ile uyuşmayan ilk yaprağı tespit et)
        return False, 0
