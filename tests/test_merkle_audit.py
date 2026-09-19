import pytest
from backend.app.core.merkle_audit import MerkleAuditProof, MerkleAuditTree


def test_empty_merkle_tree():
    """Boş ağacın 64 sıfırlı kök hash ürettiğini test eder."""
    tree = MerkleAuditTree()
    root = tree.get_root_hash()
    assert root == "0" * 64
    assert len(root) == 64


def test_single_leaf_merkle_tree():
    """Tek yapraklı ağacın yaprak özetini kök olarak kullandığını test eder."""
    tree = MerkleAuditTree(["AUDIT_EVENT_001"])
    root = tree.get_root_hash()
    assert root == MerkleAuditTree.hash_data("AUDIT_EVENT_001")


def test_merkle_tree_construction_and_determinism():
    """Ağaç inşasının deterministik olduğunu ve yaprak sıralamasının kökü değiştirdiğini test eder."""
    events1 = [
        {"action": "PHASE_A_START", "doc_hash": "abc1234"},
        {"action": "PHASE_A_FREEZE", "features": {"slant": 12.0}},
        {"action": "PHASE_B_UNLOCK", "refs": ["REF_1", "REF_2"]},
        {"action": "GENERATE_REPORT", "status": "SUCCESS"},
    ]
    tree1 = MerkleAuditTree(events1)
    root1 = tree1.get_root_hash()

    tree2 = MerkleAuditTree(events1)
    root2 = tree2.get_root_hash()

    assert root1 == root2
    assert len(root1) == 64

    # Sırası değişmiş olaylar farklı kök üretmeli
    events_shuffled = [events1[1], events1[0], events1[2], events1[3]]
    tree_shuffled = MerkleAuditTree(events_shuffled)
    assert tree_shuffled.get_root_hash() != root1


def test_merkle_proof_generation_and_verification():
    """Her bir yaprak için katılım kanıtının (Proof) doğrulandığını test eder."""
    events = [
        {"id": 1, "action": "UPLOAD_SAMPLE_Q"},
        {"id": 2, "action": "UPLOAD_SAMPLE_R1"},
        {"id": 3, "action": "UPLOAD_SAMPLE_R2"},
        {"id": 4, "action": "EXTRACT_FEATURES"},
        {"id": 5, "action": "FUSION_ANALYSIS"},
    ]
    tree = MerkleAuditTree(events)
    root = tree.get_root_hash()

    for idx, event in enumerate(events):
        proof = tree.get_proof(idx)
        leaf_hash = MerkleAuditTree.hash_data(event)
        assert proof.leaf_hash == leaf_hash
        assert proof.merkle_root == root

        is_valid = MerkleAuditTree.verify_proof(leaf_hash, proof)
        assert is_valid is True


def test_merkle_tamper_detection():
    """Kütükteki tek bir baytlık tahrifatın Merkle doğrulaması ile yakalandığını test eder."""
    original_events = [
        {"action": "CREATE_CASE", "case_id": "CASE_100"},
        {"action": "FREEZE_PHASE_A", "integrity_hash": "sha_q_valid"},
        {"action": "COMPARE", "similarity": 0.88},
    ]
    tree = MerkleAuditTree(original_events)
    original_root = tree.get_root_hash()

    # Tahrif edilmiş kütük: İkinci olayın içeriği manipüle edildi
    tampered_events = [
        {"action": "CREATE_CASE", "case_id": "CASE_100"},
        {"action": "FREEZE_PHASE_A", "integrity_hash": "sha_q_TAMPERED"},
        {"action": "COMPARE", "similarity": 0.88},
    ]

    is_valid, corrupted_idx = MerkleAuditTree.verify_log_chain(tampered_events, expected_root=original_root)
    assert is_valid is False
    assert corrupted_idx is not None
