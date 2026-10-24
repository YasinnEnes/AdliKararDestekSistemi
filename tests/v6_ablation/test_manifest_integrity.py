"""
FHDSS SPRINT 28 — Test: Manifest and Baseline Freeze Integrity
Verifies:
- Model manifests for V5 and V6 exist, are valid JSON, and retain required fields.
- SHA-256 hashes recorded in V6_ABLATION_BASELINE_FREEZE.md match the files on disk.
"""

from pathlib import Path
import hashlib
import json
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def test_manifest_files_exist_and_valid():
    v5_manifest = PROJECT_ROOT / "artifacts" / "weights" / "model_manifest.json"
    v6_manifest = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "model_manifest_v6.json"
    
    assert v5_manifest.exists(), "V5 manifest must exist"
    assert v6_manifest.exists(), "V6 manifest must exist"
    
    with open(v5_manifest, "r", encoding="utf-8") as f:
        v5_data = json.load(f)
    with open(v6_manifest, "r", encoding="utf-8") as f:
        v6_data = json.load(f)
        
    assert "model_name" in v5_data or "architecture" in v5_data
    assert "model_name" in v6_data or "architecture" in v6_data


def test_freeze_report_integrity():
    freeze_path = PROJECT_ROOT / "reports" / "v6_ablation" / "V6_ABLATION_BASELINE_FREEZE.md"
    assert freeze_path.exists(), "Freeze report missing"
    content = freeze_path.read_text(encoding="utf-8")
    
    v5_pt = PROJECT_ROOT / "artifacts" / "weights" / "convnext_tiny_forensic_multidataset_v5_best.pt"
    v6_pt = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "convnext_tiny_forensic_multidataset_v6_best.pt"
    
    if v5_pt.exists():
        v5_hash = compute_sha256(v5_pt)
        assert v5_hash in content, f"V5 PT hash {v5_hash} not matching freeze report"
        
    if v6_pt.exists():
        v6_hash = compute_sha256(v6_pt)
        assert v6_hash in content, f"V6 PT hash {v6_hash} not matching freeze report"
