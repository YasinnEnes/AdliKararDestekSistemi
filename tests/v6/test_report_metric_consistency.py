"""
FHDSS SPRINT 27 — Test: Report Metric Consistency (Phase 1)
Verifies:
- All metrics in model_manifest_v6.json match reported benchmark numbers.
- No discrepancy between manifest and report metrics.
"""

from pathlib import Path
import json
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_manifest_and_report_metric_consistency():
    manifest_path = PROJECT_ROOT / "artifacts" / "weights" / "v6" / "model_manifest_v6.json"
    report_path = PROJECT_ROOT / "reports" / "v6" / "SPRINT_26_V6_MODEL_RESEARCH_FINAL_REPORT.md"

    assert manifest_path.exists(), "Manifest missing"
    assert report_path.exists(), "Report missing"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = manifest["metrics"]
    report_text = report_path.read_text(encoding="utf-8")

    # Check key metrics exist in report text
    for key, val in metrics.items():
        val_str = f"{val:.4f}"
        assert val_str in report_text or f"{val}" in report_text, f"Metric {key}={val_str} missing from report!"
