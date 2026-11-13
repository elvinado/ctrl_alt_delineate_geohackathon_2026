"""Tests for the shared processing-worker output contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts_utils import validate_outputs


def test_validate_outputs_writes_success_result(tmp_path, monkeypatch):
    result_path = tmp_path / "result.json"
    output_path = tmp_path / "summary.json"
    monkeypatch.setenv("APGCE_RESULT_PATH", str(result_path))

    @validate_outputs(expected_outputs=["summary.json"])
    def worker():
        output_path.write_text("{}", encoding="utf-8")
        return "complete"

    assert worker() == "complete"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "succeeded"
    assert result["outputs"] == [str(output_path.resolve())]


def test_validate_outputs_writes_failure_result_for_missing_output(tmp_path, monkeypatch):
    result_path = tmp_path / "result.json"
    monkeypatch.setenv("APGCE_RESULT_PATH", str(result_path))

    @validate_outputs(expected_outputs=["missing.json"])
    def worker():
        return None

    with pytest.raises(FileNotFoundError, match="Expected output files not found"):
        worker()

    result = json.loads(result_path.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert "missing.json" in result["error_message"]
