from __future__ import annotations

import json
from pathlib import Path

import pytest

from geogg_fresh.context import StageContext
from geogg_fresh.contracts import StageStatus
from geogg_fresh.runner import _write_provenance
from geogg_fresh.stage_compat import SpecialistStageContext, StageBlocked


def test_stage_context_uses_existing_job_layout(tmp_path):
    base = StageContext(stage_id="00_inventory", job_dir=tmp_path, config={"a": 1}, run_id="run-1")
    compat = SpecialistStageContext(base, {}, "00_inventory")
    assert compat.run.run_id == "run-1"
    assert compat.run.run_dir == tmp_path / "stages"
    assert base.layout.provenance == tmp_path / "provenance"
    assert not (tmp_path / "geogg_fresh").exists()
    assert compat.stage_dir == compat.run.run_dir / "00_inventory"
    assert (compat.stage_dir / "tables").is_dir()
    assert (compat.stage_dir / "figures").is_dir()
    assert not (compat.stage_dir / "logs").exists()
    assert not hasattr(compat, "logs_dir")


def test_stage_context_input_hash_and_provenance(tmp_path):
    source = tmp_path / "source.txt"
    source.write_text("input", encoding="utf-8")
    base = StageContext(stage_id="00_inventory", job_dir=tmp_path / "job", config={}, run_id="run-1")
    digest = base.add_input(source, "test")
    assert len(digest) == 64
    result_path = base.resolve_output("tables/result.csv")
    result_path.write_text("a\n1\n", encoding="utf-8")
    compat = SpecialistStageContext(base, {}, "00_inventory")
    compat.check("reader_ok", True)
    from geogg_fresh.contracts import StageResult
    result = StageResult("00_inventory", StageStatus.COMPLETED, "ok")
    _write_provenance(base, result)
    record = json.loads((base.layout.provenance / "00_inventory.json").read_text())
    assert record["status"] == "COMPLETED"
    assert record["outputs"][0]["path"] == "tables/result.csv"
    assert record["inputs"][0]["sha256"] == digest


def test_stage_blocked_is_explicit():
    with pytest.raises(StageBlocked):
        raise StageBlocked("missing upstream inventory")


def test_provenance_keeps_backend_dataset_uid_when_input_is_hashed(tmp_path):
    from geogg_fresh.contracts import InputRecord, StageResult

    source = tmp_path / "registered.las"
    source.write_text("LAS", encoding="utf-8")
    base = StageContext(
        stage_id="00_inventory", job_dir=tmp_path / "job",
        inputs=[InputRecord(
            dataset_id="backend-uid-123", kind="well_log", format="las",
            name="registered.las", path=str(source), well_id="well-uid-456",
            well_name="WELL_A",
        )],
        run_id="run-uid",
    )
    base.add_input(source, "las:WELL_A")
    _write_provenance(base, StageResult("00_inventory", StageStatus.COMPLETED, "ok"))

    payload = json.loads((base.layout.provenance / "00_inventory.json").read_text())
    assert len(payload["inputs"]) == 1
    assert payload["inputs"][0]["dataset_id"] == "backend-uid-123"
    assert payload["inputs"][0]["well_id"] == "well-uid-456"
    assert payload["inputs"][0]["sha256"]
