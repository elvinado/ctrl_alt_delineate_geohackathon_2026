from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.geogg_fresh.config import GeoGGConfigError, config_sha256, normalize_config
from scripts.geogg_fresh.context import StageContext
from scripts.geogg_fresh.contracts import OutputRecord, StageResult, StageStatus
from scripts.geogg_fresh.provenance import register_output, write_stage_provenance


def test_normalize_config_resolves_roots_and_normalizes_wells(tmp_path):
    cfg = normalize_config(
        {
            "paths": {"project_root": "project"},
            "project": {"output_root": "output"},
            "wells": {"expected": [" WELL_A ", "Well_B"]},
        },
        config_path=tmp_path / "config.yaml",
    )
    assert cfg["paths"]["project_root"] == str((tmp_path / "project").resolve())
    assert cfg["project"]["output_root"] == str((tmp_path / "output").resolve())
    assert cfg["wells"]["expected"] == ["well_a", "well_b"]
    assert cfg["_runtime"]["config_sha256"] == config_sha256(cfg)


def test_normalize_config_requires_explicit_roots():
    with pytest.raises(GeoGGConfigError, match="paths.project_root"):
        normalize_config({})


def test_context_reads_existing_application_input_manifest(tmp_path, monkeypatch):
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    source = tmp_path / "source.sgy"
    source.write_bytes(b"seismic")
    manifest = job_dir / "inputs.json"
    manifest.write_text(json.dumps({"inputs": [{
        "dataset_id": "dataset-1", "kind": "seismic", "format": "sgy",
        "name": "source.sgy", "path": str(source), "survey_id": "survey-1",
        "survey_name": "Survey A", "metadata": {"geogg_volume": "full"},
    }]}))
    monkeypatch.setenv("APGCE_RESULT_PATH", str(job_dir / "result.json"))
    monkeypatch.setenv("APGCE_INPUT_MANIFEST", str(manifest))

    context = StageContext.from_environment(stage_id="00_inventory")
    assert context.input_paths(kind="seismic") == [source.resolve()]
    record = context.inputs[0]
    assert record.dataset_id == "dataset-1"
    assert record.survey_id == "survey-1"
    assert record.survey_name == "Survey A"
    assert record.metadata == {"geogg_volume": "full"}
    assert context.stage_dir == job_dir / "stages" / "00_inventory"
    assert context.layout.provenance == job_dir / "provenance"
    assert not (job_dir / "geogg_fresh").exists()


def test_output_registration_and_provenance_are_integrity_checked(tmp_path):
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    output = stage_dir / "inventory.json"
    output.write_text('{"ok": true}')
    record = register_output(stage_dir, output, role="inventory", media_type="application/json")
    assert record.sha256
    assert record.size_bytes == output.stat().st_size

    result = StageResult(stage_id="00_inventory", status=StageStatus.COMPLETED,
                         outputs=[record], summary="Inventory complete")
    provenance = write_stage_provenance(
        provenance_dir=tmp_path / "provenance", stage_id=result.stage_id,
        run_id="run-1", config_hash="abc", result=result, inputs=[],
    )
    payload = json.loads(provenance.read_text())
    assert payload["outputs"][0]["sha256"] == record.sha256
    with pytest.raises(FileExistsError):
        write_stage_provenance(
            provenance_dir=tmp_path / "provenance", stage_id=result.stage_id,
            run_id="run-1", config_hash="abc", result=result, inputs=[],
        )


def test_output_registration_rejects_path_escape(tmp_path):
    stage_dir = tmp_path / "stage"
    stage_dir.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("no")
    with pytest.raises(ValueError, match="inside"):
        register_output(stage_dir, outside, role="bad")


def test_stage01_uses_manifest_wells_when_config_expected_is_empty():
    import pandas as pd
    from scripts.geogg_fresh.stages.s01_well_ties import _resolve_inventory_wells

    availability = pd.DataFrame(index=["eo-1", "m-1", "no-1"])
    assert _resolve_inventory_wells([], availability) == ["eo-1", "m-1", "no-1"]
    assert _resolve_inventory_wells([" M-1 "], availability) == ["m-1"]


def test_stage00_well_list_selection_uses_registered_metadata_only():
    from types import SimpleNamespace
    from scripts.geogg_fresh.stages.s00_inventory import _registered_well_list_records

    workbook = SimpleNamespace(kind="table", metadata={"table_type": "well_list"})
    unrelated = SimpleNamespace(kind="table", metadata={"table_type": "other"})
    filename_lookalike = SimpleNamespace(kind="table", name="well_list.xlsx", metadata={})
    assert _registered_well_list_records([unrelated, filename_lookalike, workbook]) == [workbook]



def test_stage01_optional_tdr_metrics_do_not_require_md_kb_or_datum():
    import pandas as pd
    from scripts.geogg_fresh.stages.s01_well_ties import _finite_qc_values, _optional_qc_max

    # Specialist two-column TVDSS/TWT tables legitimately have neither
    # an implied datum nor MDKB-based trajectory residuals.
    records = [
        {"well": "m-1", "datum_difference_m": float("nan")},
        {"well": "no-1", "datum_difference_m": float("nan")},
    ]
    frame = pd.DataFrame(records)
    assert _finite_qc_values(frame, "datum_difference_m").empty
    assert _finite_qc_values(frame, "tvdss_residual_max_abs_m").empty
    assert _optional_qc_max(records, "tvdss_residual_max_abs_m") is None


def test_stage01_optional_tdr_metrics_preserve_available_values():
    from scripts.geogg_fresh.stages.s01_well_ties import _finite_qc_values, _optional_qc_max
    import pandas as pd

    records = [
        {"well": "m-1", "datum_difference_m": 0.2, "tvdss_residual_max_abs_m": 1.1},
        {"well": "no-1", "datum_difference_m": float("nan"), "tvdss_residual_max_abs_m": 0.8},
    ]
    frame = pd.DataFrame(records)
    assert _finite_qc_values(frame, "datum_difference_m").tolist() == [0.2]
    assert _optional_qc_max(records, "tvdss_residual_max_abs_m") == 1.1



def test_stage01_full_stack_uses_registered_stage00_inventory_path(tmp_path):
    import pandas as pd
    from types import SimpleNamespace
    from scripts.geogg_fresh.stages.s01_well_ties import _load_full_seismic_path

    run_dir = tmp_path / "run"
    inventory_dir = run_dir / "00_inventory" / "tables"
    inventory_dir.mkdir(parents=True)
    registered = tmp_path / "registered_full.sgy"
    registered.write_bytes(b"segy-placeholder")
    pd.DataFrame([
        {"volume": "full", "path": str(registered), "exists": True},
        {"volume": "near", "path": "unavailable/near.sgy", "exists": False},
    ]).to_csv(inventory_dir / "seismic_volume_inventory.csv", index=False)

    class Context:
        def __init__(self):
            self.run = SimpleNamespace(run_dir=run_dir)
            self.inputs = []
        def add_input(self, path, role):
            self.inputs.append((Path(path), role))

    ctx = Context()
    assert _load_full_seismic_path(ctx) == registered
    assert ctx.inputs == [(registered, "upstream:registered_full_stack")]


def test_stage00_maps_registered_seismic_datasets_from_delivery_filenames():
    from types import SimpleNamespace
    from scripts.geogg_fresh.stages.s00_inventory import _registered_seismic_volume

    cases = {
        "Challenge1_Challenge2_Full_Stack_cropped.sgy": "full",
        "Challenge1_Challenge2_Near_Stack_cropped.sgy": "near",
        "Challenge1_Challenge2_Mid_Stack_cropped.sgy": "mid",
        "Challenge1_Challenge2_Far_Stack_cropped.sgy": "far",
        "Challenge1_Challenge2_UltraFar_Stack_cropped.sgy": "ultrafar",
        "Challenge1_Challenge2_Inversion_P_Impedance_cropped.sgy": "pimp",
        "Challenge1_Challenge2_Inversion_S_Impedance_cropped.sgy": "simp",
        "Challenge1_Challenge2_Inversion_VpVs_cropped.sgy": "vpvs",
        "Challenge1_Challenge2_Inversion_TotalSandProbability_cropped.sgy": "totalsandprob",
    }
    for name, expected in cases.items():
        record = SimpleNamespace(name=name, metadata={})
        assert _registered_seismic_volume(record) == expected


def test_stage00_prefers_explicit_registered_seismic_volume_metadata():
    from types import SimpleNamespace
    from scripts.geogg_fresh.stages.s00_inventory import _registered_seismic_volume

    record = SimpleNamespace(name="ambiguous.sgy", metadata={"geogg_volume": "full"})
    assert _registered_seismic_volume(record) == "full"
