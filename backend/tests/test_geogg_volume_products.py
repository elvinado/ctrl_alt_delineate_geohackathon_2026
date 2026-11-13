import json
from pathlib import Path

import numpy as np
import yaml

from scripts.geogg_fresh.volume_products import (
    _is_unavailable_linked_input, _png_data_uri, _recorded_dataset_id,
    _resolve_recorded_path, _normalize_volume_axis_order, run_volume_products,
)


def test_volume_products_reads_separate_inference_job_and_writes_to_current_job(tmp_path, monkeypatch):
    jobs = tmp_path / "processing_jobs"
    source_id = "e87b8e3a-2b70-4762-a257-563762492246"
    source_job = jobs / source_id
    source_volumes = source_job / "stages" / "09_inference" / "volumes"
    source_volumes.mkdir(parents=True)
    stage_g_job = jobs / "stage-g-job"
    stage_g_job.mkdir(parents=True)
    result_path = stage_g_job / "result.json"
    result_path.write_text("{}")

    geometry = {"shape": [2, 2, 3], "axis_order": ["inline", "crossline", "time"],
                "foundation_job_id": "foundation-job", "dt_ms": 2.0, "t0_ms": 0.0,
                "input_datasets": {"full": {"dataset_id": "full-id"}}}
    inference = {"tasks": ["lithology"], "outputs": {"lithology": "volumes/prediction_lithology.npy"},
                 "release_id": "release-test"}
    (source_volumes / "volume_geometry.json").write_text(json.dumps(geometry))
    (source_job / "stages" / "09_inference" / "volume_inference_summary.json").write_text(json.dumps(inference))
    np.save(source_volumes / "prediction_valid.npy", np.ones((2, 2, 3), dtype=np.uint8))
    np.save(source_volumes / "trace_index_grid.npy", np.asarray([[0, 1], [2, 3]], dtype=np.int64))
    np.save(source_volumes / "prediction_lithology.npy", np.full((2, 2, 3), 0.5, dtype=np.float32))
    import hashlib
    release_dir = tmp_path / "model_releases" / "release-test"
    release_dir.mkdir(parents=True)
    model_file = release_dir / "models" / "lithology.pt"
    model_file.parent.mkdir()
    model_file.write_bytes(b"immutable-test-model")
    release_manifest = {"model_release_id": "release-test", "status": "ready",
                        "foundation_job_id": "foundation-job",
                        "models": [{"task": "lithology", "input": "patch_stacks", "model": "cnn3d_multitask",
                                    "model_file": "models/lithology.pt", "final_params": {"depth": 3}}],
                        "files": {"models/lithology.pt": {"bytes": model_file.stat().st_size,
                                    "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest()}}}
    (release_dir / "manifest.json").write_text(json.dumps(release_manifest))
    (source_job / "provenance").mkdir()
    prediction_path = source_volumes / "prediction_lithology.npy"
    (source_job / "provenance" / "09_inference.json").write_text(json.dumps({"status": "COMPLETED",
        "stage_id": "09_inference", "recorded_at": "2026-11-09T00:00:00Z", "config_sha256": "abc",
        "inputs": [{"role": "prediction", "path": "stages/09_inference/volumes/prediction_lithology.npy",
                    "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()}], "outputs": []}))

    # A recorded upstream artifact exercises job-relative provenance resolution.
    foundation_job = jobs / "foundation-job"
    foundation_stage = foundation_job / "stages" / "00_inventory"
    foundation_stage.mkdir(parents=True)
    upstream_artifact = foundation_stage / "inventory.txt"
    upstream_artifact.write_text("inventory evidence")
    (foundation_job / "provenance").mkdir()
    (foundation_job / "provenance" / "00_inventory.json").write_text(json.dumps({
        "status": "COMPLETED", "stage_id": "00_inventory",
        "outputs": [{"role": "inventory", "path": "stages/00_inventory/inventory.txt",
                     "sha256": hashlib.sha256(upstream_artifact.read_bytes()).hexdigest()}],
        "inputs": []}))

    # Release-matched Stage 08 evaluation is imported as evidence, not recalculated.
    evaluation_stage = source_job / "stages" / "08_evaluation"
    (evaluation_stage / "tables").mkdir(parents=True)
    (evaluation_stage / "evaluation_summary.json").write_text(json.dumps({
        "release_id": "release-test", "evaluation_policy": "outer_grouped_well_holdout",
        "prediction_rows": 12}))
    (evaluation_stage / "tables" / "model_performance.csv").write_text(
        "task,input,model,n_rows,n_wells,outer_roc_auc,outer_rmse\n"
        "lithology,patch_stacks,cnn3d_multitask,12,3,0.91,0.12\n")
    (source_job / "provenance" / "08_evaluation.json").write_text(json.dumps({
        "status": "COMPLETED", "stage_id": "08_evaluation", "inputs": [], "outputs": []}))

    config = tmp_path / "C_volume_products.yaml"
    config.write_text(yaml.safe_dump({"volume_products": {"source_inference_job_id": source_id,
                                                         "generate_sections": False, "segy_export": False}}))
    monkeypatch.setenv("APGCE_RESULT_PATH", str(result_path))

    assert run_volume_products(["--config", str(config)]) == 0
    output = stage_g_job / "stages" / "10_summary"
    manifest = json.loads((output / "volume_product_manifest.json").read_text())
    job_summary = json.loads((stage_g_job / "geogg_volume_products_summary.json").read_text())
    assert manifest["source_inference_job_id"] == source_id
    assert job_summary["source_inference_job_id"] == source_id
    assert (output / "volume_summary.json").is_file()
    summary = json.loads((output / "volume_summary.json").read_text())
    assert summary["attributes"]["lithology"]["percentiles"]["p50"] == 0.5
    assert (output / "summary_audit.json").is_file()
    assert (output / "tables" / "volume_statistics.csv").is_file()
    assert (output / "tables" / "stage_scorecard.csv").is_file()
    assert (output / "tables" / "stage_artifact_audit.csv").is_file()
    assert (output / "tables" / "output_integrity.csv").is_file()
    assert (output / "tables" / "lineage_consistency.csv").is_file()
    assert (output / "tables" / "results_scorecard.csv").is_file()
    assert (output / "tables" / "claims_register.csv").is_file()
    model_scorecard = (output / "tables" / "model_results_scorecard.csv").read_text()
    assert "0.91" in model_scorecard and "IMPORTED_NOT_RECALCULATED" in model_scorecard
    stage_audit = (output / "tables" / "stage_artifact_audit.csv").read_text()
    assert "inventory.txt" in stage_audit and "PASS" in stage_audit
    lineage = (output / "tables" / "lineage_consistency.csv").read_text()
    assert "prediction_lithology.npy" in lineage and "PASS" in lineage
    claims = (output / "tables" / "claims_register.csv").read_text()
    assert "Registered held-out evaluation evidence" in claims
    assert "upstream_recorded_artifact_integrity" in claims
    assert "Upstream stage execution: 00_inventory" in claims
    assert "upstream_stage_execution_status" in claims
    assert (output / "tables" / "environment.csv").is_file()
    assert (output / "figures" / "volume_coverage.svg").is_file()
    assert (output / "figures" / "attribute_means.svg").is_file()
    assert (output / "figures" / "pipeline_status.svg").is_file()
    assert (output / "figures" / "claims_register.svg").is_file()
    assert (output / "reproduce" / "run_stage_g.sh").is_file()
    assert (output / "dashboard.html").is_file()
    dashboard = (output / "dashboard.html").read_text()
    assert "<svg " in dashboard
    assert "figures/volume_coverage.svg" not in dashboard
    assert (output / "HANDOFF.md").is_file()
    model_card = output / "model_cards" / "lithology__cnn3d_multitask__patch_stacks.md"
    assert model_card.is_file()
    card_text = model_card.read_text()
    assert "## Model specification" in card_text
    assert "## Data provenance and validation" in card_text
    assert "## Limitations and prohibited interpretations" in card_text
    assert "outer_roc_auc" in card_text and "0.91" in card_text
    assert "outer held-out metrics are not stage 09 blind-well metrics" in card_text.lower()
    scorecard_rows = (output / "tables" / "stage_scorecard.csv").read_text()
    assert "09_inference" in scorecard_rows and "COMPLETED" in scorecard_rows
    audit_bytes = (output / "summary_audit.json").read_bytes()
    assert audit_bytes.endswith(b"\n") and not audit_bytes.endswith(b"\\n")
    audit = json.loads(audit_bytes)
    assert any(item["check"] == "immutable_model_release_files" and item["status"] == "PASS"
               for item in audit["checks"])
    stage11 = stage_g_job / "stages" / "11_volume_export"
    assert (stage11 / "native_volume_export_manifest.json").is_file()
    assert (stage11 / "validation_summary.json").is_file()
    assert (stage11 / "tables" / "product_inventory.csv").is_file()
    assert (stage11 / "tables" / "product_integrity.csv").is_file()
    assert (stage11 / "tables" / "value_domain_qc.csv").is_file()
    assert (stage11 / "tables" / "geometry_qc.csv").is_file()
    stage11_manifest = json.loads((stage11 / "native_volume_export_manifest.json").read_text())
    assert stage11_manifest["format"] == "native_npy"
    assert stage11_manifest["derived_segy_export"]["status"] == "DISABLED"
    assert stage11_manifest["status"] == "completed"
    assert (output / "summary_performance.json").is_file()
    assert "summary_performance.json" in manifest["summary_artifacts"]
    performance = json.loads((output / "summary_performance.json").read_text())
    assert {row["phase"] for row in performance["timings"]} >= {"native_statistics_and_product_hashes", "stage11_native_export_qc", "total"}
    job_summary = json.loads((stage_g_job / "geogg_volume_products_summary.json").read_text())
    assert job_summary["derived_segy_export"]["status"] == "DISABLED"
    assert job_summary["prediction_format"] == "native_npy"
    assert job_summary["volume_export_stage"]["stage"] == "stages/11_volume_export"
    assert not (source_job / "stages" / "10_summary").exists()


def _write_tiny_segy(path: Path):
    import struct

    ns, dt_us, ntr = 3, 2000, 2
    preamble = bytearray(b" " * 3200 + bytes(400))
    struct.pack_into(">H", preamble, 3200 + 16, dt_us)
    struct.pack_into(">H", preamble, 3200 + 20, ns)
    struct.pack_into(">H", preamble, 3200 + 24, 5)
    with path.open("wb") as stream:
        stream.write(preamble)
        headers = []
        for ti in range(ntr):
            header = bytearray(240)
            struct.pack_into(">i", header, 188, 100 + ti)  # inline, bytes 189-192
            struct.pack_into(">i", header, 192, 200)       # crossline, bytes 193-196
            struct.pack_into(">h", header, 108, 0)         # delay
            struct.pack_into(">H", header, 114, ns)
            struct.pack_into(">H", header, 116, dt_us)
            headers.append(bytes(header))
            stream.write(header)
            stream.write(np.asarray([ti + 0.1, ti + 0.2, ti + 0.3], dtype=">f4").tobytes())
    return headers


def test_volume_products_blocks_export_when_prediction_outside_physical_domain(tmp_path, monkeypatch):
    jobs = tmp_path / "processing_jobs"
    source_id = "inference-job"
    source_job = jobs / source_id
    source_volumes = source_job / "stages" / "09_inference" / "volumes"
    source_volumes.mkdir(parents=True)
    stage_g_job = jobs / "stage-g-job"
    stage_g_job.mkdir(parents=True)
    result_path = stage_g_job / "result.json"
    result_path.write_text("{}")

    geometry = {"shape": [1, 1, 2], "axis_order": ["inline", "crossline", "time"],
                "foundation_job_id": "foundation-job", "dt_ms": 2.0, "t0_ms": 0.0,
                "input_datasets": {"full": {"dataset_id": "full-id"}}}
    inference = {"tasks": ["vsh"], "outputs": {"vsh": "volumes/prediction_vsh.npy"}}
    (source_volumes / "volume_geometry.json").write_text(json.dumps(geometry))
    (source_job / "stages" / "09_inference" / "volume_inference_summary.json").write_text(json.dumps(inference))
    np.save(source_volumes / "prediction_valid.npy", np.ones((1, 1, 2), dtype=np.uint8))
    np.save(source_volumes / "prediction_vsh.npy", np.asarray([[[0.5, 1.2]]], dtype=np.float32))

    config = tmp_path / "C_volume_products.yaml"
    config.write_text(yaml.safe_dump({"volume_products": {"source_inference_job_id": source_id, "segy_export": False}}))
    monkeypatch.setenv("APGCE_RESULT_PATH", str(result_path))

    import pytest
    with pytest.raises(RuntimeError, match="value-domain audit failed"):
        run_volume_products(["--config", str(config)])
    audit = json.loads((stage_g_job / "stages" / "10_summary" / "summary_audit.json").read_text())
    checks = {item['check']: item['status'] for item in audit['checks']}
    assert checks['prediction_values_within_physical_domain_0_1'] == 'FAIL'
    stage11 = stage_g_job / "stages" / "11_volume_export"
    assert (stage11 / "validation_summary.json").is_file()
    stage11_summary = json.loads((stage11 / "validation_summary.json").read_text())
    assert stage11_summary["status"] == "conditional"
    assert stage11_summary["derived_segy_export"]["status"] == "DISABLED"



def test_seismic_section_generation_writes_inline_crossline_and_time_slice_pngs(tmp_path):
    import csv
    from scripts.geogg_fresh.volume_products import _build_seismic_sections

    jobs = tmp_path / "processing_jobs"
    foundation = jobs / "foundation-job"
    inventory = foundation / "stages" / "00_inventory"
    inventory.mkdir(parents=True)
    reference = tmp_path / "reference.sgy"
    _write_tiny_segy(reference)
    (inventory / "data_manifest.json").write_text(json.dumps({
        "inputs": [{"kind": "seismic", "dataset_id": "full-id", "path": str(reference)}]
    }))
    well_tables = foundation / "stages" / "01_well_ties" / "tables"
    well_tables.mkdir(parents=True)
    (well_tables / "tz_samples_all_wells.csv").write_text(
        "well,md_m,tvdss_m,twt_ms,il_nearest,xl_nearest,seismic_sample_frac,trace_offset_xy,in_seismic_time_range,tz_extrapolation_m\n"
        "well-a,1000,900,2,100,200,1,5.0,True,0\n")

    source_job = jobs / "inference-job"
    volumes = source_job / "stages" / "09_inference" / "volumes"
    volumes.mkdir(parents=True)
    prediction = np.full((2, 1, 3), 0.5, dtype=np.float32)
    np.save(volumes / "prediction_lithology.npy", prediction)
    np.save(volumes / "prediction_vsh.npy", prediction * 0.5)
    out = tmp_path / "stage-g" / "stages" / "10_summary"
    (out / "tables").mkdir(parents=True)

    geometry = {
        "foundation_job_id": "foundation-job",
        "shape": [2, 1, 3],
        "il_min": 100, "il_step": 1,
        "xl_min": 200, "xl_step": 1,
        "t0_ms": 0.0, "dt_ms": 2.0,
        "input_datasets": {"full": {"dataset_id": "full-id"}},
        "spatial_reference": {"xy_mapping_status": "requires_trace_coordinate_qc"},
    }
    valid = np.ones((2, 1, 3), dtype=np.uint8)
    trace_grid = np.asarray([[0], [1]], dtype=np.int32)
    info = _build_seismic_sections(
        storage_root=tmp_path, geometry=geometry, source_id="inference-job",
        source_job=source_job, volumes=volumes, out=out, tasks=["lithology", "vsh"],
        shape=(2, 1, 3), valid=valid, trace_grid=trace_grid,
    )
    assert info["section_count"] == 12
    assert info["well_trajectory_overlay"] == "generated_from_stage01_tz_samples"
    assert info["well_trajectory_sample_count"] == 1
    intersections = (out / "tables" / "well_volume_intersections.csv").read_text()
    assert "well-a" in intersections and "prediction_valid" in intersections
    rows = (out / "tables" / "seismic_sections.csv").read_text()
    assert "inline" in rows and "crossline" in rows and "time_slice" in rows
    images = list((out / "figures" / "sections").rglob("*.png"))
    assert len(images) == 12
    overlay_rows = list(csv.DictReader((out / "tables" / "well_section_overlays.csv").open()))
    assert len(overlay_rows) == 2  # geometry overlays are not duplicated per attribute
    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in images)



def test_dashboard_png_data_uri_is_self_contained(tmp_path):
    image = tmp_path / "preview.png"
    image.write_bytes(b"\\x89PNG\\r\\n\\x1a\\nsynthetic-png")
    uri = _png_data_uri(image)
    assert uri.startswith("data:image/png;base64,")
    assert "preview.png" not in uri
    import base64
    assert base64.b64decode(uri.split(",", 1)[1]) == image.read_bytes()


def test_stage08_evaluation_evidence_is_imported_only_for_matching_release(tmp_path):
    import csv
    import hashlib
    from scripts.geogg_fresh.volume_products import _import_evaluation_evidence

    source_job = tmp_path / "processing_jobs" / "inference-job"
    stage08 = source_job / "stages" / "08_evaluation"
    (stage08 / "tables").mkdir(parents=True)
    (stage08 / "figures").mkdir(parents=True)
    (stage08 / "evaluation_summary.json").write_text(json.dumps({
        "release_id": "release-1",
        "evaluation_policy": "sealed outer-test predictions only; no refitting or model reselection",
        "prediction_rows": 12,
    }))
    with (stage08 / "tables" / "model_performance.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task", "input", "model", "outer_roc_auc", "n_wells"])
        writer.writeheader()
        writer.writerow({"task": "lithology", "input": "patch_stacks", "model": "cnn3d_multitask",
                         "outer_roc_auc": "0.65", "n_wells": "13"})
        writer.writerow({"task": "lithology", "input": "stacks", "model": "extratrees",
                         "outer_roc_auc": "0.61", "n_wells": "13"})
    with (stage08 / "tables" / "per_well_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["well", "task", "input", "model", "roc_auc"])
        writer.writeheader()
        writer.writerow({"well": "no-1", "task": "lithology", "input": "patch_stacks",
                         "model": "cnn3d_multitask", "roc_auc": "0.7"})
    (stage08 / "tables" / "outer_oof_predictions.csv").write_text("task,well,y_true,y_pred\nlithology,no-1,1,0.8\n")
    (stage08 / "figures" / "model_comparison_outer.png").write_bytes(b"synthetic-png")

    output = tmp_path / "stage-g" / "stages" / "10_summary"
    output.mkdir(parents=True)
    result = _import_evaluation_evidence(
        source_job=source_job, storage_root=tmp_path, foundation_id="", release_id="release-1",
        release_manifest={"models": [{"task": "lithology", "input": "patch_stacks", "model": "cnn3d_multitask"}]},
        out=output,
    )

    assert result["status"] == "AVAILABLE"
    assert result["registered_model_rows"] == 1
    assert result["registered_per_well_rows"] == 1
    assert result["source_outer_oof_rows"] == 1
    assert result["source_evaluation_policy"].startswith("sealed outer-test")
    scorecard = (output / "tables" / "evaluation_scorecard.csv").read_text()
    assert "cnn3d_multitask" in scorecard
    assert "extratrees" not in scorecard
    assert (output / "tables" / "evaluation_source" / "outer_oof_predictions.csv").read_text().endswith("lithology,no-1,1,0.8\n")
    assert (output / "figures" / "evaluation" / "model_comparison_outer.png").read_bytes() == b"synthetic-png"
    assert (output / "tables" / "evaluation_evidence_inventory.csv").is_file()


def test_stage08_evaluation_evidence_rejects_release_mismatch(tmp_path):
    from scripts.geogg_fresh.volume_products import _import_evaluation_evidence

    source_job = tmp_path / "processing_jobs" / "inference-job"
    stage08 = source_job / "stages" / "08_evaluation"
    stage08.mkdir(parents=True)
    (stage08 / "evaluation_summary.json").write_text(json.dumps({"release_id": "different-release"}))
    output = tmp_path / "stage-g" / "stages" / "10_summary"
    output.mkdir(parents=True)

    result = _import_evaluation_evidence(
        source_job=source_job, storage_root=tmp_path, foundation_id="", release_id="release-1",
        release_manifest={"models": []}, out=output,
    )
    assert result["status"] == "NOT_AVAILABLE"
    assert result["registered_model_rows"] == 0


import pytest

@pytest.mark.parametrize(
    "scenario,well_rows,trace_grid,valid_mask,expected_status,expected_well_status",
    [
        ("no_trajectory_data", [], [[0], [1]], [[[1, 1, 1]], [[1, 1, 1]]],
         "no_trajectory_data", None),
        ("outside_grid", [("well-out", 99, 200, 1)], [[0], [1]],
         [[[1, 1, 1]], [[1, 1, 1]]], "no_volume_grid_samples", "no_volume_grid_samples"),
        ("unmapped_trace", [("well-unmapped", 100, 200, 1)], [[-1], [1]],
         [[[1, 1, 1]], [[1, 1, 1]]], "inside_grid_no_valid_prediction", "inside_grid_no_valid_prediction"),
        ("invalid_mask", [("well-invalid", 100, 200, 1)], [[0], [1]],
         [[[1, 0, 1]], [[1, 1, 1]]], "inside_grid_no_valid_prediction", "inside_grid_no_valid_prediction"),
        ("valid_prediction", [("well-valid", 100, 200, 1)], [[0], [1]],
         [[[1, 1, 1]], [[1, 1, 1]]], "intersecting_valid_prediction", "intersecting_valid_prediction"),
        ("mixed_wells", [("well-hit", 100, 200, 1), ("well-miss", 99, 200, 1)], [[0], [1]],
         [[[1, 1, 1]], [[1, 1, 1]]], "intersecting_valid_prediction", None),
    ],
)
def test_well_intersection_statuses_are_explicit(
    tmp_path, scenario, well_rows, trace_grid, valid_mask, expected_status, expected_well_status
):
    import csv
    from scripts.geogg_fresh.volume_products import _build_seismic_sections

    jobs = tmp_path / "processing_jobs"
    foundation = jobs / "foundation-job"
    inventory = foundation / "stages" / "00_inventory"
    inventory.mkdir(parents=True)
    reference = tmp_path / "reference.sgy"
    _write_tiny_segy(reference)
    (inventory / "data_manifest.json").write_text(json.dumps({
        "inputs": [{"kind": "seismic", "dataset_id": "full-id", "path": str(reference)}]
    }))
    well_tables = foundation / "stages" / "01_well_ties" / "tables"
    well_tables.mkdir(parents=True)
    if scenario != "no_trajectory_data":
        lines = ["well,md_m,tvdss_m,twt_ms,il_nearest,xl_nearest,seismic_sample_frac,trace_offset_xy,in_seismic_time_range,tz_extrapolation_m"]
        for well, il, xl, sample in well_rows:
            lines.append(f"{well},1000,900,2,{il},{xl},{sample},5.0,True,0")
        (well_tables / "tz_samples_all_wells.csv").write_text("\n".join(lines) + "\n")

    source_job = jobs / "inference-job"
    volumes = source_job / "stages" / "09_inference" / "volumes"
    volumes.mkdir(parents=True)
    np.save(volumes / "prediction_lithology.npy", np.full((2, 1, 3), 0.5, dtype=np.float32))
    out = tmp_path / "stage-g" / "stages" / "10_summary"
    (out / "tables").mkdir(parents=True)
    geometry = {
        "foundation_job_id": "foundation-job", "shape": [2, 1, 3],
        "il_min": 100, "il_step": 1, "xl_min": 200, "xl_step": 1,
        "t0_ms": 0.0, "dt_ms": 2.0,
        "input_datasets": {"full": {"dataset_id": "full-id"}},
    }
    info = _build_seismic_sections(
        storage_root=tmp_path, geometry=geometry, source_id="inference-job",
        source_job=source_job, volumes=volumes, out=out, tasks=["lithology"],
        shape=(2, 1, 3), valid=np.asarray(valid_mask, dtype=np.uint8),
        trace_grid=np.asarray(trace_grid, dtype=np.int32),
    )
    assert info["well_intersection_status"] == expected_status
    status_rows = list(csv.DictReader((out / "tables" / "well_intersection_status.csv").open()))
    if scenario == "no_trajectory_data":
        assert status_rows == []
    elif scenario == "mixed_wells":
        statuses = {row["well"]: row["status"] for row in status_rows}
        assert statuses == {"well-hit": "intersecting_valid_prediction", "well-miss": "no_volume_grid_samples"}
        assert info["well_status_counts"]["no_volume_grid_samples"] == 1
    else:
        assert status_rows[0]["status"] == expected_well_status


def test_recorded_stage_relative_outputs_and_legacy_absolute_paths_resolve(tmp_path):
    storage = tmp_path / "site_data"
    jobs = storage / "processing_jobs"
    foundation = jobs / "foundation-job"
    stage = foundation / "stages" / "07_models"
    stage.mkdir(parents=True)
    provenance = foundation / "provenance" / "07_models.json"
    provenance.parent.mkdir()
    stage_output = stage / "figures" / "model.png"
    stage_output.parent.mkdir()
    stage_output.write_bytes(b"stage-output")

    resolved_stage = _resolve_recorded_path(
        "figures/model.png", provenance_path=provenance, owner_job=foundation,
        stage_id="07_models")
    assert resolved_stage == stage_output

    # Old provenance stores host-absolute paths. The stable processing-job
    # identity is sufficient to remap the path into the current storage root.
    legacy = ("/old/host/site_data/processing_jobs/foundation-job/"
              "stages/07_models/figures/model.png")
    resolved_legacy = _resolve_recorded_path(
        legacy, provenance_path=provenance, owner_job=foundation,
        stage_id="07_models")
    assert resolved_legacy == stage_output

    windows_legacy = (r"Z:\\old\\site_data\\processing_jobs\\foundation-job"
                      r"\\stages\\07_models\\figures\\model.png")
    resolved_windows = _resolve_recorded_path(
        windows_legacy, provenance_path=provenance, owner_job=foundation,
        stage_id="07_models")
    assert resolved_windows == stage_output


def test_unavailable_linked_inputs_are_not_classified_as_missing(tmp_path):
    storage = tmp_path / "site_data"
    jobs = storage / "processing_jobs"
    owner = jobs / "foundation-job"
    owner.mkdir(parents=True)
    index = {"dataset-1": {"dataset_id": "dataset-1", "storage_mode": "linked",
                           "storage_key": "linked/dataset-1.sgy"}}
    resolved = _resolve_recorded_path(
        "/old/mount/seismic.sgy", provenance_path=None, owner_job=owner,
        dataset_id="dataset-1", dataset_index=index)
    assert resolved is not None and not resolved.is_file()
    assert _is_unavailable_linked_input("dataset-1", index)


def test_registered_dataset_paths_resolve_from_dataset_id_and_storage_key(tmp_path):
    storage = tmp_path / "site_data"
    owner = storage / "processing_jobs" / "inference-job"
    owner.mkdir(parents=True)
    uploads = storage / "uploads"
    uploads.mkdir()
    managed_file = uploads / "well-list.xlsx"
    managed_file.write_bytes(b"managed dataset")
    index = {"managed-id": {"dataset_id": "managed-id", "storage_mode": "managed",
                            "storage_key": "uploads/well-list.xlsx"}}
    role = "registered seismic input: full; dataset_id=linked-id"
    assert _recorded_dataset_id({"role": role}) == "linked-id"
    resolved = _resolve_recorded_path(
        "/old/host/uploads/well-list.xlsx", provenance_path=None, owner_job=owner,
        dataset_id="managed-id", dataset_index=index)
    assert resolved == managed_file


def test_volume_axis_order_accepts_native_manifest_aliases():
    assert _normalize_volume_axis_order(["inline_bin", "crossline_bin", "time_sample"]) == ["inline", "crossline", "time"]
    assert _normalize_volume_axis_order(["inline", "crossline", "time"]) == ["inline", "crossline", "time"]
    assert _normalize_volume_axis_order(["crossline", "inline", "time"]) != ["inline", "crossline", "time"]
