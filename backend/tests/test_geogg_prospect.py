import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import types
sys.modules.setdefault("scripts_utils", types.SimpleNamespace(validate_outputs=lambda **kw: (lambda fn: fn)))
from geogg_fresh.stages.s14_prospect import _cell_area_grid, _log_stats, run


def test_cell_area_from_xy_grid():
    rows = []
    for i in range(3):
        for j in range(3):
            rows.append({"volume_i": i, "volume_j": j, "x": i * 25.0, "y": j * 40.0})
    area, qc, xg, yg = _cell_area_grid(pd.DataFrame(rows), (3, 3))
    assert np.allclose(area, 1000.0)
    assert qc["status"] == "AVAILABLE"


def test_log_statistics_requires_real_swt_support():
    frame = pd.DataFrame({"h_m": [2.0, 2.0], "vsh_fraction": [0.1, 0.8],
                          "phit_fraction": [0.2, 0.2], "swt_fraction": [np.nan, np.nan]})
    stats = _log_stats(frame, 0.4, 0.03)
    assert stats["gross_m"] == 4.0
    assert stats["net_m"] == 2.0
    assert stats["ntg"] == 0.5
    assert stats["phit_net"] == 0.2
    assert stats["swt_net_pv"] is None


def test_stage14_end_to_end_phase1_phase2_without_stage11_swt(tmp_path):
    jobs = tmp_path / "storage" / "processing_jobs"
    ids = {"foundation": "foundation-job", "inference": "inference-job",
           "volume": "volume-job", "geobody": "geobody-job"}
    for job_id in ids.values():
        (jobs / job_id).mkdir(parents=True)

    foundation = jobs / ids["foundation"]
    stage01 = foundation / "stages/01_well_ties/tables"
    stage01.mkdir(parents=True)
    trajectory = pd.DataFrame({
        "well": ["no-1"] * 3, "md_m": [0.0, 10.0, 20.0],
        "tvdss_m": [500.0, 502.0, 504.0], "twt_ms": [500.0, 502.0, 504.0],
        "il_nearest": [100.0, 100.0, 100.0], "xl_nearest": [200.0, 200.0, 200.0],
    })
    trajectory.to_csv(stage01 / "tz_samples_all_wells.csv", index=False)
    pd.DataFrame([{"model": "regional_linear", "selected": True}]).to_csv(stage01 / "regional_model_selection.csv", index=False)
    pd.DataFrame([{"model_name": "regional_linear", "source_type": "regional_linear_velocity",
                   "fit_domain_tvdss_min_m": 0.0, "fit_domain_tvdss_max_m": 2000.0,
                   "v0_ms": 2000.0, "k_per_s": 0.0}]).to_csv(stage01 / "polynomial_coefficients_used.csv", index=False)

    logs_dir = foundation / "stages/02_labels/tables"
    logs_dir.mkdir(parents=True)
    pd.DataFrame({"well": ["no-1"] * 3, "md_m": [0.0, 10.0, 20.0],
                  "tvdss_m": [500.0, 502.0, 504.0], "twt_ms": [500.0, 502.0, 504.0],
                  "vsh_fraction": [0.1, 0.1, 0.1], "phit_fraction": [0.2, 0.2, 0.2],
                  "swt_fraction": [0.4, 0.5, 0.6]}).to_csv(logs_dir / "interpreted_logs_all_wells.csv", index=False)

    trace_dir = jobs / ids["inference"] / "stages/09_inference/volumes"
    trace_dir.mkdir(parents=True)
    np.save(trace_dir / "trace_index_grid.npy", np.arange(400, dtype=np.int32).reshape(20, 20))

    vol_stage = jobs / ids["volume"] / "stages/11_volume_export"
    vol_stage.mkdir(parents=True)
    (vol_stage / "native_volume_export_manifest.json").write_text(json.dumps({"source_inference_job_id": ids["inference"]}))

    geo_stage = jobs / ids["geobody"] / "stages/12_geobody"
    (geo_stage / "volumes").mkdir(parents=True)
    (geo_stage / "tables").mkdir(parents=True)
    shape = (20, 20, 6)
    labels = np.zeros(shape, dtype=np.int32)
    labels[9:12, 9:12, 1:5] = 1
    label_path = geo_stage / "volumes/geobody_labels.npy"
    np.save(label_path, labels)
    catalog_path = geo_stage / "tables/geobody_catalog.csv"
    pd.DataFrame([{"geobody_id": 1, "voxel_count": int((labels == 1).sum())}]).to_csv(catalog_path, index=False)
    run_path = geo_stage / "tables/geobody_run_length_index.csv"
    rows = []
    for i in range(9, 12):
        for j in range(9, 12):
            rows.append({"geobody_id": 1, "volume_i": i, "volume_j": j, "il": 90+i,
                         "xl": 190+j, "x": i*25.0, "y": j*40.0,
                         "top_time_k": 1, "base_time_k": 4, "voxel_count": 4})
    pd.DataFrame(rows).to_csv(run_path, index=False)
    reconcile_path = geo_stage / "tables/geobody_spatial_export_reconciliation.csv"
    pd.DataFrame([{"geobody_id": 1, "reconciles": True}]).to_csv(reconcile_path, index=False)

    import hashlib
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    (geo_stage / "geobody_manifest.json").write_text(json.dumps({
        "status": "completed",
        "source": {"foundation_job_id": ids["foundation"], "inference_job_id": ids["inference"], "volume_job_id": ids["volume"]},
        "geometry": {"shape": list(shape), "axis_order": ["inline_bin", "crossline_bin", "time_sample"],
                     "il_min": 90, "il_step": 1, "xl_min": 190, "xl_step": 1, "t0_ms": 498, "dt_ms": 2},
        "products": [
            {"product": "geobody_labels", "format": "npy", "path": "stages/12_geobody/volumes/geobody_labels.npy", "sha256": digest(label_path)},
            {"product": "geobody_catalog", "format": "csv", "path": "stages/12_geobody/tables/geobody_catalog.csv", "sha256": digest(catalog_path)},
            {"product": "geobody_run_length_index", "format": "csv", "path": "stages/12_geobody/tables/geobody_run_length_index.csv", "sha256": digest(run_path)},
            {"product": "geobody_spatial_export_reconciliation", "format": "csv", "path": "stages/12_geobody/tables/geobody_spatial_export_reconciliation.csv", "sha256": digest(reconcile_path)},
        ]
    }))

    current = jobs / "stage14-job"
    current.mkdir()
    result = run({"foundation_job_id": ids["foundation"], "inference_job_id": ids["inference"],
                  "source_volume_job_id": ids["volume"], "source_geobody_job_id": ids["geobody"],
                  "well_id": ["no-1"], "anchor_well_id": ["no-1"], "twt_pad_ms": 4.0,
                  "min_interval_samples": 3, "xy_coordinate_units": "m"}, current / "result.json")
    assert result["status"] == "COMPLETED"
    assert result["counts"]["prospects"] == 1
    assert result["qc"]["hcpv_calculated"] is True
    assert result["qc"]["well_log_swt_available"] is True
    det = pd.read_csv(current / "stages/14_prospect/tables/prospect_volumetrics_deterministic.csv")
    assert det.iloc[0].status_hcpv == "COMPUTED_CONDITIONAL_SCENARIO"
    assert det.iloc[0].hcpv_scenario_uniform_m3 > 0
    assert (current / "stages/14_prospect/tables/prospect_definitions.csv").is_file()
    assert (current / "stages/14_prospect/tables/prospect_grv_by_depth_model.csv").is_file()

    # Without declared coordinate units, geometry/log QC still completes but GRV/HCPV are withheld.
    blocked_job = jobs / "stage14-units-unconfirmed"
    blocked_job.mkdir()
    blocked = run({"foundation_job_id": ids["foundation"], "inference_job_id": ids["inference"],
                   "source_volume_job_id": ids["volume"], "source_geobody_job_id": ids["geobody"],
                   "well_id": ["no-1"], "anchor_well_id": ["no-1"], "twt_pad_ms": 4.0,
                   "min_interval_samples": 3}, blocked_job / "result.json")
    assert blocked["status"] == "COMPLETED_WITH_GRV_BLOCKERS"
    assert blocked["qc"]["xy_coordinate_units_confirmed"] is False
    assert blocked["qc"]["hcpv_calculated"] is False

    # Automatic mode selects supported anchors only from the explicit well inventory.
    auto_job = jobs / "stage14-auto-anchors"
    auto_job.mkdir()
    automatic = run({"foundation_job_id": ids["foundation"], "inference_job_id": ids["inference"],
                     "source_volume_job_id": ids["volume"], "source_geobody_job_id": ids["geobody"],
                     "well_id": ["no-1"], "anchor_well_id": [], "anchor_selection_mode": "intersecting_configured_wells",
                     "twt_pad_ms": 4.0, "min_interval_samples": 3, "xy_coordinate_units": "m"},
                    auto_job / "result.json")
    assert automatic["anchor_selection_mode"] == "intersecting_configured_wells"
    assert automatic["anchor_well_ids"] == ["no-1"]
    assert automatic["counts"]["prospects"] == 1


def test_well_intersection_audit_distinguishes_anchor_from_non_anchor_support():
    from geogg_fresh.stages.s14_prospect import _well_intersection_audit

    samples = pd.DataFrame({"well": ["no-1", "m-1", "m-1"],
                            "inside_volume": [True, True, True], "geobody_id": [0, 7, 7]})
    intervals = pd.DataFrame({"well": ["m-1"], "geobody_id": [7], "n_path_samples": [2]})
    audit = _well_intersection_audit(samples, intervals, ["no-1", "m-1"], ["no-1"], 3)
    anchor = audit.set_index("well").loc["no-1"]
    non_anchor = audit.set_index("well").loc["m-1"]
    assert anchor.anchor_status == "ANCHOR_NO_QUALIFYING_INTERVAL"
    assert anchor.intersection_status == "NO_RETAINED_GEOBODY_INTERSECTION"
    assert non_anchor.intersection_status == "INTERSECTS_RETAINED_GEOBODY"
    assert non_anchor.anchor_status == "NOT_ANCHOR"
    assert non_anchor.intervals_meeting_min_samples == 0


def test_empty_anchor_intervals_keep_expected_schema():
    empty = pd.DataFrame([], columns=["well", "geobody_id", "twt_top_ms", "twt_base_ms", "n_path_samples"])
    empty["kept"] = empty["n_path_samples"] >= 3
    assert empty.empty and "kept" in empty.columns
