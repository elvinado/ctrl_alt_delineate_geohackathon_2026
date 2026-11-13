import hashlib
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
from geogg_fresh.stages.s13_well_geobody_analysis import (
    run, _intervals, _body_geometry, _property_stats, _body_column_geometry,
    _make_depth_model, _apply_column_depths,
)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_intervals_split_when_path_leaves_and_reenters_body():
    samples = pd.DataFrame({
        "well": ["no-1"] * 5,
        "md_m": [0, 10, 20, 30, 40],
        "tvdss_m": [100, 110, 120, 130, 140],
        "twt_ms": [100, 110, 120, 130, 140],
        "il": [1] * 5, "xl": [2] * 5,
        "volume_i": [0] * 5, "volume_j": [1] * 5, "time_k": list(range(5)),
        "geobody_id": [2, 2, 0, 2, 2], "inside_volume": [True] * 5,
        "has_geobody": [True, True, False, True, True],
    })
    _, intervals = _intervals(samples, "no-1")
    assert len(intervals) == 2
    assert intervals.geobody_id.tolist() == [2, 2]
    assert intervals.n_path_samples.tolist() == [2, 2]


def test_body_geometry_and_property_statistics():
    labels = np.zeros((3, 3, 4), dtype=np.int32)
    labels[0, 1, 1:3] = 4
    labels[2, 2, 2] = 4
    geometry = {"il_min": 100, "il_step": 2, "xl_min": 200, "xl_step": 4,
                "t0_ms": 500, "dt_ms": 2}
    body = _body_geometry(labels, 4, geometry)
    assert body["voxel_count_recounted"] == 3
    assert body["occupied_columns"] == 2
    assert body["twt_top_ms"] == 502
    assert body["twt_base_ms"] == 504
    phit = np.full(labels.shape, np.nan, dtype=np.float32)
    swt = np.full(labels.shape, np.nan, dtype=np.float32)
    phit[labels == 4] = [0.1, 0.2, 0.3]
    swt[labels == 4] = [0.4, 0.5, 0.6]
    stats = _property_stats(labels, 4, phit, swt)
    assert stats["joint_support_fraction"] == 1.0
    assert np.isclose(stats["phit_mean_voxel"], 0.2)
    assert np.isclose(stats["swt_mean_voxel"], 0.5)
    assert stats["property_coverage_gate"] == "FAIL_INSUFFICIENT_JOINT_SUPPORT"
    assert stats["phit_p10"] is not None and stats["phit_p90"] is not None
    assert np.isclose(stats["hcpf_mean_joint_support"], np.mean([0.1 * 0.6, 0.2 * 0.5, 0.3 * 0.4]))
    assert stats["provisional_volume_weighted_status"] == "NOT_COMPUTED_NO_SUPPORTED_DEPTH_MODEL"
    from geogg_fresh.stages.s13_well_geobody_analysis import _make_depth_model
    depth_model = _make_depth_model("linear", "test", np.array([0.0, 1000.0]),
                                    np.array([0.0, 2000.0]), (0.0, 1000.0))
    weighted = _property_stats(labels, 4, phit, swt, 0.5, 1, 100, depth_model,
                               {"t0_ms": 500.0, "dt_ms": 2.0})
    assert weighted["provisional_volume_weighted_status"] == "PROVISIONAL_VERTICAL_THICKNESS_WEIGHTED_PROXY"
    assert np.isclose(weighted["hcpf_mean_provisional_volume_weighted"], weighted["hcpf_mean_joint_support"])


def test_column_geometry_and_provisional_depth_conversion():
    labels = np.zeros((2, 2, 6), dtype=np.int32)
    labels[0, 0, [1, 2, 4]] = 7
    geometry = {"il_min": 100, "il_step": 2, "xl_min": 200, "xl_step": 4,
                "t0_ms": 500, "dt_ms": 2}
    columns = _body_column_geometry(labels, 7, geometry)
    assert len(columns) == 1
    assert columns.iloc[0].n_vertical_runs == 2
    assert columns.iloc[0].net_time_thickness_ms == 6
    model = _make_depth_model("linear_test", "regional_polynomial",
                              np.array([0, 1000]), np.array([0, 1000]), (0, 1000))
    alternate = _make_depth_model("alternate_regional", "regional_polynomial",
                                  np.array([0, 1000]), np.array([0, 1100]), (0, 1000))
    other_well_tdr = _make_depth_model("tdr_table_no-2", "well_specific_tdr_table",
                                       np.array([0, 1000]), np.array([0, 1500]), (0, 1000),
                                       {"well": "no-2"})
    result = _apply_column_depths(
        columns, "no-1", 7,
        {"linear_test": model, "alternate_regional": alternate, "tdr_table_no-2": other_well_tdr},
        {"no-1": "linear_test"}, None, geometry)
    assert result.iloc[0].depth_status == "PROVISIONAL_CONVERSION_WITHIN_MODEL_SUPPORT"
    assert result.iloc[0].tvdss_top_primary_m == 502
    assert result.iloc[0].tvdss_base_primary_m == 508
    assert result.iloc[0].net_vertical_thickness_primary_m == 6
    assert result.iloc[0].n_models_covering_column == 2
    assert result.iloc[0].tvdss_top_model_spread_m > 0


def test_end_to_end_job_linked_intersection_and_no_legacy_paths(tmp_path):
    storage = tmp_path / "storage"
    jobs = storage / "processing_jobs"
    ids = {"foundation": "foundation-job", "inference": "inference-job",
           "volume": "volume-job", "geobody": "geobody-job"}
    for job_id in ids.values():
        (jobs / job_id).mkdir(parents=True)

    foundation = jobs / ids["foundation"]
    trajectory_dir = foundation / "stages/01_well_ties/tables"
    trajectory_dir.mkdir(parents=True)
    one_well = pd.DataFrame({
        "well": ["no-1"] * 4, "md_m": [0, 10, 20, 30], "tvdss_m": [100, 110, 120, 130],
        "twt_ms": [500, 502, 504, 506], "il_nearest": [100, 100, 102, 102],
        "xl_nearest": [200, 200, 204, 204], "tz_source": ["test"] * 4,
        "tie_status": ["test"] * 4, "tz_strong_extrapolation": [False] * 4,
    })
    second_well = one_well.copy()
    second_well["well"] = "no-2"
    second_well["md_m"] = second_well.md_m + 1000
    pd.concat([one_well, second_well], ignore_index=True).to_csv(
        trajectory_dir / "tz_samples_all_wells.csv", index=False)

    pd.DataFrame([{"model": "regional_linear", "selected": True}]).to_csv(
        trajectory_dir / "regional_model_selection.csv", index=False)
    pd.DataFrame([{"model_name": "regional_linear", "source_type": "regional_linear_velocity",
                   "fit_domain_tvdss_min_m": 0.0, "fit_domain_tvdss_max_m": 2000.0,
                   "v0_ms": 2000.0, "k_per_s": 0.0}]).to_csv(
        trajectory_dir / "polynomial_coefficients_used.csv", index=False)

    inference = jobs / ids["inference"]
    vol_dir = inference / "stages/09_inference/volumes"
    vol_dir.mkdir(parents=True)
    shape = (2, 2, 5)
    phit = np.full(shape, 0.2, dtype=np.float32)
    swt = np.full(shape, 0.4, dtype=np.float32)
    for name, arr in (("prediction_phit", phit), ("prediction_swt", swt)):
        np.save(vol_dir / f"{name}.npy", arr)

    volume_job = jobs / ids["volume"]
    volume_stage = volume_job / "stages/11_volume_export"
    volume_stage.mkdir(parents=True)
    volume_products = []
    geometry = {"shape": list(shape), "axis_order": ["inline_bin", "crossline_bin", "time_sample"],
                "il_min": 100, "il_step": 2, "xl_min": 200, "xl_step": 4,
                "t0_ms": 500, "dt_ms": 2}
    for name in ("prediction_phit", "prediction_swt"):
        p = vol_dir / f"{name}.npy"
        volume_products.append({"product": name, "format": "npy",
            "storage_root_relative_path": f"processing_jobs/{ids['inference']}/stages/09_inference/volumes/{p.name}",
            "sha256": _hash(p)})
    (volume_stage / "native_volume_export_manifest.json").write_text(json.dumps({
        "source_inference_job_id": ids["inference"], "geometry": geometry, "products": volume_products
    }))

    geobody_job = jobs / ids["geobody"]
    geobody_stage = geobody_job / "stages/12_geobody"
    (geobody_stage / "volumes").mkdir(parents=True)
    (geobody_stage / "tables").mkdir(parents=True)
    labels = np.zeros(shape, dtype=np.int32)
    labels[0, 0, 0:3] = 1
    labels[1, 1, 3:5] = 2
    labels_path = geobody_stage / "volumes/geobody_labels.npy"
    np.save(labels_path, labels)
    catalog_path = geobody_stage / "tables/geobody_catalog.csv"
    pd.DataFrame([{"geobody_id": 1, "voxel_count": 3, "stable_id": "GB-1"},
                  {"geobody_id": 2, "voxel_count": 2, "stable_id": "GB-2"}]).to_csv(catalog_path, index=False)
    run_index_path = geobody_stage / "tables/geobody_run_length_index.csv"
    pd.DataFrame([{"geobody_id": 1, "voxel_count": 3}, {"geobody_id": 2, "voxel_count": 2}]).to_csv(run_index_path, index=False)
    reconciliation_path = geobody_stage / "tables/geobody_spatial_export_reconciliation.csv"
    pd.DataFrame([{"geobody_id": 1, "reconciles": True}, {"geobody_id": 2, "reconciles": True}]).to_csv(reconciliation_path, index=False)
    (geobody_stage / "geobody_manifest.json").write_text(json.dumps({
        "status": "completed", "source": {"foundation_job_id": ids["foundation"], "inference_job_id": ids["inference"],
                   "volume_job_id": ids["volume"]}, "geometry": geometry,
        "products": [
            {"product": "geobody_labels", "format": "npy", "path": "stages/12_geobody/volumes/geobody_labels.npy", "sha256": _hash(labels_path)},
            {"product": "geobody_catalog", "format": "csv", "path": "stages/12_geobody/tables/geobody_catalog.csv", "sha256": _hash(catalog_path)},
            {"product": "geobody_run_length_index", "format": "csv", "path": "stages/12_geobody/tables/geobody_run_length_index.csv", "sha256": _hash(run_index_path)},
            {"product": "geobody_spatial_export_reconciliation", "format": "csv", "path": "stages/12_geobody/tables/geobody_spatial_export_reconciliation.csv", "sha256": _hash(reconciliation_path)},
        ],
        "exports": {"spatial": {"catalogue_reconciliation": {
            "path": "stages/12_geobody/tables/geobody_spatial_export_reconciliation.csv",
            "all_counts_reconcile": True}, "optional_voxel_xyz": {"enabled": False},
            "xy_coordinate_status": "AVAILABLE"}}
    }))

    current = jobs / "stage13-job"
    current.mkdir()
    result_path = current / "result.json"
    result = run({"foundation_job_id": ids["foundation"], "inference_job_id": ids["inference"],
                  "source_volume_job_id": ids["volume"], "source_geobody_job_id": ids["geobody"],
                  "well_id": ["no-1"]}, result_path)
    assert result["status"] == "COMPLETED"
    assert result["counts"]["distinct_intersected_geobodies"] == 2
    assert result["selected_geobody"]["geobody_id"] == 1
    assert result["selected_geobody"]["catalogue_voxel_count_reconciles"] is True
    assert result["selected_geobody"]["property_status"] == "COMPUTED_FROM_FROZEN_STAGE11_PREDICTIONS"
    assert result["selected_geobody"]["property_coverage_gate"] == "FAIL_INSUFFICIENT_JOINT_SUPPORT"
    assert np.isclose(result["selected_geobody"]["hcpf_mean_joint_support"], 0.12)
    assert result["selected_geobody"]["provisional_volume_weighted_status"] == "PROVISIONAL_VERTICAL_THICKNESS_WEIGHTED_PROXY"
    assert (current / "stages/13_well_geobody_analysis/tables/specialist_phit_swt_summary.csv").is_file()
    assert result["specialist_phit_swt_analysis"]["source"] == "Stage 11 canonical frozen prediction_phit.npy and prediction_swt.npy"

    selected_columns = pd.read_csv(current / "stages/13_well_geobody_analysis/tables/selected_body_columns.csv")
    assert not selected_columns.empty
    assert selected_columns.iloc[0].primary_depth_model == "regional_linear"
    assert selected_columns.iloc[0].depth_status == "PROVISIONAL_CONVERSION_WITHIN_MODEL_SUPPORT"
    depth_comparison = pd.read_csv(current / "stages/13_well_geobody_analysis/tables/depth_model_comparison.csv")
    assert "COMPARABLE_WITHIN_MODEL_SUPPORT" in depth_comparison.status.tolist()
    assert (current / "stages/13_well_geobody_analysis/tables/selected_body_columns.csv").is_file()
    stage13 = current / "stages/13_well_geobody_analysis"
    progress = json.loads((stage13 / "progress.json").read_text())
    assert progress["phase"] == "COMPLETED"
    assert not (stage13 / "checkpoint.json").exists()
    assert not list((stage13 / "tables").glob("*.partial.csv"))
    assert result["spatial_exports"]["owner_stage"] == "12_geobody"
    assert result["spatial_exports"]["source_job_id"] == ids["geobody"]
    assert result["spatial_exports"]["run_length_index"]["path"].endswith("geobody_run_length_index.csv")
    assert not (stage13 / "tables/geobody_run_length_index.csv").exists()

    # Multiple wells are analysed only when explicitly enumerated in the config.
    current_multi = jobs / "stage13-multi-well-job"
    current_multi.mkdir()
    multi_wells = run({"foundation_job_id": ids["foundation"], "inference_job_id": ids["inference"],
                       "source_volume_job_id": ids["volume"], "source_geobody_job_id": ids["geobody"],
                       "well_id": ["no-1", "no-2"]}, current_multi / "result.json")
    assert multi_wells["well_selection"]["mode"] == "explicit_list"
    assert multi_wells["well_ids"] == ["no-1", "no-2"]
    assert multi_wells["counts"]["wells_analysed"] == 2
    assert multi_wells["counts"]["wells_with_intersections"] == 2
    multi_samples = pd.read_csv(current_multi / "stages/13_well_geobody_analysis/tables/well_geobody_path_samples.csv")
    assert sorted(multi_samples.well.unique().tolist()) == ["no-1", "no-2"]
    assert multi_wells["spatial_exports"]["owner_stage"] == "12_geobody"

    # A valid run with no path/volume intersection must complete explicitly, not fail.
    labels[:] = 0
    np.save(labels_path, labels)
    manifest_path = geobody_stage / "geobody_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["products"][0]["sha256"] = _hash(labels_path)
    manifest["exports"]["spatial"]["catalogue_reconciliation"]["all_counts_reconcile"] = False
    reconciliation_path = geobody_stage / "tables/geobody_spatial_export_reconciliation.csv"
    pd.DataFrame([{"geobody_id": 1, "reconciles": False}, {"geobody_id": 2, "reconciles": False}]).to_csv(reconciliation_path, index=False)
    for row in manifest["products"]:
        if row["product"] == "geobody_spatial_export_reconciliation":
            row["sha256"] = _hash(reconciliation_path)
    manifest_path.write_text(json.dumps(manifest))
    current_no_hit = jobs / "stage13-no-hit-job"
    current_no_hit.mkdir()
    no_hit = run({"foundation_job_id": ids["foundation"], "inference_job_id": ids["inference"],
                  "source_volume_job_id": ids["volume"], "source_geobody_job_id": ids["geobody"],
                  "well_id": ["no-1"]}, current_no_hit / "result.json")
    assert no_hit["status"] == "COMPLETED_WITH_UPSTREAM_SPATIAL_EXPORT_WARNINGS"
    assert no_hit["counts"]["distinct_intersected_geobodies"] == 0
    assert no_hit["spatial_exports"]["catalogue_reconciliation"]["all_counts_reconcile"] is False
    assert no_hit["selected_geobody"] is None


def test_stage13_yaml_uses_confirmed_geobody_job_id():
    import yaml

    config_path = SCRIPTS / "E_well_geobody_analysis.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["well_geobody_analysis"]
    assert config["source_geobody_job_id"] == "45a68c60-0e1d-40c8-a6e8-f3c4a0643800"
    assert config["foundation_job_id"] == "d94ffa95-ca92-4807-9057-b91f077161da"
    assert config["inference_job_id"] == "b8176911-618c-48a3-961b-04e89343a3e7"
    assert config["source_volume_job_id"] == "21b13e5f-158f-4d27-9d2c-86bae4d0df41"
    assert config["well_id"] == ["eo-1", "eo-2", "m-1", "no-1", "no-2", "no-4", "np-1", "np-2", "o-1", "o-2", "o-3", "p-1", "s-1"]
    assert "export_voxel_xyz" not in config
    stage12_config = yaml.safe_load((SCRIPTS / "D_geobody.yaml").read_text(encoding="utf-8"))["geobody"]
    assert stage12_config["export_voxel_xyz"] is False
    assert stage12_config["voxel_export_chunk_size"] == 250000


def test_vectorized_column_grouping_matches_expected_runs_across_slabs():
    rng = np.random.default_rng(73)
    labels = np.zeros((19, 11, 37), dtype=np.int32)
    labels[rng.random(labels.shape) < 0.17] = 9
    geometry = {"il_min": 100, "il_step": 2, "xl_min": 200, "xl_step": 4,
                "t0_ms": 500, "dt_ms": 2}
    result = _body_column_geometry(labels, 9, geometry).sort_values(["volume_i", "volume_j"]).reset_index(drop=True)
    expected = {}
    for i, j, k in zip(*np.nonzero(labels == 9)):
        expected.setdefault((int(i), int(j)), []).append(int(k))
    assert len(result) == len(expected)
    for row in result.itertuples():
        ks = sorted(expected[(row.volume_i, row.volume_j)])
        cuts = np.flatnonzero(np.diff(ks) > 1) + 1
        runs = [[int(g[0]), int(g[-1])] for g in np.split(np.asarray(ks), cuts)]
        assert json.loads(row.time_runs_json) == runs
        assert row.voxel_count == len(ks)
        assert row.n_vertical_runs == len(runs)
        assert row.net_time_thickness_ms == len(ks) * 2


def test_lossless_run_length_index_preserves_disconnected_runs_and_counts(tmp_path):
    from geogg_fresh.stages.s12_geobody import _write_geobody_run_length_index

    labels = np.zeros((2, 2, 8), dtype=np.int32)
    labels[0, 0, [0, 1, 4]] = 3
    labels[0, 0, 5] = 4
    labels[1, 1, 2:5] = 4
    geometry = {"il_min": 100, "il_step": 2, "xl_min": 200, "xl_step": 4,
                "t0_ms": 500, "dt_ms": 2}
    x = np.full((2, 2), np.nan)
    y = np.full((2, 2), np.nan)
    x[0, 0] = 123.5
    out = tmp_path / "runs.csv"
    stats = _write_geobody_run_length_index(labels, geometry, out, x, y, slab_rows=1)
    frame = pd.read_csv(out)
    assert stats["total_exported_voxels"] == int(np.count_nonzero(labels)) == 7
    assert stats["run_rows"] == 4
    assert frame.loc[(frame.volume_i == 0) & (frame.volume_j == 0), "geobody_id"].tolist() == [3, 3, 4]
    assert frame.loc[(frame.volume_i == 0) & (frame.volume_j == 0), "voxel_count"].tolist() == [2, 1, 1]
    assert frame.loc[0, "x"] == 123.5


def test_trace_xy_mapping_uses_live_trace_grid_and_segy_scalar(tmp_path):
    sys.path.insert(0, str(ROOT / "tests"))
    from _fixtures import write_segy
    from geogg_fresh.stages.s12_geobody import _load_trace_xy

    inference = tmp_path / "inference-job"
    inference_stage = inference / "stages/09_inference"
    (inference_stage / "volumes").mkdir(parents=True)
    trace_grid = np.array([[0, -1], [-1, 1]], dtype=np.int32)
    np.save(inference_stage / "volumes/trace_index_grid.npy", trace_grid)
    volume_stage = tmp_path / "volume-job/stages/11_volume_export"
    (volume_stage / "segy").mkdir(parents=True)
    segy_path = volume_stage / "segy/reference.sgy"
    write_segy(segy_path, np.zeros((2, 3), dtype=np.float32),
               [100, 102], [200, 204], [500000, 500025], [1800000, 1800025])
    manifest = {"geometry": {"trace_index_grid": "volumes/trace_index_grid.npy"},
                "derived_segy_export": {"products": [{"product": "prediction_lithology", "format": "segy",
                    "path": "segy/reference.sgy", "sha256": _hash(segy_path), "status": "PASS"}]}}
    x, y, status = _load_trace_xy(inference, volume_stage, manifest, (2, 2, 3))
    assert status == "AVAILABLE"
    assert x[0, 0] == 500000
    assert y[0, 0] == 1800000
    assert x[1, 1] == 500025
    assert y[1, 1] == 1800025
    assert np.isnan(x[0, 1]) and np.isnan(y[1, 0])
