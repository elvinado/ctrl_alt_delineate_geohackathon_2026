"""Regression tests for the Stage 12 worker's direct-script import mode and SEG-Y exports."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from _fixtures import write_segy


def test_geobody_worker_imports_backend_packages_when_launched_as_script(tmp_path):
    backend_root = Path(__file__).resolve().parents[1]
    script_dir = backend_root / "scripts"
    code = (
        "import sys, types; "
        "shared = types.ModuleType('scripts_utils'); "
        "shared.validate_outputs = lambda **kw: (lambda fn: fn); "
        "sys.modules['scripts_utils'] = shared; "
        f"sys.path.insert(0, {str(script_dir)!r}); "
        "import process_D_geobody; "
        "from scripts.geogg_fresh.segy_write import ReferenceSurvey, write_product, reopen_qc; "
        "print('worker imports passed')"
    )
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=tmp_path, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "worker imports passed" in result.stdout


def test_geobody_segy_exports_reopen_against_stage11_reference(tmp_path):
    # Import after the worker has established the backend package root.
    backend_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_root / "scripts"))
    import types
    shared = types.ModuleType("scripts_utils")
    shared.validate_outputs = lambda **kw: (lambda fn: fn)
    sys.modules["scripts_utils"] = shared
    import process_D_geobody as worker
    from scripts.geogg_fresh.segy import SegyFile

    jobs = tmp_path / "processing_jobs"
    source_job = jobs / "source-job"
    stage11 = source_job / "stages" / "11_volume_export"
    inference_stage = source_job / "stages" / "09_inference"
    reference_dir = stage11 / "segy"
    reference_dir.mkdir(parents=True)
    reference_path = reference_dir / "prediction_lithology.sgy"
    write_segy(
        reference_path,
        np.arange(6, dtype=np.float32).reshape(2, 3),
        np.array([100, 101]), np.array([200, 200]),
        np.array([0.0, 25.0]), np.array([0.0, 0.0]),
        dt_us=2000, fmt=5, delay_ms=0,
    )
    stage_out = jobs / "geobody-job" / "stages" / "12_geobody"
    trace_grid = np.array([[0], [1]], dtype=np.int64)
    (inference_stage / "volumes").mkdir(parents=True)
    np.save(inference_stage / "volumes" / "trace_index_grid.npy", trace_grid)
    labels = np.array([[[0, 1, 1]], [[2, 2, -1]]], dtype=np.int32)
    valid = np.array([[[1, 1, 1]], [[1, 1, 0]]], dtype=bool)
    manifest = {
        "geometry": {"dt_ms": 2.0, "trace_index_grid": "volumes/trace_index_grid.npy"},
        "derived_segy_export": {"products": [{
            "product": "prediction_lithology", "format": "segy",
            "path": "segy/prediction_lithology.sgy", "status": "PASS",
        }]},
    }

    products = worker._export_segy_products(
        stage_out, stage11, inference_stage, manifest, labels, valid,
        {"source_product": "prediction_lithology"},
    )
    assert {row["product"] for row in products} == {"geobody_mask", "geobody_id"}
    assert all(row["status"] == "PASS" for row in products)
    for name in ("geobody_mask", "geobody_id"):
        output = stage_out / "segy" / f"{name}.sgy"
        sidecar = stage_out / "segy" / f"{name}.manifest.json"
        assert output.is_file() and sidecar.is_file()
        with SegyFile(output) as segy:
            assert segy.n_traces == 2 and segy.ns == 3
            assert segy.format_code == 5
            data = segy.read_traces([0, 1])
            if name == "geobody_mask":
                np.testing.assert_array_equal(data, [[0, 1, 1], [1, 1, -999.25]])
            else:
                np.testing.assert_array_equal(data, [[0, 1, 1], [2, 2, -999.25]])


def test_geobody_segy_export_reports_missing_stage09_trace_grid(tmp_path):
    backend_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_root / "scripts"))
    import types
    shared = types.ModuleType("scripts_utils")
    shared.validate_outputs = lambda **kw: (lambda fn: fn)
    sys.modules["scripts_utils"] = shared
    import process_D_geobody as worker

    stage11 = tmp_path / "source" / "stages" / "11_volume_export"
    inference_stage = tmp_path / "source" / "stages" / "09_inference"
    stage11.mkdir(parents=True)
    inference_stage.mkdir(parents=True)
    reference = stage11 / "segy" / "prediction_lithology.sgy"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"placeholder")
    manifest = {
        "geometry": {"trace_index_grid": "volumes/trace_index_grid.npy"},
        "derived_segy_export": {"products": [{
            "product": "prediction_lithology", "format": "segy",
            "path": "segy/prediction_lithology.sgy", "status": "PASS",
        }]},
    }
    with np.testing.assert_raises_regex(RuntimeError, "Canonical Stage 09 trace_index_grid"):
        worker._export_segy_products(
            tmp_path / "out", stage11, inference_stage, manifest,
            np.zeros((1, 1, 1), dtype=np.int32),
            np.ones((1, 1, 1), dtype=bool),
            {"source_product": "prediction_lithology"},
        )


def test_geobody_visualisations_use_supported_matplotlib_colormap_api(tmp_path):
    backend_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_root / "scripts"))
    import types
    shared = types.ModuleType("scripts_utils")
    shared.validate_outputs = lambda **kw: (lambda fn: fn)
    sys.modules["scripts_utils"] = shared
    import process_D_geobody as worker

    labels = np.zeros((4, 5, 6), dtype=np.int32)
    labels[1:3, 1:4, 2:5] = 1
    catalog = [{
        "geobody_id": 1,
        "voxel_count": 18,
        "centroid_il_index": 1.5,
        "centroid_xl_index": 2.0,
        "centroid_time_sample": 3.0,
    }]
    stage_out = tmp_path / "stages" / "12_geobody"
    products = worker._write_visualisations(stage_out, labels, catalog)

    names = {item["name"] for item in products}
    assert names == {
        "geobody_overview_3d.png",
        "geobody_catalog.png",
        "geobody_0001_slices.png",
    }
    assert all((stage_out / "figures" / name).is_file() for name in names)
    assert all((stage_out / "figures" / name).stat().st_size > 0 for name in names)


def test_stage12_spatial_exports_are_lossless_and_twt_only(tmp_path):
    backend_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(backend_root / "scripts"))
    import types
    shared = types.ModuleType("scripts_utils")
    shared.validate_outputs = lambda **kw: (lambda fn: fn)
    sys.modules["scripts_utils"] = shared
    import process_D_geobody as worker
    import pandas as pd

    labels = np.zeros((2, 2, 8), dtype=np.int32)
    labels[0, 0, 1:3] = 1
    labels[0, 0, 5:7] = 1
    labels[1, 1, 2:4] = 2
    geometry = {"il_min": 1000, "il_step": 2, "xl_min": 2000, "xl_step": 4,
                "t0_ms": 500.0, "dt_ms": 2.0}
    x = np.array([[100.0, np.nan], [np.nan, 300.0]])
    y = np.array([[200.0, np.nan], [np.nan, 400.0]])

    run_path = tmp_path / "geobody_run_length_index.csv"
    stats = worker._write_geobody_run_length_index(labels, geometry, run_path, x, y, slab_rows=1)
    runs = pd.read_csv(run_path)
    assert stats["run_rows"] == 3
    assert stats["total_exported_voxels"] == 6
    assert stats["voxel_counts_by_geobody"] == {1: 4, 2: 2}
    assert runs.voxel_count.sum() == int((labels > 0).sum())
    assert list(runs[runs.geobody_id == 1].voxel_count) == [2, 2]
    assert runs.loc[runs.geobody_id == 2, "x"].iloc[0] == 300.0
    assert "tvdss_top_regional_m" not in runs.columns

    voxel_path = tmp_path / "geobody_voxel_xyz.csv"
    voxel_stats = worker._write_geobody_voxel_xyz(labels, geometry, voxel_path, x, y, chunk_voxels=2)
    voxels = pd.read_csv(voxel_path)
    assert voxel_stats["voxel_rows"] == 6
    assert len(voxels) == 6
    assert "tvdss_regional_m" not in voxels.columns


def test_stage12_output_validation_decorator_is_on_worker_entrypoint_only():
    import ast

    backend_root = Path(__file__).resolve().parents[1]
    stage_file = backend_root / "scripts" / "geogg_fresh" / "stages" / "s12_geobody.py"
    worker_file = backend_root / "scripts" / "process_D_geobody.py"

    def decorated_functions(path):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        return {
            node.name: [ast.unparse(item) for item in node.decorator_list]
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and any("validate_outputs" in ast.unparse(item) for item in node.decorator_list)
        }

    assert decorated_functions(stage_file) == {}
    assert decorated_functions(worker_file) == {
        "main": ["validate_outputs(expected_outputs=[OUTPUT, SUMMARY])"]
    }

    stage_tree = ast.parse(stage_file.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "_load_trace_xy" and node.decorator_list
        for node in ast.walk(stage_tree)
    )
