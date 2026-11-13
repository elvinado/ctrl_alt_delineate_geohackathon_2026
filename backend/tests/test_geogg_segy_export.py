import hashlib
import json
from pathlib import Path

import numpy as np

from _fixtures import write_segy
from scripts.geogg_fresh.volume_products import _write_stage11_segy_products
from scripts.geogg_fresh.segy import SegyFile


def test_stage11_exports_prediction_npy_to_segy_and_reopens_bit_exact(tmp_path):
    foundation = tmp_path / "processing_jobs" / "foundation"
    inventory = foundation / "stages" / "00_inventory"
    inventory.mkdir(parents=True)
    reference = foundation / "full.sgy"
    write_segy(reference, np.arange(6, dtype=np.float32).reshape(2, 3),
                np.array([100, 101]), np.array([200, 200]),
                np.array([0.0, 25.0]), np.array([0.0, 0.0]),
                dt_us=2000, fmt=5, delay_ms=0)
    (inventory / "data_manifest.json").write_text(json.dumps({"inputs": [{
        "kind": "seismic", "dataset_id": "full-id", "path": str(reference)}]}))

    source_job = tmp_path / "processing_jobs" / "inference-job"
    volumes = source_job / "stages" / "09_inference" / "volumes"
    volumes.mkdir(parents=True)
    prediction = np.asarray([[[0.1, 0.2, 0.3]], [[0.4, 0.5, 0.6]]], dtype=np.float32)
    valid = np.ones(prediction.shape, dtype=np.uint8)
    grid = np.asarray([[0], [1]], dtype=np.int64)
    np.save(volumes / "prediction_lithology.npy", prediction)
    np.save(volumes / "prediction_valid.npy", valid)
    np.save(volumes / "trace_index_grid.npy", grid)
    prediction_path = volumes / "prediction_lithology.npy"
    products = {"lithology": {"npy": "stages/09_inference/volumes/prediction_lithology.npy",
                               "sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()}}
    out = tmp_path / "stage11"
    (out / "tables").mkdir(parents=True)

    inventory_rows, qc_rows = _write_stage11_segy_products(
        out=out, storage_root=tmp_path,
        geometry={"foundation_job_id": "foundation", "shape": [2, 1, 3],
                  "input_datasets": {"full": {"dataset_id": "full-id"}}},
        volumes=volumes, source_job=source_job, tasks=["lithology"], products=products,
        valid_path=volumes / "prediction_valid.npy", trace_grid=grid)

    assert inventory_rows[0]["status"] == "PASS"
    assert all(row["result"] in {"PASS", "INFO"} for row in qc_rows)
    exported = out / "segy" / "prediction_lithology.sgy"
    assert exported.is_file()
    with SegyFile(exported) as segy:
        assert segy.n_traces == 2 and segy.ns == 3 and segy.format_code == 5
        np.testing.assert_array_equal(segy.read_traces([0, 1]), prediction.reshape(2, 3))
    sidecar = json.loads((out / "segy" / "prediction_lithology.manifest.json").read_text())
    assert sidecar["canonical_format"] == "NPY"
    assert sidecar["status"] == "PASS"


def test_stage11_refuses_stale_or_modified_canonical_npy(tmp_path):
    foundation = tmp_path / "processing_jobs" / "foundation"
    inventory = foundation / "stages" / "00_inventory"
    inventory.mkdir(parents=True)
    reference = foundation / "full.sgy"
    write_segy(reference, np.arange(2, dtype=np.float32).reshape(2, 1),
                np.array([100, 101]), np.array([200, 200]),
                np.array([0.0, 25.0]), np.array([0.0, 0.0]),
                dt_us=2000, fmt=5, delay_ms=0)
    (inventory / "data_manifest.json").write_text(json.dumps({"inputs": [{
        "kind": "seismic", "dataset_id": "full-id", "path": str(reference)}]}))
    source_job = tmp_path / "processing_jobs" / "inference-job"
    volumes = source_job / "stages" / "09_inference" / "volumes"
    volumes.mkdir(parents=True)
    prediction = np.asarray([[[0.1]], [[0.2]]], dtype=np.float32)
    np.save(volumes / "prediction_lithology.npy", prediction)
    np.save(volumes / "prediction_valid.npy", np.ones(prediction.shape, dtype=np.uint8))
    grid = np.asarray([[0], [1]], dtype=np.int64)
    path = volumes / "prediction_lithology.npy"
    products = {"lithology": {"npy": "stages/09_inference/volumes/prediction_lithology.npy",
                               "sha256": "0" * 64}}
    out = tmp_path / "stage11"
    (out / "tables").mkdir(parents=True)
    import pytest
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _write_stage11_segy_products(
            out=out, storage_root=tmp_path,
            geometry={"foundation_job_id": "foundation", "shape": [2, 1, 1],
                      "input_datasets": {"full": {"dataset_id": "full-id"}}},
            volumes=volumes, source_job=source_job, tasks=["lithology"], products=products,
            valid_path=volumes / "prediction_valid.npy", trace_grid=grid)
