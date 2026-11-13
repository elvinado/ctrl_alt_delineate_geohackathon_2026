import csv
import json
from pathlib import Path

import numpy as np

from app.geobody_api import (
    get_geobody, get_geobody_export, get_geobody_figure, get_geobody_mesh,
    get_geobody_slice, get_geobody_summary, list_geobodies,
)


class _FakeJob:
    def __init__(self, job_dir):
        self.job_dir = str(job_dir)


class _FakeDB:
    def __init__(self, job_id, job_dir):
        self.job_id, self.job = job_id, _FakeJob(job_dir)

    def get(self, model, key):
        return self.job if key == self.job_id else None


def test_geobody_read_only_api_products_and_slices(tmp_path):
    job_id = "test-geobody-job"
    job_dir = tmp_path / job_id
    stage = job_dir / "stages" / "12_geobody"
    for folder in ("tables", "volumes", "meshes", "segy", "figures"):
        (stage / folder).mkdir(parents=True)
    (stage / "geobody_manifest.json").write_text(json.dumps({
        "status": "completed", "interpretation_status": "EXPLORATORY_UNCALIBRATED_CANDIDATE",
        "geometry": {"shape": [2, 2, 3], "il_min": 100, "il_step": 2,
                     "xl_min": 200, "xl_step": 2, "t0_ms": 0, "dt_ms": 4},
        "exports": {
            "meshes": [{"geobody_id": 1, "status": "PASS", "stage_relative_path": "meshes/geobody_0001.ply"}],
            "visualisations": [{"geobody_id": 1, "name": "geobody_0001_slices.png"}],
        },
    }))
    with (stage / "tables" / "geobody_catalog.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["geobody_id", "voxel_count", "centroid_il_index",
            "centroid_xl_index", "centroid_time_sample", "centroid_inline", "centroid_crossline", "centroid_twt_ms"])
        writer.writeheader()
        writer.writerow({"geobody_id": 1, "voxel_count": 1, "centroid_il_index": 0,
                         "centroid_xl_index": 0, "centroid_time_sample": 1,
                         "centroid_inline": 100, "centroid_crossline": 200, "centroid_twt_ms": 4})
    labels = np.zeros((2, 2, 3), dtype=np.int32)
    labels[0, 0, 1] = 1
    np.save(stage / "volumes" / "geobody_labels.npy", labels)
    (stage / "meshes" / "geobody_0001.ply").write_bytes(b"ply")
    (stage / "segy" / "geobody_mask.sgy").write_bytes(b"mask")
    (stage / "figures" / "geobody_0001_slices.png").write_bytes(b"png")
    db = _FakeDB(job_id, job_dir)

    assert get_geobody_summary(job_id, db)["status"] == "completed"
    assert list_geobodies(job_id, 50, 0, db)["total"] == 1
    assert get_geobody(job_id, 1, db)["mesh"]["download_url"].endswith("/items/1/mesh")
    assert get_geobody_slice(job_id, 1, "time", 4, db)["body_cells"] == 1
    assert get_geobody_mesh(job_id, 1, db).path.name.endswith(".ply")
    assert get_geobody_export(job_id, "geobody_mask", db).path.name.endswith(".sgy")
    assert get_geobody_figure(job_id, "geobody_0001_slices.png", db).path.name.endswith(".png")
