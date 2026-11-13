"""Read-only API for Stage 12 geobody products and visualisation assets."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import ProcessingJob

router = APIRouter()


def _stage(job_id: str, db: Session) -> Path:
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Processing job not found")
    job_dir = Path(job.job_dir).expanduser().resolve()
    if job_dir.name != job_id or not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Processing job directory is unavailable")
    stage = (job_dir / "stages" / "12_geobody").resolve()
    try:
        stage.relative_to(job_dir)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="Invalid geobody stage path") from exc
    if not stage.is_dir():
        raise HTTPException(status_code=404, detail="Stage 12 geobody outputs are not available for this job")
    return stage


def _json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _catalog(stage: Path) -> list[dict]:
    path = stage / "tables" / "geobody_catalog.csv"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Geobody catalogue is unavailable")
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key, value in list(row.items()):
            if value == "":
                row[key] = None
                continue
            try:
                row[key] = int(value) if value.lstrip("-").isdigit() else float(value)
            except (ValueError, AttributeError):
                if value in {"True", "False"}:
                    row[key] = value == "True"
    return rows


def _body(stage: Path, geobody_id: int) -> dict:
    row = next((r for r in _catalog(stage) if int(r["geobody_id"]) == geobody_id), None)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Geobody {geobody_id} not found")
    manifest = _json(stage / "geobody_manifest.json")
    products = manifest.get("exports", {})
    meshes = products.get("meshes", [])
    mesh = next((x for x in meshes if int(x.get("geobody_id", -1)) == geobody_id and x.get("status") == "PASS"), None)
    figures = [x for x in products.get("visualisations", []) if int(x.get("geobody_id", -1)) == geobody_id]
    base = f"/api/v1/geobodies/jobs/{stage.parent.parent.name}"
    if mesh:
        mesh = {**mesh, "download_url": f"{base}/items/{geobody_id}/mesh"}
    figures = [{**item, "url": f"{base}/figures/{item['name']}"} for item in figures]
    return {"geobody": row, "mesh": mesh, "figures": figures,
            "interpretation_status": manifest.get("interpretation_status"),
            "source": manifest.get("source"), "limitations": manifest.get("limitations", [])}


@router.get("/jobs/{job_id}")
def get_geobody_summary(job_id: str, db: Session = Depends(get_db)):
    stage = _stage(job_id, db)
    summary = _json(stage / "geobody_manifest.json")
    if not summary:
        raise HTTPException(status_code=404, detail="Geobody manifest is unavailable")
    return summary


@router.get("/jobs/{job_id}/items")
def list_geobodies(job_id: str, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                   db: Session = Depends(get_db)):
    stage = _stage(job_id, db)
    rows = sorted(_catalog(stage), key=lambda r: int(r.get("voxel_count") or 0), reverse=True)
    return {"total": len(rows), "offset": offset, "limit": limit, "geobodies": rows[offset:offset + limit]}


@router.get("/jobs/{job_id}/items/{geobody_id}")
def get_geobody(job_id: str, geobody_id: int, db: Session = Depends(get_db)):
    return _body(_stage(job_id, db), geobody_id)




@router.get("/jobs/{job_id}/items/{geobody_id}/voxel-preview")
def get_geobody_voxel_preview(job_id: str, geobody_id: int, max_points: int = Query(30000, ge=1000, le=60000), db: Session = Depends(get_db)):
    """Return a bounded IJK preview from the authoritative Stage 12 label volume."""
    stage = _stage(job_id, db)
    _body(stage, geobody_id)  # Validate that the ID belongs to the catalogue.
    labels_path = stage / "volumes" / "geobody_labels.npy"
    if not labels_path.is_file():
        raise HTTPException(status_code=404, detail="Geobody label volume is unavailable")
    try:
        labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
        if labels.ndim != 3:
            raise ValueError("Expected a 3D geobody label volume")
        # Scan one inline slab at a time to bound peak memory for large volumes.
        total = 0
        mins = np.array(labels.shape, dtype=np.int64)
        maxs = np.full(3, -1, dtype=np.int64)
        for i in range(labels.shape[0]):
            ijk = np.argwhere(labels[i] == geobody_id)
            if ijk.size:
                total += int(ijk.shape[0])
                mins = np.minimum(mins, np.array([i, ijk[:, 0].min(), ijk[:, 1].min()]))
                maxs = np.maximum(maxs, np.array([i, ijk[:, 0].max(), ijk[:, 1].max()]))
        if total == 0:
            raise HTTPException(status_code=404, detail=f"No voxels found for geobody {geobody_id}")
        stride = max(1, int(np.ceil(total / max_points)))
        coords = []
        seen = 0
        for i in range(labels.shape[0]):
            ijk = np.argwhere(labels[i] == geobody_id)
            if not ijk.size:
                continue
            # Deterministic global stride without assembling the full coordinate array.
            first = (-seen) % stride
            selected = ijk[first::stride]
            if selected.size:
                coords.extend([[i, int(j), int(k)] for j, k in selected])
            seen += int(ijk.shape[0])
        return {"job_id": job_id, "geobody_id": geobody_id, "coordinate_order": ["I", "J", "K"],
                "voxel_count": total, "preview_count": len(coords), "sampling_stride": stride,
                "bounds_ijk": {"min": mins.tolist(), "max": maxs.tolist()},
                "coordinates": coords, "is_sampled": stride > 1}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read geobody label volume: {exc}") from exc


@router.get("/jobs/{job_id}/items/{geobody_id}/slice")
def get_geobody_slice(job_id: str, geobody_id: int, plane: str = Query(..., pattern="^(inline|crossline|time)$"),
                      coordinate: float | None = None, db: Session = Depends(get_db)):
    stage = _stage(job_id, db)
    body = _body(stage, geobody_id)["geobody"]
    manifest = _json(stage / "geobody_manifest.json")
    geometry = manifest.get("geometry", {})
    labels_path = stage / "volumes" / "geobody_labels.npy"
    if not labels_path.is_file():
        raise HTTPException(status_code=404, detail="Geobody label volume is unavailable")
    labels = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if plane == "inline":
        value = float(coordinate if coordinate is not None else body["centroid_inline"])
        step = float(geometry.get("il_step") or 1); origin = float(geometry.get("il_min") or 0)
        idx = int(round((value - origin) / step))
        if not 0 <= idx < labels.shape[0]: raise HTTPException(422, detail="Inline coordinate is outside the volume")
        arr, axes = np.asarray(labels[idx, :, :]).T == geobody_id, ["time_sample", "crossline_index"]
    elif plane == "crossline":
        value = float(coordinate if coordinate is not None else body["centroid_crossline"])
        step = float(geometry.get("xl_step") or 1); origin = float(geometry.get("xl_min") or 0)
        idx = int(round((value - origin) / step))
        if not 0 <= idx < labels.shape[1]: raise HTTPException(422, detail="Crossline coordinate is outside the volume")
        arr, axes = np.asarray(labels[:, idx, :]).T == geobody_id, ["time_sample", "inline_index"]
    else:
        value = float(coordinate if coordinate is not None else body["centroid_twt_ms"])
        dt = float(geometry.get("dt_ms") or 1); origin = float(geometry.get("t0_ms") or 0)
        idx = int(round((value - origin) / dt))
        if not 0 <= idx < labels.shape[2]: raise HTTPException(422, detail="Time coordinate is outside the volume")
        arr, axes = np.asarray(labels[:, :, idx]).T == geobody_id, ["crossline_index", "inline_index"]
    stride = max(1, int(np.ceil(max(arr.shape) / 512)))
    arr = arr[::stride, ::stride]
    return {"geobody_id": geobody_id, "plane": plane, "coordinate": value, "axes": axes,
            "downsample_stride": stride, "shape": list(arr.shape), "body_cells": int(arr.sum()),
            "mask": arr.astype(np.uint8).tolist(), "interpretation_status": manifest.get("interpretation_status")}


@router.get("/jobs/{job_id}/items/{geobody_id}/mesh")
def get_geobody_mesh(job_id: str, geobody_id: int, db: Session = Depends(get_db)):
    stage = _stage(job_id, db)
    body = _body(stage, geobody_id)
    mesh = body.get("mesh")
    if not mesh:
        raise HTTPException(status_code=404, detail="No mesh was exported for this geobody (mesh limits may apply)")
    path = (stage / str(mesh.get("stage_relative_path") or "")).resolve()
    try: path.relative_to(stage)
    except ValueError as exc: raise HTTPException(403, detail="Invalid mesh path") from exc
    if not path.is_file(): raise HTTPException(404, detail="Mesh file is unavailable")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@router.get("/jobs/{job_id}/exports/{product}")
def get_geobody_export(job_id: str, product: str, db: Session = Depends(get_db)):
    allowed = {"geobody_mask": "geobody_mask.sgy", "geobody_id": "geobody_id.sgy"}
    if product not in allowed:
        raise HTTPException(404, detail="Unknown geobody export")
    path = _stage(job_id, db) / "segy" / allowed[product]
    if not path.is_file(): raise HTTPException(404, detail=f"{product} SEG-Y export is unavailable")
    return FileResponse(path, media_type="application/octet-stream", filename=path.name)


@router.get("/jobs/{job_id}/figures/{name}")
def get_geobody_figure(job_id: str, name: str, db: Session = Depends(get_db)):
    if Path(name).name != name or not name.lower().endswith(".png"):
        raise HTTPException(422, detail="Figure name must be a PNG filename")
    stage = _stage(job_id, db)
    path = (stage / "figures" / name).resolve()
    try: path.relative_to((stage / "figures").resolve())
    except ValueError as exc: raise HTTPException(403, detail="Invalid figure path") from exc
    if not path.is_file(): raise HTTPException(404, detail="Geobody figure not found")
    return FileResponse(path, media_type="image/png", filename=path.name)
