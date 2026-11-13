"""Read and validate immutable GeoGG model releases in managed storage."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import shutil

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.storage import configured_storage_roots

router = APIRouter()


def _release_roots(db: Session) -> list[Path]:
    return [root / "model_releases" for root in configured_storage_roots(db)]


def _safe_release_id(value: str) -> bool:
    return bool(value) and value not in {".", ".."} and Path(value).name == value and "/" not in value and "\\" not in value


def _locate_release(release_id: str, db: Session) -> Path:
    if not _safe_release_id(release_id):
        raise HTTPException(status_code=400, detail="Invalid model release ID")
    for root in _release_roots(db):
        raw_candidate = root / release_id
        if raw_candidate.is_symlink():
            continue
        candidate = raw_candidate.resolve()
        if candidate.parent == root.resolve() and candidate.is_dir():
            return candidate
    raise HTTPException(status_code=404, detail="Model release not found")


def _load_manifest(release_dir: Path) -> dict[str, Any]:
    path = release_dir / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise HTTPException(status_code=422, detail="Model release manifest is missing")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail="Model release manifest is invalid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("model_release_id") != release_dir.name:
        raise HTTPException(status_code=422, detail="Model release manifest identity does not match its directory")
    return manifest


def _validate_release(release_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        return {"valid": False, "checked_files": 0, "errors": ["Manifest has no file inventory"]}
    errors: list[str] = []
    checked = 0
    for relative, expected in files.items():
        rel = Path(str(relative))
        if rel.is_absolute() or ".." in rel.parts or not isinstance(expected, dict):
            errors.append(f"Unsafe file inventory entry: {relative}")
            continue
        raw_target = release_dir / rel
        if raw_target.is_symlink():
            errors.append(f"Missing or unsafe file: {relative}")
            continue
        target = raw_target.resolve()
        if release_dir.resolve() not in target.parents or not target.is_file():
            errors.append(f"Missing or unsafe file: {relative}")
            continue
        if target.stat().st_size != expected.get("bytes"):
            errors.append(f"Size mismatch: {relative}")
            continue
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected.get("sha256"):
            errors.append(f"Checksum mismatch: {relative}")
            continue
        checked += 1
    return {"valid": not errors, "checked_files": checked, "errors": errors}


@router.get("/model-releases", summary="List registered immutable model releases")
def list_model_releases(db: Session = Depends(get_db), limit: int = 100, offset: int = 0):
    if limit < 1 or limit > 500 or offset < 0:
        raise HTTPException(status_code=422, detail="limit must be 1–500 and offset must be non-negative")
    found: dict[str, dict[str, Any]] = {}
    for root in _release_roots(db):
        if not root.is_dir():
            continue
        for directory in root.iterdir():
            if not directory.is_dir() or directory.is_symlink() or directory.name.startswith("."):
                continue
            try:
                manifest = _load_manifest(directory)
            except HTTPException:
                continue
            found.setdefault(directory.name, {
                "model_release_id": directory.name,
                "status": manifest.get("status", "unknown"),
                "pipeline": manifest.get("pipeline"),
                "pipeline_version": manifest.get("pipeline_version"),
                "model_family": manifest.get("model_family"),
                "processing_job_id": manifest.get("processing_job_id"),
                "training_run_id": manifest.get("training_run_id"),
                "model_count": len(manifest.get("models", [])) if isinstance(manifest.get("models"), list) else 0,
                "models": manifest.get("models", []),
            })
    items = sorted(found.values(), key=lambda item: item["model_release_id"], reverse=True)
    return {"items": items[offset:offset + limit], "count": len(items), "limit": limit, "offset": offset}


@router.get("/model-releases/{release_id}", summary="Inspect a model release manifest")
def get_model_release(release_id: str, db: Session = Depends(get_db)):
    directory = _locate_release(release_id, db)
    manifest = _load_manifest(directory)
    return {"model_release_id": release_id, "manifest": manifest}


@router.post("/model-releases/{release_id}/validate", summary="Verify model release files and checksums")
def validate_model_release(release_id: str, db: Session = Depends(get_db)):
    directory = _locate_release(release_id, db)
    manifest = _load_manifest(directory)
    result = _validate_release(directory, manifest)
    return {"model_release_id": release_id, **result}


def _source_job_id(manifest: dict[str, Any]) -> str | None:
    provenance = manifest.get("provenance") if isinstance(manifest.get("provenance"), dict) else {}
    value = (manifest.get("foundation_job_id") or manifest.get("processing_job_id")
             or provenance.get("source_processing_job_id"))
    value = str(value or "").strip()
    return value if value and _safe_release_id(value) else None


def delete_model_releases_for_processing_job(job_id: str, db: Session) -> int:
    """Remove releases whose manifest identifies this source job; missing releases are harmless."""
    removed = 0
    for root in _release_roots(db):
        if not root.is_dir():
            continue
        for directory in root.iterdir():
            if not directory.is_dir() or directory.is_symlink() or directory.name.startswith("."):
                continue
            try:
                manifest = _load_manifest(directory)
            except HTTPException:
                continue
            if _source_job_id(manifest) != job_id:
                continue
            shutil.rmtree(directory)
            removed += 1
    return removed


@router.delete("/model-releases/{release_id}", status_code=204, summary="Delete a model release and its source processing job")
def delete_model_release(release_id: str, db: Session = Depends(get_db)):
    """Delete a release and its originating job, tolerating an already-deleted job."""
    from app.models import ProcessingJob
    from app.processing_jobs import delete_processing_job

    directory = _locate_release(release_id, db)
    manifest = _load_manifest(directory)
    source_job_id = _source_job_id(manifest)
    job = db.get(ProcessingJob, source_job_id) if source_job_id else None
    if job is not None:
        # Reuse the canonical job lifecycle checks and paired-release cleanup.
        delete_processing_job(source_job_id, db)
        return None
    # The job record may have been removed independently; clean up its conventional
    # backend-managed folder too, but only at the exact configured processing_jobs path.
    if source_job_id:
        for storage_root in configured_storage_roots(db):
            job_dir = storage_root / "processing_jobs" / source_job_id
            if job_dir.is_symlink():
                continue
            if job_dir.exists() and job_dir.is_dir() and job_dir.resolve().parent == (storage_root / "processing_jobs").resolve():
                shutil.rmtree(job_dir)
    if directory.exists() and not directory.is_symlink():
        shutil.rmtree(directory)
    return None
