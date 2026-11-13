"""Atomic stage provenance and output integrity records."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import OutputRecord, StageResult


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a file SHA-256 using bounded-memory streaming reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, default=str)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def register_output(stage_dir: Path, path: str | Path, *, role: str, media_type: str = "application/octet-stream") -> OutputRecord:
    """Register a file only if it exists inside the stage directory."""
    file_path = Path(path).resolve()
    stage_root = stage_dir.resolve()
    if stage_root not in file_path.parents:
        raise ValueError("Output file must be inside the stage directory")
    if not file_path.is_file():
        raise FileNotFoundError(file_path)
    return OutputRecord(
        path=file_path.relative_to(stage_root).as_posix(),
        role=role,
        media_type=media_type,
        size_bytes=file_path.stat().st_size,
        sha256=sha256_file(file_path),
    )


def write_stage_provenance(
    *,
    provenance_dir: Path,
    stage_id: str,
    run_id: str,
    config_hash: str,
    result: StageResult,
    inputs: list[dict[str, Any]],
) -> Path:
    """Write an immutable-by-convention stage record with config/input/output lineage."""
    if not stage_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for char in stage_id):
        raise ValueError("stage_id must be a simple identifier")
    path = provenance_dir / f"{stage_id}.json"
    if path.exists():
        raise FileExistsError(f"Stage provenance already exists; refusing to overwrite: {path}")
    payload = {
        "schema_version": 1,
        "run_id": run_id,
        "stage_id": stage_id,
        "status": result.status.value,
        "started_at": None,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_hash,
        "inputs": inputs,
        "outputs": [item.__dict__ for item in result.outputs],
        "summary": result.summary,
        "warnings": result.warnings,
        "limitations": result.limitations,
        "blockers": result.blockers,
        "checks": result.checks,
        "metrics": result.metrics,
    }
    _atomic_json(path, payload)
    return path
