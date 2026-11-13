"""Read completed processing-job history from the shared job registry."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.models import ProcessingJob
from app.processing_jobs import TERMINAL_STATUSES, _read_json, _refresh_status
from app.storage import ensure_storage_root


def _worker_description(db: Session, metadata: dict[str, Any], operation: str) -> str:
    """Read the worker module's first docstring line when available."""
    recorded = metadata.get("description") or metadata.get("summary")
    if recorded:
        return str(recorded).strip()

    script_name = metadata.get("python_script")
    if isinstance(script_name, str) and Path(script_name).name == script_name and script_name.endswith(".py"):
        script_path = ensure_storage_root(db) / "scripts" / script_name
        try:
            tree = ast.parse(script_path.read_text(encoding="utf-8"))
            docstring = ast.get_docstring(tree)
            if docstring:
                return docstring.strip().splitlines()[0].strip()
        except (OSError, SyntaxError, UnicodeError):
            pass

    return operation.replace("_", " ").replace("-", " ").strip() or "Description unavailable"


def _output_names(result: dict[str, Any], job_dir: Path) -> list[str]:
    """Return concise output names, accepting common result.json formats."""
    raw_outputs = result.get("outputs") or result.get("results") or []
    if isinstance(raw_outputs, (str, dict)):
        raw_outputs = [raw_outputs]

    names: list[str] = []
    if isinstance(raw_outputs, list):
        for item in raw_outputs:
            if isinstance(item, dict):
                value = item.get("path") or item.get("file") or item.get("filename") or item.get("name")
                if value:
                    names.append(Path(str(value)).name)
            elif item:
                names.append(Path(str(item)).name)

    # Fallback to files in the job directory when result.json has no output list.
    if not names and job_dir.is_dir():
        excluded = {"job.json", "result.json", "inputs.json", "local.log"}
        names = sorted(
            path.name for path in job_dir.iterdir()
            if path.is_file() and path.name not in excluded and not path.name.startswith("slurm-")
        )
    return list(dict.fromkeys(names))


def execute(*, db: Session, project_id: str | None, parameters: dict[str, Any]) -> dict[str, Any]:
    """Refresh job states and return terminal jobs, newest first."""
    try:
        limit = max(1, min(int(parameters.get("limit", 100)), 500))
    except (TypeError, ValueError):
        limit = 100

    jobs = db.query(ProcessingJob).order_by(ProcessingJob.created_at.desc()).limit(500).all()
    items: list[dict[str, Any]] = []
    for job in jobs:
        _refresh_status(job, db)
        db.refresh(job)
        if job.status not in TERMINAL_STATUSES:
            continue

        job_dir = Path(job.job_dir)
        metadata = _read_json(job_dir / "job.json")
        result = _read_json(Path(job.result_path))
        python_file = metadata.get("python_script") or f"{job.operation}.py"
        summary = (
            result.get("summary")
            or result.get("description")
            or result.get("message")
            or metadata.get("summary")
            or metadata.get("description")
            or job.error_message
            or _worker_description(db, metadata, job.operation)
        )
        items.append({
            "python_file": Path(str(python_file)).name,
            "id": job.id,
            "status": job.status,
            "summary": str(summary).replace("\n", " ").strip(),
            "outputs": _output_names(result, job_dir),
            "created_at": job.created_at.isoformat() if job.created_at else None,
        })
        if len(items) >= limit:
            break

    return {"count": len(items), "items": items, "limit": limit}
