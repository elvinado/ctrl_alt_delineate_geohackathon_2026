"""Deterministically inspect all artifacts in one processing job folder."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

MAX_TEXT_BYTES = 2_000_000
MAX_ROWS = 100_000

def _safe_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {"value_type": type(value).__name__}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}

def _csv_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        numeric: dict[str, list[float]] = {name: [] for name in columns}
        categories: dict[str, Counter] = {name: Counter() for name in columns}
        rows = 0
        for row in reader:
            rows += 1
            if rows > MAX_ROWS:
                break
            for name in columns:
                raw = (row.get(name) or "").strip()
                if not raw:
                    continue
                try:
                    number = float(raw)
                    if math.isfinite(number):
                        numeric[name].append(number)
                        continue
                except ValueError:
                    pass
                if len(categories[name]) < 30:
                    categories[name][raw] += 1
    numeric_fields = {name: {"count": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)} for name, values in numeric.items() if values}
    categorical_fields = {name: dict(counts.most_common(8)) for name, counts in categories.items() if counts}
    return {"format": "CSV", "rows_scanned": min(rows, MAX_ROWS), "truncated": rows > MAX_ROWS, "columns": columns, "numeric_fields": numeric_fields, "categorical_fields": categorical_fields}

def _inspect_file(path: Path, relative: str) -> dict[str, Any]:
    size = path.stat().st_size
    suffix = path.suffix.lower()
    item: dict[str, Any] = {"name": relative, "size_bytes": size}
    if suffix == ".csv":
        item.update(_csv_summary(path))
    elif suffix == ".json":
        if size > MAX_TEXT_BYTES:
            item.update({"format": "JSON", "skipped": "JSON file exceeds 2 MB inspection limit."})
        else:
            data = _safe_json(path)
            item.update({"format": "JSON", "top_level_keys": list(data)[:60], "summary": str(data.get("summary") or data.get("message") or "")[:500]})
    elif suffix in {".txt", ".log", ".md", ".yaml", ".yml"}:
        if size > MAX_TEXT_BYTES:
            item.update({"format": suffix.lstrip("." ).upper(), "skipped": "Text file exceeds 2 MB inspection limit."})
        else:
            content = path.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            item.update({"format": suffix.lstrip(".").upper(), "line_count": len(lines), "preview": "\n".join(lines[:8])[:1200]})
    elif suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
        item.update({"format": suffix.lstrip(".").upper(), "type": "visualization/document", "note": "File identified; visual contents are not interpreted by this report."})
    else:
        item.update({"format": suffix.lstrip(".").upper() or "unknown", "type": "binary or unsupported", "note": "File inventoried; contents were not parsed."})
    return item

def execute(*, db, project_id: str | None, parameters: dict[str, Any]) -> dict[str, Any]:
    from app.processing_jobs import _safe_json
    from app.storage import configured_storage_roots

    job_id = str(parameters.get("job_id") or "").strip()
    if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("A valid processing job ID is required.")
    job_dir = None
    for root in configured_storage_roots(db):
        jobs_root = (root / "processing_jobs").resolve()
        candidate_path = root / "processing_jobs" / job_id
        if candidate_path.is_symlink():
            continue
        candidate = candidate_path.resolve()
        if jobs_root in candidate.parents and candidate.is_dir():
            job_dir = candidate
            break
    if job_dir is None:
        raise ValueError(f"Processing run {job_id} was not found.")

    metadata = _safe_json(job_dir / "job.json")
    result = _safe_json(job_dir / "result.json")
    artifacts = []
    skipped = []
    for path in sorted(job_dir.rglob("*"), key=lambda p: str(p).lower()):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            resolved = path.resolve()
            if job_dir not in resolved.parents:
                continue
            relative = str(path.relative_to(job_dir))
            if relative in {"job.json", "result.json"}:
                continue
            artifacts.append(_inspect_file(path, relative))
        except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
            skipped.append({"name": str(path.relative_to(job_dir)), "reason": str(exc)[:240]})

    csv_files = [item for item in artifacts if item.get("format") == "CSV"]
    script_name = str(metadata.get("python_script") or result.get("python_script") or "")
    operation = str(metadata.get("operation") or result.get("operation") or "")
    target_well = str(metadata.get("target_well") or result.get("target_well") or "").strip()
    findings = [f"{len(artifacts)} output artifact(s) were inspected, including {len(csv_files)} CSV file(s)."]
    for item in csv_files:
        findings.append(f"{item['name']}: {item.get('rows_scanned', 0)} row(s) scanned across {len(item.get('columns', []))} column(s).")
    if not artifacts:
        findings.append("No parseable output artifacts were found in this processing folder.")
    return {
        "skill": "processing_output_folder_analysis", "job_id": job_id,
        "python_script": script_name or None,
        "operation": operation or None,
        "target_well": target_well or None,
        "status": result.get("status") or metadata.get("status") or "unknown",
        "created_at": metadata.get("created_at") or result.get("created_at"),
        "summary": " ".join(findings), "findings": findings,
        "artifacts": artifacts, "skipped": skipped,
        "limitations": ["This is a deterministic inventory and statistical summary, not an AI-generated geological interpretation.", "Visualization and unsupported binary contents are identified but not interpreted.", "Numeric summaries describe file contents only; scientific meaning depends on the processing method and data context."],
    }
