"""Pre-flight validation for Process D (GeoGG geobody extraction, stage 12).

Process D has no datasets of its own: it segments the sand-score cube packaged by a completed
Process C run. This handler checks that the named Process C job is fit to build on, forecasts
whether the thresholds will produce one giant body, and returns a plan. It does NOT submit the
job: submission must go through the approved backend processing service after the user confirms.
It never executes user-provided code and never opens a cube.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

SKILL_NAME = "process_d_geobody"
PYTHON_SCRIPT = "process_D_geobody.py"
SOURCE_SUMMARY = "geogg_volume_products_summary.json"
SOURCE_PRODUCT = "prediction_lithology"
MASK_PRODUCT = "prediction_valid"
S10 = "stages/10_summary"
S11 = "stages/11_volume_export"
MAX_JSON_BYTES = 8_000_000

# Settings of the one reference run. The script reads its own configuration; these are used only to forecast.
REFERENCE_LOW_THRESHOLD = 0.35
REFERENCE_HIGH_THRESHOLD = 0.65
REFERENCE_RUN = {"note": "Observed in one reference run; yours will differ with survey size and hardware.", "run_time_minutes": 18}
# Site-percolation threshold of a simple cubic lattice with face (6-) connectivity.
PERCOLATION_6 = 0.3116
PERCENTILES = (("p01", 0.01), ("p05", 0.05), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p95", 0.95), ("p99", 0.99))

REQUIREMENTS: list[dict[str, str]] = [
    {"input": "A completed Process C (volume products) job", "need": "required",
     "why": "Registers the sand-score cube and its validity mask with verified checksums."},
    {"input": "The Process B job it packaged, with its cube files", "need": "required, linked automatically",
     "why": "The cube itself stays in the Process B job; Process D reads it from there."},
    {"input": "Free disk for label volumes, two SEG-Y files and a run-length index", "need": "required",
     "why": "The outputs are several times the size of one seismic volume."},
]

WHAT_HAPPENS: list[str] = [
    "Preflight: confirms the three source jobs link up and the cube matches its checksum.",
    "Segmentation: keeps voxels above the low threshold, joins touching ones into groups, and keeps the groups that contain "
    "at least one voxel above the high threshold and meet the minimum size.",
    "Exports: a catalogue of bodies, label volumes, two SEG-Y files (mask and body number), 3-D meshes and figures.",
]


def _issue(level: str, code: str, message: str, fix: str) -> dict[str, str]:
    """level: error (blocks the run) | warning (run allowed, expect a poor result)."""
    return {"level": level, "code": code, "message": message, "how_to_fix": fix}


def _is_file(path: Path, root: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and root in path.resolve().parents
    except OSError:
        return False


def _read_json(path: Path, root: Path) -> dict[str, Any]:
    try:
        if not _is_file(path, root) or path.stat().st_size > MAX_JSON_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _read_rows(path: Path, root: Path) -> list[dict[str, str]]:
    try:
        if not _is_file(path, root) or path.stat().st_size > 1_000_000:
            return []
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def share_above(threshold: float, row: dict[str, Any]) -> float | None:
    """Share of the cube above a score, by straight-line interpolation between the recorded percentiles."""
    points = [(value, level) for name, level in PERCENTILES if (value := _num(row.get(name))) is not None]
    low, high = _num(row.get("min")), _num(row.get("max"))
    if low is not None:
        points.insert(0, (low, 0.0))
    if high is not None:
        points.append((high, 1.0))
    if len(points) < 2:
        return None
    if threshold <= points[0][0]:
        return 1.0
    if threshold >= points[-1][0]:
        return 0.0
    for (x0, c0), (x1, c1) in zip(points, points[1:]):
        if x0 <= threshold <= x1:
            below = c0 if x1 == x0 else c0 + (c1 - c0) * (threshold - x0) / (x1 - x0)
            return round(1.0 - below, 3)
    return None


def validate_source(job_dir: Path, low: float = REFERENCE_LOW_THRESHOLD) -> dict[str, Any]:
    """Check one Process C job folder and the Process B job it points to. Pure file inspection."""
    job_dir = job_dir.resolve()
    jobs_root = job_dir.parent
    issues: list[dict[str, str]] = []
    result = _read_json(job_dir / "result.json", job_dir)
    summary = _read_json(job_dir / SOURCE_SUMMARY, job_dir)
    validation = _read_json(job_dir / S11 / "validation_summary.json", job_dir)
    manifest = _read_json(job_dir / S11 / "native_volume_export_manifest.json", job_dir)
    status = str(result.get("status") or summary.get("status") or "unknown")
    if not summary or "volume_export_stage" not in summary:
        issues.append(_issue("error", "no_summary", f"'{SOURCE_SUMMARY}' was not found, or is not a Process C summary.",
                             "Choose a job produced by process_C_volume_products.py that ran to the end."))
    if status.lower() not in {"succeeded", "success", "completed"}:
        issues.append(_issue("error", "not_succeeded", f"The Process C job status is '{status}'.",
                             "Wait for it to finish, or re-run Process C."))
    if summary and str(validation.get("status") or manifest.get("status") or "").lower() != "completed":
        issues.append(_issue("error", "export_stage_incomplete", "The export stage (11) of the Process C job is not completed.",
                             "Re-run Process C so that stage 11 completes."))
    products = {str(p.get("product")): p for p in manifest.get("products") or [] if isinstance(p, dict)}
    integrity = {r.get("product"): str(r.get("status")).upper() for r in _read_rows(job_dir / S11 / "tables" / "product_integrity.csv", job_dir)}
    inference_id = str(summary.get("source_inference_job_id") or "")
    for name, role in ((SOURCE_PRODUCT, "the sand-score cube"), (MASK_PRODUCT, "the validity mask")):
        if summary and name not in products:
            issues.append(_issue("error", "product_not_registered", f"{role.capitalize()} ('{name}') is not registered in the Process C manifest.",
                                 "Re-run Process C; it registers the cubes of the Process B job."))
            continue
        if integrity.get(name) not in (None, "PASS"):
            issues.append(_issue("error", "checksum_failed", f"{role.capitalize()} failed its checksum in Process C.",
                                 "The file changed after it was predicted. Re-run Process B and Process C."))
        relative = str((products.get(name) or {}).get("storage_root_relative_path") or "")
        parts = Path(relative).parts
        if len(parts) > 2 and parts[0] == "processing_jobs" and ".." not in parts:
            target = jobs_root.joinpath(*parts[1:])
            if not _is_file(target, jobs_root):
                issues.append(_issue("error", "cube_missing", f"{role.capitalize()} file is missing from the Process B job: {relative}.",
                                     "Restore the file, or re-run Process B and Process C."))

    stats = next((r for r in _read_rows(job_dir / S10 / "tables" / "volume_statistics.csv", job_dir)
                  if r.get("attribute") == SOURCE_PRODUCT.replace("prediction_", "")), None)
    forecast = None
    if stats:
        switched_on = share_above(low, stats)
        median = _num(stats.get("p50"))
        forecast = {"low_threshold": low, "share_of_cube_above_low_threshold": switched_on, "cube_median_score": median,
                    "percolation_limit": PERCOLATION_6,
                    "method": "straight-line interpolation between the cube's recorded percentiles"}
        if switched_on is not None and switched_on > PERCOLATION_6:
            safer = next((round(v, 2) for name, _ in PERCENTILES[4:] if (v := _num(stats.get(name))) is not None and v > low), None)
            issues.append(_issue(
                "warning", "giant_body_expected",
                f"About {switched_on * 100:.0f}% of the cube lies above the low threshold of {low:g}. Above roughly "
                f"{PERCOLATION_6 * 100:.0f}%, touching cells join into one body that spans the whole survey.",
                "Expect one giant body and little else. Raise the low threshold in the process configuration"
                + (f" (about {safer:g} would switch on a quarter of the cube or less)" if safer else "") + " before running."))
    elif summary:
        issues.append(_issue("warning", "no_statistics", "The cube's value statistics were not found in the Process C job.",
                             "The run can proceed, but its outcome cannot be forecast."))

    geometry = manifest.get("geometry") if isinstance(manifest.get("geometry"), dict) else {}
    shape = geometry.get("shape") or []
    cells = shape[0] * shape[1] * shape[2] if len(shape) == 3 and all(isinstance(n, int) for n in shape) else None
    segy_bytes = next((_num(p.get("bytes")) for p in (summary.get("derived_segy_export") or {}).get("products") or []
                       if isinstance(p, dict) and _num(p.get("bytes"))), None)
    disk = round((cells * 5 + 2 * (segy_bytes or 0)) / 1e9, 1) if cells else None
    return {
        "ready": not any(i["level"] == "error" for i in issues), "issues": issues,
        "source": {
            "status": status, "source_inference_job_id": inference_id or None, "foundation_job_id": geometry.get("foundation_job_id"),
            "cube": SOURCE_PRODUCT, "cube_shape": shape or None, "forecast": forecast,
            "expected_disk_gb_at_least": disk,
            "disk_note": "Label volume (4 bytes per cell), mask (1 byte per cell) and two SEG-Y files. The run-length index is extra.",
        },
    }


def execute(*, project_id: str, parameters: dict[str, Any], db: Any) -> dict[str, Any]:
    """Validate the Process C job that Process D will segment and return a plan for the user to confirm.

    parameters["source_volume_job_id"]: the processing job ID of a completed Process C run.
    parameters["low_threshold"] (optional): a low threshold to forecast with, between 0 and 1. It does not change
    the process configuration; it only answers "what would happen at this setting".
    """
    base = {
        "skill": SKILL_NAME, "project_id": project_id, "python_script": PYTHON_SCRIPT,
        "target_well": None, "dataset_ids": [], "submitted": False, "confirmation_required": True,
        "requirements": REQUIREMENTS, "what_will_happen": WHAT_HAPPENS, "reference_run": REFERENCE_RUN,
        "reference_thresholds": {"low": REFERENCE_LOW_THRESHOLD, "high": REFERENCE_HIGH_THRESHOLD,
                                 "note": "Used in the reference run. The script reads its own configuration."},
    }
    job_id = str(parameters.get("source_volume_job_id") or "").strip()
    if not job_id:
        return {**base, "status": "needs_inputs", "issues": [],
                "summary": "No Process C job was named. Process D needs a completed volume-products run to segment."}
    if Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("A valid Process C processing job ID is required.")
    low = _num(parameters.get("low_threshold"))
    if parameters.get("low_threshold") not in (None, "") and (low is None or not 0.0 < low < 1.0):
        raise ValueError("low_threshold must be a number between 0 and 1.")

    from app.storage import configured_storage_roots

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
        return {**base, "status": "not_ready", "source_volume_job_id": job_id,
                "summary": f"Processing run {job_id} was not found.",
                "issues": [_issue("error", "job_not_found", f"Processing run {job_id} was not found.",
                                  "Check the job ID in the Ledger and try again.")]}
    report = validate_source(job_dir, low if low is not None else REFERENCE_LOW_THRESHOLD)
    errors = sum(1 for i in report["issues"] if i["level"] == "error")
    source = report["source"]
    summary = (f"{'Ready to run' if report['ready'] else 'Not ready'}: Process C job {job_id} is '{source['status']}'; "
               f"{errors} blocking issue(s), {len(report['issues']) - errors} warning(s).")
    forecast = source["forecast"]
    if forecast and forecast["share_of_cube_above_low_threshold"] is not None:
        summary += (f" At a low threshold of {forecast['low_threshold']:g}, about "
                    f"{forecast['share_of_cube_above_low_threshold'] * 100:.0f}% of the cube would be switched on.")
    return {
        **base, "status": "ready_for_confirmation" if report["ready"] else "not_ready", "summary": summary,
        "source_volume_job_id": job_id, "source": source, "issues": report["issues"],
        "expected_outputs": [
            "geogg_geobody_preflight.json and geogg_geobody_summary.json: checks, settings, counts and file list",
            "stages/12_geobody/tables: the body catalogue, the audit of every voxel group and a run-length index",
            "stages/12_geobody/volumes, segy, meshes and figures: label volumes, SEG-Y files, 3-D meshes and pictures",
        ],
        "next_action": (
            "Show this plan to the user, including any giant-body warning. After they confirm, submit the script through the "
            "approved processing service and report the returned job ID and status." if report["ready"] else
            "Explain each blocking issue and how to fix it. Do not submit."),
    }
