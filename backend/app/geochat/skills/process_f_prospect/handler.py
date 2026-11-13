"""Pre-flight validation for Process F (GeoGG prospect definition, stage 14).

Process F has no datasets of its own: it builds on the geobodies of a completed Process D run and
on the well paths and interpreted logs of the Process A run behind it. This handler checks those
inputs, forecasts which results will be blocked, and returns a plan. It does NOT submit the job:
submission must go through the approved backend processing service after the user confirms. It
never executes user-provided code and never opens a volume.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

SKILL_NAME = "process_f_prospect"
PYTHON_SCRIPT = "process_F_prospect.py"
SOURCE_SUMMARY = "geogg_geobody_summary.json"
S12 = "stages/12_geobody"
MAX_JSON_BYTES = 8_000_000
MAX_CSV_BYTES = 20_000_000
DOMINANT_SHARE = 0.90

GEOBODY_FILES: list[tuple[str, str]] = [
    (f"{S12}/volumes/geobody_labels.npy", "the volume that says which body each voxel belongs to"),
    (f"{S12}/tables/geobody_catalog.csv", "the list of bodies"),
    (f"{S12}/tables/geobody_run_length_index.csv", "the per-trace index of body extents"),
]
FOUNDATION_FILES: list[tuple[str, str]] = [
    ("stages/01_well_ties/tables/tz_samples_all_wells.csv", "well paths in seismic time"),
    ("stages/01_well_ties/tables/regional_model_selection.csv", "the chosen time-depth model"),
    ("stages/01_well_ties/tables/polynomial_coefficients_used.csv", "time-depth model coefficients"),
    ("stages/02_labels/tables/interpreted_logs_all_wells.csv", "interpreted logs for net-to-gross, porosity and water saturation"),
]

# Cost of the one reference run (13 wells, 244 prospects cut from one survey-wide body).
REFERENCE_RUN = {"note": "Observed in one reference run where every prospect was a slice of one survey-wide body. "
                         "A run on well-separated bodies should be far smaller and faster.",
                 "run_time_hours": 4.4, "disk_written_gb": 188, "prospects": 244}

REQUIREMENTS: list[dict[str, str]] = [
    {"input": "A completed Process D (geobody) job", "need": "required",
     "why": "Supplies the body label volume, the catalogue and the per-trace index."},
    {"input": "The Process A job it descends from", "need": "required, linked automatically",
     "why": "Supplies well paths, the time-depth model and the interpreted logs."},
    {"input": "Confirmed map coordinate units", "need": "required for any volume",
     "why": "Without them cell area is unknown, so gross rock volume and everything built on it is blocked."},
    {"input": "A list of anchor wells", "need": "optional",
     "why": "Without one, every well that crosses a body is used."},
]

WHAT_HAPPENS: list[str] = [
    "Prospect definition: each continuous stretch of a well inside a body becomes one prospect, the part of the body within that time window.",
    "Geometry and volume: measures each prospect and tries to compute gross rock volume under several time-depth models.",
    "Well-log characterisation: reads net-to-gross, porosity and water saturation from the logs crossing each prospect, "
    "tests cutoffs and computes low, middle and high cases.",
    "Gates: decides per prospect what may be reported. A hydrocarbon pore volume is released only if every gate passes.",
]


def _issue(level: str, code: str, message: str, fix: str) -> dict[str, str]:
    """level: error (blocks the run) | warning (run allowed, expect a limited result)."""
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
        if not _is_file(path, root) or path.stat().st_size > MAX_CSV_BYTES:
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


def _sibling(jobs_root: Path, job_id: Any) -> Path | None:
    name = str(job_id or "")
    if not name or Path(name).name != name or name in {".", ".."}:
        return None
    path = jobs_root / name
    try:
        if path.is_symlink() or not path.is_dir() or jobs_root not in path.resolve().parents:
            return None
    except OSError:
        return None
    return path.resolve()


def validate_source(job_dir: Path, wells: list[str] | None = None) -> dict[str, Any]:
    """Check one Process D job folder and the jobs it points to. Pure file inspection."""
    job_dir = job_dir.resolve()
    jobs_root = job_dir.parent
    issues: list[dict[str, str]] = []
    result = _read_json(job_dir / "result.json", job_dir)
    summary = _read_json(job_dir / SOURCE_SUMMARY, job_dir)
    status = str(result.get("status") or summary.get("status") or "unknown")
    if not summary or str(summary.get("stage")) != "12_geobody":
        issues.append(_issue("error", "no_summary", f"'{SOURCE_SUMMARY}' was not found, or is not a Process D summary.",
                             "Choose a job produced by process_D_geobody.py that ran to the end."))
    if status.lower() not in {"succeeded", "success", "completed"}:
        issues.append(_issue("error", "not_succeeded", f"The Process D job status is '{status}'.",
                             "Wait for it to finish, or re-run Process D."))
    exports = summary.get("exports") if isinstance(summary.get("exports"), dict) else {}
    spatial = exports.get("spatial") if isinstance(exports.get("spatial"), dict) else {}
    if summary and (spatial.get("catalogue_reconciliation") or {}).get("all_counts_reconcile") is not True:
        issues.append(_issue("error", "exports_inconsistent", "The Process D exports do not agree with each other on voxel counts.",
                             "Re-run Process D. Its outputs must be consistent before prospects are cut from them."))
    for relative, role in GEOBODY_FILES:
        if summary and not _is_file(job_dir / relative, job_dir):
            issues.append(_issue("error", "missing_file", f"'{relative}' is missing from the Process D job ({role}).",
                                 "Restore the file or re-run Process D."))

    catalogue = _read_rows(job_dir / S12 / "tables" / "geobody_catalog.csv", job_dir)
    sizes = sorted((int(v) for r in catalogue if (v := _num(r.get("voxel_count"))) is not None), reverse=True)
    top_share = sizes[0] / sum(sizes) if sizes and sum(sizes) else None
    if top_share is not None and top_share >= DOMINANT_SHARE:
        issues.append(_issue(
            "warning", "one_dominant_body",
            f"One body holds {top_share * 100:.2f}% of all body voxels in the Process D job.",
            "Every prospect will be a survey-wide time slice of that background body, one per stretch of well path. The run will be "
            "long, write a great deal of data and yield no real prospect. Re-run Process D with a higher low threshold first."))
    elif summary and not sizes:
        issues.append(_issue("error", "no_bodies", "The Process D catalogue lists no bodies.",
                             "There is nothing to cut prospects from. Re-run Process D with settings that keep some bodies."))

    source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
    foundation = _sibling(jobs_root, source.get("foundation_job_id"))
    available: list[str] = []
    units_known = None
    if summary and foundation is None:
        issues.append(_issue("error", "foundation_missing", f"The Process A job '{source.get('foundation_job_id')}' was not found.",
                             "Well paths and logs come from that job. Restore it, or rebuild the chain from Process A."))
    elif foundation is not None:
        for relative, role in FOUNDATION_FILES:
            if not _is_file(foundation / relative, foundation):
                issues.append(_issue("error", "missing_file", f"'{relative}' is missing from the Process A job ({role}).",
                                     "Restore the file or re-run Process A."))
        rows = _read_rows(foundation / FOUNDATION_FILES[0][0], foundation)
        available = sorted({str(r.get("well")) for r in rows if r.get("well")})
        ties = {str(r.get("tie_status")) for r in rows if r.get("tie_status")}
        if ties and not any(t.startswith(("measured", "synthetic")) or t == "validated" for t in ties):
            issues.append(_issue("warning", "ties_not_validated", "No well in the Process A job has a validated seismic tie.",
                                 "Depths are likely to be flagged invalid, as they were in the reference run, and a prospect with invalid depth gets no volume."))
        crs = str(_read_json(foundation / "stages/00_inventory/data_manifest.json", foundation).get("crs") or "")
        if crs:
            units_known = not crs.upper().startswith(("UNDISCLOSED", "UNKNOWN", "UNCONFIRMED"))
            if not units_known:
                issues.append(_issue(
                    "warning", "map_units_unconfirmed",
                    "The coordinate system of the data is recorded as undisclosed, so map units are unconfirmed.",
                    "Gross rock volume will be blocked for every prospect, and hydrocarbon pore volume with it. Only the data owner "
                    "can unblock this by confirming the units."))
    requested = [w.strip() for w in (wells or []) if str(w).strip()]
    lookup = {w.lower(): w for w in available}
    unknown = [w for w in requested if w.lower() not in lookup]
    if requested and available and unknown:
        issues.append(_issue("error", "unknown_wells", f"These wells have no path in the Process A job: {', '.join(unknown)}.",
                             f"Choose from: {', '.join(available)}."))
    inference = _sibling(jobs_root, source.get("inference_job_id"))
    if summary and (inference is None or not _is_file(inference / "stages/09_inference/volumes/trace_index_grid.npy", inference)):
        issues.append(_issue("error", "trace_grid_missing", "The trace index grid of the Process B job was not found.",
                             "It links grid cells to map positions. Restore it or re-run Process B."))
    volumes_possible = bool(units_known) and not any(i["code"] == "ties_not_validated" for i in issues)
    return {
        "ready": not any(i["level"] == "error" for i in issues), "issues": issues,
        "source": {
            "status": status, "bodies_in_catalogue": len(sizes), "largest_body_share": round(top_share, 4) if top_share is not None else None,
            "foundation_job_id": source.get("foundation_job_id"), "inference_job_id": source.get("inference_job_id"),
            "volume_job_id": source.get("volume_job_id"),
            "wells_available": available, "anchor_wells": [lookup[w.lower()] for w in requested if w.lower() in lookup] or available,
        },
        "forecast": {
            "volumes_can_be_released": volumes_possible,
            "prospects_will_be_background_slices": bool(top_share is not None and top_share >= DOMINANT_SHARE),
            "note": "A forecast from the inputs. The run's own gates make the final decision per prospect.",
        },
    }


def execute(*, project_id: str, parameters: dict[str, Any], db: Any) -> dict[str, Any]:
    """Validate the Process D job that Process F will build on and return a plan for the user to confirm.

    parameters["source_geobody_job_id"]: the processing job ID of a completed Process D run.
    parameters["wells"] (optional): anchor wells, as a list of names or one comma-separated string. Default: all.
    """
    base = {
        "skill": SKILL_NAME, "project_id": project_id, "python_script": PYTHON_SCRIPT,
        "target_well": None, "dataset_ids": [], "submitted": False, "confirmation_required": True,
        "requirements": REQUIREMENTS, "what_will_happen": WHAT_HAPPENS, "reference_run": REFERENCE_RUN,
    }
    job_id = str(parameters.get("source_geobody_job_id") or "").strip()
    if not job_id:
        return {**base, "status": "needs_inputs", "issues": [],
                "summary": "No Process D job was named. Process F needs a completed geobody run to build on."}
    if Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("A valid Process D processing job ID is required.")
    wells = parameters.get("wells")
    if isinstance(wells, str):
        wells = wells.split(",")
    if wells is not None and not isinstance(wells, list):
        raise ValueError("wells must be a list of well names or a comma-separated string.")

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
        return {**base, "status": "not_ready", "source_geobody_job_id": job_id,
                "summary": f"Processing run {job_id} was not found.",
                "issues": [_issue("error", "job_not_found", f"Processing run {job_id} was not found.",
                                  "Check the job ID in the Ledger and try again.")]}
    report = validate_source(job_dir, [str(w) for w in wells] if wells else None)
    errors = sum(1 for i in report["issues"] if i["level"] == "error")
    source, forecast = report["source"], report["forecast"]
    summary = (f"{'Ready to run' if report['ready'] else 'Not ready'}: Process D job {job_id} is '{source['status']}' with "
               f"{source['bodies_in_catalogue']} bodies; {len(source['anchor_wells'])} anchor well(s); "
               f"{errors} blocking issue(s), {len(report['issues']) - errors} warning(s).")
    if report["ready"]:
        summary += (" Volumes can be released if the run's gates pass." if forecast["volumes_can_be_released"]
                    else " No volume will be released on these inputs.")
    return {
        **base, "status": "ready_for_confirmation" if report["ready"] else "not_ready", "summary": summary,
        "source_geobody_job_id": job_id, "source": source, "forecast": forecast, "issues": report["issues"],
        "expected_outputs": [
            "geogg_prospect_preflight.json and geogg_prospect_summary.json: status, gates and file list",
            "tables: prospect definitions, volumes by depth model, deterministic volumetrics, well properties, cutoff "
            "sensitivity and low, middle and high cases",
            "per-prospect voxel and point files, which can be very large",
        ],
        "next_action": (
            "Show this plan to the user, including the forecast and the cost of the reference run. After they confirm, submit the "
            "script through the approved processing service and report the returned job ID and status." if report["ready"] else
            "Explain each blocking issue and how to fix it. Do not submit."),
    }
