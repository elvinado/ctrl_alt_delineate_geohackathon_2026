"""Pre-flight validation for Process E (GeoGG well-geobody analysis, stage 13).

Process E has no datasets of its own: it lays the well paths of Process A over the geobodies of
a completed Process D run. This handler checks that the named Process D job is fit to build on,
forecasts whether the result can be informative, and returns a plan. It does NOT submit the job:
submission must go through the approved backend processing service after the user confirms. It
never executes user-provided code and never opens a volume.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

SKILL_NAME = "process_e_well_geobody_analysis"
PYTHON_SCRIPT = "process_E_well_geobody_analysis.py"
SOURCE_SUMMARY = "geogg_geobody_summary.json"
S12 = "stages/12_geobody"
TRAJECTORIES = "stages/01_well_ties/tables/tz_samples_all_wells.csv"
MAX_JSON_BYTES = 8_000_000
MAX_CSV_BYTES = 20_000_000
DOMINANT_SHARE = 0.90

REQUIRED_FILES: list[tuple[str, str]] = [
    (f"{S12}/volumes/geobody_labels.npy", "the volume that says which body each voxel belongs to"),
    (f"{S12}/tables/geobody_catalog.csv", "the list of bodies"),
    (f"{S12}/tables/geobody_run_length_index.csv", "the per-trace index of body extents"),
    (f"{S12}/tables/geobody_spatial_export_reconciliation.csv", "the check that the exports agree with each other"),
]

# Cost of the one reference run (13 wells against one survey-wide body).
REFERENCE_RUN = {"note": "Observed in one reference run; yours will differ with the number of wells and the size of the bodies.",
                 "run_time_minutes": 74}

REQUIREMENTS: list[dict[str, str]] = [
    {"input": "A completed Process D (geobody) job", "need": "required",
     "why": "Supplies the body label volume, the catalogue and the per-trace index."},
    {"input": "The Process A job it descends from, with its well paths", "need": "required, linked automatically",
     "why": "Well positions in seismic time come from its depth-to-time conversion."},
    {"input": "A list of wells", "need": "optional",
     "why": "Without one, every well with a complete path is analysed."},
    {"input": "Porosity and water-saturation cubes", "need": "optional",
     "why": "Needed only for the property statistics. If one is missing that part is skipped."},
]

WHAT_HAPPENS: list[str] = [
    "Preflight: confirms the source jobs link up and the well paths, label volume and catalogue match their checksums.",
    "Intersections: for each well, records which path samples lie inside a body and groups them into continuous stretches.",
    "Selected body: for each well, takes the largest body it meets and tries to give it a depth range and predicted "
    "porosity and water-saturation statistics.",
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
    reconciliation = ((exports.get("spatial") or {}).get("catalogue_reconciliation") or {}) if isinstance(exports.get("spatial"), dict) else {}
    if summary and reconciliation.get("all_counts_reconcile") is not True:
        issues.append(_issue("error", "exports_inconsistent", "The Process D exports do not agree with each other on voxel counts.",
                             "Re-run Process D. Its outputs must be consistent before wells are laid over them."))
    for relative, role in REQUIRED_FILES:
        if summary and not _is_file(job_dir / relative, job_dir):
            issues.append(_issue("error", "missing_file", f"'{relative}' is missing ({role}).",
                                 "Restore the file or re-run Process D."))

    catalogue = _read_rows(job_dir / S12 / "tables" / "geobody_catalog.csv", job_dir)
    sizes = sorted((int(v) for r in catalogue if (v := _num(r.get("voxel_count"))) is not None), reverse=True)
    top_share = sizes[0] / sum(sizes) if sizes and sum(sizes) else None
    if top_share is not None and top_share >= DOMINANT_SHARE:
        issues.append(_issue(
            "warning", "one_dominant_body",
            f"One body holds {top_share * 100:.2f}% of all body voxels in the Process D job.",
            "Every well will meet that same background body, so the result will not tell the wells apart. Re-run Process D with a "
            "higher low threshold first, unless you only want to test the workflow."))
    elif summary and not sizes:
        issues.append(_issue("error", "no_bodies", "The Process D catalogue lists no bodies.",
                             "There is nothing to intersect. Re-run Process D with settings that keep some bodies."))

    source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
    foundation = _sibling(jobs_root, source.get("foundation_job_id"))
    available: list[str] = []
    if summary and foundation is None:
        issues.append(_issue("error", "foundation_missing", f"The Process A job '{source.get('foundation_job_id')}' was not found.",
                             "Well paths come from that job. Restore it, or rebuild the chain from Process A."))
    elif foundation is not None:
        rows = _read_rows(foundation / TRAJECTORIES, foundation)
        available = sorted({str(r.get("well")) for r in rows if r.get("well")})
        if not available:
            issues.append(_issue("error", "no_trajectories", f"No well paths were found in the Process A job ('{TRAJECTORIES}').",
                                 "Re-run Process A so that the depth-to-time conversion stage completes."))
        ties = {str(r.get("tie_status")) for r in rows if r.get("tie_status")}
        if ties and not any(t.startswith(("measured", "synthetic")) or t == "validated" for t in ties):
            issues.append(_issue("warning", "ties_not_validated", "No well in the Process A job has a validated seismic tie.",
                                 "The run will work, but each well's position in the cube is an estimate. Thin bodies can be missed or hit by mistake."))
    requested = [w.strip() for w in (wells or []) if str(w).strip()]
    lookup = {w.lower(): w for w in available}
    unknown = [w for w in requested if w.lower() not in lookup]
    if requested and available and unknown:
        issues.append(_issue("error", "unknown_wells", f"These wells have no path in the Process A job: {', '.join(unknown)}.",
                             f"Choose from: {', '.join(available)}."))

    volume = _sibling(jobs_root, source.get("volume_job_id"))
    registered: set[str] = set()
    if volume is not None:
        manifest = _read_json(volume / "stages/11_volume_export/native_volume_export_manifest.json", volume)
        registered = {str(p.get("product")) for p in manifest.get("products") or [] if isinstance(p, dict)}
        missing = [n.replace("prediction_", "").upper() for n in ("prediction_phit", "prediction_swt") if n not in registered]
        if registered and missing:
            issues.append(_issue("warning", "property_cube_missing",
                                 f"No {' or '.join(missing)} cube is registered in the Process C job.",
                                 "Porosity and water-saturation statistics will be skipped. Everything else will run."))
    return {
        "ready": not any(i["level"] == "error" for i in issues), "issues": issues,
        "source": {
            "status": status, "bodies_in_catalogue": len(sizes), "largest_body_share": round(top_share, 4) if top_share is not None else None,
            "foundation_job_id": source.get("foundation_job_id"), "inference_job_id": source.get("inference_job_id"),
            "volume_job_id": source.get("volume_job_id"),
            "wells_available": available, "wells_to_analyse": [lookup[w.lower()] for w in requested if w.lower() in lookup] or available,
            "property_cubes_registered": sorted(n for n in registered if n in {"prediction_phit", "prediction_swt"}),
        },
    }


def execute(*, project_id: str, parameters: dict[str, Any], db: Any) -> dict[str, Any]:
    """Validate the Process D job that Process E will analyse and return a plan for the user to confirm.

    parameters["source_geobody_job_id"]: the processing job ID of a completed Process D run.
    parameters["wells"] (optional): a list of well names, or one comma-separated string. Default: all wells.
    """
    base = {
        "skill": SKILL_NAME, "project_id": project_id, "python_script": PYTHON_SCRIPT,
        "target_well": None, "dataset_ids": [], "submitted": False, "confirmation_required": True,
        "requirements": REQUIREMENTS, "what_will_happen": WHAT_HAPPENS, "reference_run": REFERENCE_RUN,
    }
    job_id = str(parameters.get("source_geobody_job_id") or "").strip()
    if not job_id:
        return {**base, "status": "needs_inputs", "issues": [],
                "summary": "No Process D job was named. Process E needs a completed geobody run to analyse."}
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
    source = report["source"]
    summary = (f"{'Ready to run' if report['ready'] else 'Not ready'}: Process D job {job_id} is '{source['status']}' with "
               f"{source['bodies_in_catalogue']} bodies; {len(source['wells_to_analyse'])} well(s) to analyse; "
               f"{errors} blocking issue(s), {len(report['issues']) - errors} warning(s).")
    return {
        **base, "status": "ready_for_confirmation" if report["ready"] else "not_ready", "summary": summary,
        "source_geobody_job_id": job_id, "source": source, "issues": report["issues"],
        "expected_outputs": [
            "geogg_well_geobody_analysis_preflight.json and ..._summary.json: checks, counts and per-well results",
            "stages/13_well_geobody_analysis/tables: path samples, in-body stretches, well-to-body intersections, the selected "
            "body per well, time-depth model comparison and property statistics",
        ],
        "next_action": (
            "Show this plan to the user, including any warning that the result will be uninformative. After they confirm, submit "
            "the script through the approved processing service and report the returned job ID and status." if report["ready"] else
            "Explain each blocking issue and how to fix it. Do not submit."),
    }
