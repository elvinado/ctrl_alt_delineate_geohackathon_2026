"""Pre-flight validation for Process B (GeoGG evaluation and inference, stages 08-09).

Process B has no datasets of its own: it consumes the model release and sealed predictions of a
completed Process A run. This handler checks that the named Process A job is fit to build on and
returns a plan. It does NOT submit the job: submission must go through the approved backend
processing service after the user confirms. It never executes user-provided code.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SKILL_NAME = "process_b_inference"
PYTHON_SCRIPT = "process_B_inference.py"
OPERATION = "geogg_stage_f_08_09"
FOUNDATION_OPERATION = "geogg_fresh_stages_00_07"
FOUNDATION_SUMMARY = "geogg_foundations_summary.json"
FOUNDATION_STAGES = ("00_inventory", "01_well_ties", "02_labels", "03_extraction", "04_features",
                     "05_training_tables", "06_feature_selection", "07_models")
MAX_JSON_BYTES = 8_000_000

# Files of the Process A job that Process B is known to depend on.
REQUIRED_FILES: list[tuple[str, str]] = [
    ("stages/07_models/tables/outer_test_predictions.csv", "sealed held-out predictions, scored by stage 08"),
    ("stages/07_models/tables/model_registry.csv", "list of trained candidates and their practice scores"),
    ("stages/07_models/models/final/cnn3d_multitask.pt", "the deep model used to build the prediction cubes"),
    ("stages/03_extraction/patches/patch_manifest_v1.json", "definition of the seismic cut-out the model expects"),
]

# Cost of the one reference run (13 wells, 1.45 million live traces, 876 samples per trace, one GPU).
REFERENCE_RUN = {
    "note": "Observed in one reference run on one GPU; your run will differ with survey size and hardware.",
    "run_time_hours": 9.3, "prediction_cubes": 3, "disk_written_gb": 25,
    "live_traces": 1_453_018, "samples_per_trace": 876,
}

REQUIREMENTS: list[dict[str, str]] = [
    {"input": "A completed Process A (foundations) job", "need": "required",
     "why": "Supplies the trained models, the sealed held-out predictions and the seismic cut-out definition."},
    {"input": "Its model release, status 'ready'", "need": "required",
     "why": "The packaged models are what Process B loads to build the prediction cubes."},
    {"input": "The same registered seismic volumes used by Process A", "need": "required, already linked",
     "why": "The cubes are predicted from the full, near, mid, far and ultra-far stacks on the same grid."},
    {"input": "A GPU node and roughly 25 GB of free disk", "need": "recommended",
     "why": "The whole-survey prediction is the slow part and writes several multi-gigabyte cubes."},
]

WHAT_HAPPENS: list[str] = [
    "08 Final exam on unseen wells: scores every Process A model on wells it never trained on, with intervals, "
    "paired comparisons, calibration and prediction bands. This stage is quick.",
    "09 Prediction cubes for the whole survey: runs the deep model over every live trace, tile by tile, and writes one cube "
    "each for sand versus shale, shale volume and porosity. This stage takes hours and can resume if interrupted.",
]


def _issue(level: str, code: str, message: str, fix: str) -> dict[str, str]:
    """level: error (blocks the run) | warning (run allowed, quality reduced)."""
    return {"level": level, "code": code, "message": message, "how_to_fix": fix}


def _read_json(path: Path, root: Path) -> dict[str, Any]:
    try:
        if path.is_symlink() or not path.is_file() or root not in path.resolve().parents:
            return {}
        if path.stat().st_size > MAX_JSON_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_file(path: Path, root: Path) -> bool:
    try:
        return path.is_file() and not path.is_symlink() and root in path.resolve().parents
    except OSError:
        return False


def validate_foundation(job_dir: Path, storage_root: Path | None = None) -> dict[str, Any]:
    """Check one Process A job folder. Pure file inspection."""
    job_dir = job_dir.resolve()
    issues: list[dict[str, str]] = []
    result = _read_json(job_dir / "result.json", job_dir)
    summary = _read_json(job_dir / FOUNDATION_SUMMARY, job_dir)
    status = str(result.get("status") or summary.get("status") or "unknown")
    operation = str(summary.get("operation") or "")
    if not summary:
        issues.append(_issue("error", "no_summary", f"'{FOUNDATION_SUMMARY}' was not found in the job folder.",
                             "Choose a job produced by process_A_foundations.py that ran to the end."))
    elif operation != FOUNDATION_OPERATION:
        issues.append(_issue("error", "wrong_process", f"The job's operation is '{operation}', not '{FOUNDATION_OPERATION}'.",
                             "Choose a Process A (foundations) job."))
    if status.lower() not in {"succeeded", "success", "completed"}:
        issues.append(_issue("error", "not_succeeded", f"The Process A job status is '{status}'.",
                             "Wait for it to finish, or re-run Process A. Process B must not build on a failed or running job."))

    stages = {str(s.get("stage_id")): s for s in summary.get("stages", []) if isinstance(s, dict)}
    missing = [sid for sid in FOUNDATION_STAGES if sid not in stages]
    if summary and missing:
        issues.append(_issue("error", "missing_stages", f"Stages without a record: {', '.join(missing)}.",
                             "Re-run Process A so that all eight stages complete."))
    failed, blockers, limitations = [], 0, 0
    for sid, record in stages.items():
        checks = record.get("checks") if isinstance(record.get("checks"), dict) else {}
        failed += [f"{sid}/{name}" for name, body in checks.items() if isinstance(body, dict) and body.get("passed") is not True]
        blockers += len(record.get("blockers") or [])
        limitations += len(record.get("limitations") or [])
    if failed or blockers:
        issues.append(_issue("error", "failed_checks",
                             f"Process A has {len(failed)} failed self-check(s) and {blockers} blocker(s)"
                             + (f": {', '.join(failed[:5])}." if failed else "."),
                             "Fix these in Process A first. Evaluating or deploying models from a run with failed checks is unsafe."))

    release = summary.get("model_release") if isinstance(summary.get("model_release"), dict) else {}
    release_id = str(release.get("model_release_id") or "")
    if summary and str(release.get("status") or "").lower() != "ready":
        issues.append(_issue("error", "release_not_ready",
                             f"The model release status is '{release.get('status') or 'missing'}'.",
                             "Re-run Process A; it packages the release at the end of stage 07."))
    release_manifest_found = None
    if storage_root is not None and release_id and Path(release_id).name == release_id:
        releases = (storage_root / "model_releases").resolve()
        release_manifest_found = _is_file(storage_root / "model_releases" / release_id / "manifest.json", releases)
        if not release_manifest_found:
            issues.append(_issue("warning", "release_manifest_not_found",
                                 f"No manifest was found for release '{release_id}' under the storage root.",
                                 "Confirm the release folder still exists; Process B loads its models from there."))
    for relative, role in REQUIRED_FILES:
        if summary and not _is_file(job_dir / relative, job_dir):
            issues.append(_issue("error", "missing_file", f"'{relative}' is missing ({role}).",
                                 "Re-run Process A, or restore the file from the original job folder."))

    ties = (stages.get("01_well_ties") or {}).get("metrics") or {}
    n_valid, n_wells = ties.get("n_wells_validated"), ties.get("n_wells_processed")
    if n_valid == 0 and n_wells:
        issues.append(_issue("warning", "ties_not_validated",
                             f"None of the {n_wells} wells in the Process A job has a validated seismic tie.",
                             "Process B will run, but every score and cube stays exploratory until measured checkshots or synthetic ties exist."))
    if limitations:
        issues.append(_issue("warning", "inherited_limitations",
                             f"The Process A job states {limitations} limitation(s) about its inputs.",
                             "Process B inherits all of them without restating them. Read the Process A analysis before trusting the results."))
    return {
        "ready": not any(i["level"] == "error" for i in issues), "issues": issues,
        "foundation": {
            "status": status, "operation": operation or None, "stages_recorded": len(stages),
            "model_release_id": release_id or None, "model_release_status": release.get("status"),
            "model_count": release.get("model_count"), "release_manifest_found": release_manifest_found,
            "wells": n_wells, "limitations": limitations,
        },
    }


def execute(*, project_id: str, parameters: dict[str, Any], db: Any) -> dict[str, Any]:
    """Validate the Process A job that Process B will build on and return a plan for the user to confirm.

    parameters["foundation_job_id"]: the processing job ID of a completed Process A run.
    The process takes no datasets and no numeric parameters.
    """
    base = {
        "skill": SKILL_NAME, "project_id": project_id, "python_script": PYTHON_SCRIPT, "operation": OPERATION,
        "target_well": None, "dataset_ids": [], "submitted": False, "confirmation_required": True,
        "requirements": REQUIREMENTS, "what_will_happen": WHAT_HAPPENS, "reference_run": REFERENCE_RUN,
    }
    job_id = str(parameters.get("foundation_job_id") or "").strip()
    if not job_id:
        return {**base, "status": "needs_inputs", "issues": [],
                "summary": "No Process A job was named. Process B needs a completed foundations run to build on."}
    if Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("A valid Process A processing job ID is required.")

    from app.storage import configured_storage_roots

    job_dir, storage_root = None, None
    for root in configured_storage_roots(db):
        jobs_root = (root / "processing_jobs").resolve()
        candidate_path = root / "processing_jobs" / job_id
        if candidate_path.is_symlink():
            continue
        candidate = candidate_path.resolve()
        if jobs_root in candidate.parents and candidate.is_dir():
            job_dir, storage_root = candidate, root
            break
    if job_dir is None:
        return {**base, "status": "not_ready", "foundation_job_id": job_id,
                "summary": f"Processing run {job_id} was not found.",
                "issues": [_issue("error", "job_not_found", f"Processing run {job_id} was not found.",
                                  "Check the job ID in the Ledger and try again.")]}
    report = validate_foundation(job_dir, storage_root)
    errors = sum(1 for i in report["issues"] if i["level"] == "error")
    warnings = len(report["issues"]) - errors
    foundation = report["foundation"]
    summary = (
        f"{'Ready to run' if report['ready'] else 'Not ready'}: Process A job {job_id} is '{foundation['status']}' with "
        f"{foundation['stages_recorded']} of {len(FOUNDATION_STAGES)} stages and a model release that is "
        f"'{foundation['model_release_status'] or 'missing'}'; {errors} blocking issue(s), {warnings} warning(s).")
    return {
        **base, "status": "ready_for_confirmation" if report["ready"] else "not_ready", "summary": summary,
        "foundation_job_id": job_id, "foundation": foundation, "issues": report["issues"],
        "expected_outputs": [
            "geogg_inference_summary.json: status and self-checks of both stages",
            "stages/08_evaluation: held-out performance tables and figures for every candidate model",
            "stages/09_inference/volumes: one prediction cube per property, a validity mask and the cube geometry",
        ],
        "next_action": (
            "Show this plan to the user, including the expected run time. After they confirm, submit the script through the "
            "approved processing service and report the returned job ID and status." if report["ready"] else
            "Explain each blocking issue and how to fix it. Do not submit."),
    }
