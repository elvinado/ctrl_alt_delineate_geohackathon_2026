"""Pre-flight validation for Process C (GeoGG volume products, stages 10-11).

Process C has no datasets of its own: it audits and packages the prediction cubes of a completed
Process B run. This handler checks that the named Process B job is fit to package and returns a
plan. It does NOT submit the job: submission must go through the approved backend processing
service after the user confirms. It never executes user-provided code and never opens a cube.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

SKILL_NAME = "process_c_volume_products"
PYTHON_SCRIPT = "process_C_volume_products.py"
SOURCE_OPERATION = "geogg_stage_f_08_09"
SOURCE_SUMMARY = "geogg_inference_summary.json"
EVAL = "stages/08_evaluation"
INFER = "stages/09_inference"
MAX_JSON_BYTES = 8_000_000
MAX_CSV_BYTES = 5_000_000

SUPPORT_FILES: list[tuple[str, str]] = [
    (f"{INFER}/volumes/prediction_valid.npy", "mask of where a prediction exists"),
    (f"{INFER}/volumes/volume_geometry.json", "grid definition of the cubes"),
    (f"{INFER}/volumes/trace_index_grid.npy", "link between grid cells and seismic traces"),
    (f"{EVAL}/tables/model_performance.csv", "held-out scores imported into the package"),
]

# Cost of the one reference run (three cubes, 1.45 million live traces, 876 samples per trace).
REFERENCE_RUN = {"note": "Observed in one reference run; yours will differ with survey size and hardware.", "run_time_minutes": 14}

REQUIREMENTS: list[dict[str, str]] = [
    {"input": "A completed Process B (inference) job", "need": "required",
     "why": "Supplies the prediction cubes, their grid and the held-out scores."},
    {"input": "All cube tiles processed and every cube file present", "need": "required",
     "why": "A partial cube would be exported as if it were complete."},
    {"input": "The full-stack SEG-Y used for the predictions", "need": "required, already linked",
     "why": "Its headers are copied so the exported cubes load with the same geometry."},
    {"input": "Free disk for one SEG-Y file per cube", "need": "required",
     "why": "Each file is about as large as the full-stack volume."},
]

WHAT_HAPPENS: list[str] = [
    "10 Audit and summary: verifies inputs and outputs by checksum, summarises the cubes, imports the held-out scores, and "
    "writes a dashboard, section figures and model cards.",
    "11 Export and validation set-up: writes each cube as SEG-Y, checks it bit for bit against the original, freezes the "
    "predictions and prepares a blind-well test.",
]


def _issue(level: str, code: str, message: str, fix: str) -> dict[str, str]:
    """level: error (blocks the run) | warning (run allowed, read the result with care)."""
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


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _no_skill(row: dict[str, str]) -> bool:
    """Same convention as the output-analysis skills: ROC-AUC below 0.55, or R2 at or below 0.02."""
    auc, r2 = _num(row.get("outer_roc_auc")), _num(row.get("outer_r2"))
    if auc is not None:
        return auc < 0.55
    return r2 is not None and r2 <= 0.02


def validate_source(job_dir: Path) -> dict[str, Any]:
    """Check one Process B job folder. Pure file inspection."""
    job_dir = job_dir.resolve()
    issues: list[dict[str, str]] = []
    result = _read_json(job_dir / "result.json", job_dir)
    summary = _read_json(job_dir / SOURCE_SUMMARY, job_dir)
    volume = _read_json(job_dir / INFER / "volume_inference_summary.json", job_dir)
    status = str(result.get("status") or summary.get("status") or "unknown")
    if not summary:
        issues.append(_issue("error", "no_summary", f"'{SOURCE_SUMMARY}' was not found in the job folder.",
                             "Choose a job produced by process_B_inference.py that ran to the end."))
    elif str(summary.get("operation")) != SOURCE_OPERATION:
        issues.append(_issue("error", "wrong_process", f"The job's operation is '{summary.get('operation')}', not '{SOURCE_OPERATION}'.",
                             "Choose a Process B (inference) job."))
    if status.lower() not in {"succeeded", "success", "completed"}:
        issues.append(_issue("error", "not_succeeded", f"The Process B job status is '{status}'.",
                             "Wait for it to finish, or re-run Process B. A failed or running job must not be packaged."))
    failed, blockers = [], 0
    for record in (s for s in summary.get("stages", []) if isinstance(s, dict)):
        checks = record.get("checks") if isinstance(record.get("checks"), dict) else {}
        failed += [f"{record.get('stage_id')}/{name}" for name, body in checks.items()
                   if isinstance(body, dict) and body.get("passed") is not True]
        blockers += len(record.get("blockers") or [])
    if failed or blockers:
        issues.append(_issue("error", "failed_checks",
                             f"Process B has {len(failed)} failed self-check(s) and {blockers} blocker(s).",
                             "Fix these in Process B first."))
    tiling = volume.get("tiling") if isinstance(volume.get("tiling"), dict) else {}
    done, total = tiling.get("completed_tiles"), tiling.get("tile_count")
    if summary and not volume:
        issues.append(_issue("error", "no_inference_summary", "The cube summary of stage 09 is missing.",
                             "Re-run Process B so that stage 09 completes."))
    elif total and done != total:
        issues.append(_issue("error", "cube_incomplete", f"Only {done} of {total} cube tiles were processed.",
                             "Resume Process B until every tile is done. A partial cube must not be exported."))

    outputs = volume.get("outputs") if isinstance(volume.get("outputs"), dict) else {}
    cubes = {str(task): f"{INFER}/{path}" for task, path in outputs.items()}
    for task, relative in cubes.items():
        if not _is_file(job_dir / relative, job_dir):
            issues.append(_issue("error", "missing_cube", f"The {task} cube file '{relative}' is missing.",
                                 "Restore the file or re-run Process B."))
    for relative, role in SUPPORT_FILES:
        if volume and not _is_file(job_dir / relative, job_dir):
            issues.append(_issue("error", "missing_file", f"'{relative}' is missing ({role}).",
                                 "Restore the file or re-run Process B."))

    model = volume.get("model") if isinstance(volume.get("model"), dict) else {}
    no_skill: list[str] = []
    table = job_dir / EVAL / "tables" / "model_performance.csv"
    try:
        if _is_file(table, job_dir) and table.stat().st_size <= MAX_CSV_BYTES:
            with table.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    if (row.get("task") in cubes and row.get("input") == model.get("input")
                            and row.get("model") == model.get("model") and _no_skill(row)):
                        no_skill.append(str(row.get("task")))
    except (OSError, UnicodeDecodeError, csv.Error):
        pass
    if no_skill:
        issues.append(_issue("warning", "cube_without_skill",
                             f"The model that made these cubes showed no held-out skill for: {', '.join(no_skill)}.",
                             "Process C will still export them as verified SEG-Y files, which can make them look official. "
                             "Decide whether to label or withhold them."))

    region = volume.get("region") if isinstance(volume.get("region"), dict) else {}
    shape = volume.get("volume_shape") or []
    live, samples = _num(region.get("live_traces")), (_num(shape[2]) if len(shape) == 3 else None)
    # SEG-Y size: 3600-byte file header plus, per trace, a 240-byte header and 4 bytes per sample.
    per_cube_gb = round((3600 + live * (240 + 4 * samples)) / 1e9, 2) if live and samples else None
    return {
        "ready": not any(i["level"] == "error" for i in issues), "issues": issues,
        "source": {
            "status": status, "operation": summary.get("operation"), "model_release_id": summary.get("model_release_id"),
            "foundation_job_id": volume.get("foundation_job_id"), "cube_model": model.get("model"),
            "cubes": sorted(cubes), "tiles": [done, total],
            "expected_segy_gb_per_cube": per_cube_gb,
            "expected_segy_gb_total": round(per_cube_gb * len(cubes), 1) if per_cube_gb else None,
        },
    }


def execute(*, project_id: str, parameters: dict[str, Any], db: Any) -> dict[str, Any]:
    """Validate the Process B job that Process C will package and return a plan for the user to confirm.

    parameters["source_inference_job_id"]: the processing job ID of a completed Process B run.
    The process takes no datasets and no numeric parameters.
    """
    base = {
        "skill": SKILL_NAME, "project_id": project_id, "python_script": PYTHON_SCRIPT,
        "target_well": None, "dataset_ids": [], "submitted": False, "confirmation_required": True,
        "requirements": REQUIREMENTS, "what_will_happen": WHAT_HAPPENS, "reference_run": REFERENCE_RUN,
    }
    job_id = str(parameters.get("source_inference_job_id") or "").strip()
    if not job_id:
        return {**base, "status": "needs_inputs", "issues": [],
                "summary": "No Process B job was named. Process C needs a completed inference run to package."}
    if Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("A valid Process B processing job ID is required.")

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
        return {**base, "status": "not_ready", "source_inference_job_id": job_id,
                "summary": f"Processing run {job_id} was not found.",
                "issues": [_issue("error", "job_not_found", f"Processing run {job_id} was not found.",
                                  "Check the job ID in the Ledger and try again.")]}
    report = validate_source(job_dir)
    errors = sum(1 for i in report["issues"] if i["level"] == "error")
    source = report["source"]
    summary = (
        f"{'Ready to run' if report['ready'] else 'Not ready'}: Process B job {job_id} is '{source['status']}' with "
        f"{len(source['cubes'])} cube(s); {errors} blocking issue(s), {len(report['issues']) - errors} warning(s).")
    if report["ready"] and source["expected_segy_gb_total"]:
        summary += f" The export will write about {source['expected_segy_gb_total']} GB of SEG-Y."
    return {
        **base, "status": "ready_for_confirmation" if report["ready"] else "not_ready", "summary": summary,
        "source_inference_job_id": job_id, "source": source, "issues": report["issues"],
        "expected_outputs": [
            "geogg_volume_products_summary.json: status of the audit and of each exported file",
            "stages/10_summary: audit tables, cube statistics, dashboard, section figures and model cards",
            "stages/11_volume_export: one SEG-Y file per cube with its checks, plus the blind-well protocol",
        ],
        "next_action": (
            "Show this plan to the user, including the disk space needed. After they confirm, submit the script through the "
            "approved processing service and report the returned job ID and status." if report["ready"] else
            "Explain each blocking issue and how to fix it. Do not submit."),
    }
