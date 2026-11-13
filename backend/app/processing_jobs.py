"""Submit Python processing scripts locally or through SLURM, and track their status."""
import csv
import io
import json
import math
import statistics
import os
import re
import shutil
import subprocess
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy.orm import Session

from app.storage import ensure_storage_root, configured_storage_roots
from app.database import get_db
from app.models import Dataset, ProcessingJob, ProcessingJobInput
from app.storage import resolve_dataset_path
from app.runtime_settings import get_effective_settings

router = APIRouter()
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SLURM_JOB_ID = re.compile(r"Submitted batch job (\d+)")
RUNNERS = {"pyrunner.local.sh": "local", "pyrunner.sbatch.sh": "slurm"}
TERMINAL_STATUSES = {"submission_failed", "succeeded", "failed", "cancelled"}
SLURM_STATUSES = {
    "COMPLETED": "succeeded",
    "FAILED": "failed",
    "CANCELLED": "cancelled",
    "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed",
    "NODE_FAIL": "failed",
    "PREEMPTED": "failed",
    "BOOT_FAIL": "failed",
}


class EvidenceCompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    python_script: str = Field(..., min_length=1, max_length=255)
    job_ids: list[str] = Field(..., min_length=2, max_length=8)


class ProcessingJobCreate(BaseModel):
    """Choose a Python worker and one of the supported runners."""

    model_config = ConfigDict(extra="forbid")

    python_script: str = Field(..., examples=["gpu_test.py"])
    shell_script: str | None = Field(
        default=None, examples=["pyrunner.local.sh", "pyrunner.sbatch.sh"],
        description="Optional legacy override. Normally runners come from runtime settings.",
    )
    dataset_ids: list[str] = Field(default_factory=list, description="Dataset UUIDs to pass to the worker via inputs.json")
    target_well: str | None = Field(default=None, min_length=1, max_length=64, description="Optional target well passed to compatible workers via APGCE_TARGET_WELL")


def _response(job: ProcessingJob) -> dict:
    """Build the common API response."""
    return {
        "id": job.id,
        "operation": job.operation,
        "status": job.status,
        "slurm_job_id": job.slurm_job_id,
        "created_at": job.created_at,
        "log_path": job.log_path,
        "result_path": job.result_path,
        "error_message": job.error_message,
    }


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _refresh_status(job: ProcessingJob, db: Session) -> None:
    """Update status from the result file, local process, or SLURM."""
    result_path = Path(job.result_path)
    if result_path.is_file():
        result = _read_json(result_path)
        if result:
            job.status = result.get("status", "succeeded")
            job.error_message = result.get("error_message")
            db.commit()
        return

    metadata = _read_json(Path(job.job_dir) / "job.json")
    if metadata.get("execution_mode") == "local":
        if job.status in TERMINAL_STATUSES:
            return

        pid = metadata.get("process_id")
        if not isinstance(pid, int):
            return
        try:
            stat = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
            state = stat.rsplit(")", 1)[1].split()[0]
            if state == "Z":
                job.status = "failed"
                job.error_message = "Local process exited without writing result.json."
            else:
                job.status = "running"
            db.commit()
        except FileNotFoundError:
            job.status = "failed"
            job.error_message = "Local process exited without writing result.json."
            db.commit()
        except (OSError, IndexError):
            pass
        return

    if not job.slurm_job_id or job.status in TERMINAL_STATUSES:
        return
    if not shutil.which("squeue"):
        return

    try:
        queue = subprocess.run(
            ["squeue", "-h", "-j", job.slurm_job_id, "-o", "%T"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        states = queue.stdout.strip().splitlines()
        if states:
            job.status = "running" if states[0].upper() == "RUNNING" else "queued"
            db.commit()
            return

        # Finished jobs leave squeue; sacct can report their final state.
        if shutil.which("sacct"):
            accounting = subprocess.run(
                ["sacct", "-n", "-X", "-j", job.slurm_job_id, "--format=State"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            states = accounting.stdout.strip().splitlines()
            if states:
                state = states[0].split()[0].upper().split("+")[0]
                if state in SLURM_STATUSES:
                    job.status = SLURM_STATUSES[state]
                    if job.status == "succeeded" and not result_path.exists():
                        job.status = "failed"
                        job.error_message = (
                            "SLURM completed but the worker did not write result.json."
                        )
                    db.commit()
    except (OSError, subprocess.TimeoutExpired):
        pass


def _resolve_runner_candidates(request: ProcessingJobCreate, db: Session) -> list[tuple[str, str, Path, Path]]:
    """Resolve primary and fallback runner candidates from runtime settings."""
    settings = get_effective_settings(db).processing
    names = [request.shell_script or settings.default_runner, settings.fallback_runner]
    candidates: list[tuple[str, str, Path, Path]] = []
    scripts_dir = (ensure_storage_root(db) / "scripts").resolve()
    worker_name = request.python_script
    if (
        not worker_name.endswith(".py")
        or Path(worker_name).name != worker_name
        or worker_name in {".", ".."}
    ):
        raise HTTPException(status_code=422, detail="python_script must be a Python filename located directly in scripts/.")
    worker = (scripts_dir / worker_name).resolve()
    if scripts_dir not in worker.parents or not worker.is_file():
        raise HTTPException(status_code=404, detail=f"Python script not found: {worker_name}")

    errors = []
    for name in dict.fromkeys(names):
        if Path(name).name != name or not name.endswith(".sh"):
            errors.append(f"{name}: invalid runner filename")
            continue
        runner = (scripts_dir / name).resolve()
        if scripts_dir not in runner.parents or not runner.is_file():
            errors.append(f"{name}: runner file not found in scripts/")
            continue
        mode = RUNNERS.get(name, "slurm" if "sbatch" in name.lower() else "local")
        if mode == "slurm" and not shutil.which("sbatch"):
            errors.append(f"{name}: SLURM sbatch command is unavailable")
            continue
        candidates.append((name, mode, worker, runner))

    if not candidates:
        raise HTTPException(status_code=503, detail={
            "message": "Neither the primary nor fallback processing runner is available.",
            "errors": errors,
        })
    return candidates


def _prepare_job_files(
    db: Session, request: ProcessingJobCreate, mode: str, worker: Path, runner: Path
) -> tuple[str, Path, Path, Path, dict[str, str | int | None], list[Dataset], Path | None]:
    """Create the job directory and write its initial metadata."""
    # Resolve all inputs before creating or launching a job. Dataset IDs are UUID strings.
    if len(request.dataset_ids) != len(set(request.dataset_ids)):
        raise HTTPException(status_code=422, detail="dataset_ids must not contain duplicates")
    datasets: list[Dataset] = []
    manifest_inputs = []
    for dataset_id in request.dataset_ids:
        dataset = db.get(Dataset, dataset_id)
        if dataset is None:
            raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
        path = resolve_dataset_path(dataset, db)
        datasets.append(dataset)
        # Carry existing relational identities into the worker manifest. The
        # scientific runner consumes these references; it never creates its own IDs.
        well = getattr(dataset, "well", None)
        survey = getattr(dataset, "survey", None)
        try:
            dataset_metadata = json.loads(dataset.metadata_json or "{}")
        except (TypeError, json.JSONDecodeError):
            dataset_metadata = {}
        try:
            well_metadata = json.loads(well.metadata_json or "{}") if well else {}
        except (TypeError, json.JSONDecodeError):
            well_metadata = {}
        manifest_inputs.append({
            "dataset_id": dataset.id,
            "kind": dataset.kind,
            "format": dataset.format,
            "name": dataset.name,
            # The API resolves the authoritative dataset record before launch.
            # Keep the worker path plus storage provenance for audit/debugging.
            "path": str(path),
            "storage_mode": getattr(dataset, "storage_mode", "managed"),
            "storage_root": dataset.storage_root,
            "storage_key": dataset.storage_key,
            "well_id": getattr(dataset, "well_id", None),
            "well_name": getattr(well, "name", None),
            "survey_id": getattr(dataset, "survey_id", None),
            "survey_name": getattr(survey, "name", None),
            "well_x": getattr(well, "x", None),
            "well_y": getattr(well, "y", None),
            "well_crs": getattr(well, "crs", None),
            "well_metadata": well_metadata if isinstance(well_metadata, dict) else {},
            "metadata": dataset_metadata if isinstance(dataset_metadata, dict) else {},
        })

    job_id = str(uuid.uuid4())
    job_dir = ensure_storage_root(db) / "processing_jobs" / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    result_path = job_dir / "result.json"
    manifest_path: Path | None = None
    if manifest_inputs:
        manifest_path = job_dir / "inputs.json"
        manifest_path.write_text(
            json.dumps({"inputs": manifest_inputs}, indent=2), encoding="utf-8"
        )
    log_path = job_dir / ("local.log" if mode == "local" else "slurm-%j.log")
    metadata_path = job_dir / "job.json"
    metadata: dict[str, str | int | None] = {
        "job_id": job_id,
        "python_script": request.python_script,
        "shell_script": request.shell_script,
        "python_script_path": str(worker),
        "shell_script_path": str(runner),
        "execution_mode": mode,
        "dataset_ids": list(request.dataset_ids),
        "target_well": request.target_well,
        "input_manifest_path": str(manifest_path) if manifest_path else None,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return job_id, job_dir, result_path, log_path, metadata, datasets, manifest_path


def _create_processing_job(
    db: Session, job_id: str, worker: Path, job_dir: Path, log_path: Path, result_path: Path
) -> ProcessingJob:
    """Persist the job record before starting the worker."""
    job = ProcessingJob(
        id=job_id,
        operation=worker.stem,
        status="submitted",
        job_dir=str(job_dir),
        log_path=str(log_path),
        result_path=str(result_path),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def _runner_environment(worker: Path, result_path: Path, manifest_path: Path | None, target_well: str | None = None) -> dict[str, str]:
    """Build the environment passed to either runner script."""
    env = os.environ.copy()
    env["APGCE_PYTHON_SCRIPT"] = str(worker)
    env["APGCE_RESULT_PATH"] = str(result_path)
    # ProcessingJob.id is the single authoritative run identity for workers.
    env["APGCE_JOB_ID"] = result_path.parent.name
    env.pop("APGCE_INPUT_MANIFEST", None)
    if manifest_path is not None:
        env["APGCE_INPUT_MANIFEST"] = str(manifest_path)
    if target_well:
        env["APGCE_TARGET_WELL"] = target_well
    else:
        env.pop("APGCE_TARGET_WELL", None)
    return env


def _start_local_worker(
    runner: Path, worker: Path, result_path: Path, log_path: Path,
    job_dir: Path, metadata: dict[str, str | int | None], manifest_path: Path | None, target_well: str | None = None,
) -> None:
    """Launch the local runner and save its process ID for status polling."""
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["bash", str(runner)],
            cwd=str(worker.parent.parent),
            env=_runner_environment(worker, result_path, manifest_path, target_well),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )

    metadata["process_id"] = process.pid
    (job_dir / "job.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _submit_slurm_worker(
    runner: Path, worker: Path, result_path: Path, job_dir: Path, manifest_path: Path | None, target_well: str | None = None
) -> str:
    """Submit the runner to SLURM and return its job ID."""
    log_path = job_dir / "slurm-%j.log"
    export_vars = (
        f"ALL,APGCE_PYTHON_SCRIPT={worker},APGCE_RESULT_PATH={result_path},"
        f"APGCE_JOB_ID={result_path.parent.name}"
    )
    if manifest_path is not None:
        export_vars += f",APGCE_INPUT_MANIFEST={manifest_path}"
    if target_well:
        export_vars += f",APGCE_TARGET_WELL={target_well}"
    submission = subprocess.run(
        [
            "sbatch",
            f"--output={log_path}",
            f"--error={log_path}",
            f"--export={export_vars}",
            str(runner),
        ],
        cwd=str(worker.parent.parent),
        env=_runner_environment(worker, result_path, manifest_path, target_well),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    match = SLURM_JOB_ID.fullmatch(submission.stdout.strip())
    if submission.returncode != 0 or not match:
        error = (
            submission.stderr or submission.stdout or "sbatch submission failed"
        ).strip()[:2000]
        raise HTTPException(
            status_code=502,
            detail={"message": "SLURM submission failed", "error": error},
        )
    return match.group(1)


def _mark_submission_failed(db: Session, job: ProcessingJob, message: str) -> None:
    """Persist a failed submission and its diagnostic message."""
    job.status = "submission_failed"
    job.error_message = message
    db.commit()


def submit_processing_job_internal(
    request: ProcessingJobCreate,
    db: Session,
):
    """Shared submission service used by the API and GeoChat processing skills."""
    # Stage 1: Resolve the configured primary and fallback runners.
    candidates = _resolve_runner_candidates(request, db)
    _, mode, worker, runner = candidates[0]

    # Stage 2: Prepare job files and initial metadata.
    job_id, job_dir, result_path, log_path, metadata, datasets, manifest_path = _prepare_job_files(
        db, request, mode, worker, runner
    )

    # Stage 3: Save the job record before launching external work.
    job = _create_processing_job(db, job_id, worker, job_dir, log_path, result_path)
    for dataset in datasets:
        db.add(ProcessingJobInput(processing_job_id=job.id, dataset_id=dataset.id))
    db.commit()

    try:
        # Stage 4: Try primary, then fallback only if runner startup/submission fails.
        failures: list[str] = []
        selected_name = ""
        selected_mode = ""
        for runner_name, candidate_mode, candidate_worker, candidate_runner in candidates:
            try:
                if candidate_mode == "local":
                    candidate_log = job_dir / "local.log"
                    _start_local_worker(candidate_runner, candidate_worker, result_path, candidate_log, job_dir, metadata, manifest_path, request.target_well)
                    log_path = candidate_log
                    job.status = "running"
                else:
                    slurm_job_id = _submit_slurm_worker(candidate_runner, candidate_worker, result_path, job_dir, manifest_path, request.target_well)
                    job.slurm_job_id = slurm_job_id
                    job.status = "queued"
                    log_path = job_dir / f"slurm-{slurm_job_id}.log"
                selected_name, selected_mode = runner_name, candidate_mode
                metadata["shell_script"] = runner_name
                metadata["shell_script_path"] = str(candidate_runner)
                metadata["execution_mode"] = candidate_mode
                metadata["runner_attempts"] = failures + [f"{runner_name}: accepted"]
                (job_dir / "job.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
                break
            except (OSError, HTTPException) as exc:
                detail = exc.detail if isinstance(exc, HTTPException) else str(exc)
                if isinstance(detail, dict):
                    detail = detail.get("error", detail.get("message", str(detail)))
                failures.append(f"{runner_name}: {detail}")
        if not selected_name:
            message = "Both processing runners failed. " + " | ".join(failures)
            _mark_submission_failed(db, job, message)
            raise HTTPException(status_code=503, detail={"message": message, "job_id": job.id, "runner_errors": failures})
        job.log_path = str(log_path)

        # Stage 5: Save the launch result and build the API response.
        db.commit()
        db.refresh(job)
        response = _response(job)
        response.update({
            "python_script": request.python_script,
            "shell_script": selected_name,
            "execution_mode": selected_mode,
            "runner_attempts": metadata.get("runner_attempts", []),
            "dataset_ids": [dataset.id for dataset in datasets],
            "target_well": request.target_well,
            "input_manifest_path": str(manifest_path) if manifest_path else None,
        })
        return response

    except HTTPException:
        raise
    except subprocess.TimeoutExpired as exc:
        message = (
            "Timed out waiting for sbatch. Check SLURM before retrying to avoid duplicate jobs."
        )
        _mark_submission_failed(db, job, message)
        raise HTTPException(
            status_code=504,
            detail={"message": message, "job_id": job.id},
        ) from exc
    except OSError as exc:
        _mark_submission_failed(db, job, str(exc))
        raise HTTPException(
            status_code=503,
            detail={"message": "Unable to start processing job", "error": str(exc)},
        ) from exc



@router.post(
    "/processing-jobs",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Run a Python script locally or through SLURM",
)
def submit_processing_job(
    request: ProcessingJobCreate,
    db: Session = Depends(get_db),
):
    return submit_processing_job_internal(request, db)



@router.delete("/processing-jobs/{job_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a processing job and its folder")
def delete_processing_job(job_id: str, db: Session = Depends(get_db)):
    """Delete a terminal job record, its input associations, and its on-disk folder."""
    job = db.get(ProcessingJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Processing job not found")

    # Do not remove files while a worker may still be writing to the job folder.
    _refresh_status(job, db)
    db.refresh(job)
    if job.status not in TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot delete a processing job while it is {job.status}. Wait until it finishes.",
        )

    job_dir = Path(job.job_dir).resolve()
    # Restrict deletion to the job's own directory, never an arbitrary stored path.
    if job_dir.name != job.id or job_dir.parent.name != "processing_jobs":
        raise HTTPException(status_code=500, detail="Stored processing job directory is invalid")

    try:
        # A model release is a paired artifact of its training job. Remove it first;
        # an already-missing release is intentionally treated as a normal condition.
        from app.model_releases import delete_model_releases_for_processing_job
        delete_model_releases_for_processing_job(job.id, db)
        if job_dir.exists():
            if not job_dir.is_dir() or job_dir.is_symlink():
                raise HTTPException(status_code=500, detail="Stored processing job path is not a directory")
            shutil.rmtree(job_dir)
        db.query(ProcessingJobInput).filter(ProcessingJobInput.processing_job_id == job.id).delete(synchronize_session=False)
        db.delete(job)
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except OSError as exc:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Could not delete processing job folder: {exc}") from exc


@router.get("/processing-jobs/{job_id}", summary="Get processing job status")
def get_processing_job(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processing job not found")

    _refresh_status(job, db)
    db.refresh(job)
    response = _response(job)
    metadata = _read_json(Path(job.job_dir) / "job.json")
    response["python_script"] = metadata.get("python_script")
    response["shell_script"] = metadata.get("shell_script")
    response["dataset_ids"] = metadata.get("dataset_ids", [])
    response["input_manifest_path"] = metadata.get("input_manifest_path")
    response["target_well"] = metadata.get("target_well")
    return response



def _safe_json(path: Path) -> dict:
    """Read a job metadata file without failing the catalogue scan."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}


@router.get("/evidence/catalog", summary="Scan processing-job folders for evidence")
def list_evidence_catalog(db: Session = Depends(get_db)):
    """Discover existing processing results directly from configured data roots.

    Folder discovery is intentional: older results remain visible even if their
    processing-job database row is missing. Only direct child job folders are scanned.
    """
    discovered: dict[str, dict] = {}
    scanned_roots: list[str] = []
    for root in configured_storage_roots(db):
        jobs_root = root / "processing_jobs"
        if not jobs_root.is_dir():
            continue
        scanned_roots.append(str(jobs_root))
        try:
            job_dirs = list(jobs_root.iterdir())
        except OSError:
            continue
        for job_dir in job_dirs:
            if not job_dir.is_dir() or job_dir.is_symlink():
                continue
            result_path = job_dir / "result.json"
            job_meta_path = job_dir / "job.json"
            result = _safe_json(result_path)
            metadata = _safe_json(job_meta_path)
            # Include folders with any result metadata or at least one output file.
            try:
                files = [p for p in job_dir.rglob("*") if p.is_file() and not p.is_symlink() and job_dir.resolve() in p.resolve().parents]
            except OSError:
                files = []
            if not files and not result and not metadata:
                continue
            artifacts = []
            for file_path in sorted(files, key=lambda item: item.name.lower()):
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                if file_path.name in {"job.json", "result.json"}:
                    role = "metadata"
                elif file_path.suffix.lower() in {".csv", ".parquet", ".feather", ".npy", ".npz"}:
                    role = "data"
                elif file_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}:
                    role = "visualization"
                else:
                    role = "supporting"
                artifacts.append({"name": str(file_path.relative_to(job_dir)), "size_bytes": stat.st_size, "updated_at": stat.st_mtime, "role": role})
            job_id = job_dir.name
            item = {
                "id": job_id,
                "job_id": job_id,
                "status": result.get("status") or metadata.get("status") or "unknown",
                "operation": metadata.get("operation") or metadata.get("python_script") or result.get("operation"),
                "python_script": metadata.get("python_script") or result.get("python_script"),
                "target_well": metadata.get("target_well") or result.get("target_well"),
                "created_at": metadata.get("created_at") or result.get("created_at"),
                "summary": result.get("summary") or result.get("message") or metadata.get("description"),
                "artifacts": artifacts,
                "artifact_count": len([a for a in artifacts if a["role"] != "metadata"]),
                "has_result": result_path.is_file(),
            }
            # Prefer the first configured root (primary before fallback).
            discovered.setdefault(job_id, item)
    items = sorted(discovered.values(), key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return {"items": items, "count": len(items), "scanned_roots": scanned_roots}


def _specialised_explanation(job_id: str, db: Session) -> dict | None:
    """Explain one run with the output-analysis skill that owns its process, when there is one."""
    from importlib import import_module
    from openai import OpenAI
    from app.geochat.agent import explain_skill_result
    from app.geochat.skills.registry import SKILL_REGISTRY

    if Path(job_id).name != job_id or job_id in {".", ".."}:
        return None
    metadata: dict = {}
    for root in configured_storage_roots(db):
        candidate = root / "processing_jobs" / job_id
        if candidate.is_dir() and not candidate.is_symlink():
            metadata = _safe_json(candidate / "job.json")
            break
    for skill in SKILL_REGISTRY.values():
        if not (skill.name.startswith("process_") and skill.name.endswith("_output_analysis") and skill.handler):
            continue
        try:
            supports_job = getattr(import_module(f"app.geochat.skills.{skill.name}.handler"), "supports_job", None)
            if not callable(supports_job) or not supports_job(metadata):
                continue
            report = skill.handler(db=db, project_id=None, parameters={"job_id": job_id})
        except (ImportError, ValueError, OSError):
            continue
        if not report.get("applicable"):
            continue
        report["llm_explanation"] = None
        try:
            config = get_effective_settings(db).geochat
            client = OpenAI(base_url=config.base_url.rstrip("/"), api_key="token-not-needed", timeout=config.timeout_seconds)
            report["llm_explanation"], _ = explain_skill_result(
                client, model=config.model, system_prompt=config.system_prompt, skill=skill, result=report,
                question="What is this run, is it good enough to use, and what should I know first?",
                generation={"temperature": config.temperature, "max_tokens": config.max_tokens, "top_p": config.top_p},
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            report["llm_explanation"] = None
        # The same fixed layout Ask Geo shows; the dialog renders it instead of raw JSON.
        report["narrative"] = report["llm_explanation"]
        report["markdown"] = skill.renderer(report) if skill.renderer else None
        report["artifacts"] = [
            {**item, "format": (Path(str(item.get("name") or "")).suffix.lstrip(".").upper() or "FILE")}
            for item in report.get("artifacts", []) if isinstance(item, dict)
        ]
        report["findings"] = [str(item) for item in report.get("findings", [])]
        return report
    return None


@router.post("/evidence/{job_id}/explain", summary="Summarize all outputs from a processing run")
def explain_evidence_item(job_id: str, db: Session = Depends(get_db)):
    """Inspect a complete processing folder and add a grounded LLM-written explanation."""
    from openai import OpenAI, OpenAIError
    from app.geochat.skills.processing_output_folder_analysis.handler import execute

    # A registered process_*_output_analysis skill that recognises the run reads it as an expert
    # would; every other run gets the generic folder inventory. No workflow is named here.
    specialised = _specialised_explanation(job_id, db)
    if specialised is not None:
        return specialised

    try:
        report = execute(db=db, project_id=None, parameters={"job_id": job_id})
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    # Deterministic inspection is authoritative; the LLM only explains those facts.
    report["llm_explanation"] = None
    try:
        config = get_effective_settings(db).geochat
        client = OpenAI(
            base_url=config.base_url.rstrip("/"),
            api_key="token-not-needed",
            timeout=config.timeout_seconds,
        )
        evidence_context = {
            "python_script": report.get("python_script"),
            "operation": report.get("operation"),
            "target_well": report.get("target_well"),
            "status": report.get("status"),
            "artifacts": report.get("artifacts", []),
            "skipped": report.get("skipped", []),
            "domain_analysis": report.get("domain_analysis"),
            "deterministic_findings": report.get("findings", []),
            "limitations": report.get("limitations", []),
            "domain_interpretation_limits": report.get("domain_interpretation_limits", []),
        }
        completion = client.chat.completions.create(
            model=config.model,
            messages=[
                {"role": "system", "content": (
                    "You explain scientific-computing processing outputs to a geoscience user. "
                    "Write a clear, descriptive explanation of what this processing run appears "
                    "to do and what its output files contain. Start with the purpose of the run, "
                    "then explain the principal outputs and what their fields or statistics mean "
                    "at a practical level. Use only the supplied evidence. Do not infer a workflow "
                    "purpose from a filename alone; if purpose is unclear, say so. Distinguish "
                    "computed values from model predictions. Do not claim geological interpretations, "
                    "validated accuracy, or calibrated probabilities unless explicitly supported. "
                    "Mention important limitations naturally. Write 2–4 concise paragraphs in "
                    "plain language, without a heading or bullets."
                )},
                {"role": "user", "content": "Explain this processing run and its output folder using only these inspected facts:\n" + json.dumps(evidence_context, ensure_ascii=False, default=str)},
            ],
            temperature=min(float(config.temperature), 0.3),
            max_tokens=min(int(config.max_tokens), 650),
            top_p=config.top_p,
        )
        narrative = (completion.choices[0].message.content or "").strip()
        if narrative:
            report["llm_explanation"] = narrative
    except (OpenAIError, OSError, RuntimeError, TypeError, ValueError):
        # Keep deterministic findings usable if the configured LLM service is unavailable.
        report["llm_explanation"] = None
    return report


@router.post("/evidence/compare", summary="Compare CSV outputs from processing runs")
def compare_evidence_results(payload: EvidenceCompareRequest, db: Session = Depends(get_db)):
    """Compare shared numeric CSV columns from selected runs of one Python script."""
    if len(set(payload.job_ids)) != len(payload.job_ids):
        raise HTTPException(status_code=422, detail="Select each processing run only once.")
    if Path(payload.python_script).name != payload.python_script or not payload.python_script.endswith(".py"):
        raise HTTPException(status_code=422, detail="Invalid Python script name.")

    selected: list[tuple[str, Path, dict]] = []
    for job_id in payload.job_ids:
        if Path(job_id).name != job_id or job_id in {".", ".."}:
            raise HTTPException(status_code=422, detail="Invalid processing job ID.")
        job_dir = None
        for root in configured_storage_roots(db):
            jobs_root = (root / "processing_jobs").resolve()
            unresolved = root / "processing_jobs" / job_id
            if unresolved.is_symlink():
                continue
            candidate = unresolved.resolve()
            if jobs_root in candidate.parents and candidate.is_dir():
                job_dir = candidate
                break
        if job_dir is None:
            raise HTTPException(status_code=404, detail=f"Processing run not found: {job_id}")
        metadata = _safe_json(job_dir / "job.json")
        result = _safe_json(job_dir / "result.json")
        script = metadata.get("python_script") or result.get("python_script")
        if script != payload.python_script:
            raise HTTPException(status_code=422, detail=f"Run {job_id} was not produced by {payload.python_script}.")
        if (result.get("status") or metadata.get("status") or "").lower() not in {"succeeded", "completed", "success"}:
            raise HTTPException(status_code=422, detail=f"Run {job_id} has no successful completed result.")
        selected.append((job_id, job_dir, metadata))

    # Resolve declared outputs first, then recursively discover files for legacy jobs.
    # Matching is based on CSV schema, not filename: many workers include a well/run
    # identifier in their output filename even when the table structure is identical.
    def csv_outputs(job_dir: Path) -> list[Path]:
        result = _safe_json(job_dir / "result.json")
        raw = result.get("outputs") or result.get("results") or []
        if isinstance(raw, (str, dict)):
            raw = [raw]
        candidates: list[Path] = []
        for item in raw if isinstance(raw, list) else []:
            value = item.get("path") or item.get("file") or item.get("filename") if isinstance(item, dict) else item
            if not isinstance(value, str) or not value.strip():
                continue
            path = Path(value)
            path = path if path.is_absolute() else job_dir / path
            try:
                resolved = path.resolve()
                if job_dir.resolve() in resolved.parents and resolved.is_file() and resolved.suffix.lower() == ".csv":
                    candidates.append(resolved)
            except OSError:
                continue
        # Include nested outputs and legacy files not explicitly listed in result.json.
        try:
            for path in job_dir.rglob("*.csv"):
                if path.is_symlink():
                    continue
                resolved = path.resolve()
                if job_dir.resolve() in resolved.parents and resolved.is_file():
                    candidates.append(resolved)
        except OSError:
            pass
        return sorted(set(candidates), key=lambda path: str(path.relative_to(job_dir)).lower())

    def csv_schema(path: Path) -> tuple[str, ...]:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                header = next(csv.reader(handle), [])
            return tuple(str(name).strip() for name in header if str(name).strip())
        except (OSError, UnicodeDecodeError, csv.Error):
            return ()

    output_maps = [csv_outputs(job_dir) for _, job_dir, _ in selected]
    schema_maps = [{schema: path for path in paths if (schema := csv_schema(path))} for paths in output_maps]
    common_schemas = set.intersection(*(set(mapping) for mapping in schema_maps)) if schema_maps else set()
    if not common_schemas:
        discovered = [sorted({path.suffix.lower() for path in job_dir.rglob("*") if path.is_file()}) for _, job_dir, _ in selected]
        details = "; ".join(
            f"{job_id}: {len(paths)} CSV file(s), formats {', '.join(formats) or 'none'}"
            for (job_id, _, _), paths, formats in zip(selected, output_maps, discovered)
        )
        raise HTTPException(
            status_code=422,
            detail=("No compatible CSV output schemas were found across the selected runs. "
                    "Output filenames may differ, but CSV column headers must match. " + details),
        )

    comparisons = []
    for schema in sorted(common_schemas):
        run_rows = []
        paths_for_schema = [mapping[schema] for mapping in schema_maps]
        for (job_id, _, _), path in zip(selected, paths_for_schema):
            try:
                with path.open("r", encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle)
                    numeric: dict[str, list[float]] = {}
                    for row in reader:
                        for key, value in row.items():
                            if not key or value is None or not value.strip():
                                continue
                            try:
                                number = float(value)
                            except (TypeError, ValueError):
                                continue
                            if math.isfinite(number):
                                numeric.setdefault(key, []).append(number)
                run_rows.append((job_id, numeric))
            except (OSError, UnicodeDecodeError, csv.Error) as exc:
                raise HTTPException(status_code=422, detail=f"Could not read {path.name} from run {job_id}: {exc}") from exc
        common_columns = set.intersection(*(set(values) for _, values in run_rows)) if run_rows else set()
        columns = []
        for column in sorted(common_columns):
            stats = []
            for job_id, values in run_rows:
                numbers = values[column]
                if not numbers:
                    continue
                stats.append({
                    "job_id": job_id,
                    "count": len(numbers),
                    "mean": statistics.fmean(numbers),
                    "median": statistics.median(numbers),
                    "min": min(numbers),
                    "max": max(numbers),
                    "stddev": statistics.pstdev(numbers) if len(numbers) > 1 else 0.0,
                })
            if len(stats) == len(selected):
                columns.append({"name": column, "runs": stats})
        if columns:
            comparisons.append({
                "file_name": paths_for_schema[0].name if len({p.name for p in paths_for_schema}) == 1 else " / ".join(p.name for p in paths_for_schema),
                "schema": list(schema),
                "run_files": [{"job_id": job_id, "file_name": path.name, "relative_path": str(path.relative_to(job_dir))}
                              for (job_id, job_dir, _), path in zip(selected, paths_for_schema)],
                "columns": columns,
            })

    if not comparisons:
        raise HTTPException(status_code=422, detail="Compatible CSV files were found, but they contain no shared numeric columns with values in every selected run.")

    # Optional domain analyzers are discovered as independent output-analysis
    # skills. The processing API does not know any workflow-specific names.
    skill_analysis = None
    try:
        from importlib import import_module
        from app.geochat.skills.registry import SKILL_REGISTRY

        for skill in SKILL_REGISTRY.values():
            if not (skill.name.startswith("process_") and skill.name.endswith("_output_analysis")):
                continue
            module = import_module(f"app.geochat.skills.{skill.name}.handler")
            supports_job = getattr(module, "supports_job", None)
            compare_selected = getattr(module, "compare_selected_job_outputs", None)
            if callable(supports_job) and callable(compare_selected) and all(
                supports_job(metadata) for _, _, metadata in selected
            ):
                skill_analysis = compare_selected(db=db, job_ids=payload.job_ids)
                break
    except (ImportError, ValueError, OSError) as exc:
        skill_analysis = {"warning": f"A specialised output analysis skill was unavailable: {exc}"}

    return {
        "python_script": payload.python_script,
        "runs": [{"job_id": job_id, "created_at": metadata.get("created_at")} for job_id, _, metadata in selected],
        "comparisons": comparisons,
        "skill_analysis": skill_analysis,
        "method": "Descriptive statistics calculated from numeric columns in CSV outputs with matching column schemas. Output filenames may differ between runs.",
    }



@router.get("/processing-jobs", summary="List processing jobs")
def list_processing_jobs(db: Session = Depends(get_db), limit: int = 100, offset: int = 0):
    """List recent processing jobs for the Processing Job Inspector."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")
    if offset < 0:
        raise HTTPException(status_code=422, detail="offset must be non-negative")
    jobs = db.query(ProcessingJob).order_by(ProcessingJob.created_at.desc()).offset(offset).limit(limit).all()
    items = []
    for job in jobs:
        _refresh_status(job, db)
        db.refresh(job)
        item = _response(job)
        metadata = _read_json(Path(job.job_dir) / "job.json")
        item.update({
            "python_script": metadata.get("python_script"),
            "shell_script": metadata.get("shell_script"),
            "dataset_ids": metadata.get("dataset_ids", []),
            "input_manifest_path": metadata.get("input_manifest_path"),
            "target_well": metadata.get("target_well"),
            "job_dir": job.job_dir,
        })
        items.append(item)
    return {"items": items, "count": len(items), "limit": limit, "offset": offset}


@router.get("/processing-jobs/{job_id}/files", summary="List files in a processing job folder")
def list_processing_job_files(job_id: str, db: Session = Depends(get_db)):
    """List direct files in a job directory without exposing arbitrary filesystem paths."""
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processing job not found")
    job_dir = Path(job.job_dir).resolve()
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Processing job folder not found")
    items = []
    for path in sorted(job_dir.iterdir(), key=lambda item: item.name.lower()):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            items.append({"name": path.name, "size_bytes": stat.st_size, "updated_at": stat.st_mtime})
    return {"job_id": job.id, "job_dir": str(job_dir), "items": items, "count": len(items)}


@router.get("/processing-jobs/{job_id}/files/{file_name}", summary="Read a processing job file")
def read_processing_job_file(job_id: str, file_name: str, db: Session = Depends(get_db)):
    """Read a permitted text/JSON/log/Markdown/CSV file from the selected job folder."""
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processing job not found")
    if Path(file_name).name != file_name or file_name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid file name")
    allowed_suffixes = {".json", ".log", ".txt", ".out", ".err", ".csv", ".md"}
    if Path(file_name).suffix.lower() not in allowed_suffixes:
        raise HTTPException(status_code=415, detail="Only JSON, log, text, Markdown, and CSV files can be read")
    job_dir = Path(job.job_dir).resolve()
    target = (job_dir / file_name).resolve()
    if target.parent != job_dir or target.is_symlink() or not target.is_file():
        raise HTTPException(status_code=404, detail="Job file not found")
    max_bytes = 2 * 1024 * 1024
    size = target.stat().st_size
    if size > max_bytes:
        raise HTTPException(status_code=413, detail="File is larger than the 2 MiB read limit")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail="File is not UTF-8 text") from exc
    return {"job_id": job.id, "name": target.name, "size_bytes": size, "content": content}



@router.get("/processing-jobs/{job_id}/inference-slice", summary="Read a coordinate-aware inference prediction slice")
def get_inference_prediction_slice(
    job_id: str,
    attribute: str = "lithology",
    plane: str = "inline",
    coordinate: int = 0,
    max_dimension: int = 512,
    db: Session = Depends(get_db),
):
    """Return one downsampled prediction slice; never transfer the full NPY cube."""
    import numpy as np

    if attribute not in {"lithology", "vsh", "phit"}:
        raise HTTPException(status_code=422, detail="Unsupported prediction attribute")
    if plane not in {"inline", "crossline", "time"}:
        raise HTTPException(status_code=422, detail="Plane must be inline, crossline, or time")
    if max_dimension < 32 or max_dimension > 1024:
        raise HTTPException(status_code=422, detail="max_dimension must be between 32 and 1024")
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processing job not found")
    job_dir = Path(job.job_dir).resolve()
    candidates = [
        job_dir / "stages" / "09_inference",
        job_dir / "09_inference",
    ]
    stage_dir = next((p for p in candidates if (p / "volume_inference_summary.json").is_file()), None)
    if stage_dir is None:
        raise HTTPException(status_code=404, detail="This job does not contain registered seismic-volume inference outputs")
    summary = _read_json(stage_dir / "volume_inference_summary.json")
    geometry = _read_json(stage_dir / "volumes" / "volume_geometry.json")
    if not geometry:
        raise HTTPException(status_code=404, detail="Inference volume geometry is unavailable")
    cube_path = stage_dir / "volumes" / f"prediction_{attribute}.npy"
    if cube_path.is_symlink() or not cube_path.is_file():
        raise HTTPException(status_code=404, detail=f"Prediction cube is unavailable: {attribute}")
    try:
        cube = np.load(cube_path, mmap_mode="r", allow_pickle=False)
        valid_path = stage_dir / "volumes" / "prediction_valid.npy"
        valid = np.load(valid_path, mmap_mode="r", allow_pickle=False) if valid_path.is_file() and not valid_path.is_symlink() else None
        il_min, il_step = int(geometry.get("il_min", 0)), int(geometry.get("il_step", 1))
        xl_min, xl_step = int(geometry.get("xl_min", 0)), int(geometry.get("xl_step", 1))
        t0, dt = float(geometry.get("t0_ms", 0)), float(geometry.get("dt_ms", 1))
        if plane == "inline":
            coordinate_min, coordinate_step, coordinate_count = il_min, il_step, cube.shape[0]
            index = (cube.shape[0] // 2) if coordinate == 0 else round((coordinate - il_min) / il_step)
            if index < 0 or index >= cube.shape[0]: raise HTTPException(status_code=422, detail="Inline coordinate outside prediction region")
            data = np.asarray(cube[index, :, :])
            mask = np.asarray(valid[index, :, :]) if valid is not None else np.isfinite(data)
            axes = {"x": [xl_min + i * xl_step for i in range(cube.shape[1])], "y": [t0 + i * dt for i in range(cube.shape[2])]}
            actual_coordinate = il_min + index * il_step
        elif plane == "crossline":
            coordinate_min, coordinate_step, coordinate_count = xl_min, xl_step, cube.shape[1]
            index = (cube.shape[1] // 2) if coordinate == 0 else round((coordinate - xl_min) / xl_step)
            if index < 0 or index >= cube.shape[1]: raise HTTPException(status_code=422, detail="Crossline coordinate outside prediction region")
            data = np.asarray(cube[:, index, :])
            mask = np.asarray(valid[:, index, :]) if valid is not None else np.isfinite(data)
            axes = {"x": [il_min + i * il_step for i in range(cube.shape[0])], "y": [t0 + i * dt for i in range(cube.shape[2])]}
            actual_coordinate = xl_min + index * xl_step
        else:
            coordinate_min, coordinate_step, coordinate_count = t0, dt, cube.shape[2]
            index = (cube.shape[2] // 2) if coordinate == 0 else round((coordinate - t0) / dt)
            if index < 0 or index >= cube.shape[2]: raise HTTPException(status_code=422, detail="Time coordinate outside prediction region")
            data = np.asarray(cube[:, :, index])
            mask = np.asarray(valid[:, :, index]) if valid is not None else np.isfinite(data)
            axes = {"x": [xl_min + i * xl_step for i in range(cube.shape[1])], "y": [il_min + i * il_step for i in range(cube.shape[0])]}
            actual_coordinate = t0 + index * dt
        stride = max(1, int(np.ceil(max(data.shape) / max_dimension)))
        data = data[::stride, ::stride]
        mask = mask[::stride, ::stride] & np.isfinite(data)
        safe = np.where(mask, data, 0).astype(float)
        return {
            "job_id": job_id, "attribute": attribute, "plane": plane,
            "coordinate": actual_coordinate, "coordinate_min": coordinate_min,
            "coordinate_step": coordinate_step, "coordinate_count": coordinate_count,
            "axes": axes, "shape": list(data.shape),
            "downsample_stride": stride, "values": safe.tolist(),
            "valid_mask": mask.astype(bool).tolist(),
            "valid_count": int(mask.sum()), "invalid_count": int(mask.size - mask.sum()),
            "volume_shape": list(cube.shape), "summary": summary,
            "geometry_flags": geometry.get("geometry_flags", []),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not read inference slice: {exc}") from exc


VOLUME_PRODUCT_ROOTS = ("stages/10_summary", "stages/11_volume_export")
VOLUME_PRODUCT_SUFFIXES = {".json", ".csv", ".md", ".png", ".svg", ".html", ".segy", ".sgy", ".npy"}


def _volume_product_file(job_dir: Path, relative_path: str) -> Path:
    if not relative_path or "\\" in relative_path:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not any(str(relative).startswith(root + "/") for root in VOLUME_PRODUCT_ROOTS):
        raise HTTPException(status_code=403, detail="Artifact is outside the Volume Products output directories")
    target = (job_dir / relative).resolve()
    root = job_dir.resolve()
    if root not in target.parents or target.is_symlink() or not target.is_file():
        raise HTTPException(status_code=404, detail="Volume Products artifact not found")
    if target.suffix.lower() not in VOLUME_PRODUCT_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported Volume Products artifact type")
    return target


@router.get("/processing-jobs/{job_id}/volume-products", summary="List Volume Products result artifacts")
def list_volume_product_artifacts(job_id: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processing job not found")
    job_dir = Path(job.job_dir).resolve()
    items = []
    for relative_root in VOLUME_PRODUCT_ROOTS:
        folder = job_dir / relative_root
        if not folder.is_dir() or folder.is_symlink():
            continue
        for path in folder.rglob("*"):
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in VOLUME_PRODUCT_SUFFIXES:
                continue
            relative = path.relative_to(job_dir).as_posix()
            stat = path.stat()
            items.append({"path": relative, "name": path.name, "suffix": path.suffix.lower(), "size_bytes": stat.st_size, "category": "summary" if relative.startswith("stages/10_summary/") else "export"})
    items.sort(key=lambda item: item["path"].lower())
    return {"job_id": job_id, "items": items, "count": len(items)}


@router.get("/processing-jobs/{job_id}/volume-products/file", summary="Read or serve a Volume Products artifact")
def get_volume_product_artifact(job_id: str, path: str, db: Session = Depends(get_db)):
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processing job not found")
    target = _volume_product_file(Path(job.job_dir).resolve(), path)
    if target.suffix.lower() in {".png", ".svg", ".html", ".segy", ".sgy", ".npy"}:
        media = {".png": "image/png", ".svg": "image/svg+xml", ".html": "text/html", ".segy": "application/octet-stream", ".sgy": "application/octet-stream", ".npy": "application/octet-stream"}.get(target.suffix.lower(), "application/octet-stream")
        return FileResponse(target, media_type=media, filename=target.name)
    max_bytes = 5 * 1024 * 1024
    if target.stat().st_size > max_bytes:
        raise HTTPException(status_code=413, detail="Text artifact exceeds the 5 MiB read limit")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail="Artifact is not UTF-8 text") from exc
    return {"job_id": job_id, "path": path, "name": target.name, "size_bytes": target.stat().st_size, "content": content}


STAGE_RESULT_ROOTS = {
    "well-geobody": ("stages/13_well_geobody_analysis",),
    "prospect": ("stages/14_prospect",),
    "foundations": ("stages/00_inventory", "stages/01_well_ties", "stages/02_labels", "stages/03_extraction", "stages/04_features", "stages/05_training_tables", "stages/06_feature_selection", "stages/07_models"),
}
STAGE_RESULT_ROOT_FILES = {
    "well-geobody": ("geogg_well_geobody_analysis_preflight.json", "geogg_well_geobody_analysis_summary.json"),
    "prospect": ("geogg_prospect_preflight.json", "geogg_prospect_summary.json"),
    "foundations": ("geogg_foundations_summary.json", "result.json", "inputs.json", "job.json"),
}
STAGE_RESULT_SUFFIXES = {".json", ".csv", ".md", ".txt", ".png", ".jpg", ".jpeg", ".svg", ".pdf", ".npy", ".npz", ".ply", ".csv.gz", ".joblib", ".pt", ".pth", ".yaml", ".yml", ".log"}


def _stage_result_relative_allowed(relative: Path, stage_id: str) -> bool:
    """Allow only the selected stage subtree or its known root summary files.

    Some older/local runners place the job payload one directory below the DB
    job_dir. Match the stage path as a complete path segment, not by basename.
    """
    parts = relative.parts
    roots = STAGE_RESULT_ROOTS.get(stage_id, ())
    for root in roots:
        root_parts = Path(root).parts
        if any(parts[index:index + len(root_parts)] == root_parts for index in range(len(parts) - len(root_parts) + 1)):
            return True
    return relative.name in STAGE_RESULT_ROOT_FILES.get(stage_id, ())


def _stage_result_target(job_dir: Path, stage_id: str, relative_path: str) -> Path:
    if stage_id not in STAGE_RESULT_ROOTS or not relative_path or "\\" in relative_path:
        raise HTTPException(status_code=400, detail="Invalid stage result path")
    relative = Path(relative_path)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise HTTPException(status_code=400, detail="Invalid stage result path")
    if not _stage_result_relative_allowed(relative, stage_id):
        raise HTTPException(status_code=403, detail="Artifact is outside the selected stage output directory")
    base = job_dir.resolve()
    target = (base / relative).resolve()
    if base not in target.parents or target.is_symlink() or not target.is_file():
        raise HTTPException(status_code=404, detail="Stage result artifact not found")
    if target.suffix.lower() not in STAGE_RESULT_SUFFIXES:
        raise HTTPException(status_code=415, detail="Unsupported stage result artifact type")
    return target


@router.get("/processing-jobs/{job_id}/stage-results", summary="List Well Geobody or Prospect result artifacts")
def list_stage_result_artifacts(job_id: str, stage: str, db: Session = Depends(get_db)):
    if stage not in STAGE_RESULT_ROOTS:
        raise HTTPException(status_code=400, detail="Unsupported stage result type")
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processing job not found")
    job_dir = Path(job.job_dir).resolve()
    if not job_dir.is_dir():
        raise HTTPException(status_code=404, detail="Processing job output directory not found")

    # Search the full job directory for the exact stage subtree. This supports
    # legacy/local jobs whose runner stored the payload under an extra folder.
    stage_roots = []
    for root in STAGE_RESULT_ROOTS[stage]:
        root_path = Path(root)
        direct = job_dir / root_path
        if direct.is_dir() and not direct.is_symlink():
            stage_roots.append(direct)
        else:
            suffix_parts = root_path.parts
            for candidate in job_dir.rglob(suffix_parts[-1]):
                if candidate.is_dir() and not candidate.is_symlink() and candidate.parts[-len(suffix_parts):] == suffix_parts:
                    try:
                        resolved = candidate.resolve()
                        if job_dir in resolved.parents:
                            stage_roots.append(candidate)
                    except OSError:
                        continue

    items = []
    seen = set()
    for name in STAGE_RESULT_ROOT_FILES[stage]:
        candidates = [job_dir / name]
        candidates.extend(job_dir.rglob(name))
        for path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            try:
                relative = path.relative_to(job_dir).as_posix()
                _stage_result_target(job_dir, stage, relative)
            except (ValueError, HTTPException, OSError):
                continue
            if relative in seen:
                continue
            seen.add(relative)
            stat = path.stat()
            items.append({"path": relative, "name": path.name, "suffix": path.suffix.lower(), "size_bytes": stat.st_size, "category": "summary"})

    for root in stage_roots:
        for path in root.rglob("*"):
            if not path.is_file() or path.is_symlink() or path.suffix.lower() not in STAGE_RESULT_SUFFIXES:
                continue
            try:
                relative = path.relative_to(job_dir).as_posix()
                _stage_result_target(job_dir, stage, relative)
            except (ValueError, HTTPException, OSError):
                continue
            if relative in seen:
                continue
            seen.add(relative)
            stat = path.stat()
            suffix = path.suffix.lower()
            category = "figure" if suffix in {".png", ".jpg", ".jpeg", ".svg"} else "table" if suffix == ".csv" else "model" if suffix in {".joblib", ".pt", ".pth"} else "data" if suffix in {".npy", ".npz", ".ply"} or "/patches/" in relative or "/transforms/" in relative else "document" if suffix in {".pdf", ".md", ".txt", ".yaml", ".yml", ".log"} else "summary"
            items.append({"path": relative, "name": path.name, "suffix": suffix, "size_bytes": stat.st_size, "category": category})
    items.sort(key=lambda item: item["path"].lower())
    return {"job_id": job_id, "stage": stage, "items": items, "count": len(items), "job_dir_exists": True, "stage_roots_found": len(stage_roots)}


@router.get("/processing-jobs/{job_id}/prospect-voxel-preview", summary="Return a bounded preview of one Prospect voxel set")
def get_prospect_voxel_preview(job_id: str, prospect_id: str, max_points: int = 30000, db: Session = Depends(get_db)):
    if not re.fullmatch(r"P\d{3}", prospect_id or ""):
        raise HTTPException(status_code=400, detail="Invalid prospect ID")
    if max_points < 1000 or max_points > 60000:
        raise HTTPException(status_code=400, detail="max_points must be between 1000 and 60000")
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processing job not found")
    job_dir = Path(job.job_dir).resolve()
    # Locate the stage subtree using the same legacy-aware path matching as the artifact browser.
    candidates = [job_dir / "stages/14_prospect/intermediate/prospect_voxels" / f"{prospect_id}_ijk.npy"]
    candidates.extend(job_dir.rglob(f"{prospect_id}_ijk.npy"))
    target = None
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            relative = resolved.relative_to(job_dir)
            if candidate.is_file() and not candidate.is_symlink() and _stage_result_relative_allowed(relative, "prospect"):
                target = resolved
                break
        except (ValueError, OSError):
            continue
    if target is None:
        raise HTTPException(status_code=404, detail=f"Voxel data for {prospect_id} is not available")
    try:
        import numpy as np
        coords = np.load(target, mmap_mode="r", allow_pickle=False)
        if coords.ndim != 2 or coords.shape[1] != 3 or coords.dtype.kind not in "iu":
            raise ValueError("Expected an integer N×3 IJK array")
        total = int(coords.shape[0])
        stride = max(1, int(np.ceil(total / max_points)))
        preview = np.asarray(coords[::stride], dtype=np.int32)
        mins = np.asarray(coords.min(axis=0), dtype=np.int64).tolist() if total else [0, 0, 0]
        maxs = np.asarray(coords.max(axis=0), dtype=np.int64).tolist() if total else [0, 0, 0]
        return {"job_id": job_id, "prospect_id": prospect_id, "coordinate_order": ["I", "J", "K"], "voxel_count": total, "preview_count": int(preview.shape[0]), "sampling_stride": stride, "bounds_ijk": {"min": mins, "max": maxs}, "coordinates": preview.tolist(), "is_sampled": stride > 1}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not read voxel array: {exc}") from exc


@router.get("/processing-jobs/{job_id}/stage-results/file", summary="Read or serve a Well Geobody or Prospect result artifact")
def get_stage_result_artifact(job_id: str, stage: str, path: str, db: Session = Depends(get_db)):
    if stage not in STAGE_RESULT_ROOTS:
        raise HTTPException(status_code=400, detail="Unsupported stage result type")
    job = db.get(ProcessingJob, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Processing job not found")
    target = _stage_result_target(Path(job.job_dir), stage, path)
    suffix = target.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".npy", ".npz", ".ply"}:
        media = {".png":"image/png", ".jpg":"image/jpeg", ".jpeg":"image/jpeg", ".svg":"image/svg+xml", ".pdf":"application/pdf"}.get(suffix, "application/octet-stream")
        return FileResponse(target, media_type=media, filename=target.name)
    max_bytes = 5 * 1024 * 1024
    if target.stat().st_size > max_bytes:
        raise HTTPException(status_code=413, detail="Text artifact exceeds the 5 MiB read limit")
    try:
        content = target.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=415, detail="Artifact is not UTF-8 text") from exc
    return {"job_id": job_id, "stage": stage, "path": path, "name": target.name, "size_bytes": target.stat().st_size, "content": content}


# Project-wide Well 3D catalogue. Each well is an independent folder under
# site_data/well_3d/<well-name>; only known presentation formats are exposed.
WELL3D_IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp'}
WELL3D_HTML_SUFFIXES = {'.html', '.htm'}
WELL3D_TABLE_SUFFIXES = {'.csv', '.tsv'}
WELL3D_DATA_SUFFIXES = {'.json', '.npz', '.npy'}


def _well3d_root(db: Session) -> Path | None:
    for root in configured_storage_roots(db):
        candidate = root / 'well_3d'
        if candidate.is_dir() and not candidate.is_symlink():
            return candidate.resolve()
    return None


def _well3d_file(root: Path, well: str, relative_path: str) -> Path:
    lexical_well_dir = root / well
    if lexical_well_dir.is_symlink():
        raise HTTPException(status_code=404, detail='Symbolic links are not served from Well 3D folders')
    well_dir = lexical_well_dir.resolve()
    if root not in well_dir.parents or not well_dir.is_dir():
        raise HTTPException(status_code=404, detail='Well folder not found')
    lexical_target = well_dir / relative_path
    # Reject symlinks before resolving; resolving first would hide the link itself.
    if any(part.is_symlink() for part in [well_dir, *lexical_target.parents] if part != root) or lexical_target.is_symlink():
        raise HTTPException(status_code=404, detail='Symbolic links are not served from Well 3D folders')
    target = lexical_target.resolve()
    if well_dir not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail='Well 3D file not found')
    if target.suffix.lower() not in WELL3D_IMAGE_SUFFIXES | WELL3D_HTML_SUFFIXES | WELL3D_TABLE_SUFFIXES | WELL3D_DATA_SUFFIXES:
        raise HTTPException(status_code=415, detail='Unsupported Well 3D file type')
    return target


@router.get('/evidence/well-3d/wells', summary='List available Well 3D folders')
def list_well3d_wells(db: Session = Depends(get_db)):
    root = _well3d_root(db)
    if root is None:
        return {'items': [], 'count': 0, 'available': False}
    items = []
    for folder in sorted(root.iterdir(), key=lambda p: p.name.lower()):
        if not folder.is_dir() or folder.is_symlink():
            continue
        files = [p for p in folder.rglob('*') if p.is_file() and not p.is_symlink() and folder.resolve() in p.resolve().parents]
        html = [p for p in files if p.suffix.lower() in WELL3D_HTML_SUFFIXES]
        images = [p for p in files if p.suffix.lower() in WELL3D_IMAGE_SUFFIXES]
        tables = [p for p in files if p.suffix.lower() in WELL3D_TABLE_SUFFIXES]
        others = [p for p in files if p.suffix.lower() in WELL3D_DATA_SUFFIXES]
        items.append({'well': folder.name, 'html': [str(p.relative_to(folder)) for p in html], 'images': [str(p.relative_to(folder)) for p in images], 'tables': [str(p.relative_to(folder)) for p in tables], 'others': [str(p.relative_to(folder)) for p in others], 'file_count': len(files)})
    return {'items': items, 'count': len(items), 'available': True}


@router.get('/evidence/well-3d/{well}/file', summary='Serve a Well 3D asset')
def serve_well3d_file(well: str, path: str, db: Session = Depends(get_db)):
    root = _well3d_root(db)
    if root is None:
        raise HTTPException(status_code=404, detail='site_data/well_3d is not available')
    target = _well3d_file(root, well, path)
    suffix = target.suffix.lower()
    media = {'.png':'image/png', '.jpg':'image/jpeg', '.jpeg':'image/jpeg', '.webp':'image/webp', '.html':'text/html', '.htm':'text/html', '.csv':'text/csv; charset=utf-8', '.tsv':'text/tab-separated-values; charset=utf-8', '.json':'application/json'}.get(suffix, 'application/octet-stream')
    headers = {'Content-Security-Policy': "default-src 'self' 'unsafe-inline' 'unsafe-eval' data: blob: https:; img-src 'self' data: blob: https:; connect-src 'self' data: blob: https:; frame-ancestors 'self'", 'X-Content-Type-Options': 'nosniff'} if suffix in WELL3D_HTML_SUFFIXES else None
    return FileResponse(target, media_type=media, filename=None, headers=headers)


@router.get('/evidence/well-3d/{well}/table', summary='Preview a Well 3D CSV or TSV')
def preview_well3d_table(well: str, path: str, offset: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    if offset < 0 or limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail='Invalid table pagination')
    root = _well3d_root(db)
    if root is None:
        raise HTTPException(status_code=404, detail='site_data/well_3d is not available')
    target = _well3d_file(root, well, path)
    if target.suffix.lower() not in WELL3D_TABLE_SUFFIXES:
        raise HTTPException(status_code=415, detail='File is not a CSV or TSV table')
    if target.stat().st_size > 100 * 1024 * 1024:
        raise HTTPException(status_code=413, detail='Table exceeds the 100 MiB preview limit')
    delimiter = '\t' if target.suffix.lower() == '.tsv' else ','
    try:
        with target.open('r', encoding='utf-8-sig', newline='') as stream:
            reader = csv.reader(stream, delimiter=delimiter)
            headers = next(reader, [])
            rows = []
            total = 0
            for index, row in enumerate(reader):
                total += 1
                if offset <= index < offset + limit:
                    rows.append(row)
        return {'name': target.name, 'columns': headers, 'rows': rows, 'offset': offset, 'limit': limit, 'total_rows': total}
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise HTTPException(status_code=422, detail=f'Could not read table: {exc}') from exc
