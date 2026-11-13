"""Run migrated specialist stages inside one existing APGCE processing job."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback
from typing import Any

from .config import load_yaml_config
from .context import StageContext
from .contracts import StageResult, StageStatus
from .provenance import register_output, write_stage_provenance
from .model_release import create_model_release
from .stage_compat import SpecialistStageContext, StageBlocked


STAGE_MODULES = {
    "00_inventory": "geogg_fresh.stages.s00_inventory",
    "01_well_ties": "geogg_fresh.stages.s01_well_ties",
    "02_labels": "geogg_fresh.stages.s02_labels",
    "03_extraction": "geogg_fresh.stages.s03_extraction",
    "04_features": "geogg_fresh.stages.s04_features",
    "05_training_tables": "geogg_fresh.stages.s05_training_tables",
    "06_feature_selection": "geogg_fresh.stages.s06_feature_selection",
    "07_models": "geogg_fresh.stages.s07_models",
}


def run_foundations(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GeoGG stages 00–07: foundations, labels, extraction, features, selection and modelling")
    default_config = os.environ.get("APGCE_GEOGG_CONFIG")
    if not default_config:
        candidate = Path(__file__).resolve().parents[1] / "A_foundations.yaml"
        default_config = str(candidate) if candidate.is_file() else None
    parser.add_argument("--config", default=default_config,
                        help="Editable GeoGG YAML config (or APGCE_GEOGG_CONFIG)")
    parser.add_argument(
        "--input-manifest",
        default=os.environ.get("APGCE_INPUT_MANIFEST"),
        help="Backend-generated inputs.json (or APGCE_INPUT_MANIFEST)",
    )
    args = parser.parse_args(argv)
    # The APGCE runner currently supplies the manifest as a CLI argument.
    # StageContext deliberately reads the same value from the environment,
    # keeping manifest parsing and dataset identity handling in one place.
    if args.input_manifest:
        os.environ["APGCE_INPUT_MANIFEST"] = str(Path(args.input_manifest).expanduser().resolve())
    if not args.config:
        parser.error("--config or APGCE_GEOGG_CONFIG is required")

    result_value = os.environ.get("APGCE_RESULT_PATH")
    if not result_value:
        raise RuntimeError("APGCE_RESULT_PATH is required by the existing processing runner")
    job_dir = Path(result_value).expanduser().resolve().parent
    # The APGCE processing-job UUID is the authoritative GeoGG run identity.
    # The backend stores each job at processing_jobs/<job-uuid>/; deriving the
    # ID from that directory also supports older runners that do not export it.
    processing_job_id = os.environ.get("APGCE_JOB_ID") or job_dir.name
    if processing_job_id != job_dir.name:
        raise RuntimeError(
            f"APGCE_JOB_ID ({processing_job_id}) does not match the processing "
            f"job directory ID ({job_dir.name}). Refusing to write outputs outside the tracked job."
        )
    run_id = os.environ.get("APGCE_GEOGG_RUN_ID") or processing_job_id
    if run_id != processing_job_id:
        raise RuntimeError(
            "APGCE_GEOGG_RUN_ID must match the APGCE processing job ID; "
            "independent GeoGG run IDs are disabled to preserve job tracking."
        )
    cfg = load_yaml_config(args.config, output_root=job_dir)
    stage_results: list[dict[str, Any]] = []
    overall_status = "succeeded"

    for stage_id, module_name in STAGE_MODULES.items():
        base = StageContext.from_environment(stage_id=stage_id, config=cfg, run_id=run_id)
        if any(base.stage_dir.iterdir()):
            raise RuntimeError(
                f"Refusing to overwrite existing stage output: {base.stage_dir}. "
                "Use a new APGCE_GEOGG_RUN_ID for a fresh run."
            )
        compat = SpecialistStageContext(base, cfg, stage_id)
        module = __import__(module_name, fromlist=["run"])
        try:
            module.run(compat)
            failed_checks = [name for name, item in compat.checks.items() if not item.get("passed")]
            if compat.blockers:
                status = StageStatus.BLOCKED
            else:
                status = StageStatus.CONDITIONAL if compat.limitations or failed_checks else StageStatus.COMPLETED
            summary = (f"{stage_id} completed with {len(compat.limitations)} limitation(s) and "
                       f"{len(failed_checks)} failed check(s).")
        except StageBlocked as exc:
            compat.blockers.append(str(exc))
            status, summary = StageStatus.BLOCKED, str(exc)
        except Exception:
            error = traceback.format_exc()
            base.resolve_output("traceback.txt").write_text(error, encoding="utf-8")
            failed = StageResult(stage_id, StageStatus.FAILED, "Stage execution failed.",
                                 warnings=compat.warnings, limitations=compat.limitations,
                                 blockers=compat.blockers, checks=compat.checks, metrics=compat.metrics)
            _write_provenance(base, failed)
            stage_results.append({"stage_id": stage_id, **failed.to_dict()})
            _write_foundations_summary(job_dir, run_id, "failed", stage_results, error)
            raise

        result = StageResult(
            stage_id=stage_id, status=status, summary=summary,
            warnings=compat.warnings, limitations=compat.limitations,
            quality_flags=compat.quality_flags,
            blockers=compat.blockers, checks=compat.checks,
            metrics={**compat.metrics, **compat.extra},
        )
        _write_provenance(base, result)
        stage_results.append({"stage_id": stage_id, **result.to_dict()})
        print(f"GeoGG {stage_id}: {status.value} — {summary}")
        if status == StageStatus.BLOCKED:
            overall_status = "failed"
            break

    model_release = None
    completed_stage_ids = {item.get("stage_id") for item in stage_results}
    if overall_status == "succeeded" and "07_models" in completed_stage_ids:
        model_release = create_model_release(job_dir=job_dir, config=cfg, run_id=run_id)
        if model_release is None:
            print("GeoGG model release: no successful final model artifacts; no release registered.")
        else:
            print(f"GeoGG model release: {model_release['model_release_id']} — "
                  f"{model_release['model_count']} model artifact(s) packaged.")
    _write_foundations_summary(job_dir, run_id, overall_status, stage_results, model_release=model_release)
    return 0 if overall_status == "succeeded" else 2


def _write_provenance(base: StageContext, result: StageResult) -> None:
    outputs = []
    for path in sorted(base.stage_dir.rglob("*")):
        if path.is_file() and path.name != "provenance.json":
            outputs.append(register_output(base.stage_dir, path, role=path.parent.name or "artifact"))
    result.outputs = outputs
    inputs = [dict(item.__dict__) for item in base.inputs]
    manifest_by_path = {str(Path(item["path"]).resolve()): item for item in inputs}
    for path, digest in base._input_hashes.items():
        registered = manifest_by_path.get(str(Path(path).resolve()))
        if registered is not None:
            registered["sha256"] = digest
            continue
        inputs.append({"dataset_id": None, "kind": "scientific_input", "format": Path(path).suffix.lstrip("."),
                       "name": Path(path).name, "path": path, "sha256": digest})
    write_stage_provenance(
        provenance_dir=base.layout.provenance, stage_id=base.stage_id,
        run_id=base.run_id, config_hash=base.config_hash, result=result, inputs=inputs,
    )


def _write_foundations_summary(
    job_dir: Path, run_id: str, status: str, stages: list[dict[str, Any]], error: str | None = None,
    model_release: dict[str, Any] | None = None,
) -> Path:
    """Write GeoGG stage details separately from the decorator-owned result.json."""
    payload = {
        "status": status,
        "operation": "geogg_fresh_stages_00_07",
        "processing_job_id": job_dir.name,
        "run_id": run_id,
        "job_dir": str(job_dir),
        "stages": stages,
        "model_release": model_release,
    }
    if error:
        payload["error_message"] = error[-8000:]
    path = job_dir / "geogg_foundations_summary.json"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path
