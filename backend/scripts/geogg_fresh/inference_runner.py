"""Independent GeoGG Stage F runner (08 evaluation + 09 inference)."""
from __future__ import annotations
import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

from .config import load_yaml_config
from .context import StageContext
from .stage_compat import SpecialistStageContext, StageBlocked
from .contracts import StageResult, StageStatus
from .provenance import register_output, sha256_file, write_stage_provenance


def _resolve_release(job_dir: Path, release_id: str) -> Path:
    if not release_id or Path(release_id).name != release_id or release_id in {".", ".."}:
        raise ValueError("inference.model_release_id must be a simple release ID")
    storage_root = job_dir.parent.parent.resolve()
    release_root = (storage_root / "model_releases").resolve()
    if release_root.parent != storage_root:
        raise ValueError("Invalid configured model release root")
    release = release_root / release_id
    if release.is_symlink() or not release.is_dir() or release.resolve().parent != release_root:
        raise FileNotFoundError(f"Model release not found or unsafe: {release_id}")
    manifest_path = release / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Model release manifest is missing or unsafe")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model_release_id") != release_id or manifest.get("status") != "ready":
        raise ValueError("Model release identity/status does not match requested release")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Model release has no verifiable file inventory")
    for relative, metadata in files.items():
        rel = Path(relative)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"Unsafe path in release inventory: {relative}")
        target = release / rel
        if target.is_symlink() or not target.is_file() or release.resolve() not in target.resolve().parents:
            raise ValueError(f"Missing/unsafe release artifact: {relative}")
        if target.stat().st_size != int(metadata.get("bytes", -1)) or sha256_file(target) != metadata.get("sha256"):
            raise ValueError(f"Release checksum/size mismatch: {relative}")
    return release.resolve()



def _inference_stage_status(
    stage_id: str,
    checks: dict[str, dict[str, Any]],
    limitations: list[str] | None = None,
    blockers: list[str] | None = None,
) -> StageStatus:
    """Apply stage-specific status rules without hiding scientific limitations."""
    if blockers:
        return StageStatus.BLOCKED
    failed_checks = [
        name for name, check in checks.items()
        if not check.get("passed", False)
    ]
    if failed_checks:
        return StageStatus.CONDITIONAL
    # Stage 08 limitations can qualify evaluation completeness. Stage 09
    # geometry limitations are downstream spatial-quality constraints and do
    # not invalidate a successfully generated native IL/XL prediction cube.
    if stage_id == "08_evaluation" and limitations:
        return StageStatus.CONDITIONAL
    return StageStatus.COMPLETED


def run_inference(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run GeoGG Stage F: held-out evaluation and released-model inference")
    default_config = os.environ.get("APGCE_GEOGG_INFERENCE_CONFIG")
    if not default_config:
        candidate = Path(__file__).resolve().parents[1] / "B_inference.yaml"
        default_config = str(candidate) if candidate.is_file() else None
    parser.add_argument("--config", default=default_config, help="Dedicated Stage F YAML config")
    parser.add_argument("--input-manifest", default=os.environ.get("APGCE_INPUT_MANIFEST"))
    args = parser.parse_args(argv)
    if not args.config:
        parser.error("--config or APGCE_GEOGG_INFERENCE_CONFIG is required")
    if args.input_manifest:
        os.environ["APGCE_INPUT_MANIFEST"] = str(Path(args.input_manifest).expanduser().resolve())
    result_value = os.environ.get("APGCE_RESULT_PATH")
    if not result_value:
        raise RuntimeError("APGCE_RESULT_PATH is required")
    result_path = Path(result_value).expanduser().resolve()
    job_dir = result_path.parent
    job_id = os.environ.get("APGCE_JOB_ID") or job_dir.name
    if job_id != job_dir.name:
        raise RuntimeError("APGCE_JOB_ID does not match the active processing job directory")
    cfg = load_yaml_config(args.config, output_root=job_dir)
    inf = cfg.get("inference") or {}
    release_id = str(inf.get("model_release_id") or "").strip()
    release = _resolve_release(job_dir, release_id)
    release_manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    provenance = release_manifest.get("provenance") or {}
    foundation_job_id = str(
        release_manifest.get("foundation_job_id")
        or provenance.get("source_processing_job_id")
        or release_manifest.get("processing_job_id")
        or ""
    ).strip()
    if foundation_job_id and (
        Path(foundation_job_id).name != foundation_job_id or foundation_job_id in {".", ".."}
    ):
        raise ValueError(f"Model release {release_id} contains an invalid foundation_job_id")
    stages = ["08_evaluation", "09_inference"]
    summaries: list[dict[str, Any]] = []
    for stage_id in stages:
        base = StageContext.from_environment(stage_id=stage_id, config=cfg, run_id=job_id)
        if any(base.stage_dir.iterdir()):
            raise RuntimeError(f"Refusing to overwrite existing Stage F output: {base.stage_dir}")
        compat = SpecialistStageContext(base, cfg, stage_id)
        compat.release = release
        compat.release_id = release_id
        compat.release_manifest = release_manifest
        compat.foundation_job_id = foundation_job_id
        compat.input_records = list(base.inputs)
        module = __import__(f"geogg_fresh.stages.s{stage_id[:2]}_{'evaluation' if stage_id.startswith('08') else 'inference'}", fromlist=["run"])
        try:
            module.run(compat)
            # Optional downstream limitations remain visible as quality
            # metadata; failed scientific checks still make the stage conditional.
            status = _inference_stage_status(stage_id, compat.checks, compat.limitations, compat.blockers)
            result = StageResult(
                stage_id, status, f"{stage_id} completed.",
                warnings=compat.warnings,
                limitations=compat.limitations,
                quality_flags=compat.quality_flags,
                blockers=compat.blockers,
                checks=compat.checks,
                metrics={**compat.metrics, **compat.extra},
            )
        except StageBlocked as exc:
            result = StageResult(
                stage_id, StageStatus.BLOCKED, str(exc),
                limitations=compat.limitations,
                quality_flags=compat.quality_flags,
                blockers=[str(exc)],
                checks=compat.checks,
            )
        except Exception:
            error = traceback.format_exc()
            (base.stage_dir / "traceback.txt").write_text(error, encoding="utf-8")
            raise
        outputs = [register_output(base.stage_dir, p, role=p.parent.name or "artifact")
                   for p in sorted(base.stage_dir.rglob("*")) if p.is_file() and p.name != "provenance.json"]
        result.outputs = outputs
        stage_inputs = [dict(item.__dict__) for item in base.inputs]
        known_paths = {str(item.get("path")) for item in stage_inputs}
        for input_path, metadata in compat.inputs.items():
            if input_path in known_paths:
                continue
            stage_inputs.append({
                "path": input_path,
                "role": metadata.get("role"),
                "sha256": metadata.get("sha256"),
                "size_bytes": metadata.get("bytes"),
                "source": "release_source_artifact",
            })
        write_stage_provenance(provenance_dir=base.layout.provenance, stage_id=stage_id, run_id=job_id,
                               config_hash=base.config_hash, result=result, inputs=stage_inputs)
        summaries.append({"stage_id": stage_id, **result.to_dict()})
        print(f"GeoGG {stage_id}: {result.status.value} — {result.summary}")
        if result.status == StageStatus.BLOCKED:
            break
    payload = {"operation": "geogg_stage_f_08_09", "status": "succeeded" if len(summaries) == 2 and
               all(x["status"] in {"COMPLETED", "CONDITIONAL"} for x in summaries) else "failed",
               "processing_job_id": job_id, "model_release_id": release_id, "stages": summaries}
    out = job_dir / "geogg_inference_summary.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return 0 if payload["status"] == "succeeded" else 2
