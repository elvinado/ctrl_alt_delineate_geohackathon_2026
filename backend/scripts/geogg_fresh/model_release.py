"""Create immutable, portable GeoGG model releases in backend-managed storage."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .provenance import sha256_file


class ModelReleaseError(RuntimeError):
    """A trained model cannot be safely packaged as a reusable release."""


def _copy_required(source: Path, destination: Path, copied: dict[str, dict[str, Any]], root: Path) -> None:
    if not source.is_file():
        raise ModelReleaseError(f"Required release artifact is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    copied[destination.relative_to(root).as_posix()] = {
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def create_model_release(*, job_dir: Path, config: dict, run_id: str) -> dict[str, Any] | None:
    """Package final fitted models and their inference/training contracts.

    The release lives beside processing_jobs under the same configured backend
    storage root. It is created atomically and never overwrites an existing ID.
    """
    job_dir = job_dir.resolve()
    stages = job_dir / "stages"
    s03, s05, s06, s07 = (stages / name for name in
                          ("03_extraction", "05_training_tables", "06_feature_selection", "07_models"))
    registry_path = s07 / "tables" / "model_registry.csv"
    if not registry_path.is_file():
        raise ModelReleaseError("Stage 07 model_registry.csv is missing; cannot register a model release")

    with registry_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    successful = [row for row in rows if row.get("status") == "OK" and row.get("final_model_file")]
    if not successful:
        # A completed Stage 07 can legitimately produce no usable final models.
        return None

    model_files: list[tuple[Path, str]] = []
    for row in successful:
        rel = Path(row["final_model_file"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ModelReleaseError(f"Unsafe model path in registry: {rel}")
        source = s07 / rel
        if not source.is_file():
            raise ModelReleaseError(f"Registered final model is missing: {source}")
        model_files.append((source, rel.as_posix()))

    release_id = f"geogg-{run_id}"
    storage_root = job_dir.parent.parent
    release_root = (storage_root / "model_releases").resolve()
    if release_root.parent != storage_root:
        raise ModelReleaseError("Invalid model release storage path")
    release_root.mkdir(parents=True, exist_ok=True)
    final_dir = release_root / release_id
    if final_dir.exists():
        raise ModelReleaseError(f"Model release already exists and is immutable: {release_id}")

    temp_dir = Path(tempfile.mkdtemp(prefix=f".{release_id}.", dir=release_root))
    copied: dict[str, dict[str, Any]] = {}
    try:
        # Model files plus the exact fitted transforms and schemas used by Stage 07.
        for source, relative in model_files:
            _copy_required(source, temp_dir / relative, copied, temp_dir)
        required = [
            (registry_path, Path("training/model_registry.csv")),
            (s07 / "tables" / "outer_test_predictions.csv", Path("training/outer_test_predictions.csv")),
            (s07 / "tables" / "fit_audit.csv", Path("training/fit_audit.csv")),
            (s07 / "architectures" / "model_architectures.json", Path("training/model_architectures.json")),
            (s05 / "feature_schema.json", Path("contracts/feature_schema.json")),
            (s05 / "training_manifest.json", Path("training/training_manifest.json")),
            (s05 / "tables" / "split_plan_wells.csv", Path("training/split_plan_wells.csv")),
            (s06 / "feature_sets.json", Path("preprocessing/feature_sets.json")),
        ]
        for source, relative in required:
            _copy_required(source, temp_dir / relative, copied, temp_dir)

        # Include all fitted Stage 06 transform JSONs, preserving resolution names.
        transform_dir = s06 / "transforms"
        transform_files = sorted(transform_dir.glob("transforms_*ms.json")) if transform_dir.is_dir() else []
        if not transform_files:
            raise ModelReleaseError("Stage 06 fitted transform files are missing")
        for source in transform_files:
            _copy_required(source, temp_dir / "preprocessing" / "transforms" / source.name, copied, temp_dir)

        # Preserve patch-generation contract, not the potentially very large training patch array.
        extraction_meta = s03 / "tables" / "extraction_metadata.csv"
        _copy_required(extraction_meta, temp_dir / "contracts/extraction_metadata.csv", copied, temp_dir)
        patch_manifests = sorted((s03 / "patches").glob("patch_manifest_*.json")) if (s03 / "patches").is_dir() else []
        if patch_manifests:
            for source in patch_manifests:
                _copy_required(source, temp_dir / "contracts/patches" / source.name, copied, temp_dir)

        config_path = config.get("_runtime", {}).get("config_path")
        if config_path and Path(config_path).is_file():
            _copy_required(Path(config_path), temp_dir / "config/training.yaml", copied, temp_dir)
        else:
            # A normalized snapshot remains portable even if the original YAML was moved.
            snapshot = {k: v for k, v in config.items() if not str(k).startswith("_")}
            snapshot_path = temp_dir / "config/training_config.json"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_text(json.dumps(snapshot, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            copied[snapshot_path.relative_to(temp_dir).as_posix()] = {
                "bytes": snapshot_path.stat().st_size, "sha256": sha256_file(snapshot_path)}

        models = []
        for row in successful:
            models.append({
                "task": row.get("task"), "input": row.get("input"), "model": row.get("model"),
                "model_file": Path(row["final_model_file"]).as_posix(),
                "final_params": row.get("final_params"),
                "final_threshold": row.get("final_threshold"),
                "scaling_mode": row.get("scaling_mode"),
            })
        manifest = {
            "schema_version": 1,
            "model_release_id": release_id,
            "status": "ready",
            "pipeline": "geogg_fresh",
            "pipeline_version": "v82",
            "processing_job_id": job_dir.name,
            # Authoritative origin link: this exact foundation job created the release.
            "foundation_job_id": job_dir.name,
            "training_run_id": run_id,
            "config_sha256": config.get("_runtime", {}).get("config_sha256"),
            "model_family": "deep_and_tabular" if any(Path(m["model_file"]).suffix == ".joblib" for m in models) and
                            any(Path(m["model_file"]).suffix == ".pt" for m in models) else
                            "deep" if any(Path(m["model_file"]).suffix == ".pt" for m in models) else "tabular",
            "models": models,
            "training_stages": ["00_inventory", "01_well_ties", "02_labels", "03_extraction",
                                "04_features", "05_training_tables", "06_feature_selection", "07_models"],
            "contracts": {
                "feature_schema": "contracts/feature_schema.json",
                "extraction_metadata": "contracts/extraction_metadata.csv",
                "feature_sets": "preprocessing/feature_sets.json",
                "transforms_directory": "preprocessing/transforms/",
                "patch_contract_directory": "contracts/patches/",
            },
            "provenance": {
                "source_processing_job_id": job_dir.name,
                "source_job_relative_path": f"processing_jobs/{job_dir.name}",
                "immutable_release": True,
            },
            "files": copied,
        }
        manifest_path = temp_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        # Verify every copied payload before making the release visible.
        for relative, details in copied.items():
            target = temp_dir / relative
            if not target.is_file() or target.stat().st_size != details["bytes"] or sha256_file(target) != details["sha256"]:
                raise ModelReleaseError(f"Release integrity check failed: {relative}")
        os.replace(temp_dir, final_dir)
        return {
            "model_release_id": release_id,
            "status": "ready",
            "path": str(final_dir),
            "model_count": len({item["model_file"] for item in models}),
            "manifest": str(final_dir / "manifest.json"),
        }
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
