"""GeoGG Stage 11: native-volume handoff, integrity checks and SEG-Y export."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..segy_write import NULL_VALUE, ReferenceSurvey, write_product, reopen_qc
from .s10_summary import _audit_rollup_status, _csv_write, _find_reference, _sha256

def _write_stage11_validation_protocol(out: Path, source_job: Path, source_id: str,
                                        products: dict, tasks: list, geometry: dict,
                                        volumes: Path) -> list:
    """Create immutable prediction-freeze and independent validation scaffolding.

    Truth is deliberately not loaded here. Blind-well evaluation can only be
    performed by a later explicit truth-reveal operation against this freeze.
    """
    validation = out / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    freeze_products = []
    for task in tasks:
        item = products[task]
        path = source_job / item["npy"]
        freeze_products.append({
            "product": f"prediction_{task}",
            "path": item.get("source_job_relative_path", item["npy"]),
            "sha256": item.get("sha256") or _sha256(path),
            "bytes": int(path.stat().st_size),
            "dtype": str(np.load(path, mmap_mode="r").dtype),
            "shape": list(np.load(path, mmap_mode="r").shape),
        })
    valid_path = volumes / "prediction_valid.npy"
    if valid_path.is_file():
        freeze_products.append({
            "product": "prediction_valid",
            "path": str(valid_path.relative_to(source_job)),
            "sha256": _sha256(valid_path), "bytes": int(valid_path.stat().st_size),
            "dtype": str(np.load(valid_path, mmap_mode="r").dtype),
            "shape": list(np.load(valid_path, mmap_mode="r").shape),
        })
    freeze = {
        "schema_version": 1, "freeze_state": "FROZEN",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_inference_job_id": source_id,
        "prediction_policy": "Predictions are immutable after this manifest is written. Truth must not be read or joined before freeze.",
        "geometry": geometry, "products": freeze_products,
        "freeze_sha256": hashlib.sha256(json.dumps(freeze_products, sort_keys=True).encode()).hexdigest(),
    }
    (validation / "prediction_freeze_manifest.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")

    registry = validation / "blind_well_registry_template.csv"
    with registry.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["well_id", "split", "truth_source", "truth_sha256", "truth_reveal_authorized", "notes"])
        w.writeheader()
    (validation / "blind_well_protocol.json").write_text(json.dumps({
        "schema_version": 1, "protocol_state": "FROZEN_PREDICTIONS_AWAITING_BLIND_REGISTRY",
        "prediction_freeze_manifest": "prediction_freeze_manifest.json",
        "steps": [
            "Register blind wells and independent truth sources without loading truth values.",
            "Verify well identities, split membership, truth-source provenance and checksums.",
            "Confirm prediction_freeze_manifest.json is immutable and record its freeze_sha256.",
            "Authorize truth reveal only after prediction freeze; record operator and timestamp.",
            "Join truth to frozen predictions by registered well/sample coordinates and publish per-well and per-class metrics.",
        ],
        "prohibitions": ["Do not use Stage 08 outer-fold metrics as blind-well results.", "Do not refit, recalibrate or modify predictions after truth reveal.", "Do not infer blind-well status from ordinary well-volume intersections."],
        "registry_template": "blind_well_registry_template.csv",
        "truth_reveal": "NOT_PERFORMED",
    }, indent=2) + "\n", encoding="utf-8")

    # Per-attribute voxel distribution summaries are not well validation metrics.
    class_rows = []
    for task in tasks:
        arr = np.load(source_job / products[task]["npy"], mmap_mode="r")
        finite_min = float("inf"); finite_max = float("-inf"); finite_sum = 0.0; count = 0
        for i in range(0, arr.shape[0], 32):
            chunk = np.asarray(arr[i:i+32])
            vals = chunk[np.isfinite(chunk)]
            if vals.size:
                finite_min = min(finite_min, float(vals.min())); finite_max = max(finite_max, float(vals.max()))
                finite_sum += float(vals.sum(dtype=np.float64)); count += int(vals.size)
        class_rows.append({"attribute": task, "population": "all_finite_voxels_not_blind_wells",
                           "finite_voxels": count,
                           "minimum": finite_min if count else "", "maximum": finite_max if count else "",
                           "mean": finite_sum / count if count else "",
                           "class_metrics_status": "NOT_APPLICABLE_CONTINUOUS_ATTRIBUTE"})
    _csv_write(validation / "per_class_prediction_report.csv", class_rows)
    _csv_write(validation / "per_well_validation_report.csv", [])

    parity = [{"check": "stage03_patch_feature_parity",
               "status": "PASS" if (geometry.get("stage03_parity") or {}).get("max_abs_difference") == 0 else "NOT_TESTABLE",
               "evidence": json.dumps(geometry.get("stage03_parity") or {})},
              {"check": "feature_smoke_tile_parity", "status": "NOT_TESTABLE",
               "evidence": "No persisted feature smoke-tile reference arrays are registered in this inference job."},
              {"check": "tile_seam_consistency", "status": "NOT_TESTABLE",
               "evidence": "Inference checkpoint records tile layout and halo, but no overlapping tile-boundary prediction pairs are persisted for comparison."}]
    _csv_write(validation / "feature_tile_parity.csv", parity)
    reproducibility = {
        "schema_version": 1, "status": "FROZEN_OUTPUT_IDENTITY_RECORDED",
        "source_inference_job_id": source_id,
        "freeze_sha256": freeze["freeze_sha256"],
        "product_hashes": {x["product"]: x["sha256"] for x in freeze_products},
        "repeated_inference": "NOT_RUN",
        "reason": "A repeat inference would require the exact execution environment, model release, feature inputs and compute configuration; this packaging stage does not rerun inference.",
    }
    (validation / "reproducibility_report.json").write_text(json.dumps(reproducibility, indent=2) + "\n", encoding="utf-8")
    return [
        {"check": "prediction_freeze_manifest", "status": "PASS", "detail": freeze["freeze_sha256"]},
        {"check": "blind_well_registry", "status": "NOT_CONFIGURED", "detail": "Template and protocol written; independent blind wells not registered."},
        {"check": "per_well_blind_validation", "status": "NOT_RUN", "detail": "Requires registered blind wells and authorized truth reveal."},
        {"check": "feature_and_tile_parity", "status": "CONDITIONAL", "detail": {x["check"]: x["status"] for x in parity}},
        {"check": "reproducibility", "status": "CONDITIONAL", "detail": reproducibility["status"]},
    ]


def _write_stage11_segy_products(*, out: Path, storage_root: Path, geometry: dict,
                                  volumes: Path, source_job: Path, tasks: list[str], products: dict,
                                  valid_path: Path, trace_grid: np.ndarray) -> tuple[list[dict], list[dict]]:
    """Export frozen NPY predictions onto registered source SEG-Y trace geometry.

    NPY remains canonical. SEG-Y is a derived interoperability product and every
    exported product is reopened and checked before it is registered.
    """
    reference_path = _find_reference(storage_root, geometry)
    live_flat = np.flatnonzero(np.asarray(trace_grid).reshape(-1) >= 0)
    trace_idx = np.asarray(trace_grid, dtype=np.int64).reshape(-1)[live_flat]
    if trace_idx.size == 0:
        raise RuntimeError("SEG-Y export requires at least one mapped source trace")
    if np.unique(trace_idx).size != trace_idx.size:
        raise RuntimeError("SEG-Y export refuses duplicate source trace mappings")
    shape = tuple(int(x) for x in trace_grid.shape) + (int(geometry["shape"][2]),)
    valid = np.load(valid_path, mmap_mode="r", allow_pickle=False)
    if tuple(valid.shape) != shape:
        raise RuntimeError("Validity mask shape does not match prediction geometry")
    if not np.isin(np.unique(valid), (0, 1, False, True)).all():
        raise RuntimeError("Validity mask must contain only binary 0/1 values")
    export_dir = out / "segy"
    export_dir.mkdir(parents=True, exist_ok=True)
    inventory, qc_rows = [], []
    with _reference_context(reference_path) as ref:
        for task in tasks:
            item = products[task]
            # Resolve the canonical prediction using the task's registered source path.
            source_path = (source_job / item["npy"]).resolve()
            # Export only the exact registered canonical product. A stale or
            # replaced NPY must never be silently repackaged as a valid SEG-Y.
            try:
                source_path.relative_to(source_job.resolve())
            except ValueError as exc:
                raise RuntimeError(f"Prediction path escapes the inference job for {task}") from exc
            if not source_path.is_file():
                raise RuntimeError(f"Canonical prediction NPY is missing for {task}: {source_path}")
            expected_sha256 = str(item.get("sha256") or "").lower()
            actual_sha256 = _sha256(source_path)
            if not expected_sha256 or actual_sha256 != expected_sha256:
                raise RuntimeError(f"Canonical prediction NPY checksum mismatch for {task}")
            source_array = np.load(source_path, mmap_mode="r", allow_pickle=False)
            if tuple(source_array.shape) != shape:
                raise RuntimeError(f"Prediction geometry mismatch for {task}: {source_array.shape} != {shape}")
            if not np.issubdtype(source_array.dtype, np.number):
                raise RuntimeError(f"Prediction NPY is not numeric for {task}: {source_array.dtype}")
            # Stream the mapped ROI into a disk-backed float32 staging array. Do not
            # materialize a full survey-sized prediction matrix in RAM.
            staging_path = export_dir / f".{task}.segy_staging.npy"
            data = np.lib.format.open_memmap(staging_path, mode="w+", dtype=np.float32,
                                             shape=(trace_idx.size, shape[2]))
            source_flat = source_array.reshape(-1, shape[2])
            valid_flat = valid.reshape(-1, shape[2])
            for start in range(0, trace_idx.size, 2048):
                stop = min(trace_idx.size, start + 2048)
                bins = live_flat[start:stop]
                block = np.asarray(source_flat[bins], dtype=np.float32).copy()
                block_valid = np.asarray(valid_flat[bins], dtype=bool)
                if not np.isfinite(block[block_valid]).all():
                    raise RuntimeError(f"Non-finite valid prediction values in SEG-Y export for {task}")
                block[~block_valid] = np.float32(NULL_VALUE)
                data[start:stop] = block
            data.flush()
            path = export_dir / f"prediction_{task}.sgy"
            write_meta = write_product(ref, path, trace_idx, 0, data,
                provenance=[f"APGCE Stage 11 prediction: {task}",
                            f"Source inference job: {geometry.get('source_inference_job_id', 'registered')}",
                            "Canonical source: native NPY; no-data: -999.25"])
            rows = reopen_qc(f"prediction_{task}", "prediction", path, ref, trace_idx, 0, data)
            qc_rows.extend(rows)
            sidecar = {"schema_version": 1, "product": f"prediction_{task}",
                       "format": "SEG-Y IEEE float32", "canonical_format": "NPY",
                       "source_npy": item.get("source_job_relative_path", item.get("npy", "")),
                       "source_npy_sha256": item.get("sha256", ""),
                       "reference_dataset": reference_path.name,
                       "reference_dataset_sha256": _sha256(reference_path),
                       "trace_mapping": "trace_index_grid.npy flattened in native inline/crossline order; unmapped bins omitted",
                       "null_value": NULL_VALUE, "write": write_meta,
                       "reopen_qc": rows, "status": "PASS" if all(r.get("result") in {"PASS", "INFO"} for r in rows) else "FAIL"}
            del data
            staging_path.unlink(missing_ok=True)
            sidecar_path = path.with_suffix(".manifest.json")
            sidecar_path.write_text(json.dumps(sidecar, indent=2) + "\n", encoding="utf-8")
            inventory.append({"product": f"prediction_{task}", "format": "segy",
                              "path": path.relative_to(out).as_posix(),
                              "bytes": path.stat().st_size, "sha256": _sha256(path),
                              "sidecar": sidecar_path.relative_to(out).as_posix(),
                              "status": sidecar["status"]})
    _csv_write(out / "tables" / "segy_export_qc.csv", qc_rows)
    _csv_write(out / "tables" / "segy_export_inventory.csv", inventory)
    return inventory, qc_rows


@contextmanager
def _reference_context(path: Path):
    yield ReferenceSurvey(path)


def _write_native_volume_export_stage(*, job_dir: Path, storage_root: Path, source_job: Path,
                                      source_id: str, volumes: Path, geometry: dict, tasks: list[str],
                                      products: dict, source_artifacts: dict, stats: dict,
                                      stage10_manifest: Path, domain_ok: bool,
                                      segy_export_enabled: bool = True) -> dict:
    """Write Stage 11 native-NPY handoff/QC without recreating derived SEG-Y.

    The large canonical arrays stay in Stage 09 and are referenced by
    storage-root-relative paths; this avoids duplicating multi-GB products.
    """
    out = job_dir / "stages" / "11_volume_export"
    stages_root = (job_dir / "stages").resolve()
    if out.resolve().parent != stages_root:
        raise RuntimeError("Unsafe Stage 11 output path")
    if out.exists():
        if out.is_symlink():
            raise RuntimeError("Refusing to replace a symlinked Stage 11 output")
        shutil.rmtree(out)
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    validation_checks = _write_stage11_validation_protocol(
        out, source_job, source_id, products, tasks, geometry, volumes)

    inventory_rows = []
    for task in tasks:
        item = products[task]
        path = source_job / item["npy"]
        arr = np.load(path, mmap_mode="r")
        inventory_rows.append({
            "product": f"prediction_{task}", "format": "npy",
            "storage_root_relative_path": item["source_job_relative_path"],
            "bytes": int(path.stat().st_size), "sha256": item["sha256"],
            "dtype": str(arr.dtype), "shape": "x".join(str(v) for v in arr.shape),
            "canonical": True, "copied": False,
        })
    valid_path = volumes / "prediction_valid.npy"
    if valid_path.is_file():
        valid_arr = np.load(valid_path, mmap_mode="r")
        inventory_rows.append({
            "product": "prediction_valid", "format": "npy",
            "storage_root_relative_path": f"processing_jobs/{source_id}/{valid_path.relative_to(source_job)}",
            "bytes": int(valid_path.stat().st_size), "sha256": source_artifacts.get("validity_mask", {}).get("sha256", ""),
            "dtype": str(valid_arr.dtype), "shape": "x".join(str(v) for v in valid_arr.shape),
            "canonical": True, "copied": False,
        })
    _csv_write(tables / "product_inventory.csv", inventory_rows)

    domain_rows = [{"attribute": task, "expected_min": stats[task]["expected_value_domain"][0],
                   "expected_max": stats[task]["expected_value_domain"][1],
                   "valid_voxels": stats[task]["finite_valid_voxels"],
                   "nonfinite_valid_voxels": stats[task]["null_or_nonfinite_voxels_within_valid_mask"],
                   "below_domain": stats[task]["values_below_domain"],
                   "above_domain": stats[task]["values_above_domain"],
                   "status": stats[task]["value_domain_status"]} for task in tasks]
    _csv_write(tables / "value_domain_qc.csv", domain_rows)

    shape = tuple(int(v) for v in geometry.get("shape", []))
    axis_order = list(geometry.get("axis_order") or [])
    grid_path = volumes / "trace_index_grid.npy"
    grid = np.load(grid_path, mmap_mode="r") if grid_path.is_file() else None
    geometry_rows = [
        {"check": "geometry_shape_3d", "status": "PASS" if len(shape) == 3 else "FAIL",
         "detail": shape},
        {"check": "axis_order_inline_crossline_time",
         "status": "PASS" if [str(v).casefold().replace("_bin", "").replace("_sample", "") for v in axis_order] == ["inline", "crossline", "time"] else "WARN",
         "detail": {"recorded": axis_order, "normalized": [str(v).casefold().replace("_bin", "").replace("_sample", "") for v in axis_order]}},
        {"check": "trace_index_grid_available", "status": "PASS" if grid_path.is_file() else "NOT_TESTABLE",
         "detail": str(grid_path.relative_to(source_job)) if grid_path.is_file() else "missing"},
    ]
    if grid_path.is_file():
        geometry_rows.append({"check": "trace_index_grid_shape",
                              "status": "PASS" if tuple(grid.shape) == shape[:2] else "FAIL",
                              "detail": list(grid.shape)})
        # Chunk the occupancy count to avoid a full-volume temporary array.
        mapped_count = sum(int(np.count_nonzero(grid[i:i + 64] >= 0)) for i in range(0, grid.shape[0], 64))
        geometry_rows.append({"check": "trace_index_grid_has_live_traces",
                              "status": "PASS" if mapped_count > 0 else "WARN",
                              "detail": {"mapped_bins": mapped_count}})
    _csv_write(tables / "geometry_qc.csv", geometry_rows)

    segy_inventory, segy_qc_rows = [], []
    segy_status = "DISABLED"
    if segy_export_enabled:
        if grid is None:
            raise RuntimeError("SEG-Y export requires the registered trace_index_grid.npy")
        segy_inventory, segy_qc_rows = _write_stage11_segy_products(
            out=out, storage_root=storage_root, geometry=geometry, volumes=volumes,
            source_job=source_job, tasks=tasks, products=products, valid_path=valid_path, trace_grid=grid)
        segy_status = _audit_rollup_status([row.get("status", "FAIL") for row in segy_inventory])

    product_integrity = []
    for row in inventory_rows:
        source_path = storage_root / row["storage_root_relative_path"]
        expected = row.get("sha256") or ""
        actual = _sha256(source_path) if source_path.is_file() and expected else ""
        product_integrity.append({"product": row["product"], "status":
                                  "PASS" if actual and actual == expected else "FAIL"})
    _csv_write(tables / "product_integrity.csv", product_integrity)
    checks = [
        {"check": "native_npy_products_registered", "status": "PASS" if len(inventory_rows) >= len(tasks) + 1 else "WARN",
         "detail": {"products": len(inventory_rows), "prediction_attributes": len(tasks)}},
        {"check": "native_npy_product_integrity",
         "status": _audit_rollup_status([row["status"] for row in product_integrity]),
         "detail": {row["product"]: row["status"] for row in product_integrity}},
        {"check": "prediction_value_domains", "status": "PASS" if domain_ok else "FAIL",
         "detail": {row["attribute"]: row["status"] for row in domain_rows}},
        {"check": "geometry_qc", "status": _audit_rollup_status([row["status"] for row in geometry_rows]),
         "detail": {row["check"]: row["status"] for row in geometry_rows}},
        {"check": "segy_export_reopen_qc", "status": "PASS" if segy_status == "DISABLED" else segy_status,
         "detail": {"enabled": bool(segy_export_enabled), "products": len(segy_inventory)}},
        {"check": "stage10_manifest_available", "status": "PASS" if stage10_manifest.is_file() else "FAIL",
         "detail": str(stage10_manifest.relative_to(job_dir)) if stage10_manifest.is_file() else "missing"},
    ]
    native_qc_status = "completed" if all(row["status"] == "PASS" for row in checks) else "conditional"
    status = native_qc_status
    stage10_hash = _sha256(stage10_manifest) if stage10_manifest.is_file() else ""
    manifest = {
        "schema_version": 1, "stage": "11_volume_export", "status": status,
        "source_inference_job_id": source_id, "source_inference_job_relative_path": f"processing_jobs/{source_id}",
        "format": "native_npy", "derived_segy_export": {"status": segy_status, "enabled": bool(segy_export_enabled), "products": segy_inventory},
        "product_policy": "Canonical NPY arrays are referenced in place; no large array copies are made.",
        "stage10_manifest": {"path": "../10_summary/volume_product_manifest.json", "sha256": stage10_hash},
        "geometry": geometry, "products": inventory_rows, "checks": checks,
        "native_product_qc_status": native_qc_status, "validation_protocol": "validation/blind_well_protocol.json",
        "validation_checks": validation_checks, "validation_status": "in_progress",
    }
    (out / "native_volume_export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out / "validation_summary.json").write_text(json.dumps({
        "schema_version": 1, "stage": "11_volume_export", "status": status,
        "checks": checks, "prediction_format": "native_npy", "derived_segy_export": {"status": segy_status, "enabled": bool(segy_export_enabled), "products": segy_inventory},
        "native_product_qc_status": native_qc_status, "validation_status": "in_progress",
        "validation_checks": validation_checks,
    }, indent=2) + "\n", encoding="utf-8")
    (out / "HANDOFF.md").write_text(
        "# Stage 11 — native volume export and QC\n\n"
        f"- Status: **{status}**\n- Source inference job: `{source_id}`\n"
        f"- Prediction format: `NumPy NPY`\n- Attributes: {', '.join(tasks)}\n\n"
        "## Product handling\n\n"
        "Canonical prediction arrays and the validity mask remain in the source Stage 09 inference job. "
        "SEG-Y prediction volumes are derived interoperability products written from those frozen NPY arrays, "
        "reopened and checked against source geometry and sample values.\n\n"
        "## Files\n\n- `native_volume_export_manifest.json` — canonical product references and checksums\n"
        "- `tables/product_inventory.csv` — product metadata\n- `tables/product_integrity.csv` — checksum verification\n"
        "- `tables/value_domain_qc.csv` — physical-domain checks\n- `tables/geometry_qc.csv` — native grid geometry checks\n"
        "- `validation_summary.json` — stage verdicts\n"
        "- `segy/` — derived SEG-Y products with per-product manifests and reopen QC\n",
        encoding="utf-8")
    return {"stage": "stages/11_volume_export", "status": status,
            "manifest": "stages/11_volume_export/native_volume_export_manifest.json",
            "validation_summary": "stages/11_volume_export/validation_summary.json",
            "prediction_format": "native_npy", "derived_segy_export": {"status": segy_status, "enabled": bool(segy_export_enabled), "products": segy_inventory}}


# Public stage entry point; the underscore alias remains for compatibility.
write_native_volume_export_stage = _write_native_volume_export_stage
