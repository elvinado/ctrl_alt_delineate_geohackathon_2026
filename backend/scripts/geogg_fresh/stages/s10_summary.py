"""GeoGG Stage 10: audit and summarise registered prediction volumes."""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import json
import os
import platform
import re
import shutil
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from ..segy import SegyFile


_SHA256_CACHE: dict[tuple[str, int, int, int, int], str] = {}

def _sha256(path: Path) -> str:
    """Hash a file once per unchanged filesystem identity during this run."""
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns), int(getattr(stat, "st_ino", 0)))
    cached = _SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    digest = h.hexdigest()
    _SHA256_CACHE[key] = digest
    return digest


def _find_reference(storage_root: Path, geometry: dict) -> Path:
    foundation_id = str(geometry.get('foundation_job_id') or '').strip()
    if not foundation_id or Path(foundation_id).name != foundation_id:
        raise RuntimeError('Stage F geometry does not contain a safe foundation_job_id')
    jobs_root = (storage_root / 'processing_jobs').resolve()
    foundation = jobs_root / foundation_id
    if foundation.is_symlink() or foundation.parent != jobs_root or not foundation.is_dir():
        raise RuntimeError(f'Foundation processing job is unavailable: {foundation_id}')
    manifest_path = foundation / 'stages' / '00_inventory' / 'data_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('Foundation data_manifest.json is missing')
    manifest = json.loads(manifest_path.read_text())
    full_meta = (geometry.get('input_datasets') or {}).get('full') or {}
    wanted = str(full_meta.get('dataset_id') or '')
    records = manifest.get("inputs") or []
    candidates = []
    for row in records:
        if not isinstance(row, dict) or str(row.get("kind", "")).casefold() != "seismic":
            continue
        if wanted and str(row.get("dataset_id") or "") != wanted:
            continue
        role_text = " ".join(str(row.get(k) or "") for k in ("name", "path", "metadata")).casefold()
        if not wanted and not any(x in role_text for x in ("full_stack", "full stack", "full-stack")):
            continue
        raw_path = row.get("path")
        if raw_path:
            p = Path(raw_path).expanduser().resolve()
            if p.is_file():
                expected = str(row.get("sha256") or "").lower()
                if expected and _sha256(p) != expected:
                    raise RuntimeError("Reference full-stack SEG-Y checksum differs from foundation inventory")
                candidates.append(p)
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) != 1:
        raise RuntimeError(f'Expected one registered full-stack SEG-Y reference; found {len(candidates)}')
    return candidates[0]



def _safe_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _csv_write(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _audit_rollup_status(statuses: list[str]) -> str:
    """Reduce artifact-level states without treating missing evidence as success."""
    if not statuses:
        return "NOT_TESTABLE"
    if any(value in {"FAIL", "MISSING", "CHANGED"} for value in statuses):
        return "FAIL"
    if any(value in {"WARN", "NOT_HASHED", "NOT_RECORDED", "NOT_RETESTABLE", "UNAVAILABLE_EXTERNAL"} for value in statuses):
        return "WARN"
    if all(value == "PASS" for value in statuses):
        return "PASS"
    return "NOT_TESTABLE"


def _stage_execution_status(value: object) -> str:
    status = str(value or "NOT_RECORDED").strip().upper().replace("-", "_").replace(" ", "_")
    if status in {"COMPLETED", "COMPLETE", "SUCCEEDED", "SUCCESS", "DONE"}:
        return "PASS"
    if status in {"FAILED", "FAILURE", "ERROR", "CANCELLED", "CANCELED"}:
        return "FAIL"
    if status in {"NOT_RECORDED", "MISSING", "UNKNOWN", ""}:
        return "NOT_TESTABLE"
    return "WARN"


def _safe_artifact_label(value: object) -> str:
    """Create a stable, path-safe filename component from manifest metadata."""
    label = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown"))
    return label.strip("._")[:100] or "unknown"


def _load_dataset_index(*job_dirs: Path) -> dict[str, dict]:
    """Read persisted input registrations so old absolute paths can be resolved portably."""
    index: dict[str, dict] = {}
    for job_dir in job_dirs:
        if not job_dir:
            continue
        payload = _safe_json(job_dir / "inputs.json")
        for item in payload.get("inputs", []) if isinstance(payload.get("inputs"), list) else []:
            if isinstance(item, dict) and item.get("dataset_id"):
                index[str(item["dataset_id"])] = item
    return index


def _resolve_recorded_path(raw: object, *, provenance_path: Path | None, owner_job: Path,
                           stage_id: str | None = None, dataset_id: str | None = None,
                           dataset_index: dict[str, dict] | None = None) -> Path | None:
    """Resolve provenance paths across stage-relative, job-relative and historical absolute forms.

    Stage outputs in the specialist provenance schema are relative to their stage
    directory (not the processing-job root). Older records may instead be job- or
    storage-root-relative. Absolute paths from a previous host are remapped only
    when they contain the stable processing_jobs/<job-id>/ segment.
    """
    if raw is None or not str(raw).strip():
        return None
    storage_root = owner_job.parent.parent
    raw_text = str(raw).strip()
    candidate = Path(raw_text).expanduser()

    def usable(path: Path, boundary: Path | None = None) -> bool:
        try:
            if path.is_symlink() or not path.is_file():
                return False
            resolved = path.resolve(strict=True)
            if boundary is not None:
                resolved.relative_to(boundary.resolve())
            return True
        except (OSError, ValueError):
            return False

    absolute_like = candidate.is_absolute() or bool(re.match(r"^[A-Za-z]:[\\\\/]", raw_text)) or raw_text.startswith("\\\\")
    if absolute_like:
        # First honour the exact path if it is still valid on this host.
        if candidate.is_absolute() and usable(candidate):
            return candidate
        # Remap stale POSIX or Windows host paths through the stable backend job identity.
        normalized = re.sub(r"/+", "/", raw_text.replace("\\", "/"))
        match = re.search(r"(?:^|/)processing_jobs/([^/]+)/(.+)$", normalized)
        if match:
            job_id, relative = match.groups()
            if job_id not in {"", ".", ".."} and Path(job_id).name == job_id:
                jobs_root = storage_root / "processing_jobs"
                remapped_job = jobs_root / job_id
                remapped = remapped_job / Path(relative)
                if usable(remapped, remapped_job):
                    return remapped
        # Managed uploads are portable by dataset ID/storage key. Linked inputs
        # remain external references and are classified as not re-testable below.
        item = (dataset_index or {}).get(str(dataset_id or ""))
        if item and str(item.get("storage_mode") or "").casefold() == "managed":
            storage_key = str(item.get("storage_key") or "")
            if storage_key:
                managed = storage_root / storage_key
                if usable(managed, storage_root):
                    return managed
        return candidate

    # Specialist stage outputs are relative to stages/<stage_id>/.
    options: list[tuple[Path, Path]] = []
    if stage_id:
        options.append((owner_job / "stages" / stage_id / candidate, owner_job))
        options.append((owner_job / stage_id / candidate, owner_job))
    options.append((owner_job / candidate, owner_job))
    options.append((storage_root / candidate, storage_root))
    for path, boundary in options:
        if usable(path, boundary):
            return path
    # Return the most specific expected location for an accurate MISSING record.
    return options[0][0] if options else owner_job / candidate


def _recorded_dataset_id(entry: object) -> str:
    if not isinstance(entry, dict):
        return ""
    direct = str(entry.get("dataset_id") or "").strip()
    if direct:
        return direct
    role = str(entry.get("role") or "")
    match = re.search(r"(?:^|[;\s])dataset_id=([A-Za-z0-9-]+)", role)
    return match.group(1) if match else ""


def _is_unavailable_linked_input(dataset_id: str | None,
                                 dataset_index: dict[str, dict] | None) -> bool:
    item = (dataset_index or {}).get(str(dataset_id or ""))
    return bool(item and str(item.get("storage_mode") or "").casefold() == "linked")


def _resolve_release_manifest(storage_root: Path, geometry: dict, inference: dict) -> tuple[str, dict, list[dict]]:
    release_id = str(inference.get("release_id") or inference.get("model_release_id") or "").strip()
    if not release_id or Path(release_id).name != release_id or release_id in {".", ".."}:
        return release_id, {}, [{"check": "model_release_manifest_available", "status": "NOT_TESTABLE",
                                  "detail": "Stage F summary does not identify a model release"}]
    release_root = (storage_root / "model_releases").resolve()
    release = release_root / release_id
    if release_root.parent != storage_root or release.is_symlink() or not release.is_dir() or release.resolve().parent != release_root:
        return release_id, {}, [{"check": "model_release_manifest_available", "status": "FAIL",
                                  "detail": f"Model release is unavailable or unsafe: {release_id}"}]
    manifest_path = release / "manifest.json"
    manifest = _safe_json(manifest_path)
    checks = [{"check": "model_release_manifest_available", "status": "PASS" if manifest else "FAIL",
               "detail": str(manifest_path)}]
    if manifest.get("model_release_id") != release_id or manifest.get("status") != "ready":
        checks.append({"check": "model_release_identity", "status": "FAIL", "detail": "Release identity/status mismatch"})
    else:
        checks.append({"check": "model_release_identity", "status": "PASS", "detail": release_id})
    files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    bad = []
    for relative, metadata in files.items():
        rel = Path(relative)
        target = release / rel
        if rel.is_absolute() or ".." in rel.parts or target.is_symlink() or not target.is_file() or release.resolve() not in target.resolve().parents:
            bad.append(relative)
            continue
        if target.stat().st_size != int(metadata.get("bytes", -1)) or _sha256(target) != metadata.get("sha256"):
            bad.append(relative)
    checks.append({"check": "immutable_model_release_files", "status": "PASS" if files and not bad else "FAIL",
                   "detail": {"files_checked": len(files), "failed_files": bad}})
    expected_foundation = str(geometry.get("foundation_job_id") or "")
    actual_foundation = str(manifest.get("foundation_job_id") or (manifest.get("provenance") or {}).get("source_processing_job_id") or "")
    checks.append({"check": "release_foundation_job_matches_geometry",
                   "status": "PASS" if expected_foundation and expected_foundation == actual_foundation else "FAIL",
                   "detail": {"geometry_foundation_job_id": expected_foundation,
                              "release_foundation_job_id": actual_foundation}})
    return release_id, manifest, checks


def _collect_pipeline_status(storage_root: Path, source_job: Path, foundation_id: str) -> list[dict]:
    """Collect only recorded provenance; never infer missing stages as completed."""
    rows = []
    foundation = storage_root / "processing_jobs" / foundation_id if foundation_id else None
    for stage_id in [f"{i:02d}_{name}" for i, name in enumerate(
            ["inventory", "well_ties", "labels", "extraction", "features", "training_tables",
             "feature_selection", "models", "evaluation", "inference"] )]:
        owner = source_job if stage_id in {"08_evaluation", "09_inference"} else foundation
        path = owner / "provenance" / f"{stage_id}.json" if owner else None
        record = _safe_json(path) if path else {}
        inputs = record.get("inputs", []) if isinstance(record.get("inputs"), list) else []
        outputs = record.get("outputs", []) if isinstance(record.get("outputs"), list) else []
        rows.append({"stage": stage_id, "processing_job_id": owner.name if owner else "",
                     "status": record.get("status", "NOT_RECORDED"),
                     "recorded_at": record.get("recorded_at", ""),
                     "started_at": record.get("started_at", ""),
                     "completed_at": record.get("completed_at", ""),
                     "config_sha256": record.get("config_sha256", ""),
                     "code_fingerprint": (record.get("code_fingerprint") or record.get("code_sha256") or
                                          record.get("source_code_sha256") or ""),
                     "code_version": record.get("code_version", record.get("git_commit", "")),
                     "provenance_exists": bool(path and path.is_file()),
                     "input_count": len(inputs), "output_count": len(outputs),
                     "warning_count": len(record.get("warnings", []) or []),
                     "error_count": len(record.get("errors", []) or []),
                     "provenance_file": str(path.relative_to(storage_root)) if path and path.is_file() else "",
                     "summary": record.get("summary", "")})
    return rows



def _normalize_volume_axis_order(axis_order: object) -> list[str]:
    """Normalize equivalent manifest axis labels without changing stored metadata."""
    if not isinstance(axis_order, (list, tuple)):
        return []
    aliases = {"inline_bin": "inline", "crossline_bin": "crossline", "time_sample": "time"}
    return [aliases.get(str(value).strip().casefold(), str(value).strip().casefold())
            for value in axis_order]


def _png_rgb_bytes(rgb: np.ndarray) -> bytes:
    """Encode an RGB uint8 array as PNG using only the Python standard library."""
    image = np.asarray(rgb, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("PNG input must have shape [height, width, 3]")
    height, width, _ = image.shape

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xffffffff)

    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, level=6))
            + chunk(b"IEND", b""))


def _seismic_rgb(values: np.ndarray) -> np.ndarray:
    """Robust symmetric grayscale rendering for seismic amplitudes."""
    data = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(data)
    if not finite.any():
        return np.full((*data.shape, 3), 128, dtype=np.uint8)
    scale = float(np.percentile(np.abs(data[finite]), 98))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    normalized = np.clip(np.nan_to_num(data / scale, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
    gray = np.rint(127.5 + normalized * 127.5).astype(np.uint8)
    gray[~finite] = 128
    return np.repeat(gray[..., None], 3, axis=2)


def _attribute_rgb(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Compact blue-cyan-yellow-red sequential map for bounded [0, 1] attributes."""
    data = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(data)
    x = np.clip(np.nan_to_num(data, nan=0.0), 0.0, 1.0)
    stops = np.asarray([[35, 55, 130], [30, 150, 180], [235, 220, 80], [180, 35, 45]], dtype=np.float32)
    position = x * (len(stops) - 1)
    lower = np.floor(position).astype(int).clip(0, len(stops) - 2)
    fraction = (position - lower)[..., None]
    rgb = stops[lower] * (1.0 - fraction) + stops[lower + 1] * fraction
    rgb = np.rint(rgb).astype(np.uint8)
    rgb[~mask] = np.asarray([220, 220, 220], dtype=np.uint8)
    return rgb


def _write_section_figure(path: Path, amplitude: np.ndarray, prediction: np.ndarray,
                          valid: np.ndarray, well_points: list[tuple[int, int]] | None = None) -> dict:
    """Write a two-panel PNG (seismic amplitude and prediction) with time vertically."""
    amp = np.asarray(amplitude, dtype=np.float32)
    pred = np.asarray(prediction, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool)
    if amp.shape != pred.shape or pred.shape != mask.shape or amp.ndim != 2:
        raise RuntimeError("Section amplitude, prediction and validity arrays must share a 2-D shape")
    # Input arrays are [horizontal, time]; display with time increasing downwards.
    left = _seismic_rgb(amp.T)
    right = _attribute_rgb(pred.T, mask.T)
    spacer = np.full((left.shape[0], 12, 3), 245, dtype=np.uint8)
    canvas = np.concatenate([left, spacer, right], axis=1)
    # Overlay well TWT samples on both seismic and prediction panels. The
    # coordinates are native IL/XL bins and sample indices from Stage 01.
    for horizontal_index, sample_index in well_points or []:
        if not (0 <= horizontal_index < amp.shape[0] and 0 <= sample_index < amp.shape[1]):
            continue
        row = int(sample_index)
        col = int(horizontal_index)
        for panel_offset in (0, left.shape[1] + spacer.shape[1]):
            x0, y0 = panel_offset + col, row
            y1, y2 = max(0, y0 - 3), min(canvas.shape[0], y0 + 4)
            x1, x2 = max(panel_offset, x0 - 3), min(panel_offset + left.shape[1], x0 + 4)
            canvas[y1:y2, x1:x2] = np.asarray([255, 40, 40], dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_rgb_bytes(canvas))
    return {"path": path.as_posix(), "width": int(canvas.shape[1]), "height": int(canvas.shape[0]),
            "horizontal_bins": int(amp.shape[0]), "time_samples": int(amp.shape[1]),
            "amplitude_display": "symmetric p98 grayscale", "prediction_display": "bounded 0-1 sequential",
            "invalid_prediction_display": "light grey"}


def _build_seismic_sections(*, storage_root: Path, geometry: dict, source_id: str,
                            source_job: Path, volumes: Path, out: Path, tasks: list[str],
                            shape: tuple[int, int, int], valid: np.ndarray,
                            trace_grid: np.ndarray) -> dict:
    """Create inline, crossline and time-slice sections from the registered full stack.

    Section coordinates are native IL/XL bins. XY coordinates are deliberately not
    synthesized because the registered survey may have non-affine XY geometry.
    """
    reference = _find_reference(storage_root, geometry)
    il_min, il_step = int(geometry["il_min"]), int(geometry["il_step"])
    xl_min, xl_step = int(geometry["xl_min"]), int(geometry["xl_step"])
    t0, dt = float(geometry["t0_ms"]), float(geometry["dt_ms"])
    n_il, n_xl, ns = shape
    grid = np.asarray(trace_grid, dtype=np.int64)
    if grid.shape != (n_il, n_xl):
        raise RuntimeError("Trace-index grid does not match the inference ROI")
    if np.any(grid >= 0) and int(np.max(grid)) >= 2**63:
        raise RuntimeError("Invalid trace index in inference grid")

    def picks(length: int) -> list[int]:
        return sorted(set([0, length // 4, length // 2, (3 * length) // 4, length - 1]))

    il_picks, xl_picks = picks(n_il), picks(n_xl)
    time_picks = sorted(set([ns // 4, ns // 2, (3 * ns) // 4]))
    # Stage 01 already computes minimum-curvature trajectory positions and
    # MD/TVDSS-to-TWT samples tied to the registered full-stack geometry.
    # Reuse that evidence; do not independently invent a time-depth transform.
    foundation_id = str(geometry.get("foundation_job_id") or "")
    well_samples_path = storage_root / "processing_jobs" / foundation_id / "stages" / "01_well_ties" / "tables" / "tz_samples_all_wells.csv"
    well_samples: list[dict] = []
    if well_samples_path.is_file():
        with well_samples_path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                try:
                    il_value = float(row["il_nearest"]); xl_value = float(row["xl_nearest"])
                    sample_value = float(row["seismic_sample_frac"])
                    if not all(np.isfinite(v) for v in (il_value, xl_value, sample_value)):
                        continue
                    well_samples.append({"well": row.get("well", ""),
                        "il_index": int(round((il_value - il_min) / il_step)),
                        "xl_index": int(round((xl_value - xl_min) / xl_step)),
                        "sample_index": int(round(sample_value)),
                        "twt_ms": float(row.get("twt_ms") or (t0 + sample_value * dt)),
                        "md_m": row.get("md_m", ""), "tvdss_m": row.get("tvdss_m", ""),
                        "trace_offset_xy": row.get("trace_offset_xy", ""),
                        "in_seismic_time_range": str(row.get("in_seismic_time_range", "")).lower() in {"true", "1"},
                        "tz_extrapolation_m": row.get("tz_extrapolation_m", "")})
                except (KeyError, TypeError, ValueError):
                    continue
    well_overlay_rows = []
    well_intersection_rows = []
    well_counts: dict[str, dict] = {}
    prediction_arrays = {task: np.load(volumes / f"prediction_{task}.npy", mmap_mode="r") for task in tasks}
    for row in well_samples:
        ii, jj, kk = row["il_index"], row["xl_index"], row["sample_index"]
        inside = 0 <= ii < n_il and 0 <= jj < n_xl and 0 <= kk < ns
        trace_id = int(grid[ii, jj]) if inside else -1
        mapped = bool(inside and trace_id >= 0)
        is_valid = bool(mapped and valid[ii, jj, kk] == 1)
        item = {**row, "sample_inside_volume_grid": inside,
                "sample_has_trace_mapping": mapped, "trace_index": trace_id,
                "sample_has_valid_prediction": is_valid,
                # Retain the v119 names for downstream compatibility.
                "inside_volume_grid": inside, "prediction_valid": is_valid}
        for task, prediction_array in prediction_arrays.items():
            item[f"prediction_{task}"] = float(prediction_array[ii, jj, kk]) if is_valid else ""
        well_intersection_rows.append(item)
        well_name = str(row.get("well") or "(unnamed)")
        counts = well_counts.setdefault(well_name, {
            "well": well_name, "trajectory_sample_count": 0,
            "samples_inside_volume_grid": 0, "samples_with_trace_mapping": 0,
            "samples_with_valid_prediction": 0,
        })
        counts["trajectory_sample_count"] += 1
        counts["samples_inside_volume_grid"] += int(inside)
        counts["samples_with_trace_mapping"] += int(mapped)
        counts["samples_with_valid_prediction"] += int(is_valid)
    for counts in well_counts.values():
        if counts["samples_with_valid_prediction"]:
            status = "intersecting_valid_prediction"
        elif counts["samples_inside_volume_grid"]:
            status = "inside_grid_no_valid_prediction"
        else:
            status = "no_volume_grid_samples"
        counts["status"] = status
    well_status_rows = list(well_counts.values())
    # Load only the seismic samples required for the three time slices.
    time_amp = np.full((n_il, n_xl, len(time_picks)), np.nan, dtype=np.float32)
    flat_grid = grid.reshape(-1)
    with SegyFile(reference) as ref:
        if ref.ns != ns or abs(ref.dt_us / 1000.0 - dt) > 1e-6:
            raise RuntimeError("Reference SEG-Y time axis does not match inference volume")
        if np.any(flat_grid >= ref.n_traces):
            raise RuntimeError("Trace-index grid references a trace outside the registered full stack")
        live_indices = flat_grid[flat_grid >= 0]
        if np.unique(live_indices).size != live_indices.size:
            raise RuntimeError("Trace-index grid contains duplicate reference-trace mappings")
        reference_t0 = float(np.median(ref.scan_headers(fields=("delay_ms",), show_progress=False)["delay_ms"]))
        if abs(reference_t0 - t0) > 1e-6:
            raise RuntimeError("Reference SEG-Y delay/t0 does not match inference volume")
        live_flat = np.flatnonzero(flat_grid >= 0)
        for start in range(0, live_flat.size, 2048):
            bins = live_flat[start:start + 2048]
            traces = ref.read_traces(flat_grid[bins])
            time_amp.reshape(-1, len(time_picks))[bins] = traces[:, time_picks]

        # Read each source seismic section once, then reuse it across all
        # prediction attributes. Previously the same SEG-Y traces were read
        # again for every task, multiplying I/O by the number of attributes.
        inline_amplitudes = {i: ref.read_traces(grid[i, :]) for i in il_picks}
        crossline_amplitudes = {j: ref.read_traces(grid[:, j]) for j in xl_picks}
        well_by_inline: dict[int, list[dict]] = {i: [] for i in il_picks}
        well_by_crossline: dict[int, list[dict]] = {j: [] for j in xl_picks}
        for row in well_samples:
            if not row["in_seismic_time_range"] or not 0 <= row["sample_index"] < ns:
                continue
            if row["il_index"] in well_by_inline and 0 <= row["xl_index"] < n_xl:
                well_by_inline[row["il_index"]].append(row)
            if row["xl_index"] in well_by_crossline and 0 <= row["il_index"] < n_il:
                well_by_crossline[row["xl_index"]].append(row)

        figure_rows = []
        for task in tasks:
            prediction = prediction_arrays[task]
            for i in il_picks:
                amplitude = inline_amplitudes[i]
                pred = np.asarray(prediction[i, :, :], dtype=np.float32)
                mask = np.asarray(valid[i, :, :]) == 1
                il_number = il_min + i * il_step
                points = [(r["xl_index"], r["sample_index"]) for r in well_by_inline[i]]
                rel = Path("figures") / "sections" / task / f"inline_{il_number}.png"
                meta = _write_section_figure(out / rel, amplitude, pred, mask, points)
                meta["well_overlay_points"] = len(points)
                meta["path"] = rel.as_posix()
                figure_rows.append({"attribute": task, "section_type": "inline", "coordinate": il_number,
                                    "sample_index": "", "time_ms": "", **meta})
            for j in xl_picks:
                amplitude = crossline_amplitudes[j]
                pred = np.asarray(prediction[:, j, :], dtype=np.float32)
                mask = np.asarray(valid[:, j, :]) == 1
                xl_number = xl_min + j * xl_step
                points = [(r["il_index"], r["sample_index"]) for r in well_by_crossline[j]]
                rel = Path("figures") / "sections" / task / f"crossline_{xl_number}.png"
                meta = _write_section_figure(out / rel, amplitude, pred, mask, points)
                meta["well_overlay_points"] = len(points)
                meta["path"] = rel.as_posix()
                figure_rows.append({"attribute": task, "section_type": "crossline", "coordinate": xl_number,
                                    "sample_index": "", "time_ms": "", **meta})
            for k, sample_index in enumerate(time_picks):
                amp = time_amp[:, :, k]
                pred = np.asarray(prediction[:, :, sample_index], dtype=np.float32)
                mask = np.asarray(valid[:, :, sample_index]) == 1
                # A time slice is spatial, so do not transpose its native IL/XL axes.
                left, right = _seismic_rgb(amp), _attribute_rgb(pred, mask)
                canvas = np.concatenate([left, np.full((n_il, 12, 3), 245, dtype=np.uint8), right], axis=1)
                rel = Path("figures") / "sections" / task / f"time_slice_{sample_index:04d}.png"
                target = out / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(_png_rgb_bytes(canvas))
                figure_rows.append({"attribute": task, "section_type": "time_slice",
                                    "coordinate": "", "sample_index": sample_index,
                                    "time_ms": round(t0 + sample_index * dt, 6),
                                    "path": rel.as_posix(), "width": int(canvas.shape[1]),
                                    "height": int(canvas.shape[0]), "horizontal_bins": n_xl,
                                    "time_samples": 1, "amplitude_display": "symmetric p98 grayscale",
                                    "prediction_display": "bounded 0-1 sequential",
                                    "invalid_prediction_display": "light grey"})

        # Well overlays describe geometry/validity, not a specific attribute.
        # Emit each trajectory sample once rather than once per prediction task.
        for row in well_samples:
            ii, jj, kk = row["il_index"], row["xl_index"], row["sample_index"]
            if not row["in_seismic_time_range"] or not (0 <= ii < n_il and 0 <= jj < n_xl and 0 <= kk < ns):
                continue
            trace_index = int(grid[ii, jj])
            for section_type, coordinate, horizontal_index in (
                ("inline", il_min + ii * il_step, jj),
                ("crossline", xl_min + jj * xl_step, ii),
            ):
                selected = ii in il_picks if section_type == "inline" else jj in xl_picks
                if selected:
                    well_overlay_rows.append({**row, "section_type": section_type,
                        "section_coordinate": coordinate, "horizontal_bin_index": horizontal_index,
                        "trace_index": trace_index,
                        "prediction_valid": bool(trace_index >= 0 and valid[ii, jj, kk] == 1)})
        _csv_write(out / "tables" / "seismic_sections.csv", figure_rows)
        _csv_write(out / "tables" / "well_volume_intersections.csv", well_intersection_rows)
        _csv_write(out / "tables" / "well_section_overlays.csv", well_overlay_rows)
        _csv_write(out / "tables" / "well_intersection_status.csv", well_status_rows,
                   ["well", "status", "trajectory_sample_count", "samples_inside_volume_grid",
                    "samples_with_trace_mapping", "samples_with_valid_prediction"])
    if not well_samples:
        well_status = "no_trajectory_data"
    elif any(r["samples_with_valid_prediction"] for r in well_status_rows):
        well_status = "intersecting_valid_prediction"
    elif any(r["samples_inside_volume_grid"] for r in well_status_rows):
        well_status = "inside_grid_no_valid_prediction"
    else:
        well_status = "no_volume_grid_samples"
    payload = {
        "schema_version": 1, "source_inference_job_id": source_id,
        "reference_seismic": str(reference),
        "geometry": {"axis_order": "inline,crossline,time", "il_min": il_min, "il_step": il_step,
                     "xl_min": xl_min, "xl_step": xl_step, "t0_ms": t0, "dt_ms": dt, "shape": list(shape),
                     "xy_mapping_status": (geometry.get("spatial_reference") or {}).get("xy_mapping_status",
                                                                                         "not_recorded")},
        "inline_indices": [{"index": i, "inline": il_min + i * il_step} for i in il_picks],
        "crossline_indices": [{"index": j, "crossline": xl_min + j * xl_step} for j in xl_picks],
        "time_slices": [{"sample_index": k, "time_ms": round(t0 + k * dt, 6)} for k in time_picks],
        "section_count": len(figure_rows),
        "well_trajectory_overlay": ("generated_from_stage01_tz_samples" if well_samples else
                                    "not_available: Stage 01 tz_samples_all_wells.csv missing or contains no usable trajectory/TWT samples"),
        "well_intersection_status": well_status,
        "well_intersection_method": "Stage 01 registered IL/XL/TWT samples mapped to inference-grid indices; not a continuous 3D wellbore/volume geometry intersection",
        "well_trajectory_sample_count": len(well_samples),
        "well_count": len(well_status_rows),
        "well_status_counts": {status: sum(r["status"] == status for r in well_status_rows)
                                for status in ("intersecting_valid_prediction",
                                               "inside_grid_no_valid_prediction",
                                               "no_volume_grid_samples")},
        "well_section_overlay_count": len(well_overlay_rows),
        "well_volume_sample_count": len(well_intersection_rows),
        "well_samples_inside_volume_grid": sum(bool(r["sample_inside_volume_grid"]) for r in well_intersection_rows),
        "well_samples_with_trace_mapping": sum(bool(r["sample_has_trace_mapping"]) for r in well_intersection_rows),
        "well_valid_prediction_intersections": sum(bool(r["prediction_valid"]) for r in well_intersection_rows),
        "well_overlay_source": str(well_samples_path) if well_samples else None,
    }
    (out / "seismic_sections.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload



_EVALUATION_TABLES = (
    "model_performance.csv", "per_well_metrics.csv", "per_class_metrics.csv",
    "paired_comparisons.csv", "beyond_depth.csv", "conformal_intervals.csv",
    "calibration_bins.csv", "errors_by_interval.csv", "outer_oof_predictions.csv",
)
_EVALUATION_FIGURES = (
    "model_comparison_outer.png", "per_well_metrics.png", "beyond_depth.png",
    "calibration.png", "true_vs_predicted.png", "roc_pr_curves.png",
    "confusion_matrices.png",
)

def _evaluation_stage_dir(source_job: Path, storage_root: Path, foundation_id: str, release_id: str) -> Path | None:
    """Find the evaluation stage whose recorded release matches this inference."""
    candidates = [source_job / "stages" / "08_evaluation"]
    if foundation_id:
        candidates.append(storage_root / "processing_jobs" / foundation_id / "stages" / "08_evaluation")
    matching = []
    for candidate in candidates:
        if not candidate.is_dir():
            continue
        summary_path = candidate / "evaluation_summary.json"
        recorded = _safe_json(summary_path)
        recorded_release = str(recorded.get("release_id") or recorded.get("model_release_id") or "")
        if recorded_release and release_id and recorded_release == release_id:
            matching.append(candidate)
    if len(matching) > 1:
        # The source inference job is the authoritative owner when both paths
        # expose the same release-bound evaluation snapshot.
        return matching[0]
    return matching[0] if matching else None

def _read_csv_rows(path: Path) -> tuple[list[str], list[dict]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)

def _csv_data_row_count(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return max(0, sum(1 for _ in csv.reader(stream)) - 1)

def _import_evaluation_evidence(*, source_job: Path, storage_root: Path, foundation_id: str,
                                release_id: str, release_manifest: dict, out: Path) -> dict:
    """Copy immutable Stage 08 evidence; never recalculate evaluation metrics."""
    stage_dir = _evaluation_stage_dir(source_job, storage_root, foundation_id, release_id)
    tables_out = out / "tables" / "evaluation_source"
    figures_out = out / "figures" / "evaluation"
    tables_out.mkdir(parents=True, exist_ok=True)
    figures_out.mkdir(parents=True, exist_ok=True)
    if stage_dir is None:
        _csv_write(out / "tables" / "evaluation_scorecard.csv", [], ["task", "input", "model"])
        _csv_write(out / "tables" / "evaluation_per_well_metrics.csv", [], ["well", "task", "input", "model"])
        _csv_write(out / "tables" / "evaluation_conformal_intervals.csv", [], ["task", "input", "model"])
        _csv_write(out / "tables" / "evaluation_evidence_inventory.csv", [],
                   ["artifact", "source_path", "output_path", "bytes", "source_sha256", "sha256", "rows", "status"])
        return {"status": "NOT_AVAILABLE", "source_stage": "", "files": [],
                "registered_model_rows": 0, "registered_per_well_rows": 0,
                "registered_conformal_rows": 0, "figure_paths": [],
                "limitation": "No Stage 08 evaluation summary matching the immutable model release was found."}

    inventory, copied_tables, figure_paths = [], {}, []
    for filename in ("evaluation_summary.json", *_EVALUATION_TABLES):
        source = stage_dir / "tables" / filename if filename.endswith(".csv") else stage_dir / filename
        if not source.is_file():
            continue
        target = tables_out / filename
        shutil.copy2(source, target)
        rows = 0
        if filename.endswith(".csv"):
            rows = _csv_data_row_count(target)
            # The outer OOF table can contain hundreds of thousands of rows.
            # Preserve it byte-for-byte without loading the entire file in RAM.
            if filename != "outer_oof_predictions.csv":
                _, copied_tables[filename] = _read_csv_rows(target)
        source_hash, copied_hash = _sha256(source), _sha256(target)
        if source_hash != copied_hash:
            raise RuntimeError(f"Stage 08 evaluation evidence copy failed checksum verification: {filename}")
        inventory.append({"artifact": filename, "source_path": source.relative_to(storage_root).as_posix(),
                          "output_path": str(target.relative_to(out)), "bytes": target.stat().st_size,
                          "source_sha256": source_hash, "sha256": copied_hash, "rows": rows,
                          "status": "COPIED_VERIFIED"})

    for filename in _EVALUATION_FIGURES:
        source = stage_dir / "figures" / filename
        if not source.is_file():
            continue
        target = figures_out / filename
        shutil.copy2(source, target)
        figure_paths.append(str(target.relative_to(out)))
        inventory.append({"artifact": filename, "source_path": source.relative_to(storage_root).as_posix(),
                          "output_path": str(target.relative_to(out)), "bytes": target.stat().st_size,
                          "sha256": _sha256(target), "rows": "", "status": "COPIED_VERIFIED"})

    models = release_manifest.get("models", []) if isinstance(release_manifest.get("models"), list) else []
    registered = {(str(row.get("task", "")), str(row.get("input", "")), str(row.get("model", "")))
                  for row in models if isinstance(row, dict)}
    performance = copied_tables.get("model_performance.csv", [])
    matched = [row for row in performance if (row.get("task", ""), row.get("input", ""), row.get("model", "")) in registered]
    metric_fields = list(performance[0].keys()) if performance else ["task", "input", "model"]
    _csv_write(out / "tables" / "evaluation_scorecard.csv", matched, metric_fields)

    per_well = copied_tables.get("per_well_metrics.csv", [])
    matched_well = [row for row in per_well if (row.get("task", ""), row.get("input", ""), row.get("model", "")) in registered]
    well_fields = list(per_well[0].keys()) if per_well else ["well", "task", "input", "model"]
    _csv_write(out / "tables" / "evaluation_per_well_metrics.csv", matched_well, well_fields)

    conformal = copied_tables.get("conformal_intervals.csv", [])
    matched_conformal = [row for row in conformal if (row.get("task", ""), row.get("input", ""), row.get("model", "")) in registered]
    conformal_fields = list(conformal[0].keys()) if conformal else ["task", "input", "model"]
    _csv_write(out / "tables" / "evaluation_conformal_intervals.csv", matched_conformal, conformal_fields)

    summary = _safe_json(stage_dir / "evaluation_summary.json")
    _csv_write(out / "tables" / "evaluation_evidence_inventory.csv", inventory)
    status = "AVAILABLE" if matched else "AVAILABLE_NO_RELEASE_MODEL_MATCH"
    return {"status": status, "source_stage": stage_dir.relative_to(storage_root).as_posix(),
            "source_release_id": str(summary.get("release_id") or summary.get("model_release_id") or ""),
            "source_evaluation_policy": summary.get("evaluation_policy", "not recorded"),
            "source_prediction_rows": summary.get("prediction_rows"),
            "registered_models": [list(key) for key in sorted(registered)],
            "registered_model_rows": len(matched), "registered_per_well_rows": len(matched_well),
            "registered_conformal_rows": len(matched_conformal),
            "source_outer_oof_rows": next((row["rows"] for row in inventory
                                            if row["artifact"] == "outer_oof_predictions.csv"), 0),
            "files": inventory, "figure_paths": figure_paths, "_model_metrics": matched,
            "limitation": "These are imported Stage 08 held-out evaluation results. They are not Stage 09 blind-well validation."}

def _write_specialist_summary_package(*, out: Path, storage_root: Path, source_job: Path, source_id: str,
                                      foundation_id: str, geometry: dict, inference: dict, release_id: str,
                                      stats: dict, summary: dict, audit_checks: list[dict],
                                      source_artifacts: dict, products: dict, source_hash_checks: list[bool],
                                      parity: dict, valid_count: int) -> None:
    """Write Stage 10 in the specialist's stable tables/figures/model_cards/reproduce layout."""
    tables = out / "tables"
    figures = out / "figures"
    cards = out / "model_cards"
    reproduce = out / "reproduce"
    for folder in (tables, figures, cards, reproduce):
        folder.mkdir(parents=True, exist_ok=True)

    pipeline_rows = _collect_pipeline_status(storage_root, source_job, foundation_id)
    foundation_job = storage_root / "processing_jobs" / foundation_id if foundation_id else None
    dataset_index = _load_dataset_index(source_job, foundation_job) if foundation_job else _load_dataset_index(source_job)
    # Expand the upstream audit from stage-level status to individual recorded
    # inputs/outputs. Missing paths remain explicit; they are never silently
    # interpreted as successful or reconstructed from filenames.
    stage_artifact_rows = []
    for stage_row in pipeline_rows:
        provenance_rel = stage_row.get("provenance_file")
        provenance_path = storage_root / provenance_rel if provenance_rel else None
        record = _safe_json(provenance_path) if provenance_path else {}
        for direction in ("inputs", "outputs"):
            entries = record.get(direction, []) if isinstance(record.get(direction), list) else []
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    entry = {"path": str(entry)}
                raw = entry.get("path") or entry.get("relative_path") or ""
                owner_job = provenance_path.parent.parent if provenance_path else source_job
                dataset_id = _recorded_dataset_id(entry)
                candidate = _resolve_recorded_path(
                    raw, provenance_path=provenance_path, owner_job=owner_job,
                    stage_id=stage_row["stage"], dataset_id=dataset_id, dataset_index=dataset_index)
                exists = bool(candidate and candidate.is_file() and not candidate.is_symlink())
                expected = str(entry.get("sha256") or "")
                actual = _sha256(candidate) if exists and expected else ""
                artifact_status = ("NOT_RECORDED" if not raw else
                                   "NOT_RETESTABLE" if not exists and direction == "inputs" and
                                   _is_unavailable_linked_input(dataset_id, dataset_index) else
                                   "UNAVAILABLE_EXTERNAL" if not exists and (Path(str(raw)).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", str(raw)) or str(raw).startswith("\\\\")) else
                                   "MISSING" if not exists else
                                   "NOT_HASHED" if not expected else
                                   "PASS" if actual == expected else "CHANGED")
                raw_label = (Path(str(raw).replace("\\", "/")).name
                             if Path(str(raw)).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", str(raw))
                             else str(raw))
                resolved_label = ""
                if candidate:
                    try:
                        resolved_label = candidate.resolve().relative_to(storage_root.resolve()).as_posix()
                    except (OSError, ValueError):
                        resolved_label = candidate.name
                stage_artifact_rows.append({"stage": stage_row["stage"], "direction": direction,
                    "index": index, "role": entry.get("role", ""), "dataset_id": dataset_id,
                    "path": raw_label, "resolved_path": resolved_label,
                    "expected_sha256": expected, "actual_sha256": actual,
                    "status": artifact_status})
    _csv_write(tables / "stage_artifact_audit.csv", stage_artifact_rows)
    for stage_row in pipeline_rows:
        stage_artifacts = [row for row in stage_artifact_rows if row["stage"] == stage_row["stage"]]
        stage_row["artifact_count"] = len(stage_artifacts)
        stage_row["artifact_audit_status"] = _audit_rollup_status(
            [str(row["status"]) for row in stage_artifacts])
        stage_row["artifact_pass_count"] = sum(row["status"] == "PASS" for row in stage_artifacts)
        stage_row["artifact_issue_count"] = sum(row["status"] in {"FAIL", "MISSING", "CHANGED"}
                                                for row in stage_artifacts)
        stage_row["artifact_not_retestable_count"] = sum(
            row["status"] in {"NOT_RETESTABLE", "UNAVAILABLE_EXTERNAL"} for row in stage_artifacts)
        stage_row["execution_audit_status"] = _stage_execution_status(stage_row.get("status"))
    for row in pipeline_rows:
        if not row.get("code_fingerprint"):
            row["code_fingerprint"] = "NOT_RECORDED"
        if not row.get("code_version"):
            row["code_version"] = "NOT_RECORDED"
    _csv_write(tables / "stage_scorecard.csv", pipeline_rows)

    integrity_rows = []
    for label, item in {**source_artifacts, **{f"prediction_{k}": v for k, v in products.items()}}.items():
        rel = item.get("source_job_relative_path")
        if item.get("storage_root_relative_path"):
            path = storage_root / item["storage_root_relative_path"]
        elif rel:
            path = storage_root / rel
        else:
            continue
        actual = _sha256(path) if path.is_file() else ""
        integrity_rows.append({"artifact": label, "path": rel or item.get("storage_root_relative_path"),
                               "expected_bytes": item.get("bytes"),
                               "actual_bytes": path.stat().st_size if path.is_file() else "",
                               "expected_sha256": item.get("sha256"), "actual_sha256": actual,
                               "status": "PASS" if actual and actual == item.get("sha256") else "FAIL"})
    _csv_write(tables / "output_integrity.csv", integrity_rows)

    lineage_rows = []
    for stage_id in ("08_evaluation", "09_inference"):
        record = _safe_json(source_job / "provenance" / f"{stage_id}.json")
        for item in record.get("inputs", []) if isinstance(record.get("inputs"), list) else []:
            raw = item.get("path") if isinstance(item, dict) else None
            expected = item.get("sha256") if isinstance(item, dict) else None
            owner_job = source_job
            dataset_id = _recorded_dataset_id(item)
            path = _resolve_recorded_path(
                raw, provenance_path=source_job / "provenance" / f"{stage_id}.json",
                owner_job=owner_job, stage_id=stage_id, dataset_id=dataset_id,
                dataset_index=dataset_index)
            exists = bool(path and path.is_file() and not path.is_symlink())
            actual = _sha256(path) if exists and expected else ""
            status = ("NOT_RECORDED" if not raw else
                      "NOT_RETESTABLE" if not exists and _is_unavailable_linked_input(dataset_id, dataset_index) else
                      "UNAVAILABLE_EXTERNAL" if not exists and (Path(str(raw)).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", str(raw)) or str(raw).startswith("\\\\")) else
                      "MISSING" if not exists else
                      "NOT_HASHED" if not expected else "PASS" if actual == expected else "CHANGED")
            raw_label = (Path(str(raw).replace("\\", "/")).name
                         if raw and (Path(str(raw)).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", str(raw)))
                         else str(raw or ""))
            resolved_label = ""
            if path:
                try:
                    resolved_label = path.resolve().relative_to(storage_root.resolve()).as_posix()
                except (OSError, ValueError):
                    resolved_label = path.name
            lineage_rows.append({"stage": stage_id, "role": item.get("role", "") if isinstance(item, dict) else "",
                                 "dataset_id": dataset_id, "path": raw_label,
                                 "resolved_path": resolved_label, "expected_sha256": expected or "",
                                 "actual_sha256": actual, "status": status})
    _csv_write(tables / "lineage_consistency.csv", lineage_rows)

    # Stage 10 now explicitly closes the audit loop: stage provenance, recorded
    # artifact integrity, source lineage and generated prediction integrity are
    # represented as acceptance checks and claims, not merely sidecar tables.
    stage_status_values = [str(row.get("status") or "NOT_RECORDED").upper() for row in pipeline_rows]
    stage_artifact_statuses = [str(row.get("status") or "NOT_TESTABLE") for row in stage_artifact_rows]
    package_audit_checks = [
        {"check": "upstream_stage_provenance_coverage",
         "status": "PASS" if stage_status_values and all(v not in {"NOT_RECORDED", "MISSING", "UNKNOWN"} for v in stage_status_values) else "WARN",
         "detail": {"expected_stages": len(pipeline_rows),
                    "recorded_stages": sum(v not in {"NOT_RECORDED", "MISSING", "UNKNOWN"} for v in stage_status_values),
                    "stage_statuses": {row["stage"]: row["status"] for row in pipeline_rows}}},
        {"check": "upstream_stage_execution_status",
         "status": _audit_rollup_status([_stage_execution_status(row.get("status"))
                                          for row in pipeline_rows]),
         "detail": {"stage_statuses": {row["stage"]: row.get("status", "NOT_RECORDED")
                                        for row in pipeline_rows}}},
        {"check": "upstream_recorded_artifact_integrity",
         "status": _audit_rollup_status(stage_artifact_statuses),
         "detail": {"artifacts_checked": len(stage_artifact_rows),
                    "status_counts": {v: stage_artifact_statuses.count(v) for v in sorted(set(stage_artifact_statuses))}}},
        {"check": "source_lineage_integrity",
         "status": _audit_rollup_status([str(row["status"]) for row in lineage_rows]),
         "detail": {"inputs_checked": len(lineage_rows),
                    "status_counts": {v: sum(row["status"] == v for row in lineage_rows)
                                      for v in sorted({str(row["status"]) for row in lineage_rows})}}},
        {"check": "stage_g_prediction_output_integrity",
         "status": _audit_rollup_status([str(row["status"]) for row in integrity_rows]),
         "detail": {"products_checked": len(integrity_rows),
                    "failed_products": [row["artifact"] for row in integrity_rows if row["status"] != "PASS"]}},
    ]

    result_rows = []
    for task, item in stats.items():
        result_rows.append({"attribute": task, "valid_voxels": item["finite_valid_voxels"],
                            "valid_voxel_fraction": summary["valid_voxel_fraction"], "min": item["min"],
                            "mean": item["mean"], "std": item["std"], "max": item["max"],
                            "p05": item["percentiles"]["p05"], "p50": item["percentiles"]["p50"],
                            "p95": item["percentiles"]["p95"],
                            "domain_status": item["value_domain_status"]})
    _csv_write(tables / "results_scorecard.csv", result_rows, [
        "attribute", "valid_voxels", "valid_voxel_fraction", "min", "mean", "std", "max",
        "p05", "p50", "p95", "domain_status"])
    _csv_write(tables / "volume_statistics.csv", [
        {"attribute": task, "finite_valid_voxels": item["finite_valid_voxels"],
         "null_or_nonfinite_voxels_within_valid_mask": item["null_or_nonfinite_voxels_within_valid_mask"],
         "min": item["min"], **item["percentiles"], "max": item["max"], "mean": item["mean"], "std": item["std"]}
        for task, item in stats.items()])

    release_id, release_manifest, release_checks = _resolve_release_manifest(storage_root, geometry, inference)
    evaluation_evidence = _import_evaluation_evidence(
        source_job=source_job, storage_root=storage_root, foundation_id=foundation_id,
        release_id=release_id, release_manifest=release_manifest, out=out,
    )
    model_scorecard_rows = []
    for metric_row in evaluation_evidence.get("_model_metrics", []):
        model_scorecard_rows.append({
            "task": metric_row.get("task", ""), "input": metric_row.get("input", ""),
            "model": metric_row.get("model", ""), "evaluation_stage": evaluation_evidence.get("source_stage", ""),
            "evaluation_policy": evaluation_evidence.get("source_evaluation_policy", "not recorded"),
            "n_rows": metric_row.get("n_rows", ""), "n_wells": metric_row.get("n_wells", ""),
            "outer_roc_auc": metric_row.get("outer_roc_auc", ""),
            "outer_rmse": metric_row.get("outer_rmse", ""),
            "outer_mae": metric_row.get("outer_mae", ""),
            "outer_r2": metric_row.get("outer_r2", ""),
            "outer_roc_auc_ci_lo": metric_row.get("outer_roc_auc_ci_lo", ""),
            "outer_roc_auc_ci_hi": metric_row.get("outer_roc_auc_ci_hi", ""),
            "outer_rmse_ci_lo": metric_row.get("outer_rmse_ci_lo", ""),
            "outer_rmse_ci_hi": metric_row.get("outer_rmse_ci_hi", ""),
            "evidence_status": "IMPORTED_NOT_RECALCULATED"})
    _csv_write(tables / "model_results_scorecard.csv", model_scorecard_rows, [
        "task", "input", "model", "evaluation_stage", "evaluation_policy", "n_rows", "n_wells",
        "outer_roc_auc", "outer_rmse", "outer_mae", "outer_r2", "outer_roc_auc_ci_lo",
        "outer_roc_auc_ci_hi", "outer_rmse_ci_lo", "outer_rmse_ci_hi", "evidence_status"])

    summary["evaluation_evidence"] = {key: value for key, value in evaluation_evidence.items()
                                      if not key.startswith("_")}
    evaluation_check_status = (
        "PASS" if evaluation_evidence["status"] == "AVAILABLE" else
        "WARN" if evaluation_evidence["status"] == "AVAILABLE_NO_RELEASE_MODEL_MATCH" else "NOT_TESTABLE"
    )
    evaluation_check = {"check": "registered_model_held_out_evaluation_evidence",
                        "status": evaluation_check_status,
                        "detail": {"registered_model_rows": evaluation_evidence.get("registered_model_rows", 0),
                                   "registered_per_well_rows": evaluation_evidence.get("registered_per_well_rows", 0),
                                   "registered_conformal_rows": evaluation_evidence.get("registered_conformal_rows", 0),
                                   "source_stage": evaluation_evidence.get("source_stage", "")}}
    audit_path = out / "summary_audit.json"
    audit_payload = _safe_json(audit_path)
    audit_payload["checks"] = (list(audit_payload.get("checks", [])) + package_audit_checks +
                                release_checks + [evaluation_check])
    if evaluation_evidence.get("limitation"):
        audit_payload.setdefault("limitations", []).append(evaluation_evidence["limitation"])
    audit_payload["model_release_id"] = release_id
    audit_payload["status"] = "completed" if all(x.get("status") == "PASS" for x in audit_payload["checks"]) else "conditional"
    audit_payload["limitations"] = list(audit_payload.get("limitations", []))
    missing_code_identity = [row["stage"] for row in pipeline_rows
                             if row.get("code_fingerprint") in (None, "", "NOT_RECORDED")
                             and row.get("code_version") in (None, "", "NOT_RECORDED")]
    if missing_code_identity:
        audit_payload["limitations"].append(
            f"Code identity was not recorded at execution time for {len(missing_code_identity)} upstream stages; "
            "it cannot be reconstructed reliably after the run.")
    audit_path.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")
    claim_rows = []
    all_checks = list(audit_checks) + package_audit_checks + release_checks + [evaluation_check]
    for check in all_checks:
        claim_rows.append({"id": f"Q{len(claim_rows)+1:03d}", "claim": check["check"],
                           "verdict": check["status"], "evidence": json.dumps(check.get("detail", ""), default=str),
                           "rule": "Explicit acceptance rule; missing evidence is not treated as PASS",
                           "source": "summary_audit.json"})
    for row in pipeline_rows:
        claim_rows.append({"id": f"Q{len(claim_rows)+1:03d}",
            "claim": f"Upstream stage execution: {row['stage']}",
            "verdict": row.get("execution_audit_status", "NOT_TESTABLE"),
            "evidence": json.dumps({key: row.get(key) for key in
                ("processing_job_id", "status", "recorded_at", "started_at", "completed_at",
                 "config_sha256", "code_fingerprint", "code_version", "input_count", "output_count",
                 "warning_count", "error_count", "provenance_file")}, default=str),
            "rule": "Stage execution state is derived only from its recorded provenance status",
            "source": "tables/stage_scorecard.csv"})
    for row in stage_artifact_rows:
        claim_rows.append({"id": f"Q{len(claim_rows)+1:03d}",
            "claim": f"{row['stage']} {row['direction']} artifact: {row['path'] or 'path not recorded'}",
            "verdict": row["status"], "evidence": json.dumps({k: row.get(k) for k in
                ("role", "dataset_id", "resolved_path", "expected_sha256", "actual_sha256", "status")}),
            "rule": "Resolve stage-relative/job-relative/portable paths; verify recorded SHA-256. Unavailable linked inputs are NOT_RETESTABLE, not MISSING.",
            "source": "tables/stage_artifact_audit.csv"})
    for row in integrity_rows:
        claim_rows.append({"id": f"Q{len(claim_rows)+1:03d}",
            "claim": f"Generated product integrity: {row['artifact']}",
            "verdict": row["status"], "evidence": json.dumps(row, default=str),
            "rule": "Actual byte count and SHA-256 must match the recorded product inventory",
            "source": "tables/output_integrity.csv"})
    for row in lineage_rows:
        claim_rows.append({"id": f"Q{len(claim_rows)+1:03d}",
            "claim": f"Source lineage input: {row['stage']} {row['path']}",
            "verdict": row["status"], "evidence": json.dumps(row, default=str),
            "rule": "Verify portable source inputs by SHA-256; unavailable external linked inputs are NOT_RETESTABLE, not MISSING.",
            "source": "tables/lineage_consistency.csv"})
    # Link release-specific scientific performance claims to the exact copied
    # Stage 08 source table and row identity. These are evidence references,
    # not new validation results or causal/geological interpretations.
    for metric_index, metric_row in enumerate(evaluation_evidence.get("_model_metrics", []), start=1):
        metric_label = "/".join(str(metric_row.get(k, "")) for k in ("task", "input", "model"))
        metrics = {k: v for k, v in metric_row.items() if k.startswith("outer_") or
                   k in {"n_rows", "n_wells", "kind"}}
        claim_rows.append({"id": f"Q{len(claim_rows)+1:03d}",
            "claim": f"Registered held-out evaluation evidence: {metric_label}",
            "verdict": "EVIDENCE_AVAILABLE" if metrics else "NOT_TESTABLE",
            "evidence": json.dumps(metrics, default=str),
            "rule": "Copied verbatim from release-matched Stage 08 evaluation; no recomputation",
            "source": f"tables/evaluation_source/model_performance.csv row {metric_index}"})
    for task, item in stats.items():
        claim_rows.append({"id": f"Q{len(claim_rows)+1:03d}",
            "claim": f"Volume attribute summary: {task}",
            "verdict": item["value_domain_status"],
            "evidence": json.dumps({"finite_valid_voxels": item["finite_valid_voxels"],
                "min": item["min"], "mean": item["mean"], "max": item["max"],
                "p05": item["percentiles"]["p05"], "p50": item["percentiles"]["p50"],
                "p95": item["percentiles"]["p95"]}),
            "rule": "Descriptive statistics over finite values where prediction_valid == 1",
            "source": "tables/volume_statistics.csv"})
    _csv_write(tables / "claims_register.csv", claim_rows)
    _csv_write(tables / "acceptance_checks.csv", all_checks)
    _csv_write(tables / "issues_register.csv", [
        {"stage": "10_summary", "severity": "LIMITATION", "issue": text}
        for text in _safe_json(out / "summary_audit.json").get("limitations", [])
    ])
    _csv_write(tables / "code_currency.csv", [{
        "stage": row["stage"],
        "status": "RECORDED_NOT_VERIFIED" if (row.get("code_fingerprint") not in (None, "", "NOT_RECORDED") or row.get("code_version") not in (None, "", "NOT_RECORDED")) else "NOT_RECORDED",
        "code_fingerprint": row.get("code_fingerprint", "") or "NOT_RECORDED",
        "code_version": row.get("code_version", "") or "NOT_RECORDED",
        "detail": "Historical provenance did not record a code fingerprint/version. Stage 10 does not invent one retrospectively; a future execution must record it at launch."
    } for row in pipeline_rows])
    _csv_write(tables / "confidentiality_scan.csv", [{"status": "NOT_APPLICABLE",
        "detail": "The specialist scans its in-run package directory; this backend packaging job does not own that code directory."}])

    environment = [{"item": "python", "value": sys.version.split()[0]},
                   {"item": "platform", "value": platform.platform()},
                   {"item": "numpy", "value": np.__version__},
                   {"item": "hostname", "value": os.environ.get("HOSTNAME", "")},
                   {"item": "stage_g_job_id", "value": out.parents[1].name},
                   {"item": "source_inference_job_id", "value": source_id}]
    _csv_write(tables / "environment.csv", environment)

    model_rows = release_manifest.get("models", []) if isinstance(release_manifest.get("models"), list) else []
    card_paths = []
    for model in model_rows:
        task = str(model.get("task") or "unknown_task")
        model_name = str(model.get("model") or "unknown_model")
        input_name = str(model.get("input") or "unknown_input")
        card_name = "__".join(_safe_artifact_label(value) for value in (task, model_name, input_name)) + ".md"
        model_metrics = [row for row in evaluation_evidence.get("_model_metrics", [])
                        if row.get("task") == task and row.get("input") == input_name and row.get("model") == model_name]
        card_metric_text = "\n".join(
            f"- `{key}`: `{value}`" for key, value in (model_metrics[0].items() if model_metrics else [])
            if key.startswith("outer_") or key in {"n_rows", "n_wells", "kind"}
        ) if model_metrics else "No matching Stage 08 held-out evaluation row was available for this release model."
        model_file = str(model.get("model_file") or "")
        model_file_meta = (release_manifest.get("files") or {}).get(model_file, {}) if isinstance(release_manifest.get("files"), dict) else {}
        eval_summary = _safe_json(out / "tables" / "evaluation_source" / "evaluation_summary.json")
        card_text = (
            f"# Model card — {task}\n\n"
            "## Identity and release\n\n"
            f"- Task / target: `{task}`\n- Model family: `{model.get('family', model.get('kind', model_name))}`\n"
            f"- Registered model name: `{model_name}`\n- Input contract: `{input_name}`\n"
            f"- Model release ID: `{release_id or 'NOT_AVAILABLE'}`\n"
            f"- Release manifest SHA-256: `{_sha256(storage_root / 'model_releases' / release_id / 'manifest.json') if release_id and (storage_root / 'model_releases' / release_id / 'manifest.json').is_file() else 'NOT_AVAILABLE'}`\n"
            f"- Foundation processing job: `{foundation_id or 'NOT_AVAILABLE'}`\n"
            f"- Inference processing job: `{source_id}`\n"
            f"- Model artifact path: `{model_file or 'NOT_RECORDED'}`\n"
            f"- Model artifact SHA-256: `{model_file_meta.get('sha256', 'NOT_RECORDED')}`\n\n"
            "## Model specification\n\n"
            f"- Target definition: `{model.get('target', model.get('target_definition', task))}`\n"
            f"- Feature set: `{json.dumps(model.get('features', model.get('feature_set', 'NOT_RECORDED')), sort_keys=True, default=str)}`\n"
            f"- Preprocessing contract: `{json.dumps(model.get('preprocessing', model.get('preprocessing_config', 'see immutable release manifest')), sort_keys=True, default=str)}`\n"
            f"- Final parameters: `{json.dumps(model.get('final_params'), sort_keys=True, default=str)}`\n"
            f"- Decision threshold: `{model.get('final_threshold', 'NOT_RECORDED')}`\n"
            f"- Training rows: `{model.get('n_train', model.get('training_rows', 'NOT_RECORDED'))}`\n"
            f"- Training wells: `{json.dumps(model.get('training_wells', 'NOT_RECORDED'), default=str)}`\n"
            f"- Class / target labels: `{json.dumps(model.get('classes', model.get('target_classes', 'NOT_RECORDED')), default=str)}`\n\n"
            "## Evaluation protocol and results\n\n"
            f"- Evaluation source: `{evaluation_evidence.get('source_stage', 'NOT_AVAILABLE')}`\n"
            f"- Evaluation policy: `{evaluation_evidence.get('source_evaluation_policy', 'NOT_RECORDED')}`\n"
            f"- Evaluation summary policy: `{eval_summary.get('evaluation_policy', 'NOT_RECORDED')}`\n"
            f"- Matching model rows: `{len(model_metrics)}`\n"
            + (card_metric_text if model_metrics else "No matching release-bound Stage 08 evaluation row was available.") + "\n\n"
            "Metrics are imported verbatim from the release-matched Stage 08 evidence. Stage 10 does not refit, "
            "recalibrate, recompute, or reinterpret these metrics. Outer held-out metrics are not Stage 09 blind-well metrics.\n\n"
            "## Intended use and operating conditions\n\n"
            f"The model is registered for `{task}` using the `{input_name}` input contract. Apply only to data "
            "prepared with the exact feature, preprocessing, geometry and sampling contracts in the immutable release. "
            "Confirm spatial registration, valid-mask coverage and attribute-domain QC before interpretation.\n\n"
            "## Data provenance and validation\n\n"
            f"- Evaluation evidence status: `{evaluation_evidence.get('status', 'NOT_AVAILABLE')}`\n"
            f"- Evaluation prediction rows: `{evaluation_evidence.get('source_prediction_rows', 'NOT_RECORDED')}`\n"
            f"- Registered per-well metric rows: `{evaluation_evidence.get('registered_per_well_rows', 0)}`\n"
            f"- Registered conformal interval rows: `{evaluation_evidence.get('registered_conformal_rows', 0)}`\n"
            "- Model identity and artifact checks are recorded in `summary_audit.json` and `tables/output_integrity.csv`.\n\n"
            "## Limitations and prohibited interpretations\n\n"
            "Performance evidence applies only to the evaluation population and protocol recorded by Stage 08. "
            "No independent blind-well generalisation claim is made unless a separately registered Stage 09 "
            "blind-well protocol and its frozen prediction/truth evidence are available. Do not interpret a "
            "lithology probability as fluid classification, reserve estimate, or direct geological certainty. "
            "Missing metadata is explicitly unreported; it is not inferred.\n"
        )
        (cards / card_name).write_text(card_text, encoding="utf-8")
        card_paths.append(f"model_cards/{card_name}")
    if not card_paths:
        (cards / "MODEL_CARDS_UNAVAILABLE.md").write_text(
            "# Model cards unavailable\n\nThe source inference summary or immutable release manifest did not provide model entries. "
            "No model identity or performance claims have been inferred.\n", encoding="utf-8")

    # Small self-contained SVG figures: no plotting dependency and no external assets.
    coverage = float(summary.get("valid_voxel_fraction", 0.0)) * 100.0
    (figures / "volume_coverage.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="720" height="150" viewBox="0 0 720 150">'
        f'<text x="20" y="28" font-size="18">Valid voxel coverage</text>'
        f'<rect x="20" y="55" width="660" height="34" fill="#e5e7eb"/>'
        f'<rect x="20" y="55" width="{660*max(0,min(100,coverage))/100:.2f}" height="34" fill="#287a58"/>'
        f'<text x="20" y="120" font-size="16">{coverage:.2f}% ({valid_count:,} valid voxels)</text></svg>',
        encoding="utf-8")
    bars = []
    for idx, (task, item) in enumerate(stats.items()):
        y = 35 + idx * 42
        mean = float(item["mean"] or 0.0)
        bars.append(f'<text x="12" y="{y+16}" font-size="13">{html.escape(task)}</text>'
                    f'<rect x="180" y="{y}" width="{max(0,min(1,mean))*440:.2f}" height="22" fill="#416f9c"/>'
                    f'<text x="635" y="{y+16}" font-size="12">{mean:.3f}</text>')
    (figures / "attribute_means.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="720" height="220" viewBox="0 0 720 220">'
        '<text x="12" y="20" font-size="16">Mean prediction attributes (0–1 scale)</text>' + ''.join(bars) + '</svg>',
        encoding="utf-8")
    status_colors = {"COMPLETED": "#287a58", "CONDITIONAL": "#b7791f", "BLOCKED": "#c05621",
                     "FAILED": "#b83232", "NOT_RECORDED": "#89939e"}
    status_parts = []
    for idx, row in enumerate(pipeline_rows):
        y = 32 + idx * 24
        color = status_colors.get(str(row["status"]), "#89939e")
        status_parts.append(f'<text x="10" y="{y+13}" font-size="12">{html.escape(str(row["stage"]))}</text>'
                            f'<rect x="190" y="{y}" width="150" height="17" rx="3" fill="{color}"/>'
                            f'<text x="350" y="{y+13}" font-size="11">{html.escape(str(row["status"]))}</text>')
    (figures / "pipeline_status.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="620" height="{max(100, 45+len(status_parts)*24)}">'
        '<text x="10" y="20" font-size="16">Recorded upstream stage statuses</text>' + ''.join(status_parts) + '</svg>',
        encoding="utf-8")
    claim_parts = []
    for idx, row in enumerate(claim_rows):
        y = 32 + idx * 26
        claim_parts.append(f'<text x="10" y="{y+13}" font-size="11">{html.escape(row["id"])} {html.escape(row["claim"][:72])}</text>'
                           f'<text x="570" y="{y+13}" font-size="11">{html.escape(row["verdict"])}</text>')
    (figures / "claims_register.svg").write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="760" height="{max(100, 45+len(claim_parts)*26)}">'
        '<text x="10" y="20" font-size="16">Stage 10 evidence claims and checks</text>' + ''.join(claim_parts) + '</svg>',
        encoding="utf-8")

    reproduce_script = (
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        "# Re-run Stage G as a new APGCE processing job using its dedicated YAML.\n"
        "# The APGCE job runner must provide APGCE_RESULT_PATH and APGCE_JOB_ID.\n"
        "python scripts/process_C_volume_products.py \\\n"
        "  --config scripts/C_volume_products.yaml\n"
    )
    (reproduce / "run_stage_g.sh").write_text(reproduce_script, encoding="utf-8")

    stage_rows = "".join(f"<tr><td>{html.escape(str(r['stage']))}</td><td>{html.escape(str(r['status']))}</td>"
                         f"<td>{html.escape(str(r['processing_job_id']))}</td></tr>" for r in pipeline_rows)
    claim_html = "".join(f"<tr><td>{html.escape(r['id'])}</td><td>{html.escape(r['claim'])}</td>"
                         f"<td>{html.escape(r['verdict'])}</td></tr>" for r in claim_rows)
    # Inline SVG source directly: dashboard viewers that restrict external SVG
    # image references can still render the figures in this self-contained HTML.
    inline_figures = []
    for figure_name in ("volume_coverage.svg", "attribute_means.svg",
                        "pipeline_status.svg", "claims_register.svg"):
        svg_text = (figures / figure_name).read_text(encoding="utf-8")
        inline_figures.append(f"<figure><figcaption>{html.escape(figure_name)}</figcaption>{svg_text}</figure>")
    evaluation_gallery = ["<h2>Held-out evaluation evidence</h2>",
                          "<p class='muted'>Imported from the matching Stage 08 evaluation release. "
                          "These results are not Stage 09 blind-well validation.</p>"]
    scorecard_path = out / "tables" / "evaluation_scorecard.csv"
    if scorecard_path.is_file():
        score_rows = _read_csv_rows(scorecard_path)[1]
        if score_rows:
            columns = [key for key in ("task", "input", "model", "n_rows", "n_wells",
                                       "outer_roc_auc", "outer_rmse", "outer_roc_auc_ci_lo",
                                       "outer_roc_auc_ci_hi", "outer_rmse_ci_lo", "outer_rmse_ci_hi")
                       if key in score_rows[0]]
            evaluation_gallery.append("<table><tr>" + "".join(f"<th>{html.escape(key)}</th>" for key in columns) + "</tr>" +
                "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in columns) + "</tr>"
                        for row in score_rows) + "</table>")
    for relative in evaluation_evidence.get("figure_paths", []):
        image_path = out / relative
        if image_path.is_file():
            evaluation_gallery.append(f"<figure><figcaption>{html.escape(image_path.name)}</figcaption>"
                                      f"<img alt='{html.escape(image_path.name)}' src='{_png_data_uri(image_path)}'></figure>")

    (out / "dashboard.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>GeoGG Stage 10 Summary</title>"
        "<style>body{font:15px system-ui;max-width:1100px;margin:36px auto;padding:0 20px;color:#202b36}"
        "table{border-collapse:collapse;width:100%;margin:14px 0 28px}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}"
        "figure{margin:24px 0}figure svg{display:block;width:100%;height:auto;max-height:520px}"
        "img{max-width:100%;height:auto}.muted{color:#65717d}</style></head><body>"
        f"<h1>GeoGG Stage 10 Summary</h1><p class='muted'>Stage G job: {html.escape(out.parents[1].name)} | "
        f"Source inference job: {html.escape(source_id)} | Model release: {html.escape(release_id)}</p>"
        f"<h2>Volume coverage</h2><p>{coverage:.2f}% valid voxels; shape {html.escape(str(summary['shape']))}</p>"
        + "".join(inline_figures) + "".join(evaluation_gallery) +
        f"<h2>Recorded pipeline stages</h2><table><tr><th>Stage</th><th>Status</th><th>Job</th></tr>{stage_rows}</table>"
        f"<h2>Audit claims</h2><table><tr><th>ID</th><th>Check</th><th>Verdict</th></tr>{claim_html}</table>"
        "<p>See tables/, model_cards/, figures/sections/ and HANDOFF.md for the complete evidence package.</p></body></html>",
        encoding="utf-8")

    limitations = [
        "This is a separate-job audit of recorded provenance and registered products, not a reconstruction of the original specialist in-process context.",
        "Stage 08 outer held-out evaluation metrics are imported without recalculation; they must not be described as Stage 09 blind-well validation.",
        "A separate Stage 09 blind-well evaluation is not generated because this backend's volume inference does not currently freeze predictions against an independent well-truth protocol.",
        "Confidentiality scanning of the specialist code directory is not applicable to this independent backend job.",
    ]
    handoff = ["# GeoGG Stage 10 — audit and handoff", "",
               f"- Stage G job: `{out.parents[1].name}`", f"- Source inference job: `{source_id}`",
               f"- Foundation job: `{foundation_id}`", f"- Model release: `{release_id or 'NOT_AVAILABLE'}`",
               f"- Shape: `{summary['shape']}` (axis order: `{summary['axis_order']}`)",
               f"- Valid voxels: {valid_count:,} / {summary['candidate_voxels']:,} ({coverage:.2f}%)",
               f"- Live ROI traces: {summary['live_trace_count']:,}", "",
               "## Reading order", "1. `dashboard.html` — offline overview.",
               "2. `tables/claims_register.csv` — explicit checks and verdicts.",
               "3. `tables/results_scorecard.csv` and `tables/volume_statistics.csv` — attribute statistics.",
               "4. `tables/model_results_scorecard.csv` and `tables/evaluation_scorecard.csv` — release-matched model evaluation evidence.",
               "5. `tables/evaluation_per_well_metrics.csv` and `tables/evaluation_conformal_intervals.csv` — supporting held-out evidence.",
               "6. `tables/evaluation_source/` — exact source Stage 08 evaluation tables and checksums.",
               "7. `model_cards/` — immutable release model identities and matching evaluation evidence.",
               "8. `tables/stage_scorecard.csv`, `stage_artifact_audit.csv`, `output_integrity.csv`, `lineage_consistency.csv` — stage and artifact-level audit trail.",
               "9. `figures/evaluation/` — source Stage 08 evaluation figures.",
               "10. `figures/sections/` — inline, crossline and time-slice seismic/prediction panels.",
               "11. `figures/` — lightweight coverage and attribute figures.", "",
               "## Scientific limitations", *[f"- {item}" for item in limitations], "",
               "## Geometry", "Use native IL/XL geometry for grid operations. The source geometry's XY mapping "
               "quality flags remain authoritative; do not assume affine XY mapping.",
               "Sections use native IL/XL trace mapping. Well-trajectory overlays are not generated unless a "
               "registered trajectory-to-volume intersection contract is available.", ""]
    (out / "HANDOFF.md").write_text("\n".join(handoff), encoding="utf-8")

    # Inventory all package artifacts before the manifest is rewritten by caller.
    package_inventory = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name not in {"volume_product_manifest.json", "volume_summary.json"}:
            package_inventory[path.relative_to(out).as_posix()] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    summary["specialist_aligned_layout"] = {
        "directories": ["tables/", "figures/", "model_cards/", "reproduce/"],
        "dashboard": "dashboard.html", "handoff": "HANDOFF.md",
        "model_card_count": len(card_paths), "package_artifacts": package_inventory,
    }
    (out / "volume_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")







def _png_data_uri(path: Path) -> str:
    """Return a self-contained PNG data URI for offline HTML dashboards."""
    if not path.is_file():
        raise FileNotFoundError(f"Dashboard preview image is missing: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def run_volume_products(argv=None) -> int:
    parser = argparse.ArgumentParser(description='GeoGG Stages 10–11: volume summary and export')
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text()) or {}
    options = cfg.get('volume_products') or {}
    result = Path(os.environ['APGCE_RESULT_PATH']).expanduser().resolve()
    job_dir = result.parent
    storage_root = job_dir.parent.parent.resolve()
    jobs_root = (storage_root / 'processing_jobs').resolve()
    if jobs_root.parent != storage_root or not jobs_root.is_dir():
        raise RuntimeError(f'Backend processing_jobs storage root is unavailable: {jobs_root}')

    source_id = str(options.get('source_inference_job_id') or '').strip()
    if not source_id or Path(source_id).name != source_id or source_id in {'.', '..'}:
        raise RuntimeError('volume_products.source_inference_job_id must be a safe processing-job ID')
    source_job = jobs_root / source_id
    if source_job.is_symlink() or source_job.resolve().parent != jobs_root or not source_job.is_dir():
        raise RuntimeError(f'Source inference processing job is unavailable or unsafe: {source_id}')
    source = source_job / 'stages' / '09_inference'
    if not source.is_dir(): source = source_job / '09_inference'
    out = job_dir / 'stages' / '10_summary'
    if not source.is_dir(): raise RuntimeError(f'Stage F inference outputs not found in job {source_id}: {source}')
    inference_summary_path = source / 'volume_inference_summary.json'
    geometry_path = source / 'volumes' / 'volume_geometry.json'
    if not inference_summary_path.is_file() or not geometry_path.is_file():
        raise RuntimeError(f'Source inference job {source_id} is missing its completed volume summary or geometry')
    if out.exists() and any(out.iterdir()): raise RuntimeError(f'Refusing to overwrite Stage G outputs: {out}')
    out.mkdir(parents=True, exist_ok=True)
    volumes = source / 'volumes'
    geometry = json.loads(geometry_path.read_text())
    inference = json.loads(inference_summary_path.read_text())
    tasks = list(inference.get('tasks') or [])
    if not tasks or not inference.get('outputs'):
        raise RuntimeError(f'Source inference job {source_id} has no completed prediction tasks/outputs')
    run_started = time.perf_counter()
    phase_timings = []
    phase_started = time.perf_counter()
    shape = tuple(int(x) for x in geometry['shape'])
    valid = np.load(volumes / 'prediction_valid.npy', mmap_mode='r')
    if tuple(valid.shape) != shape: raise RuntimeError('Validity mask shape disagrees with volume geometry')
    valid_count = sum(int(np.count_nonzero(valid[i:i + 16])) for i in range(0, shape[0], 16))
    stats, products = {}, {}
    for task in tasks:
        p = volumes / f'prediction_{task}.npy'
        arr = np.load(p, mmap_mode='r')
        if tuple(arr.shape) != shape: raise RuntimeError(f'{task}: prediction shape mismatch')
        # Work in bounded inline slabs. This avoids allocating a full-volume
        # boolean mask and keeps percentile samples in float32 instead of
        # eagerly widening the entire prediction volume to float64.
        value_chunks = []
        for slab_start in range(0, shape[0], 8):
            slab_stop = min(shape[0], slab_start + 8)
            slab = np.asarray(arr[slab_start:slab_stop], dtype=np.float32)
            slab_valid = np.asarray(valid[slab_start:slab_stop]) == 1
            finite_mask = np.isfinite(slab) & slab_valid
            if finite_mask.any():
                value_chunks.append(slab[finite_mask])
        values = np.concatenate(value_chunks) if value_chunks else np.empty(0, dtype=np.float32)
        percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99]) if values.size else [None] * 7
        # Current Stage F targets are bounded attributes: lithology P(sand),
        # Vsh and Phit. Record an explicit domain check; do not infer class
        # probability sums from the single lithology probability channel.
        domain_min, domain_max = 0.0, 1.0
        domain_tolerance = 1e-6
        below_domain = int(np.count_nonzero(values < domain_min - domain_tolerance))
        above_domain = int(np.count_nonzero(values > domain_max + domain_tolerance))
        stats[task] = {
            'finite_valid_voxels': int(values.size),
            'null_or_nonfinite_voxels_within_valid_mask': int(valid_count - values.size),
            'expected_value_domain': [domain_min, domain_max],
            'domain_tolerance': domain_tolerance,
            'values_below_domain': below_domain,
            'values_above_domain': above_domain,
            'value_domain_status': 'PASS' if below_domain == 0 and above_domain == 0 else 'FAIL',
            'min': float(values.min()) if values.size else None,
            'max': float(values.max()) if values.size else None,
            'mean': float(values.mean(dtype=np.float64)) if values.size else None,
            'std': float(values.std(dtype=np.float64)) if values.size else None,
            'percentiles': {f'p{q:02d}': (float(v) if v is not None else None)
                            for q, v in zip((1, 5, 25, 50, 75, 95, 99), percentiles)},
        }
        products[task] = {
            'npy': str(p.relative_to(source_job)),
            'source_job_relative_path': f'processing_jobs/{source_id}/{p.relative_to(source_job)}',
            'bytes': int(p.stat().st_size),
            'sha256': _sha256(p),
        }
    phase_timings.append({"phase": "native_statistics_and_product_hashes",
                          "seconds": round(time.perf_counter() - phase_started, 3)})
    phase_started = time.perf_counter()
    source_artifacts = {}
    foundation_id = str(geometry.get('foundation_job_id') or '').strip()
    foundation_manifest = (
        jobs_root / foundation_id / 'stages' / '00_inventory' / 'data_manifest.json'
        if foundation_id and Path(foundation_id).name == foundation_id else None
    )
    artifact_paths = {
        'inference_summary': inference_summary_path,
        'volume_geometry': geometry_path,
        'validity_mask': volumes / 'prediction_valid.npy',
        'trace_index_grid': volumes / 'trace_index_grid.npy',
    }
    for label, artifact_path in artifact_paths.items():
        if artifact_path.is_file():
            source_artifacts[label] = {
                'source_job_relative_path': f'processing_jobs/{source_id}/{artifact_path.relative_to(source_job)}',
                'bytes': int(artifact_path.stat().st_size),
                'sha256': _sha256(artifact_path),
            }
    if foundation_manifest is not None and foundation_manifest.is_file():
        source_artifacts['foundation_data_manifest'] = {
            'storage_root_relative_path': str(foundation_manifest.relative_to(storage_root)),
            'bytes': int(foundation_manifest.stat().st_size),
            'sha256': _sha256(foundation_manifest),
        }

    # Stage 10 is deliberately limited to prediction summaries, statistics,
    # geometry metadata and a manifest of the canonical NPY products.
    summary = {
        'schema_version': 1,
        'stage': '10_summary',
        'source_inference_job_id': source_id,
        'shape': list(shape),
        'axis_order': geometry['axis_order'],
        'geometry': geometry,
        'valid_voxels': valid_count,
        'candidate_voxels': int(valid.size),
        'valid_voxel_fraction': float(valid_count / valid.size) if valid.size else 0.0,
        'live_trace_count': (
            sum(int(np.count_nonzero(np.load(volumes / 'trace_index_grid.npy', mmap_mode='r')[i:i + 64] >= 0))
                for i in range(0, shape[0], 64))
            if (volumes / 'trace_index_grid.npy').is_file()
            else int((geometry.get('region') or {}).get('live_traces', 0))
        ),
        'attributes': stats,
        'products': products,
        'source_artifacts': source_artifacts,
    }
    (out / 'volume_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    manifest = {
        'schema_version': 1,
        'stage': '10_summary',
        'source_inference_job_id': source_id,
        'source_inference_job_relative_path': f'processing_jobs/{source_id}',
        'foundation_job_id': geometry.get('foundation_job_id'),
        'geometry': geometry,
        'products': products,
        'source_artifacts': source_artifacts,
        'summary': 'volume_summary.json',
    }
    (out / 'volume_product_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')

    # Re-hash source products after statistics are computed. This mirrors the
    # specialist summary's integrity audit without pretending this independent
    # packaging job has access to the specialist's in-run stage tables.
    source_hash_checks = []
    for task, item in products.items():
        source_path = source_job / item['npy']
        source_hash_checks.append(_sha256(source_path) == item['sha256'])
    for label, item in source_artifacts.items():
        if item.get('storage_root_relative_path'):
            source_path = storage_root / item['storage_root_relative_path']
        else:
            rel = item['source_job_relative_path'].split(f'processing_jobs/{source_id}/', 1)[-1]
            source_path = source_job / rel
        source_hash_checks.append(source_path.is_file() and _sha256(source_path) == item['sha256'])

    parity = geometry.get('stage03_parity') or {}
    parity_ok = (
        int(parity.get('rows_compared', 0)) > 0
        and parity.get('nan_pattern_equal') is True
        and float(parity.get('max_abs_difference', float('inf'))) == 0.0
    )
    domain_ok = all(item['value_domain_status'] == 'PASS' for item in stats.values())
    finite_ok = all(item['null_or_nonfinite_voxels_within_valid_mask'] == 0 for item in stats.values())
    audit_checks = [
        {'check': 'source_inference_summary_present', 'status': 'PASS' if inference_summary_path.is_file() else 'FAIL'},
        {'check': 'volume_geometry_present', 'status': 'PASS' if geometry_path.is_file() else 'FAIL'},
        {'check': 'prediction_shapes_match_geometry', 'status': 'PASS'},
        {'check': 'validity_mask_shape_matches_geometry', 'status': 'PASS'},
        {'check': 'stage03_patch_parity', 'status': 'PASS' if parity_ok else 'WARN',
         'detail': {'rows_compared': parity.get('rows_compared', 0),
                    'nan_pattern_equal': parity.get('nan_pattern_equal'),
                    'max_abs_difference': parity.get('max_abs_difference')}},
        {'check': 'valid_voxels_are_finite', 'status': 'PASS' if finite_ok else 'FAIL'},
        {'check': 'prediction_values_within_physical_domain_0_1',
         'status': 'PASS' if domain_ok else 'FAIL',
         'detail': {task: {'below': item['values_below_domain'], 'above': item['values_above_domain']}
                    for task, item in stats.items()}},
        {'check': 'source_artifacts_unchanged_during_summary',
         'status': 'PASS' if source_hash_checks and all(source_hash_checks) else 'FAIL'},
        {'check': 'all_prediction_products_have_checksums',
         'status': 'PASS' if all(len(item['sha256']) == 64 for item in products.values()) else 'FAIL'},
    ]
    audit_status = 'completed' if all(x['status'] == 'PASS' for x in audit_checks) else 'conditional'
    (out / 'summary_audit.json').write_text(json.dumps({
        'schema_version': 2, 'stage': '10_summary', 'source_inference_job_id': source_id,
        'status': audit_status,
        'checks': audit_checks,
        'limitations': [
            'Blind-well accuracy and conformal coverage are not recomputed by this packaging stage.',
            'Pipeline-wide status, code-fingerprint and claims-register audits require the upstream run context; '
            'this separate-job summary does not infer them from missing local stage folders.',
        ],
    }, indent=2) + '\n')
    # Specialist-aligned Stage 10 package: tables/, figures/, model_cards/, reproduce/.
    _write_specialist_summary_package(
        out=out, storage_root=storage_root, source_job=source_job, source_id=source_id,
        foundation_id=foundation_id, geometry=geometry, inference=inference,
        release_id=str(inference.get('release_id') or inference.get('model_release_id') or ''), stats=stats, summary=summary,
        audit_checks=audit_checks, source_artifacts=source_artifacts, products=products,
        source_hash_checks=source_hash_checks, parity=parity, valid_count=valid_count,
    )
    phase_timings.append({"phase": "stage10_audit_and_dashboard",
                          "seconds": round(time.perf_counter() - phase_started, 3)})
    phase_started = time.perf_counter()

    # Generate native-grid seismic sections from the registered full-stack SEG-Y.
    # This is a read-only visualization product; prediction NPYs remain canonical.
    section_info = None
    if options.get("generate_sections", True) and domain_ok:
        trace_grid = np.load(volumes / "trace_index_grid.npy", mmap_mode="r")
        section_info = _build_seismic_sections(
            storage_root=storage_root, geometry=geometry, source_id=source_id,
            source_job=source_job, volumes=volumes, out=out, tasks=tasks,
            shape=shape, valid=valid, trace_grid=trace_grid,
        )
        # Embed preview PNGs as data URIs. Relative paths are unreliable when
        # dashboards are opened through job viewers, downloaded individually,
        # or served from a different base URL. Keep the full-resolution section
        # files in figures/sections/ for independent access.
        dashboard_path = out / "dashboard.html"
        dashboard = dashboard_path.read_text(encoding="utf-8")
        gallery = ["<h2>Seismic sections and time slices</h2>",
                   "<p class='muted'>Left: registered full-stack amplitude. Right: predicted attribute. "
                   "Sections use native IL/XL bins; no affine XY interpolation is applied.</p>"]
        for task in tasks:
            gallery.append(f"<h3>{html.escape(str(task))}</h3>")
            previews = [
                ("inline", f"inline_{section_info['inline_indices'][len(section_info['inline_indices']) // 2]['inline']}.png"),
                ("crossline", f"crossline_{section_info['crossline_indices'][len(section_info['crossline_indices']) // 2]['crossline']}.png"),
            ]
            mid_time = section_info["time_slices"][len(section_info["time_slices"]) // 2]
            previews.append((f"Time slice {mid_time['time_ms']:.2f} ms",
                             f"time_slice_{mid_time['sample_index']:04d}.png"))
            for label, filename in previews:
                image_path = out / "figures" / "sections" / str(task) / filename
                image_uri = _png_data_uri(image_path)
                gallery.append(
                    f"<figure><figcaption>{html.escape(label)}</figcaption>"
                    f"<img loading='lazy' alt='{html.escape(str(task))} {html.escape(label)}' "
                    f"src='{image_uri}'></figure>"
                )
        dashboard = dashboard.replace("<h2>Recorded pipeline stages</h2>",
                                      "".join(gallery) + "<h2>Recorded pipeline stages</h2>")
        dashboard_path.write_text(dashboard, encoding="utf-8")

    # Record hashes for the Stage 10 handoff itself, analogous to the
    # specialist output-integrity table. The manifest is excluded to avoid a
    # self-referential checksum.
    if section_info is not None:
        manifest["seismic_sections"] = section_info
        audit_path = out / "summary_audit.json"
        audit_payload = _safe_json(audit_path)
        audit_payload.setdefault("checks", []).append({
            "check": "native_grid_seismic_sections",
            "status": "PASS" if section_info.get("section_count", 0) > 0 else "FAIL",
            "detail": {"section_count": section_info.get("section_count", 0),
                       "well_trajectory_overlay": section_info.get("well_trajectory_overlay")},
        })
        audit_payload["status"] = "completed" if all(
            item.get("status") == "PASS" for item in audit_payload["checks"]
        ) else "conditional"
        audit_path.write_text(json.dumps(audit_payload, indent=2) + "\n", encoding="utf-8")
        section_check = audit_payload["checks"][-1]
        _csv_write(out / "tables" / "acceptance_checks.csv",
                   [*list(csv.DictReader((out / "tables" / "acceptance_checks.csv").open(encoding="utf-8"))),
                    section_check])
        claim_path = out / "tables" / "claims_register.csv"
        claim_rows = list(csv.DictReader(claim_path.open(encoding="utf-8")))
        claim_id = f"Q{len(claim_rows) + 1:02d}"
        claim_rows.append({"id": claim_id, "claim": section_check["check"],
                           "verdict": section_check["status"],
                           "evidence": json.dumps(section_check.get("detail", {})),
                           "rule": "Registered full-stack trace mapping and native IL/XL section generation",
                           "source": "seismic_sections.json / seismic_sections.csv"})
        _csv_write(claim_path, claim_rows)
        dashboard_path = out / "dashboard.html"
        dashboard = dashboard_path.read_text(encoding="utf-8")
        audit_heading = "<h2>Audit claims</h2>"
        row_html = (f"<tr><td>{claim_id}</td><td>{section_check['check']}</td>"
                    f"<td>{section_check['status']}</td></tr>")
        # Append the section-generation check to the audit table, not the stage table.
        audit_pos = dashboard.find(audit_heading)
        if audit_pos >= 0:
            table_end = dashboard.find("</table>", audit_pos)
            if table_end >= 0:
                dashboard = dashboard[:table_end] + row_html + dashboard[table_end:]
                dashboard_path.write_text(dashboard, encoding="utf-8")

    manifest["stage11_handoff"] = "../11_volume_export/native_volume_export_manifest.json"
    manifest['summary_artifacts'] = {
        p.relative_to(out).as_posix(): {'bytes': int(p.stat().st_size), 'sha256': _sha256(p)}
        for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'volume_product_manifest.json'
    }
    stage10_manifest_path = out / 'volume_product_manifest.json'
    stage10_manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    phase_timings.append({"phase": "seismic_sections_and_stage10_manifest",
                          "seconds": round(time.perf_counter() - phase_started, 3)})
    phase_started = time.perf_counter()

    # Stage 11 remains a real handoff/QC stage, but its products are native
    # NPY references and sidecars rather than derived SEG-Y files.
    from .s11_volume_export import write_native_volume_export_stage

    stage11 = write_native_volume_export_stage(
        job_dir=job_dir, storage_root=storage_root, source_job=source_job, source_id=source_id,
        volumes=volumes, geometry=geometry, tasks=tasks, products=products,
        source_artifacts=source_artifacts, stats=stats, stage10_manifest=stage10_manifest_path,
        domain_ok=domain_ok, segy_export_enabled=bool(options.get("segy_export", {}).get("enabled", True)
            if isinstance(options.get("segy_export"), dict) else options.get("segy_export", True)),
    )
    phase_timings.append({"phase": "stage11_native_export_qc",
                          "seconds": round(time.perf_counter() - phase_started, 3)})
    phase_timings.append({"phase": "total", "seconds": round(time.perf_counter() - run_started, 3)})
    performance = {"schema_version": 1, "stage": "10_summary", "timings": phase_timings,
                   "optimizations": ["process-local SHA-256 cache keyed by file identity",
                                     "chunked finite-value extraction",
                                     "reused seismic section reads across prediction attributes",
                                     "pre-indexed well overlay samples"]}
    performance_path = out / "summary_performance.json"
    performance_path.write_text(json.dumps(performance, indent=2) + '\n')
    # Include the timing sidecar in the Stage 10 artifact inventory, then
    # refresh the Stage 11 pointer to the final manifest checksum.
    manifest = _safe_json(stage10_manifest_path)
    manifest.setdefault("summary_artifacts", {})["summary_performance.json"] = {
        "bytes": int(performance_path.stat().st_size), "sha256": _sha256(performance_path)}
    stage10_manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    stage11_manifest_path = job_dir / "stages" / "11_volume_export" / "native_volume_export_manifest.json"
    stage11_manifest = _safe_json(stage11_manifest_path)
    stage11_manifest.setdefault("stage10_manifest", {})["sha256"] = _sha256(stage10_manifest_path)
    stage11_manifest_path.write_text(json.dumps(stage11_manifest, indent=2) + "\n", encoding="utf-8")
    job_summary = {
        'status': 'completed' if domain_ok and stage11["status"] == "completed" else 'conditional',
        'source_inference_job_id': source_id,
        'summary_stage': 'stages/10_summary',
        'summary': 'stages/10_summary/volume_summary.json',
        'product_manifest': 'stages/10_summary/volume_product_manifest.json',
        'volume_export_stage': stage11,
        'prediction_format': 'native_npy',
        'derived_segy_export': stage11['derived_segy_export'],
    }
    (job_dir / 'geogg_volume_products_summary.json').write_text(json.dumps(job_summary, indent=2) + '\n')
    if not domain_ok:
        raise RuntimeError('Stage 10 value-domain audit failed; inspect summary_audit.json, claims_register.csv and Stage 11 QC')
    return 0
