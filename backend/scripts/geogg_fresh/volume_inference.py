"""Label-independent, bounded-memory inference over registered seismic SEG-Y volumes.

The module deliberately writes prediction cubes rather than a multi-billion-row CSV.
Each cube uses [inline_bin, crossline_bin, time_sample] and is accompanied by geometry
metadata so Stage G can export a correctly indexed SEG-Y volume.
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .extraction import fill_patches, patch_window
from .features import normalized_difference, stack_attributes, stack_zero_missing, trend_vs_order
from .segy import SegyFile
from .stage_compat import StageBlocked
from .provenance import sha256_file

STACKS = ("full", "near", "mid", "far", "ultrafar")


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").casefold()).strip("_")


def resolve_stack_records(records, required: tuple[str, ...] | list[str] = STACKS, *, require_existing: bool = True) -> dict[str, dict[str, Any]]:
    """Resolve one registered SEG-Y input per amplitude stack, using names/metadata only."""
    resolved: dict[str, dict[str, Any]] = {}
    aliases = {
        "full": {"full", "full_stack", "stack_full", "fullstack"},
        "near": {"near", "near_stack", "stack_near", "nearstack"},
        "mid": {"mid", "middle", "mid_stack", "stack_mid"},
        "far": {"far", "far_stack", "stack_far", "farstack"},
        "ultrafar": {"ultrafar", "ultra_far", "ultra_far_stack", "stack_ultrafar"},
    }
    for item in records:
        get = item.get if isinstance(item, dict) else lambda k, d=None: getattr(item, k, d)
        path = Path(str(get("path", "") or "")).expanduser()
        if path.suffix.casefold() not in {".sgy", ".segy"} or (require_existing and not path.is_file()):
            continue
        metadata = get("metadata", {}) or {}
        tokens = {_norm(path.stem), _norm(get("name", ""))}
        if isinstance(metadata, dict):
            for key in ("volume", "stack", "seismic_role", "channel", "volume_name"):
                if metadata.get(key):
                    tokens.add(_norm(metadata[key]))
        matches = [role for role, names in aliases.items()
                   if any(token in names or any(f"_{name}_" in f"_{token}_" for name in names)
                          for token in tokens)]
        if len(matches) > 1:
            raise StageBlocked(f"SEG-Y input {path.name} ambiguously matches stacks {matches}")
        if not matches:
            continue
        role = matches[0]
        if role in resolved:
            raise StageBlocked(f"Multiple registered SEG-Y datasets match the '{role}' stack")
        resolved[role] = {"path": path.resolve(), "dataset_id": get("dataset_id"),
                          "name": get("name") or path.name}
    missing = sorted(set(required) - set(resolved))
    if missing:
        raise StageBlocked("Survey-volume inference requires one registered SEG-Y for each amplitude stack; "
                           f"missing {missing}. Register/select full, near, mid, far and ultrafar volumes.")
    return resolved


def _same_sampling(reference, candidate, role: str) -> None:
    """Require a common time axis, but allow each stack its own IL/XL/XY geometry.

    Specialist Stage 03 defines patch grid indices on the full-stack geometry and
    maps those IL/XL coordinates independently into each stack geometry. Requiring
    identical trace grids or affine XY transforms would reject valid surveys.
    """
    fields = ("dt_ms", "t0_ms", "ns")
    differences = [f for f in fields if not np.isclose(
        float(getattr(reference, f)), float(getattr(candidate, f)), rtol=0.0, atol=1e-6
    )]
    if differences:
        raise StageBlocked(
            f"{role} SEG-Y time sampling differs from the full-stack reference: {differences}. "
            "Time-axis resampling is not implicit in inference."
        )



def _xy_mapping_quality(geometry) -> dict[str, Any]:
    """Describe whether affine IL/XL-to-XY mapping is safe for downstream use."""
    threshold = 0.25 * min(geometry.bin_spacing())
    supported = (
        geometry.affine is not None
        and geometry.affine_residual_max <= threshold
        and not geometry.flags
    )
    return {
        "native_grid": "IL/XL",
        "xy_mapping_supported": bool(supported),
        "xy_mapping_status": "supported" if supported else "requires_trace_coordinate_qc",
        "affine_residual_threshold_m": float(threshold),
        "affine_residual_max_m": float(geometry.affine_residual_max),
        "affine_residual_rms_m": float(geometry.affine_residual_rms),
        "geometry_flags": list(geometry.flags),
    }


def _patch_keep_mask(patches: np.ndarray, *, deep: bool, max_nan_fraction: float) -> np.ndarray:
    """Apply the specialist missing-patch rule only to deep models.

    Tabular Stage 04 features retain per-feature missingness and the released
    frozen transform handles it; a global patch completeness gate is not used.
    """
    if not deep:
        return np.ones(len(patches), dtype=bool)
    finite_fraction = np.isfinite(patches).reshape(len(patches), -1).mean(axis=1)
    return (1.0 - finite_fraction) <= float(max_nan_fraction)


def _foundation_stage_records(job_dir: Path) -> dict:
    summary_path = job_dir / "geogg_foundations_summary.json"
    provenance_path = job_dir / "provenance" / "00_inventory.json"
    if not summary_path.is_file() or summary_path.is_symlink():
        raise StageBlocked(f"Foundation prerequisite has no geogg_foundations_summary.json: {job_dir.name}")
    if not provenance_path.is_file() or provenance_path.is_symlink():
        raise StageBlocked(f"Foundation prerequisite has no Stage 00 provenance: {job_dir.name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "succeeded":
        raise StageBlocked(f"Foundation prerequisite {job_dir.name} did not succeed")
    stages = {str(row.get("stage_id")): str(row.get("status"))
              for row in summary.get("stages", []) if isinstance(row, dict)}
    required_stages = ["00_inventory", "01_well_ties", "02_labels", "03_extraction",
                       "04_features", "05_training_tables", "06_feature_selection", "07_models"]
    missing = [stage for stage in required_stages if stages.get(stage) not in {"COMPLETED", "CONDITIONAL"}]
    if missing:
        raise StageBlocked(f"Foundation prerequisite {job_dir.name} is incomplete; missing/failed stages: {missing}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    return {str(item.get("role")): item for item in provenance.get("inputs", [])
            if isinstance(item, dict) and item.get("role")}


def _resolve_originating_stack_records(foundation_job: Path, foundation_job_id: str) -> dict[str, dict[str, Any]]:
    """Resolve stack files directly from the originating Stage 00 data manifest.

    The manifest is the immutable inventory snapshot produced by the foundation
    job. The inference worker is intentionally independent of the APGCE web app,
    database package, and deployment directory layout.
    """
    manifest_path = foundation_job / "stages" / "00_inventory" / "data_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise StageBlocked(f"Foundation job {foundation_job_id} has no Stage 00 data_manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StageBlocked(f"Cannot read foundation data manifest: {exc}") from exc

    manifest_records = []
    for item in manifest.get("inputs", []):
        if not isinstance(item, dict) or str(item.get("kind", "")).casefold() != "seismic":
            continue
        dataset_id = str(item.get("dataset_id") or "").strip()
        raw_path = str(item.get("path") or "").strip()
        if not dataset_id or not raw_path:
            continue
        manifest_records.append({
            "path": Path(raw_path).expanduser(),
            "dataset_id": dataset_id,
            "name": item.get("name") or "",
            "metadata": item.get("metadata") or {},
            "manifest_sha256": str(item.get("sha256") or "").lower(),
        })

    # Role identification uses inventory names/metadata; path existence is
    # checked below so errors can identify the exact missing dataset.
    selected = resolve_stack_records(manifest_records, required=STACKS, require_existing=False)
    source_by_id = {str(row["dataset_id"]): row for row in manifest_records}
    resolved: dict[str, dict[str, Any]] = {}
    for role, item in selected.items():
        item = {**item, **source_by_id.get(str(item["dataset_id"]), {})}
        path = Path(item["path"]).expanduser().resolve()
        if not path.is_file():
            raise StageBlocked(
                f"Foundation inventory {role} dataset {item['dataset_id']} points to an unavailable file: {path}. "
                "The Stage 00 manifest must reference a path mounted and readable by the inference worker."
            )
        expected_hash = str(item.get("manifest_sha256") or "").lower()
        if not expected_hash:
            raise StageBlocked(f"Foundation inventory has no SHA-256 checksum for {role} dataset {item['dataset_id']}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise StageBlocked(
                f"Foundation inventory {role} dataset {item['dataset_id']} failed its SHA-256 check"
            )
        resolved[role] = {
            "path": path,
            "dataset_id": str(item["dataset_id"]),
            "name": item.get("name") or path.name,
            "metadata": item.get("metadata") or {},
            "manifest_sha256": expected_hash,
        }
    return resolved


def _resolve_foundation_job(ctx, records: dict[str, dict[str, Any]], foundation_job_id: str) -> Path:
    """Resolve and validate the exact foundation job recorded by the model release.

    A release is created by one foundation job, so inference must follow that
    immutable provenance link rather than searching for a job by timestamp or hash.
    SEG-Y hashes are still checked to prevent applying the release to another survey.
    """
    selected_id = str(foundation_job_id or "").strip()
    if not selected_id or Path(selected_id).name != selected_id or selected_id in {".", ".."}:
        raise StageBlocked("The model release manifest has no valid foundation_job_id")
    storage_root = ctx.base.job_dir.parent.parent.resolve()
    jobs_root = (storage_root / "processing_jobs").resolve()
    if jobs_root.parent != storage_root or not jobs_root.is_dir():
        raise StageBlocked("Configured backend processing_jobs storage root is unavailable")
    candidate_path = jobs_root / selected_id
    if candidate_path.is_symlink() or not candidate_path.is_dir() or candidate_path.resolve().parent != jobs_root:
        raise StageBlocked(
            f"The foundation job recorded by the model release is unavailable or unsafe: {selected_id}"
        )
    # Stage 00 provenance records inventory inputs as dataset records (dataset_id,
    # kind, name, path, sha256); it does not guarantee synthetic role keys such as
    # ``seismic:full``. The authoritative role-to-dataset mapping and checksum are
    # in data_manifest.json and are verified by _resolve_originating_stack_records.
    # Do not compare against a fabricated role-keyed provenance map here.
    _foundation_stage_records(candidate_path)  # validate successful foundation stages
    return candidate_path.resolve()


def _verify_stage03_patch_parity(ctx, foundation_job: Path, reference, geometries, open_files,
                                 channels: list[str], offsets, *, max_rows: int = 32) -> dict:
    """Compare fresh volume cuts with a deterministic sample of Stage 03 patches."""
    stage_dir = foundation_job / "stages" / "03_extraction"
    index_path = stage_dir / "tables" / "sample_index.csv"
    manifest_candidates = sorted(stage_dir.glob("patches/patch_manifest_*.json"))
    if not index_path.is_file() or len(manifest_candidates) != 1:
        raise StageBlocked("Foundation prerequisite must contain one Stage 03 sample_index.csv and patch manifest")
    manifest = json.loads(manifest_candidates[0].read_text(encoding="utf-8"))
    patch_name = str(manifest.get("file") or "")
    if not patch_name or Path(patch_name).name != patch_name:
        raise StageBlocked("Stage 03 patch manifest contains an unsafe patch filename")
    patch_path = stage_dir / "patches" / patch_name
    if not patch_path.is_file() or sha256_file(patch_path) != manifest.get("sha256"):
        raise StageBlocked("Foundation Stage 03 patch tensor is missing or fails its manifest checksum")
    index = pd.read_csv(index_path)
    required = {"patch_row", "patch_status", "grid_i", "grid_j", "desired_twt_ms"}
    if not required.issubset(index.columns):
        raise StageBlocked(f"Stage 03 sample_index.csv lacks parity columns: {sorted(required - set(index.columns))}")
    source_channels = list(manifest.get("channels") or [])
    if not set(channels).issubset(source_channels):
        raise StageBlocked("Stage 03 patch channels do not contain the selected model channels")
    valid = index["patch_status"].astype(str).eq("extracted")
    valid &= pd.to_numeric(index["patch_row"], errors="coerce").notna()
    rows = index.loc[valid].copy()
    if rows.empty:
        raise StageBlocked("Stage 03 has no extracted patches available for parity verification")
    # Evenly distributed deterministic sample across the extracted row population.
    take = np.linspace(0, len(rows) - 1, min(max_rows, len(rows))).round().astype(int)
    rows = rows.iloc[take].reset_index(drop=True)
    patch_rows = rows["patch_row"].to_numpy(dtype=int)
    source_array = np.load(patch_path, mmap_mode="r", allow_pickle=False)
    if source_array.ndim != 5 or source_array.shape[1] != len(source_channels):
        raise StageBlocked("Stage 03 patch tensor shape disagrees with its manifest channel list")
    if np.any(patch_rows < 0) or np.any(patch_rows >= source_array.shape[0]):
        raise StageBlocked("Stage 03 sample index contains patch rows outside its tensor")
    ci = rows["grid_i"].to_numpy(dtype=int)
    cj = rows["grid_j"].to_numpy(dtype=int)
    ck = np.round((rows["desired_twt_ms"].to_numpy(float) - reference.t0_ms) / reference.dt_ms).astype(int)
    actual = _patch_batch(open_files, geometries, reference, ci, cj, ck, offsets, channels)
    source_idx = [source_channels.index(ch) for ch in channels]
    expected = np.asarray(source_array[patch_rows][:, source_idx], dtype=np.float32)
    same_nan = np.array_equal(np.isfinite(actual), np.isfinite(expected))
    finite = np.isfinite(actual) & np.isfinite(expected)
    max_abs = float(np.max(np.abs(actual[finite] - expected[finite]))) if finite.any() else 0.0
    if not same_nan or max_abs > 1e-5:
        raise StageBlocked(
            f"Stage 09 extraction failed Stage 03 parity: rows={len(rows)}, nan_pattern_equal={same_nan}, "
            f"max_abs_difference={max_abs:.6g}"
        )
    ctx.add_input(index_path, "foundation prerequisite: Stage 03 sample index")
    ctx.add_input(patch_path, "foundation prerequisite: Stage 03 parity patch tensor")
    return {"rows_compared": int(len(rows)), "nan_pattern_equal": same_nan,
            "max_abs_difference": max_abs, "patch_channels": channels}


def _required_channels(features: list[str]) -> list[str]:
    found = set()
    for name in features:
        if "__" in name:
            prefix = name.split("__", 1)[0]
            if prefix in STACKS:
                found.add(prefix)
        for pair in ("near_far", "near_ultrafar", "mid_far"):
            if name.startswith(pair + "__"):
                found.update(pair.split("_"))
        if name.startswith("partials__"):
            found.update(("near", "mid", "far", "ultrafar"))
    return [x for x in STACKS if x in found]


def _features_from_patches(patches: np.ndarray, channels: list[str], required_features: list[str],
                           dt_ms: float, zero_run_min: int = 5) -> pd.DataFrame:
    """Reproduce Stage 04 stack feature definitions for a batch of unlabeled patches."""
    result: dict[str, np.ndarray] = {}
    by_stack: dict[str, dict[str, np.ndarray]] = {}
    for stack in _required_channels(required_features):
        if stack not in channels:
            raise StageBlocked(f"Required model stack '{stack}' is absent from the input patch")
        values = np.asarray(patches[:, channels.index(stack)], dtype=np.float64)
        _, missing = stack_zero_missing(values, zero_run_min)
        values[missing] = np.nan
        by_stack[stack] = stack_attributes(values, dt_ms)
        for name, vector in by_stack[stack].items():
            result[f"{stack}__{name}"] = vector
    pairs = (("near", "far"), ("near", "ultrafar"), ("mid", "far"))
    for a, b in pairs:
        if a in by_stack and b in by_stack:
            result[f"{a}_{b}__amp_diff"] = by_stack[b]["amp_c"] - by_stack[a]["amp_c"]
            result[f"{a}_{b}__rms_nd"] = normalized_difference(by_stack[b]["rms_long"],
                                                               by_stack[a]["rms_long"])
    partial = [s for s in ("near", "mid", "far", "ultrafar") if s in by_stack]
    if len(partial) >= 3:
        amps = np.column_stack([by_stack[s]["amp_c"] for s in partial])
        slope, intercept = trend_vs_order(amps)
        result["partials__amp_slope_vs_order"] = slope
        result["partials__amp_intercept_vs_order"] = intercept
    missing_features = sorted(set(required_features) - set(result))
    if missing_features:
        raise StageBlocked(f"Volume feature builder cannot reproduce released predictors: {missing_features}")
    return pd.DataFrame({name: result[name] for name in required_features})


def _iter_centres(geom, batch_size: int):
    """Yield live trace cells in batches, each with its full time axis.

    Grouping all samples of adjacent IL/XL cells together lets each batch read its
    overlapping 5x5 neighbourhood once instead of rereading it for every time sample.
    """
    live_i, live_j = np.where(geom.grid >= 0)
    cell_batch = max(1, int(batch_size) // int(geom.ns))
    time = np.arange(geom.ns, dtype=np.int64)
    for start in range(0, len(live_i), cell_batch):
        stop = min(start + cell_batch, len(live_i))
        ci = np.repeat(live_i[start:stop].astype(np.int64), geom.ns)
        cj = np.repeat(live_j[start:stop].astype(np.int64), geom.ns)
        ck = np.tile(time, stop - start)
        yield ci, cj, ck



def _iter_cell_batches(live_i: np.ndarray, live_j: np.ndarray, ns: int, batch_size: int):
    """Yield all time samples for bounded batches of live IL/XL cells."""
    cell_batch = max(1, int(batch_size) // int(ns))
    time = np.arange(ns, dtype=np.int64)
    for start in range(0, len(live_i), cell_batch):
        stop = min(start + cell_batch, len(live_i))
        ci = np.repeat(live_i[start:stop].astype(np.int64), ns)
        cj = np.repeat(live_j[start:stop].astype(np.int64), ns)
        ck = np.tile(time, stop - start)
        yield ci, cj, ck


def _iter_spatial_tiles(geom, tile_i: int, tile_j: int):
    """Yield non-overlapping spatial output tiles and their live reference cells.

    Patch halos are read by _patch_batch around each centre; only centre voxels are
    written. This is the same halo rule used by the specialist SectionCutter.
    """
    ni, nj = geom.grid.shape
    tile_id = 0
    for i0 in range(0, ni, tile_i):
        for j0 in range(0, nj, tile_j):
            i1, j1 = min(i0 + tile_i, ni), min(j0 + tile_j, nj)
            local_i, local_j = np.where(geom.grid[i0:i1, j0:j1] >= 0)
            yield tile_id, (i0, i1, j0, j1), (local_i + i0, local_j + j0)
            tile_id += 1


def _resolve_volume_region(reference, cfg: dict):
    """Resolve an optional IL/XL and/or affine-XY ROI into a cropped reference grid.

    IL/XL bounds are inclusive survey coordinates. XY bounds are inclusive map
    coordinates and use the Stage 00 affine transform; XY selection is rejected
    when the stored geometry is materially non-affine.
    """
    region = cfg.get("volume_region") or {}
    if not isinstance(region, dict):
        raise StageBlocked("inference.volume_region must be a mapping")
    ni, nj = reference.grid.shape
    i0, i1, j0, j1 = 0, ni, 0, nj

    def index_range(key, minimum, step, size):
        value = region.get(key)
        if value is None:
            return 0, size
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise StageBlocked(f"volume_region.{key} must be [minimum, maximum]")
        lo, hi = map(float, value)
        if lo > hi:
            raise StageBlocked(f"volume_region.{key} minimum exceeds maximum")
        first = max(0, int(np.ceil((lo - minimum) / step - 1e-9)))
        stop = min(size, int(np.floor((hi - minimum) / step + 1e-9)) + 1)
        if first >= stop:
            raise StageBlocked(f"volume_region.{key} does not intersect the survey")
        return first, stop

    i0, i1 = index_range("inline", reference.il_min, reference.il_step, ni)
    j0, j1 = index_range("crossline", reference.xl_min, reference.xl_step, nj)
    region_grid = reference.grid[i0:i1, j0:j1].copy()
    xy = region.get("xy")
    xy_summary = None
    if xy is not None:
        if not isinstance(xy, dict):
            raise StageBlocked("volume_region.xy must contain x_min, x_max, y_min and y_max")
        required = ("x_min", "x_max", "y_min", "y_max")
        if any(k not in xy for k in required):
            raise StageBlocked("volume_region.xy requires x_min, x_max, y_min and y_max")
        xmin, xmax, ymin, ymax = (float(xy[k]) for k in required)
        if xmin > xmax or ymin > ymax:
            raise StageBlocked("volume_region.xy minimum bounds must not exceed maximum bounds")
        if reference.affine_residual_max > 0.25 * min(reference.bin_spacing()):
            raise StageBlocked("XY region selection is unsafe because the stored survey geometry is materially non-affine; use IL/XL bounds")
        il = reference.il_min + np.arange(i0, i1, dtype=float) * reference.il_step
        xl = reference.xl_min + np.arange(j0, j1, dtype=float) * reference.xl_step
        il_grid, xl_grid = np.meshgrid(il, xl, indexing="ij")
        x, y = reference.ilxl_to_xy(il_grid, xl_grid)
        xy_mask = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)
        if not xy_mask.any():
            raise StageBlocked("Configured XY volume_region does not intersect the survey")
        rows = np.flatnonzero(xy_mask.any(axis=1))
        cols = np.flatnonzero(xy_mask.any(axis=0))
        row0, row1 = int(rows[0]), int(rows[-1]) + 1
        col0, col1 = int(cols[0]), int(cols[-1]) + 1
        region_grid = region_grid[row0:row1, col0:col1]
        xy_mask = xy_mask[row0:row1, col0:col1]
        i0, i1 = i0 + row0, i0 + row1
        j0, j1 = j0 + col0, j0 + col1
        region_grid[~xy_mask] = -1
        xy_summary = {"x_min": xmin, "x_max": xmax, "y_min": ymin, "y_max": ymax,
                      "selection_method": "stage00_affine_transform",
                      "affine_residual_max_m": float(reference.affine_residual_max)}
    if not np.any(region_grid >= 0):
        raise StageBlocked("Configured volume_region contains no live full-stack traces")
    resolved = {
        "inline_index_range": [i0, i1], "crossline_index_range": [j0, j1],
        "inline_range": [reference.il_min + i0 * reference.il_step,
                         reference.il_min + (i1 - 1) * reference.il_step],
        "crossline_range": [reference.xl_min + j0 * reference.xl_step,
                            reference.xl_min + (j1 - 1) * reference.xl_step],
        "xy": xy_summary,
        "live_traces": int(np.count_nonzero(region_grid >= 0)),
        "full_survey_shape": [ni, nj, int(reference.ns)],
    }
    return i0, i1, j0, j1, region_grid, resolved


def _summarize_tile_valid_counts(tile_valid_counts: dict[str, int], completed_tiles: set[str],
                                tile_count: int, expected_live: int) -> int:
    """Return exact prediction count without scanning the full validity memmap."""
    if len(completed_tiles) != tile_count:
        raise StageBlocked(f"Inference checkpoint incomplete: {len(completed_tiles)}/{tile_count} tiles")
    if len(tile_valid_counts) != tile_count or set(tile_valid_counts) != completed_tiles:
        raise StageBlocked(f"Inference checkpoint has {len(tile_valid_counts)}/{tile_count} tile valid counts")
    if any(int(value) < 0 for value in tile_valid_counts.values()):
        raise StageBlocked("Inference checkpoint contains a negative tile valid-voxel count")
    total = sum(int(value) for value in tile_valid_counts.values())
    if total > int(expected_live):
        raise StageBlocked(f"Prediction validity exceeds live survey voxels: {total}>{expected_live}")
    return total


def _checkpoint_write(path: Path, payload: dict) -> None:
    """Atomically persist resumable tile state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(temp, path)


def _volume_fingerprint(ctx, selection, records, reference, channels, tasks, cfg) -> str:
    import hashlib
    payload = {
        'inference_engine_version': 4,
        'release_id': str(ctx.release_id), 'selection': selection, 'channels': channels,
        'tasks': tasks, 'shape': list(reference.grid.shape) + [int(reference.ns)],
        'geometry': {'il_min': reference.il_min, 'il_step': reference.il_step,
                     'xl_min': reference.xl_min, 'xl_step': reference.xl_step,
                     't0_ms': reference.t0_ms, 'dt_ms': reference.dt_ms},
        'inputs': {role: {'dataset_id': item.get('dataset_id'),
                          'sha256': item.get('manifest_sha256') or sha256_file(item['path'])}
                   for role, item in sorted(records.items()) if role in set(['full', *channels])},
        'tile_shape': [int(cfg.get('volume_tile_inline', 32)), int(cfg.get('volume_tile_crossline', 32))],
        'deep_precision': str(cfg.get('deep_precision', 'fp32')).casefold(),
        'deep_max_nan_fraction': float(cfg.get('deep_max_nan_fraction', 0.2)),
        'volume_region': cfg.get('volume_region', {}),
        'resolved_volume_region': cfg.get('_resolved_volume_region', {}),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

def _patch_batch(open_files, geometries, reference, ci, cj, ck, offsets, channels):
    oi, oj, _ = offsets
    n = len(ci)
    patches = np.full((n, len(channels), len(oi), len(oj), offsets[2].size), np.nan, dtype=np.float32)
    # The reference grid is identical for every selected stack; map every touched cell to its trace.
    ii = (ci[:, None, None] + oi[None, :, None]).repeat(oj.size, axis=2).ravel()
    jj = (cj[:, None, None] + oj[None, None, :]).repeat(oi.size, axis=1).ravel()
    cells = np.unique(np.stack([ii, jj], axis=1), axis=0)
    lut_cells = {(int(a), int(b)): i for i, (a, b) in enumerate(cells)}
    lookup = lambda i, j: lut_cells.get((int(i), int(j)), -1)
    for c, role in enumerate(channels):
        geom = geometries[role]
        trace_ids = geom.trace_index(reference.il_min + cells[:, 0] * reference.il_step,
                                     reference.xl_min + cells[:, 1] * reference.xl_step)
        traces = open_files[role].read_traces(trace_ids)
        fill_patches(patches, c, ci, cj, ck, lookup, traces, geom.ns, offsets)
    return patches



def _read_tile_trace_blocks(open_files, geometries, reference, channels,
                            global_i0: int, global_i1: int,
                            global_j0: int, global_j1: int, halo: int = 2):
    """Read each channel's tile plus spatial halo once, in native trace order.

    The full-stack grid defines the requested IL/XL coordinates. Each input
    channel resolves those coordinates through its own SurveyGeometry.
    """
    grid_i = np.arange(global_i0 - halo, global_i1 + halo, dtype=np.int64)
    grid_j = np.arange(global_j0 - halo, global_j1 + halo, dtype=np.int64)
    ii, jj = np.meshgrid(grid_i, grid_j, indexing="ij")
    il = reference.il_min + ii.ravel() * reference.il_step
    xl = reference.xl_min + jj.ravel() * reference.xl_step
    blocks = {}
    for role in channels:
        geom = geometries[role]
        trace_ids = geom.trace_index(il, xl)
        traces = open_files[role].read_traces(trace_ids)
        blocks[role] = traces.reshape(len(grid_i), len(grid_j), geom.ns)
    return blocks, int(grid_i[0]), int(grid_j[0])


def _extract_patches_from_tile(blocks, channels, tile_i0: int, tile_j0: int,
                               ci: np.ndarray, cj: np.ndarray, ck: np.ndarray, offsets):
    """Vectorised Stage-03-compatible patch gather from a cached tile+halo.

    This replaces the former Python loop over every centre and every 5x5
    neighbour. Time samples outside the native volume remain NaN.
    """
    oi, oj, ok = offsets
    local_i = np.asarray(ci, dtype=np.int64) - int(tile_i0)
    local_j = np.asarray(cj, dtype=np.int64) - int(tile_j0)
    sample = np.asarray(ck, dtype=np.int64)[:, None] + ok[None, :]
    sample_valid = (sample >= 0) & (sample < next(iter(blocks.values())).shape[2])
    sample_safe = np.clip(sample, 0, next(iter(blocks.values())).shape[2] - 1)
    row_idx = local_i[:, None, None, None] + oi[None, :, None, None]
    col_idx = local_j[:, None, None, None] + oj[None, None, :, None]
    time_idx = sample_safe[:, None, None, :]
    patches = np.empty((len(ci), len(channels), len(oi), len(oj), len(ok)), dtype=np.float32)
    for channel_index, role in enumerate(channels):
        values = blocks[role][row_idx, col_idx, time_idx]
        patches[:, channel_index] = np.where(sample_valid[:, None, None, :], values, np.nan)
    return patches


def _iter_voxel_batches(live_i: np.ndarray, live_j: np.ndarray, ns: int, batch_size: int):
    """Yield bounded flat voxel batches without expanding entire traces at once."""
    total = int(len(live_i)) * int(ns)
    for start in range(0, total, max(1, int(batch_size))):
        flat = np.arange(start, min(start + int(batch_size), total), dtype=np.int64)
        cell = flat // int(ns)
        yield live_i[cell], live_j[cell], flat % int(ns), flat


def run_volume_inference(ctx, selection: dict, cfg: dict) -> None:
    """Stream a complete registered seismic survey through one released model."""
    from .metrics import apply_bounds

    model_name = str(selection.get("model") or "").casefold()
    deep = model_name in {"tcn", "cnn3d_multitask"}
    if deep:
        import torch
        from . import models_deep as mdeep
        if str(selection.get("input") or "") != "patch_stacks":
            raise StageBlocked("Released deep volume inference requires input='patch_stacks'")
        registry = pd.read_csv(ctx.release / "training" / "model_registry.csv")
        selected = registry[(registry.status == "OK") & (registry.input.astype(str) == "patch_stacks") &
                            (registry.model.astype(str) == model_name)]
        if selected.empty:
            raise StageBlocked(f"Release contains no successful deep model '{model_name}'")
        requested_task = str(selection.get("task") or "all").casefold()
        if requested_task not in {"all", "lithology", "vsh", "phit"}:
            raise StageBlocked("Deep task must be all, lithology, vsh or phit")
        if requested_task != "all" and requested_task not in set(selected.task.astype(str)):
            raise StageBlocked(f"Release has no successful {model_name} model for task '{requested_task}'")
        rel = Path(str(selected.iloc[0].final_model_file))
        checkpoint_path = (ctx.release / rel).resolve()
        if ctx.release.resolve() not in checkpoint_path.parents or not checkpoint_path.is_file():
            raise StageBlocked("Released deep checkpoint is missing")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model") != model_name:
            raise StageBlocked("Deep checkpoint architecture does not match the selected registry model")
        channels = list(checkpoint.get("channels") or [])
        if not channels or any(c not in STACKS for c in channels):
            raise StageBlocked(f"Unsupported deep checkpoint channel contract: {channels}")
        patch_contract_path = ctx.release / "contracts" / "patches" / "patch_manifest_v1.json"
        if not patch_contract_path.is_file():
            raise StageBlocked("Release is missing its immutable patch contract")
        patch_contract = json.loads(patch_contract_path.read_text(encoding="utf-8"))
        if float(patch_contract.get("dt_ms", -1)) != 4.0 or patch_contract.get("shape", [])[2:] != [5, 5, 21]:
            raise StageBlocked("Released deep patch contract is not the supported 4 ms, 5 x 5 x 21 contract")
        if not set(channels).issubset(set(patch_contract.get("channels") or [])):
            raise StageBlocked("Deep checkpoint channels are not present in the released extraction contract")
        device = mdeep.device_name(str(cfg.get("deep_device", "auto")))
        model = mdeep.build(model_name, len(channels), dropout=0.2).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        scaling = checkpoint.get("scaling")
        if not isinstance(scaling, dict):
            raise StageBlocked("Deep checkpoint has no frozen scaling parameters")
        tasks = ["lithology", "vsh", "phit"] if requested_task == "all" else [requested_task]
        zero_run_min = 5
        training_config = ctx.release / "config" / "training.yaml"
        if training_config.is_file():
            try:
                import yaml
                release_cfg = yaml.safe_load(training_config.read_text(encoding="utf-8")) or {}
                zero_run_min = int((release_cfg.get("missing_data") or {}).get("stack_zero_run_missing_min", 5))
            except Exception as exc:
                raise StageBlocked(f"Cannot read the release's missing-data feature contract: {exc}") from exc
        features = []
        tabular = None
    else:
        import joblib
        from .preprocess import apply_transform
        from .models_tabular import predict as predict_tabular
        if str(selection.get("input") or "") != "stacks":
            raise StageBlocked("Survey-volume tabular inference currently supports input='stacks' only")
        registry = pd.read_csv(ctx.release / "training" / "model_registry.csv")
        chosen = registry[(registry.status == "OK") & (registry.task.astype(str) == str(selection.get("task"))) &
                          (registry.input.astype(str) == "stacks") & (registry.model.astype(str) == model_name)]
        if len(chosen) != 1:
            raise StageBlocked(f"Tabular selector matched {len(chosen)} successful model rows; expected exactly one")
        rel = Path(str(chosen.iloc[0].final_model_file))
        artifact_path = (ctx.release / rel).resolve()
        if ctx.release.resolve() not in artifact_path.parents or not artifact_path.is_file():
            raise StageBlocked("Released tabular model artifact is missing")
        tabular = joblib.load(artifact_path)
        features = list(tabular.get("features") or [])
        if not features:
            raise StageBlocked("Released tabular artifact has no predictor feature list")
        transform_rel = Path(str(tabular.get("transform_file") or ""))
        transform_path = (ctx.release / transform_rel).resolve()
        if ctx.release.resolve() not in transform_path.parents or not transform_path.is_file():
            transform_path = (ctx.release / "preprocessing" / "transforms" / transform_rel.name).resolve()
        if ctx.release.resolve() not in transform_path.parents or not transform_path.is_file():
            raise StageBlocked("Released frozen transform file is missing")
        transforms = json.loads(transform_path.read_text(encoding="utf-8"))
        try:
            params = transforms["folds"][tabular["transform_key"]]["params"]
        except (KeyError, TypeError) as exc:
            raise StageBlocked("Released frozen transform key is missing") from exc
        tasks = [str(selection["task"])]
        channels = _required_channels(features)
        if not channels:
            raise StageBlocked("Could not derive required seismic stacks from released predictor columns")
        zero_run_min = 5
        training_config = ctx.release / "config" / "training.yaml"
        if training_config.is_file():
            try:
                import yaml
                release_cfg = yaml.safe_load(training_config.read_text(encoding="utf-8")) or {}
                zero_run_min = int((release_cfg.get("missing_data") or {}).get("stack_zero_run_missing_min", 5))
            except Exception as exc:
                raise StageBlocked(f"Cannot read the release's missing-data feature contract: {exc}") from exc
        device = "cpu"

    # The specialist extraction contract is fixed at 5 x 5 x 21 samples.
    offsets = patch_window(5, 5, 21)
    # Full-stack geometry is always the authoritative grid, even when the selected
    # model's predictors do not include full-stack amplitudes.
    input_channels = list(dict.fromkeys(["full", *channels]))
    foundation_id = getattr(ctx, "foundation_job_id", None)
    # The release's foundation job is the source of truth. Do not require users
    # to re-select the same five datasets on every inference job.
    foundation_job = _resolve_foundation_job(ctx, {}, str(foundation_id or ""))
    records = _resolve_originating_stack_records(foundation_job, str(foundation_id or ""))
    missing_channels = sorted(set(input_channels) - set(records))
    if missing_channels:
        raise StageBlocked(f"Originating foundation job is missing required stack datasets: {missing_channels}")
    # The inventory resolver has already verified every selected file against
    # its data_manifest.json SHA-256. No second, role-keyed provenance comparison.
    foundation_geometry_dir = foundation_job / "stages" / "00_inventory" / "tables" / "geometry"
    from .segy import SurveyGeometry
    open_files, geometries = {}, {}
    try:
        for role in input_channels:
            geometry_path = foundation_geometry_dir / f"{role}_geometry.npz"
            if not geometry_path.is_file() or geometry_path.is_symlink():
                raise StageBlocked(f"Foundation Stage 00 geometry is missing for required stack '{role}'")
            geometries[role] = SurveyGeometry.load_npz(geometry_path)
            path = records[role]["path"]
            sgy = SegyFile(path)
            open_files[role] = sgy
            _same_sampling(geometries["full"], geometries[role], role)
            if sgy.ns != geometries[role].ns:
                raise StageBlocked(f"{role} SEG-Y sample count differs from its foundation geometry")
            ctx.add_input(geometry_path, f"foundation prerequisite: Stage 00 {role} geometry")
        reference = geometries["full"]
        parity = _verify_stage03_patch_parity(ctx, foundation_job, reference, geometries,
                                              open_files, channels, offsets)
        if reference.flags:
            ctx.limit("Reference SEG-Y geometry has flags: " + "; ".join(reference.flags), category="xy_geometry")
        if reference.affine_residual_max > 0.25 * min(reference.bin_spacing()):
            ctx.limit("Reference SEG-Y XY geometry is materially non-affine; prediction cube remains indexed by "
                      "native IL/XL, and XY interpretation requires the original trace coordinates/QC",
                      category="xy_geometry")
        if reference.dt_ms != 4.0:
            raise StageBlocked(f"Released patch contract requires 4 ms samples; survey dt is {reference.dt_ms} ms")
        if deep and len(channels) != 5:
            raise StageBlocked("Released deep model expects exactly five stack channels")

        # Extraction and GPU model batches are independent controls. Small
        # extraction batches bound host memory; deep_batch_size controls CUDA.
        extract_batch_size = max(128, int(cfg.get(
            "volume_extract_batch_size", cfg.get("volume_batch_size", 2048))))
        deep_batch_size = max(1, int(cfg.get("deep_batch_size", 512)))
        deep_precision = str(cfg.get("deep_precision", "fp32")).casefold()
        if deep_precision not in {"fp32", "fp16", "bf16"}:
            raise StageBlocked("deep_precision must be fp32, fp16 or bf16")
        pin_memory = bool(cfg.get("deep_pin_memory", True))
        tile_i = max(1, int(cfg.get("volume_tile_inline", 32)))
        tile_j = max(1, int(cfg.get("volume_tile_crossline", 32)))
        region_i0, region_i1, region_j0, region_j1, region_grid, region_meta = _resolve_volume_region(reference, cfg)
        shape = (region_grid.shape[0], region_grid.shape[1], reference.ns)
        fingerprint_cfg = dict(cfg)
        fingerprint_cfg["_resolved_volume_region"] = region_meta
        output_dir = ctx.stage_dir / "volumes"
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = ctx.stage_dir / "checkpoints"
        checkpoint_path = checkpoint_dir / "volume_inference_state.json"
        fingerprint = _volume_fingerprint(ctx, selection, records, reference, channels, tasks, fingerprint_cfg)
        expected_outputs = [output_dir / f"prediction_{task}.npy" for task in tasks]
        expected_outputs += [output_dir / "prediction_valid.npy"]
        prior = {}
        if checkpoint_path.is_file():
            try:
                prior = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                prior = {}
        # v101 checkpoints include exact per-tile valid counts. Older checkpoints
        # cannot safely reconstruct the summary without rescanning the full mask.
        resumable = (prior.get("fingerprint") == fingerprint and
                     isinstance(prior.get("tile_valid_counts"), dict) and
                     all(path.is_file() and not path.is_symlink() for path in expected_outputs))
        if resumable:
            try:
                resumable = all(tuple(np.load(path, mmap_mode="r", allow_pickle=False).shape) == shape
                                for path in expected_outputs)
            except (OSError, ValueError):
                resumable = False
        output_maps = {}
        for task in tasks:
            filename = output_dir / f"prediction_{task}.npy"
            output_maps[task] = np.lib.format.open_memmap(
                filename, mode="r+" if resumable else "w+", dtype=np.float32, shape=shape)
            if not resumable:
                output_maps[task][:] = np.nan
        valid_path = output_dir / "prediction_valid.npy"
        valid_map = np.lib.format.open_memmap(
            valid_path, mode="r+" if resumable else "w+", dtype=np.uint8, shape=shape)
        if not resumable:
            valid_map[:] = 0
        completed_tiles = set(prior.get("completed_tiles", [])) if resumable else set()
        tile_valid_counts = ({str(k): int(v) for k, v in prior.get("tile_valid_counts", {}).items()}
                             if resumable else {})
        tile_count = 0
        n_pred = 0
        inference_started = time.perf_counter()
        phase_seconds = {"patch_extraction": 0.0, "preprocessing": 0.0,
                         "model_inference": 0.0, "output_write": 0.0}

        def sync_cuda():
            if deep and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
        ctx.info("Stage 09 volume inference: starting ROI %s x %s x %s; %d live traces; tile %d x %d; extraction batch %d voxels; deep batch %d; %d prior tiles resumable",
                 *shape, region_meta["live_traces"], tile_i, tile_j, extract_batch_size,
                 deep_batch_size if deep else 0, len(completed_tiles))
        from types import SimpleNamespace
        region_geometry = SimpleNamespace(grid=region_grid)
        for tile_id, bounds, (local_i, local_j) in _iter_spatial_tiles(region_geometry, tile_i, tile_j):
            live_i, live_j = local_i + region_i0, local_j + region_j0
            tile_count += 1
            if str(tile_id) in completed_tiles:
                # The exact count is checkpointed when the tile is committed.
                if str(tile_id) not in tile_valid_counts:
                    raise StageBlocked(f"Checkpoint is missing valid-voxel count for tile {tile_id}")
                continue
            tile_predicted = 0
            tile_global_i0 = region_i0 + bounds[0]
            tile_global_i1 = region_i0 + bounds[1]
            tile_global_j0 = region_j0 + bounds[2]
            tile_global_j1 = region_j0 + bounds[3]
            phase_started = time.perf_counter()
            tile_blocks, cache_i0, cache_j0 = _read_tile_trace_blocks(
                open_files, geometries, reference, channels,
                tile_global_i0, tile_global_i1, tile_global_j0, tile_global_j1, halo=2)
            phase_seconds["patch_extraction"] += time.perf_counter() - phase_started

            batch_specs = _iter_voxel_batches(live_i, live_j, reference.ns, extract_batch_size)

            def prepare_batch(spec):
                ci, cj, ck, flat = spec
                t0 = time.perf_counter()
                patches = _extract_patches_from_tile(
                    tile_blocks, channels, cache_i0, cache_j0, ci, cj, ck, offsets)
                extract_seconds = time.perf_counter() - t0
                t0 = time.perf_counter()
                keep = _patch_keep_mask(
                    patches, deep=deep,
                    max_nan_fraction=float(cfg.get("deep_max_nan_fraction", 0.2)),
                )
                if deep:
                    prepared = mdeep.apply_patch_scaling(patches[keep], scaling) if keep.any() else None
                else:
                    prepared = patches[keep] if keep.any() else None
                prep_seconds = time.perf_counter() - t0
                return ci, cj, ck, flat, keep, prepared, extract_seconds, prep_seconds

            if deep:
                # One bounded producer prefetches the next CPU patch batch while
                # the main thread submits the current batch to CUDA. Only one
                # future batch is retained, keeping memory bounded.
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=1, thread_name_prefix="geogg-prefetch") as pool:
                    pending = None
                    try:
                        first_spec = next(batch_specs, None)
                        if first_spec is not None:
                            pending = pool.submit(prepare_batch, first_spec)
                        while pending is not None:
                            prepared_batch = pending.result()
                            next_spec = next(batch_specs, None)
                            pending = pool.submit(prepare_batch, next_spec) if next_spec is not None else None
                            ci, cj, ck, flat, keep, scaled, extract_s, prep_s = prepared_batch
                            phase_seconds["patch_extraction"] += extract_s
                            phase_seconds["preprocessing"] += prep_s
                            if scaled is None:
                                continue
                            sync_cuda()
                            phase_started = time.perf_counter()
                            prediction = mdeep.predict(
                                model, scaled, device=device, batch_size=deep_batch_size,
                                pin_memory=pin_memory, precision=deep_precision)
                            sync_cuda()
                            phase_seconds["model_inference"] += time.perf_counter() - phase_started
                            values = {}
                            for task in tasks:
                                raw = np.asarray(prediction[task], dtype=float)
                                if task == "lithology":
                                    values[task] = raw
                                else:
                                    values[task], _ = apply_bounds(task, raw)
                            phase_started = time.perf_counter()
                            flat_indices = ((ci[keep] - region_i0) * shape[1] +
                                            (cj[keep] - region_j0)) * reference.ns + ck[keep]
                            for task, vector in values.items():
                                output_maps[task].reshape(-1)[flat_indices] = np.asarray(vector, dtype=np.float32)
                            valid_map.reshape(-1)[flat_indices] = 1
                            tile_predicted += int(keep.sum())
                            n_pred += int(keep.sum())
                            phase_seconds["output_write"] += time.perf_counter() - phase_started
                    finally:
                        if pending is not None:
                            pending.cancel()
            else:
                for spec in batch_specs:
                    ci, cj, ck, flat, keep, batch_patches, extract_s, prep_s = prepare_batch(spec)
                    phase_seconds["patch_extraction"] += extract_s
                    phase_seconds["preprocessing"] += prep_s
                    if batch_patches is None:
                        continue
                    batch_features = _features_from_patches(
                        batch_patches, channels, features, reference.dt_ms, zero_run_min)
                    X = apply_transform(batch_features, params, features,
                                        mode=tabular.get("scaling_mode", "independent"),
                                        add_indicators=False).to_numpy(float)
                    phase_started = time.perf_counter()
                    raw = predict_tabular(tabular["estimator"], str(tabular["kind"]), X)
                    phase_seconds["model_inference"] += time.perf_counter() - phase_started
                    values = {tasks[0]: apply_bounds(tasks[0], raw)[0] if tabular["kind"] == "regression"
                              else np.asarray(raw, dtype=float)}
                    phase_started = time.perf_counter()
                    flat_indices = ((ci[keep] - region_i0) * shape[1] +
                                    (cj[keep] - region_j0)) * reference.ns + ck[keep]
                    for task, vector in values.items():
                        output_maps[task].reshape(-1)[flat_indices] = np.asarray(vector, dtype=np.float32)
                    valid_map.reshape(-1)[flat_indices] = 1
                    tile_predicted += int(keep.sum())
                    n_pred += int(keep.sum())
                    phase_seconds["output_write"] += time.perf_counter() - phase_started
            for arr in output_maps.values():
                arr.flush()
            valid_map.flush()
            completed_tiles.add(str(tile_id))
            tile_valid_counts[str(tile_id)] = tile_predicted
            _checkpoint_write(checkpoint_path, {
                "fingerprint": fingerprint, "completed_tiles": sorted(completed_tiles, key=int),
                "tile_valid_counts": tile_valid_counts,
                "tile_count": tile_count, "tile_shape": [tile_i, tile_j],
                "output_shape": list(shape), "updated_tiles": tile_count,
            })
            log_every = max(1, int(cfg.get("volume_progress_log_tiles", 10)))
            if len(completed_tiles) == 1 or len(completed_tiles) % log_every == 0:
                ctx.info("Stage 09 volume inference: completed %d tiles; latest tile predicted %d voxels",
                         len(completed_tiles), tile_predicted)
        ctx.info("Stage 09 volume inference: flushing prediction arrays and validity mask")
        for arr in output_maps.values():
            arr.flush()
        valid_map.flush()
        # Summarise from the per-tile counters. Do not scan the multi-GB uint8
        # memmap: this count is exact because each output tile owns disjoint centres.
        expected_live = int(region_meta["live_traces"] * reference.ns)
        n_pred = _summarize_tile_valid_counts(tile_valid_counts, completed_tiles, tile_count, expected_live)
        ctx.info("Stage 09 volume inference: tile processing complete; %d/%d valid voxels; writing geometry and summary",
                 n_pred, expected_live)
        for role, item in records.items():
            if role in input_channels:
                ctx.add_input(item["path"], f"registered seismic input: {role}; dataset_id={item['dataset_id']}")
        trace_grid_path = output_dir / "trace_index_grid.npy"
        np.save(trace_grid_path, region_grid, allow_pickle=False)
        reference_channel = "full"
        geom_meta = {
            "reference_channel": reference_channel,
            "foundation_job_id": foundation_job.name,
            "stage03_parity": parity,
            "axis_order": ["inline_bin", "crossline_bin", "time_sample"],
            "shape": list(shape), "il_min": reference.il_min + region_i0 * reference.il_step,
            "il_step": reference.il_step,
            "xl_min": reference.xl_min + region_j0 * reference.xl_step, "xl_step": reference.xl_step,
            "region": region_meta,
            "t0_ms": reference.t0_ms, "dt_ms": reference.dt_ms, "ns": reference.ns,
            "t_max_ms": reference.t0_ms + (reference.ns - 1) * reference.dt_ms,
            "affine": reference.affine, "affine_residual_max_m": reference.affine_residual_max,
            "affine_residual_rms_m": reference.affine_residual_rms,
            "geometry_flags": reference.flags,
            "spatial_reference": _xy_mapping_quality(reference),
            "trace_index_grid": "volumes/trace_index_grid.npy",
            "channels": channels,
            "input_channels": input_channels,
            "tiling": {"tile_shape_inline_crossline": [tile_i, tile_j],
                       "tile_count": tile_count, "completed_tiles": len(completed_tiles),
                       "valid_voxel_count_source": "checkpointed_per_tile_counts",
                       "halo_samples": [2, 2, 10], "checkpoint": "checkpoints/volume_inference_state.json"},
            "input_datasets": {k: {"dataset_id": v["dataset_id"], "name": v["name"],
                                    "file_name": v["path"].name} for k, v in records.items() if k in input_channels},
        }
        ctx.save_json(geom_meta, "volumes/volume_geometry.json", "Prediction cube geometry and registered input provenance")
        summary = {"release_id": ctx.release_id, "model": selection, "input_mode": "registered_seismic_volume",
                   "tasks": tasks, "predicted_voxels": n_pred,
                   "foundation_job_id": foundation_job.name, "stage03_parity": parity,
                   "candidate_voxels": expected_live, "volume_shape": list(shape),
                   "region": region_meta,
                   "channels": channels, "device": device,
                   "performance_configuration": {
                       "extraction_batch_size_voxels": extract_batch_size,
                       "deep_batch_size": deep_batch_size if deep else None,
                       "deep_precision": deep_precision if deep else None,
                       "pinned_memory": pin_memory if deep else False,
                       "cpu_prefetch_workers": 1 if deep else 0,
                   },
                   "tiling": {"tile_shape_inline_crossline": [tile_i, tile_j],
                              "tile_count": tile_count, "completed_tiles": len(completed_tiles),
                              "resume_used": bool(resumable),
                              "valid_voxel_count_source": "checkpointed_per_tile_counts",
                              "checkpoint": "checkpoints/volume_inference_state.json"},
                   "outputs": {task: f"volumes/prediction_{task}.npy" for task in tasks},
                   "valid_mask": "volumes/prediction_valid.npy", "geometry": "volumes/volume_geometry.json",
                   "label_independent": True,
                   "note": "Predictions are stored as [inline_bin,crossline_bin,time_sample] cubes for the configured ROI. "
                           "Empty traces and rejected low-coverage patches remain NaN; no labels or well coordinates are used."}
        ctx.save_json(summary, "volume_inference_summary.json", "Stage 09 survey-volume inference provenance")
        ctx.metric("volume_prediction_voxels", n_pred)
        total_elapsed = time.perf_counter() - inference_started
        phase_summary = {key: round(value, 3) for key, value in phase_seconds.items()}
        phase_summary["total_elapsed_seconds"] = round(total_elapsed, 3)
        phase_summary["unaccounted_seconds"] = round(max(0.0, total_elapsed - sum(phase_seconds.values())), 3)
        phase_summary["predicted_voxels_per_second"] = round(n_pred / total_elapsed, 3) if total_elapsed > 0 else 0.0
        summary["performance"] = phase_summary
        # Rewrite summary with measured phase timings.
        ctx.save_json(summary, "volume_inference_summary.json", "Stage 09 survey-volume inference provenance and performance")
        ctx.info("Stage 09 volume inference: finalisation complete; predicted %d voxels; phase seconds=%s",
                 n_pred, phase_summary)
        ctx.metric("volume_shape", list(shape))
        ctx.metric("volume_inference_performance", phase_summary)
        ctx.metric("volume_inference_device", device)
        ctx.check("full_stack_reference_geometry", True,
                  "All prediction coordinates and patch centres are indexed on the foundation full-stack grid")
        ctx.check("stage03_patch_parity", True,
                  f"{parity['rows_compared']} Stage 03 patches reproduced; max absolute difference "
                  f"{parity['max_abs_difference']:.6g}")
    finally:
        for sgy in open_files.values():
            sgy.close()
