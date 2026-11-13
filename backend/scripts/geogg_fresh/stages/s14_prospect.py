"""Stage 14: well-anchored prospect definition, GRV and well-log screening.

This stage consumes immutable Foundation, Stage 11 and Stage 12 products. It does
not run inference, modify geobodies, or calculate HCPV / in-place resources.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .s13_well_geobody_analysis import (
    _depth_from_model, _intervals, _load_stage01_depth_models,
    _make_depth_model, _read_json, _safe_job, _find_product,
)

STAGE = "14_prospect"
PREFLIGHT = "geogg_prospect_preflight.json"
SUMMARY = "geogg_prospect_summary.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _product_path(job: Path, row: dict) -> Path:
    rel = str(row.get("path") or row.get("storage_root_relative_path") or "").strip()
    if not rel:
        raise RuntimeError(f"Registered product {row.get('product')} has no path")
    path = (job / rel).resolve()
    try:
        path.relative_to(job.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Product path escapes source job: {rel}") from exc
    return path


def _atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def _cell_area_grid(run_length: pd.DataFrame, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    """Estimate live-bin area from Stage 12 CDP XY using local IL/XL vectors."""
    required = {"volume_i", "volume_j", "x", "y"}
    if not required.issubset(run_length.columns):
        raise RuntimeError(f"Stage 12 run-length index lacks XY columns: {sorted(required - set(run_length.columns))}")
    xy = run_length[["volume_i", "volume_j", "x", "y"]].drop_duplicates(["volume_i", "volume_j"])
    nx, ny = shape
    xg = np.full((nx, ny), np.nan, dtype=np.float64)
    yg = np.full((nx, ny), np.nan, dtype=np.float64)
    ii = xy.volume_i.to_numpy(dtype=int); jj = xy.volume_j.to_numpy(dtype=int)
    xx = pd.to_numeric(xy.x, errors="coerce").to_numpy(dtype=float)
    yy = pd.to_numeric(xy.y, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy) & (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
    xg[ii[ok], jj[ok]] = xx[ok]; yg[ii[ok], jj[ok]] = yy[ok]
    # Use local forward differences (backward at the final row/column) to estimate
    # the XY Jacobian determinant without a Python loop over the full survey grid.
    if nx < 2 or ny < 2:
        raise RuntimeError("At least two inline and two crossline bins are required to derive cell area")
    ip = np.where(np.arange(nx) < nx - 1, np.arange(nx) + 1, np.arange(nx) - 1)
    jp = np.where(np.arange(ny) < ny - 1, np.arange(ny) + 1, np.arange(ny) - 1)
    dix = xg[ip, :] - xg; diy = yg[ip, :] - yg
    djx = xg[:, jp] - xg; djy = yg[:, jp] - yg
    area = np.abs(dix * djy - diy * djx)
    finite = area[np.isfinite(area) & (area > 0)]
    if not len(finite):
        raise RuntimeError("Cannot derive positive live-bin footprint area from Stage 12 CDP XY coordinates")
    median = float(np.median(finite))
    # Preserve measured local areas; use robust median only where the grid lacks local XY neighbours.
    area[~np.isfinite(area) | (area <= 0)] = median
    return area, {"status": "AVAILABLE", "method": "local_CDP_XY_vector_cross_product",
                  "median_cell_area_m2": median, "measured_cells": int(len(finite)),
                  "filled_cells": int(area.size - len(finite)), "units": "m2"}, xg, yg


def _path_samples(trajectory: pd.DataFrame, labels: np.ndarray, geometry: dict,
                  well_ids: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    il0, ils = int(geometry["il_min"]), int(geometry["il_step"])
    xl0, xls = int(geometry["xl_min"]), int(geometry["xl_step"])
    t0, dt = float(geometry["t0_ms"]), float(geometry["dt_ms"])
    selected = trajectory[trajectory.well.astype(str).str.strip().str.casefold().isin(well_ids)].copy()
    selected["_well"] = selected.well.astype(str).str.strip().str.casefold()
    rows = []
    for well, frame in selected.groupby("_well", sort=False):
        frame = frame.sort_values("md_m", kind="stable")
        for idx, r in enumerate(frame.itertuples()):
            i = int(round((float(r.il_nearest) - il0) / ils))
            j = int(round((float(r.xl_nearest) - xl0) / xls))
            k = int(round((float(r.twt_ms) - t0) / dt))
            inside = 0 <= i < labels.shape[0] and 0 <= j < labels.shape[1] and 0 <= k < labels.shape[2]
            gid = int(labels[i, j, k]) if inside else 0
            rows.append({"well": well, "sample_index": idx, "md_m": float(r.md_m),
                         "tvdss_m": float(r.tvdss_m), "twt_ms": float(r.twt_ms),
                         "il": float(r.il_nearest), "xl": float(r.xl_nearest),
                         "volume_i": i, "volume_j": j, "time_k": k,
                         "geobody_id": gid, "inside_volume": bool(inside)})
    samples = pd.DataFrame(rows)
    intervals = []
    for well in well_ids:
        _, iv = _intervals(samples[samples.well == well].copy(), well)
        if not iv.empty:
            intervals.append(iv)
    return samples, pd.concat(intervals, ignore_index=True) if intervals else pd.DataFrame()


def _well_intersection_audit(samples: pd.DataFrame, intervals: pd.DataFrame,
                             well_ids: list[str], anchor_wells: list[str],
                             min_samples: int) -> pd.DataFrame:
    """Summarise trajectory support for every configured well and anchor."""
    rows = []
    for well in well_ids:
        path = samples[samples.well == well] if not samples.empty else pd.DataFrame()
        hit = path[path.geobody_id > 0] if not path.empty else pd.DataFrame()
        iv = intervals[intervals.well == well] if not intervals.empty else pd.DataFrame()
        rows.append({
            "well": well, "is_anchor": well in anchor_wells,
            "trajectory_samples_in_volume": int(path.inside_volume.sum()) if not path.empty else 0,
            "trajectory_samples_on_geobody": int(len(hit)),
            "intersected_geobody_ids": ";".join(map(str, sorted(hit.geobody_id.unique().tolist()))) if not hit.empty else "",
            "raw_intervals": int(len(iv)),
            "intervals_meeting_min_samples": int((iv.n_path_samples >= min_samples).sum()) if not iv.empty else 0,
            "intersection_status": "INTERSECTS_RETAINED_GEOBODY" if not hit.empty else "NO_RETAINED_GEOBODY_INTERSECTION",
            "anchor_status": ("ANCHOR_SUPPORTED" if not iv.empty and bool((iv.n_path_samples >= min_samples).any())
                             else "ANCHOR_NO_QUALIFYING_INTERVAL") if well in anchor_wells else "NOT_ANCHOR",
        })
    return pd.DataFrame(rows)


def _prospect_voxels(labels: np.ndarray, gid: int, k0: int, k1: int,
                     seeds: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    from scipy import ndimage
    sub = np.asarray(labels[:, :, k0:k1 + 1]) == int(gid)
    components, _ = ndimage.label(sub, structure=ndimage.generate_binary_structure(3, 1))
    seeds = seeds[(seeds.time_k >= k0) & (seeds.time_k <= k1)]
    if seeds.empty:
        return np.array([], int), np.array([], int), np.array([], int), False
    si = seeds.volume_i.to_numpy(dtype=int); sj = seeds.volume_j.to_numpy(dtype=int)
    sk = seeds.time_k.to_numpy(dtype=int) - k0
    valid = (si >= 0) & (si < components.shape[0]) & (sj >= 0) & (sj < components.shape[1]) & (sk >= 0) & (sk < components.shape[2])
    component_ids = np.unique(components[si[valid], sj[valid], sk[valid]])
    component_ids = component_ids[component_ids > 0]
    if not len(component_ids):
        return np.array([], int), np.array([], int), np.array([], int), False
    mask = np.isin(components, component_ids)
    vi, vj, vk = np.nonzero(mask)
    touches = False
    if k0 > 0:
        touches |= bool(np.any(np.asarray(labels[:, :, k0 - 1])[vi[vk == 0], vj[vk == 0]] == gid))
    if k1 + 1 < labels.shape[2]:
        edge = k1 - k0
        touches |= bool(np.any(np.asarray(labels[:, :, k1 + 1])[vi[vk == edge], vj[vk == edge]] == gid))
    return vi, vj, vk + k0, touches


def _logs_for_prospect(logs: pd.DataFrame, samples: pd.DataFrame, voxels: tuple,
                       dt: float, n_j: int, n_k: int) -> pd.DataFrame:
    vi, vj, vk = voxels
    if not len(vi):
        return pd.DataFrame()
    keys = ((vi.astype(np.int64) * int(n_j) + vj.astype(np.int64)) * int(n_k) + vk.astype(np.int64))
    p = samples.copy()
    sample_keys = ((p.volume_i.to_numpy(dtype=np.int64) * int(n_j) +
                    p.volume_j.to_numpy(dtype=np.int64)) * int(n_k) +
                   p.time_k.to_numpy(dtype=np.int64))
    inside = p[np.isin(sample_keys, keys, assume_unique=False) & (p.geobody_id > 0)]
    out = []
    for well, path in inside.groupby("well"):
        L = logs[logs.well.astype(str).str.strip().str.casefold() == well].sort_values("md_m").copy()
        if L.empty:
            continue
        cells = np.sort(path.twt_ms.to_numpy(dtype=float))
        t = pd.to_numeric(L.twt_ms, errors="coerce").to_numpy(dtype=float)
        idx = np.searchsorted(cells, t)
        best = np.full(len(t), -1, dtype=int)
        for cand in (idx - 1, idx):
            ok = (cand >= 0) & (cand < len(cells))
            clipped = np.clip(cand, 0, max(len(cells) - 1, 0))
            hit = ok & (np.abs(t - cells[clipped]) <= dt / 2)
            best[(best < 0) & hit] = cand[(best < 0) & hit]
        valid = best >= 0
        if not valid.any():
            continue
        L = L.iloc[np.flatnonzero(valid)].copy()
        L["cell"] = cells[best[valid]]
        tvd = pd.to_numeric(L.tvdss_m, errors="coerce").to_numpy(dtype=float)
        L["h_m"] = np.abs(np.gradient(tvd)) if len(tvd) > 1 else np.zeros(len(tvd))
        L["well"] = well
        out.append(L)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def _log_stats(frame: pd.DataFrame, vsh_max: float, phit_min: float) -> dict:
    if frame.empty:
        return {"gross_m": 0.0, "net_m": 0.0, "ntg": None, "phit_net": None,
                "swt_net_pv": None, "swt_support_m": 0.0, "n_log_samples": 0}
    h = pd.to_numeric(frame.h_m, errors="coerce").to_numpy(dtype=float)
    vsh = pd.to_numeric(frame.vsh_fraction, errors="coerce").to_numpy(dtype=float)
    phit = pd.to_numeric(frame.phit_fraction, errors="coerce").to_numpy(dtype=float)
    swt = pd.to_numeric(frame.swt_fraction, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(h) & (h > 0) & np.isfinite(vsh) & np.isfinite(phit)
    net = valid & (vsh <= vsh_max) & (phit >= phit_min)
    gross = float(h[valid].sum()); net_h = float(h[net].sum())
    sw_ok = net & np.isfinite(swt) & (swt >= 0) & (swt <= 1)
    pore = phit[sw_ok] * h[sw_ok]
    return {"gross_m": gross, "net_m": net_h, "ntg": net_h / gross if gross > 0 else None,
            "phit_net": float(np.sum(phit[net] * h[net]) / net_h) if net_h > 0 else None,
            "swt_net_pv": float(np.sum(pore * swt[sw_ok]) / pore.sum()) if pore.sum() > 0 else None,
            "swt_support_m": float(h[sw_ok].sum()), "n_log_samples": int(len(frame)),
            "net_swt_samples": int(sw_ok.sum())}


def _bootstrap_screening(frame: pd.DataFrame, n: int, rng: np.random.Generator,
                         vsh_max: float, phit_min: float, grv_values: tuple[float, float, float] | None,
                         velocity_fraction: float, hcpv_allowed: bool) -> dict:
    """Block bootstrap interpreted log cells; HCPV remains a conditional scenario."""
    if frame.empty or n < 1:
        return {"status": "NO_LOG_BLOCKS", "replicates": 0}
    groups = [g for _, g in frame.groupby(["well", "cell"], sort=False)]
    if not groups:
        return {"status": "NO_LOG_BLOCKS", "replicates": 0}
    draws = {k: [] for k in ("ntg", "phit_net", "swt_net_pv", "hcpv_m3")}
    for _ in range(int(n)):
        selected = rng.integers(0, len(groups), len(groups))
        sample = pd.concat([groups[i] for i in selected], ignore_index=True)
        stat = _log_stats(sample, vsh_max, phit_min)
        for key in ("ntg", "phit_net", "swt_net_pv"):
            value = stat.get(key)
            draws[key].append(float(value) if value is not None and np.isfinite(value) else np.nan)
        hcpv = np.nan
        if hcpv_allowed and grv_values and all(np.isfinite(grv_values)):
            ntg, phi, sw = (stat.get(k) for k in ("ntg", "phit_net", "swt_net_pv"))
            if all(v is not None and np.isfinite(v) and 0 <= v <= 1 for v in (ntg, phi, sw)):
                lo, mode, hi = grv_values
                grv = rng.triangular(lo, mode, hi) if hi > lo else mode
                scale = rng.uniform(max(0.0, 1.0 - velocity_fraction), 1.0 + velocity_fraction)
                hcpv = grv * scale * ntg * phi * (1.0 - sw)
        draws["hcpv_m3"].append(hcpv)
    result = {"status": "BLOCK_BOOTSTRAP", "replicates": int(n),
              "block_unit": "well x 4ms seismic cell",
              "uncertainty_basis": "paired log-cell block resampling + regional T-Z model range + velocity scaling",
              "excludes": "threshold/model-score uncertainty, spatial extrapolation, closure/contact, fill, fluid and geological risk"}
    for key, values in draws.items():
        a = np.asarray(values, dtype=float); a = a[np.isfinite(a)]
        result[f"{key}_P90"] = float(np.percentile(a, 10)) if len(a) else None
        result[f"{key}_P50"] = float(np.percentile(a, 50)) if len(a) else None
        result[f"{key}_P10"] = float(np.percentile(a, 90)) if len(a) else None
        result[f"{key}_valid_replicates"] = int(len(a))
    return result


def run(config: dict, result_path: Path) -> dict:
    current = result_path.parent
    jobs_root = next((p for p in current.parents if p.name == "processing_jobs"), None)
    if jobs_root is None:
        raise RuntimeError(f"Cannot locate definitive processing_jobs directory from {result_path}")
    ids = {k: str(config.get(k) or "").strip() for k in
           ("foundation_job_id", "inference_job_id", "source_volume_job_id", "source_geobody_job_id")}
    missing = [k for k, v in ids.items() if not v]
    if missing:
        raise RuntimeError("Missing required source job IDs: " + ", ".join(missing))
    jobs = {k: _safe_job(jobs_root, v, k) for k, v in ids.items()}
    well_setting = config.get("well_id")
    if not isinstance(well_setting, (list, tuple)):
        raise RuntimeError("prospect.well_id must be an explicit YAML list; wildcard and string selectors are not supported")
    well_ids = [str(v).strip().casefold() for v in well_setting if str(v).strip()]
    if not well_ids or len(well_ids) != len(set(well_ids)):
        raise RuntimeError("prospect.well_id must be an explicit, non-empty list without duplicates")
    anchor_mode = str(config.get("anchor_selection_mode", "explicit")).strip().casefold()
    if anchor_mode not in {"explicit", "intersecting_configured_wells"}:
        raise RuntimeError("prospect.anchor_selection_mode must be 'explicit' or 'intersecting_configured_wells'")
    anchor_setting = config.get("anchor_well_id", [])
    if not isinstance(anchor_setting, (list, tuple)):
        raise RuntimeError("prospect.anchor_well_id must be an explicit YAML list")
    anchor_wells = [str(v).strip().casefold() for v in anchor_setting if str(v).strip()]
    if len(anchor_wells) != len(set(anchor_wells)) or not set(anchor_wells).issubset(set(well_ids)):
        raise RuntimeError("prospect.anchor_well_id must be a unique subset of explicit well_id")
    if anchor_mode == "explicit" and not anchor_wells:
        raise RuntimeError("prospect.anchor_well_id must be non-empty when anchor_selection_mode is 'explicit'")
    if anchor_mode == "intersecting_configured_wells" and anchor_wells:
        raise RuntimeError("anchor_well_id must be empty when anchor_selection_mode is 'intersecting_configured_wells'; supported anchors are selected from well_id")
    if int(config.get("min_interval_samples", 3)) < 1 or float(config.get("twt_pad_ms", 8.0)) < 0 or float(config.get("merge_gap_ms", 8.0)) < 0:
        raise RuntimeError("Prospect interval sample count must be positive and TWT pad / merge gap cannot be negative")
    cutoffs = config.get("net_cutoffs", {"vsh_max": 0.4, "phit_min": 0.03})
    if not all(0.0 <= float(cutoffs[k]) <= 1.0 for k in ("vsh_max", "phit_min")):
        raise RuntimeError("net_cutoffs vsh_max and phit_min must be fractions in [0, 1]")
    if int(config.get("bootstrap_replicates", 1000)) < 1 or not 0 <= float(config.get("velocity_uncertainty_fraction", 0.10)) < 1:
        raise RuntimeError("bootstrap_replicates must be positive and velocity_uncertainty_fraction in [0, 1)")

    foundation = jobs["foundation_job_id"]
    trajectory_stage_id = str(config.get("trajectory_stage_id", "01_well_ties"))
    well_log_stage_id = str(config.get("well_log_stage_id", "02_labels"))
    stage01 = foundation / "stages" / trajectory_stage_id / "tables"
    trajectory_path = stage01 / "tz_samples_all_wells.csv"
    logs_path = foundation / "stages" / well_log_stage_id / "tables" / "interpreted_logs_all_wells.csv"
    if not trajectory_path.is_file():
        raise RuntimeError(f"Foundation Stage 01 trajectory table missing: {trajectory_path}")
    if not logs_path.is_file():
        raise RuntimeError(f"Foundation Stage 02 interpreted well logs missing: {logs_path}")
    trajectory = pd.read_csv(trajectory_path)
    needed_t = {"well", "md_m", "tvdss_m", "twt_ms", "il_nearest", "xl_nearest"}
    if not needed_t.issubset(trajectory.columns):
        raise RuntimeError(f"Stage 01 trajectory table missing columns: {sorted(needed_t - set(trajectory.columns))}")
    trajectory = trajectory.dropna(subset=list(needed_t)).copy()
    available = set(trajectory.well.astype(str).str.strip().str.casefold())
    unknown = sorted(set(well_ids) - available)
    if unknown:
        raise RuntimeError(f"Configured well IDs absent from Stage 01 trajectory: {unknown}")
    logs = pd.read_csv(logs_path)
    needed_l = {"well", "md_m", "tvdss_m", "twt_ms", "vsh_fraction", "phit_fraction", "swt_fraction"}
    if not needed_l.issubset(logs.columns):
        raise RuntimeError(f"Stage 02 interpreted logs missing columns: {sorted(needed_l - set(logs.columns))}")
    logs = logs[logs.well.astype(str).str.strip().str.casefold().isin(well_ids)].copy()

    volume_stage_id = str(config.get("source_volume_stage_id", "11_volume_export"))
    volume_manifest = _read_json(jobs["source_volume_job_id"] / "stages" / volume_stage_id / "native_volume_export_manifest.json", "Stage 11 volume manifest")
    if str(volume_manifest.get("source_inference_job_id") or (volume_manifest.get("provenance") or {}).get("source_inference_job_id") or "") != ids["inference_job_id"]:
        raise RuntimeError("Stage 11 volume manifest does not reference configured inference job")
    geobody_stage = jobs["source_geobody_job_id"] / "stages" / str(config.get("source_geobody_stage_id", "12_geobody"))
    gm = _read_json(geobody_stage / "geobody_manifest.json", "Stage 12 geobody manifest")
    if str(gm.get("status", "")).casefold() != "completed":
        raise RuntimeError("Source Stage 12 geobody manifest is not completed")
    expected = {"foundation_job_id": ids["foundation_job_id"], "inference_job_id": ids["inference_job_id"], "volume_job_id": ids["source_volume_job_id"]}
    src = gm.get("source", {})
    for k, v in expected.items():
        if str(src.get(k) or "") != v:
            raise RuntimeError(f"Stage 12 provenance mismatch for {k}: {src.get(k)!r} != {v!r}")
    geom = gm.get("geometry", {})
    shape = tuple(int(v) for v in geom.get("shape", []))
    if len(shape) != 3 or geom.get("axis_order") != ["inline_bin", "crossline_bin", "time_sample"]:
        raise RuntimeError("Stage 12 geobody geometry must declare [inline,crossline,time] shape")
    volume_geometry = volume_manifest.get("geometry") if isinstance(volume_manifest.get("geometry"), dict) else {}
    for field in ("shape", "il_min", "il_step", "xl_min", "xl_step", "t0_ms", "dt_ms"):
        if field in volume_geometry and geom.get(field) != volume_geometry.get(field):
            raise RuntimeError(f"Stage 12 geometry {field}={geom.get(field)!r} does not match Stage 11 {volume_geometry.get(field)!r}")
    products = gm.get("products", [])
    required_products = {"geobody_labels": "npy", "geobody_catalog": "csv", "geobody_run_length_index": "csv", "geobody_spatial_export_reconciliation": "csv"}
    paths = {}
    for name, fmt in required_products.items():
        row = _find_product(products, name, fmt)
        if not row:
            raise RuntimeError(f"Stage 12 manifest does not register required product {name}")
        p = _product_path(jobs["source_geobody_job_id"], row)
        if not p.is_file() or (row.get("sha256") and _sha256(p) != str(row["sha256"]).lower()):
            raise RuntimeError(f"Stage 12 product missing or checksum mismatch: {name}")
        paths[name] = p
    labels = np.load(paths["geobody_labels"], mmap_mode="r", allow_pickle=False)
    if tuple(labels.shape) != shape:
        raise RuntimeError(f"Stage 12 label array shape {labels.shape} != manifest {shape}")
    catalog = pd.read_csv(paths["geobody_catalog"])
    if not {"geobody_id", "voxel_count"}.issubset(catalog.columns):
        raise RuntimeError("Stage 12 catalogue is missing geobody_id / voxel_count")
    reconciliation = pd.read_csv(paths["geobody_spatial_export_reconciliation"])
    if "reconciles" not in reconciliation or not reconciliation.reconciles.astype(str).str.casefold().isin(["true", "1", "yes"]).all():
        raise RuntimeError("Stage 12 spatial-export reconciliation contains failures")
    actual_counts: dict[int, int] = {}
    for i0 in range(0, shape[0], 16):
        label_ids, label_counts = np.unique(np.asarray(labels[i0:i0 + 16]), return_counts=True)
        for gid, count in zip(label_ids, label_counts):
            if int(gid) > 0:
                actual_counts[int(gid)] = actual_counts.get(int(gid), 0) + int(count)
    for r in catalog.itertuples():
        if actual_counts.get(int(r.geobody_id), 0) != int(r.voxel_count):
            raise RuntimeError(f"Stage 12 catalogue voxel count mismatch for geobody {r.geobody_id}")
    run_length = pd.read_csv(paths["geobody_run_length_index"], usecols=["geobody_id", "volume_i", "volume_j", "il", "xl", "x", "y", "top_time_k", "base_time_k", "voxel_count"])
    area, area_qc, x_grid, y_grid = _cell_area_grid(run_length, shape[:2])
    area_coordinate_units = area.copy()
    xy_units = str(config.get("xy_coordinate_units") or "").strip().casefold()
    unit_to_m = {"m": 1.0, "metre": 1.0, "meter": 1.0,
                 "ft": 0.3048, "foot": 0.3048, "feet": 0.3048,
                 "us_survey_foot": 1200.0 / 3937.0}
    xy_to_m = unit_to_m.get(xy_units)
    area_qc.update({"coordinate_units": xy_units or None,
                    "units_status": "CONFIRMED_SUPPORTED" if xy_to_m is not None else "UNCONFIRMED_OR_UNSUPPORTED",
                    "conversion_factor_to_m": xy_to_m})
    if xy_to_m is not None:
        area = area * (xy_to_m ** 2)
    trace_grid_path = jobs["inference_job_id"] / "stages" / "09_inference" / "volumes" / "trace_index_grid.npy"
    if not trace_grid_path.is_file():
        raise RuntimeError(f"Canonical Stage 09 trace_index_grid is missing: {trace_grid_path}")
    trace_grid = np.load(trace_grid_path, mmap_mode="r", allow_pickle=False)
    if tuple(trace_grid.shape) != shape[:2]:
        raise RuntimeError(f"Stage 09 trace_index_grid shape {trace_grid.shape} != geobody live-bin shape {shape[:2]}")
    live_columns = int(np.count_nonzero(np.asarray(trace_grid) >= 0))

    samples, intervals = _path_samples(trajectory, labels, geom, well_ids)
    cfg = config
    pad = float(cfg.get("twt_pad_ms", 8.0)); min_samples = int(cfg.get("min_interval_samples", 3)); gap = float(cfg.get("merge_gap_ms", 8.0))
    if anchor_mode == "intersecting_configured_wells":
        supported = sorted(
            well for well in well_ids
            if not intervals.empty
            and bool(((intervals.well == well) & (intervals.n_path_samples >= min_samples)).any())
        )
        anchor_wells = supported
    anchor = samples[samples.well.isin(anchor_wells)].copy()
    out = current / "stages" / STAGE
    tables = out / "tables"; tables.mkdir(parents=True, exist_ok=True)
    audit = _well_intersection_audit(samples, intervals, well_ids, anchor_wells, min_samples)
    audit_path = tables / "prospect_well_intersection_audit.csv"
    audit.to_csv(audit_path, index=False)
    if intervals.empty:
        detail = ", ".join(f"{r.well}: {r.intersection_status}" for r in audit.itertuples())
        raise RuntimeError("No configured wells intersect any retained geobody. Per-well audit: " + detail +
                           f". Audit exported to {audit_path}")
    # Merge nearby same-body intervals per anchor well, then apply minimum sample support.
    merged_rows = []
    for (well, gid), frame in intervals[intervals.well.isin(anchor_wells)].sort_values("twt_top_ms").groupby(["well", "geobody_id"]):
        current_iv = None
        for r in frame.itertuples():
            if current_iv and float(r.twt_top_ms) - current_iv["twt_base_ms"] <= gap + float(geom["dt_ms"]):
                current_iv["twt_base_ms"] = float(r.twt_base_ms); current_iv["n_path_samples"] += int(r.n_path_samples)
            else:
                if current_iv: merged_rows.append(current_iv)
                current_iv = {"well": well, "geobody_id": int(gid), "twt_top_ms": float(r.twt_top_ms), "twt_base_ms": float(r.twt_base_ms), "n_path_samples": int(r.n_path_samples)}
        if current_iv: merged_rows.append(current_iv)
    anchor_intervals = pd.DataFrame(merged_rows, columns=[
        "well", "geobody_id", "twt_top_ms", "twt_base_ms", "n_path_samples"
    ])
    anchor_intervals["kept"] = anchor_intervals["n_path_samples"] >= min_samples
    anchor_intervals = anchor_intervals[anchor_intervals.kept].reset_index(drop=True)
    if anchor_intervals.empty:
        anchor_detail = ", ".join(
            f"{r.well}: {r.anchor_status} ({r.trajectory_samples_on_geobody} path samples, "
            f"{r.raw_intervals} raw intervals)" for r in audit[audit.is_anchor].itertuples()
        )
        raise RuntimeError(
            f"No configured anchor well has an interval meeting min_interval_samples={min_samples}. "
            f"Anchor audit: {anchor_detail}. Non-anchor intersections do not automatically become prospect anchors. "
            f"Review {audit_path}; in automatic mode ensure at least one configured well intersects a retained geobody, "
            f"or in explicit mode configure a supported anchor_well_id."
        )

    depth_models, well_depth_models, selected_regional = _load_stage01_depth_models(stage01)
    grv_models = {name: model for name, model in depth_models.items()
                  if str(model.get("source_type", "")).startswith("regional_")}
    if not grv_models:
        raise RuntimeError("Foundation Stage 01 has no usable regional T-Z model for survey-wide GRV")
    depth_model_selection_status = "STAGE01_SELECTED_REGIONAL_MODEL"
    if selected_regional not in grv_models:
        selected_regional = sorted(grv_models)[0]
        depth_model_selection_status = "FALLBACK_FIRST_AVAILABLE_REGIONAL_MODEL_REQUIRES_REVIEW"
    t0, dt = float(geom["t0_ms"]), float(geom["dt_ms"])
    nK = labels.shape[2]
    prospects, prospect_voxels, well_property_rows, sensitivity_rows, grv_rows = [], {}, [], [], []
    point_frames: dict[str, pd.DataFrame] = {}
    pooled_logs: dict[str, pd.DataFrame] = {}
    cut = cfg.get("net_cutoffs", {"vsh_max": 0.4, "phit_min": 0.03})
    cut_grid = cfg.get("cutoff_grid", {"vsh_max": [0.3, 0.4, 0.5], "phit_min": [0.03, 0.08, 0.12]})
    for pidx, r in enumerate(anchor_intervals.itertuples(), start=1):
        pid = f"P{pidx:03d}"
        ka = max(0, int(round((r.twt_top_ms - t0) / dt)) - int(round(pad / dt)))
        kb = min(nK - 1, int(round((r.twt_base_ms - t0) / dt)) + int(round(pad / dt)))
        seed = anchor[(anchor.well == r.well) & (anchor.geobody_id == r.geobody_id) & anchor.twt_ms.between(r.twt_top_ms, r.twt_base_ms)]
        vi, vj, vk, truncated = _prospect_voxels(labels, int(r.geobody_id), ka, kb, seed)
        if not len(vi):
            continue
        prospect_voxels[pid] = (vi, vj, vk)
        cols = np.unique(vi.astype(np.int64) * shape[1] + vj)
        kmin, kmax = int(vk.min()), int(vk.max())
        top_twt, base_twt = t0 + kmin * dt, t0 + kmax * dt
        top_depth, base_depth = _depth_from_model(grv_models[selected_regional],
                                                   np.asarray([top_twt - dt / 2, base_twt + dt / 2]))
        xvals, yvals = x_grid[vi, vj], y_grid[vi, vj]
        finite_xy = np.isfinite(xvals) & np.isfinite(yvals)
        prospects.append({"prospect": pid, "anchor_well": r.well, "geobody_id": int(r.geobody_id),
                          "anchor_twt_top_ms": r.twt_top_ms, "anchor_twt_base_ms": r.twt_base_ms,
                          "anchor_path_samples": int(r.n_path_samples), "window_twt_top_ms": t0 + ka * dt,
                          "window_twt_base_ms": t0 + kb * dt, "n_voxels": int(len(vi)),
                          "n_occupied_columns": int(len(cols)),
                          "footprint_area_km2": float(np.sum(area.ravel()[cols]) / 1e6) if xy_to_m is not None else None,
                          "footprint_area_coordinate_units2": float(np.sum(area_coordinate_units.ravel()[cols])),
                          "il_min": int(geom["il_min"] + vi.min() * geom["il_step"]),
                          "il_max": int(geom["il_min"] + vi.max() * geom["il_step"]),
                          "xl_min": int(geom["xl_min"] + vj.min() * geom["xl_step"]),
                          "xl_max": int(geom["xl_min"] + vj.max() * geom["xl_step"]),
                          "centroid_il": float(geom["il_min"] + np.mean(vi) * geom["il_step"]),
                          "centroid_xl": float(geom["xl_min"] + np.mean(vj) * geom["xl_step"]),
                          "centroid_x": float(np.mean(xvals[finite_xy])) if finite_xy.any() else None,
                          "centroid_y": float(np.mean(yvals[finite_xy])) if finite_xy.any() else None,
                          "xy_support_fraction": float(finite_xy.mean()),
                          "twt_top_ms": top_twt, "twt_base_ms": base_twt,
                          "top_tvdss_m": float(top_depth) if np.isfinite(top_depth) else None,
                          "base_tvdss_m": float(base_depth) if np.isfinite(base_depth) else None,
                          "window_truncated": bool(truncated),
                          "touches_survey_edge": bool(vi.min() == 0 or vj.min() == 0 or vi.max() == shape[0]-1 or vj.max() == shape[1]-1),
                          "fill_assumption": "NO_CONTACT: 100% of the candidate sand body counted; upside geometric assumption",
                          "interpretation_status": "EXPLORATORY_UNCALIBRATED_CANDIDATE"})
        point_frames[pid] = pd.DataFrame({"prospect": pid, "parent_geobody_id": int(r.geobody_id),
            "volume_i": vi, "volume_j": vj, "time_k": vk,
            "il": geom["il_min"] + vi * geom["il_step"], "xl": geom["xl_min"] + vj * geom["xl_step"],
            "x": xvals, "y": yvals, "twt_ms": t0 + vk * dt,
            "tvdss_m": _depth_from_model(grv_models[selected_regional], t0 + vk * dt)})
        # GRV for every available Stage 01 model, no extrapolation beyond model support.
        for model_name, model in grv_models.items():
            top = t0 + vk * dt - dt / 2; base = t0 + vk * dt + dt / 2
            ztop = _depth_from_model(model, top); zbase = _depth_from_model(model, base)
            dz = np.abs(zbase - ztop)
            a = area[vi, vj]
            valid = np.isfinite(dz) & np.isfinite(a) & (dz > 0) & (a > 0)
            grv = float(np.sum(a[valid] * dz[valid])) if valid.any() and xy_to_m is not None else None
            status = ("COMPUTED" if valid.any() and xy_to_m is not None else
                      "BLOCKED_XY_UNITS_UNCONFIRMED" if xy_to_m is None else "NO_DEPTH_SUPPORT")
            grv_rows.append({"prospect": pid, "depth_model": model_name,
                             "is_selected_regional_model": model_name == selected_regional,
                             "grv_m3": grv, "valid_voxels": int(valid.sum()) if xy_to_m is not None else 0,
                             "total_voxels": int(len(vk)), "valid_fraction": float(valid.mean()) if xy_to_m is not None else 0.0,
                             "status": status})
        # Compute the prospect's intersecting log samples once, then partition by well.
        Lall = _logs_for_prospect(logs, samples, prospect_voxels[pid], dt, shape[1], shape[2])
        pooled_logs[pid] = Lall
        for well in well_ids:
            Lw = Lall[Lall.well == well] if not Lall.empty else pd.DataFrame()
            stat = _log_stats(Lw, float(cut["vsh_max"]), float(cut["phit_min"]))
            well_property_rows.append({"prospect": pid, "well": well, "intersects_prospect": not Lw.empty, **stat,
                                       "swt_status": "SUPPORTED" if stat.get("swt_net_pv") is not None else "UNAVAILABLE_OR_INSUFFICIENT_LOG_SUPPORT"})
        for vm in cut_grid.get("vsh_max", []):
            for pm in cut_grid.get("phit_min", []):
                sensitivity_rows.append({"prospect": pid, "vsh_max": float(vm), "phit_min": float(pm), **_log_stats(Lall, float(vm), float(pm))})

    if not prospects:
        raise RuntimeError("No non-empty well-anchored prospect components were extracted")
    prospect_df = pd.DataFrame(prospects)
    grv_df = pd.DataFrame(grv_rows)
    well_df = pd.DataFrame(well_property_rows)
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    # Prospect-level well-pooled log statistics are explicitly labelled as pooled samples.
    pooled_rows = []
    for pid, L in pooled_logs.items():
        pooled_rows.append({"prospect": pid, "well": "ALL_INTERSECTING_WELLS", **_log_stats(L, float(cut["vsh_max"]), float(cut["phit_min"]))})
    pooled = pd.DataFrame(pooled_rows)
    prospect_df = prospect_df.merge(pooled, on="prospect", how="left", suffixes=("", "_logs"))
    # Conditional uniform-property HCPV scenario. It is withheld when any hard
    # screening gate fails; no geological qualification is assigned automatically.
    min_net = float(cfg.get("min_net_m_per_well", 1.0))
    max_footprint_fraction = float(cfg.get("review_max_footprint_fraction", 0.05))
    water_sw_min = float(cfg.get("water_sw_min", 0.90))
    hc_sw_max = float(cfg.get("hc_sw_max", 0.60))
    deterministic_rows = []
    uncertainty_rows = []
    bootstrap_n = int(cfg.get("bootstrap_replicates", 1000))
    rng = np.random.default_rng(int(cfg.get("bootstrap_seed", 14)))
    velocity_fraction = float(cfg.get("velocity_uncertainty_fraction", 0.10))
    for row in prospect_df.itertuples():
        pid = row.prospect
        g = grv_df[grv_df.prospect == pid]
        chosen = g[g.depth_model == selected_regional] if selected_regional else pd.DataFrame()
        chosen = chosen[chosen.status == "COMPUTED"] if not chosen.empty else chosen
        grv_value = float(chosen.grv_m3.iloc[0]) if not chosen.empty else None
        grv_fraction = float(chosen.valid_fraction.iloc[0]) if not chosen.empty else 0.0
        L = pooled_logs[pid]
        net_wells = well_df[(well_df.prospect == pid) & (well_df.net_m >= min_net)]
        water_wells = net_wells[net_wells.swt_net_pv >= water_sw_min]
        hc_wells = net_wells[net_wells.swt_net_pv <= hc_sw_max]
        footprint_fraction = float(row.n_occupied_columns / max(live_columns, 1))
        flags = {
            "survey_scale_object": footprint_fraction > max_footprint_fraction,
            "mixed_fluid_wells": bool(len(water_wells) and len(hc_wells)),
            "water_bearing_at_wells": bool(row.swt_net_pv is not None and pd.notna(row.swt_net_pv) and row.swt_net_pv >= water_sw_min),
            "no_net_support": not len(net_wells),
            "insufficient_swt_support": not (pd.notna(row.swt_net_pv) and float(row.swt_support_m) >= min_net),
            "depth_invalid": grv_fraction < 0.95,
            "empty": int(row.n_voxels) == 0,
        }
        blockers = [k for k, v in flags.items() if v]
        ntg = float(row.ntg) if pd.notna(row.ntg) else None
        phi = float(row.phit_net) if pd.notna(row.phit_net) else None
        sw = float(row.swt_net_pv) if pd.notna(row.swt_net_pv) else None
        scenario = (grv_value * ntg * phi * (1.0 - sw)
                    if not blockers and grv_value is not None and all(v is not None for v in (ntg, phi, sw))
                    else None)
        deterministic_rows.append({"prospect": pid, "reporting_level": "SCENARIO_HCPV" if scenario is not None else "GEOMETRY_ONLY",
                                    "level_reasons": ";".join(blockers) if blockers else "CONDITIONAL_LOG_PROPERTY_SCENARIO",
                                    "selected_depth_model": selected_regional, "grv_m3": grv_value,
                                    "grv_valid_fraction": grv_fraction, "ntg_logs": ntg,
                                    "phit_logs": phi, "swt_logs_pv": sw,
                                    "hcpv_scenario_uniform_m3": scenario,
                                    "hcpv_basis": "GRV x log-derived NTG x total PHIT x (1 - pore-volume-weighted SWT); uniform properties over candidate body",
                                    "porosity_type": "PHIT_TOTAL_NOT_PHIE",
                                    "fill_assumption": "NO_CONTACT_FULL_CANDIDATE_BODY_UPSIDE_CASE",
                                    "footprint_fraction_of_live_survey": footprint_fraction,
                                    "n_wells_with_net": int(len(net_wells)),
                                    "wells_water_bearing": ";".join(water_wells.well.astype(str)),
                                    "wells_low_swt": ";".join(hc_wells.well.astype(str)),
                                    "flag_xy_units_unconfirmed": xy_to_m is None,
                                    "flag_window_truncated": bool(row.window_truncated),
                                    "flag_touches_survey_edge": bool(row.touches_survey_edge),
                                    **{f"flag_{k}": v for k, v in flags.items()},
                                    "status_hcpv": "COMPUTED_CONDITIONAL_SCENARIO" if scenario is not None else "WITHHELD"})
        regional_names = list(grv_models)
        grv_all = g[g.depth_model.isin(regional_names) & (g.status == "COMPUTED")].grv_m3.dropna().to_numpy(dtype=float)
        grv_tuple = None
        if grv_value is not None and len(grv_all):
            grv_tuple = (float(min(grv_all.min(), grv_value)), grv_value, float(max(grv_all.max(), grv_value)))
        uncertainty_rows.append({"prospect": pid, **_bootstrap_screening(
            pooled_logs[pid], bootstrap_n, rng, float(cut["vsh_max"]), float(cut["phit_min"]),
            grv_tuple, velocity_fraction, scenario is not None)})
    deterministic_df = pd.DataFrame(deterministic_rows)
    uncertainty_df = pd.DataFrame(uncertainty_rows)
    prospect_df = prospect_df.merge(deterministic_df[["prospect", "reporting_level", "level_reasons", "hcpv_scenario_uniform_m3", "status_hcpv"]], on="prospect", how="left")
    # Export voxel-count sufficient statistics for alternate T-Z / area assumptions.
    hist_rows = []
    for pid, (_, _, vk) in prospect_voxels.items():
        counts = np.bincount(vk, minlength=nK)
        for k in np.flatnonzero(counts):
            hist_rows.append({"prospect": pid, "time_k": int(k), "twt_ms": t0 + int(k) * dt, "voxel_count": int(counts[k])})

    outputs = {
        "prospect_well_intersection_audit.csv": audit,
        "prospect_anchor_intervals.csv": anchor_intervals,
        "prospect_definitions.csv": prospect_df,
        "prospect_well_properties.csv": well_df,
        "prospect_cutoff_sensitivity.csv": sensitivity_df,
        "prospect_grv_by_depth_model.csv": grv_df,
        "prospect_volumetrics_deterministic.csv": deterministic_df,
        "prospect_uncertainty_screening.csv": uncertainty_df,
        "prospect_voxel_count_by_sample.csv": pd.DataFrame(hist_rows),
    }
    for name, frame in outputs.items():
        frame.to_csv(tables / name, index=False)
    # Persist exact prospect voxel coordinates as compact NPY index triplets for later specialist work.
    voxel_dir = out / "intermediate" / "prospect_voxels"; voxel_dir.mkdir(parents=True, exist_ok=True)
    point_dir = out / "outputs" / "points"; point_dir.mkdir(parents=True, exist_ok=True)
    max_points_rows = int(cfg.get("max_points_csv_rows", 20_000_000))
    point_export_status = {}
    for pid, (vi, vj, vk) in prospect_voxels.items():
        np.save(voxel_dir / f"{pid}_ijk.npy", np.column_stack([vi, vj, vk]).astype(np.int32), allow_pickle=False)
        if len(vi) <= max_points_rows:
            point_path = point_dir / f"{pid}_points_xyz.csv"
            point_frames[pid].to_csv(point_path, index=False)
            point_export_status[pid] = {"status": "EXPORTED", "path": str(point_path.relative_to(current)), "rows": int(len(vi))}
        else:
            point_export_status[pid] = {"status": "SKIPPED_ROW_LIMIT", "rows": int(len(vi)), "limit": max_points_rows}
    source = {"foundation_job_id": ids["foundation_job_id"], "inference_job_id": ids["inference_job_id"],
              "source_volume_job_id": ids["source_volume_job_id"], "source_geobody_job_id": ids["source_geobody_job_id"]}
    source_products = {
        "stage01_trajectory": trajectory_path, "stage01_regional_model_selection": stage01 / "regional_model_selection.csv",
        "stage01_depth_model_coefficients": stage01 / "polynomial_coefficients_used.csv",
        "stage02_interpreted_logs": logs_path, "stage09_trace_index_grid": trace_grid_path,
        "stage11_manifest": jobs["source_volume_job_id"] / "stages" / volume_stage_id / "native_volume_export_manifest.json",
        **{f"stage12_{name}": path for name, path in paths.items()},
    }
    source_fingerprints = {name: {"path": str(path.relative_to(jobs_root.parent)),
                                   "sha256": _sha256(path), "bytes": path.stat().st_size}
                           for name, path in source_products.items() if path.is_file()}
    products = []
    for p in sorted(tables.glob("*.csv")):
        products.append({"product": p.stem, "format": "csv", "path": str(p.relative_to(current)), "sha256": _sha256(p), "bytes": p.stat().st_size})
    for p in sorted(voxel_dir.glob("*.npy")):
        products.append({"product": p.stem, "format": "npy", "path": str(p.relative_to(current)), "sha256": _sha256(p), "bytes": p.stat().st_size})
    for p in sorted(point_dir.glob("*.csv")):
        products.append({"product": p.stem, "format": "csv", "path": str(p.relative_to(current)), "sha256": _sha256(p), "bytes": p.stat().st_size})
    swt_supported = bool(well_df.swt_net_pv.notna().any())
    grv_supported = bool((grv_df.status == "COMPUTED").any())
    summary = {"schema_version": 1, "stage": STAGE, "status": "COMPLETED" if grv_supported else "COMPLETED_WITH_GRV_BLOCKERS",
               "interpretation_status": "EXPLORATORY_UNCALIBRATED_CANDIDATE",
               "phase_scope": ["PHASE_1_PROSPECT_DEFINITION_GEOMETRY_GRV", "PHASE_2_WELL_LOG_CHARACTERISATION"],
               "source": source, "source_products": source_fingerprints,
               "depth_model_selection": {"model": selected_regional, "status": depth_model_selection_status},
               "well_ids": well_ids, "anchor_well_ids": anchor_wells,
                "anchor_selection_mode": anchor_mode,
               "counts": {"prospects": len(prospect_df), "anchor_intervals": len(anchor_intervals),
                          "wells_with_log_support": int(well_df.groupby("well").gross_m.sum().gt(0).sum())},
               "point_exports": point_export_status,
               "qc": {"cell_area": area_qc, "grv_available": grv_supported,
                      "well_log_swt_available": swt_supported,
                      "xy_coordinate_units": xy_units or None,
                      "xy_coordinate_units_confirmed": xy_to_m is not None,
                      "hcpv_calculated": bool(deterministic_df.hcpv_scenario_uniform_m3.notna().any()),
                      "hcpv_status": "CONDITIONAL_SCENARIO_ONLY" if deterministic_df.hcpv_scenario_uniform_m3.notna().any() else "WITHHELD_BY_QUALIFICATION_GATES",
                      "swt_note": "SWT is derived only from interpreted Stage 02 well logs; no Stage 11 SWT volume is required. Missing or insufficient well-log SWT remains unavailable."},
               "products": products,
               "limitations": ["Geobodies are exploratory uncalibrated seismic candidates, not confirmed reservoirs.",
                               "TVDSS and GRV depend on provisional Stage 01 depth models and Stage 12 CDP XY footprint estimates.",
                               "Well-log properties are sampled only where registered well trajectories intersect prospect voxels; spatial extrapolation is not independently validated.",
                               "No structural closure, fluid contact, hydrocarbon presence, net pay, HCPV, STOIIP, GIIP or reserves claim is made.",
                               "HCPV, when reported, is a conditional uniform-property screening scenario only; it is not a resource estimate."]}
    _atomic_json(out / "prospect_manifest.json", summary)
    return summary


def main() -> None:
    result_path = Path(os.environ["APGCE_RESULT_PATH"]).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path(os.environ.get("APGCE_GEOGG_PROSPECT_CONFIG") or Path(__file__).resolve().parents[2] / "F_prospect.yaml").expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Stage 14 requires PyYAML") from exc
    root = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = root.get("prospect", {})
    if not isinstance(config, dict):
        raise RuntimeError("prospect YAML section must be a mapping")
    result = run(config, result_path)
    preflight = {"schema_version": 1, "stage": STAGE, "status": result["status"],
                 "source": result["source"], "well_ids": result["well_ids"],
                 "anchor_well_ids": result["anchor_well_ids"], "anchor_selection_mode": result["anchor_selection_mode"],
                  "counts": result["counts"], "qc": result["qc"]}
    _atomic_json(result_path.parent / PREFLIGHT, preflight)
    _atomic_json(result_path.parent / SUMMARY, result)


if __name__ == "__main__":
    main()
