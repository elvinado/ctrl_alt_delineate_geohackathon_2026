"""Stage 13: intersect registered deviated-well trajectories with Stage 12 geobodies.

This stage consumes immutable upstream processing jobs. It does not rerun inference or
geobody extraction, and it does not treat geometric candidates as confirmed reservoirs.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


PREFLIGHT = "geogg_well_geobody_analysis_preflight.json"
SUMMARY = "geogg_well_geobody_analysis_summary.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_json(path: Path, label: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return value


def _safe_job(jobs_root: Path, job_id: str, label: str) -> Path:
    if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
        raise RuntimeError(f"{label} job ID is missing or unsafe: {job_id!r}")
    path = jobs_root / job_id
    if path.is_symlink() or not path.is_dir() or path.resolve().parent != jobs_root.resolve():
        raise RuntimeError(f"{label} processing job is unavailable or unsafe: {job_id}")
    return path.resolve()


def _product_path(jobs_root: Path, row: dict) -> Path:
    rel = str(row.get("storage_root_relative_path") or row.get("path") or "").strip()
    if not rel:
        raise RuntimeError(f"Registered product {row.get('product')} has no path")
    path = (jobs_root.parent / rel).resolve()
    try:
        path.relative_to(jobs_root.parent.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Registered product path escapes storage root: {rel}") from exc
    return path


def _find_product(products: list, name: str, fmt: str | None = None) -> dict | None:
    for row in products:
        if not isinstance(row, dict) or str(row.get("product", "")).strip() != name:
            continue
        if fmt is None or str(row.get("format", "")).lower() == fmt.lower():
            return row
    return None


def _intervals(samples: pd.DataFrame, well: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sampled path intersections and contiguous MD-ordered geobody intervals."""
    cols = ["well", "sample_index", "md_m", "tvdss_m", "twt_ms", "il", "xl", "volume_i", "volume_j",
            "time_k", "geobody_id", "inside_volume", "has_geobody"]
    if samples.empty:
        return pd.DataFrame(columns=cols), pd.DataFrame()
    s = samples.sort_values("md_m", kind="stable").reset_index(drop=True).copy()
    ids = s["geobody_id"].to_numpy(dtype=np.int64)
    # A non-geobody sample (0) breaks an interval, as does a change in ID.
    run_start = np.r_[True, ids[1:] != ids[:-1]]
    s["_run"] = np.cumsum(run_start)
    hits = s[s["geobody_id"] > 0]
    if hits.empty:
        return s.drop(columns=["_run"]), pd.DataFrame(columns=["well", "geobody_id", "interval_index", "md_top_m",
            "md_base_m", "twt_top_ms", "twt_base_ms", "n_path_samples", "il_top", "xl_top", "il_base", "xl_base"])
    rows = []
    for (_, gid, run), group in hits.groupby(["well", "geobody_id", "_run"], sort=False):
        first, last = group.iloc[0], group.iloc[-1]
        rows.append({"well": well, "geobody_id": int(gid), "interval_index": int(run),
                     "md_top_m": float(group.md_m.min()), "md_base_m": float(group.md_m.max()),
                     "twt_top_ms": float(group.twt_ms.min()), "twt_base_ms": float(group.twt_ms.max()),
                     "n_path_samples": int(len(group)), "il_top": int(first.il), "xl_top": int(first.xl),
                     "il_base": int(last.il), "xl_base": int(last.xl)})
    return s.drop(columns=["_run"]), pd.DataFrame(rows)


def _body_geometry(labels: np.ndarray, gid: int, geometry: dict) -> dict:
    """Compute exact voxel/column geometry for one body without materialising a full-volume mask."""
    il_min, il_step = int(geometry["il_min"]), int(geometry["il_step"])
    xl_min, xl_step = int(geometry["xl_min"]), int(geometry["xl_step"])
    t0, dt = float(geometry["t0_ms"]), float(geometry["dt_ms"])
    nvox = ncols = 0
    sum_i = sum_j = sum_k = 0.0
    i_min, i_max = labels.shape[0], -1
    j_min, j_max = labels.shape[1], -1
    k_min, k_max = labels.shape[2], -1
    # Iterate inline slabs so memory use remains bounded for large candidate bodies.
    for i0 in range(0, labels.shape[0], 16):
        block = np.asarray(labels[i0:i0 + 16])
        ii, jj, kk = np.nonzero(block == gid)
        if not len(ii):
            continue
        ii = ii.astype(np.int64) + i0
        nvox += int(len(ii))
        ncols += int(len(np.unique(ii * labels.shape[1] + jj)))
        sum_i += float(ii.sum()); sum_j += float(jj.sum()); sum_k += float(kk.sum())
        i_min = min(i_min, int(ii.min())); i_max = max(i_max, int(ii.max()))
        j_min = min(j_min, int(jj.min())); j_max = max(j_max, int(jj.max()))
        k_min = min(k_min, int(kk.min())); k_max = max(k_max, int(kk.max()))
    if not nvox:
        return {"geobody_id": gid, "voxel_count_recounted": 0, "geometry_status": "NOT_FOUND_IN_LABEL_VOLUME"}
    # Centroids are voxel-weighted; footprint is occupied IL/XL columns, not bbox area.
    return {"geobody_id": gid, "voxel_count_recounted": nvox, "occupied_columns": ncols,
            "il_min": il_min + i_min * il_step, "il_max": il_min + i_max * il_step,
            "xl_min": xl_min + j_min * xl_step, "xl_max": xl_min + j_max * xl_step,
            "centroid_il": il_min + (sum_i / nvox) * il_step,
            "centroid_xl": xl_min + (sum_j / nvox) * xl_step,
            "twt_top_ms": t0 + k_min * dt, "twt_base_ms": t0 + k_max * dt,
            "centroid_twt_ms": t0 + (sum_k / nvox) * dt,
            "gross_time_span_ms": (k_max - k_min + 1) * dt,
            "geometry_status": "COMPUTED_FROM_LABEL_VOLUME"}


def _hist_quantile(hist: np.ndarray, edges: np.ndarray, q: float) -> float | None:
    total = float(hist.sum())
    if total <= 0:
        return None
    target = q * total
    cumulative = np.cumsum(hist)
    idx = min(int(np.searchsorted(cumulative, target, side="left")), len(hist) - 1)
    before = float(cumulative[idx - 1]) if idx else 0.0
    count = float(hist[idx])
    fraction = 0.0 if count <= 0 else max(0.0, min(1.0, (target - before) / count))
    return float(edges[idx] + fraction * (edges[idx + 1] - edges[idx]))


def _property_stats(labels: np.ndarray, gid: int, phit: np.ndarray | None,
                    swt: np.ndarray | None, min_joint_support_fraction: float = 0.5,
                    min_joint_voxels: int = 100, histogram_bins: int = 1000,
                    volume_weight_model: dict | None = None, geometry: dict | None = None) -> dict:
    """Bounded-memory PHIT/SWT statistics over canonical Stage 11 predictions.

    Histograms provide deterministic, bounded-memory P10/P50/P90 estimates over
    the expected fraction domain [0, 1]. Physical-volume weighting is deliberately
    not inferred from time samples; Stage 13 does not have calibrated lateral cell
    area and depth-cell volumes for every voxel.
    """
    if phit is None or swt is None:
        return {"property_status": "NOT_COMPUTED_SOURCE_ARRAY_UNAVAILABLE",
                "property_coverage_gate": "FAIL_SOURCE_ARRAY_UNAVAILABLE"}
    edges = np.linspace(0.0, 1.0, int(histogram_bins) + 1)
    h_phit = np.zeros(histogram_bins, dtype=np.int64)
    h_swt = np.zeros(histogram_bins, dtype=np.int64)
    h_phit_joint = np.zeros(histogram_bins, dtype=np.int64)
    h_swt_joint = np.zeros(histogram_bins, dtype=np.int64)
    h_hcpf = np.zeros(histogram_bins, dtype=np.int64)
    h_phit_vw = np.zeros(histogram_bins, dtype=np.float64)
    h_swt_vw = np.zeros(histogram_bins, dtype=np.float64)
    h_phit_joint_vw = np.zeros(histogram_bins, dtype=np.float64)
    h_swt_joint_vw = np.zeros(histogram_bins, dtype=np.float64)
    h_hcpf_vw = np.zeros(histogram_bins, dtype=np.float64)
    n_body = n_phit = n_swt = n_joint = 0
    n_volume_weighted = 0
    weighted_sums = {"phit": 0.0, "swt": 0.0, "phit_joint": 0.0,
                     "swt_joint": 0.0, "hcpf": 0.0}
    weighted_totals = {"phit": 0.0, "swt": 0.0, "phit_joint": 0.0,
                       "swt_joint": 0.0, "hcpf": 0.0}
    n_phit_out = n_swt_out = 0
    sums = {"phit": 0.0, "swt": 0.0, "joint_phit": 0.0, "joint_swt": 0.0, "hcpf": 0.0}
    minima = {"phit": float("inf"), "swt": float("inf"), "hcpf": float("inf")}
    maxima = {"phit": float("-inf"), "swt": float("-inf"), "hcpf": float("-inf")}
    for i0 in range(0, labels.shape[0], 16):
        mask = np.asarray(labels[i0:i0 + 16]) == gid
        if not mask.any():
            continue
        a = np.asarray(phit[i0:i0 + 16], dtype=np.float64)[mask]
        b = np.asarray(swt[i0:i0 + 16], dtype=np.float64)[mask]
        n_body += int(mask.sum())
        finite_a, finite_b = np.isfinite(a), np.isfinite(b)
        valid_a = finite_a & (a >= 0.0) & (a <= 1.0)
        valid_b = finite_b & (b >= 0.0) & (b <= 1.0)
        n_phit_out += int((finite_a & ~valid_a).sum())
        n_swt_out += int((finite_b & ~valid_b).sum())
        n_phit += int(valid_a.sum()); n_swt += int(valid_b.sum())
        if valid_a.any():
            av = a[valid_a]
            sums["phit"] += float(av.sum()); minima["phit"] = min(minima["phit"], float(av.min()))
            maxima["phit"] = max(maxima["phit"], float(av.max()))
            h_phit += np.histogram(av, bins=edges)[0]
        if valid_b.any():
            bv = b[valid_b]
            sums["swt"] += float(bv.sum()); minima["swt"] = min(minima["swt"], float(bv.min()))
            maxima["swt"] = max(maxima["swt"], float(bv.max()))
            h_swt += np.histogram(bv, bins=edges)[0]
        both = valid_a & valid_b
        n_joint += int(both.sum())
        if volume_weight_model is not None and geometry is not None:
            ii, jj, kk = np.nonzero(mask)
            twt_centres = float(geometry["t0_ms"]) + kk.astype(np.float64) * float(geometry["dt_ms"])
            half_dt = 0.5 * float(geometry["dt_ms"])
            lower, upper = twt_centres - half_dt, twt_centres + half_dt
            support = ((lower >= volume_weight_model["twt_min"]) &
                       (upper <= volume_weight_model["twt_max"]))
            local_weights = np.zeros(len(kk), dtype=np.float64)
            if support.any():
                z_lower = np.interp(lower[support], volume_weight_model["twt"], volume_weight_model["z"])
                z_upper = np.interp(upper[support], volume_weight_model["twt"], volume_weight_model["z"])
                local_weights[support] = np.abs(z_upper - z_lower)
            n_volume_weighted += int(np.count_nonzero(local_weights > 0))
            va_weighted = valid_a & (local_weights > 0)
            vb_weighted = valid_b & (local_weights > 0)
            both_weighted = both & (local_weights > 0)
            if va_weighted.any():
                w = local_weights[va_weighted]; v = a[va_weighted]
                h_phit_vw += np.histogram(v, bins=edges, weights=w)[0]
                weighted_sums["phit"] += float(np.dot(v, w)); weighted_totals["phit"] += float(w.sum())
            if vb_weighted.any():
                w = local_weights[vb_weighted]; v = b[vb_weighted]
                h_swt_vw += np.histogram(v, bins=edges, weights=w)[0]
                weighted_sums["swt"] += float(np.dot(v, w)); weighted_totals["swt"] += float(w.sum())
            if both_weighted.any():
                w = local_weights[both_weighted]; v = a[both_weighted] * (1.0 - b[both_weighted])
                joint_phit_values, joint_swt_values = a[both_weighted], b[both_weighted]
                h_phit_joint_vw += np.histogram(joint_phit_values, bins=edges, weights=w)[0]
                h_swt_joint_vw += np.histogram(joint_swt_values, bins=edges, weights=w)[0]
                h_hcpf_vw += np.histogram(v, bins=edges, weights=w)[0]
                weighted_sums["phit_joint"] += float(np.dot(joint_phit_values, w))
                weighted_sums["swt_joint"] += float(np.dot(joint_swt_values, w))
                weighted_totals["phit_joint"] += float(w.sum()); weighted_totals["swt_joint"] += float(w.sum())
                weighted_sums["hcpf"] += float(np.dot(v, w)); weighted_totals["hcpf"] += float(w.sum())
        if both.any():
            pa, sw = a[both], b[both]
            hcpf = pa * (1.0 - sw)
            sums["joint_phit"] += float(pa.sum()); sums["joint_swt"] += float(sw.sum())
            sums["hcpf"] += float(hcpf.sum())
            minima["hcpf"] = min(minima["hcpf"], float(hcpf.min()))
            maxima["hcpf"] = max(maxima["hcpf"], float(hcpf.max()))
            h_phit_joint += np.histogram(pa, bins=edges)[0]
            h_swt_joint += np.histogram(sw, bins=edges)[0]
            h_hcpf += np.histogram(hcpf, bins=edges)[0]
    phit_support = n_phit / n_body if n_body else 0.0
    swt_support = n_swt / n_body if n_body else 0.0
    joint_support = n_joint / n_body if n_body else 0.0
    gate = ("PASS" if n_joint >= int(min_joint_voxels) and joint_support >= float(min_joint_support_fraction)
            else "FAIL_INSUFFICIENT_JOINT_SUPPORT")
    result = {
        "property_status": "COMPUTED_FROM_FROZEN_STAGE11_PREDICTIONS",
        "property_coverage_gate": gate,
        "property_gate_min_joint_support_fraction": float(min_joint_support_fraction),
        "property_gate_min_joint_voxels": int(min_joint_voxels),
        "property_histogram_bins": int(histogram_bins),
        "property_statistic_method": "exact sums/min/max; histogram-estimated quantiles over [0,1]",
        "phit_voxels": n_phit, "swt_voxels": n_swt, "joint_phit_swt_voxels": n_joint,
        "phit_out_of_domain_voxels": n_phit_out, "swt_out_of_domain_voxels": n_swt_out,
        "phit_support_fraction": phit_support, "swt_support_fraction": swt_support,
        "joint_support_fraction": joint_support,
        "phit_mean_voxel": sums["phit"] / n_phit if n_phit else None,
        "swt_mean_voxel": sums["swt"] / n_swt if n_swt else None,
        "phit_mean_joint_support": sums["joint_phit"] / n_joint if n_joint else None,
        "swt_mean_joint_support": sums["joint_swt"] / n_joint if n_joint else None,
        "phit_min": minima["phit"] if n_phit else None, "phit_max": maxima["phit"] if n_phit else None,
        "swt_min": minima["swt"] if n_swt else None, "swt_max": maxima["swt"] if n_swt else None,
        "hcpf_mean_joint_support": sums["hcpf"] / n_joint if n_joint else None,
        "hcpf_min_joint_support": minima["hcpf"] if n_joint else None,
        "hcpf_max_joint_support": maxima["hcpf"] if n_joint else None,
        "phit_weighting": "voxel_count_weighted_on_regular_grid",
        "swt_weighting": "voxel_count_weighted_on_regular_grid",
        "provisional_volume_weighted_status": ("PROVISIONAL_VERTICAL_THICKNESS_WEIGHTED_PROXY"
            if volume_weight_model is not None and n_volume_weighted else "NOT_COMPUTED_NO_SUPPORTED_DEPTH_MODEL"),
        "provisional_volume_weighted_voxels": n_volume_weighted,
        "provisional_volume_weighted_limitation": "Weights use per-sample TVDSS thickness from the selected monotonic T-Z model and assume constant lateral cell area. This is a relative volume proxy, not calibrated m3; unsupported depth samples are excluded.",
    }
    for key, hist in (("phit", h_phit), ("swt", h_swt),
                      ("phit_joint", h_phit_joint), ("swt_joint", h_swt_joint),
                      ("hcpf", h_hcpf)):
        for q, label in ((0.10, "p10"), (0.50, "p50"), (0.90, "p90")):
            result[f"{key}_{label}"] = _hist_quantile(hist, edges, q) if hist.sum() else None
    # On a regular grid each valid voxel has equal count weight; report weighted
    # means explicitly rather than duplicating the same numbers under an ambiguous name.
    result["phit_mean_voxel_weighted"] = result["phit_mean_voxel"]
    result["swt_mean_voxel_weighted"] = result["swt_mean_voxel"]
    result["phit_mean_joint_voxel_weighted"] = result["phit_mean_joint_support"]
    result["swt_mean_joint_voxel_weighted"] = result["swt_mean_joint_support"]
    result["hcpf_mean_voxel_weighted"] = result["hcpf_mean_joint_support"]
    for key in ("phit", "swt", "phit_joint", "swt_joint", "hcpf"):
        result[f"{key}_mean_provisional_volume_weighted"] = (
            weighted_sums[key] / weighted_totals[key] if weighted_totals[key] > 0 else None)
        for q, label in ((0.10, "p10"), (0.50, "p50"), (0.90, "p90")):
            hist = {"phit": h_phit_vw, "swt": h_swt_vw,
                    "phit_joint": h_phit_joint_vw, "swt_joint": h_swt_joint_vw,
                    "hcpf": h_hcpf_vw}[key]
            result[f"{key}_{label}_provisional_volume_weighted"] = (
                _hist_quantile(hist, edges, q) if hist.sum() > 0 else None)
    return result



def _body_column_geometry(labels: np.ndarray, gid: int, geometry: dict) -> pd.DataFrame:
    """Group body voxels by trace in bounded slabs; never rescan all voxels per column."""
    il_min, il_step = int(geometry["il_min"]), int(geometry["il_step"])
    xl_min, xl_step = int(geometry["xl_min"]), int(geometry["xl_step"])
    t0, dt = float(geometry["t0_ms"]), float(geometry["dt_ms"])
    nxl = int(labels.shape[1])
    rows = []
    for i0 in range(0, labels.shape[0], 16):
        ii, jj, kk = np.nonzero(np.asarray(labels[i0:i0 + 16]) == gid)
        if not len(ii):
            continue
        ii = ii.astype(np.int64) + i0
        jj = jj.astype(np.int64)
        kk = kk.astype(np.int64)
        column_key = ii * nxl + jj
        order = np.lexsort((kk, column_key))
        keys, ks = column_key[order], kk[order]
        ii_sorted, jj_sorted = ii[order], jj[order]
        # A run starts at a new trace or after a missing time sample.
        run_start = np.r_[True, (keys[1:] != keys[:-1]) | (np.diff(ks) > 1)]
        run_idx = np.flatnonzero(run_start)
        run_end = np.r_[run_idx[1:] - 1, len(ks) - 1]
        run_keys = keys[run_idx]
        unique_keys, first_run, run_counts = np.unique(run_keys, return_index=True, return_counts=True)
        for col_pos, key in enumerate(unique_keys):
            r0 = int(first_run[col_pos]); r1 = r0 + int(run_counts[col_pos])
            a, b = int(run_idx[r0]), int(run_end[r1 - 1])
            run_pairs = [[int(ks[int(run_idx[r])]), int(ks[int(run_end[r])])] for r in range(r0, r1)]
            i, j = int(ii_sorted[a]), int(jj_sorted[a])
            count = int(b - a + 1)
            kmin, kmax = int(ks[a]), int(ks[b])
            rows.append({
                "volume_i": i, "volume_j": j,
                "il": il_min + i * il_step, "xl": xl_min + j * xl_step,
                "voxel_count": count, "n_vertical_runs": len(run_pairs),
                "time_runs_json": json.dumps(run_pairs, separators=(",", ":")),
                "top_time_k": kmin, "base_time_k": kmax,
                "twt_top_ms": t0 + kmin * dt, "twt_base_ms": t0 + kmax * dt,
                "gross_time_span_ms": (kmax - kmin + 1) * dt,
                "net_time_thickness_ms": count * dt,
            })
    return pd.DataFrame(rows)



def _make_depth_model(name: str, source_type: str, z: np.ndarray, twt: np.ndarray,
                      support: tuple[float, float], metadata: dict | None = None) -> dict | None:
    z = np.asarray(z, dtype=float)
    twt = np.asarray(twt, dtype=float)
    valid = np.isfinite(z) & np.isfinite(twt)
    z, twt = z[valid], twt[valid]
    if len(z) < 2:
        return None
    order = np.argsort(twt, kind="stable")
    twt, z = twt[order], z[order]
    unique_t, unique_idx = np.unique(twt, return_index=True)
    twt, z = unique_t, z[unique_idx]
    if len(twt) < 2 or np.any(np.diff(twt) <= 0) or np.any(np.diff(z) <= 0):
        return None
    return {"name": name, "source_type": source_type, "twt": twt, "z": z,
            "twt_min": float(twt.min()), "twt_max": float(twt.max()),
            "z_min": float(z.min()), "z_max": float(z.max()),
            "metadata": metadata or {}, "support": support}


def _depth_from_model(model: dict, twt_values) -> np.ndarray:
    """Invert a monotonic TWT(TVDSS) model without extrapolating beyond its controls."""
    values = np.asarray(twt_values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values) & (values >= model["twt_min"]) & (values <= model["twt_max"])
    if valid.any():
        out[valid] = np.interp(values[valid], model["twt"], model["z"])
    return out


def _load_stage01_depth_models(stage01_tables: Path) -> tuple[dict[str, dict], dict[str, str | None], str | None]:
    """Load available well-specific T-Z tables and every saved regional candidate."""
    models: dict[str, dict] = {}
    well_model: dict[str, str] = {}

    # Per-well T-Z tables are the strongest local controls where their source files remain accessible.
    qc_path = stage01_tables / "tdr_table_qc.csv"
    if qc_path.is_file():
        try:
            qc = pd.read_csv(qc_path)
            if {"well", "file"}.issubset(qc.columns):
                from geogg_fresh.io_wells import parse_tdr_table
                for _, row in qc.iterrows():
                    well = str(row.get("well", "")).strip().casefold()
                    source_path = Path(str(row.get("file", ""))).expanduser()
                    if not well or not source_path.is_file():
                        continue
                    qc_flags = str(row.get("flags", "")).casefold()
                    if "twt_not_strictly_increasing" in qc_flags or "tvdss_not_strictly_increasing" in qc_flags:
                        continue
                    try:
                        parsed = parse_tdr_table(source_path, well)
                        parsed_flags = " ".join(parsed.meta.get("flags", [])).casefold()
                        if ("twt_not_strictly_increasing" in parsed_flags
                                or "tvdss_not_strictly_increasing" in parsed_flags):
                            continue
                        frame = parsed.data
                        if not {"tvdss_m", "twt_ms"}.issubset(frame.columns):
                            continue
                        model = _make_depth_model(
                            f"tdr_table_{well}", "well_specific_tdr_table",
                            frame.tvdss_m.to_numpy(), frame.twt_ms.to_numpy(),
                            (float(frame.tvdss_m.min()), float(frame.tvdss_m.max())),
                            {"well": well, "source_path": str(source_path),
                             "flags": parsed.meta.get("flags", [])})
                        if model:
                            models[model["name"]] = model
                            well_model[well] = model["name"]
                    except (OSError, ValueError, RuntimeError, KeyError):
                        continue
        except (OSError, ValueError, pd.errors.ParserError):
            pass

    # Reconstruct all fitted regional candidates from Stage 01's retained coefficient table.
    coef_path = stage01_tables / "polynomial_coefficients_used.csv"
    selection_path = stage01_tables / "regional_model_selection.csv"
    selected_regional = None
    if selection_path.is_file():
        try:
            selection = pd.read_csv(selection_path)
            if {"model", "selected"}.issubset(selection.columns):
                selected = selection[selection.selected.astype(str).str.casefold().isin(["true", "1", "yes"])]
                if not selected.empty:
                    selected_regional = str(selected.iloc[0]["model"])
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    if coef_path.is_file():
        try:
            coefficients = pd.read_csv(coef_path)
            for _, row in coefficients.iterrows():
                name = str(row.get("model_name", "")).strip()
                source_type = str(row.get("source_type", "")).strip()
                if not name or source_type not in {"regional_polynomial", "regional_linear_velocity"}:
                    continue
                zmin = float(row.get("fit_domain_tvdss_min_m", 0.0))
                zmax = float(row.get("fit_domain_tvdss_max_m", np.nan))
                if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
                    continue
                zgrid = np.linspace(zmin, zmax, 20001)
                if source_type == "regional_linear_velocity":
                    v0, k = float(row["v0_ms"]), float(row["k_per_s"])
                    if v0 <= 0 or not np.isfinite(v0 + k):
                        continue
                    twt_grid = (2000.0 * zgrid / v0 if abs(k) < 1e-9
                                else 2000.0 / k * np.log1p(k * zgrid / v0))
                else:
                    scale = float(row.get("scale_m", np.nan))
                    if not np.isfinite(scale) or scale <= 0:
                        continue
                    coeffs = []
                    degree = int(row.get("degree", 0))
                    for power in range(1, degree + 1):
                        value = row.get(f"c{power}_normalised_ms")
                        if value is None or pd.isna(value):
                            coeffs = []
                            break
                        coeffs.append(float(value))
                    if not coeffs:
                        continue
                    u = zgrid / scale
                    twt_grid = sum(c * u ** power for power, c in enumerate(coeffs, start=1))
                model = _make_depth_model(name, source_type, zgrid, twt_grid,
                                          (zmin, zmax), {"fit_domain_tvdss_m": [zmin, zmax],
                                                         "selected_regional_model": name == selected_regional})
                if model:
                    models[name] = model
        except (OSError, ValueError, KeyError, TypeError, pd.errors.ParserError):
            pass
    # Stage 01 tie summary identifies the T-Z function actually assigned to each well.
    tie_path = stage01_tables / "tie_summary.csv"
    if tie_path.is_file():
        try:
            ties = pd.read_csv(tie_path)
            if {"well", "tz_source"}.issubset(ties.columns):
                for _, row in ties.iterrows():
                    well = str(row.get("well", "")).strip().casefold()
                    source = str(row.get("tz_source", "")).strip().casefold()
                    if not well or well in well_model:
                        continue
                    if "historic" in source:
                        well_model[well] = None
                    elif "regional" in source and selected_regional in models:
                        well_model[well] = selected_regional
                    elif "tdr" in source or "checkshot" in source:
                        # Keep a missing model explicit; do not silently substitute a regional curve.
                        well_model[well] = None
                    else:
                        well_model[well] = selected_regional if selected_regional in models else None
        except (OSError, ValueError, pd.errors.ParserError):
            pass
    return models, well_model, selected_regional


def _depth_comparison_rows(well: str, trajectory: pd.DataFrame,
                           models: dict[str, dict], well_model: dict[str, str],
                           selected_regional: str | None = None) -> list[dict]:
    rows = []
    candidates = []
    local_name = well_model.get(well)
    if local_name and local_name in models:
        candidates.append(local_name)
    if selected_regional and selected_regional in models and selected_regional not in candidates:
        candidates.append(selected_regional)
    candidates.extend(name for name, model in models.items()
                      if model["source_type"].startswith("regional_") and name not in candidates)
    if not candidates:
        return [{"well": well, "model_name": None, "status": "NO_USABLE_DEPTH_MODEL",
                 "comparison_basis": "Foundation Stage 01 registered trajectory samples"}]
    for name in candidates:
        model = models[name]
        predicted = _depth_from_model(model, trajectory.twt_ms.to_numpy(dtype=float))
        observed = trajectory.tvdss_m.to_numpy(dtype=float)
        valid = np.isfinite(predicted) & np.isfinite(observed)
        if valid.any():
            residual = predicted[valid] - observed[valid]
            rows.append({"well": well, "model_name": name, "source_type": model["source_type"],
                         "n_comparison_samples": int(valid.sum()),
                         "twt_overlap_min_ms": float(trajectory.twt_ms.to_numpy()[valid].min()),
                         "twt_overlap_max_ms": float(trajectory.twt_ms.to_numpy()[valid].max()),
                         "depth_residual_mean_m": float(np.mean(residual)),
                         "depth_residual_mae_m": float(np.mean(np.abs(residual))),
                         "depth_residual_rmse_m": float(np.sqrt(np.mean(residual ** 2))),
                         "depth_residual_max_abs_m": float(np.max(np.abs(residual))),
                         "is_primary_model_for_well": bool(name == local_name or
                             (well not in well_model and name == selected_regional)),
                         "status": "COMPARABLE_WITHIN_MODEL_SUPPORT",
                         "comparison_basis": "Foundation Stage 01 registered trajectory samples; not independent validation"})
        else:
            rows.append({"well": well, "model_name": name, "source_type": model["source_type"],
                         "n_comparison_samples": 0,
                         "is_primary_model_for_well": bool(name == local_name or
                             (well not in well_model and name == selected_regional)),
                         "status": "NO_TWT_OVERLAP_WITH_MODEL_SUPPORT",
                         "comparison_basis": "Foundation Stage 01 registered trajectory samples; not independent validation"})
    return rows


def _apply_column_depths(columns: pd.DataFrame, well: str, geobody_id: int,
                         models: dict[str, dict], well_model: dict[str, str | None],
                         selected_regional: str | None, geometry: dict) -> pd.DataFrame:
    if columns.empty:
        return columns
    local_name = well_model.get(well)
    primary_name = local_name if local_name in models else (selected_regional if selected_regional in models else None)
    output = columns.copy()
    output.insert(0, "geobody_id", int(geobody_id))
    output.insert(0, "well", well)
    t0, dt = float(geometry["t0_ms"]), float(geometry["dt_ms"])
    top_t = output.twt_top_ms.to_numpy(dtype=float)
    base_t = output.twt_base_ms.to_numpy(dtype=float)
    parsed_runs = [json.loads(v) if isinstance(v, str) else [] for v in output.time_runs_json]
    primary_top = np.full(len(output), np.nan); primary_base = np.full(len(output), np.nan)
    primary_net = np.full(len(output), np.nan)
    candidate_top, candidate_base, candidate_net = [], [], []
    for name, model in models.items():
        if model["source_type"] == "well_specific_tdr_table" and model["metadata"].get("well") != well:
            continue
        ztop = _depth_from_model(model, top_t)
        zbase = _depth_from_model(model, base_t)
        valid = np.isfinite(ztop) & np.isfinite(zbase)
        net = np.full(len(output), np.nan)
        # Convert every run edge in a single interpolation batch per model.
        run_counts = np.fromiter((len(v) for v in parsed_runs), dtype=np.int64, count=len(parsed_runs))
        if run_counts.sum():
            owners = np.repeat(np.arange(len(output), dtype=np.int64), run_counts)
            flat_edges = np.fromiter(
                (t0 + (edge_k + offset) * dt
                 for runs in parsed_runs for k0, k1 in runs
                 for edge_k, offset in ((k0, -0.5), (k1, 0.5))),
                dtype=float, count=int(run_counts.sum() * 2))
            edge_depths = _depth_from_model(model, flat_edges).reshape(-1, 2)
            edge_valid = np.isfinite(edge_depths).all(axis=1) & (edge_depths[:, 1] >= edge_depths[:, 0])
            sums = np.zeros(len(output), dtype=float)
            valid_runs = np.ones(len(output), dtype=bool)
            np.add.at(sums, owners[edge_valid], edge_depths[edge_valid, 1] - edge_depths[edge_valid, 0])
            np.logical_and.at(valid_runs, owners, edge_valid)
            valid_net = valid & valid_runs & (run_counts > 0)
            net[valid_net] = sums[valid_net]
        if name == primary_name:
            primary_top[valid] = ztop[valid]
            primary_base[valid] = zbase[valid]
            primary_net[:] = net
        candidate_top.append(ztop)
        candidate_base.append(zbase)
        candidate_net.append(net)
    output["primary_depth_model"] = primary_name
    output["tvdss_top_primary_m"] = primary_top
    output["tvdss_base_primary_m"] = primary_base
    output["gross_vertical_extent_primary_m"] = primary_base - primary_top
    output["net_vertical_thickness_primary_m"] = primary_net
    output["n_models_covering_column"] = (np.sum(np.vstack([np.isfinite(v) for v in candidate_top]), axis=0)
                                           if candidate_top else np.zeros(len(output), dtype=int))
    for prefix, values in (("top", candidate_top), ("base", candidate_base), ("net", candidate_net)):
        if values:
            matrix = np.vstack(values)
            finite = np.isfinite(matrix)
            mins = np.min(np.where(finite, matrix, np.inf), axis=0)
            maxs = np.max(np.where(finite, matrix, -np.inf), axis=0)
            no_values = ~finite.any(axis=0)
            mins[no_values] = np.nan; maxs[no_values] = np.nan
        else:
            mins = maxs = np.full(len(output), np.nan)
        min_name = f"tvdss_{prefix}_model_min_m" if prefix != "net" else "net_vertical_thickness_model_min_m"
        max_name = f"tvdss_{prefix}_model_max_m" if prefix != "net" else "net_vertical_thickness_model_max_m"
        spread_name = f"tvdss_{prefix}_model_spread_m" if prefix != "net" else "net_vertical_thickness_model_spread_m"
        output[min_name], output[max_name], output[spread_name] = mins, maxs, maxs - mins
    output["depth_status"] = np.where(np.isfinite(output["gross_vertical_extent_primary_m"]),
        "PROVISIONAL_CONVERSION_WITHIN_MODEL_SUPPORT", "PRIMARY_MODEL_UNAVAILABLE_OR_OUTSIDE_SUPPORT")
    output["depth_uncertainty_note"] = "Inter-model spread only; not a statistical confidence interval or calibrated uncertainty"
    return output


def _atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def run(config: dict, result_path: Path) -> dict:
    if not isinstance(config, dict):
        raise RuntimeError("well_geobody_analysis configuration must be a mapping")
    current = result_path.parent
    out_stage = current / "stages" / "13_well_geobody_analysis"
    tables = out_stage / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    progress_path = out_stage / "progress.json"
    def update_progress(phase: str, **details) -> None:
        payload = {"schema_version": 1, "stage": "13_well_geobody_analysis", "phase": phase,
                   "updated_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                   **details}
        _atomic_json(progress_path, payload)
        print(f"[Stage 13] {phase}: {details}", flush=True)
    update_progress("INITIALISING")
    jobs_root = next((p for p in current.parents if p.name == "processing_jobs"), None)
    if jobs_root is None:
        raise RuntimeError(f"Cannot locate definitive processing_jobs directory from {result_path}")
    jobs_root = jobs_root.resolve()
    ids = {key: str(config.get(key) or "").strip() for key in
           ("foundation_job_id", "inference_job_id", "source_volume_job_id", "source_geobody_job_id")}
    missing_ids = [key for key, value in ids.items() if not value]
    if missing_ids:
        raise RuntimeError("Required upstream processing job IDs are missing: " + ", ".join(missing_ids))
    labels = {"foundation_job_id": "Foundation", "inference_job_id": "Inference",
              "source_volume_job_id": "Volume-products", "source_geobody_job_id": "Geobody"}
    update_progress("VALIDATING_UPSTREAM", source_jobs=ids)
    jobs = {key: _safe_job(jobs_root, value, labels[key]) for key, value in ids.items()}

    # Stage 01 is the authoritative registered trajectory source: minimum-curvature
    # positions and TWT from the Foundation job's own configured T-Z relationship.
    trajectory_path = jobs["foundation_job_id"] / "stages/01_well_ties/tables/tz_samples_all_wells.csv"
    if not trajectory_path.is_file():
        raise RuntimeError(f"Foundation Stage 01 trajectory samples are missing: {trajectory_path}")
    trajectory_all = pd.read_csv(trajectory_path)
    required_trajectory = {"well", "md_m", "tvdss_m", "twt_ms", "il_nearest", "xl_nearest"}
    missing = required_trajectory - set(trajectory_all.columns)
    if missing:
        raise RuntimeError(f"Stage 01 trajectory table is missing columns: {sorted(missing)}")
    trajectory_all = trajectory_all.dropna(subset=["well"]).copy()
    trajectory_all["_well_key"] = trajectory_all.well.astype(str).str.strip().str.casefold()
    available_wells = sorted(trajectory_all.loc[trajectory_all._well_key != "", "_well_key"].unique().tolist())
    if not available_wells:
        raise RuntimeError("Stage 01 trajectory table contains no identifiable wells")
    stage01_tables = trajectory_path.parent
    depth_models, well_depth_models, selected_regional_model = _load_stage01_depth_models(stage01_tables)
    complete_trajectory = trajectory_all.dropna(subset=["md_m", "tvdss_m", "twt_ms", "il_nearest", "xl_nearest"]).copy()
    well_setting = config.get("well_id")
    if not isinstance(well_setting, (list, tuple)):
        raise RuntimeError(
            "well_geobody_analysis.well_id must be an explicit YAML list of well IDs; "
            "implicit 'all' and single-string selectors are not supported"
        )
    requested_wells = [str(value).strip().casefold() for value in well_setting if str(value).strip()]
    if not requested_wells:
        raise RuntimeError("well_geobody_analysis.well_id list must contain at least one well")
    if len(requested_wells) != len(set(requested_wells)):
        raise RuntimeError("well_geobody_analysis.well_id contains duplicate well IDs")
    selection_mode = "explicit_list"
    if not requested_wells:
        raise RuntimeError("No wells have complete registered Stage 01 trajectory samples")
    unknown_wells = sorted(set(requested_wells) - set(available_wells))
    if unknown_wells:
        raise RuntimeError(f"Requested wells are not present in Stage 01: {unknown_wells}; available wells: {available_wells}")
    complete_wells = set(complete_trajectory._well_key.unique().tolist())
    wells_without_complete_trajectory = sorted(set(requested_wells) - complete_wells)
    selected_wells = [well for well in requested_wells if well in complete_wells]
    if not selected_wells:
        raise RuntimeError(f"None of the requested wells have complete trajectory samples: {requested_wells}")

    # Validate cross-stage provenance before opening the label volume.
    volume_stage_id = str(config.get("source_volume_stage_id", "11_volume_export")).strip()
    volume_stage = jobs["source_volume_job_id"] / "stages" / volume_stage_id
    volume_manifest = _read_json(volume_stage / "native_volume_export_manifest.json", "Stage 11 native volume manifest")
    recorded_inference = str(volume_manifest.get("source_inference_job_id") or
                             (volume_manifest.get("provenance") or {}).get("source_inference_job_id") or "")
    if recorded_inference != ids["inference_job_id"]:
        raise RuntimeError(f"Volume-products job links inference {recorded_inference!r}, not configured {ids['inference_job_id']!r}")

    geobody_stage_id = str(config.get("source_geobody_stage_id", "12_geobody")).strip()
    geobody_stage = jobs["source_geobody_job_id"] / "stages" / geobody_stage_id
    geobody_manifest = _read_json(geobody_stage / "geobody_manifest.json", "Stage 12 geobody manifest")
    if str(geobody_manifest.get("status", "")).lower() != "completed":
        raise RuntimeError(f"Source Stage 12 geobody job is not completed: {geobody_manifest.get('status', 'missing')}")
    source = geobody_manifest.get("source") if isinstance(geobody_manifest.get("source"), dict) else {}
    expected_links = {"foundation_job_id": ids["foundation_job_id"],
                      "inference_job_id": ids["inference_job_id"],
                      "volume_job_id": ids["source_volume_job_id"]}
    for field, expected in expected_links.items():
        if str(source.get(field) or "") != expected:
            raise RuntimeError(f"Geobody manifest {field}={source.get(field)!r}; configured source is {expected!r}")
    geometry = geobody_manifest.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("axis_order") != ["inline_bin", "crossline_bin", "time_sample"]:
        # Stage 12 schema stores axis order as list; reject ambiguous axis ordering.
        raise RuntimeError("Geobody manifest has missing or unsupported [inline,crossline,time] geometry")
    shape = tuple(int(v) for v in geometry.get("shape", []))
    if len(shape) != 3:
        raise RuntimeError(f"Invalid geobody volume shape: {shape}")
    volume_geometry = volume_manifest.get("geometry") if isinstance(volume_manifest.get("geometry"), dict) else {}
    for field in ("shape", "il_min", "il_step", "xl_min", "xl_step", "t0_ms", "dt_ms"):
        if field in volume_geometry and geometry.get(field) != volume_geometry.get(field):
            raise RuntimeError(f"Geobody geometry {field}={geometry.get(field)!r} does not match Stage 11 {field}={volume_geometry.get(field)!r}")
    products = geobody_manifest.get("products") if isinstance(geobody_manifest.get("products"), list) else []
    label_row = _find_product(products, "geobody_labels", "npy")
    catalog_row = _find_product(products, "geobody_catalog", "csv")
    run_length_row = _find_product(products, "geobody_run_length_index", "csv")
    spatial_reconciliation_row = _find_product(products, "geobody_spatial_export_reconciliation", "csv")
    if not label_row or not catalog_row:
        raise RuntimeError("Geobody manifest must register geobody_labels NPY and geobody_catalog CSV")
    if not run_length_row or not spatial_reconciliation_row:
        raise RuntimeError("Stage 12 manifest must register its run-length index and spatial reconciliation exports")
    def _geobody_product_path(row: dict) -> Path:
        rel = str(row.get("storage_root_relative_path") or row.get("path") or "").strip()
        if not rel:
            raise RuntimeError(f"Geobody product {row.get('product')} has no registered path")
        candidate = Path(rel)
        path = (jobs["source_geobody_job_id"] / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        try:
            path.relative_to(jobs["source_geobody_job_id"].resolve())
        except ValueError as exc:
            raise RuntimeError(f"Geobody product path escapes its source job: {rel}") from exc
        return path
    labels_path, catalog_path = _geobody_product_path(label_row), _geobody_product_path(catalog_row)
    run_length_source_path = _geobody_product_path(run_length_row)
    spatial_reconciliation_source_path = _geobody_product_path(spatial_reconciliation_row)
    for path, row in ((labels_path, label_row), (catalog_path, catalog_row),
                      (run_length_source_path, run_length_row),
                      (spatial_reconciliation_source_path, spatial_reconciliation_row)):
        if not path.is_file():
            raise RuntimeError(f"Registered geobody product is missing: {path}")
        expected_hash = str(row.get("sha256") or "").lower()
        if expected_hash and _sha256(path) != expected_hash:
            raise RuntimeError(f"Registered geobody product checksum mismatch: {path}")
    update_progress("LOADING_VOLUME", label_volume=str(labels_path), shape=shape)
    labels_array = np.load(labels_path, mmap_mode="r", allow_pickle=False)
    if tuple(labels_array.shape) != shape or labels_array.dtype.kind not in "iu":
        raise RuntimeError(f"Geobody label volume shape/dtype mismatch: {labels_array.shape}/{labels_array.dtype}, expected {shape}")
    catalog = pd.read_csv(catalog_path)
    if not {"geobody_id", "voxel_count"}.issubset(catalog.columns):
        raise RuntimeError("Geobody catalogue must contain geobody_id and voxel_count")

    # Load Stage 11 canonical PHIT/SWT arrays when registered. Missing properties
    # are conditional outputs, not a reason to lose the geometric intersection.
    volume_products = volume_manifest.get("products") if isinstance(volume_manifest.get("products"), list) else []
    prop_arrays = {}
    for product in ("prediction_phit", "prediction_swt"):
        row = _find_product(volume_products, product, "npy")
        if row:
            p = _product_path(jobs_root, row)
            if not p.is_file():
                raise RuntimeError(f"Registered Stage 11 property array is missing: {p}")
            expected_hash = str(row.get("sha256") or "").lower()
            actual_hash = _sha256(p)
            if not expected_hash or actual_hash != expected_hash:
                raise RuntimeError(f"Registered Stage 11 property array checksum is missing or mismatched: {p}")
            arr = np.load(p, mmap_mode="r", allow_pickle=False)
            if tuple(arr.shape) != shape:
                raise RuntimeError(f"Stage 11 property array shape mismatch for {product}: {arr.shape}, expected {shape}")
            prop_arrays[product] = arr
    phit = prop_arrays.get("prediction_phit")
    swt = prop_arrays.get("prediction_swt")
    inference_primary_models_path = jobs["inference_job_id"] / "stages/09_inference/tables/primary_models.csv"
    property_model_identity = []
    if inference_primary_models_path.is_file():
        try:
            primary_models = pd.read_csv(inference_primary_models_path)
            for task in ("phit", "swt"):
                task_rows = primary_models[primary_models.get("task", pd.Series(dtype=str)).astype(str).str.casefold() == task]
                if not task_rows.empty:
                    record = task_rows.iloc[0].to_dict()
                    property_model_identity.append({key: (None if pd.isna(value) else value)
                        for key, value in record.items() if key in {"task", "input", "model", "model_name", "status"}})
        except (OSError, ValueError, pd.errors.ParserError):
            property_model_identity = []
    property_model_identity_status = "PRIMARY_MODEL_IDENTITIES_READ" if property_model_identity else "MODEL_IDENTITY_NOT_SERIALIZED_OR_UNAVAILABLE"
    property_config = config.get("specialist_phit_swt", {})
    if not isinstance(property_config, dict):
        raise RuntimeError("specialist_phit_swt configuration must be a mapping")
    min_joint_support_fraction = float(property_config.get("min_joint_support_fraction", 0.5))
    min_joint_voxels = int(property_config.get("min_joint_voxels", 100))
    histogram_bins = int(property_config.get("histogram_bins", 1000))
    if not 0.0 <= min_joint_support_fraction <= 1.0 or min_joint_voxels < 1 or histogram_bins < 10:
        raise RuntimeError("Invalid specialist PHIT/SWT coverage-gate or histogram configuration")

    # General all-body spatial products are generated and owned by Stage 12.
    # Stage 13 validates and references those immutable upstream products below.
    il_min, il_step = int(geometry["il_min"]), int(geometry["il_step"])
    xl_min, xl_step = int(geometry["xl_min"]), int(geometry["xl_step"])
    t0, dt = float(geometry["t0_ms"]), float(geometry["dt_ms"])
    if il_step <= 0 or xl_step <= 0 or dt <= 0:
        raise RuntimeError("Geobody geometry increments must be positive")
    sampled_frames = []
    interval_frames = []
    intersection_frames = []
    selected_rows = []
    column_frames = []
    depth_comparison_frames = []
    per_well = []
    geometry_cache: dict[int, dict] = {}
    property_cache: dict[tuple[int, str | None], dict] = {}
    columns_cache: dict[int, pd.DataFrame] = {}
    update_progress("ANALYSING_WELLS", total_wells=len(selected_wells), completed_wells=0,
                    selected_wells=selected_wells)
    for well_index, well in enumerate(selected_wells, start=1):
        update_progress("ANALYSING_WELL", well=well, well_index=well_index,
                        total_wells=len(selected_wells), completed_wells=well_index - 1)
        trajectory = complete_trajectory[complete_trajectory._well_key == well].copy()
        trajectory = trajectory.sort_values("md_m", kind="stable")
        depth_comparison_frames.append(pd.DataFrame(
            _depth_comparison_rows(well, trajectory, depth_models, well_depth_models, selected_regional_model)))
        rows = []
        for _, row in trajectory.iterrows():
            il, xl, twt = float(row.il_nearest), float(row.xl_nearest), float(row.twt_ms)
            i = int(round((il - il_min) / il_step))
            j = int(round((xl - xl_min) / xl_step))
            k = int(round((twt - t0) / dt))
            inside = 0 <= i < shape[0] and 0 <= j < shape[1] and 0 <= k < shape[2]
            gid = int(labels_array[i, j, k]) if inside else 0
            rows.append({"well": well, "sample_index": int(len(rows)), "md_m": float(row.md_m),
                         "tvdss_m": float(row.tvdss_m), "twt_ms": twt, "il": il, "xl": xl,
                         "volume_i": i, "volume_j": j, "time_k": k, "geobody_id": gid,
                         "inside_volume": bool(inside), "has_geobody": bool(gid > 0),
                         "trajectory_tz_source": str(row.get("tz_source", "")),
                         "trajectory_tie_status": str(row.get("tie_status", "")),
                         "tz_strong_extrapolation": bool(row.get("tz_strong_extrapolation", False))})
        sampled, intervals = _intervals(pd.DataFrame(rows), well)
        sampled_frames.append(sampled)
        interval_frames.append(intervals)
        crossed = sorted(set(int(v) for v in sampled.loc[sampled.geobody_id > 0, "geobody_id"].tolist()))
        intersection_rows = []
        for gid in crossed:
            body_intervals = intervals[intervals.geobody_id == gid]
            cat_rows = catalog[catalog.geobody_id.astype(int) == gid]
            cat_row = cat_rows.iloc[0].to_dict() if len(cat_rows) else {}
            intersection_rows.append({"well": well, "geobody_id": gid,
                "n_intervals": int(len(body_intervals)), "n_well_path_samples": int((sampled.geobody_id == gid).sum()),
                "md_top_m": float(body_intervals.md_top_m.min()), "md_base_m": float(body_intervals.md_base_m.max()),
                "twt_top_on_path_ms": float(body_intervals.twt_top_ms.min()),
                "twt_base_on_path_ms": float(body_intervals.twt_base_ms.max()),
                "catalogue_voxel_count": int(cat_row.get("voxel_count", 0)),
                "stable_id": cat_row.get("stable_id", ""),
                "closure_status": cat_row.get("closure_status", ""),
                "boundary_flags": cat_row.get("boundary_flags", "")})
        intersections = pd.DataFrame(intersection_rows)
        if not intersections.empty:
            intersections = intersections.sort_values(["catalogue_voxel_count", "geobody_id"], ascending=[False, True]).reset_index(drop=True)
            intersections["selected_largest_by_voxel_count"] = intersections.index == 0
            intersection_frames.append(intersections)
            selected_id = int(intersections.iloc[0].geobody_id)
            if selected_id not in geometry_cache:
                update_progress("COMPUTING_BODY_GEOMETRY", well=well, geobody_id=selected_id)
                geometry_cache[selected_id] = _body_geometry(labels_array, selected_id, geometry)
                columns_cache[selected_id] = _body_column_geometry(labels_array, selected_id, geometry)
            property_model_name = (well_depth_models[well] if well in well_depth_models else selected_regional_model)
            property_key = (selected_id, property_model_name)
            if property_key not in property_cache:
                property_model = depth_models.get(property_model_name) if property_model_name else None
                property_cache[property_key] = _property_stats(
                    labels_array, selected_id, phit, swt, min_joint_support_fraction,
                    min_joint_voxels, histogram_bins, property_model, geometry)
            selected_properties = property_cache[property_key]
            selected = dict(geometry_cache[selected_id])
            cat_match = catalog[catalog.geobody_id.astype(int) == selected_id]
            selected.update({"well": well, "selected_by": "largest_catalogue_voxel_count_among_intersected_bodies",
                             "catalogue_voxel_count": int(cat_match.iloc[0].voxel_count) if len(cat_match) else None,
                             **selected_properties})
            selected["catalogue_voxel_count_reconciles"] = selected.get("voxel_count_recounted") == selected.get("catalogue_voxel_count")
            column_geometry = columns_cache[selected_id]
            column_depths = _apply_column_depths(column_geometry, well, selected_id, depth_models,
                                                 well_depth_models, selected_regional_model, geometry)
            if not column_depths.empty:
                column_frames.append(column_depths)
                valid_top = column_depths.tvdss_top_primary_m.dropna()
                valid_base = column_depths.tvdss_base_primary_m.dropna()
                selected.update({
                    "primary_depth_model": next((v for v in column_depths.primary_depth_model if isinstance(v, str)), None),
                    "depth_status": ("PROVISIONAL_CONVERSION_WITHIN_MODEL_SUPPORT"
                                     if len(valid_top) == len(column_depths) and len(valid_base) == len(column_depths)
                                     else "PARTIAL_OR_UNAVAILABLE_MODEL_SUPPORT"),
                    "tvdss_top_primary_m": float(valid_top.min()) if len(valid_top) else None,
                    "tvdss_base_primary_m": float(valid_base.max()) if len(valid_base) else None,
                    "gross_vertical_extent_primary_m": float(valid_base.max() - valid_top.min())
                                                       if len(valid_top) and len(valid_base) else None,
                    "max_top_model_spread_m": float(column_depths.tvdss_top_model_spread_m.max())
                                              if column_depths.tvdss_top_model_spread_m.notna().any() else None,
                    "max_base_model_spread_m": float(column_depths.tvdss_base_model_spread_m.max())
                                               if column_depths.tvdss_base_model_spread_m.notna().any() else None,
                    "max_net_thickness_model_spread_m": float(column_depths.net_vertical_thickness_model_spread_m.max())
                                                        if column_depths.net_vertical_thickness_model_spread_m.notna().any() else None,
                    "columns_with_primary_depth": int(column_depths.tvdss_top_primary_m.notna().sum()),
                    "columns_total": int(len(column_depths)),
                })
            else:
                selected["depth_status"] = "NO_SELECTED_BODY_COLUMNS"
            selected["depth_uncertainty_interpretation"] = (
                "Inter-model spread only; not a statistical confidence interval")
            selected["status"] = "COMPLETED"
            selected_rows.append(selected)
            well_status = "COMPLETED"
        else:
            selected_id = None
            selected_rows.append({"well": well, "status": "NO_INTERSECTED_GEOBODY"})
            well_status = "COMPLETED_NO_INTERSECTION"
        per_well.append({"well": well, "status": well_status,
            "counts": {"trajectory_samples": int(len(sampled)),
                       "samples_inside_geobody_volume": int(sampled.inside_volume.sum()),
                       "samples_intersecting_geobodies": int(sampled.has_geobody.sum()),
                       "distinct_intersected_geobodies": len(crossed),
                       "path_intervals": int(len(intervals))},
            "intersected_geobody_ids": crossed,
            "selected_geobody_id": selected_id})
        # Durable per-well checkpoint: a later failure cannot erase completed wells.
        _atomic_csv(tables / "well_geobody_path_samples.partial.csv", pd.concat(sampled_frames, ignore_index=True))
        _atomic_csv(tables / "well_geobody_path_intervals.partial.csv", pd.concat(interval_frames, ignore_index=True))
        _atomic_csv(tables / "well_geobody_intersections.partial.csv",
                    pd.concat(intersection_frames, ignore_index=True) if intersection_frames else pd.DataFrame())
        _atomic_csv(tables / "selected_well_geobody_analysis.partial.csv", pd.DataFrame(selected_rows))
        _atomic_csv(tables / "selected_body_columns.partial.csv",
                    pd.concat(column_frames, ignore_index=True) if column_frames else pd.DataFrame())
        _atomic_json(out_stage / "checkpoint.json", {"status": "IN_PROGRESS", "completed_wells": well_index,
            "total_wells": len(selected_wells), "last_completed_well": well, "per_well": per_well})
        update_progress("WELL_COMPLETED", well=well, well_index=well_index,
                        total_wells=len(selected_wells), completed_wells=well_index)

    sampled = pd.concat(sampled_frames, ignore_index=True) if sampled_frames else pd.DataFrame()
    intervals = pd.concat(interval_frames, ignore_index=True) if interval_frames else pd.DataFrame()
    intersections = pd.concat(intersection_frames, ignore_index=True) if intersection_frames else pd.DataFrame(
        columns=["well", "geobody_id", "n_intervals", "n_well_path_samples", "md_top_m", "md_base_m",
                 "twt_top_on_path_ms", "twt_base_on_path_ms", "catalogue_voxel_count", "stable_id",
                 "closure_status", "boundary_flags", "selected_largest_by_voxel_count"])
    selected_table = pd.DataFrame(selected_rows)
    wells_with_intersections = sum(item["status"] == "COMPLETED" for item in per_well)
    wells_without_intersections = sum(item["status"] == "COMPLETED_NO_INTERSECTION" for item in per_well)
    total_counts = {key: int(sum(item["counts"][key] for item in per_well)) for key in
                    ("trajectory_samples", "samples_inside_geobody_volume", "samples_intersecting_geobodies",
                     "distinct_intersected_geobodies", "path_intervals")}
    if wells_without_complete_trajectory:
        overall_status = "COMPLETED_WITH_WELL_WARNINGS"
    elif wells_with_intersections == 0:
        overall_status = "COMPLETED_NO_INTERSECTION"
    else:
        overall_status = "COMPLETED"
    update_progress("WRITING_OUTPUTS", completed_wells=len(per_well), total_wells=len(selected_wells))
    sampled_path = tables / "well_geobody_path_samples.csv"
    interval_path = tables / "well_geobody_path_intervals.csv"
    intersections_path = tables / "well_geobody_intersections.csv"
    selected_path = tables / "selected_well_geobody_analysis.csv"
    columns_path = tables / "selected_body_columns.csv"
    depth_comparison_path = tables / "depth_model_comparison.csv"
    property_summary_path = tables / "specialist_phit_swt_summary.csv"
    _atomic_csv(sampled_path, sampled)
    _atomic_csv(interval_path, intervals)
    _atomic_csv(intersections_path, intersections)
    _atomic_csv(selected_path, selected_table)
    column_table = pd.concat(column_frames, ignore_index=True) if column_frames else pd.DataFrame(
        columns=["well", "geobody_id", "il", "xl", "twt_top_ms", "twt_base_ms",
                 "tvdss_top_primary_m", "tvdss_base_primary_m", "vertical_thickness_primary_m", "depth_status"])
    depth_comparison = pd.concat(depth_comparison_frames, ignore_index=True) if depth_comparison_frames else pd.DataFrame()
    _atomic_csv(columns_path, column_table)
    _atomic_csv(depth_comparison_path, depth_comparison)
    property_columns = ["well", "geobody_id"] + sorted({key for row in selected_rows for key in row
        if key.startswith(("phit_", "swt_", "joint_", "hcpf_", "property_", "provisional_volume_"))})
    property_table = pd.DataFrame([{key: row.get(key) for key in property_columns}
                                   for row in selected_rows if row.get("geobody_id") is not None])
    _atomic_csv(property_summary_path, property_table)

    # Spatial reconciliation is performed once by Stage 12 against its own
    # canonical label volume and catalogue. Preserve the upstream result here.
    spatial_export_summary = geobody_manifest.get("exports", {}).get("spatial", {})
    spatial_reconciliation = spatial_export_summary.get("catalogue_reconciliation", {})
    if not spatial_reconciliation.get("all_counts_reconcile", False):
        overall_status = "COMPLETED_WITH_UPSTREAM_SPATIAL_EXPORT_WARNINGS"

    summary = {"schema_version": 4, "stage": "13_well_geobody_analysis", "status": overall_status,
        "interpretation_status": "EXPLORATORY_GEOMETRIC_CANDIDATE_ASSESSMENT",
        "source": {**ids, "source_volume_stage_id": volume_stage_id,
                   "source_geobody_stage_id": geobody_stage_id,
                   "trajectory_source": "Foundation Stage 01 tz_samples_all_wells.csv",
                   "trajectory_sha256": _sha256(trajectory_path),
                   "geobody_labels_sha256": _sha256(labels_path),
                   "geobody_catalogue_sha256": _sha256(catalog_path),
                   "geobody_run_length_index_sha256": str(run_length_row.get("sha256") or ""),
                   "geobody_spatial_reconciliation_sha256": str(spatial_reconciliation_row.get("sha256") or "")},
        "well_selection": {"mode": selection_mode, "requested": requested_wells,
                           "analysed": selected_wells,
                           "without_complete_trajectory": wells_without_complete_trajectory,
                           "available_in_stage01": available_wells},
        "well_ids": selected_wells,
        "counts": {**total_counts, "wells_analysed": len(per_well),
                   "wells_with_intersections": int(wells_with_intersections),
                   "wells_without_intersections": int(wells_without_intersections),
                   "wells_without_complete_trajectory": len(wells_without_complete_trajectory)},
        "per_well": per_well,
        "spatial_exports": {
            "owner_stage": "12_geobody",
            "source_job_id": ids["source_geobody_job_id"],
            "source_manifest": str((geobody_stage / "geobody_manifest.json").relative_to(jobs_root)),
            "run_length_index": next(({"path": row.get("path"), "sha256": row.get("sha256")}
                for row in products if row.get("product") == "geobody_run_length_index"), None),
            "catalogue_reconciliation": spatial_reconciliation,
            "optional_voxel_xyz": spatial_export_summary.get("optional_voxel_xyz", {"enabled": False}),
            "xy_coordinate_status": spatial_export_summary.get("xy_coordinate_status", "UNAVAILABLE_NOT_RECORDED"),
            "depth_conversion": "Well-specific TVDSS conversion is computed in Stage 13; Stage 12 exports TWT only.",
        },
        "specialist_phit_swt_analysis": {
            "status": "COMPUTED_WHERE_SOURCE_ARRAYS_AVAILABLE" if phit is not None and swt is not None else "SOURCE_ARRAYS_UNAVAILABLE",
            "source": "Stage 11 canonical frozen prediction_phit.npy and prediction_swt.npy",
            "source_volume_job_id": ids["source_volume_job_id"],
            "source_inference_job_id": ids["inference_job_id"],
            "model_identity_status": property_model_identity_status,
            "primary_model_identities": property_model_identity,
            "primary_models_csv_sha256": _sha256(inference_primary_models_path) if inference_primary_models_path.is_file() else None,
            "prediction_arrays": {name: {"sha256": str(next((row.get("sha256") for row in volume_products if row.get("product") == name), "") or ""),
                                          "shape": list(prop_arrays[name].shape) if name in prop_arrays else None,
                                          "dtype": str(prop_arrays[name].dtype) if name in prop_arrays else None}
                                  for name in ("prediction_phit", "prediction_swt")},
            "coverage_gate": {"min_joint_support_fraction": min_joint_support_fraction,
                              "min_joint_voxels": min_joint_voxels,
                              "histogram_bins": histogram_bins},
            "statistics": ["mean, P10, P50, P90, min and max for PHIT and SWT",
                           "joint-support PHIT and SWT statistics, including P10/P50/P90",
                           "PHIT × (1 − SWT) screening statistic"],
            "weighting": {"voxel_weighted": "Each valid voxel contributes one count; regular-grid count weighting.",
                          "provisional_volume_weighted": "Vertical thickness-weighted proxy from the selected monotonic T-Z model; assumes constant lateral cell area, excludes unsupported samples, and is not calibrated m3."},
            "interpretation": "Exploratory model-prediction screening only; PHIT × (1 − SWT) is not HCPV or net-pay.",
        },
        "depth_characterisation": {
            "available_models": [{"model_name": name, "source_type": model["source_type"],
                                  "twt_support_ms": [model["twt_min"], model["twt_max"]],
                                  "tvdss_support_m": [model["z_min"], model["z_max"]]}
                                 for name, model in depth_models.items()],
            "selected_regional_model": selected_regional_model,
            "well_specific_models": well_depth_models,
            "columns_exported": int(len(column_table)),
            "comparison_rows": int(len(depth_comparison)),
            "uncertainty_method": "range across available monotonic models covering both column endpoints",
            "limitations": ["Model spread is not a statistical confidence interval.",
                            "No depth extrapolation is performed outside model support.",
                            "Only accessible TDR tables and regional candidates with retained coefficients are reconstructed; historic polynomial models lacking complete serialized parameters are not reconstructed.",
                            "Trajectory comparison uses Stage 01 registered samples and is not independent validation."]
        },
        "selected_geobodies": [row for row in selected_rows if row.get("geobody_id") is not None],
        # Preserve the v141/v142 singular field for single-well callers.
        "selected_geobody": next((row for row in selected_rows if row.get("geobody_id") is not None), None)
                             if len(selected_wells) == 1 else None,
        "products": [{"product": "well_geobody_path_samples", "format": "csv", "path": str(sampled_path.relative_to(current)), "sha256": _sha256(sampled_path)},
                     {"product": "well_geobody_path_intervals", "format": "csv", "path": str(interval_path.relative_to(current)), "sha256": _sha256(interval_path)},
                     {"product": "well_geobody_intersections", "format": "csv", "path": str(intersections_path.relative_to(current)), "sha256": _sha256(intersections_path)},
                     {"product": "selected_well_geobody_analysis", "format": "csv", "path": str(selected_path.relative_to(current)), "sha256": _sha256(selected_path)},
                     {"product": "selected_body_columns", "format": "csv", "path": str(columns_path.relative_to(current)), "sha256": _sha256(columns_path)},
                     {"product": "depth_model_comparison", "format": "csv", "path": str(depth_comparison_path.relative_to(current)), "sha256": _sha256(depth_comparison_path)},
                     {"product": "specialist_phit_swt_summary", "format": "csv", "path": str(property_summary_path.relative_to(current)), "sha256": _sha256(property_summary_path)}],
        "limitations": ["Intersection is evaluated at registered Stage 01 trajectory sample positions, not by continuous wellbore interpolation.",
            "The Stage 01 well-specific T-Z trajectory is provisional and carries its own extrapolation/tie-status flags.",
            "Largest intersected body means largest by catalogue voxel count; it is not a geological quality ranking.",
            "PHIT/SWT summaries, when available, are frozen model predictions on body voxels, not direct measurements.",
            "Geobodies are exploratory uncalibrated candidates; no reservoir, hydrocarbon, fluid, net-pay, HCPV or reserves claim is made."]}
    _atomic_json(out_stage / "well_geobody_analysis_manifest.json", summary)
    # Successful finalisation: keep the terminal progress record, but remove all
    # transient per-well checkpoint CSVs and checkpoint metadata. Failed runs retain them.
    for partial in tables.glob("*.partial.csv"):
        partial.unlink(missing_ok=True)
    (out_stage / "checkpoint.json").unlink(missing_ok=True)
    update_progress("COMPLETED", status=overall_status, completed_wells=len(per_well),
                    total_wells=len(selected_wells), output_products=len(summary["products"]))
    return summary


def main() -> None:
    result_path = Path(os.environ["APGCE_RESULT_PATH"]).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = Path(os.environ.get("APGCE_GEOGG_WELL_GEOBODY_ANALYSIS_CONFIG") or
                       Path(__file__).resolve().parents[2] / "E_well_geobody_analysis.yaml").expanduser().resolve()
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Stage 13 requires PyYAML") from exc
    root = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    config = root.get("well_geobody_analysis", {})
    if not isinstance(config, dict):
        raise RuntimeError("well_geobody_analysis YAML section must be a mapping")
    # Persist a concise preflight before any large array work; the decorated worker
    # then validates both this report and the final summary.
    result = run(config, result_path)
    preflight = {"schema_version": 2, "stage": "13_well_geobody_analysis", "status": result["status"],
                 "source": result["source"], "well_selection": result["well_selection"],
                 "well_ids": result["well_ids"], "counts": result["counts"],
                 "checks": [{"check": "source_jobs_and_provenance", "status": "PASS"},
                            {"check": "trajectory_and_label_geometry", "status": "PASS"},
                            {"check": "geobody_catalogue_and_labels", "status": "PASS"}]}
    (result_path.parent / PREFLIGHT).write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    (result_path.parent / SUMMARY).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
