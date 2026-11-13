"""Pure functions for stage 03: move log labels onto the seismic time grid and cut patches.

Two problems are solved here, both without touching any file:

1. **Label binning.**  A log sample every 0.15 m becomes one label per seismic
   time bin.  Bins are centred EXACTLY on native seismic samples (t0 + k*dt), so
   the seismic value needs no interpolation (``desired_twt_ms == sampled_twt_ms``)
   and there is no half-sample ambiguity:

       block index  b = floor((twt - t0) / W + 0.5)        (half-open: [c - W/2, c + W/2))
       centre       c = t0 + b * W,   native sample index k = b * W / dt

   For W = 12 or 24 ms the blocks do NOT overlap (stride = W), so coarser
   datasets never count the same log interval twice.

   Inside a bin, every log sample is weighted by the TIME it represents
   (MD step x local dTWT/dMD).  Rules (all configurable):
   * coverage  = represented time / W; below ``min_time_coverage`` -> labels masked
   * continuous targets = weighted mean of VALID samples, only if their weight
     >= ``min_valid_fraction`` of the bin's logged time
   * classes   = majority by time, only if its share >= ``class_purity_min``
     (otherwise the bin is ``mixed`` -> -1).  All shares are kept as soft labels.

2. **Patch assembly.**  For a centre (IL, XL, k) the patch is the native-grid
   neighbourhood ``[channel, il_offset, xl_offset, time_offset]`` of shape
   ``[C, 5, 5, 21]`` by default.  Offsets are in GRID STEPS (an IL step of 2 line
   numbers = one bin), not metres.  Anything outside a volume or in an empty bin
   is NaN - never zero-padded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CHANNELS_DEFAULT = ["full", "near", "mid", "far", "ultrafar", "pimp", "simp", "vpvs"]
STACK_CHANNELS = ["full", "near", "mid", "far", "ultrafar"]
LITH_FRACTIONS = ["sand", "shale", "heterolithic", "tight_nonshale", "not_interpretable"]
FLUID_FRACTIONS = ["water_bearing", "hydrocarbon_potential", "unresolved", "conflict"]
CLASS_FRACTIONS = ["shale", "brine_sand", "oil_sand", "gas_sand", "undetermined_sand", "other_rock", "unlabelled"]
CLASS_IDS = {"shale": 0, "brine_sand": 1, "oil_sand": 2, "gas_sand": 3, "undetermined_sand": 4,
             "other_rock": 5, "unlabelled": -1}

DEFAULT_RULES = {"min_time_coverage": 0.8, "min_valid_fraction": 0.6, "class_purity_min": 0.6}


def _bool(series: pd.Series) -> np.ndarray:
    """Robust bool for columns read back from CSV ('True'/'False' strings or real bools)."""
    if series.dtype == bool:
        return np.array(series, dtype=bool)                  # writable copy (pandas may return read-only views)
    return np.array(series.astype(str).str.strip().str.lower().isin(["true", "1", "1.0"]), dtype=bool)


def block_index(twt_ms, t0_ms: float, dt_ms: float, window_ms: float) -> np.ndarray:
    """Block number b of each TWT (half-open bins centred on t0 + b*W).  NaN TWT -> -1."""
    if window_ms % dt_ms:
        raise ValueError(f"window {window_ms} ms must be a multiple of the sample interval {dt_ms} ms")
    t = np.asarray(twt_ms, dtype=float)
    out = np.full(t.shape, -1, dtype=np.int64)
    ok = np.isfinite(t) & (t >= t0_ms - window_ms / 2)
    out[ok] = np.floor((t[ok] - t0_ms) / window_ms + 0.5).astype(np.int64)
    return out


def time_weights(md_m, md_step_m, twt_ms) -> np.ndarray:
    """Time (ms) represented by each log sample: MD step x local dTWT/dMD."""
    md = np.asarray(md_m, dtype=float)
    t = np.asarray(twt_ms, dtype=float)
    if md.size < 2:
        return np.zeros_like(md)
    slope = np.gradient(t, md)
    w = np.asarray(md_step_m, dtype=float) * np.clip(slope, 0, None)
    return np.where(np.isfinite(w), w, 0.0)


def aggregate_well(logs: pd.DataFrame, t0_ms: float, dt_ms: float, window_ms: float,
                   rules: dict | None = None) -> pd.DataFrame:
    """One row per seismic time block for ONE well (input = that well's 02_labels rows, MD order)."""
    r = {**DEFAULT_RULES, **(rules or {})}
    d = logs.sort_values("md_m").reset_index(drop=True)
    d = d[np.isfinite(d["twt_ms"].to_numpy(float))].reset_index(drop=True)
    if d.empty:
        return pd.DataFrame()
    stride = int(round(window_ms / dt_ms))
    d["_b"] = block_index(d["twt_ms"], t0_ms, dt_ms, window_ms)
    d["_w"] = time_weights(d["md_m"], d["md_step_m"], d["twt_ms"])
    d = d[d["_b"] >= 0]
    rows = []
    md_all, twt_all = d["md_m"].to_numpy(float), d["twt_ms"].to_numpy(float)
    for b, g in d.groupby("_b", sort=True):
        w = g["_w"].to_numpy(float)
        wsum = float(w.sum())
        centre = t0_ms + b * window_ms
        row = {"resolution_ms": window_ms, "twt_bin_index": int(b * stride), "block_index": int(b),
               "desired_twt_ms": centre, "sampled_twt_ms": centre, "interpolation_offset_ms": 0.0,
               "n_log_samples": int(len(g)), "logged_time_ms": wsum,
               "time_coverage": min(1.0, wsum / window_ms),
               "label_twt_centroid_ms": float(np.average(g["twt_ms"], weights=w)) if wsum > 0 else np.nan,
               "md_top_m": float(g["md_m"].min()), "md_base_m": float(g["md_m"].max()),
               "md_centre_m": float(np.interp(centre, twt_all, md_all)),
               "tvdss_centre_m": float(np.interp(centre, d["twt_ms"].to_numpy(float), d["tvdss_m"].to_numpy(float)))
               if "tvdss_m" in d else np.nan,
               "tz_strong_extrapolation": bool(_bool(g["tz_strong_extrapolation"]).any())
               if "tz_strong_extrapolation" in g else False}
        row["centroid_offset_ms"] = row["label_twt_centroid_ms"] - centre
        covered = row["time_coverage"] >= r["min_time_coverage"] and wsum > 0
        row["label_mask_reason"] = "" if covered else "partial_log_coverage"
        # ---- continuous targets
        for tgt in ("vsh", "phit", "swt"):
            m = _bool(g[f"{tgt}_mask"]) if f"{tgt}_mask" in g else np.zeros(len(g), bool)
            vals = g[f"{tgt}_fraction"].to_numpy(float) if f"{tgt}_fraction" in g else np.full(len(g), np.nan)
            m = m & np.isfinite(vals)
            wv = float(w[m].sum())
            frac = wv / wsum if wsum > 0 else 0.0
            ok = covered and frac >= r["min_valid_fraction"] and wv > 0
            mean = float(np.average(vals[m], weights=w[m])) if wv > 0 else np.nan
            std = float(np.sqrt(np.average((vals[m] - mean) ** 2, weights=w[m]))) if wv > 0 else np.nan
            row[f"{tgt}_valid_fraction"] = frac
            row[f"y_{tgt}"] = mean if ok else np.nan
            row[f"{tgt}_within_bin_std"] = std if ok else np.nan
            row[f"{tgt}_mask"] = bool(ok)
        # ---- soft labels (time shares) for lithology, fluid and class
        for col, cats, prefix in (("lithology_label", LITH_FRACTIONS, "lith_frac_"),
                                  ("fluid_label", FLUID_FRACTIONS, "fluid_frac_"),
                                  ("class_name", CLASS_FRACTIONS, "class_frac_")):
            vals = g[col].astype(str).to_numpy() if col in g else np.full(len(g), "")
            for c in cats:
                row[prefix + c] = float(w[vals == c].sum() / wsum) if wsum > 0 else 0.0
        # ---- hard labels with purity
        pur = r["class_purity_min"]
        y_lith = -1
        if covered:
            if row["lith_frac_sand"] >= pur:
                y_lith = 1
            elif row["lith_frac_shale"] >= pur:
                y_lith = 0
        row["y_lithology"] = y_lith
        y_fl = -1
        if y_lith == 1:
            sand_w = row["fluid_frac_water_bearing"] + row["fluid_frac_hydrocarbon_potential"] + \
                row["fluid_frac_unresolved"] + row["fluid_frac_conflict"]
            if sand_w > 0:
                if row["fluid_frac_water_bearing"] / sand_w >= pur:
                    y_fl = 0
                elif row["fluid_frac_hydrocarbon_potential"] / sand_w >= pur:
                    y_fl = 1
        row["y_sand_fluid"] = y_fl
        best = max(CLASS_FRACTIONS, key=lambda c: row["class_frac_" + c])
        if covered and best != "unlabelled" and row["class_frac_" + best] >= pur:
            row["class_name"], row["class_id"] = best, CLASS_IDS[best]
        else:
            row["class_name"], row["class_id"] = "unlabelled", -1
            if covered and not row["label_mask_reason"]:
                row["label_mask_reason"] = "mixed_bin" if best != "unlabelled" else "mostly_unlabelled_logs"
        # ---- grade: share of grade-B time inside the assigned class
        if row["class_id"] != -1 and "confidence_grade" in g:
            cls_m = g["class_name"].astype(str).to_numpy() == row["class_name"]
            gb = float(w[cls_m & (g["confidence_grade"].astype(str).to_numpy() == "B")].sum())
            row["label_grade"] = "B" if gb >= 0.5 * float(w[cls_m].sum()) else "C"
        else:
            row["label_grade"] = "U"
        rows.append(row)
    return pd.DataFrame(rows)


def patch_window(n_il: int, n_xl: int, n_t: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetric integer offsets (grid steps / samples) for an odd-sized patch."""
    for n in (n_il, n_xl, n_t):
        if n % 2 == 0:
            raise ValueError("patch dimensions must be odd so the centre is a real sample")
    return (np.arange(n_il) - n_il // 2, np.arange(n_xl) - n_xl // 2, np.arange(n_t) - n_t // 2)


def fill_patches(out: np.ndarray, ch: int, centre_i: np.ndarray, centre_j: np.ndarray, centre_k: np.ndarray,
                 trace_lookup, traces: np.ndarray, n_samples: int, offsets) -> None:
    """Write one channel of many patches into ``out[N, C, nI, nJ, nT]`` (in place).

    ``trace_lookup(i, j)`` returns a row of ``traces`` (or -1) for grid indices;
    samples outside ``[0, n_samples)`` stay NaN.
    """
    oi, oj, ok = offsets
    for n in range(centre_i.size):
        ks = centre_k[n] + ok
        kin = (ks >= 0) & (ks < n_samples)
        for a, di in enumerate(oi):
            for b, dj in enumerate(oj):
                row = trace_lookup(centre_i[n] + di, centre_j[n] + dj)
                if row < 0:
                    continue
                vals = np.full(ok.size, np.nan, dtype=np.float32)
                vals[kin] = traces[row, ks[kin]]
                out[n, ch, a, b, :] = vals
