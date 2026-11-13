"""03_extraction - labels MD -> TVDSS -> TWT onto the native seismic grid; 5x5x21 patches from real SEG-Y.

What this stage answers
-----------------------
"For each well and each seismic time sample along its real (deviated) path:
which label does the log support, where exactly is the sample in the survey,
and what do the eight seismic channels look like around it?"

Steps (each with a progress bar)
--------------------------------
1. Upstream      - 02 labels (+TVDSS/TWT), 01 trajectory/tie tables, 00 volume inventory + geometry.
2. Label bins    - 4 / 12 / 24 ms non-overlapping bins centred on native samples (extraction.py):
                   coverage, valid fractions, purity, soft labels.  Missing stays missing.
3. Positions     - bin centre TWT -> MD (log) -> X/Y (01 trajectory) -> IL/XL (00 geometry)
                   -> nearest live trace.  The XY offset to the bin centre is recorded.
4. Patches       - per well, read only the traces needed (memory-mapped, no full-cube load),
                   assemble [N, 8, 5, 5, 21] float32 with NaN where a volume has no data.
5. Tables        - sample_index (row i = patch i), labels_<W>ms, alignment audit, metadata.
6. Figures       - survey footprint, real example patch + context sections, label/time
                   alignment, coverage per well.
7. Summary       - gate checks: unique keys, row/patch order, bounds, alignment documented.

Status: BLOCKED without a readable full stack + geometry (labels are still binned and
written so the alignment can be reviewed); CONDITIONAL while ties are provisional,
volumes are missing or inversion provenance is unaudited.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..stage_compat import StageBlocked, SpecialistStageContext as StageContext
from ..config import cfg_get, resolve_path, resolve_expected_wells
from ..extraction import (CHANNELS_DEFAULT, STACK_CHANNELS, aggregate_well, fill_patches,
                          patch_window)
from ..plotting import (CLASS_COLORS, CLASS_ORDER, COORD_NOTE, NEUTRAL, SEISMIC_CMAP, SERIES, STATUS, TEXT_2,
                        footer, not_available_figure, setup_style)
from ..progress import StepTracker, progress
from ..provenance import sha256_file
from ..segy import SegyFile, SurveyGeometry

STEPS = ["upstream tables", "label bins", "positions", "patches", "tables", "figures", "summary"]
VERSION_DEFAULT = "v1"


# =============================================================================== stage
def run(ctx: StageContext) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_style()
    steps = StepTracker(STEPS, ctx.stage)
    cfg = ctx.cfg
    ex = cfg.get("extraction") or {}
    reserved = [w.lower() for w in (cfg_get(cfg, "wells.reserved_reviewed", []) or [])]
    channels = list(ex.get("channels") or CHANNELS_DEFAULT)
    if "totalsandprob" in channels:
        raise StageBlocked("TotalSandProbability must never be a predictor channel (AGENTS.md); remove it")
    n_il, n_xl, n_t = int(ex.get("patch_il", 5)), int(ex.get("patch_xl", 5)), int(ex.get("patch_t", 21))
    offsets = patch_window(n_il, n_xl, n_t)
    resolutions = [int(r) for r in (ex.get("resolutions_ms") or [4, 12, 24])]
    rules = {k: float(ex[k]) for k in ("min_time_coverage", "min_valid_fraction", "class_purity_min") if k in ex}
    version = str(ex.get("version", VERSION_DEFAULT))
    ctx.parameters.update({"channels": channels, "patch_shape": [n_il, n_xl, n_t], "resolutions_ms": resolutions,
                           "rules": rules, "version": version})
    run_dir = ctx.run.run_dir

    # ---------------------------------------------------------------- 1 upstream
    steps.next("upstream tables")
    labels = _read_labels(ctx, run_dir)
    wells = resolve_expected_wells(cfg, labels=labels)
    if not wells:
        raise StageBlocked("No wells found in Stage 02 label outputs and wells.expected is empty; cannot extract.")
    ctx.metric("n_wells_expected", len(wells))
    ctx.metric("well_population_source", "configuration" if cfg_get(cfg, "wells.expected", []) else "02_labels outputs")
    tz_p = run_dir / "01_well_ties" / "tables" / "tz_samples_all_wells.csv"
    if not tz_p.exists():
        raise StageBlocked(f"Missing {tz_p}: run 01_well_ties first (needed for the well paths)")
    ctx.add_input(tz_p, "upstream:01_well_ties/tz_samples_all_wells.csv")
    tz = pd.read_csv(tz_p)
    tz["well"] = tz["well"].astype(str).str.lower()
    tie = _optional_csv(ctx, run_dir / "01_well_ties" / "tables" / "tie_summary.csv", "tie_summary")
    sel = _optional_csv(ctx, run_dir / "01_well_ties" / "tables" / "regional_model_selection.csv",
                        "regional_model_selection")
    inv = _optional_csv(ctx, run_dir / "00_inventory" / "tables" / "seismic_volume_inventory.csv",
                        "seismic_volume_inventory")
    unc_rms, unc_basis = _twt_uncertainty(sel)
    volumes, geoms = _open_geometries(ctx, run_dir, channels, inv)
    full_geom = geoms.get("full")
    dt = float(full_geom.dt_ms) if full_geom is not None else float(ex.get("dt_ms_if_no_seismic", 4.0))
    t0 = float(full_geom.t0_ms) if full_geom is not None else 0.0
    for r in resolutions:
        if r % dt:
            raise StageBlocked(f"resolution {r} ms is not a multiple of the seismic sample interval {dt} ms")

    # ---------------------------------------------------------------- 2 label bins
    steps.next("label bins")
    label_tables = {}
    for res in resolutions:
        parts = []
        for w in progress(wells, desc=f"bins {res} ms", unit="well"):
            g = labels[labels["well"] == w]
            if g.empty:
                continue
            a = aggregate_well(g, t0, dt, res, rules)
            if len(a):
                a.insert(0, "well", w)
                parts.append(a)
        if not parts:
            raise StageBlocked("No label rows with TWT: check 02_labels / 01_well_ties outputs")
        label_tables[res] = pd.concat(parts, ignore_index=True)

    # ---------------------------------------------------------------- 3 positions
    steps.next("positions")
    keys = (pd.concat([t[["well", "twt_bin_index", "desired_twt_ms", "md_centre_m", "tvdss_centre_m",
                          "tz_strong_extrapolation"]] for t in label_tables.values()])
            .drop_duplicates(["well", "twt_bin_index"]))
    keys["_order"] = keys["well"].map({w: i for i, w in enumerate(wells)})
    idx = keys.sort_values(["_order", "twt_bin_index"]).drop(columns="_order").reset_index(drop=True)
    for res, t in label_tables.items():
        member = set(zip(t["well"], t["twt_bin_index"]))
        idx[f"in_{res}ms"] = [(w, k) in member for w, k in zip(idx["well"], idx["twt_bin_index"])]
    idx.insert(2, "sampled_twt_ms", idx["desired_twt_ms"])
    idx.insert(3, "interpolation_offset_ms", 0.0)
    idx = _positions(idx, tz, full_geom)
    idx = _well_attributes(idx, tie, reserved, unc_rms, unc_basis, dt)

    # ---------------------------------------------------------------- 4 patches
    steps.next("patches")
    patches = None
    if full_geom is None or "full" not in volumes:
        ctx.block("Full-stack SEG-Y or its 00_inventory geometry is not available in this run: no patches "
                  "were extracted (labels are binned and written for review)")
        idx["patch_status"] = np.where(idx["patch_status"] == "", "no_seismic_in_run", idx["patch_status"])
    else:
        patches = _extract(ctx, idx, channels, volumes, geoms, offsets, version)
    idx.insert(0, "patch_row", np.arange(len(idx)) if patches is not None else -1)

    # ---------------------------------------------------------------- 5 tables
    steps.next("tables")
    _write_tables(ctx, idx, label_tables, channels, volumes, geoms, inv, wells, reserved, version,
                  (n_il, n_xl, n_t), patches is not None, rules, unc_basis)

    # ---------------------------------------------------------------- 6 figures
    steps.next("figures")
    _figure_footprints(ctx, idx, tz, full_geom, wells)
    lab4 = label_tables[min(label_tables)][["well", "twt_bin_index", "class_name", "y_lithology"]]
    _figure_example_patch(ctx, idx.merge(lab4, on=["well", "twt_bin_index"], how="left"), patches, channels,
                          volumes, geoms, ex.get("example_well"), offsets)
    _figure_alignment(ctx, labels, label_tables, idx, ex.get("alignment_wells"), wells, patches is not None)
    _figure_coverage(ctx, idx, label_tables, wells, reserved, patches is not None)

    # ---------------------------------------------------------------- 7 summary
    steps.next("summary")
    _summarise(ctx, idx, label_tables, patches, channels, volumes, wells, unc_rms, dt)
    steps.close()


# =============================================================================== step 1
def _optional_csv(ctx, path: Path, role: str):
    if path.exists():
        ctx.add_input(path, f"upstream:{role}")
        return pd.read_csv(path)
    ctx.warn(f"optional upstream table missing: {path}")
    return None


def _read_labels(ctx: StageContext, run_dir: Path) -> pd.DataFrame:
    override = cfg_get(ctx.cfg, "extraction.labels_table_override")
    if override:
        path = resolve_path(ctx.cfg, override)
        ctx.limit(f"Labels read from extraction.labels_table_override ({path}), NOT from this run's 02_labels - "
                  "use only to review a table produced by another run of this pipeline")
    else:
        path = run_dir / "02_labels" / "tables" / "interpreted_logs_all_wells.csv"
    if path is None or not Path(path).exists():
        raise StageBlocked(f"Missing label table {path}: run 02_labels first")
    ctx.add_input(path, "labels (02_labels/interpreted_logs_all_wells.csv)")
    cols = ["well", "md_m", "md_step_m", "tvdss_m", "twt_ms", "tz_strong_extrapolation", "lithology_label",
            "fluid_label", "class_name", "confidence_grade", "vsh_fraction", "phit_fraction", "swt_fraction",
            "vsh_mask", "phit_mask", "swt_mask"]
    df = pd.read_csv(path, usecols=lambda c: c in cols, low_memory=False)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise StageBlocked(f"Label table {path} lacks columns {missing} (rerun 02_labels with release >= 0.2)")
    df["well"] = df["well"].astype(str).str.lower()
    for c in ("lithology_label", "fluid_label", "class_name", "confidence_grade"):
        df[c] = df[c].fillna("").astype(str)
    return df


def _twt_uncertainty(sel: pd.DataFrame | None) -> tuple[float, str]:
    if sel is not None and "selected" in sel:
        s = sel[sel["selected"].astype(str).str.lower() == "true"]
        if len(s) and "mean_lowo_rms_ms" in s:
            r = s.iloc[0]
            return (float(r["mean_lowo_rms_ms"]),
                    f"leave-one-well-out RMS of the selected regional T-Z model '{r['model']}' "
                    f"(max abs {float(r.get('max_lowo_abs_ms', np.nan)):.1f} ms); proxy for every provisional tie")
    return np.nan, "unknown (01_well_ties regional_model_selection.csv not available)"


def _open_geometries(ctx: StageContext, run_dir: Path, channels: list[str], inv: pd.DataFrame | None):
    """Open only the registered SEG-Y paths recorded by Stage 00 inventory."""
    volumes, geoms = {}, {}
    if inv is None or inv.empty:
        ctx.warn("00_inventory seismic_volume_inventory.csv is unavailable; no seismic channels can be opened")
        return volumes, geoms
    for ch in channels:
        rows = inv[inv["volume"].astype(str).str.casefold() == ch.casefold()]
        gpath = run_dir / "00_inventory" / "tables" / "geometry" / f"{ch}_geometry.npz"
        if rows.empty:
            ctx.warn(f"channel {ch}: no registered dataset was inventoried; its patch channel will be NaN")
            continue
        row = rows.iloc[0]
        raw_path = row.get("path")
        path = Path(str(raw_path)).expanduser() if isinstance(raw_path, str) and raw_path.strip() else None
        if path is None or not path.is_file():
            ctx.warn(f"channel {ch}: registered SEG-Y path is unavailable ({raw_path}); its patch channel will be NaN")
            continue
        if not gpath.is_file():
            ctx.warn(f"channel {ch}: no 00_inventory geometry ({gpath}); rerun 00_inventory")
            continue
        ctx.add_input(gpath, f"upstream:00_inventory geometry {ch}")
        ctx.add_input(path, f"registered SEG-Y {ch}; dataset_id={row.get('dataset_id')}")
        volumes[ch] = path
        geoms[ch] = SurveyGeometry.load_npz(gpath)
    if "full" in geoms:
        g0 = geoms["full"]
        for ch, geom in geoms.items():
            same = (geom.il_min, geom.il_step, geom.xl_min, geom.xl_step, geom.dt_ms, geom.t0_ms) == \
                   (g0.il_min, g0.il_step, g0.xl_min, g0.xl_step, g0.dt_ms, g0.t0_ms)
            if not same:
                ctx.limit(f"channel {ch}: IL/XL numbering or time axis differs from the full stack - "
                          "registration by IL/XL number is NOT safe; channel kept but flagged")
    return volumes, geoms


# =============================================================================== step 3
def _positions(idx: pd.DataFrame, tz: pd.DataFrame, geom: SurveyGeometry | None) -> pd.DataFrame:
    out = idx.copy()
    out["x"], out["y"] = np.nan, np.nan
    for w, rows in out.groupby("well").groups.items():
        t = tz[tz["well"] == w].sort_values("md_m")
        if t.empty or "x" not in t:
            continue
        md = out.loc[rows, "md_centre_m"].to_numpy(float)
        inside = (md >= t["md_m"].iloc[0]) & (md <= t["md_m"].iloc[-1])
        out.loc[rows, "x"] = np.where(inside, np.interp(md, t["md_m"], t["x"]), np.nan)
        out.loc[rows, "y"] = np.where(inside, np.interp(md, t["md_m"], t["y"]), np.nan)
    out["patch_status"] = np.where(out["x"].isna(), "no_trajectory_position", "")
    for c in ("il_frac", "xl_frac", "trace_offset_xy_m"):
        out[c] = np.nan
    for c in ("il", "xl", "grid_i", "grid_j", "centre_trace_full"):
        out[c] = -1
    if geom is None:
        return out
    ok = out["x"].notna().to_numpy()
    il_f, xl_f = geom.xy_to_ilxl(out.loc[ok, "x"].to_numpy(float), out.loc[ok, "y"].to_numpy(float))
    il_n, xl_n = geom.nearest_bin(il_f, xl_f)
    tx, ty = geom.ilxl_to_xy(il_n, xl_n)
    out.loc[ok, "il_frac"], out.loc[ok, "xl_frac"] = il_f, xl_f
    out.loc[ok, "il"], out.loc[ok, "xl"] = il_n, xl_n
    out.loc[ok, "grid_i"] = (il_n - geom.il_min) // geom.il_step
    out.loc[ok, "grid_j"] = (xl_n - geom.xl_min) // geom.xl_step
    out.loc[ok, "trace_offset_xy_m"] = np.hypot(out.loc[ok, "x"].to_numpy(float) - tx,
                                                out.loc[ok, "y"].to_numpy(float) - ty)
    out.loc[ok, "centre_trace_full"] = geom.trace_index(il_n, xl_n)
    k = np.round((out["desired_twt_ms"].to_numpy(float) - geom.t0_ms) / geom.dt_ms).astype(int)
    outside_t = (k < 0) | (k >= geom.ns)
    no_trace = ok & (out["centre_trace_full"].to_numpy() < 0)
    out.loc[no_trace, "patch_status"] = "outside_survey_or_empty_bin"
    out.loc[ok & ~no_trace & outside_t, "patch_status"] = "outside_seismic_time_range"
    return out


def _well_attributes(idx, tie, reserved, unc_rms, unc_basis, dt):
    out = idx.copy()
    out["reserved_reviewed_well"] = out["well"].isin(reserved)
    out["tie_status"], out["extraction_allowed"] = "", True
    if tie is not None:
        t = tie.assign(well=tie["well"].astype(str).str.lower()).set_index("well")
        out["tie_status"] = out["well"].map(t["tie_status"]).fillna("")
        if "extraction_allowed" in t:
            allowed = t["extraction_allowed"].astype(str).str.lower().eq("true")
            out["extraction_allowed"] = out["well"].map(allowed).fillna(False).astype(bool)
    out["twt_uncertainty_rms_ms"] = unc_rms
    out["twt_uncertainty_samples"] = unc_rms / dt if np.isfinite(unc_rms) else np.nan
    out["twt_uncertainty_basis"] = unc_basis
    blocked = ~out["extraction_allowed"] & (out["patch_status"] == "")
    out.loc[blocked, "patch_status"] = "blocked_by_01_tie_qc"
    return out


# =============================================================================== step 4
def _extract(ctx, idx, channels, volumes, geoms, offsets, version):
    """Read only the needed traces per well and volume; write patches to a memory-mapped .npy."""
    oi, oj, ok = offsets
    n = len(idx)
    path = ctx.stage_dir / "patches" / f"patches_4ms_{version}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32,
                                    shape=(n, len(channels), oi.size, oj.size, ok.size))
    arr[:] = np.nan
    gfull = geoms["full"]
    todo = (idx["patch_status"] == "").to_numpy()
    k_all = np.round((idx["desired_twt_ms"].to_numpy(float) - gfull.t0_ms) / gfull.dt_ms).astype(int)
    for c, ch in enumerate(channels):
        if ch not in volumes:
            continue
        g = geoms[ch]
        with SegyFile(volumes[ch], cfg_get(ctx.cfg, "seismic.byte_locations")) as sgy:
            if sgy.ns != g.ns:
                ctx.limit(f"{ch}: SEG-Y has {sgy.ns} samples but its geometry says {g.ns}")
            for w in progress(idx["well"].unique(), desc=f"patches {ch}", unit="well"):
                rows = np.flatnonzero(todo & (idx["well"] == w).to_numpy())
                if rows.size == 0:
                    continue
                ci = idx["grid_i"].to_numpy()[rows]
                cj = idx["grid_j"].to_numpy()[rows]
                # every grid cell the patches of this well touch
                ii = (ci[:, None, None] + oi[None, :, None]).repeat(oj.size, 2).ravel()
                jj = (cj[:, None, None] + oj[None, None, :]).repeat(oi.size, 1).ravel()
                cells = np.unique(np.stack([ii, jj], 1), axis=0)
                # grid indices are defined on the FULL-stack grid; convert to line numbers for this volume
                tr = g.trace_index(gfull.il_min + cells[:, 0] * gfull.il_step,
                                   gfull.xl_min + cells[:, 1] * gfull.xl_step)
                traces = sgy.read_traces(tr)                       # NaN rows for empty / outside bins
                lut = {(int(a), int(b)): (r if t >= 0 else -1) for r, ((a, b), t) in enumerate(zip(cells, tr))}

                def lookup(i, j):
                    return lut.get((int(i), int(j)), -1)

                block = np.full((rows.size, 1, oi.size, oj.size, ok.size), np.nan, dtype=np.float32)
                fill_patches(block, 0, ci, cj, k_all[rows], lookup, traces, sgy.ns, offsets)
                arr[rows, c] = block[:, 0]
    arr.flush()
    idx.loc[todo, "patch_status"] = "extracted"
    # per-channel finite fraction + the centre value for the order check
    for c, ch in enumerate(channels):
        idx[f"valid_frac_{ch}"] = np.isfinite(arr[:, c]).reshape(n, -1).mean(1)
    idx["patch_complete_stacks"] = np.all([idx[f"valid_frac_{ch}"] == 1.0 for ch in channels
                                           if ch in STACK_CHANNELS and ch in volumes], axis=0) \
        if any(ch in volumes for ch in STACK_CHANNELS) else False
    centre = arr[:, channels.index("full"), oi.size // 2, oj.size // 2, ok.size // 2]
    idx["centre_amp_full"] = centre
    # independent re-read of the centre samples straight from the file (verifies row <-> patch order)
    with SegyFile(volumes["full"], cfg_get(ctx.cfg, "seismic.byte_locations")) as sgy:
        rows = np.flatnonzero((idx["patch_status"] == "extracted").to_numpy())
        tr = idx["centre_trace_full"].to_numpy()[rows]
        direct = sgy.read_traces(tr)[np.arange(rows.size), k_all[rows]] if rows.size else np.zeros(0)
    same = np.allclose(direct, centre[rows], equal_nan=True)
    ctx.check("patch_rows_match_direct_segy_reads", bool(same),
              f"{rows.size} centre samples re-read from the full-stack file equal patch[:, full, centre]")
    del arr
    return path


# =============================================================================== step 5
def _write_tables(ctx, idx, label_tables, channels, volumes, geoms, inv, wells, reserved, version, shape,
                  have_patches, rules, unc_basis):
    units = {"desired_twt_ms": "ms", "sampled_twt_ms": "ms", "md_centre_m": "m MD", "tvdss_centre_m": "m",
             "x": "survey XY (CRS undisclosed)", "y": "survey XY (CRS undisclosed)",
             "trace_offset_xy_m": "m (XY units)", "twt_uncertainty_rms_ms": "ms"}
    ctx.save_table(idx, "sample_index.csv",
                   "One row per (well, twt_bin_index); row i == patch i in patches_4ms_<version>.npy",
                   units=units, required=True)
    key = idx.set_index(["well", "twt_bin_index"])["patch_row"]
    for res, t in label_tables.items():
        t = t.copy()
        t.insert(1, "patch_row", [int(key.get((w, k), -1)) for w, k in zip(t["well"], t["twt_bin_index"])])
        t.insert(2, "label_version", f"labels_{res}ms_{version}")
        t["reserved_reviewed_well"] = t["well"].isin(reserved)
        ctx.save_table(t, f"labels_{res}ms_{version}.csv",
                       f"{res} ms non-overlapping label bins (targets, masks, soft labels) keyed to patch_row",
                       required=True)
    # ---- per-well rows = sample index + 4 ms labels
    base = min(label_tables)
    lab4 = label_tables[base].drop(columns=["desired_twt_ms", "sampled_twt_ms", "interpolation_offset_ms",
                                            "md_centre_m", "tvdss_centre_m", "tz_strong_extrapolation"])
    merged = idx.merge(lab4, on=["well", "twt_bin_index"], how="left")
    for w in wells:
        d = merged[merged["well"] == w]
        if d.empty:
            d = pd.DataFrame([{"well": w, "not_available_reason": "no labelled log samples with TWT"}])
        ctx.save_table(d, f"per_well/{w}_samples.csv", f"{w}: sample index + {base} ms labels", required=True)
    # ---- alignment audit (one row per well)
    rows = []
    for w in wells:
        s = idx[idx["well"] == w]
        r = {"well": w, "reserved_reviewed_well": w in reserved, "n_patch_rows": int(len(s))}
        if len(s):
            r.update({"twt_top_ms": float(s["desired_twt_ms"].min()), "twt_base_ms": float(s["desired_twt_ms"].max()),
                      "tie_status": s["tie_status"].iloc[0], "extraction_allowed": bool(s["extraction_allowed"].iloc[0]),
                      "twt_uncertainty_rms_ms": float(s["twt_uncertainty_rms_ms"].iloc[0]),
                      "twt_uncertainty_samples": float(s["twt_uncertainty_samples"].iloc[0]),
                      "n_bins_strong_tz_extrapolation": int(s["tz_strong_extrapolation"].astype(bool).sum()),
                      "n_distinct_centre_traces": int(s.loc[s["centre_trace_full"] >= 0, "centre_trace_full"].nunique()),
                      "max_trace_offset_xy_m": float(s["trace_offset_xy_m"].max()) if s["trace_offset_xy_m"].notna().any() else np.nan,
                      "max_interpolation_offset_ms": float(s["interpolation_offset_ms"].abs().max()),
                      **{f"n_patch_{k}": int(v) for k, v in s["patch_status"].value_counts().items()}})
        for res, t in label_tables.items():
            u = t[t["well"] == w]
            r[f"n_bins_{res}ms"] = int(len(u))
            r[f"n_lithology_labels_{res}ms"] = int((u["y_lithology"] >= 0).sum()) if len(u) else 0
            r[f"n_sand_fluid_labels_{res}ms"] = int((u["y_sand_fluid"] >= 0).sum()) if len(u) else 0
            r[f"n_vsh_targets_{res}ms"] = int(u["vsh_mask"].sum()) if len(u) else 0
            r[f"mean_abs_centroid_offset_ms_{res}ms"] = float(u["centroid_offset_ms"].abs().mean()) if len(u) else np.nan
            for reason, n in (u["label_mask_reason"].value_counts().items() if len(u) else []):
                if reason:
                    r[f"n_{reason}_{res}ms"] = int(n)
        rows.append(r)
    ctx.save_table(pd.DataFrame(rows), "label_alignment_audit.csv",
                   "Per well: TWT range, tie status and uncertainty, bins per resolution, masking reasons, "
                   "XY offsets to traces, patch status counts", required=True)
    # ---- extraction metadata (one row per channel)
    rows = []
    invd = inv.set_index("volume") if inv is not None and "volume" in inv else None
    for c, ch in enumerate(channels):
        g = geoms.get(ch)
        role = ("amplitude stack (angle range UNVERIFIED)" if ch in STACK_CHANNELS and ch != "full"
                else "full stack" if ch == "full"
                else "conditional predictor: inversion provenance unaudited")
        rows.append({"channel_index": c, "channel": ch, "role": role, "path": str(volumes.get(ch, "")),
                     "available_in_run": ch in volumes, "sha256_from_00_inventory":
                     (invd.loc[ch, "sha256"] if invd is not None and ch in invd.index and "sha256" in invd else ""),
                     "ns": g.ns if g else np.nan, "dt_ms": g.dt_ms if g else np.nan, "t0_ms": g.t0_ms if g else np.nan,
                     "t_max_ms": (g.t0_ms + g.dt_ms * (g.ns - 1)) if g else np.nan,
                     "n_traces": g.n_traces if g else np.nan, "il_min": g.il_min if g else np.nan,
                     "il_step": g.il_step if g else np.nan, "xl_min": g.xl_min if g else np.nan,
                     "xl_step": g.xl_step if g else np.nan,
                     "affine_residual_max_m": g.affine_residual_max if g else np.nan,
                     "patch_axis_order": "row, channel, il_offset, xl_offset, time_offset",
                     "patch_shape_per_channel": "x".join(map(str, shape)),
                     "il_offset_unit": f"grid steps ({g.il_step if g else '?'} inline numbers each)",
                     "missing_value_policy": "NaN (never zero-padded)", "normalisation": "none (raw values)"})
    ctx.save_table(pd.DataFrame(rows), "extraction_metadata.csv",
                   "Channel order, source volumes, time axes, geometry and patch conventions", required=True)
    if have_patches:
        p = ctx.stage_dir / "patches" / f"patches_4ms_{version}.npy"
        ctx._register(p, "patches", f"float32 patches [N, {len(channels)}, {shape[0]}, {shape[1]}, {shape[2]}]",
                      required=True)
        ctx.save_json({"file": p.name, "sha256": sha256_file(p), "dtype": "float32",
                       "shape": [int(len(idx)), len(channels), *shape],
                       "axis_order": ["row (= sample_index.csv patch_row)", "channel", "il_offset", "xl_offset",
                                      "time_offset"],
                       "channels": channels, "time_offsets_samples": list(range(-(shape[2] // 2), shape[2] // 2 + 1)),
                       "dt_ms": float(geoms["full"].dt_ms), "missing_value": "NaN",
                       "rows_without_patch": "all-NaN; see sample_index.patch_status",
                       "label_tables": [f"labels_{r}ms_{version}.csv" for r in label_tables],
                       "rules": rules, "twt_uncertainty_basis": unc_basis,
                       "excluded_channels": ["totalsandprob (target-like derived product)"]},
                      f"patches/patch_manifest_{version}.json", "Patch array contract (shape, axes, channels, hash)",
                      required=True)


# =============================================================================== step 6: figures
def _figure_footprints(ctx, idx, tz, geom, wells):
    import matplotlib.pyplot as plt

    if geom is None:
        fig = not_available_figure("Well paths vs seismic survey footprint",
                                   ["Full-stack geometry (00_inventory/tables/geometry/full_geometry.npz) is not "
                                    "available in this run; run 00_inventory where the SEG-Y files are reachable."])
        ctx.save_figure(fig, "well_vs_survey_footprints.png", "NOT AVAILABLE", required=True)
        return
    fig, (a, b) = plt.subplots(1, 2, figsize=(19, 9), gridspec_kw={"width_ratios": [1.3, 1]})
    step = max(1, int(np.sqrt(geom.grid.size / 60000)))
    gi, gj = np.nonzero(geom.grid[::step, ::step] >= 0)
    gx, gy = geom.ilxl_to_xy(geom.il_min + gi * step * geom.il_step, geom.xl_min + gj * step * geom.xl_step)
    a.scatter(gx, gy, s=1, color="#d9d8d4", rasterized=True, label="live full-stack bins (subsampled)")
    for w in wells:
        t = tz[tz["well"] == w]
        if t.empty:
            continue
        a.plot(t["x"], t["y"], color=TEXT_2, lw=1.2)
        s = idx[(idx["well"] == w) & (idx["patch_status"] == "extracted")]
        a.scatter(s["x"], s["y"], s=6, color=SERIES[0], zorder=3)
        a.text(t["x"].iloc[-1], t["y"].iloc[-1], f" {w}", fontsize=9, va="center")
    a.scatter([], [], s=12, color=SERIES[0], label="extracted patch centres (4 ms rows)")
    a.set_aspect("equal", adjustable="datalim")
    a.set_xlabel("X")
    a.set_ylabel("Y")
    a.set_title("Survey footprint, deviated well paths and patch centres")
    a.legend(loc="lower left", fontsize=8, markerscale=3)
    data = [idx.loc[(idx["well"] == w) & idx["trace_offset_xy_m"].notna(), "trace_offset_xy_m"].to_numpy()
            for w in wells]
    keep = [i for i, d in enumerate(data) if d.size]
    if keep:
        b.boxplot([data[i] for i in keep], vert=False, widths=0.6, showfliers=False)
        b.set_yticks(np.arange(1, len(keep) + 1))
        b.set_yticklabels([f"{wells[i]} ({idx.loc[idx['well'] == wells[i], 'centre_trace_full'].nunique()} traces)"
                           for i in keep])
    half_diag = 0.5 * float(np.hypot(*geom.bin_spacing()))
    b.axvline(half_diag, color=STATUS["warning"], ls="--", lw=1)
    b.text(half_diag, 0.02, f" half bin diagonal {half_diag:.1f} m", fontsize=8, color=TEXT_2,
           transform=b.get_xaxis_transform())
    b.set_xlabel("XY distance from path point to the nearest bin centre (m)")
    b.set_title("Lateral snapping error per well (and distinct centre traces)")
    fig.suptitle("03_extraction: where the patches come from", fontsize=13, fontweight="bold")
    footer(fig, COORD_NOTE + " Nearest-bin snapping; no lateral interpolation of seismic.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ctx.save_figure(fig, "well_vs_survey_footprints.png", "Survey footprint, well paths, patch centres, "
                    "snapping offsets", required=True, multipanel=True)


def _figure_example_patch(ctx, idx, patches_path, channels, volumes, geoms, example_well, offsets):
    import matplotlib.pyplot as plt

    if patches_path is None:
        fig = not_available_figure("Example 5x5x21 patch from the real full stack",
                                   ["No SEG-Y patches were extracted in this run (full stack not reachable).",
                                    "Run 03_extraction where the registered SEG-Y inputs are mounted and readable by the worker."])
        ctx.save_figure(fig, "example_patch_fullstack.png", "NOT AVAILABLE", required=True)
        return
    arr = np.load(patches_path, mmap_mode="r")
    ok = idx["patch_status"].eq("extracted") & idx.get("patch_complete_stacks", False)
    sand = ok & idx["y_lithology"].eq(1)                     # prefer a labelled sand bin for the example
    dev = ~idx["reserved_reviewed_well"].astype(bool) if "reserved_reviewed_well" in idx else True
    cand = idx[sand & dev] if (sand & dev).any() else idx[sand] if sand.any() else idx[ok]
    if example_well and (cand["well"] == example_well).any():
        cand = cand[cand["well"] == example_well]
    elif len(cand):
        drift = cand.groupby("well")["centre_trace_full"].nunique()
        cand = cand[cand["well"] == drift.idxmax()]
    if cand.empty:
        cand = idx[idx["patch_status"].eq("extracted")]
    if cand.empty:
        fig = not_available_figure("Example patch", ["no extracted patch rows"])
        ctx.save_figure(fig, "example_patch_fullstack.png", "NOT AVAILABLE", required=True)
        return
    row = cand.iloc[len(cand) // 2]
    r = int(row["patch_row"])
    p = np.array(arr[r], dtype=float)                         # [C, nI, nJ, nT] (copy, then release the file)
    del arr
    oi, oj, ok_ = offsets
    full = p[channels.index("full")]
    clip = np.nanpercentile(np.abs(full), 98) if np.isfinite(full).any() else 1.0
    g = geoms["full"]
    il, xl, t_c = int(row["il"]), int(row["xl"]), float(row["desired_twt_ms"])
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.25, 1])
    ax_il, ax_xl = fig.add_subplot(gs[0, :2]), fig.add_subplot(gs[0, 2:])
    with SegyFile(volumes["full"], cfg_get(ctx.cfg, "seismic.byte_locations")) as sgy:
        for ax, (sec, axis_vals), label, centre in (
                (ax_il, sgy.read_inline(g, il), f"inline {il}", xl),
                (ax_xl, sgy.read_crossline(g, xl), f"crossline {xl}", il)):
            step = g.xl_step if ax is ax_il else g.il_step
            sel = np.abs(axis_vals - centre) <= 60 * step
            k0, k1 = max(0, int((t_c - 300 - g.t0_ms) / g.dt_ms)), min(g.ns, int((t_c + 300 - g.t0_ms) / g.dt_ms))
            img = sec[sel][:, k0:k1].T
            sclip = np.nanpercentile(np.abs(img), 98) if np.isfinite(img).any() else 1.0
            ax.imshow(img, aspect="auto", cmap=SEISMIC_CMAP, vmin=-sclip, vmax=sclip, interpolation="nearest",
                      extent=[axis_vals[sel][0], axis_vals[sel][-1], g.t0_ms + k1 * g.dt_ms, g.t0_ms + k0 * g.dt_ms])
            half_w = (oj.size // 2 if ax is ax_il else oi.size // 2) * step
            ax.add_patch(plt.Rectangle((centre - half_w, t_c - (ok_.size // 2) * g.dt_ms), 2 * half_w,
                                       (ok_.size - 1) * g.dt_ms, fill=False, ec="#0b0b0b", lw=2))
            # well path samples that fall on this section
            on = idx[(idx["well"] == row["well"]) & ((idx["il"] == il) if ax is ax_il else (idx["xl"] == xl))]
            ax.plot(on["xl"] if ax is ax_il else on["il"], on["desired_twt_ms"], "o", ms=3, color=SERIES[1],
                    label=f"{row['well']} path samples on this {label.split()[0]}")
            ax.set_xlabel("crossline number" if ax is ax_il else "inline number")
            ax.set_ylabel("TWT (ms)")
            ax.set_title(f"Real full stack, {label} (+/-60 bins, +/-300 ms); black box = patch")
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(False)
    t_axis = t_c + ok_ * g.dt_ms
    for a_i, (img, xlab, title) in enumerate((
            (full[:, oj.size // 2, :].T, "IL offset (bins)", "patch: IL-time (centre XL)"),
            (full[oi.size // 2, :, :].T, "XL offset (bins)", "patch: XL-time (centre IL)"),
            (full[:, :, ok_.size // 2], "XL offset (bins)", "patch: IL-XL at centre time"))):
        ax = fig.add_subplot(gs[1, a_i])
        if a_i < 2:
            ax.imshow(img, aspect="auto", cmap=SEISMIC_CMAP, vmin=-clip, vmax=clip, interpolation="nearest",
                      extent=[-2.5, 2.5, t_axis[-1] + g.dt_ms / 2, t_axis[0] - g.dt_ms / 2])
            ax.set_ylabel("TWT (ms)")
        else:
            ax.imshow(img, cmap=SEISMIC_CMAP, vmin=-clip, vmax=clip, interpolation="nearest",
                      extent=[-2.5, 2.5, 2.5, -2.5])
            ax.set_ylabel("IL offset (bins)")
        ax.set_xlabel(xlab)
        ax.set_title(title, fontsize=10)
        ax.grid(False)
    ax = fig.add_subplot(gs[1, 3])
    for c, ch in enumerate(channels):
        tr = p[c, oi.size // 2, oj.size // 2, :]
        if not np.isfinite(tr).any():
            ax.text(c, t_axis[ok_.size // 2], "n/a", ha="center", fontsize=7, color=STATUS["critical"])
            continue
        if ch not in STACK_CHANNELS:
            tr = tr - np.nanmean(tr)                           # inversion values: show variation, not level
        scale = np.nanmax(np.abs(tr)) or 1.0
        ax.plot(c + 0.4 * tr / scale, t_axis, color=SERIES[0] if ch in STACK_CHANNELS else SERIES[1], lw=1.2)
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, fontsize=8)
    ax.invert_yaxis()
    ax.set_ylabel("TWT (ms)")
    ax.set_title("Centre trace of all 8 channels\n(each scaled to its own max; inversion de-meaned; display only)",
                 fontsize=10)
    fig.suptitle(f"03_extraction: example patch {row['well']} at {t_c:.0f} ms (IL {il}, XL {xl}, class "
                 f"{row.get('class_name', '?')}) - real SEG-Y values", fontsize=13, fontweight="bold")
    footer(fig, "Patch axes are native grid steps (bins) and samples, not metres. Colour clip = 98th percentile "
                "of |amplitude|. Inversion channels in their own units. No interpolation.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ctx.save_figure(fig, "example_patch_fullstack.png", "Real full-stack context sections, patch slices and "
                    "8-channel centre traces", required=True, multipanel=True)


def _figure_alignment(ctx, labels, label_tables, idx, chosen, wells, have_seismic):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    have = [w for w in wells if (labels["well"] == w).any()]
    picks = [w for w in (chosen or []) if w in have] or [w for w in ("s-1", "o-1", "m-1") if w in have] or have[:3]
    picks = picks[:3]
    res = sorted(label_tables)
    fig = plt.figure(figsize=(20, 12))
    outer = fig.add_gridspec(1, len(picks), wspace=0.25)
    cmap = ListedColormap([CLASS_COLORS[c] for c in CLASS_ORDER])
    for p_i, w in enumerate(picks):
        inner = outer[p_i].subgridspec(1, 3, width_ratios=[1.3, 1.0, 0.8], wspace=0.08)
        ax_v = fig.add_subplot(inner[0])
        ax_c = fig.add_subplot(inner[1], sharey=ax_v)
        ax_s = fig.add_subplot(inner[2], sharey=ax_v)
        L = labels[(labels["well"] == w) & labels["twt_ms"].notna()].sort_values("twt_ms")
        ax_v.plot(L["vsh_fraction"], L["twt_ms"], color=NEUTRAL, lw=0.5, label="log VSH (valid)")
        for r_i, r in enumerate(res):
            t = label_tables[r][label_tables[r]["well"] == w]
            ax_v.step(t["y_vsh"], t["desired_twt_ms"], where="mid", lw=1.3 if r_i else 1.0,
                      color=[SERIES[0], SERIES[1], SERIES[2]][r_i % 3], label=f"{r} ms bins")
        ax_v.set_xlim(0, 1)
        ax_v.set_xlabel("VSH (v/v)")
        ax_v.set_title(w, fontsize=12, fontweight="bold")
        if p_i == 0:
            ax_v.set_ylabel("TWT (ms)")
        ax_v.legend(fontsize=7, loc="lower left")
        # class strips: log level + each resolution
        tops = [("log", L["twt_ms"].to_numpy(float), L["class_name"].to_numpy(object))]
        for r in res:
            t = label_tables[r][label_tables[r]["well"] == w]
            tops.append((f"{r} ms", t["desired_twt_ms"].to_numpy(float), t["class_name"].to_numpy(object)))
        # resample every strip onto one regular 0.5 ms axis (nearest sample) and draw it as one image
        lo = min(tt.min() for _, tt, _ in tops if tt.size)
        hi = max(tt.max() for _, tt, _ in tops if tt.size)
        grid_t = np.arange(lo, hi + 0.5, 0.5)
        img = np.full((grid_t.size, len(tops)), np.nan)
        for s_i, (name, tt, cls) in enumerate(tops):
            if tt.size == 0:
                continue
            codes = pd.Categorical(cls, categories=CLASS_ORDER).codes.astype(float)
            codes[codes < 0] = np.nan
            pos = np.clip(np.searchsorted(tt, grid_t), 0, tt.size - 1)
            prev = np.clip(pos - 1, 0, tt.size - 1)
            nearest = np.where(np.abs(tt[prev] - grid_t) < np.abs(tt[pos] - grid_t), prev, pos)
            half = 0.5 * (float(name.split()[0]) if name != "log" else 0.5)
            col = codes[nearest]
            col[np.abs(tt[nearest] - grid_t) > max(half, 1.0)] = np.nan        # no colour across gaps
            img[:, s_i] = col
        ax_c.imshow(img, aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(CLASS_ORDER) - 0.5, interpolation="nearest",
                    extent=[0, len(tops), grid_t[-1], grid_t[0]])
        ax_c.set_xticks(np.arange(len(tops)) + 0.5)
        ax_c.set_xticklabels([n for n, *_ in tops], fontsize=8, rotation=30)
        ax_c.set_title("class: log vs bins", fontsize=9)
        ax_c.grid(False)
        plt.setp(ax_c.get_yticklabels(), visible=False)
        s = idx[(idx["well"] == w) & (idx["patch_status"] == "extracted")] if have_seismic else pd.DataFrame()
        if len(s) and "centre_amp_full" in s:
            amp = s["centre_amp_full"].to_numpy(float)
            sc = np.nanpercentile(np.abs(amp), 98) or 1.0
            ax_s.fill_betweenx(s["desired_twt_ms"], 0, np.clip(amp / sc, -1, 1), where=amp > 0, color="#0b0b0b", lw=0)
            ax_s.plot(np.clip(amp / sc, -1, 1), s["desired_twt_ms"], color="#0b0b0b", lw=0.6)
            ax_s.set_xlim(-1.1, 1.1)
            ax_s.set_title("full stack along path\n(centre trace, 4 ms)", fontsize=9)
        else:
            ax_s.set_facecolor("#f3f2ef")
            ax_s.text(0.5, 0.5, "SEISMIC NOT AVAILABLE\nin this run", rotation=90, ha="center", va="center",
                      transform=ax_s.transAxes, color=STATUS["critical"], fontweight="bold")
            ax_s.set_title("full stack along path", fontsize=9)
        ax_s.set_xticks([])
        plt.setp(ax_s.get_yticklabels(), visible=False)
        if len(L):
            ax_v.set_ylim(float(L["twt_ms"].max()) + 5, float(L["twt_ms"].min()) - 5)   # TWT down
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=CLASS_COLORS[c], label=c) for c in CLASS_ORDER], loc="lower center",
               ncol=len(CLASS_ORDER), fontsize=9)
    fig.suptitle("03_extraction: log labels moved to seismic time (MD -> TVDSS -> TWT) and binned at "
                 + "/".join(f"{r}" for r in res) + " ms", fontsize=13, fontweight="bold")
    footer(fig, "Bins centred on native seismic samples (no time interpolation). A bin keeps a class only if one class "
                "covers >= purity of its logged time; otherwise it is 'mixed' (unlabelled). TWT from the provisional "
                "01_well_ties model: expect several ms of tie error.", y=0.035)
    ctx.save_figure(fig, "label_time_alignment.png", "Log vs binned labels in TWT for selected wells, with the "
                    "real full-stack trace along the path", required=True, multipanel=True)


def _figure_coverage(ctx, idx, label_tables, wells, reserved, have_seismic):
    import matplotlib.pyplot as plt

    res = sorted(label_tables)
    fig, axs = plt.subplots(1, 3, figsize=(20, 8.5))
    y = np.arange(len(wells))[::-1]
    names = [f"{w} (reserved)" if w in reserved else w for w in wells]
    t4 = label_tables[res[0]]
    groups = [("lithology label", t4["y_lithology"] >= 0, SERIES[0]),
              ("mixed bin", t4["label_mask_reason"] == "mixed_bin", STATUS["warning"]),
              ("mostly unlabelled logs", t4["label_mask_reason"] == "mostly_unlabelled_logs", NEUTRAL),
              ("partial log coverage", t4["label_mask_reason"] == "partial_log_coverage", STATUS["serious"]),
              ("other (class only / no lithology)", (t4["y_lithology"] < 0) & (t4["label_mask_reason"] == ""), "#c9b79c")]
    left = np.zeros(len(wells))
    for name, m, col in groups:
        v = t4[m].groupby("well").size().reindex(wells).fillna(0).to_numpy()
        axs[0].barh(y, v, left=left, color=col, label=name, edgecolor="white", lw=0.4)
        left += v
    axs[0].set_title(f"{res[0]} ms bins per well and why some carry no lithology label")
    axs[0].set_xlabel("number of bins")
    axs[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3)
    width = 0.8 / len(res)
    for r_i, r in enumerate(res):
        t = label_tables[r]
        v = t[t["y_lithology"] >= 0].groupby("well").size().reindex(wells).fillna(0).to_numpy()
        axs[1].barh(y + (r_i - (len(res) - 1) / 2) * width, v, height=width, color=[SERIES[0], SERIES[1], SERIES[2]][r_i % 3],
                    label=f"{r} ms")
    axs[1].set_title("Usable lithology labels per resolution (independent bins)")
    axs[1].set_xlabel("number of labelled bins")
    axs[1].legend(fontsize=8)
    if have_seismic:
        st = idx.groupby(["well", "patch_status"]).size().unstack(fill_value=0).reindex(wells).fillna(0)
        left = np.zeros(len(wells))
        colours = {"extracted": STATUS["good"], "outside_survey_or_empty_bin": STATUS["critical"],
                   "outside_seismic_time_range": STATUS["serious"], "blocked_by_01_tie_qc": STATUS["warning"],
                   "no_trajectory_position": NEUTRAL}
        for s in st.columns:
            axs[2].barh(y, st[s].to_numpy(), left=left, color=colours.get(s, NEUTRAL), label=s, edgecolor="white", lw=0.4)
            left += st[s].to_numpy()
        axs[2].set_title("Patch status per well (4 ms rows)")
        axs[2].set_xlabel("number of patch rows")
        axs[2].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2)
    else:
        axs[2].set_axis_off()
        axs[2].text(0.5, 0.55, "Patch extraction\nNOT AVAILABLE", ha="center", fontsize=20, fontweight="bold",
                    color=STATUS["critical"], transform=axs[2].transAxes)
        axs[2].text(0.5, 0.35, "SEG-Y volumes are not reachable in this run.", ha="center", fontsize=10,
                    transform=axs[2].transAxes)
    for a in axs[:2] if not have_seismic else axs:
        a.set_yticks(y)
        a.set_yticklabels(names)
    fig.suptitle("03_extraction: coverage by well", fontsize=13, fontweight="bold")
    footer(fig, "Coarser bins are non-overlapping, so 12/24 ms counts are the honest number of independent samples.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ctx.save_figure(fig, "extraction_coverage_by_well.png", "Bins, labels and patch status per well",
                    required=True, multipanel=True)


# =============================================================================== step 7
def _summarise(ctx, idx, label_tables, patches, channels, volumes, wells, unc_rms, dt):
    dup = idx.duplicated(["well", "twt_bin_index"]).sum()
    ctx.check("unique_well_twt_bin_keys", dup == 0, f"{len(idx)} rows, {dup} duplicates")
    ctx.check("bins_on_native_samples", bool(np.allclose(idx["interpolation_offset_ms"], 0)),
              "desired_twt_ms == sampled_twt_ms for every row (bins centred on native samples)")
    for res, t in label_tables.items():
        ctx.check(f"unique_keys_{res}ms", not t.duplicated(["well", "twt_bin_index"]).any(), f"{len(t)} bins")
    no_impute = all(t.loc[~t[f"{v}_mask"], f"y_{v}"].isna().all() for t in label_tables.values()
                    for v in ("vsh", "phit", "swt"))
    ctx.check("missing_targets_stay_missing", bool(no_impute), "every masked bin target is NaN")
    lab_wells = sorted(label_tables[min(label_tables)]["well"].unique())
    ctx.check("labels_binned_for_every_well", set(lab_wells) == set(wells), f"{len(lab_wells)}/{len(wells)} wells")
    t4 = label_tables[min(label_tables)]
    for res, t in label_tables.items():
        ctx.metric(f"n_bins_{res}ms", int(len(t)))
        ctx.metric(f"n_lithology_labels_{res}ms", int((t["y_lithology"] >= 0).sum()))
        ctx.metric(f"n_sand_fluid_labels_{res}ms", int((t["y_sand_fluid"] >= 0).sum()))
        ctx.metric(f"n_vsh_targets_{res}ms", int(t["vsh_mask"].sum()))
        ctx.metric(f"class_counts_{res}ms", t["class_name"].value_counts().to_dict())
    ctx.metric("twt_uncertainty_rms_ms", unc_rms)
    ctx.metric("mean_abs_centroid_offset_ms_4ms", float(t4["centroid_offset_ms"].abs().mean()))
    if patches is not None:
        st = idx["patch_status"].value_counts().to_dict()
        ctx.metric("patch_status_counts", st)
        ctx.metric("patch_array", {"rows": int(len(idx)), "channels": channels})
        missing = [c for c in channels if c not in volumes]
        if missing:
            ctx.limit(f"Channels without a readable volume (NaN in every patch): {missing}")
        n_bad = sum(v for k, v in st.items() if k != "extracted")
        if n_bad:
            ctx.warn(f"{n_bad} patch rows not extracted: {({k: v for k, v in st.items() if k != 'extracted'})}")
    if np.isfinite(unc_rms) and unc_rms > dt:
        ctx.limit(f"TWT tie uncertainty ~{unc_rms:.1f} ms RMS = {unc_rms / dt:.1f} samples: 4 ms labels can be "
                  "misplaced by more than one sample - compare the 12/24 ms datasets before trusting 4 ms results")
    ctx.limit("Inversion channels (pimp, simp, vpvs) are conditional predictors: their provenance and calibration "
              "wells are unaudited")
    if any(c in volumes for c in ("near", "mid", "far", "ultrafar")):
        ctx.limit("Partial-stack angle ranges are undocumented: near/mid/far/ultrafar are amplitude channels, "
                  "not verified angle gathers")
    ctx.readme("Datasets", "\n".join(
        f"* `labels_{r}ms_*.csv`: {len(t)} bins, {int((t['y_lithology'] >= 0).sum())} with a lithology label, "
        f"{int((t['y_sand_fluid'] >= 0).sum())} with a sand-fluid label" for r, t in label_tables.items())
        + "\n* `sample_index.csv` row i == patch i (`patch_row`).  Coarser bins point to the patch at their centre.")
