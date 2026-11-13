"""02_labels - interpreted logs -> curated Vsh/PHIT/SWT targets, lithology/fluid classes, evidence.

What this stage answers
-----------------------
"For every logged depth of every well: which values can be trusted as training
truth, what rock is it, what does the evidence say about its fluid, and how sure
are we?"  Nothing here looks at seismic; stage 03 moves these labels to TWT.

Steps (each shows a progress bar in the terminal)
--------------------------------------------------
1. Upstream tables   - LAS paths (00), curve inventory (00), MD->TVDSS->TWT samples (01).
2. Read logs         - the original LAS files; a compiled workbook is used ONLY when the
                       LAS files are not reachable (``labels.log_source``), and is then
                       verified against the stage-00 inventory.
3. Verify source     - sample counts, valid counts, min / median / max per curve must match
                       the stage-00 inventory exactly.
4. Recompute         - GR->Vsh, density/neutron porosity, Archie Sw: applied only when the
                       curves exist AND parameters are configured; otherwise NOT_APPLIED + reason.
5. Curve QC          - bounds, placeholders (VSH = 0 & PHIT = 0), artifacts; raw values kept.
6. Provider flags    - what PS_GIP / PS_OIP really encode (implied cutoffs, conflicts).
7. Labels            - lithology, fluid evidence, class, status, confidence grade (labelling.py).
8. Depth / time      - TVDSS and TWT interpolated from 01_well_ties along the real trajectory.
9. Support + tables  - which tasks the labels can support (TASK_UNSUPPORTED where not).
10. Figures          - per-well petrophysical panels, class distribution, crossplots.
11. Summary          - checks, metrics, limitations -> STATUS.json.

Status logic: CONDITIONAL whenever upstream ties are provisional, the labels rest on
unreviewed interpreted curves, or fluid classes are unsupported (all true for the
Challenge 1/2 data).  BLOCKED if no log data can be read at all.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..stage_compat import StageBlocked, SpecialistStageContext as StageContext
from ..config import cfg_get, resolve_path, resolve_expected_wells
from ..io_wells import classify_curve, read_las
from ..labelling import (CLASS_NAME, CLASS_TABLE, assign_labels, cutoff_sensitivity,
                         demote_unsupported_classes, effective_policy, qc_curves, reason_codes,
                         sample_thickness, task_support)
from ..petrophysics import (density_porosity, gamma_ray_index, neutron_density_porosity,
                            archie_sw, vsh_larionov_older, vsh_larionov_tertiary, vsh_linear)
from ..plotting import (CLASS_COLORS, CLASS_ORDER, FLUID_COLORS, GRADE_COLORS, GRADE_ORDER, LITH_COLORS,
                        LITH_ORDER, NEUTRAL, SERIES, STATUS, TEXT_2, footer, not_available_figure, setup_style)
from ..progress import StepTracker, progress

STEPS = ["upstream tables", "read logs", "verify source", "recompute where possible", "curve QC",
         "provider flag audit", "labels and classes", "depth/time attach", "support and tables",
         "figures", "summary"]

# provider curves -> column names used in this stage
PROVIDER = {"VSH": "vsh_raw", "PHIT": "phit_raw", "SWT": "swt_raw",
            "HC_INDICATOR_GAS": "gas_flag_raw", "HC_INDICATOR_OIL": "oil_flag_raw"}
# raw logs that would allow independent recomputation
RAW_LOGS = {"GR": "gr_raw", "RHOB": "rhob_raw", "NPHI": "nphi_raw", "RT_DEEP": "rt_raw", "DT": "dt_raw"}
COMPARE_STATS = ("n_samples", "n_valid", "min", "p50", "max")


# =============================================================================== helpers
def _pick_curve(mnemonics: list[str], family: str) -> str | None:
    """First curve of a family; an exact family-name mnemonic wins (e.g. 'VSH' before 'VCL')."""
    hits = [m for m in mnemonics if classify_curve(m) == family]
    for m in hits:
        if m.strip().upper().split(":")[0] == family:
            return m
    return hits[0] if hits else None


def _curve_stats(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    ok = v[np.isfinite(v)]
    out = {"n_samples": int(v.size), "n_valid": int(ok.size)}
    if ok.size:
        out.update({"min": float(ok.min()), "p50": float(np.percentile(ok, 50)), "max": float(ok.max())})
    return out


def _read_workbook(path: Path, sheet: str | None) -> pd.DataFrame:
    """Read a compiled log workbook: one sheet, columns WELL, [SOURCE_FILE], DEPTH (m|ft), curves..."""
    book = pd.read_excel(path, sheet_name=None)
    name = sheet if sheet in book else next(iter(book))
    df = book[name]
    cols = {c: str(c).strip() for c in df.columns}
    well_col = next((c for c, n in cols.items() if n.upper() in ("WELL", "WELL_NAME", "WELLNAME", "UWI")), None)
    depth_col = next((c for c, n in cols.items() if re.match(r"^(DEPTH|DEPT|MD)\b", n.upper())), None)
    if well_col is None or depth_col is None:
        raise StageBlocked(f"Workbook {path} sheet '{name}': no WELL or DEPTH column ({list(cols.values())})")
    factor = 0.3048 if re.search(r"\((ft|feet|f)\)", cols[depth_col], re.I) else 1.0
    src_col = next((c for c, n in cols.items() if n.upper() in ("SOURCE_FILE", "SOURCE", "FILE")), None)
    out = pd.DataFrame({"well": df[well_col].astype(str).str.strip().str.lower(),
                        "md_m": pd.to_numeric(df[depth_col], errors="coerce") * factor,
                        "source_file": df[src_col].astype(str) if src_col is not None else str(path.name)})
    for c, n in cols.items():
        if c in (well_col, depth_col, src_col):
            continue
        out[n] = pd.to_numeric(df[c], errors="coerce")
    out.attrs["sheet"] = name
    return out


def _inventory_for(ctx: StageContext, upstream_inv: pd.DataFrame | None) -> tuple[pd.DataFrame | None, str]:
    """Stage-00 curve inventory of THIS run, or a configured reference copy (sandbox runs)."""
    if upstream_inv is not None and len(upstream_inv) and "mnemonic" in upstream_inv:
        if upstream_inv["mnemonic"].notna().any():
            return upstream_inv, "this run: 00_inventory/tables/log_curve_inventory.csv"
    ref = cfg_get(ctx.cfg, "labels.reference_curve_inventory")
    if ref:
        p = resolve_path(ctx.cfg, ref)
        if p is not None and p.exists():
            ctx.add_input(p, "reference stage-00 curve inventory (another run of this pipeline)")
            return pd.read_csv(p), f"reference: {p}"
    return None, "none"


# =============================================================================== stage
def run(ctx: StageContext) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_style()
    steps = StepTracker(STEPS, ctx.stage)
    cfg = ctx.cfg
    reserved = [w.lower() for w in (cfg_get(cfg, "wells.reserved_reviewed", []) or [])]
    pol = effective_policy(cfg)
    ctx.parameters["label_policy"] = pol

    # ---------------------------------------------------------------- 1 upstream tables
    steps.next("upstream tables")
    run_dir = ctx.run.run_dir
    avail_p = run_dir / "00_inventory" / "tables" / "well_availability.csv"
    if not avail_p.exists():
        raise StageBlocked(f"Missing {avail_p} - run 00_inventory first")
    ctx.add_input(avail_p, "upstream:00_inventory/well_availability.csv")
    avail = pd.read_csv(avail_p)
    avail["well"] = avail["well"].astype(str).str.strip().str.casefold()
    wells = resolve_expected_wells(cfg, inventory=avail)
    if not wells:
        raise StageBlocked("Stage 00 well_availability.csv contains no wells and wells.expected is empty; cannot label.")
    dev_wells = [w for w in wells if w not in reserved]
    ctx.metric("n_wells_expected", len(wells))
    ctx.metric("well_population_source", "configuration" if cfg_get(cfg, "wells.expected", []) else "00_inventory/well_availability.csv")
    inv_p = run_dir / "00_inventory" / "tables" / "log_curve_inventory.csv"
    upstream_inv = pd.read_csv(inv_p) if inv_p.exists() else None
    if inv_p.exists():
        ctx.add_input(inv_p, "upstream:00_inventory/log_curve_inventory.csv")
    tz_p = run_dir / "01_well_ties" / "tables" / "tz_samples_all_wells.csv"
    tz = pd.read_csv(tz_p) if tz_p.exists() else None
    if tz is not None:
        ctx.add_input(tz_p, "upstream:01_well_ties/tz_samples_all_wells.csv")
    else:
        ctx.limit("01_well_ties/tables/tz_samples_all_wells.csv missing: TVDSS/TWT columns left empty")
    traj_p = run_dir / "01_well_ties" / "tables" / "trajectory_summary.csv"
    traj = pd.read_csv(traj_p) if traj_p.exists() else None
    tie_p = run_dir / "01_well_ties" / "tables" / "tie_summary.csv"
    tie = pd.read_csv(tie_p) if tie_p.exists() else None
    for p, role in ((traj_p, "trajectory_summary"), (tie_p, "tie_summary")):
        if p.exists():
            ctx.add_input(p, f"upstream:01_well_ties/{role}.csv")

    # ---------------------------------------------------------------- 2 read logs
    steps.next("read logs")
    frames, meta_rows, source_rows = _read_logs(ctx, avail, wells)
    if not frames:
        raise StageBlocked("No well-log data could be read (no LAS found and no usable "
                           "labels.combined_log_workbook) - nothing to label")
    logs = pd.concat(frames, ignore_index=True)
    meta = pd.DataFrame(meta_rows)
    sources = pd.DataFrame(source_rows)
    ctx.save_table(sources, "log_sources.csv", "Per well: where the log data came from (LAS or workbook)")
    ctx.save_table(meta, "curve_metadata.csv", "Curves used per well: family, mnemonic, unit, description")

    # ---------------------------------------------------------------- 3 verify source
    steps.next("verify source")
    inventory, inv_source = _inventory_for(ctx, upstream_inv)
    verify = _verify_source(logs, meta, inventory)
    ctx.save_table(verify, "source_verification.csv",
                   f"Log data read here vs stage-00 curve inventory ({inv_source}): counts, min, median, max")
    if len(verify):
        n_bad = int((~verify["match"]).sum())
        ctx.check("source_matches_stage00_inventory", n_bad == 0,
                  f"{len(verify) - n_bad}/{len(verify)} well-curve statistics identical ({inv_source})")
    else:
        ctx.warn("No stage-00 curve inventory available to verify the log source against")
    used_wb = sorted(sources.loc[sources["log_source"] == "workbook", "well"].tolist())
    if used_wb:
        ctx.limit(f"Logs for {len(used_wb)} wells read from the compiled workbook, not the original LAS files "
                  f"(verified against the stage-00 inventory statistics: see source_verification.csv): {used_wb}")
    depth_check = _depth_is_md(logs, traj)
    ctx.save_table(depth_check, "depth_index_check.csv",
                   "LAS depth range vs trajectory MD/TVD at TD: proves the depth index is MD for deviated wells")
    if len(depth_check):
        bad = depth_check[~depth_check["consistent_with_md"]]
        ctx.check("depth_index_is_md", bad.empty,
                  "every log is consistent with an MD index; deviated wells prove it (log base > TVD at TD)"
                  if bad.empty else f"inconsistent wells: {bad['well'].tolist()}")
        for _, r in depth_check[depth_check["log_below_listed_td_m"] > 1.0].iterrows():
            ctx.warn(f"{r['well']}: log continues {r['log_below_listed_td_m']:.1f} m below the listed TD / last "
                     "deviation station; TVDSS/TWT there rely on extrapolating the last survey direction")

    # per-sample thickness (MD) - every count below is in metres, not samples
    logs["md_step_m"] = np.concatenate([sample_thickness(g["md_m"].to_numpy())
                                        for _, g in logs.groupby("well", sort=False)])

    # ---------------------------------------------------------------- 4 recompute
    steps.next("recompute where possible")
    logs, methods = _recompute(ctx, logs, meta)
    ctx.save_table(methods, "method_comparison.csv",
                   "Independent Vsh/PHIT/SWT recomputation: APPLIED with comparison, or NOT_APPLIED with reason")
    present = {fam for fam in RAW_LOGS if f"{RAW_LOGS[fam]}" in logs and logs[RAW_LOGS[fam]].notna().any()}
    if not present:
        ctx.limit("No GR, RHOB, NPHI, DT or resistivity curve exists in any well: Vsh/PHIT/SWT cannot be "
                  "recomputed or independently checked; labels rest on the provider's interpreted curves")

    # ---------------------------------------------------------------- 5 curve QC
    steps.next("curve QC")
    for col in PROVIDER.values():
        if col not in logs:
            logs[col] = np.nan
    lab = qc_curves(logs, pol)
    qc_summary = _qc_summary(lab)
    ctx.save_table(qc_summary, "curve_qc_summary.csv",
                   "Per well and curve: MD thickness per QC flag (raw values retained in the main table)")

    # ---------------------------------------------------------------- 6 provider flag audit
    steps.next("provider flag audit")
    flag_audit, flag_inference = _provider_flag_audit(lab, pol)
    ctx.save_table(flag_audit, "provider_flag_audit.csv",
                   "PS_GIP / PS_OIP: thickness, implied Vsh/PHIT/SWT cutoffs, conflicts, fractional values")

    # ---------------------------------------------------------------- 7 labels
    steps.next("labels and classes")
    lab = assign_labels(lab, pol)

    # ---------------------------------------------------------------- 8 depth / time
    steps.next("depth/time attach")
    lab = _attach_depth_time(lab, tz, tie)
    lab["reserved_reviewed_well"] = lab["well"].isin(reserved)
    no_t = lab.loc[lab["twt_ms"].isna()].groupby("well")["md_step_m"].sum()
    ctx.metric("md_m_without_twt_by_well", {k: round(float(v), 1) for k, v in no_t.items()})
    if len(no_t):
        ctx.warn(f"Samples without TVDSS/TWT (outside the 01_well_ties MD range): "
                 f"{', '.join(f'{k} {v:.1f} m' for k, v in no_t.items())} - stage 03 cannot place them in time")

    # ---------------------------------------------------------------- 9 support + tables
    steps.next("support and tables")
    support = task_support(lab, pol, dev_wells)
    lab, demoted = demote_unsupported_classes(lab, support)
    if demoted:
        ctx.warn(f"Classes without multi-well support demoted to undetermined_sand: {demoted}")
    _write_tables(ctx, lab, pol, support, wells, dev_wells, flag_inference, sources, meta, methods)

    # ---------------------------------------------------------------- 10 figures
    steps.next("figures")
    for w in progress(wells, desc="petrophysical panels", unit="well"):
        d = lab[lab["well"] == w]
        if d.empty:
            fig = not_available_figure(f"{w}: petrophysical panel", [_no_data_reason(sources, w)])
            ctx.save_figure(fig, f"per_well/{w}_petrophysical_panel.png", f"{w}: NOT AVAILABLE", required=True)
            continue
        _figure_panel(ctx, w, d, pol, tz)
    _figure_class_distribution(ctx, lab, wells, reserved)
    _figure_crossplots(ctx, lab, pol)

    # ---------------------------------------------------------------- 11 summary
    steps.next("summary")
    _summarise(ctx, lab, pol, support, wells, dev_wells, flag_inference)
    steps.close()


# =============================================================================== step 2
def _read_logs(ctx: StageContext, avail: pd.DataFrame, wells: list[str]):
    lab_cfg = ctx.cfg.get("labels") or {}
    mode = str(lab_cfg.get("log_source", "auto")).lower()
    if mode not in ("auto", "las", "workbook"):
        raise StageBlocked(f"labels.log_source must be auto | las | workbook, not '{mode}'")
    wb_path = resolve_path(ctx.cfg, lab_cfg.get("combined_log_workbook")) if lab_cfg.get("combined_log_workbook") else None
    workbook = None
    frames, meta_rows, source_rows = [], [], []
    for w in progress(wells, desc="read logs", unit="well"):
        row = avail[avail["well"] == w]
        las_path = None
        if len(row) and isinstance(row.iloc[0].get("las_file"), str) and bool(row.iloc[0].get("las_found")):
            las_path = Path(row.iloc[0]["las_file"])
        if mode in ("auto", "las") and las_path is not None and las_path.exists():
            ctx.add_input(las_path, f"LAS {w}")
            las = read_las(las_path, w)
            mnems = [c["mnemonic"] for c in las.curves]
            frame = pd.DataFrame({"well": w, "md_m": las.depth_m, "source_file": las_path.name,
                                  "log_source": "las"})
            for fam, col in {**PROVIDER, **RAW_LOGS}.items():
                m = _pick_curve(mnems[1:], fam)
                if m is None:
                    continue
                vals = las.data[m].to_numpy(dtype=float)
                if las.null_value is not None:
                    vals = np.where(np.isclose(vals, las.null_value), np.nan, vals)
                frame[col] = vals
                c = next(c for c in las.curves if c["mnemonic"] == m)
                meta_rows.append({"well": w, "family": fam, "column": col, "mnemonic": m, "unit": c["unit"],
                                  "description": c["descr"], "source": "LAS header"})
            frames.append(frame)
            source_rows.append({"well": w, "log_source": "las", "path": str(las_path), "reader": las.reader,
                                "n_samples": len(frame), "md_top_m": float(np.nanmin(las.depth_m)),
                                "md_base_m": float(np.nanmax(las.depth_m)), "note": "; ".join(las.warnings)})
            continue
        if mode in ("auto", "workbook") and wb_path is not None and wb_path.exists():
            if workbook is None:
                ctx.add_input(wb_path, "compiled log workbook (fallback when LAS files are not reachable)")
                workbook = _read_workbook(wb_path, lab_cfg.get("workbook_sheet"))
            sub = workbook[workbook["well"] == w]
            if len(sub):
                curves = [c for c in sub.columns if c not in ("well", "md_m", "source_file")]
                frame = pd.DataFrame({"well": w, "md_m": sub["md_m"].to_numpy(float),
                                      "source_file": sub["source_file"].to_numpy(), "log_source": "workbook"})
                for fam, col in {**PROVIDER, **RAW_LOGS}.items():
                    m = _pick_curve(curves, fam)
                    if m is None:
                        continue
                    frame[col] = sub[m].to_numpy(dtype=float)
                    meta_rows.append({"well": w, "family": fam, "column": col, "mnemonic": m, "unit": "",
                                      "description": "", "source": f"workbook {wb_path.name}"})
                frames.append(frame)
                source_rows.append({"well": w, "log_source": "workbook", "path": str(wb_path),
                                    "reader": f"pandas.read_excel sheet '{workbook.attrs.get('sheet')}'",
                                    "n_samples": len(frame), "md_top_m": float(frame["md_m"].min()),
                                    "md_base_m": float(frame["md_m"].max()),
                                    "note": f"source_file in workbook: {sub['source_file'].iloc[0]}"})
                continue
        reason = ("LAS not found by 00_inventory" if las_path is None else f"LAS missing: {las_path}")
        if wb_path is not None:
            reason += f"; workbook {'has no rows for this well' if wb_path.exists() else 'not found: ' + str(wb_path)}"
        source_rows.append({"well": w, "log_source": "none", "path": "", "reader": "", "n_samples": 0,
                            "md_top_m": np.nan, "md_base_m": np.nan, "note": reason})
        ctx.limit(f"{w}: no log data ({reason})")
    return frames, meta_rows, source_rows


def _no_data_reason(sources: pd.DataFrame, w: str) -> str:
    r = sources[sources["well"] == w]
    return str(r.iloc[0]["note"]) if len(r) else "well not processed"


# =============================================================================== step 3
def _verify_source(logs: pd.DataFrame, meta: pd.DataFrame, inventory: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    if inventory is None or meta.empty:
        return pd.DataFrame(rows)
    inv = inventory.copy()
    inv["well"] = inv["well"].astype(str).str.lower()
    for _, m in meta.iterrows():
        ref = inv[(inv["well"] == m["well"]) & (inv["mnemonic"].astype(str).str.upper() == str(m["mnemonic"]).upper())]
        if ref.empty:
            rows.append({"well": m["well"], "mnemonic": m["mnemonic"], "match": False,
                         "detail": "curve not in stage-00 inventory"})
            continue
        ref = ref.iloc[0]
        got = _curve_stats(logs.loc[logs["well"] == m["well"], m["column"]].to_numpy(float))
        row = {"well": m["well"], "mnemonic": m["mnemonic"]}
        ok = True
        for k in COMPARE_STATS:
            a, b = got.get(k, np.nan), ref.get(k, np.nan)
            row[f"{k}_here"], row[f"{k}_stage00"] = a, b
            same = (pd.isna(a) and pd.isna(b)) or (not pd.isna(a) and not pd.isna(b)
                                                   and np.isclose(float(a), float(b), rtol=1e-7, atol=1e-9))
            ok &= bool(same)
        row["match"] = ok
        row["detail"] = "identical" if ok else "statistics differ"
        rows.append(row)
    return pd.DataFrame(rows)


def _depth_is_md(logs: pd.DataFrame, traj: pd.DataFrame | None) -> pd.DataFrame:
    rows = []
    if traj is None:
        return pd.DataFrame(rows)
    for w, g in logs.groupby("well", sort=False):
        t = traj[traj["well"].astype(str).str.lower() == w]
        if t.empty:
            continue
        t = t.iloc[0]
        base = float(g["md_m"].max())
        md_td, tvd_td = float(t["md_td_m"]), float(t["tvd_rt_td_m"])
        proves_md = base > tvd_td + 1.0          # a TVD-indexed log cannot be deeper than TVD at TD
        within_td = base <= md_td + 1.0
        below = max(0.0, base - md_td)
        rows.append({"well": w, "log_base_m": base, "md_td_m": md_td, "tvd_rt_td_m": tvd_td,
                     "log_below_listed_td_m": below,
                     "consistent_with_md": bool(proves_md or within_td),
                     "evidence": ("log base deeper than TVD at TD -> index must be MD" if proves_md else
                                  "near-vertical: MD ~ TVD, both interpretations fit")
                     + (f"; log extends {below:.1f} m below the listed TD / last survey station "
                        "(trajectory extrapolated there)" if below > 1.0 else "")})
    return pd.DataFrame(rows)


# =============================================================================== step 4
def _recompute(ctx: StageContext, logs: pd.DataFrame, meta: pd.DataFrame):
    """Apply textbook methods only where curves AND documented parameters exist."""
    rc = cfg_get(ctx.cfg, "labels.recompute", {}) or {}
    rows = []
    out = logs.copy()

    def compare(w, method, target_col, values, provider_col, note):
        g = out["well"] == w
        prov = out.loc[g, provider_col].to_numpy(float) if provider_col in out else np.full(g.sum(), np.nan)
        ok = np.isfinite(values) & np.isfinite(prov)
        r = {"well": w, "method": method, "status": "APPLIED", "reason": note,
             "n_compared": int(ok.sum())}
        if ok.sum() > 2:
            diff = values[ok] - prov[ok]
            r.update({"mae_vs_provider": float(np.mean(np.abs(diff))), "bias_vs_provider": float(np.mean(diff)),
                      "pearson_r_vs_provider": float(np.corrcoef(values[ok], prov[ok])[0, 1])})
        out.loc[g, target_col] = values
        return r

    for w, g in out.groupby("well", sort=False):
        have = {fam for fam, col in RAW_LOGS.items() if col in out and g[col].notna().any()}

        def na(method, curves, reason):
            rows.append({"well": w, "method": method, "status": "NOT_APPLIED", "required_curves": curves,
                         "reason": reason})

        # ---- Vsh from GR
        if "GR" not in have:
            na("vsh_gr_linear", "GR", "curve GR absent from this well's logs")
            na("vsh_gr_larionov", "GR", "curve GR absent from this well's logs")
        else:
            gr = g["gr_raw"].to_numpy(float)
            mode = rc.get("gr_endpoint_mode")
            if mode == "fixed" and rc.get("gr_clean") is not None and rc.get("gr_shale") is not None:
                clean, shale = float(rc["gr_clean"]), float(rc["gr_shale"])
            elif mode == "per_well_percentile":
                clean = float(np.nanpercentile(gr, float(rc.get("gr_clean_pct", 5))))
                shale = float(np.nanpercentile(gr, float(rc.get("gr_shale_pct", 95))))
            else:
                clean = shale = None
            if clean is None or not shale > clean:
                na("vsh_gr_linear", "GR", "GR present but labels.recompute.gr_endpoint_mode / endpoints not configured")
                na("vsh_gr_larionov", "GR", "GR present but endpoints not configured")
            else:
                igr = gamma_ray_index(gr, clean, shale)
                note = f"IGR endpoints {mode}: GR_clean={clean:.1f}, GR_shale={shale:.1f} API ({rc.get('provenance', 'no provenance note')})"
                rows.append(compare(w, "vsh_gr_linear", "vsh_gr_linear", vsh_linear(igr), "vsh_raw", note))
                var = rc.get("larionov_variant")
                if var in ("tertiary", "older"):
                    f = vsh_larionov_tertiary if var == "tertiary" else vsh_larionov_older
                    rows.append(compare(w, "vsh_gr_larionov", "vsh_gr_larionov", f(igr), "vsh_raw",
                                        f"Larionov {var}; {note}"))
                else:
                    na("vsh_gr_larionov", "GR", "labels.recompute.larionov_variant not set (tertiary | older)")
        # ---- porosity from density (and neutron)
        rho_ma, rho_fl = rc.get("rho_matrix"), rc.get("rho_fluid")
        if "RHOB" not in have:
            na("phit_density", "RHOB", "curve RHOB absent from this well's logs")
            na("phit_neutron_density", "RHOB+NPHI", "curves RHOB/NPHI absent from this well's logs")
        elif rho_ma is None or rho_fl is None:
            na("phit_density", "RHOB", "RHOB present but labels.recompute.rho_matrix / rho_fluid not configured")
            na("phit_neutron_density", "RHOB+NPHI", "matrix/fluid densities not configured")
        else:
            rhob = g["rhob_raw"].to_numpy(float)
            rhob = np.where(np.nanmedian(rhob) > 100, rhob / 1000.0, rhob)       # kg/m3 -> g/cc
            phid = density_porosity(rhob, float(rho_ma), float(rho_fl))
            note = (f"rho_ma={rho_ma}, rho_fl={rho_fl} g/cc ({rc.get('provenance', 'no provenance note')}); "
                    "density porosity is NOT relabelled PHIT - comparison only")
            rows.append(compare(w, "phit_density", "phit_density", phid, "phit_raw", note))
            if "NPHI" in have:
                nphi = g["nphi_raw"].to_numpy(float)
                nphi = np.where(np.nanmedian(nphi) > 1.5, nphi / 100.0, nphi)    # percent -> fraction
                rows.append(compare(w, "phit_neutron_density", "phit_neutron_density",
                                    neutron_density_porosity(phid, nphi), "phit_raw", note))
            else:
                na("phit_neutron_density", "NPHI", "curve NPHI absent from this well's logs")
        # ---- Archie
        if "RT_DEEP" not in have:
            na("swt_archie", "RT_DEEP", "deep resistivity curve absent from this well's logs")
        elif any(rc.get(k) is None for k in ("rw", "a", "m", "n")):
            na("swt_archie", "RT_DEEP", "resistivity present but Rw, a, m, n not configured (no defaults are assumed)")
        else:
            sw = archie_sw(g["rt_raw"].to_numpy(float), g["phit_raw"].to_numpy(float), float(rc["rw"]),
                           float(rc["a"]), float(rc["m"]), float(rc["n"]))
            rows.append(compare(w, "swt_archie", "swt_archie", sw, "swt_raw",
                                f"Archie Rw={rc['rw']}, a={rc['a']}, m={rc['m']}, n={rc['n']} with provider "
                                f"PHIT ({rc.get('provenance', 'no provenance note')}); clean-sand model"))
    return out, pd.DataFrame(rows)


# =============================================================================== step 5-6
def _qc_summary(lab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    usable = ("ok", "valid_at_lower_limit", "valid_at_upper_limit")
    for w, g in lab.groupby("well", sort=False):
        for curve in ("vsh", "phit", "swt"):
            stats = g.groupby(f"{curve}_flag")["md_step_m"].agg(["sum", "size"])
            for flag, r in stats.iterrows():
                rows.append({"well": w, "curve": curve.upper(), "flag": flag, "md_thickness_m": float(r["sum"]),
                             "n_samples": int(r["size"]), "usable_as_target": flag in usable})
    return pd.DataFrame(rows)


def _provider_flag_audit(lab: pd.DataFrame, pol: dict):
    rows = []
    thr = pol["fluid"]["provider_flag_threshold"]
    for w, g in lab.groupby("well", sort=False):
        evaluated = g["provider_pay_flag"] != "not_evaluated"
        for fluid, col in (("gas", "gas_flag_raw"), ("oil", "oil_flag_raw")):
            raw = g[col].to_numpy(float)
            on = np.isfinite(raw) & (raw >= thr)
            s = g[on]
            r = {"well": w, "flag": f"PS_{'GIP' if fluid == 'gas' else 'OIP'}",
                 "evaluated_md_m": float(g.loc[evaluated, "md_step_m"].sum()),
                 "set_md_m": float(s["md_step_m"].sum()), "n_set": int(on.sum()),
                 "n_fractional_values": int((np.isfinite(raw) & (raw > 0) & (raw < 1)).sum()),
                 "n_both_flags_set": int((g["provider_pay_flag"] == "gas_and_oil_conflict").sum())}
            if len(s):
                r.update({"md_first_m": float(s["md_m"].min()), "md_last_m": float(s["md_m"].max()),
                          "vsh_max": float(s["vsh_raw"].max()), "vsh_p95": float(s["vsh_raw"].quantile(0.95)),
                          "phit_min": float(s["phit_raw"].min()), "phit_p05": float(s["phit_raw"].quantile(0.05)),
                          "swt_max": float(s["swt_raw"].max()), "swt_p95": float(s["swt_raw"].quantile(0.95)),
                          "fraction_swt_le_hc_cutoff": float((s["swt_raw"] <= pol["fluid"]["swt_hc_potential_max"]).mean())})
            rows.append(r)
    audit = pd.DataFrame(rows)
    flagged = lab[lab["provider_pay_flag"].isin(["gas", "oil"])]
    inference = {"note": ("Observed from the data, not documented by the provider: pay-flagged samples sit inside "
                          "these bounds, i.e. the flags look like net-pay cutoffs on the same VSH/PHIT/SWT curves. "
                          "The oil-vs-gas attribution has no documented basis.")}
    if len(flagged):
        inference.update({"n_flagged_samples": int(len(flagged)),
                          "vsh_p99_of_flagged": float(flagged["vsh_raw"].quantile(0.99)),
                          "phit_p01_of_flagged": float(flagged["phit_raw"].quantile(0.01)),
                          "swt_p99_of_flagged": float(flagged["swt_raw"].quantile(0.99)),
                          "n_wells_gas_flag": int(audit.loc[(audit["flag"] == "PS_GIP") & (audit["n_set"] > 0), "well"].nunique()),
                          "n_wells_oil_flag": int(audit.loc[(audit["flag"] == "PS_OIP") & (audit["n_set"] > 0), "well"].nunique()),
                          "n_samples_both_flags": int((lab["provider_pay_flag"] == "gas_and_oil_conflict").sum()),
                          "n_fractional_flag_samples": int(lab["provider_flag_fractional"].sum())})
    return audit, inference


# =============================================================================== step 8
def _attach_depth_time(lab: pd.DataFrame, tz: pd.DataFrame | None, tie: pd.DataFrame | None) -> pd.DataFrame:
    out = lab.copy()
    out["tvdss_m"] = np.nan
    out["twt_ms"] = np.nan
    out["tz_strong_extrapolation"] = False
    out["tie_status"] = ""
    if tz is not None:
        tz = tz.copy()
        tz["well"] = tz["well"].astype(str).str.lower()
        for w, idx in out.groupby("well", sort=False).groups.items():
            t = tz[tz["well"] == w].sort_values("md_m")
            if t.empty:
                continue
            md = out.loc[idx, "md_m"].to_numpy(float)
            inside = (md >= t["md_m"].iloc[0]) & (md <= t["md_m"].iloc[-1])
            for col in ("tvdss_m", "twt_ms"):
                if col in t:
                    v = np.interp(md, t["md_m"].to_numpy(float), t[col].to_numpy(float))
                    out.loc[idx, col] = np.where(inside, v, np.nan)
            if "tz_strong_extrapolation" in t:
                flag = t["tz_strong_extrapolation"].astype(str).str.lower().eq("true").to_numpy(float)
                out.loc[idx, "tz_strong_extrapolation"] = np.interp(md, t["md_m"].to_numpy(float), flag) > 0
    if tie is not None and "tie_status" in tie:
        m = dict(zip(tie["well"].astype(str).str.lower(), tie["tie_status"]))
        out["tie_status"] = out["well"].map(m).fillna("")
    return out


# =============================================================================== step 9
MAIN_COLUMNS = [
    "well", "md_m", "md_step_m", "tvdss_m", "twt_ms", "tz_strong_extrapolation", "tie_status",
    "reserved_reviewed_well", "log_source", "source_file",
    "vsh_raw", "phit_raw", "swt_raw", "gas_flag_raw", "oil_flag_raw",
    "artifact_interval", "vsh_valid", "vsh_flag", "phit_valid", "phit_flag", "swt_valid", "swt_flag",
    "provider_pay_flag", "provider_flag_fractional",
    "lithology_label", "lithology_reason", "fluid_label", "fluid_reason", "fluid_evidence_type",
    "class_id", "class_name", "class_status", "confidence_grade", "label_source", "lith_vsh50_baseline",
    "vsh_fraction", "phit_fraction", "swt_fraction", "vsh_mask", "phit_mask", "swt_mask",
    "y_lithology", "y_sand_fluid",
]
OPTIONAL_COLUMNS = ["gr_raw", "rhob_raw", "nphi_raw", "rt_raw", "dt_raw", "vsh_gr_linear", "vsh_gr_larionov",
                    "phit_density", "phit_neutron_density", "swt_archie"]
UNITS = {"md_m": "m (MD below RT)", "md_step_m": "m", "tvdss_m": "m below MSL (positive down)",
         "twt_ms": "ms two-way time (01_well_ties T-Z)", "vsh_raw": "v/v", "phit_raw": "v/v", "swt_raw": "v/v",
         "vsh_fraction": "v/v", "phit_fraction": "v/v", "swt_fraction": "v/v",
         "gas_flag_raw": "provider flag (0/1)", "oil_flag_raw": "provider flag (0/1)"}


def _mask_reasons(g: pd.DataFrame, tgt: str, mask: np.ndarray) -> pd.Series:
    """Why a target sample is not usable: the QC flag, or 'excluded_in_shale' for SWT in shale."""
    reason = g[f"{tgt}_flag"].astype(object).copy()
    reason[g[f"{tgt}_valid"].to_numpy(bool) & ~mask] = "excluded_in_shale"
    return reason[~mask].value_counts()


def _rounded(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind == "f":
            out[c] = out[c].round(6 if c in ("md_m", "tvdss_m") else 5)
    return out


def _write_tables(ctx, lab, pol, support, wells, dev_wells, flag_inference, sources, meta, methods):
    cols = MAIN_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in lab]
    main = _rounded(lab[cols])
    ctx.save_table(main, "interpreted_logs_all_wells.csv",
                   "One row per log sample: raw provider curves, QC flags, labels, classes, grades, masked targets",
                   units=UNITS, required=True)
    for w in progress(wells, desc="per-well CSV", unit="well"):
        d = main[main["well"] == w]
        if d.empty:
            d = pd.DataFrame([{"well": w, "not_available_reason": "no log data read for this well"}])
        ctx.save_table(d, f"per_well/{w}.csv", f"{w}: interpreted logs and labels", units=UNITS, required=True)

    # ---- class definition table
    thick = lab.groupby("class_name")["md_step_m"].sum()
    n = lab.groupby("class_name").size()
    status = {r["task"].split(":", 1)[1]: r for _, r in support.iterrows() if r["task"].startswith("class:")}
    lp, fp = pol["lithology"], pol["fluid"]
    rule = {
        "shale": f"provider VSH valid and >= {lp['vsh_shale_min']}",
        "brine_sand": (f"sand (VSH <= {lp['vsh_sand_max']}, PHIT >= {lp['phit_sand_min']}) + provider SWT >= "
                       f"{fp['swt_water_min']} + no provider pay flag" if fp["brine_from_interpreted_swt"]
                       else "not assigned (brine_from_interpreted_swt = false)"),
        "oil_sand": ("sand + hydrocarbon_potential + PS_OIP (reviewed by "
                     f"{fp.get('provider_flags_reviewed_by')})" if fp.get("provider_flags_accepted")
                     else "NOT ASSIGNED - no independent oil evidence"),
        "gas_sand": ("sand + hydrocarbon_potential + PS_GIP (reviewed by "
                     f"{fp.get('provider_flags_reviewed_by')})" if fp.get("provider_flags_accepted")
                     else "NOT ASSIGNED - no independent gas evidence"),
        "undetermined_sand": "sand with hydrocarbon_potential, unresolved or conflicting fluid evidence (abstention)",
        "other_rock": "NOT ASSIGNED - no lithology log or core to confirm",
        "unlabelled": "heterolithic dead band, placeholders/artifacts, tight non-shale, missing VSH",
    }
    trainable = {"shale": "yes (lithology + class)", "brine_sand": "yes (provisional)",
                 "undetermined_sand": "as sand for lithology; abstain for fluid",
                 "oil_sand": "no", "gas_sand": "no", "other_rock": "no", "unlabelled": "no (inference only)"}
    if fp.get("provider_flags_accepted"):
        for c in ("oil_sand", "gas_sand"):
            if status.get(c, {}).get("status") != "TASK_UNSUPPORTED":
                trainable[c] = "yes (provider flags, reviewed)"
    rows = []
    for cid, name, meaning, evidence in CLASS_TABLE:
        s = status.get(name)
        rows.append({"class_id": cid, "class_name": name, "geological_meaning": meaning,
                     "minimum_evidence_required": evidence, "rule_in_this_run": rule[name],
                     "trainable_in_this_run": trainable[name],
                     "task_status": s["status"] if s is not None else "",
                     "n_samples_all_wells": int(n.get(name, 0)), "md_thickness_m_all_wells": float(thick.get(name, 0.0)),
                     "n_development_wells_supporting": int(s["n_wells_supporting"]) if s is not None else np.nan,
                     "class_map_version": pol["class_map_version"], "plot_color": CLASS_COLORS[name]})
    ctx.save_table(pd.DataFrame(rows), "class_definition_table.csv",
                   "Class IDs, meaning, evidence required vs rule used in this run, task status", required=True)

    # ---- class counts by well (long, every class for every well)
    rows = []
    for w in wells:
        g = lab[lab["well"] == w]
        logged = float(g["md_step_m"].sum())
        tvd = np.abs(np.gradient(g["tvdss_m"].to_numpy(float))) if len(g) > 1 else np.zeros(len(g))
        for cid, name, _, _ in CLASS_TABLE:
            m = (g["class_name"] == name).to_numpy()
            t = float(g.loc[m, "md_step_m"].sum())
            rows.append({"well": w, "class_id": cid, "class_name": name, "n_samples": int(m.sum()),
                         "md_thickness_m": t, "tvd_thickness_m": float(np.nansum(tvd[m])) if len(tvd) else 0.0,
                         "fraction_of_logged_md": t / logged if logged else np.nan,
                         "reserved_reviewed_well": w in (cfg_get(ctx.cfg, "wells.reserved_reviewed", []) or [])})
    ctx.save_table(pd.DataFrame(rows), "class_counts_by_well.csv",
                   "Samples, MD and TVD thickness per class and well (no-1 reserved, reported not trained)",
                   units={"md_thickness_m": "m MD", "tvd_thickness_m": "m TVD"}, required=True)

    # ---- property coverage
    rows = []
    for w in wells + ["ALL_DEVELOPMENT_WELLS"]:
        g = lab[lab["well"].isin(dev_wells)] if w == "ALL_DEVELOPMENT_WELLS" else lab[lab["well"] == w]
        for tgt in ("vsh", "phit", "swt"):
            mask = g[f"{tgt}_mask"].to_numpy(bool) if len(g) else np.zeros(0, bool)
            vals = g[f"{tgt}_fraction"].to_numpy(float)[mask] if len(g) else np.zeros(0)
            raw = g[f"{tgt}_raw"].to_numpy(float) if len(g) else np.zeros(0)
            r = {"well": w, "target": f"{tgt}_fraction", "n_samples": int(len(g)), "n_valid": int(mask.sum()),
                 "fraction_valid": float(mask.mean()) if len(g) else np.nan,
                 "valid_md_m": float(g.loc[mask, "md_step_m"].sum()) if len(g) else 0.0,
                 "n_raw_present_but_masked": int((np.isfinite(raw) & ~mask).sum()),
                 "masking_breakdown": "; ".join(f"{k}={v}" for k, v in _mask_reasons(g, tgt, mask).items())
                 if len(g) else "",
                 "n_valid_at_upper_limit": int((g.loc[mask, f"{tgt}_flag"] == "valid_at_upper_limit").sum()) if len(g) else 0}
            if vals.size:
                q = np.percentile(vals, [5, 50, 95])
                r.update({"min": float(vals.min()), "p05": float(q[0]), "p50": float(q[1]), "p95": float(q[2]),
                          "max": float(vals.max())})
            rows.append(r)
    ctx.save_table(pd.DataFrame(rows), "property_coverage.csv",
                   "Valid-target coverage per well and property, with masking reasons", required=True)
    ctx.save_table(support, "task_support.csv",
                   "Which learning tasks the labels support (TASK_UNSUPPORTED where evidence is missing)", required=True)
    ctx.save_table(cutoff_sensitivity(lab, pol, dev_wells), "cutoff_sensitivity.csv",
                   "Sand/shale/heterolithic MD thickness for alternative Vsh cutoff pairs (development wells)")

    # ---- label policy (the contract for every later stage)
    policy_doc = {
        "class_map_version": pol["class_map_version"], "policy_version": pol["policy_version"],
        "status": "PROVISIONAL - not geologist-approved", "effective_policy": pol,
        "reason_codes": reason_codes(pol),
        "classes": [{"class_id": c, "class_name": n, "meaning": m, "minimum_evidence": e}
                    for c, n, m, e in CLASS_TABLE],
        "unit_conventions": {
            "md_m": "measured depth below rotary table (m), the LAS depth index (verified: depth_index_check.csv)",
            "tvdss_m": "true vertical depth below mean sea level (m, positive down), from 01_well_ties trajectories",
            "twt_ms": "two-way time (ms) from the T-Z function chosen in 01_well_ties (provisional for most wells)",
            "vsh/phit/swt": "fractions v/v in [0,1]; PHIT = TOTAL porosity, SWT = TOTAL water saturation",
            "provider flags": f"PS_GIP / PS_OIP >= {pol['fluid']['provider_flag_threshold']} counts as set; NaN = not evaluated",
            "thickness": "md_step_m = half distance to neighbouring samples (MD); TVD thickness in class_counts_by_well.csv"},
        "curve_definitions": meta.drop_duplicates(["family", "mnemonic", "description"])[
            ["family", "mnemonic", "unit", "description"]].to_dict("records") if len(meta) else [],
        "provider_flag_inference": flag_inference,
        "independent_evidence_available": {k: cfg_get(ctx.cfg, f"auxiliary_inputs_status.{k}", "not recorded")
                                           for k in ("fluid_tests_dst_mdt_rft", "formation_tops")},
        "task_support": support.to_dict("records"),
        "log_sources": sources.to_dict("records"),
        "recompute_methods": methods.to_dict("records"),
        "targets": {"vsh_fraction": "provider VSH where vsh_mask", "phit_fraction": "provider PHIT where phit_mask",
                    "swt_fraction": ("provider SWT where swt_mask (excludes shale: Sw is a model default there)"
                                     if pol["targets"]["swt_exclude_shale"] else "provider SWT where swt_mask"),
                    "y_lithology": "1 sand, 0 shale, -1 not a training sample",
                    "y_sand_fluid": "0 water_bearing, 1 hydrocarbon_potential, -1 not a training sample",
                    "class_id": "flat class; use only classes whose task_status is not TASK_UNSUPPORTED"},
        "never": ["impute a masked target", "infer oil or gas from low Sw alone", "use labels as predictors",
                  "train on unlabelled samples", "fit supervised models on reserved well(s) "
                  + ", ".join(cfg_get(ctx.cfg, 'wells.reserved_reviewed', []) or [])],
    }
    ctx.save_json(policy_doc, "label_policy.json", "Label contract: cutoffs, unit conventions, reason codes, "
                  "class map, evidence inventory, task support", required=True)


# =============================================================================== step 10: figures
def _strip(ax, md, labels, order, colors, title):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    codes = pd.Categorical(labels, categories=order).codes.astype(float)
    codes[codes < 0] = np.nan
    cmap = ListedColormap([colors[k] for k in order])
    half = 0.5 * float(np.nanmedian(np.diff(md))) if len(md) > 1 else 0.1
    ax.imshow(codes[:, None], aspect="auto", cmap=cmap, vmin=-0.5, vmax=len(order) - 0.5,
              extent=[0, 1, md[-1] + half, md[0] - half], interpolation="nearest")
    ax.set_xticks([])
    ax.set_title(title, fontsize=9)
    ax.grid(False)
    plt.setp(ax.get_yticklabels(), visible=False)


def _na_track(ax, title, text):
    import matplotlib.pyplot as plt

    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_facecolor("#f3f2ef")
    ax.grid(False)
    ax.text(0.5, 0.5, text, rotation=90, ha="center", va="center", transform=ax.transAxes,
            fontsize=10, fontweight="bold", color=STATUS["critical"])
    ax.set_title(title, fontsize=9)
    plt.setp(ax.get_yticklabels(), visible=False)


def _figure_panel(ctx, w, d, pol, tz):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    d = d.sort_values("md_m")
    md = d["md_m"].to_numpy(float)
    lp, fp = pol["lithology"], pol["fluid"]
    fig = plt.figure(figsize=(19, 13))
    widths = [1.5, 0.75, 0.75, 1.3, 1.3, 0.75, 0.42, 0.42, 0.42]
    gs = fig.add_gridspec(1, len(widths), width_ratios=widths, wspace=0.10, left=0.055, right=0.94,
                          top=0.90, bottom=0.13)
    ax0 = fig.add_subplot(gs[0, 0])
    axes = [ax0] + [fig.add_subplot(gs[0, i], sharey=ax0) for i in range(1, len(widths))]

    # ---- track 1: VSH (+ GR if it exists)
    a = axes[0]
    vraw, vok = d["vsh_raw"].to_numpy(float), d["vsh_valid"].to_numpy(bool)
    a.fill_betweenx(md, 0, np.where(vok, vraw, np.nan), color=CLASS_COLORS["shale"], alpha=0.35, lw=0)
    a.plot(np.where(vok, vraw, np.nan), md, color="#0b0b0b", lw=0.7, label="VSH valid")
    a.plot(np.where(~vok & np.isfinite(vraw), vraw, np.nan), md, color=STATUS["critical"], lw=0.9,
           label="VSH masked (placeholder/artifact)")
    for x, lbl in ((lp["vsh_sand_max"], "sand max"), (lp["vsh_shale_min"], "shale min")):
        a.axvline(x, color=TEXT_2, ls="--", lw=0.9)
        a.text(x + 0.01, 0.997, f"{lbl} {x:g}", transform=a.get_xaxis_transform(), ha="left", va="top",
               fontsize=7, rotation=90, color=TEXT_2)
    a.set_xlim(0, 1)
    a.set_xlabel("VSH (v/v)")
    gr_note = "GR: in LAS" if "gr_raw" in d and d["gr_raw"].notna().any() else "GR: NOT IN LAS"
    a.set_title(f"VSH (provider)\n{gr_note}", fontsize=9)
    a.set_ylabel("MD below RT (m)")
    a.legend(loc="lower left", fontsize=7)
    if "gr_raw" in d and d["gr_raw"].notna().any():
        tw = a.twiny()
        tw.plot(d["gr_raw"], md, color=SERIES[2], lw=0.6)
        tw.set_xlabel("GR (API)", color=SERIES[2], fontsize=8)
        tw.grid(False)

    # ---- tracks 2-3: RHOB/NPHI and resistivity (real curves or NOT AVAILABLE)
    if "rhob_raw" in d and d["rhob_raw"].notna().any():
        axes[1].plot(d["rhob_raw"], md, color=SERIES[7], lw=0.6)
        axes[1].set_title("RHOB", fontsize=9)
        plt.setp(axes[1].get_yticklabels(), visible=False)
    else:
        _na_track(axes[1], "RHOB / NPHI", "RHOB / NPHI  NOT AVAILABLE\n(curves absent from LAS)")
    if "rt_raw" in d and d["rt_raw"].notna().any():
        axes[2].semilogx(d["rt_raw"], md, color=SERIES[6], lw=0.6)
        axes[2].set_title("Deep resistivity", fontsize=9)
        plt.setp(axes[2].get_yticklabels(), visible=False)
    else:
        _na_track(axes[2], "Resistivity", "RESISTIVITY  NOT AVAILABLE\n(curve absent from LAS)")

    # ---- track 4: PHIT
    a = axes[3]
    praw, pok = d["phit_raw"].to_numpy(float), d["phit_valid"].to_numpy(bool)
    a.plot(np.where(pok, praw, np.nan), md, color="#0b0b0b", lw=0.7, label="PHIT valid")
    bad = ~pok & np.isfinite(praw)
    a.scatter(np.clip(praw[bad], 0, 0.5), md[bad], s=2, color=STATUS["critical"], label="PHIT masked", zorder=3)
    a.axvline(pol["bounds"]["phit"][1], color=STATUS["critical"], ls=":", lw=0.9)
    a.set_xlim(0, 0.5)
    a.set_xticks([0, 0.1, 0.2, 0.3, 0.4])
    a.set_xlabel("PHIT (v/v)")
    n_oob = int((praw > pol["bounds"]["phit"][1]).sum())
    a.set_title(f"PHIT total (provider)\n{n_oob} samples > {pol['bounds']['phit'][1]} masked", fontsize=9)
    a.legend(loc="lower right", fontsize=7)
    plt.setp(a.get_yticklabels(), visible=False)

    # ---- track 5: SWT
    a = axes[4]
    sraw, sok = d["swt_raw"].to_numpy(float), d["swt_valid"].to_numpy(bool)
    a.plot(np.where(sok, sraw, np.nan), md, color=SERIES[0], lw=0.7, label="SWT valid")
    a.plot(np.where(~sok & np.isfinite(sraw), sraw, np.nan), md, color=NEUTRAL, lw=0.5,
           label="SWT not usable (low/invalid PHIT, artifact)")
    a.axvline(fp["swt_hc_potential_max"], color=FLUID_COLORS["hydrocarbon_potential"], ls="--", lw=0.9)
    a.axvline(fp["swt_water_min"], color=FLUID_COLORS["water_bearing"], ls="--", lw=0.9)
    a.set_xlim(0, 1.02)
    a.set_xlabel("SWT (v/v)")
    a.set_title(f"SWT total (provider)\ncutoffs {fp['swt_hc_potential_max']:g} / {fp['swt_water_min']:g}", fontsize=9)
    a.legend(loc="lower left", fontsize=7)
    plt.setp(a.get_yticklabels(), visible=False)

    # ---- track 6: provider pay flags
    a = axes[5]
    pay = d["provider_pay_flag"].to_numpy(object)
    a.fill_betweenx(md, 0, 0.5, where=np.isin(pay, ["gas", "gas_and_oil_conflict"]),
                    color=CLASS_COLORS["gas_sand"], lw=0, step="mid")
    a.fill_betweenx(md, 0.5, 1.0, where=np.isin(pay, ["oil", "gas_and_oil_conflict"]),
                    color=CLASS_COLORS["oil_sand"], lw=0, step="mid")
    a.fill_betweenx(md, 0, 1, where=pay == "not_evaluated", facecolor="none", edgecolor=NEUTRAL,
                    hatch="///", lw=0, step="mid")
    a.set_xlim(0, 1)
    a.set_xticks([0.25, 0.75])
    a.set_xticklabels(["GIP", "OIP"], fontsize=8)
    a.set_title("Provider pay flags\n(screening only)", fontsize=9)
    a.grid(False)
    plt.setp(a.get_yticklabels(), visible=False)

    # ---- tracks 7-9: lithology, class, confidence strips
    _strip(axes[6], md, d["lithology_label"].to_numpy(object), LITH_ORDER, LITH_COLORS, "Lithology")
    _strip(axes[7], md, d["class_name"].to_numpy(object), CLASS_ORDER, CLASS_COLORS, "Class")
    _strip(axes[8], md, d["confidence_grade"].to_numpy(object), GRADE_ORDER, GRADE_COLORS, "Grade")
    axes[0].set_ylim(md.max() + 1, md.min() - 1)

    # ---- secondary TVDSS axis from the 01_well_ties trajectory
    t = d[["md_m", "tvdss_m"]].dropna()
    if len(t) > 2 and np.all(np.diff(t["tvdss_m"].to_numpy()) > 0):
        mdv, zv = t["md_m"].to_numpy(float), t["tvdss_m"].to_numpy(float)
        from matplotlib.ticker import MaxNLocator

        sec = axes[8].secondary_yaxis("right", functions=(lambda x: np.interp(x, mdv, zv),
                                                          lambda z: np.interp(z, zv, mdv)))
        ticks = MaxNLocator(10).tick_values(zv.min(), zv.max())
        sec.set_yticks([tk for tk in ticks if zv.min() <= tk <= zv.max()])
        sec.set_ylabel("TVDSS (m, from 01_well_ties)")

    # ---- legends + titles
    handles = ([Patch(color=CLASS_COLORS[c], label=c) for c in CLASS_ORDER]
               + [Patch(color=LITH_COLORS[c], label=f"lith: {c}") for c in LITH_ORDER]
               + [Patch(color=GRADE_COLORS[g], label=f"grade {g}") for g in GRADE_ORDER]
               + [Patch(facecolor="none", edgecolor=NEUTRAL, hatch="///", label="pay flag not evaluated")])
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8, bbox_to_anchor=(0.5, 0.035))
    th = d.groupby("class_name")["md_step_m"].sum()
    summary = ", ".join(f"{c} {th.get(c, 0):.0f} m" for c in CLASS_ORDER if th.get(c, 0) > 0)
    tie = d["tie_status"].iloc[0] if "tie_status" in d and len(d) else ""
    reserved = " | RESERVED (previously reviewed) well" if bool(d["reserved_reviewed_well"].iloc[0]) else ""
    fig.suptitle(f"{w} - provider interpreted logs, QC masks and labels (EXPLORATORY; T-Z: {tie}){reserved}",
                 fontsize=13, fontweight="bold", y=0.975)
    fig.text(0.5, 0.943, summary, ha="center", fontsize=9.5, color=TEXT_2)
    footer(fig, (f"Labels {pol['policy_version']}: sand VSH <= {lp['vsh_sand_max']} & PHIT >= {lp['phit_sand_min']}; "
                 f"shale VSH >= {lp['vsh_shale_min']}; between = heterolithic (unlabelled). Fluid from provider SWT; "
                 "oil/gas TASK_UNSUPPORTED (no independent evidence); provider flags are screening only. Grades: "
                 "B interpreted log far from cutoffs, C near cutoffs/uncorroborated, U unlabelled (no grade A: "
                 "no core or tests). Cutoffs are baseline proxies, NOT geologist-approved."), y=0.004)
    ctx.save_figure(fig, f"per_well/{w}_petrophysical_panel.png",
                    f"{w}: VSH, RHOB/NPHI, resistivity, PHIT, SWT, pay flags, lithology, class and grade tracks",
                    required=True, multipanel=True)


def _figure_class_distribution(ctx, lab, wells, reserved):
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(19, 8.5), gridspec_kw={"width_ratios": [1.35, 1]})
    th = lab.groupby(["well", "class_name"])["md_step_m"].sum().unstack(fill_value=0.0).reindex(wells).fillna(0.0)
    y = np.arange(len(wells))[::-1]
    left = np.zeros(len(wells))
    for c in CLASS_ORDER:
        vals = th[c].to_numpy() if c in th else np.zeros(len(wells))
        if vals.sum() == 0:
            continue
        a1.barh(y, vals, left=left, color=CLASS_COLORS[c], edgecolor="white", lw=0.5, label=c)
        left += vals
    labelled = left - (th["unlabelled"].to_numpy() if "unlabelled" in th else 0)
    for yi, tot, lab_m in zip(y, left, labelled):
        a1.text(tot + 15, yi, f"labelled {100 * lab_m / tot:.0f}%" if tot else "no data", va="center", fontsize=8)
    names = [f"{w} (reserved)" if w in reserved else w for w in wells]
    a1.set_yticks(y)
    a1.set_yticklabels(names)
    a1.set_xlabel("logged thickness (m MD)")
    a1.set_title("Class thickness by well")
    a1.legend(loc="lower right", fontsize=8)
    a1.set_xlim(0, left.max() * 1.18 if left.max() else 1)

    unl = lab[lab["class_name"] == "unlabelled"].copy()
    groups = {"heterolithic (Vsh dead band)": unl["lithology_label"] == "heterolithic",
              "placeholder VSH=0 & PHIT=0": unl["vsh_flag"] == "placeholder_zero_vsh_and_zero_phit",
              "PHIT above physical bound": unl["vsh_flag"] == "artifact_phit_out_of_bounds",
              "tight non-shale (low PHIT)": unl["lithology_label"] == "tight_nonshale",
              "VSH missing / other": ~unl["lithology_label"].isin(["heterolithic", "tight_nonshale"])
              & ~unl["vsh_flag"].isin(["placeholder_zero_vsh_and_zero_phit", "artifact_phit_out_of_bounds"])}
    colors = [LITH_COLORS["heterolithic"], STATUS["critical"], STATUS["serious"], LITH_COLORS["tight_nonshale"], NEUTRAL]
    left = np.zeros(len(wells))
    for (gname, m), col in zip(groups.items(), colors):
        vals = unl[m].groupby("well")["md_step_m"].sum().reindex(wells).fillna(0.0).to_numpy()
        if vals.sum() == 0:
            continue
        a2.barh(y, vals, left=left, color=col, edgecolor="white", lw=0.5, label=gname)
        left += vals
    a2.set_yticks(y)
    a2.set_yticklabels(names)
    a2.set_xlabel("unlabelled thickness (m MD)")
    a2.set_title("Why samples stay unlabelled")
    a2.set_xlim(0, left.max() * 1.05 if left.max() else 1)
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8)
    fig.suptitle("02_labels: class distribution (provider-log labels, provisional; oil/gas classes unsupported)",
                 fontsize=13, fontweight="bold")
    footer(fig, "Thickness = sum of sample spacing along MD (not true stratigraphic thickness). "
                "Reserved well shown for transparency; it is never used for supervised fitting.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ctx.save_figure(fig, "class_distribution_by_well.png", "Class thickness and unlabelled reasons per well",
                    required=True, multipanel=True)


def _figure_crossplots(ctx, lab, pol):
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)                          # display subsample only; tables use all rows
    fig, axs = plt.subplots(2, 2, figsize=(18, 14))
    lp, fp = pol["lithology"], pol["fluid"]

    a = axs[0, 0]
    v = lab[lab["vsh_valid"] & lab["phit_valid"]]
    v = v.iloc[rng.permutation(len(v))[:40000]] if len(v) > 40000 else v
    for c in CLASS_ORDER:
        s = v[v["class_name"] == c]
        if len(s):
            a.scatter(s["vsh_raw"], s["phit_raw"], s=3, alpha=0.35, color=CLASS_COLORS[c], label=f"{c} ({len(s)})",
                      rasterized=True)
    for x in (lp["vsh_sand_max"], lp["vsh_shale_min"]):
        a.axvline(x, color=TEXT_2, ls="--", lw=1)
    a.axhline(lp["phit_sand_min"], color=TEXT_2, ls=":", lw=1)
    a.set_xlabel("VSH (v/v)")
    a.set_ylabel("PHIT (v/v)")
    a.set_title("(a) PHIT vs VSH by class (valid samples)")
    a.legend(markerscale=4, fontsize=8)

    a = axs[0, 1]
    s = lab[(lab["lithology_label"] == "sand") & lab["swt_valid"]]
    s = s.iloc[rng.permutation(len(s))[:40000]] if len(s) > 40000 else s
    for f in ("water_bearing", "hydrocarbon_potential", "unresolved", "conflict"):
        q = s[s["fluid_label"] == f]
        if len(q):
            a.scatter(q["phit_raw"], q["swt_raw"], s=4, alpha=0.4, color=FLUID_COLORS[f], label=f"{f} ({len(q)})",
                      rasterized=True)
    phi = np.linspace(0.03, 0.45, 200)
    for bvw in (0.03, 0.05, 0.08):
        a.plot(phi, np.where(bvw / phi <= 1.0, bvw / phi, np.nan), color=TEXT_2, lw=0.8, ls="-.")
        a.text(0.44, min(bvw / 0.44, 1.0), f"BVW {bvw}", fontsize=7, color=TEXT_2, ha="right", va="bottom")
    a.axhline(fp["swt_hc_potential_max"], color=FLUID_COLORS["hydrocarbon_potential"], ls="--", lw=1)
    a.axhline(fp["swt_water_min"], color=FLUID_COLORS["water_bearing"], ls="--", lw=1)
    a.set_xlabel("PHIT (v/v)")
    a.set_ylabel("SWT (v/v)")
    a.set_ylim(0, 1.05)
    a.set_title("(b) Sand only: SWT vs PHIT (Buckles-type, BVW = PHIT x SWT)")
    a.legend(markerscale=4, fontsize=8, loc="lower left")

    a = axs[1, 0]
    sand = lab[(lab["lithology_label"] == "sand") & lab["swt_valid"]]
    for name, m, col in (("provider gas flag", sand["provider_pay_flag"] == "gas", CLASS_COLORS["gas_sand"]),
                         ("provider oil flag", sand["provider_pay_flag"] == "oil", CLASS_COLORS["oil_sand"]),
                         ("no flag (evaluated)", sand["provider_pay_flag"] == "none", NEUTRAL)):
        if m.sum():
            x = np.sort(sand.loc[m, "swt_raw"].to_numpy(float))
            a.step(x, np.arange(1, x.size + 1) / x.size, where="post", lw=2, color=col,
                   label=f"{name} (n={x.size}; {100 * np.mean(x <= fp['swt_hc_potential_max'] + 0.05):.0f}% "
                         f"<= {fp['swt_hc_potential_max'] + 0.05:.2f})")
    a.axvline(fp["swt_hc_potential_max"], color=FLUID_COLORS["hydrocarbon_potential"], ls="--", lw=1)
    a.axvline(fp["swt_water_min"], color=FLUID_COLORS["water_bearing"], ls="--", lw=1)
    a.set_xlim(0, 1.02)
    a.set_ylim(0, 1.02)
    a.set_xlabel("SWT (v/v), sand samples")
    a.set_ylabel("cumulative fraction of samples")
    a.set_title("(c) Provider pay flags vs SWT (cumulative): flagged sand stops at SWT ~0.75\n"
                "-> the flags behave like net-pay cutoffs, not independent fluid tests")
    a.legend(fontsize=8, loc="lower right")

    a = axs[1, 1]
    if "rhob_raw" in lab and "dt_raw" in lab and lab["rhob_raw"].notna().any() and lab["dt_raw"].notna().any():
        e = lab[lab["phit_valid"] & lab["rhob_raw"].notna() & lab["dt_raw"].notna()]
        rho = np.where(e["rhob_raw"].median() > 100, e["rhob_raw"], e["rhob_raw"] * 1000.0)
        ai = rho * 304800.0 / e["dt_raw"].to_numpy(float)
        for c in CLASS_ORDER:
            m = (e["class_name"] == c).to_numpy()
            if m.any():
                a.scatter(e["phit_raw"].to_numpy()[m], ai[m], s=4, alpha=0.4, color=CLASS_COLORS[c], label=c)
        a.set_xlabel("PHIT (v/v)")
        a.set_ylabel("acoustic impedance (kg/m3 * m/s)")
        a.set_title("(d) Elastic rock physics: AI vs PHIT")
        a.legend(markerscale=4, fontsize=8)
    else:
        a.set_axis_off()
        a.text(0.5, 0.62, "(d) Elastic rock-physics crossplots", ha="center", fontsize=12, fontweight="bold",
               transform=a.transAxes)
        a.text(0.5, 0.48, "NOT AVAILABLE", ha="center", fontsize=24, fontweight="bold",
               color=STATUS["critical"], transform=a.transAxes)
        a.text(0.5, 0.30, "No sonic (DT/DTS) or density (RHOB) logs in any LAS file:\n"
               "Vp, Vs, density and impedance cannot be cross-plotted against Vsh/PHIT/SWT.\n"
               "The elastic link enters later through the inversion volumes (stage 03/04, conditional).",
               ha="center", va="top", fontsize=10, transform=a.transAxes)
    fig.suptitle("02_labels: petrophysical crossplots (provider interpreted curves; display subsample of 40k points)",
                 fontsize=13, fontweight="bold")
    footer(fig, f"Cutoffs {pol['policy_version']} (provisional). Colours: fixed class palette; labels in legends.")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    ctx.save_figure(fig, "rock_physics_crossplots.png",
                    "PHIT-VSH by class, sand SWT-PHIT by fluid evidence, pay flags vs SWT, elastic panel",
                    required=True, multipanel=True)


# =============================================================================== step 11
def _summarise(ctx, lab, pol, support, wells, dev_wells, flag_inference):
    dev = lab[lab["well"].isin(dev_wells)]
    th = dev.groupby("class_name")["md_step_m"].sum()
    for c in CLASS_ORDER:
        ctx.metric(f"dev_md_m_{c}", round(float(th.get(c, 0.0)), 1))
    ctx.metric("dev_fraction_labelled", round(float(1 - th.get("unlabelled", 0.0) / th.sum()), 4) if th.sum() else None)
    ctx.metric("task_status", {r["task"]: r["status"] for _, r in support.iterrows()})
    vsh50_sand_placeholder = float(lab.loc[(lab["lith_vsh50_baseline"] == "sand") & lab["artifact_interval"],
                                           "md_step_m"].sum())
    ctx.metric("vsh50_baseline_sand_md_m_on_placeholders", round(vsh50_sand_placeholder, 1))
    ctx.metric("provider_flag_inference", flag_inference)

    # ---- acceptance checks (things that must hold for the labels to be usable at all)
    ids_ok = lab["class_id"].isin([c for c, *_ in CLASS_TABLE]).all()
    ctx.check("one_valid_class_per_sample", bool(ids_ok), f"{len(lab)} samples, class ids in {sorted(CLASS_NAME)}")
    no_impute = True
    for t in ("vsh", "phit", "swt"):
        m = lab[f"{t}_mask"].to_numpy(bool)
        f = lab[f"{t}_fraction"].to_numpy(float)
        no_impute &= bool(np.all(np.isnan(f[~m])) and np.allclose(f[m], lab[f"{t}_raw"].to_numpy(float)[m]))
    ctx.check("no_imputed_truth", no_impute, "masked targets are NaN; valid targets equal the raw provider value")
    lo, hi = pol["bounds"]["phit"]
    inb = (lab["vsh_fraction"].dropna().between(0, 1).all() and lab["phit_fraction"].dropna().between(lo, hi).all()
           and lab["swt_fraction"].dropna().between(0, 1).all())
    ctx.check("targets_within_physical_bounds", bool(inb), f"VSH, SWT in [0,1]; PHIT in ({lo}, {hi}]")
    fluid_ok = not lab.loc[lab["class_name"].isin(["oil_sand", "gas_sand"]), "class_status"].ne(
        "provider_fluid_flag_reviewed").any()
    ctx.check("no_oil_gas_without_reviewed_evidence", bool(fluid_ok),
              "oil_sand/gas_sand only with accept_provider_fluid_flags + named reviewer")
    have = sorted(lab["well"].unique())
    ctx.check("labels_for_every_well", set(have) == set(wells), f"{len(have)}/{len(wells)} wells have log samples")

    # ---- limitations: what keeps these labels CONDITIONAL
    ctx.limit("Lithology cutoffs (VSH <= {0} sand, >= {1} shale) are baseline proxies, NOT geologist-approved; "
              "formation-specific endpoints need formation tops (organiser: markers not available)"
              .format(pol["lithology"]["vsh_sand_max"], pol["lithology"]["vsh_shale_min"]))
    unsupported = support.loc[support["status"] == "TASK_UNSUPPORTED", "task"].tolist()
    if unsupported:
        ctx.limit(f"TASK_UNSUPPORTED: {unsupported} - use the hierarchical tasks (y_lithology, y_sand_fluid)")
    if pol["fluid"]["brine_from_interpreted_swt"]:
        ctx.limit("brine_sand and water/hydrocarbon-potential labels rest on the provider SWT (model undocumented, "
                  "no resistivity to check it, no water tests) - provisional")
    ctx.warn(f"VSH50 baseline (historic 05D proxy) would label {vsh50_sand_placeholder:.0f} m MD of placeholder/"
             "artifact intervals (VSH = 0 with PHIT = 0 or PHIT > bound) as sand; this policy masks them")

    lp, fp = pol["lithology"], pol["fluid"]
    ctx.readme("Label rules (provisional)", f"""
* **Lithology** from provider VSH: sand if VSH <= {lp['vsh_sand_max']} and PHIT >= {lp['phit_sand_min']};
  shale if VSH >= {lp['vsh_shale_min']}; in between = heterolithic -> unlabelled.
* **Masked, never clipped:** VSH = 0 together with PHIT = 0 (placeholder / tight streak), PHIT outside
  {pol['bounds']['phit']}, PHIT exactly 0, SWT where PHIT < {fp['phit_min_for_swt']}.  Raw values stay in `*_raw`.
* **Fluid (sand only):** SWT >= {fp['swt_water_min']} and no pay flag -> water_bearing (brine_sand);
  SWT <= {fp['swt_hc_potential_max']} -> hydrocarbon_potential (undetermined_sand).  Never oil or gas from Sw.
* **Oil / gas:** {'accepted from provider flags (reviewed by ' + str(fp.get('provider_flags_reviewed_by')) + ')' if fp.get('provider_flags_accepted') else 'TASK_UNSUPPORTED - no independent fluid evidence; PS_GIP/PS_OIP behave like net-pay cutoffs (see provider_flag_audit.csv)'}.
* **Train on:** `y_lithology` (sand vs shale), `y_sand_fluid` (water vs hydrocarbon potential),
  `vsh_fraction`/`phit_fraction`/`swt_fraction` with their masks.  Everything is in `label_policy.json`.
""")
    ctx.readme("Task support", support[["task", "status", "n_wells_supporting"]].to_markdown(index=False)
               if hasattr(support, "to_markdown") and _has_tabulate() else support[["task", "status"]].to_string(index=False))
    ctx.extra["task_support"] = {r["task"]: r["status"] for _, r in support.iterrows()}
    ctx.extra["class_map_version"] = pol["class_map_version"]


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False
