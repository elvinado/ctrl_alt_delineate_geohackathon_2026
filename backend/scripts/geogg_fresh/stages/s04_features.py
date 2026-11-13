"""04_features - stack attributes, inversion inputs, AVO (quantitative only with verified angles).

What this stage answers
-----------------------
"Which numbers describe the seismic around every labelled sample, computed only
from the five amplitude stacks (plus inversion values as separate, conditional
inputs), and what can - and cannot - be said about AVO?"

Steps (progress bars in the terminal)
-------------------------------------
1. Upstream        - 03 patches + sample_index + manifest (channel order), 03 4-ms labels (figures only).
2. Stack attributes- 13 attributes x 5 stacks from the patches (features.py), polarity preserved.
3. Contrasts       - near/far/ultrafar differences, safe normalised differences, amplitude trend vs
                     stack ORDER (a proxy: the angles are not documented).
4. Elastic inputs  - PIMP / SIMP / VpVs: centre value, local mean, local std (conditional inputs).
5. AVO             - events picked on the full stack along each path; Shuey A/B + candidate class
                     ONLY if angle ranges, amplitude preservation and provenance are configured;
                     otherwise QUALITATIVE_PROXY_ONLY.
6. 1000-ms overview- RMS per stack per 1000 ms of TWT, survey-wide (random live bins) and along wells.
7. Dictionary      - every column: definition, window, units, polarity, predictor policy.
8. Figures, 9. Summary.

Exact zeros (config section ``missing_data``, see ``zero_fill_summary.csv``):
* inversion volumes - every zero and every negative value is no-data fill (impedance and Vp/Vs
  can never be <= 0): NaN.
* amplitude stacks - only RUNS of >= ``stack_zero_run_missing_min`` consecutive zeros along a
  trace (mutes, padding, dead traces) are no-data.  Isolated zeros are kept as amplitude 0:
  the full stack holds ~9 % isolated exact zeros at the wells that sit where the partial
  stacks are also near zero (``partner_abs_ratio_at_zeros``), i.e. small amplitudes stored
  as 0, not missing samples.  (Release 0.5 turned them into NaN, which blanked the full-stack
  envelope / semblance / spectral attributes of most rows - fixed in 0.6.)
No label is ever used to compute a feature.  Labels appear only in figures (colouring),
never in feature tables.  TotalSandProbability is not read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..stage_compat import StageBlocked, SpecialistStageContext as StageContext
from ..config import cfg_get, resolve_expected_wells
from ..features import (ATTR_DOC, STACK_ATTRS, avo_class_candidate, local_summaries, matched_event_amplitude,
                        normalized_difference, pick_events, shuey_fit, stack_attributes, stack_zero_missing,
                        trend_vs_order)
from ..plotting import CLASS_COLORS, CLASS_ORDER, NEUTRAL, SERIES, STATUS, TEXT_2, footer, setup_style
from ..progress import StepTracker, progress
from ..segy import SegyFile, SurveyGeometry

STEPS = ["upstream", "stack attributes", "contrasts", "elastic inputs", "avo", "1000-ms overview",
         "dictionary", "figures", "summary"]
KEYS = ["patch_row", "well", "twt_bin_index", "desired_twt_ms"]


# =============================================================================== stage
def run(ctx: StageContext) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_style()
    steps = StepTracker(STEPS, ctx.stage)
    cfg = ctx.cfg
    fc = cfg.get("features") or {}
    av = cfg.get("avo") or {}
    stacks = list(fc.get("stacks") or ["full", "near", "mid", "far", "ultrafar"])
    partial = list(fc.get("partial_stacks") or ["near", "mid", "far", "ultrafar"])
    elastic = list(fc.get("elastic") or ["pimp", "simp", "vpvs"])
    if "totalsandprob" in stacks + elastic:
        raise StageBlocked("TotalSandProbability must never be a feature source (AGENTS.md)")
    min_fin = float(fc.get("min_finite_fraction", 0.7))
    run_dir = ctx.run.run_dir
    wells = [w.lower() for w in cfg_get(cfg, "wells.expected", [])]
    reserved = [w.lower() for w in (cfg_get(cfg, "wells.reserved_reviewed", []) or [])]

    # ---------------------------------------------------------------- 1 upstream
    steps.next("upstream")
    ex_dir = run_dir / "03_extraction"
    version = str(cfg_get(cfg, "extraction.version", "v1"))
    idx_p = ex_dir / "tables" / "sample_index.csv"
    man_p = ex_dir / "patches" / f"patch_manifest_{version}.json"
    if not idx_p.exists() or not man_p.exists():
        raise StageBlocked(f"Missing {idx_p} or {man_p}: run 03_extraction where the SEG-Y files are reachable")
    manifest = json.loads(man_p.read_text(encoding="utf-8"))
    arr_p = ex_dir / "patches" / manifest["file"]
    if not arr_p.exists():
        raise StageBlocked(f"Missing patch array {arr_p}")
    for p, role in ((idx_p, "sample_index"), (man_p, "patch manifest"), (arr_p, "patch array")):
        ctx.add_input(p, f"upstream:03_extraction {role}")
    idx = pd.read_csv(idx_p)
    wells = resolve_expected_wells(cfg, labels=idx)
    if not wells:
        raise StageBlocked("No wells found in Stage 03 sample_index.csv and wells.expected is empty")
    patches = np.load(arr_p, mmap_mode="r")
    channels = list(manifest["channels"])
    dt = float(manifest["dt_ms"])
    if patches.shape[0] != len(idx) or not (idx["patch_row"].to_numpy() == np.arange(len(idx))).all():
        raise StageBlocked("sample_index.csv rows do not match the patch array rows - rerun 03_extraction")
    lab_p = ex_dir / "tables" / f"labels_4ms_{version}.csv"
    labels = pd.read_csv(lab_p) if lab_p.exists() else None
    if labels is not None:
        ctx.add_input(lab_p, "upstream:03 labels (figures only, never features)")
    present = {ch: bool(np.isfinite(patches[:, channels.index(ch)]).any()) for ch in channels}
    ctx.parameters.update({"stacks": stacks, "partial_stacks": partial, "elastic": elastic, "dt_ms": dt,
                           "patch_shape": list(patches.shape[1:]), "min_finite_fraction": min_fin})

    # ---------------------------------------------------------------- 2 stack attributes
    steps.next("stack attributes")
    feats = idx[KEYS].copy()
    md = cfg.get("missing_data") or {}
    min_run = int(md.get("stack_zero_run_missing_min", 5))
    el_zero_missing = bool(md.get("elastic_zero_is_missing", True))
    ctx.parameters.update({"stack_zero_run_missing_min": min_run, "elastic_zero_is_missing": el_zero_missing})
    qc_rows: list[dict] = []
    partner = _partner_means(patches, channels, [s for s in stacks if s in channels])
    for s in progress(stacks, desc="stack attributes", unit="stack"):
        if s not in channels:
            ctx.limit(f"stack '{s}' is not a channel of the patch array - its attributes are absent")
            continue
        X = _load_stack(patches, channels, s, min_run, partner.get(s), idx, qc_rows)
        for name, values in stack_attributes(X, dt, min_fin).items():
            feats[f"{s}__{name}"] = values
        if not present[s]:
            ctx.limit(f"stack '{s}' has no data in any patch (volume missing in 03): its attributes are NaN")

    # ---------------------------------------------------------------- 3 contrasts (proxies)
    steps.next("contrasts")
    contrast_doc = []

    def add(name, values, definition, sources):
        feats[name] = values
        contrast_doc.append((name, definition, sources))

    pairs = [("near", "far"), ("near", "ultrafar"), ("mid", "far")]
    for a, b in pairs:
        if f"{a}__amp_c" in feats and f"{b}__amp_c" in feats:
            add(f"{a}_{b}__amp_diff", feats[f"{b}__amp_c"] - feats[f"{a}__amp_c"],
                f"{b} centre amplitude minus {a} centre amplitude (signed)", f"{a},{b}")
            add(f"{a}_{b}__rms_nd", normalized_difference(feats[f"{b}__rms_long"], feats[f"{a}__rms_long"]),
                f"({b} RMS - {a} RMS) / ({b} RMS + {a} RMS), 80 ms window; NaN if the sum is ~0", f"{a},{b}")
    have_partial = [s for s in partial if f"{s}__amp_c" in feats]
    if len(have_partial) >= 3:
        amps = np.column_stack([feats[f"{s}__amp_c"] for s in have_partial])
        slope, inter = trend_vs_order(amps)
        add("partials__amp_slope_vs_order", slope,
            f"least-squares slope of centre amplitude vs stack ORDER {have_partial} (1..{len(have_partial)}) - "
            "proxy, not an AVO gradient", ",".join(have_partial))
        add("partials__amp_intercept_vs_order", inter,
            "intercept of the same fit at order 0 - proxy, not an AVO intercept", ",".join(have_partial))
    feats_stack = feats.copy()

    # ---------------------------------------------------------------- 4 elastic inputs
    steps.next("elastic inputs")
    el = idx[KEYS].copy()
    for e in elastic:
        if e not in channels:
            ctx.limit(f"elastic channel '{e}' not in the patch array")
            continue
        X = _load_elastic(patches, channels, e, el_zero_missing, idx, qc_rows)
        for name, values in local_summaries(X).items():
            el[f"{e}__{name}"] = values
    valid_el = {e: float(el[f"{e}__c"].notna().mean()) for e in elastic if f"{e}__c" in el}
    ctx.metric("elastic_centre_valid_fraction", valid_el)
    zf = pd.DataFrame(qc_rows)
    ctx.save_table(zf, "zero_fill_summary.csv", "Per well (and all wells) and channel: exact zeros, zero runs, "
                   "what was treated as missing, and the evidence (smallest non-zero |amplitude|, amplitude of the "
                   "other stacks at the zeros)", required=True)
    _report_zeros(ctx, zf, stacks, elastic, min_run)

    # ---------------------------------------------------------------- 5 AVO
    steps.next("avo")
    avo_mode, avo_reason, angles = _avo_mode(av, partial)
    ctx.extra["avo_mode"] = avo_mode
    events = _avo_events(idx, patches, channels, partial, labels, dt, av, avo_mode, angles)
    if avo_mode != "QUANTITATIVE":
        ctx.limit(f"AVO is QUALITATIVE_PROXY_ONLY: {avo_reason}")

    # ---------------------------------------------------------------- 6 1000-ms overview
    steps.next("1000-ms overview")
    overview = _overview(ctx, feats_stack, stacks, partial, run_dir, float(av.get("overview_bin_ms", 1000)),
                         int(av.get("overview_n_traces", 2000)), min_run)

    # ---------------------------------------------------------------- 7 dictionary + tables
    steps.next("dictionary")
    dictionary = _dictionary(stacks, elastic, contrast_doc, present, channels, dt, patches.shape[1:])
    units = {c: "amplitude units of the volume (uncalibrated)" for c in feats_stack.columns if c not in KEYS}
    ctx.save_table(feats_stack, "stack_attributes.csv", "Per patch row: attributes from the five stacks + "
                   "stack contrasts (proxies). Keys only, no labels.", units=units, required=True)
    ctx.save_table(el, "elastic_inputs.csv", "Per patch row: PIMP/SIMP/VpVs centre value, 3x3x5 local mean and "
                   "std (conditional inputs, provenance unaudited)", required=True)
    ctx.save_table(events, "avo_event_fits.csv", f"Events picked on the full stack along each path; mode "
                   f"{avo_mode}", required=True)
    ctx.save_table(overview, "avo_1000ms_overview.csv", "RMS per stack in 1000-ms TWT bins, survey-wide and "
                   "along wells - trend overview only, never an AVO class", required=True)
    ctx.save_table(dictionary, "feature_dictionary.csv", "Every column: family, definition, window, units, "
                   "polarity, predictor policy", required=True)

    # ---------------------------------------------------------------- 8 figures
    steps.next("figures")
    _figure_examples(ctx, feats_stack, el, labels, fc.get("example_well"), wells, reserved)
    _figure_distributions(ctx, feats_stack, el, labels, wells, reserved)
    _figure_avo_events(ctx, events, avo_mode, partial, angles)
    _figure_avo_crossplot(ctx, events, avo_mode)
    _figure_overview(ctx, overview, stacks)

    # ---------------------------------------------------------------- 9 summary
    steps.next("summary")
    _summarise(ctx, feats_stack, el, dictionary, events, avo_mode, present, stacks, elastic)
    steps.close()


# =============================================================================== channel loading
def _partner_means(patches, channels, stacks):
    """For every stack: the mean of the OTHER stacks at every patch sample (evidence for zero diagnostics)."""
    out = {}
    if len(stacks) < 2:
        return out
    raw = {s: np.asarray(patches[:, channels.index(s)], dtype=np.float32) for s in stacks}
    total = sum(raw.values())
    for s in stacks:
        out[s] = (total - raw[s]) / (len(stacks) - 1)
    return out


def _zero_qc(zero, missing, X_raw, partner, ci, cj, ck, rows, well, ch, rule, negative=None):
    """One QC row: how many zeros, how many treated as missing, and what the zeros look like."""
    nz = X_raw[~zero & np.isfinite(X_raw)]
    absnz = np.abs(nz)
    iso = zero & ~missing
    r = {"well": well, "channel": ch, "n_patch_samples": int(zero.size),
         "fraction_patch_samples_zero": float(zero.mean()) if zero.size else np.nan,
         "fraction_centre_samples_zero": float(zero[:, ci, cj, ck].mean()) if zero.size else np.nan,
         "fraction_patch_samples_missing": float(missing.mean()) if zero.size else np.nan,
         "fraction_centre_samples_missing": float(missing[:, ci, cj, ck].mean()) if zero.size else np.nan,
         "fraction_patch_samples_isolated_zero_kept": float(iso.mean()) if zero.size else np.nan,
         "nonzero_abs_p001": float(np.quantile(absnz, 0.001)) if absnz.size else np.nan,
         "nonzero_abs_median": float(np.median(absnz)) if absnz.size else np.nan,
         "fraction_patch_samples_negative": float(negative.mean()) if negative is not None and negative.size else np.nan,
         "partner_abs_ratio_at_zeros": np.nan, "treated_as_missing": rule}
    if partner is not None and iso.any() and (~zero).any():
        p_at = np.abs(partner[iso])
        p_else = np.abs(partner[~zero])
        p_at, p_else = p_at[np.isfinite(p_at)], p_else[np.isfinite(p_else)]
        if p_at.size and p_else.size and np.median(p_else) > 0:
            r["partner_abs_ratio_at_zeros"] = float(np.median(p_at) / np.median(p_else))
    rows.append(r)


def _load_stack(patches, channels, ch, min_run, partner, idx, qc_rows):
    """One amplitude stack as float64: runs of >= min_run exact zeros -> NaN; isolated zeros kept.

    Why not every zero?  A live trace can hold genuine exact zeros (small amplitudes stored
    as 0).  Turning those into NaN blanks every attribute that needs a complete window
    (envelope, semblance, spectrum) - exactly what happened to the full stack in release 0.5.
    """
    X = np.asarray(patches[:, channels.index(ch)], dtype=np.float64)
    zero, missing = stack_zero_missing(X, min_run)
    ci, cj, ck = X.shape[1] // 2, X.shape[2] // 2, X.shape[3] // 2
    rule = f"runs of >= {min_run} zeros along time (isolated zeros kept as amplitude 0)"
    groups = list(idx.groupby("well").groups.items())
    for w, rows in groups:
        r = np.asarray(rows)
        _zero_qc(zero[r], missing[r], X[r], partner[r] if partner is not None else None, ci, cj, ck, qc_rows, w,
                 ch, rule)
    _zero_qc(zero, missing, X, partner, ci, cj, ck, qc_rows, "ALL_WELLS", ch, rule)
    X[missing] = np.nan
    return X


def _load_elastic(patches, channels, ch, zero_is_missing, idx, qc_rows):
    """One inversion channel: zeros (no-data fill) and negative values -> NaN (never impedance = 0)."""
    X = np.asarray(patches[:, channels.index(ch)], dtype=np.float64)
    zero = X == 0.0
    neg = X < 0.0
    missing = (zero | neg) if zero_is_missing else neg
    ci, cj, ck = X.shape[1] // 2, X.shape[2] // 2, X.shape[3] // 2
    rule = "every zero and negative value" if zero_is_missing else "negative values only"
    for w, rows in idx.groupby("well").groups.items():
        r = np.asarray(rows)
        _zero_qc(zero[r], missing[r], X[r], None, ci, cj, ck, qc_rows, w, ch, rule, negative=neg[r])
    _zero_qc(zero, missing, X, None, ci, cj, ck, qc_rows, "ALL_WELLS", ch, rule, negative=neg)
    X[missing] = np.nan
    return X


def _report_zeros(ctx, zf, stacks, elastic, min_run):
    """Log the evidence and the decision for every channel (warnings where the evidence disagrees)."""
    if zf.empty:
        return
    allw = zf[zf["well"] == "ALL_WELLS"].set_index("channel")
    ctx.metric("fraction_centre_samples_missing_by_channel",
               allw["fraction_centre_samples_missing"].round(4).to_dict())
    ctx.metric("fraction_patch_samples_isolated_zero_kept_by_channel",
               allw["fraction_patch_samples_isolated_zero_kept"].round(4).to_dict())
    for s in [s for s in stacks if s in allw.index]:
        r = allw.loc[s]
        iso, ratio = float(r["fraction_patch_samples_isolated_zero_kept"]), r["partner_abs_ratio_at_zeros"]
        if iso < 0.001:
            continue
        msg = (f"{s}: {100 * iso:.1f}% of patch samples are isolated exact zeros (runs < {min_run}), kept as "
               f"amplitude 0; smallest non-zero |amp| (0.1% quantile) {r['nonzero_abs_p001']:.3g} vs median "
               f"{r['nonzero_abs_median']:.3g}; other stacks at these zeros carry {ratio:.2f} x their usual |amp|")
        if np.isfinite(ratio) and ratio > 0.6:
            ctx.warn(msg + " -> the zeros do NOT sit where the other stacks are small: they may be dropouts. "
                     "Review, and consider missing_data.stack_zero_run_missing_min: 1")
        else:
            ctx.info(msg + " -> consistent with small amplitudes stored as 0 (not missing data)")
    el_zero = allw.loc[[e for e in elastic if e in allw.index], "fraction_centre_samples_missing"]
    if (el_zero > 0).any():
        ctx.warn("Inversion volumes are zero-filled (no data) at part of the wells: " +
                 ", ".join(f"{k} {100 * v:.1f}% of centre samples" for k, v in el_zero.items()) +
                 " - treated as missing (NaN), never as impedance or Vp/Vs = 0")


# =============================================================================== AVO
def _avo_mode(av: dict, partial: list[str]):
    ranges = av.get("angle_ranges_deg") or {}
    missing = [s for s in partial if not ranges.get(s)]
    if missing:
        return "QUALITATIVE_PROXY_ONLY", f"effective angle ranges not documented for {missing}", {}
    if not av.get("amplitude_preservation_verified"):
        return "QUALITATIVE_PROXY_ONLY", "relative amplitude preservation of the partial stacks not verified", {}
    if not str(av.get("angle_provenance") or "").strip():
        return "QUALITATIVE_PROXY_ONLY", "angle ranges given without avo.angle_provenance", {}
    angles = {s: float(np.mean(ranges[s])) for s in partial}
    return "QUANTITATIVE", "", angles


def _avo_events(idx, patches, channels, partial, labels, dt, av, mode, angles):
    """Pick events on the full-stack centre trace along each well; amplitudes on the partial stacks."""
    q = float(av.get("event_quantile", 0.75))
    search = int(av.get("registration_search_samples", 1))
    cf = channels.index("full") if "full" in channels else None
    lab = labels.set_index(["well", "twt_bin_index"]) if labels is not None else None
    rows = []
    if cf is None:
        return pd.DataFrame(rows)
    ci, cj, ck = patches.shape[2] // 2, patches.shape[3] // 2, patches.shape[4] // 2
    for w, g in idx.groupby("well", sort=False):
        g = g.sort_values("desired_twt_ms")
        full_c = np.asarray(patches[g["patch_row"].to_numpy(), cf, ci, cj, ck], dtype=float)
        for i in pick_events(g["desired_twt_ms"].to_numpy(), full_c, dt, q):
            r = g.iloc[i]
            row = {"well": w, "patch_row": int(r["patch_row"]), "twt_bin_index": int(r["twt_bin_index"]),
                   "event_twt_ms": float(r["desired_twt_ms"]), "event_polarity": "peak" if full_c[i] > 0 else "trough",
                   "full_amp": float(full_c[i])}
            sign = 1.0 if full_c[i] > 0 else -1.0
            amps = []
            for s in partial:
                if s not in channels:
                    amps.append(np.nan)
                    continue
                tr = np.asarray(patches[int(r["patch_row"]), channels.index(s), ci, cj, :], dtype=float)
                a, shift = matched_event_amplitude(tr, ck, sign, search)
                row[f"{s}_amp"], row[f"{s}_shift_samples"] = a, shift
                amps.append(a)
            amps = np.asarray(amps)
            slope, inter = trend_vs_order(amps[None, :])
            row["proxy_slope_vs_order"], row["proxy_intercept_vs_order"] = float(slope[0]), float(inter[0])
            if mode == "QUANTITATIVE":
                fit = shuey_fit(amps, np.array([angles[s] for s in partial]))
                row.update({f"shuey_{k}": v for k, v in fit.items()})
            if lab is not None and (w, int(r["twt_bin_index"])) in lab.index:
                L = lab.loc[(w, int(r["twt_bin_index"]))]
                row["class_at_event_display_only"] = L["class_name"]
                row["y_lithology_display_only"] = int(L["y_lithology"])
            rows.append(row)
    ev = pd.DataFrame(rows)
    if ev.empty:
        return ev
    ev["avo_mode"] = mode
    if mode == "QUANTITATIVE":
        near_zero = 0.2 * float(np.nanmedian(np.abs(ev["shuey_A"])))
        ev["avo_class_candidate"] = [avo_class_candidate(a, b, near_zero) for a, b in zip(ev["shuey_A"], ev["shuey_B"])]
        weak = (np.abs(ev["shuey_A"]) < 2 * ev["shuey_se_A"]) | (np.abs(ev["shuey_B"]) < 2 * ev["shuey_se_B"])
        ev["class_confidence"] = np.where(weak, "low (|A| or |B| < 2 SE)", "moderate")
        ev["class_note"] = ("reflection-event interpretation under the stated polarity; NOT an oil/gas label; "
                            f"A~0 band = 0.2 x median|A| = {near_zero:.4g}")
        ev["polarity_convention"] = str(av.get("polarity", "unknown"))
    else:
        ev["avo_class_candidate"] = "UNSUPPORTED (angles unverified)"
    return ev


# =============================================================================== overview
def _overview(ctx, feats, stacks, partial, run_dir, bin_ms, n_traces, min_run=5):
    rows = []
    edges = np.arange(0.0, 4000.0 + bin_ms, bin_ms)
    # ---- along the wells (centre amplitudes of the patch rows)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (feats["desired_twt_ms"] >= lo) & (feats["desired_twt_ms"] < hi)
        if not m.any():
            continue
        r = {"source": "well_paths (patch centres)", "twt_from_ms": lo, "twt_to_ms": hi, "n_samples": int(m.sum())}
        for s in stacks:
            c = f"{s}__amp_c"
            if c in feats:
                v = feats.loc[m, c].to_numpy(float)
                r[f"rms_{s}"] = float(np.sqrt(np.nanmean(v ** 2))) if np.isfinite(v).any() else np.nan
        rows.append(r)
    # ---- survey-wide: the SAME random live bins read from every stack
    inventory_path = run_dir / "00_inventory" / "tables" / "seismic_volume_inventory.csv"
    inventory = pd.read_csv(inventory_path) if inventory_path.is_file() else pd.DataFrame()
    if inventory_path.is_file():
        ctx.add_input(inventory_path, "upstream:00_inventory seismic volume registry")
    registered = {
        str(row.get("volume", "")).casefold(): row.get("path")
        for _, row in inventory.iterrows()
    }
    gpath = run_dir / "00_inventory" / "tables" / "geometry" / "full_geometry.npz"
    if gpath.exists():
        gfull = SurveyGeometry.load_npz(gpath)
        live = np.flatnonzero(gfull.grid.ravel() >= 0)
        pick = np.sort(np.random.default_rng(0).choice(live, size=min(n_traces, live.size), replace=False))
        ii, jj = np.unravel_index(pick, gfull.grid.shape)
        il, xl = gfull.il_min + ii * gfull.il_step, gfull.xl_min + jj * gfull.xl_step
        survey = {}
        for s in progress(stacks, desc="survey RMS", unit="stack"):
            raw_path = registered.get(str(s).casefold())
            path = Path(str(raw_path)).expanduser() if raw_path else None
            gp = run_dir / "00_inventory" / "tables" / "geometry" / f"{s}_geometry.npz"
            if path is None or not path.is_file() or not gp.exists():
                ctx.limit(f"survey overview: registered {s} volume or geometry is unavailable")
                continue
            g = SurveyGeometry.load_npz(gp)
            ctx.add_input(path, f"registered SEG-Y {s}; resolved from Stage 00 inventory")
            with SegyFile(path) as sgy:
                data = sgy.read_traces(g.trace_index(il, xl)).astype(np.float64)
            t = g.t0_ms + g.dt_ms * np.arange(g.ns)
            zero, missing = stack_zero_missing(data, min_run)       # same rule as the patches
            survey[s] = (t, data, zero, missing)
        for lo, hi in zip(edges[:-1], edges[1:]):
            r = {"source": f"survey ({len(pick)} random live bins, same IL/XL in every stack)", "twt_from_ms": lo,
                 "twt_to_ms": hi}
            any_v = False
            for s, (t, data, zero, missing) in survey.items():
                m = (t >= lo) & (t < hi)
                if m.any():
                    block = data[:, m].copy()
                    r[f"zero_fraction_{s}"] = float(zero[:, m].mean())
                    r[f"missing_fraction_{s}"] = float(missing[:, m].mean())
                    block[missing[:, m]] = np.nan              # mutes / padding must not dilute the RMS
                    r[f"rms_{s}"] = float(np.sqrt(np.nanmean(block ** 2))) if np.isfinite(block).any() else np.nan
                    r["n_samples"] = int(np.isfinite(block).sum())
                    any_v = True
            if any_v:
                rows.append(r)
    else:
        ctx.warn("00_inventory full geometry not found: survey-wide 1000-ms overview skipped")
    ov = pd.DataFrame(rows)
    if ov.empty:
        return ov
    for s in partial:
        if f"rms_{s}" in ov and "rms_near" in ov and s != "near":
            ov[f"ratio_{s}_to_near"] = ov[f"rms_{s}"] / ov["rms_near"].where(ov["rms_near"] > 0)
    cols = [f"rms_{s}" for s in partial if f"rms_{s}" in ov]
    if len(cols) >= 3:
        slope, _ = trend_vs_order(ov[cols].to_numpy(float))
        ov["rms_slope_vs_stack_order"] = slope
    ov["interpretation"] = "trend overview only - no AVO class is assigned to a 1000-ms bin"
    return ov


# =============================================================================== dictionary
def _dictionary(stacks, elastic, contrast_doc, present, channels, dt, shape):
    rows = [{"column": k, "family": "key", "definition": "row key / provenance", "source_channels": "",
             "window": "", "units": "", "polarity": "", "predictor_policy": "never (key)"} for k in KEYS]
    _, ni, nj, nt = shape
    for s in stacks:
        for a in STACK_ATTRS:
            definition, unit, window = ATTR_DOC[a]
            rows.append({"column": f"{s}__{a}", "family": "stack_attribute", "definition": definition,
                         "source_channels": s, "window": window.replace("all samples", f"{nt} samples = "
                                                                       f"{(nt - 1) * dt:.0f} ms span"),
                         "units": unit, "polarity": "preserved (signed)" if a in ("amp_c", "dadt_c", "cosphase_c")
                         else "polarity-independent", "predictor_policy": "allowed" if present.get(s) else
                         "allowed (channel absent in this run: all NaN)"})
    for name, definition, sources in contrast_doc:
        rows.append({"column": name, "family": "stack_contrast_proxy", "definition": definition,
                     "source_channels": sources, "window": "centre sample / 80 ms", "units": "amplitude units or "
                     "dimensionless", "polarity": "signed differences keep polarity",
                     "predictor_policy": "allowed - amplitude-trend proxy, NOT AVO (angles unverified)"})
    for e in elastic:
        for k, d in (("c", "centre value"), ("local_mean", "mean over 3x3 bins x 5 samples"),
                     ("local_std", "std over 3x3 bins x 5 samples")):
            rows.append({"column": f"{e}__{k}", "family": "elastic_input", "definition": d, "source_channels": e,
                         "window": "3x3x5" if k != "c" else "centre", "units": "inversion volume units",
                         "polarity": "n/a", "predictor_policy": "conditional - inversion provenance and "
                         "calibration wells unaudited; report Stacks-only results as the primary benchmark"})
    rows.append({"column": "totalsandprob*", "family": "excluded", "definition": "TotalSandProbability",
                 "source_channels": "totalsandprob", "window": "", "units": "", "polarity": "",
                 "predictor_policy": "never - target-like derived product"})
    return pd.DataFrame(rows)


# =============================================================================== figures
def _well_with_most_sand(labels, wells, reserved):
    if labels is None:
        return wells[0]
    d = labels[~labels["well"].isin(reserved)]
    c = d[d["y_lithology"] == 1].groupby("well").size()
    return c.idxmax() if len(c) else d["well"].iloc[0]


def _figure_examples(ctx, feats, el, labels, example_well, wells, reserved):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    w = example_well if example_well in set(feats["well"]) else _well_with_most_sand(labels, wells, reserved)
    f = feats[feats["well"] == w].sort_values("desired_twt_ms")
    e = el[el["well"] == w].sort_values("desired_twt_ms")
    t = f["desired_twt_ms"].to_numpy()
    tracks = [("class (4 ms)", None), ("centre amplitude", ["full__amp_c", "near__amp_c", "far__amp_c"]),
              ("envelope / RMS (full)", ["full__env_c", "full__rms_long"]), ("semblance (full)", ["full__semblance"]),
              ("far-near RMS nd", ["near_far__rms_nd"]), ("P-impedance", ["pimp__c"]), ("Vp/Vs", ["vpvs__c"])]
    fig, axs = plt.subplots(1, len(tracks), figsize=(21, 12), sharey=True,
                            gridspec_kw={"width_ratios": [0.35] + [1] * (len(tracks) - 1)})
    if labels is not None:
        L = labels[labels["well"] == w].set_index("twt_bin_index")
        cls = f["twt_bin_index"].map(L["class_name"]).fillna("unlabelled")
        codes = pd.Categorical(cls, categories=CLASS_ORDER).codes.astype(float)
        axs[0].imshow(codes[:, None], aspect="auto", interpolation="nearest",
                      cmap=ListedColormap([CLASS_COLORS[c] for c in CLASS_ORDER]), vmin=-0.5,
                      vmax=len(CLASS_ORDER) - 0.5, extent=[0, 1, t[-1] + 2, t[0] - 2])
    axs[0].set_xticks([])
    axs[0].set_title("class (4 ms)\nfigure only", fontsize=9)
    axs[0].set_ylabel("TWT (ms)")
    colours = [SERIES[0], SERIES[2], SERIES[1]]
    for ax, (title, cols) in zip(axs[1:], tracks[1:]):
        src = e if cols and cols[0].split("__")[0] in ("pimp", "simp", "vpvs") else f
        drawn = False
        for k, c in enumerate(cols):
            if c in src and src[c].notna().any():
                ax.plot(src[c], src["desired_twt_ms"], lw=0.9, color=colours[k % 3], label=c)
                drawn = True
        if not drawn:
            ax.text(0.5, 0.5, "NOT AVAILABLE\n(channel absent)", ha="center", va="center", rotation=90,
                    transform=ax.transAxes, color=STATUS["critical"], fontweight="bold")
        else:
            ax.legend(fontsize=7, loc="lower right")
        if "amp" in title:
            ax.axvline(0, color=NEUTRAL, lw=0.6)
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.set_title(title, fontsize=10)
    axs[0].set_ylim(t[-1] + 4, t[0] - 4)
    fig.suptitle(f"04_features: attributes along {w} (4 ms patch rows, real values; labels only for colour)",
                 fontsize=13, fontweight="bold")
    footer(fig, "Stack attributes come only from the five amplitude stacks; P-impedance and Vp/Vs are separate "
                "conditional inputs. Amplitudes in uncalibrated volume units; polarity as recorded.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ctx.save_figure(fig, "stack_attribute_examples.png", f"Attribute tracks along {w}", required=True, multipanel=True)


def _figure_distributions(ctx, feats, el, labels, wells, reserved):
    import matplotlib.pyplot as plt

    both = feats.merge(el, on=KEYS, how="left")
    if labels is not None:
        both = both.merge(labels[["well", "twt_bin_index", "y_lithology"]], on=["well", "twt_bin_index"], how="left")
    show = [c for c in ("full__rms_long", "near_far__rms_nd", "pimp__c", "full__env_c", "full__semblance", "vpvs__c")
            if c in both and both[c].notna().any()]
    fig, axs = plt.subplots(2, 3, figsize=(20, 11))
    for ax, c in zip(axs[0], show[:3]):
        data = [both.loc[both["well"] == w, c].dropna().to_numpy() for w in wells]
        keep = [i for i, d in enumerate(data) if d.size]
        if keep:
            ax.boxplot([data[i] for i in keep], vert=False, showfliers=False, widths=0.6)
            ax.set_yticks(range(1, len(keep) + 1))
            ax.set_yticklabels([f"{wells[i]}{' (res.)' if wells[i] in reserved else ''}" for i in keep], fontsize=8)
        ax.ticklabel_format(axis="x", useOffset=False)
        ax.set_title(f"{c} by well", fontsize=10)
    dev = both[~both["well"].isin(reserved)] if "y_lithology" in both else both.iloc[0:0]
    for ax, c in zip(axs[1], show[3:6] if len(show) > 3 else show[:3]):
        if dev.empty:
            ax.text(0.5, 0.5, "labels not available", ha="center", transform=ax.transAxes)
            continue
        for val, name, col in ((1, "sand", CLASS_COLORS["undetermined_sand"]), (0, "shale", CLASS_COLORS["shale"])):
            v = dev.loc[dev["y_lithology"] == val, c].dropna()
            if len(v):
                _safe_hist(ax, v, bins=40, density=True, histtype="step", lw=2, color=col, label=f"{name} (n={len(v)})")
        ax.set_title(f"{c}: sand vs shale bins (development wells)", fontsize=10)
        ax.legend(fontsize=8)
    for ax in axs.ravel()[len(show):]:
        ax.set_axis_off()
    fig.suptitle("04_features: feature distributions (exploration only)", fontsize=13, fontweight="bold")
    footer(fig, "Bottom row uses 4-ms labels for colour only. Feature SELECTION happens later inside training "
                "folds (stage 06); nothing here may be used to choose features.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ctx.save_figure(fig, "feature_distributions_by_well.png", "Per-well and sand/shale feature distributions",
                    required=True, multipanel=True)


def _safe_hist(ax, values, **kw):
    """Histogram that also copes with constant data (draws a single bar instead of failing)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return
    span = v.max() - v.min()
    bins = kw.pop("bins", 40)
    if span <= 1e-12 * max(1.0, abs(v.max())):
        bins = [v.min() - 0.5, v.min() + 0.5]
    ax.hist(v, bins=bins, **kw)


def _banner(fig, text, y=0.945):
    fig.text(0.5, y, text, ha="center", fontsize=11, fontweight="bold", color="white",
             bbox={"boxstyle": "round,pad=0.4", "facecolor": STATUS["critical"], "edgecolor": "none"})


def _figure_avo_events(ctx, ev, mode, partial, angles):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
    if ev.empty:
        for ax in axs.ravel():
            ax.set_axis_off()
        axs[1, 1].text(0.5, 0.5, "no events picked (no full stack)", ha="center", transform=axs[1, 1].transAxes)
    else:
        e = ev.copy()
        e["_pri"] = e.get("y_lithology_display_only", pd.Series(-1, index=e.index)).fillna(-1).map({1: 0, 0: 1}).fillna(2)
        e = e.sort_values(["_pri", "full_amp"], key=lambda s: s if s.name == "_pri" else -s.abs()).head(12)
        for ax, (_, r) in zip(axs.ravel(), e.iterrows()):
            amps = np.array([r.get(f"{s}_amp", np.nan) for s in partial], dtype=float)
            if mode == "QUANTITATIVE":
                x = np.sin(np.radians([angles[s] for s in partial])) ** 2
                ax.plot(x, amps, "o", color=SERIES[0])
                xx = np.linspace(0, x.max() * 1.1, 20)
                ax.plot(xx, r["shuey_A"] + r["shuey_B"] * xx, color=SERIES[1], lw=1.4,
                        label=f"A={r['shuey_A']:.3g} B={r['shuey_B']:.3g}")
                ax.legend(fontsize=7)
            else:
                x = np.arange(1, len(partial) + 1)
                ax.plot(x, amps, "o-", color=SERIES[0])
                ax.plot(x, r["proxy_intercept_vs_order"] + r["proxy_slope_vs_order"] * x, "--", color=NEUTRAL, lw=1)
                ax.set_xticks(x)
                ax.set_xticklabels(partial, fontsize=8)
            ax.axhline(0, color=NEUTRAL, lw=0.6)
            ax.set_title(f"{r['well']} {r['event_twt_ms']:.0f} ms {r['event_polarity']} "
                         f"[{r.get('class_at_event_display_only', '?')}]", fontsize=9)
    fig.suptitle("04_features: event amplitudes on the partial stacks", fontsize=13, fontweight="bold", y=0.99)
    if mode != "QUANTITATIVE":
        _banner(fig, "QUALITATIVE PROXY ONLY - x axis is stack ORDER, not incidence angle (angle ranges undocumented)")
    footer(fig, "Events = local extrema of the full stack along the well path (|amp| >= 75th percentile); partial-stack "
                "amplitudes taken at the same-polarity extremum within +/-1 sample. Class in brackets is the 4-ms "
                "label, shown for context only.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    ctx.save_figure(fig, "avo_amplitude_vs_angle_by_event.png", f"Event amplitudes vs stack ({mode})",
                    required=True, multipanel=True)


def _figure_avo_crossplot(ctx, ev, mode):
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(19, 8.5))
    if not ev.empty:
        lab = ev.get("y_lithology_display_only", pd.Series(-1, index=ev.index)).fillna(-1)
        if mode == "QUANTITATIVE":
            x, y, xl, yl = ev["shuey_A"], ev["shuey_B"], "intercept A", "gradient B"
        else:
            x, y = ev.get("near_amp", pd.Series(np.nan, index=ev.index)), ev.get("far_amp", pd.Series(np.nan, index=ev.index)) - ev.get("near_amp", pd.Series(np.nan, index=ev.index))
            xl, yl = "near amplitude at event (proxy for intercept)", "far - near amplitude (proxy for gradient)"
        for val, name, col in ((1, "sand bins", CLASS_COLORS["undetermined_sand"]), (0, "shale bins", CLASS_COLORS["shale"]),
                               (-1, "unlabelled bins", NEUTRAL)):
            m = (lab == val).to_numpy()
            a.scatter(x[m], y[m], s=14, alpha=0.6, color=col, label=f"{name} (n={int(m.sum())})")
        a.axhline(0, color=TEXT_2, lw=0.8)
        a.axvline(0, color=TEXT_2, lw=0.8)
        a.set_xlabel(xl)
        a.set_ylabel(yl)
        a.legend(fontsize=8)
        a.set_title("Events coloured by the 4-ms lithology label (context, not truth for fluid)")
        s = ev["proxy_slope_vs_order"].dropna()
        for val, name, col in ((1, "sand bins", CLASS_COLORS["undetermined_sand"]), (0, "shale bins", CLASS_COLORS["shale"])):
            v = ev.loc[(lab == val) & ev["proxy_slope_vs_order"].notna(), "proxy_slope_vs_order"]
            if len(v):
                _safe_hist(b, v, bins=30, histtype="step", lw=2, density=True, color=col, label=f"{name} (n={len(v)})")
        b.set_xlabel("amplitude slope vs stack order (proxy)")
        b.set_title(f"Amplitude trend across partial stacks (n={len(s)} events)")
        b.legend(fontsize=8)
    fig.suptitle("04_features: AVO crossplot" + (" (Shuey A/B)" if mode == "QUANTITATIVE" else ""),
                 fontsize=13, fontweight="bold", y=0.99)
    if mode != "QUANTITATIVE":
        _banner(fig, "QUALITATIVE PROXY - not intercept/gradient: angles and amplitude preservation are unverified")
    footer(fig, "An AVO class is a reflection-event interpretation, never an oil/gas label. Polarity convention unverified.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    ctx.save_figure(fig, "avo_intercept_gradient_crossplot.png", f"AVO crossplot ({mode})", required=True,
                    multipanel=True)


def _figure_overview(ctx, ov, stacks):
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 3, figsize=(20, 7.5))
    if ov.empty:
        for ax in axs:
            ax.set_axis_off()
    else:
        for ax, src in zip(axs[:2], ("survey", "well_paths")):
            d = ov[ov["source"].str.startswith(src)]
            if d.empty:
                ax.text(0.5, 0.5, f"{src}: NOT AVAILABLE", ha="center", transform=ax.transAxes, color=STATUS["critical"])
                continue
            mid = 0.5 * (d["twt_from_ms"] + d["twt_to_ms"])
            for k, s in enumerate(stacks):
                if f"rms_{s}" in d and d[f"rms_{s}"].notna().any():
                    ax.plot(d[f"rms_{s}"], mid, "o-", color=([NEUTRAL] + SERIES)[k % 9], label=s)
            ax.set_ylim(4000, 0)
            ax.set_xlabel("RMS amplitude (volume units)")
            ax.set_ylabel("TWT bin centre (ms)")
            ax.set_title(d["source"].iloc[0], fontsize=10)
            ax.legend(fontsize=8)
        for src, col in (("survey", SERIES[0]), ("well_paths", SERIES[1])):
            d = ov[ov["source"].str.startswith(src)]
            if len(d) and "ratio_far_to_near" in d:
                axs[2].plot(d["ratio_far_to_near"], 0.5 * (d["twt_from_ms"] + d["twt_to_ms"]), "o-", color=col, label=src)
        axs[2].axvline(1, color=NEUTRAL, lw=0.8)
        axs[2].set_ylim(4000, 0)
        axs[2].set_xlabel("RMS far / RMS near")
        axs[2].set_title("Far/near RMS ratio by 1000-ms bin")
        axs[2].legend(fontsize=8)
    fig.suptitle("04_features: 1000-ms TWT overview of stack amplitudes", fontsize=13, fontweight="bold", y=0.995)
    _banner(fig, "TREND OVERVIEW ONLY - no AVO class is assigned to any 1000-ms bin", y=0.905)
    footer(fig, "Survey: the same random live IL/XL bins read from every stack. Wells: RMS of patch-centre amplitudes "
                "along the paths. A ratio trend can reflect processing (gain, NMO stretch) as much as rock properties.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig.subplots_adjust(top=0.82)
    ctx.save_figure(fig, "avo_time_interval_comparison.png", "RMS per stack per 1000-ms bin and far/near ratio",
                    required=True, multipanel=True)


# =============================================================================== summary
def _summarise(ctx, feats, el, dictionary, events, avo_mode, present, stacks, elastic):
    fcols = [c for c in feats.columns if c not in KEYS]
    ecols = [c for c in el.columns if c not in KEYS]
    forbidden = ("class", "y_lithology", "y_sand", "y_vsh", "y_phit", "y_swt", "vsh_", "phit_", "swt_", "lith_",
                 "fluid_", "label")
    ctx.check("no_label_columns_in_features", not any(k in c for c in fcols + ecols for k in forbidden),
              f"{len(fcols)} stack + {len(ecols)} elastic columns, none label-derived")
    ctx.check("dictionary_covers_every_column", set(fcols + ecols) <= set(dictionary["column"]),
              "every feature column is documented in feature_dictionary.csv")
    ctx.check("no_totalsandprob", not any("totalsandprob" in c for c in fcols + ecols), "TotalSandProbability absent")
    ctx.check("rows_match_patch_index", len(feats) == len(el), f"{len(feats)} rows")
    ctx.metric("n_stack_feature_columns", len(fcols))
    ctx.metric("n_elastic_feature_columns", len(ecols))
    ctx.metric("mean_nan_fraction", {
        "stack_features": round(float(feats[fcols].isna().mean().mean()), 4) if fcols else None,
        "elastic_features": round(float(el[ecols].isna().mean().mean()), 4) if ecols else None})
    # attributes that need a complete window must exist wherever the patch itself is complete
    comp = {}
    for s in [s for s in stacks if present.get(s)]:
        for a in ("amp_c", "env_c", "semblance", "spec_centroid_hz", "rms_long"):
            c = f"{s}__{a}"
            if c in feats:
                comp[c] = round(float(feats[c].notna().mean()), 4)
    ctx.metric("stack_attribute_completeness", comp)
    low = {c: v for c, v in comp.items() if v < 0.9}
    ctx.check("stack_attributes_mostly_complete", not low,
              f"window attributes non-NaN in >= 90% of rows; below: {low}" if low else
              f"{len(comp)} key stack attributes non-NaN in >= 90% of rows")
    ctx.metric("n_avo_events", int(len(events)))
    ctx.metric("avo_mode", avo_mode)
    ctx.metric("channels_present", present)
    ctx.limit("Inversion inputs (pimp/simp/vpvs) are conditional: provenance and calibration wells unaudited")
    ctx.readme("Feature policy", f"""
* {len(fcols)} stack-derived columns (13 attributes x {len([s for s in stacks if present.get(s)])} stacks with data + proxies),
  {len(ecols)} elastic columns (centre, local mean, local std).
* Amplitudes are raw (no scaling here). Normalisation happens inside training folds (stage 06).
* AVO mode: **{avo_mode}**. Configure `avo.angle_ranges_deg`, `avo.amplitude_preservation_verified` and
  `avo.angle_provenance` only from documented processing to enable Shuey fits.
* `feature_dictionary.csv` lists the predictor policy of every column (allowed / conditional / never).
""")
