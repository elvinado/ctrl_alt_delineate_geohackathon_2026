"""05_training_tables - one auditable table of X (predictors) and y (targets) per resolution, plus the split plan.

What this stage answers
-----------------------
"Exactly which numbers may a model see, which numbers is it asked to predict, for
which rows, and which wells test it?"  Nothing is scaled, imputed or resampled here:
the tables hold raw predictors, raw targets and masks, so every later number can be
traced back.  Normalisation and feature selection happen inside training folds (06).

Steps
-----
1. Upstream     - 03 labels (4/12/24 ms) + sample index, 04 stack attributes, elastic inputs, dictionary.
2. Join         - strictly on (well, twt_bin_index); patch_row and TWT reconciled across tables.
3. Columns      - keys | meta__* | predictors (allowlist from 04) | y_* targets | mask_* | ysoft__* .
4. Split plan   - whole wells only; no-1 reserved; balanced outer folds + nested inner folds
                  chosen by exhaustive search (splits.py); distances and patch overlap reported.
5. Leakage      - predictors never overlap targets, never label-derived, no duplicates,
                  tripwire for target-like predictors, reserved well never in a fold.
6. Write        - training_samples_all.csv (4 ms), 12/24 ms ablation tables, X / y files, per-well
                  files, feature_schema.json, training_manifest.json, figures.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..stage_compat import StageBlocked, SpecialistStageContext as StageContext
from ..config import cfg_get
from ..plotting import CLASS_COLORS, COORD_NOTE, NEUTRAL, SERIES, footer, setup_style
from ..progress import StepTracker, progress
from ..splits import nested_plan

STEPS = ["upstream", "join", "columns", "split plan", "leakage checks", "write tables", "figures", "summary"]
META_FROM_INDEX = ["md_centre_m", "tvdss_centre_m", "x", "y", "il", "xl", "grid_i", "grid_j", "trace_offset_xy_m",
                   "tie_status", "twt_uncertainty_rms_ms", "tz_strong_extrapolation", "reserved_reviewed_well",
                   "patch_status", "patch_complete_stacks"]
META_FROM_LABELS = ["time_coverage", "n_log_samples", "label_grade", "label_mask_reason", "label_version",
                    "centroid_offset_ms", "class_name"]
TARGETS = {"y_lithology": "mask_lithology", "y_sand_fluid": "mask_sand_fluid", "y_class_id": "mask_class",
           "y_vsh": "mask_vsh", "y_phit": "mask_phit", "y_swt": "mask_swt"}
BALANCE = {"n_sand": 1.0, "n_shale": 0.5, "n_water_sand": 1.0, "n_hc_potential_sand": 1.5, "n_vsh": 0.5}


def run(ctx: StageContext) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_style()
    steps = StepTracker(STEPS, ctx.stage)
    cfg = ctx.cfg
    sp = cfg.get("split") or {}
    wells = [w.lower() for w in cfg_get(cfg, "wells.expected", [])]
    reserved = [w.lower() for w in (cfg_get(cfg, "wells.reserved_reviewed", []) or [])]
    version = str(cfg_get(cfg, "extraction.version", "v1"))
    run_dir = ctx.run.run_dir

    # ---------------------------------------------------------------- 1 upstream
    steps.next("upstream")
    ex, fe = run_dir / "03_extraction" / "tables", run_dir / "04_features" / "tables"
    need = {"sample_index": ex / "sample_index.csv", "stack": fe / "stack_attributes.csv",
            "elastic": fe / "elastic_inputs.csv", "dictionary": fe / "feature_dictionary.csv"}
    for name, p in need.items():
        if not p.exists():
            raise StageBlocked(f"Missing {p}: run 03_extraction and 04_features first")
        ctx.add_input(p, f"upstream:{name}")
    idx = pd.read_csv(need["sample_index"])
    if not wells:
        wells = idx["well"].dropna().astype(str).str.casefold().drop_duplicates().tolist()
    stack = pd.read_csv(need["stack"])
    elastic = pd.read_csv(need["elastic"])
    dictionary = pd.read_csv(need["dictionary"])
    resolutions = [int(r) for r in (cfg_get(cfg, "extraction.resolutions_ms", [4, 12, 24]) or [4, 12, 24])]
    labels = {}
    for r in resolutions:
        p = ex / f"labels_{r}ms_{version}.csv"
        if p.exists():
            ctx.add_input(p, f"upstream:labels_{r}ms")
            labels[r] = pd.read_csv(p)
    if not labels:
        raise StageBlocked("No 03 label tables found")
    base = min(labels)
    support = _task_support(ctx, run_dir)

    # ---------------------------------------------------------------- 2 join
    steps.next("join")
    keys = ["patch_row", "well", "twt_bin_index", "desired_twt_ms"]
    for name, t in (("stack_attributes", stack), ("elastic_inputs", elastic)):
        same = (len(t) == len(idx)) and all((t[k].to_numpy() == idx[k].to_numpy()).all() for k in keys)
        ctx.check(f"{name}_rows_reconcile_with_sample_index", bool(same),
                  f"{len(t)} rows; patch_row, well, twt_bin_index and TWT identical row by row")
        if not same:
            raise StageBlocked(f"{name}.csv rows do not reconcile with sample_index.csv - rerun 04_features")
    pred_stack = [c for c in stack.columns if c not in keys]
    pred_el = [c for c in elastic.columns if c not in keys]
    features = idx[keys + [c for c in META_FROM_INDEX if c in idx]].merge(
        stack, on=keys, how="left").merge(elastic, on=keys, how="left")

    # ---------------------------------------------------------------- 3 columns per resolution
    steps.next("columns")
    tables = {}
    for r, lab in labels.items():
        tables[r] = _assemble(features, lab, r, keys, pred_stack, pred_el, support)
    main = tables[base]

    # ---------------------------------------------------------------- 4 split plan
    steps.next("split plan")
    dev = [w for w in wells if w not in reserved and (main["well"] == w).any()]
    counts = _well_counts(main, wells)
    n_outer, n_inner = int(sp.get("n_outer_folds", 4)), int(sp.get("n_inner_folds", 3))
    weights = {k: float(v) for k, v in (sp.get("balance_weights") or BALANCE).items()}
    if len(dev) < 2:
        raise StageBlocked(f"Only {len(dev)} development wells - whole-well validation is impossible")
    if len(dev) < n_outer:
        ctx.warn(f"{len(dev)} development wells < {n_outer} outer folds: using {len(dev)} folds (leave-one-well-out)")
    plan = nested_plan(dev, counts, weights, n_outer, n_inner)
    split = _split_table(plan, wells, reserved, counts, idx)
    overlap = _patch_overlap(idx, int(sp.get("purge_il_bins", 4)), int(sp.get("purge_xl_bins", 4)),
                             float(sp.get("purge_twt_ms", 80)))
    fold_of = dict(zip(split["well"], split["outer_fold"]))
    for r in tables:
        tables[r]["outer_fold"] = tables[r]["well"].map(fold_of).astype(int)
        tables[r]["split_role"] = np.where(tables[r]["well"].isin(reserved), "reserved_reviewed", "development")

    # ---------------------------------------------------------------- 5 leakage checks
    steps.next("leakage checks")
    predictors = pred_stack + pred_el
    leak = _leakage_checks(ctx, main, predictors, dictionary, reserved, split)

    # ---------------------------------------------------------------- 6 write
    steps.next("write tables")
    _write(ctx, tables, base, keys, predictors, pred_stack, pred_el, dictionary, split, overlap, plan, wells,
           reserved, support, leak, weights)

    # ---------------------------------------------------------------- 7 figures
    steps.next("figures")
    _figure_class_balance(ctx, tables, base, wells, reserved, fold_of)
    _figure_coverage(ctx, main, wells, reserved, pred_el)
    _figure_split_map(ctx, idx, split, plan, wells, reserved)

    # ---------------------------------------------------------------- 8 summary
    steps.next("summary")
    _summarise(ctx, tables, base, plan, split, overlap, support, predictors, pred_el)
    steps.close()


# =============================================================================== helpers
def _task_support(ctx, run_dir: Path) -> dict:
    p = run_dir / "02_labels" / "tables" / "task_support.csv"
    if not p.exists():
        ctx.warn("02_labels task_support.csv missing: class trainability taken from the class name only")
        return {}
    ctx.add_input(p, "upstream:02 task_support")
    t = pd.read_csv(p)
    return dict(zip(t["task"], t["status"]))


def _assemble(features, lab, res, keys, pred_stack, pred_el, support):
    """One row per labelled bin: keys | meta | predictors | targets | masks | soft labels."""
    L = lab.copy()
    lab_keys = ["well", "twt_bin_index"]
    out = L[lab_keys + ["patch_row"]].merge(features.drop(columns=["patch_row"]), on=lab_keys, how="left")
    out.insert(0, "sample_id", [f"{w}|{res}ms|{k}" for w, k in zip(out["well"], out["twt_bin_index"])])
    out.insert(2, "resolution_ms", res)
    meta = {}
    for c in META_FROM_LABELS:
        if c in L:
            meta[f"meta__{c}"] = L[c].to_numpy()
    for c in [c for c in out.columns if c in set(META_FROM_INDEX)]:
        meta[f"meta__{c}"] = out.pop(c).to_numpy()
    out = pd.concat([out, pd.DataFrame(meta, index=out.index)], axis=1)
    # ---- targets and masks (raw values, NaN where masked: nothing imputed)
    trainable = {name: support.get(f"class:{name}", "SUPPORTED_PROVISIONAL") != "TASK_UNSUPPORTED"
                 for name in ("shale", "brine_sand", "oil_sand", "gas_sand", "undetermined_sand", "other_rock")}
    y = pd.DataFrame(index=out.index)
    y["y_lithology"] = L["y_lithology"].to_numpy()
    y["mask_lithology"] = y["y_lithology"] >= 0
    y["y_sand_fluid"] = L["y_sand_fluid"].to_numpy()
    y["mask_sand_fluid"] = y["y_sand_fluid"] >= 0
    y["y_class_id"] = L["class_id"].to_numpy()
    y["mask_class"] = [bool(cid >= 0 and trainable.get(name, False)) for cid, name in zip(L["class_id"], L["class_name"])]
    for t in ("vsh", "phit", "swt"):
        y[f"y_{t}"] = L[f"y_{t}"].to_numpy()
        y[f"mask_{t}"] = L[f"{t}_mask"].astype(bool).to_numpy()
        y[f"ysoft__{t}_within_bin_std"] = L[f"{t}_within_bin_std"].to_numpy()
    for c in [c for c in L.columns if c.startswith(("lith_frac_", "fluid_frac_", "class_frac_"))]:
        y[f"ysoft__{c}"] = L[c].to_numpy()
    out = pd.concat([out, y], axis=1)
    return out


def _well_counts(main: pd.DataFrame, wells: list[str]) -> pd.DataFrame:
    g = main.groupby("well")
    c = pd.DataFrame({
        "n_rows": g.size(),
        "n_sand": g["y_lithology"].apply(lambda s: int((s == 1).sum())),
        "n_shale": g["y_lithology"].apply(lambda s: int((s == 0).sum())),
        "n_water_sand": g["y_sand_fluid"].apply(lambda s: int((s == 0).sum())),
        "n_hc_potential_sand": g["y_sand_fluid"].apply(lambda s: int((s == 1).sum())),
        "n_vsh": g["mask_vsh"].sum().astype(int), "n_phit": g["mask_phit"].sum().astype(int),
        "n_swt": g["mask_swt"].sum().astype(int)})
    return c.reindex([w for w in wells if w in c.index]).fillna(0)


def _split_table(plan, wells, reserved, counts, idx):
    rows = []
    path_xy = idx.groupby("well")[["x", "y"]].agg(list)
    test_of = {w: f for f, g in enumerate(plan["outer"]) for w in g}

    def min_dist(a, others):
        if a not in path_xy.index or not others:
            return np.nan
        A = np.column_stack([path_xy.loc[a, "x"], path_xy.loc[a, "y"]])
        best = np.inf
        for o in others:
            if o not in path_xy.index:
                continue
            B = np.column_stack([path_xy.loc[o, "x"], path_xy.loc[o, "y"]])
            d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
            best = min(best, float(np.nanmin(d)))
        return best if np.isfinite(best) else np.nan

    dev = [w for g in plan["outer"] for w in g]
    for w in wells:
        r = {"well": w, "role": "reserved_reviewed" if w in reserved else
             ("development" if w in test_of else "no_rows"), "outer_fold": test_of.get(w, -1)}
        r.update({k: int(counts.loc[w, k]) if w in counts.index else 0 for k in counts.columns})
        for f, info in plan["inner"].items():
            inner_of = {x: i for i, g in enumerate(info["folds"]) for x in g}
            r[f"inner_fold_when_outer_{f}"] = inner_of.get(w, -1)
        if w in test_of:
            train = [x for x in dev if x not in plan["outer"][test_of[w]]]
            r["nearest_training_well_path_distance_m"] = min_dist(w, train)
        r["nearest_any_well_path_distance_m"] = min_dist(w, [x for x in wells if x != w])
        rows.append(r)
    return pd.DataFrame(rows)


def _patch_overlap(idx, di_max, dj_max, dt_max):
    """Pairs of rows from DIFFERENT wells whose patch footprints overlap (would need purging)."""
    rows = []
    wells = sorted(idx["well"].unique())
    for a_i, a in enumerate(wells):
        A = idx[idx["well"] == a]
        for b in wells[a_i + 1:]:
            B = idx[idx["well"] == b]
            di = np.abs(A["grid_i"].to_numpy()[:, None] - B["grid_i"].to_numpy()[None, :])
            dj = np.abs(A["grid_j"].to_numpy()[:, None] - B["grid_j"].to_numpy()[None, :])
            dt = np.abs(A["desired_twt_ms"].to_numpy()[:, None] - B["desired_twt_ms"].to_numpy()[None, :])
            hit = (di <= di_max) & (dj <= dj_max) & (dt <= dt_max)
            lat = np.sqrt(di.astype(float) ** 2 + dj.astype(float) ** 2)
            rows.append({"well_a": a, "well_b": b, "min_lateral_distance_bins": float(lat.min()),
                         "n_overlapping_row_pairs": int(hit.sum()),
                         "purge_needed": bool(hit.any())})
    return pd.DataFrame(rows)


def _leakage_checks(ctx, main, predictors, dictionary, reserved, split):
    res = {}
    target_cols = [c for c in main.columns if c.startswith(("y_", "mask_", "ysoft__"))]
    meta_cols = [c for c in main.columns if c.startswith("meta__")]
    overlap = sorted(set(predictors) & set(target_cols + meta_cols))
    res["predictors_disjoint_from_targets_and_meta"] = not overlap
    ctx.check("predictors_disjoint_from_targets_and_meta", not overlap, f"{len(predictors)} predictors; overlap {overlap}")
    pol = dict(zip(dictionary["column"], dictionary["predictor_policy"].astype(str)))
    bad = [c for c in predictors if c not in pol or pol[c].startswith("never")]
    res["predictors_documented_and_allowed"] = not bad
    ctx.check("predictors_documented_and_allowed", not bad, f"undocumented/never: {bad}")
    tokens = {"class", "lith", "lithology", "fluid", "label", "vsh", "phit", "swt", "mask", "target"}
    sus = [c for c in predictors if c.startswith(("y_", "mask_", "ysoft__", "meta__"))
           or tokens & set(re.split(r"_+", c.lower()))]
    res["no_label_named_predictors"] = not sus
    ctx.check("no_label_named_predictors", not sus, f"suspicious names: {sus}")
    X = main[predictors].to_numpy(float)
    dup = []
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            a, b = X[:, i], X[:, j]
            if np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)]):
                dup.append((predictors[i], predictors[j]))
    res["no_duplicate_predictors"] = not dup
    ctx.check("no_duplicate_predictors", not dup, f"identical columns: {dup[:5]}")
    # tripwire only (never used for selection): a predictor that mirrors a target suggests leakage
    dev = main[~main["well"].isin(reserved)]
    trip = []
    for t in ("y_vsh", "y_phit", "y_swt", "y_lithology"):
        m = dev[TARGETS[t]].astype(bool)
        if m.sum() < 30:
            continue
        yv = dev.loc[m, t].rank()
        for c in predictors:
            xv = dev.loc[m, c]
            ok = xv.notna()
            if ok.sum() < 30:
                continue
            rho = float(np.corrcoef(xv[ok].rank(), yv[ok])[0, 1])
            if abs(rho) > 0.95:
                trip.append((c, t, round(rho, 3)))
    res["target_like_predictor_tripwire"] = not trip
    ctx.check("no_target_like_predictor", not trip, "tripwire |Spearman| > 0.95 on development rows "
              f"(diagnostic only, never used for selection): {trip}")
    in_folds = split.loc[split["well"].isin(reserved), "outer_fold"]
    res["reserved_wells_outside_all_folds"] = bool((in_folds == -1).all())
    ctx.check("reserved_wells_outside_all_folds", bool((in_folds == -1).all()),
              f"reserved {reserved}: outer_fold = -1 and never in any inner fold")
    return {"checks": res, "duplicates": dup, "tripwire": trip, "overlap": overlap}


def _write(ctx, tables, base, keys, predictors, pred_stack, pred_el, dictionary, split, overlap, plan, wells,
           reserved, support, leak, weights):
    main = tables[base]
    units = {"desired_twt_ms": "ms", "y_vsh": "v/v", "y_phit": "v/v", "y_swt": "v/v"}
    ctx.save_table(main, "training_samples_all.csv", f"{base} ms: one row per labelled bin - keys, meta__, raw "
                   "predictors, y_ targets, mask_ columns, ysoft__ soft labels, outer_fold", units=units, required=True)
    for r, t in tables.items():
        if r != base:
            ctx.save_table(t, f"training_samples_{r}ms.csv", f"{r} ms ablation dataset (same schema; features of "
                           "the patch at the bin centre)", units=units)
    for w in progress(wells, desc="per-well tables", unit="well"):
        d = main[main["well"] == w]
        if d.empty:
            d = pd.DataFrame([{"well": w, "not_available_reason": "no labelled rows"}])
        ctx.save_table(d, f"per_well/{w}_training.csv", f"{w}: training rows ({base} ms)", required=True)
    idcols = ["sample_id", "well", "outer_fold", "split_role"]
    ctx.save_table(main[idcols + predictors], "X_features_raw.csv", "Raw predictors (allowlist), NOT normalised",
                   required=True)
    ylab = [c for c in main.columns if c in ("y_lithology", "mask_lithology", "y_sand_fluid", "mask_sand_fluid",
                                              "y_class_id", "mask_class") or c.startswith("ysoft__lith_frac")
            or c.startswith("ysoft__fluid_frac") or c.startswith("ysoft__class_frac")]
    prov = [c for c in ("meta__label_version", "meta__label_grade", "meta__label_mask_reason", "meta__class_name")
            if c in main]
    ctx.save_table(main[idcols + ylab + prov], "y_labels.csv", "Class targets, masks, soft labels, provenance",
                   required=True)
    for t in ("vsh", "phit", "swt"):
        cols = [f"y_{t}", f"mask_{t}", f"ysoft__{t}_within_bin_std"]
        ctx.save_table(main[idcols + cols + prov[:2]], f"y_{t}.csv", f"{t.upper()} target (bin mean of valid log "
                       "values), mask and within-bin std", units={f"y_{t}": "v/v"}, required=True)
    ctx.save_table(split, "split_plan_wells.csv", "Whole-well outer test fold and inner fold of every well; "
                   "label counts; distances to the nearest training well", required=True)
    ctx.save_table(overlap, "patch_overlap_between_wells.csv", "Well pairs whose patch footprints overlap "
                   "(rows to purge when they sit on opposite sides of a split)")
    miss = main.groupby("well")[predictors].apply(lambda d: d.isna().mean()).T
    ctx.save_table(pd.DataFrame({"predictor": miss.index, "group": ["elastic" if c in pred_el else "stacks"
                                                                    for c in miss.index]}).join(miss.reset_index(drop=True)),
                   "predictor_missingness_by_well.csv", "Fraction of NaN per predictor and well (imputation is "
                   "fold-local in stage 06)")
    pol = dict(zip(dictionary["column"], dictionary["predictor_policy"]))
    schema = {
        "version": f"training_tables_{cfg_get(ctx.cfg, 'extraction.version', 'v1')}",
        "row_key": ["well", "resolution_ms", "twt_bin_index"], "sample_id": "well|<res>ms|twt_bin_index",
        "patch_link": "patch_row -> 03_extraction/patches/patches_4ms_<version>.npy[row]",
        "predictor_groups": {"stacks": pred_stack, "elastic": pred_el, "fused": pred_stack + pred_el},
        "predictor_policy": {c: pol.get(c, "") for c in predictors},
        "conditional_predictors": pred_el,
        "never_predictors": ["sample_id", "well", "outer_fold", "split_role", "patch_row", "twt_bin_index",
                             "desired_twt_ms", "resolution_ms"] + [c for c in main.columns if c.startswith(
                                 ("meta__", "y_", "mask_", "ysoft__"))] + ["totalsandprob*"],
        "targets": {t: {"mask": m, "unit": "v/v" if t in ("y_vsh", "y_phit", "y_swt") else "class id"}
                    for t, m in TARGETS.items()},
        "target_encoding": {"y_lithology": {"1": "sand", "0": "shale", "-1": "masked"},
                            "y_sand_fluid": {"0": "water_bearing", "1": "hydrocarbon_potential", "-1": "masked"},
                            "y_class_id": "02 class map (geogg_fresh_classes_v1); use only where mask_class"},
        "task_support_from_02": support,
        "datasets": {f"{r}ms": ("training_samples_all.csv" if r == base else f"training_samples_{r}ms.csv")
                     for r in tables},
        "rules": ["predictors are RAW: fit imputation and scaling inside training folds only (06)",
                  "never train on rows where the target mask is False",
                  "never use reserved wells for fitting, selection, thresholds or calibration",
                  "12/24 ms rows are independent bins; do not mix resolutions in one fit"],
    }
    ctx.save_json(schema, "feature_schema.json", "Predictor allowlist, target/mask map, encodings and rules",
                  required=True)
    manifest = {
        "rows": {f"{r}ms": int(len(t)) for r, t in tables.items()},
        "rows_by_well": {f"{r}ms": t["well"].value_counts().to_dict() for r, t in tables.items()},
        "n_predictors": {"stacks": len(pred_stack), "elastic": len(pred_el)},
        "split": {"outer_folds": [list(g) for g in plan["outer"]],
                  "inner_folds": {str(f): [list(g) for g in v["folds"]] for f, v in plan["inner"].items()},
                  "balance_weights": weights, "outer_score": plan["outer_score"],
                  "n_outer_partitions_scored": plan["n_outer_partitions_scored"],
                  "selection_rule": "exhaustive search; lowest weighted squared relative imbalance; ties broken "
                                    "alphabetically (deterministic, no random seed)"},
        "reserved_wells": {"wells": reserved, "statement": "excluded from every outer and inner fold by "
                           "config wells.reserved_reviewed (previously reviewed; never used for supervised fitting)"},
        "leakage_checks": leak["checks"],
        "inputs": {k: v.get("sha256") for k, v in ctx.inputs.items()},
        "code_version": ctx.run.code_version,
    }
    ctx.save_json(manifest, "training_manifest.json", "Row counts, split, reserved-well statement, leakage checks, "
                  "input hashes", required=True)


# =============================================================================== figures
FOLD_COLORS = [SERIES[0], SERIES[1], SERIES[2], SERIES[6], SERIES[4], SERIES[5]]


def _figure_class_balance(ctx, tables, base, wells, reserved, fold_of):
    import matplotlib.pyplot as plt

    res = sorted(tables)
    fig, axs = plt.subplots(1, len(res), figsize=(7 * len(res), 9), sharey=True)
    axs = np.atleast_1d(axs)
    y = np.arange(len(wells))[::-1]
    names = [f"{w} ({'reserved' if w in reserved else 'fold ' + str(fold_of.get(w, '-'))})" for w in wells]
    for ax, r in zip(axs, res):
        t = tables[r]
        cats = [("shale", t["y_lithology"] == 0, CLASS_COLORS["shale"]),
                ("water-bearing sand", t["y_sand_fluid"] == 0, CLASS_COLORS["brine_sand"]),
                ("HC-potential sand", t["y_sand_fluid"] == 1, SERIES[1]),
                ("sand, fluid masked", (t["y_lithology"] == 1) & (t["y_sand_fluid"] < 0), CLASS_COLORS["undetermined_sand"]),
                ("no lithology label", t["y_lithology"] < 0, "#e6e5e1")]
        left = np.zeros(len(wells))
        for name, m, col in cats:
            v = t[m].groupby("well").size().reindex(wells).fillna(0).to_numpy()
            ax.barh(y, v, left=left, color=col, edgecolor="white", lw=0.4, label=name)
            left += v
        ax.set_title(f"{r} ms rows")
        ax.set_xlabel("rows (independent bins at this resolution)")
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(names)
    axs[-1].legend(fontsize=8, loc="lower right")
    fig.suptitle("05_training_tables: class balance by well and resolution (no resampling applied)",
                 fontsize=13, fontweight="bold")
    footer(fig, "Counts are as labelled; balancing (if any) is a model-side weighting decided inside training folds.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ctx.save_figure(fig, "class_balance_by_well.png", "Class/target balance per well and resolution", required=True,
                    multipanel=True)


def _figure_coverage(ctx, main, wells, reserved, pred_el):
    import matplotlib.pyplot as plt

    cols = [("lithology", "mask_lithology"), ("sand fluid", "mask_sand_fluid"), ("class", "mask_class"),
            ("VSH", "mask_vsh"), ("PHIT", "mask_phit"), ("SWT", "mask_swt")]
    if pred_el:
        cols.append(("elastic inputs\n(non-NaN)", None))
    M = np.zeros((len(wells), len(cols)))
    for i, w in enumerate(wells):
        d = main[main["well"] == w]
        for j, (_, m) in enumerate(cols):
            if m is None:
                M[i, j] = float(d[pred_el].notna().all(axis=1).sum()) if len(d) else 0
            else:
                M[i, j] = float(d[m].sum()) if len(d) else 0
    fig, ax = plt.subplots(figsize=(16, 9))
    im = ax.imshow(M, cmap="Blues", aspect="auto")
    for i in range(len(wells)):
        for j in range(len(cols)):
            ax.text(j, i, f"{int(M[i, j])}", ha="center", va="center", fontsize=9,
                    color="white" if M[i, j] > 0.6 * M.max() else "#0b0b0b")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c for c, _ in cols])
    ax.set_yticks(range(len(wells)))
    ax.set_yticklabels([f"{w} (reserved)" if w in reserved else w for w in wells])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="rows with a valid target (4 ms)")
    ax.set_title("Usable rows per well and target (4 ms); elastic column = rows with all inversion inputs present")
    footer(fig, "Masked targets stay NaN in the tables. Elastic inputs are NaN where the inversion volumes are zero-filled.")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    ctx.save_figure(fig, "target_coverage_heatmap.png", "Valid targets per well (4 ms)", required=True)


def _figure_split_map(ctx, idx, split, plan, wells, reserved):
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(19, 9), gridspec_kw={"width_ratios": [1.2, 1]})
    fold_of = dict(zip(split["well"], split["outer_fold"]))
    for w in wells:
        d = idx[idx["well"] == w]
        if d.empty:
            continue
        f = fold_of.get(w, -1)
        col = NEUTRAL if f < 0 else FOLD_COLORS[f % len(FOLD_COLORS)]
        a.plot(d["x"], d["y"], color=col, lw=3)
        a.scatter(d["x"].iloc[0], d["y"].iloc[0], s=40, color=col, marker="s" if w in reserved else "o", zorder=3)
        a.text(d["x"].iloc[-1], d["y"].iloc[-1], f"  {w} ({'reserved' if w in reserved else 'F' + str(f)})",
               fontsize=9, va="center")
    for f in range(len(plan["outer"])):
        a.plot([], [], color=FOLD_COLORS[f % len(FOLD_COLORS)], lw=3, label=f"outer test fold {f}")
    a.plot([], [], color=NEUTRAL, lw=3, label="reserved (never in a fold)")
    a.set_aspect("equal", adjustable="datalim")
    a.set_xlabel("X")
    a.set_ylabel("Y")
    a.legend(fontsize=8, loc="best")
    a.set_title("Well paths (patch rows) coloured by outer TEST fold")
    b.set_axis_off()
    lines = ["Outer folds (test wells) and their inner validation folds", ""]
    cnt = split.set_index("well")
    for f, g in enumerate(plan["outer"]):
        s = cnt.loc[list(g)]
        lines.append(f"F{f}: test {', '.join(g)}   sand {int(s['n_sand'].sum())}, shale {int(s['n_shale'].sum())}, "
                     f"water {int(s['n_water_sand'].sum())}, HC-pot {int(s['n_hc_potential_sand'].sum())}")
        inner = plan["inner"].get(f, {}).get("folds", [])
        lines.append("     inner: " + " | ".join(",".join(x) for x in inner) if inner else "     inner: none")
    lines += ["", f"reserved: {', '.join(reserved) or 'none'}",
              f"{plan['n_outer_partitions_scored']} groupings scored; best imbalance score {plan['outer_score']:.3f}"]
    b.text(0.0, 1.0, "\n".join(lines), va="top", family="monospace", fontsize=9.5)
    fig.suptitle("05_training_tables: whole-well split plan", fontsize=13, fontweight="bold")
    footer(fig, COORD_NOTE)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ctx.save_figure(fig, "train_test_well_map.png", "Wells coloured by outer fold + fold composition", required=True,
                    multipanel=True)


# =============================================================================== summary
def _summarise(ctx, tables, base, plan, split, overlap, support, predictors, pred_el):
    main = tables[base]
    ids = pd.concat([t["sample_id"] for t in tables.values()])
    ctx.check("unique_sample_ids", not ids.duplicated().any(), f"{len(ids)} sample ids over {len(tables)} resolutions")
    ctx.check("every_development_well_in_exactly_one_outer_fold",
              sorted(w for g in plan["outer"] for w in g) == sorted(split.loc[split["role"] == "development", "well"]),
              f"{len(plan['outer'])} outer folds")
    for r, t in tables.items():
        ctx.metric(f"rows_{r}ms", int(len(t)))
        ctx.metric(f"usable_targets_{r}ms", {k: int(t[m].sum()) for k, m in TARGETS.items()})
    ctx.metric("outer_folds", [list(g) for g in plan["outer"]])
    ctx.metric("n_predictors", len(predictors))
    if pred_el:
        frac = float(main[pred_el].notna().all(axis=1).mean())
        ctx.metric("fraction_rows_with_all_elastic_inputs_4ms", round(frac, 4))
        if frac < 0.99:
            ctx.limit(f"Only {100 * frac:.1f}% of {base}-ms rows have every inversion input (zero-filled outside the "
                      "inversion window): Elastic/Fused comparisons must use matched rows (stage 07)")
    if overlap["purge_needed"].any():
        ctx.limit("Some patch footprints of different wells overlap: purge those rows across splits "
                  f"({overlap.loc[overlap['purge_needed'], ['well_a', 'well_b']].values.tolist()})")
    unsupported = [k for k, v in support.items() if v == "TASK_UNSUPPORTED"]
    if unsupported:
        ctx.limit(f"Targets inherited as TASK_UNSUPPORTED from 02: {unsupported} (their classes are masked)")
    ctx.limit("Targets are provisional log interpretations, tied with a provisional T-Z (see 02/03 limitations)")
    ctx.readme("How to use these tables", f"""
* Primary dataset: `training_samples_all.csv` ({base} ms). Ablations: `training_samples_<12|24>ms.csv`.
* Predictors = the `fused` list in `feature_schema.json` (`stacks` and `elastic` for the ablations). They are RAW.
* Train a target only on rows where its `mask_*` is True. `ysoft__*` are optional soft labels (time shares).
* Outer test folds: {', '.join('F' + str(i) + '=' + '+'.join(g) for i, g in enumerate(plan['outer']))}.
  `outer_fold` in every row says in which fold that row's well is the TEST well; `-1` = reserved.
""")
