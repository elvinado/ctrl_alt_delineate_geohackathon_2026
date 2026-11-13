"""06_feature_selection - frozen fold-local transforms and well-level feature statistics.

What this stage answers
-----------------------
"Which predictors carry information that holds up in wells the selection never saw, and
what exact, frozen transform will every model apply?"  Everything is fitted inside the
training wells of each fold; the outer TEST wells of a fold are never read for that fold,
and the reserved well (no-1) is dropped when the tables are loaded.

Steps (progress bars in the terminal)
-------------------------------------
1. Upstream          - 05 tables (4 / 12 / 24 ms), feature_schema.json, split_plan_wells.csv.
2. Per-well effects  - for every development well, task and predictor: the within-well effect
                       (rank-biserial r for sand-vs-shale and water-vs-HC-potential, Spearman rho
                       for VSH / PHIT / SWT).  Rank-based: insensitive to each well's amplitude level.
3. Folds             - for each outer fold (fit wells = development wells outside its test wells)
                       and for "all development" (the final model):
                       a. transforms   median / IQR scaling, clipping, imputation, missing indicators
                                       (preprocess.py) - frozen as JSON;
                       b. screening    label-free: too many NaN, near-constant;
                       c. redundancy   label-free: complete-linkage clusters of |Spearman| >= 0.90,
                                       one representative each;
                       d. tests        random-effects meta-analysis of the per-well effects over the
                                       fit wells (HKSJ t-test, k-1 df), Benjamini-Hochberg q, sign
                                       consistency; documented selection rule (selection.py).
4. Inner replication - the same procedure inside every outer training set: select on two inner
                       groups, then check the direction of each selected predictor in the wells of
                       the third group (never seen by that selection).  Also gives the stability
                       (how often a predictor is selected).
5. Missingness       - do the missing-value indicators predict a target? (coverage, not geology)
6. Write, 7. Figures, 8. Summary.

Nothing here trains a predictive model or reads an outer test well's targets; held-out
performance belongs to stages 07/08.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..stage_compat import StageBlocked, SpecialistStageContext as StageContext
from ..config import cfg_get
from ..plotting import CLASS_COLORS, NEUTRAL, SERIES, STATUS, TEXT_2, footer, setup_style
from ..preprocess import apply_transform, common_groups, fit_transform_params, params_table
from ..progress import StepTracker, progress
from ..selection import (aggregate_effects, apply_selection_rule, benjamini_hochberg, per_well_effects,
                         pooled_effect_xy, rank_correlation, redundancy_clusters, screen_features)

STEPS = ["upstream", "per-well effects", "folds", "inner replication", "missingness", "write", "figures", "summary"]

# task -> (target, mask, kind, task_support key in 02, meaning of a POSITIVE effect)
TASKS = {
    "lithology": ("y_lithology", "mask_lithology", "binary", "hierarchical_1_lithology_sand_vs_shale",
                  "higher in sand than in shale"),
    "sand_fluid": ("y_sand_fluid", "mask_sand_fluid", "binary", "hierarchical_2_sand_water_vs_hc_potential",
                   "higher in hydrocarbon-potential sand than in water-bearing sand"),
    "vsh": ("y_vsh", "mask_vsh", "regression", "regression_vsh", "increases with VSH"),
    "phit": ("y_phit", "mask_phit", "regression", "regression_phit", "increases with PHIT"),
    "swt": ("y_swt", "mask_swt", "regression", "regression_swt", "increases with SWT"),
}
DEFAULTS = {"resolutions_ms": None, "groups": ["stacks", "elastic", "fused"], "clip_quantiles": [0.005, 0.995],
            "indicator_min_missing": 0.01, "max_missing": 0.5, "near_constant_fraction": 0.95,
            "redundancy_abs_spearman": 0.90, "min_rows_per_class_per_well": 5, "min_rows_per_well_regression": 15,
            "min_wells": 4, "fdr_q": 0.10, "min_abs_effect": 0.10, "min_sign_consistency": 0.70,
            "max_selected": 25}
ALL_DEV = -1                                               # fold id of the "all development wells" fit


# =============================================================================== stage
def run(ctx: StageContext) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_style()
    steps = StepTracker(STEPS, ctx.stage)
    cfg = ctx.cfg
    sp = {**DEFAULTS, **(cfg.get("selection") or {})}
    ctx.parameters.update({k: v for k, v in sp.items()})
    reserved = [w.lower() for w in (cfg_get(cfg, "wells.reserved_reviewed", []) or [])]
    stacks = list((cfg.get("features") or {}).get("stacks") or ["full", "near", "mid", "far", "ultrafar"])

    # ---------------------------------------------------------------- 1 upstream
    steps.next("upstream")
    tt = ctx.run.run_dir / "05_training_tables"
    schema_p, split_p = tt / "feature_schema.json", tt / "tables" / "split_plan_wells.csv"
    for p in (schema_p, split_p):
        if not p.exists():
            raise StageBlocked(f"Missing {p}: run 05_training_tables first")
        ctx.add_input(p, f"upstream:05 {p.name}")
    schema = json.loads(schema_p.read_text(encoding="utf-8"))
    split = pd.read_csv(split_p)
    groups = {g: list(schema["predictor_groups"][g]) for g in sp["groups"] if g in schema["predictor_groups"]}
    if not groups:
        raise StageBlocked(f"None of the predictor groups {sp['groups']} is in feature_schema.json")
    predictors = list(dict.fromkeys(f for g in groups.values() for f in g))
    forbidden = [f for f in predictors if f.startswith(("y_", "mask_", "ysoft__", "meta__"))]
    ctx.check("no_target_or_meta_in_predictors", not forbidden, f"{len(predictors)} predictors; forbidden: {forbidden}")
    if forbidden:
        raise StageBlocked(f"feature_schema.json lists target/meta columns as predictors: {forbidden}")
    datasets = {int(k.replace("ms", "")): v for k, v in schema["datasets"].items()}
    want = sp["resolutions_ms"] or sorted(datasets)
    resolutions = [int(r) for r in want if int(r) in datasets]
    if not resolutions:
        raise StageBlocked(f"Requested resolutions {want} not produced by 05 ({sorted(datasets)})")
    base = min(resolutions)
    support = schema.get("task_support_from_02") or {}
    tasks = {}
    for t, (target, mask, kind, key, meaning) in TASKS.items():
        status = support.get(key, "SUPPORTED_PROVISIONAL")
        if status == "TASK_UNSUPPORTED":
            ctx.limit(f"task {t}: {key} is TASK_UNSUPPORTED in 02 - no statistics computed")
            continue
        tasks[t] = (target, mask, kind, meaning)

    dev = split.loc[split["role"] == "development", "well"].tolist()
    dev = [w for w in dev if w not in reserved]
    n_outer = int(split.loc[split["well"].isin(dev), "outer_fold"].max()) + 1
    outer = {f: sorted(split.loc[split["outer_fold"] == f, "well"]) for f in range(n_outer)}
    inner = {}
    for f in range(n_outer):
        col = f"inner_fold_when_outer_{f}"
        if col in split:
            g = split[split[col] >= 0].groupby(col)["well"].apply(sorted)
            inner[f] = [list(v) for v in g.tolist()]
        else:
            inner[f] = []
    tables: dict[int, pd.DataFrame] = {}
    for r in resolutions:
        p = tt / "tables" / datasets[r]
        if not p.exists():
            raise StageBlocked(f"Missing {p}")
        ctx.add_input(p, f"upstream:05 {r} ms table")
        d = pd.read_csv(p)
        n_res = int(d["well"].isin(reserved).sum())
        d = d[d["well"].isin(dev)].reset_index(drop=True)             # reserved rows dropped HERE, never used
        ctx.info("%d ms: %d development rows loaded; %d reserved-well rows dropped unread", r, len(d), n_res)
        missing_cols = [c for c in predictors if c not in d]
        if missing_cols:
            raise StageBlocked(f"{r} ms table lacks predictors {missing_cols[:5]} - rerun 05")
        tables[r] = d
    ctx.check("reserved_well_never_loaded", all(not t["well"].isin(reserved).any() for t in tables.values()),
              f"reserved {reserved}: 0 rows in every working table")
    ctx.extra["folds"] = {"outer_test_wells": {str(f): w for f, w in outer.items()},
                          "inner_groups": {str(f): g for f, g in inner.items()}, "development_wells": dev}

    # ---------------------------------------------------------------- 2 per-well effects
    steps.next("per-well effects")
    effects: dict[tuple[int, str], pd.DataFrame] = {}
    for r in resolutions:
        for t, (target, mask, kind, _) in progress(list(tasks.items()), desc=f"per-well effects {r} ms", unit="task"):
            effects[(r, t)] = per_well_effects(tables[r], predictors, target, mask, kind,
                                               int(sp["min_rows_per_class_per_well"]),
                                               int(sp["min_rows_per_well_regression"]))

    # ---------------------------------------------------------------- 3 folds (outer + all development)
    steps.next("folds")
    fit_sets = {f: [w for w in dev if w not in outer[f]] for f in range(n_outer)}
    fit_sets[ALL_DEV] = list(dev)
    results, transforms, used = {}, {}, {}
    for r in resolutions:
        transforms[r] = {}
        for f in progress(list(fit_sets), desc=f"folds {r} ms", unit="fold"):
            fit = fit_sets[f]
            test = outer.get(f, [])
            if set(fit) & set(test):
                raise RuntimeError(f"fold {f}: fit wells overlap test wells")       # cannot happen; guard anyway
            res = _procedure(tables[r], fit, groups, tasks, effects, r, sp, stacks)
            results[(r, f, -1)] = res
            transforms[r][str(f) if f != ALL_DEV else "all_development"] = _transform_doc(res, fit, test, r, f)
            used[(r, f)] = res["wells_used"]

    # ---------------------------------------------------------------- 4 inner replication
    steps.next("inner replication")
    rep_rows = []
    for r in resolutions:
        for f in progress(range(n_outer), desc=f"inner splits {r} ms", unit="fold"):
            if not inner.get(f):
                rep_rows.append({"resolution_ms": r, "outer_fold": f, "inner_split": -1, "group": "", "task": "",
                                 "validation_wells": "", "n_selected": 0, "n_feature_well_pairs": 0,
                                 "sign_replication_rate": np.nan, "median_heldout_to_training_effect": np.nan,
                                 "note": "fewer than 2 training wells in this outer fold: no inner splits"})
            for i, val in enumerate(inner.get(f, [])):
                fit = [w for w in fit_sets[f] if w not in val]
                res = _procedure(tables[r], fit, groups, tasks, effects, r, sp, stacks)
                results[(r, f, i)] = res
                transforms[r][f"{f}/inner{i}"] = _transform_doc(res, fit, val, r, f, inner=i)
                used[(r, f, i)] = res["wells_used"]
                rep_rows += _replication(res, val, effects, r, f, i, groups, tasks)
    replication = pd.DataFrame(rep_rows)

    # leakage audit: every statistic of fold f used only wells outside its outer test wells
    leaks = []
    for key, wells_used in used.items():
        f = key[1]
        bad = set(wells_used) & set(outer.get(f, [])) if f != ALL_DEV else set()
        bad |= set(wells_used) & set(reserved)
        if len(key) == 3:
            bad |= set(wells_used) & set(inner[f][key[2]])
        if bad:
            leaks.append((key, sorted(bad)))
    ctx.check("held_out_wells_never_used_for_fitting", not leaks,
              "transforms, screening, clusters and tests of every outer/inner fold used only its fit wells"
              if not leaks else f"violations: {leaks[:3]}")
    if leaks:
        raise RuntimeError(f"Held-out wells were used for fitting: {leaks[:3]}")

    # ---------------------------------------------------------------- 5 missingness informativeness
    steps.next("missingness")
    miss = _missingness(tables, resolutions, fit_sets, tasks, sp, transforms)

    # ---------------------------------------------------------------- 6 write
    steps.next("write")
    out = _write(ctx, results, transforms, effects, replication, miss, resolutions, base, n_outer, inner, groups,
                 tasks, fit_sets, outer, sp)

    # ---------------------------------------------------------------- 7 figures
    steps.next("figures")
    fig_group = "fused" if "fused" in groups else next(iter(groups))
    _figure_ranks(ctx, out["hyp"], base, fig_group, tasks, n_outer, sp)
    _figure_collinearity(ctx, results[(base, ALL_DEV, -1)], base, fig_group)
    _figure_distributions(ctx, tables[base], results[(base, ALL_DEV, -1)], base, fig_group, tasks)
    _figure_well_effects(ctx, effects, out["hyp"], base, fig_group, tasks, dev)
    _figure_stability(ctx, out["stability"], out["sel"], base, fig_group, tasks, n_outer)
    _figure_replication(ctx, replication, base, fig_group, tasks, n_outer)

    # ---------------------------------------------------------------- 8 summary
    steps.next("summary")
    _summarise(ctx, out, replication, miss, base, fig_group, tasks, groups, transforms, tables, fit_sets)
    steps.close()


# =============================================================================== the procedure
def _procedure(df, fit, groups, tasks, effects, res, sp, stacks) -> dict:
    """Transforms, screening, redundancy and tests using ONLY the rows of the ``fit`` wells."""
    train = df[df["well"].isin(fit)]
    predictors = list(dict.fromkeys(f for g in groups.values() for f in g))
    params = fit_transform_params(train, predictors, common_groups(predictors, stacks),
                                  tuple(sp["clip_quantiles"]), float(sp["indicator_min_missing"]))
    screen = screen_features(train, predictors, float(sp["max_missing"]), float(sp["near_constant_fraction"]))
    kept = screen.loc[screen["kept_screen"], "feature"].tolist()
    coverage = dict(zip(screen["feature"], 1 - screen["train_missing_fraction"]))
    corr = rank_correlation(train, kept) if kept else pd.DataFrame()
    clusters, stats, selected = {}, {}, {}
    task_stats = {t: aggregate_effects(effects[(res, t)], fit, kind, int(sp["min_wells"]))
                  for t, (_, _, kind, _) in tasks.items()}
    for t, (target, mask, kind, _) in tasks.items():
        st = task_stats[t]
        if len(st):
            d = train[train[mask].astype(bool)]
            yv = d[target].to_numpy(float)
            st["pooled_row_level_effect"] = [pooled_effect_xy(d[f].to_numpy(float), yv, kind) for f in st["feature"]]
    wells_used = set(train["well"])
    for t, st in task_stats.items():
        for w in (st["wells"] if len(st) else []):
            wells_used |= set(w.split(";"))
    for g, feats in groups.items():
        kg = [f for f in feats if f in kept]
        cl = redundancy_clusters(corr.loc[kg, kg], coverage, float(sp["redundancy_abs_spearman"])) if kg else \
            pd.DataFrame(columns=["feature", "cluster_id", "cluster_size", "representative", "represented_by", "order"])
        clusters[g] = cl
        cands = cl.loc[cl["representative"], "feature"].tolist() if len(cl) else []
        for t in tasks:
            st = task_stats[t]
            st = st[st["feature"].isin(feats)] if len(st) else st
            if st.empty:
                stats[(g, t)] = st
                selected[(g, t)] = []
                continue
            s = apply_selection_rule(st, cands, float(sp["fdr_q"]), float(sp["min_abs_effect"]),
                                     float(sp["min_sign_consistency"]), int(sp["max_selected"]))
            stats[(g, t)] = s
            selected[(g, t)] = s.loc[s["selected"]].sort_values("rank")["feature"].tolist()
    return {"params": params, "screen": screen, "corr": corr, "clusters": clusters, "stats": stats,
            "selected": selected, "fit": list(fit), "wells_used": sorted(wells_used), "n_rows": int(len(train))}


def _transform_doc(res, fit, held_out, r, f, inner: int | None = None) -> dict:
    return {"resolution_ms": r, "outer_fold": f, "inner_split": inner, "fit_wells": list(fit),
            "held_out_wells_never_used": list(held_out), "n_fit_rows": res["n_rows"], "params": res["params"]}


def _replication(res, val, effects, r, f, i, groups, tasks) -> list[dict]:
    """Did the predictors selected without wells ``val`` point the same way inside ``val``?"""
    rows = []
    for g in groups:
        for t in tasks:
            sel = res["selected"][(g, t)]
            st = res["stats"][(g, t)]
            e = effects[(r, t)]
            e = e[e["well"].isin(val) & e["eligible"] & e["feature"].isin(sel)]
            agree, ratio, n = [], [], 0
            sign = dict(zip(st["feature"], np.sign(st["re_mean_effect"]))) if len(st) else {}
            train_mag = dict(zip(st["feature"], st["re_mean_effect"].abs())) if len(st) else {}
            for _, x in e.iterrows():
                if not np.isfinite(x["effect"]) or x["effect"] == 0:
                    continue
                n += 1
                agree.append(np.sign(x["effect"]) == sign.get(x["feature"], 0))
                if train_mag.get(x["feature"], 0) > 0:
                    ratio.append(x["effect"] * sign[x["feature"]] / train_mag[x["feature"]])
            rows.append({"resolution_ms": r, "outer_fold": f, "inner_split": i, "group": g, "task": t,
                         "validation_wells": ";".join(val), "n_selected": len(sel), "n_feature_well_pairs": n,
                         "sign_replication_rate": float(np.mean(agree)) if agree else np.nan,
                         "median_heldout_to_training_effect": float(np.median(ratio)) if ratio else np.nan})
    return rows


def _missingness(tables, resolutions, fit_sets, tasks, sp, transforms) -> pd.DataFrame:
    """Per-well effect of each distinct missing-value pattern on each target (aggregated per fold)."""
    rows = []
    for r in resolutions:
        d = tables[r]
        allp = transforms[r]["all_development"]["params"]
        cols = [f for f, p in allp.items() if p["indicator"]]
        patterns, seen = {}, {}
        for c in cols:
            key = d[c].isna().to_numpy().tobytes()
            if key in seen:
                patterns[seen[key]].append(c)
            else:
                seen[key] = c
                patterns[c] = [c]
        if not patterns:
            continue
        ind = pd.DataFrame({f"{c}__isna": d[c].isna().astype(float) for c in patterns})
        dd = pd.concat([d[["well", "desired_twt_ms"] + [m for _, m, _, _ in tasks.values()] +
                          [y for y, _, _, _ in tasks.values()]], ind], axis=1)
        for t, (target, mask, kind, _) in tasks.items():
            eff = per_well_effects(dd, list(ind.columns), target, mask, kind, int(sp["min_rows_per_class_per_well"]),
                                   int(sp["min_rows_per_well_regression"]))
            for f, fit in fit_sets.items():
                st = aggregate_effects(eff, fit, kind, int(sp["min_wells"]))
                if st.empty:
                    continue
                st["q_value"] = benjamini_hochberg(st["p_value"].to_numpy())
                for _, x in st.iterrows():
                    base_col = x["feature"].replace("__isna", "")
                    rows.append({"resolution_ms": r, "outer_fold": f, "task": t, "indicator": x["feature"],
                                 "same_pattern_features": ";".join(patterns[base_col]),
                                 "k_wells": x["k_wells"], "re_mean_effect": x["re_mean_effect"],
                                 "re_ci_lo": x["re_ci_lo"], "re_ci_hi": x["re_ci_hi"], "q_value": x["q_value"],
                                 "sign_consistency": x["sign_consistency"],
                                 "informative": bool(np.isfinite(x["q_value"]) and x["q_value"] <= sp["fdr_q"]
                                                     and abs(x["re_mean_effect"]) >= 0.2)})
    return pd.DataFrame(rows)


# =============================================================================== write
def _fold_label(f):
    return "all_development" if f == ALL_DEV else f"outer_{f}"


def _write(ctx, results, transforms, effects, replication, miss, resolutions, base, n_outer, inner, groups, tasks,
           fit_sets, outer, sp) -> dict:
    (ctx.stage_dir / "transforms").mkdir(parents=True, exist_ok=True)
    for r in resolutions:
        doc = {"resolution_ms": r, "how_to_apply": "geogg_fresh.preprocess.apply_transform(df, params, features, "
               "mode='independent'|'common')", "steps": ["indicator <f>__isna where NaN (if params.indicator)",
                                                         "NaN -> impute_value", "clip to [clip_lo, clip_hi]",
                                                         "(x - centre) / scale  (or common_centre / common_scale)"],
               "folds": transforms[r]}
        ctx.save_json(doc, f"transforms/transforms_{r}ms.json", f"Frozen fold-local transforms ({r} ms): outer folds, "
                      "inner splits and all_development; fitted on fit_wells only", required=(r == base))
    tp = [params_table(v["params"], resolution_ms=r, fold_key=k, outer_fold=v["outer_fold"],
                       inner_split=-1 if v["inner_split"] is None else v["inner_split"])
          for r in resolutions for k, v in transforms[r].items()]
    ctx.save_table(pd.concat(tp, ignore_index=True), "transform_parameters.csv", "Every frozen transform parameter "
                   "(long form): fold, feature, impute value, clip limits, centre/scale (independent and common)",
                   required=True)
    hyp, scr, sel, stab = [], [], [], []
    for (r, f, i), res in results.items():
        if i != -1:
            continue
        s = res["screen"].copy()
        for g, cl in res["clusters"].items():
            m = s.merge(cl.drop(columns=["order"]), on="feature", how="inner") if len(cl) else s.iloc[0:0]
            if len(m) < len([x for x in groups[g]]):
                m = pd.concat([m, s[s["feature"].isin(groups[g]) & ~s["feature"].isin(m["feature"])]])
            m.insert(0, "group", g)
            m.insert(0, "fold_label", _fold_label(f))
            m.insert(0, "outer_fold", f)
            m.insert(0, "resolution_ms", r)
            scr.append(m)
        for (g, t), st in res["stats"].items():
            if st.empty:
                continue
            h = st.copy()
            h.insert(0, "effect_statistic", "rank_biserial" if tasks[t][2] == "binary" else "spearman")
            h.insert(0, "positive_means", tasks[t][3])
            for k, v in (("task", t), ("group", g), ("fold_label", _fold_label(f)), ("outer_fold", f),
                         ("resolution_ms", r)):
                h.insert(0, k, v)
            hyp.append(h)
    hyp = pd.concat(hyp, ignore_index=True) if hyp else pd.DataFrame()
    # ---- stability: inner selection frequency per outer fold; outer-fold frequency for all_development
    for r in resolutions:
        for g in groups:
            for t in tasks:
                for f in range(n_outer):
                    n_in = len(inner.get(f, []))
                    cnt = {}
                    for i in range(n_in):
                        for x in results[(r, f, i)]["selected"][(g, t)]:
                            cnt[x] = cnt.get(x, 0) + 1
                    for x in set(cnt) | set(results[(r, f, -1)]["selected"][(g, t)]):
                        stab.append({"resolution_ms": r, "outer_fold": f, "group": g, "task": t, "feature": x,
                                     "n_inner_splits": n_in, "n_inner_selected": cnt.get(x, 0),
                                     "inner_selection_frequency": cnt.get(x, 0) / n_in if n_in else np.nan,
                                     "selected_at_outer_level": x in results[(r, f, -1)]["selected"][(g, t)]})
                cnt = {}
                for f in range(n_outer):
                    for x in results[(r, f, -1)]["selected"][(g, t)]:
                        cnt[x] = cnt.get(x, 0) + 1
                for x in set(cnt) | set(results[(r, ALL_DEV, -1)]["selected"][(g, t)]):
                    stab.append({"resolution_ms": r, "outer_fold": ALL_DEV, "group": g, "task": t, "feature": x,
                                 "n_inner_splits": n_outer, "n_inner_selected": cnt.get(x, 0),
                                 "inner_selection_frequency": cnt.get(x, 0) / n_outer,
                                 "selected_at_outer_level": x in results[(r, ALL_DEV, -1)]["selected"][(g, t)]})
    stab = pd.DataFrame(stab)
    freq = {(int(a), int(b), c, d, e): v for a, b, c, d, e, v in
            stab[["resolution_ms", "outer_fold", "group", "task", "feature", "inner_selection_frequency"]]
            .itertuples(index=False)} if len(stab) else {}
    if len(hyp):
        hyp["selection_frequency"] = [freq.get((int(a), int(b), c, d, e), 0.0) for a, b, c, d, e in
                                      hyp[["resolution_ms", "outer_fold", "group", "task", "feature"]]
                                      .itertuples(index=False)]
        sel = hyp[hyp["selected"]].sort_values(["resolution_ms", "outer_fold", "group", "task", "rank"])
        sel = sel[["resolution_ms", "outer_fold", "fold_label", "group", "task", "rank", "feature", "re_mean_effect",
                   "re_ci_lo", "re_ci_hi", "q_value", "k_wells", "sign_consistency", "selection_frequency",
                   "positive_means"]].assign(outcome="selected")
    else:
        sel = pd.DataFrame(columns=["resolution_ms", "outer_fold", "fold_label", "group", "task", "rank", "feature",
                                    "outcome"])
    # an explicit row for every resolution / fold / group / task where NOTHING passed (never a silent gap)
    empty = []
    for (r, f, i), res in results.items():
        if i != -1:
            continue
        for g in groups:
            for t in tasks:
                if not res["selected"][(g, t)]:
                    st = res["stats"].get((g, t))
                    tested = int(st["p_value"].notna().sum()) if st is not None and len(st) else 0
                    empty.append({"resolution_ms": r, "outer_fold": f, "fold_label": _fold_label(f), "group": g,
                                  "task": t, "rank": np.nan, "feature": "",
                                  "outcome": "NONE_SELECTED" if tested else "NOT_TESTABLE (too few eligible wells)",
                                  "positive_means": tasks[t][3]})
    if empty:
        sel = pd.concat([sel, pd.DataFrame(empty)], ignore_index=True)
    sel = sel.sort_values(["resolution_ms", "outer_fold", "group", "task", "rank"], na_position="last")
    ctx.save_table(hyp, "hypothesis_test_results.csv", "Per resolution / fold / group / task / predictor: per-well "
                   "effect summary, random-effects mean + HKSJ 95% CI, p, BH q, sign consistency, row-level pooled "
                   "effect (descriptive), decision", required=True)
    ctx.save_table(sel, "selected_features.csv", "Selected predictors (ranked) per resolution / fold / group / task; "
                   "outer_fold -1 = all development wells (final model); outcome NONE_SELECTED / NOT_TESTABLE rows "
                   "where nothing passed", required=True)
    ctx.save_table(pd.concat(scr, ignore_index=True) if scr else pd.DataFrame(), "screening_and_clusters.csv",
                   "Label-free screening (missing, near-constant, between-well share, TWT trend) and redundancy "
                   "clusters with their representative", required=True)
    pwe = pd.concat([e.assign(resolution_ms=r, task=t) for (r, t), e in effects.items()], ignore_index=True)
    ctx.save_table(pwe[["resolution_ms", "task", "well", "feature", "n", "n_class1", "n_class0", "statistic",
                        "effect", "var", "eligible"]], "per_well_effects.csv", "Within-well effect of every "
                   "predictor on every task (development wells only; the reserved well is never computed)",
                   required=True)
    ctx.save_table(stab, "selection_stability.csv", "How often each predictor is selected: across the inner splits "
                   "of an outer fold, and across outer folds for all_development")
    ctx.save_table(replication if len(replication) else pd.DataFrame(columns=["resolution_ms"]),
                   "inner_replication.csv", "Direction check of selected predictors in inner-validation wells "
                   "(never seen by that selection)", required=True)
    ctx.save_table(miss if len(miss) else pd.DataFrame(columns=["resolution_ms", "indicator"]),
                   "missingness_informativeness.csv", "Does a missing-value pattern predict a target? (per-well "
                   "random-effects, as for predictors)")
    # ---- feature sets for 07
    sets = {"rules": {k: sp[k] for k in ("fdr_q", "min_abs_effect", "min_sign_consistency", "min_wells",
                                         "max_selected", "redundancy_abs_spearman", "max_missing")},
            "note": "screened = label-free cluster representatives (task-independent); selected = passed the "
                    "documented well-level test rule for that task. Empty 'selected' means no predictor passed.",
            "resolutions": {}}
    for r in resolutions:
        R = {}
        for (rr, f, i), res in results.items():
            if rr != r:
                continue
            key = _fold_label(f) if i == -1 else f"outer_{f}/inner_{i}"
            R[key] = {"fit_wells": res["fit"], "held_out_wells": (outer.get(f, []) if i == -1 else inner[f][i]),
                      "transform_key": (str(f) if f != ALL_DEV else "all_development") if i == -1 else f"{f}/inner{i}",
                      "groups": {g: {"screened": res["clusters"][g].loc[res["clusters"][g]["representative"],
                                                                         "feature"].tolist()
                                     if len(res["clusters"][g]) else [],
                                     "selected": {t: res["selected"][(g, t)] for t in tasks}} for g in groups}}
        sets["resolutions"][f"{r}ms"] = {"transforms_file": f"transforms/transforms_{r}ms.json", "folds": R}
    ctx.save_json(sets, "feature_sets.json", "Per resolution / fold (outer, inner, all_development) / group / task: "
                  "screened and selected predictor lists for 07", required=True)
    manifest = {"fit_wells": {_fold_label(f): w for f, w in fit_sets.items()},
                "outer_test_wells_never_used": {_fold_label(f): w for f, w in outer.items()},
                "inner_groups": {str(f): g for f, g in inner.items()},
                "tasks": {t: {"target": v[0], "mask": v[1], "kind": v[2], "positive_effect_means": v[3]}
                          for t, v in tasks.items()},
                "statistics": {"within_well": "rank-biserial r = 2*AUC-1 (binary), Spearman rho (regression)",
                               "across_wells": "random-effects (DerSimonian-Laird tau^2) mean with Hartung-Knapp-"
                                               "Sidik-Jonkman t-interval, k-1 df, Knapp-Hartung max rule; regression "
                                               "on Fisher z, reported on the rho scale",
                               "multiplicity": "Benjamini-Hochberg over the candidate predictors of one "
                                               "resolution / fold / group / task",
                               "row_level_pooled_effect": "descriptive only (rows are autocorrelated)"},
                "parameters": sp, "code_version": ctx.run.code_version}
    ctx.save_json(manifest, "selection_manifest.json", "Fit wells per fold, held-out wells, statistics and rules",
                  required=True)
    return {"hyp": hyp, "sel": sel, "stability": stab, "screen": pd.concat(scr, ignore_index=True) if scr else None}


# =============================================================================== figures
FOLD_COLORS = [SERIES[0], SERIES[1], SERIES[2], SERIES[6], SERIES[4], SERIES[5]]
TASK_TITLES = {"lithology": "sand vs shale", "sand_fluid": "water vs HC-potential sand", "vsh": "VSH",
               "phit": "PHIT", "swt": "SWT"}


def _figure_ranks(ctx, hyp, base, group, tasks, n_outer, sp):
    import matplotlib.pyplot as plt

    tl = list(tasks)
    ncol = 3
    nrow = int(np.ceil(len(tl) / ncol)) or 1
    fig, axs = plt.subplots(nrow, ncol, figsize=(21, 7.2 * nrow), squeeze=False)
    h = hyp[(hyp["resolution_ms"] == base) & (hyp["group"] == group)] if len(hyp) else hyp
    for ax, t in zip(axs.ravel(), tl):
        d = h[h["task"] == t] if len(h) else h
        a = d[(d["outer_fold"] == ALL_DEV) & d["candidate"]] if len(d) else d
        if a is None or a.empty or a["re_mean_effect"].notna().sum() == 0:
            ax.text(0.5, 0.5, f"{t}: no statistics\n(too few eligible training wells)", ha="center", va="center",
                    transform=ax.transAxes, color=STATUS["critical"])
            ax.set_axis_off()
            continue
        a = a.dropna(subset=["re_mean_effect"]).assign(_m=lambda x: x["re_mean_effect"].abs()).nlargest(15, "_m")
        feats = a["feature"].tolist()[::-1]
        y = np.arange(len(feats))
        A = a.set_index("feature")
        for j, f in enumerate(feats):
            r = A.loc[f]
            ax.plot([r["re_ci_lo"], r["re_ci_hi"]], [j, j], color=TEXT_2, lw=1.4, zorder=2)
            ax.scatter(r["re_mean_effect"], j, s=70, zorder=4, color="#0b0b0b" if r["selected"] else "white",
                       edgecolor="#0b0b0b", linewidth=1.4)
        for f_ in range(n_outer):
            o = d[d["outer_fold"] == f_].set_index("feature")
            xs = [o.loc[f, "re_mean_effect"] if f in o.index else np.nan for f in feats]
            ax.scatter(xs, y + 0.18 * (f_ - (n_outer - 1) / 2) / max(1, n_outer - 1) * 2, s=16, color=FOLD_COLORS[f_ % 6],
                       zorder=3, label=f"outer fold {f_} (its 9 training wells)")
        ax.axvline(0, color=NEUTRAL, lw=0.9)
        for s in (-1, 1):
            ax.axvline(s * float(sp["min_abs_effect"]), color=NEUTRAL, lw=0.7, ls=":")
        ax.set_yticks(y)
        ax.set_yticklabels(feats, fontsize=7.5)
        ax.set_xlim(-1, 1)
        stat = "rank-biserial r" if tasks[t][2] == "binary" else "Spearman rho"
        ax.set_xlabel(f"random-effects mean of per-well {stat} (+ = {tasks[t][3]})", fontsize=8.5)
        n_sel = int(a["selected"].sum())
        ax.set_title(f"{TASK_TITLES.get(t, t)}: top {len(feats)} candidates, {int(d[(d['outer_fold'] == ALL_DEV)]['selected'].sum())} selected (all development)",
                     fontsize=10)
        if t == tl[0]:
            ax.legend(fontsize=7, loc="lower right")
        _ = n_sel
    for ax in axs.ravel()[len(tl):]:
        ax.set_axis_off()
    fig.suptitle(f"06_feature_selection: predictor ranks by well-level effect ({base} ms, {group} predictors)",
                 fontsize=13, fontweight="bold")
    footer(fig, "Black/white dot + bar: all 12 development wells, random-effects mean with HKSJ 95% CI (filled = "
                "selected: BH q <= {q}, |effect| >= {e}, sign consistency >= {c}). Coloured dots: the same statistic "
                "inside each outer fold's training wells. Effects are within-well, so a predictor that only "
                "separates wells from each other scores ~0.".format(q=sp["fdr_q"], e=sp["min_abs_effect"],
                                                                   c=sp["min_sign_consistency"]))
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    ctx.save_figure(fig, "feature_selection_ranks.png", "Forest plot of well-level effects per task", required=True,
                    multipanel=True)


def _figure_collinearity(ctx, res, base, group):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cl = res["clusters"].get(group)
    fig = plt.figure(figsize=(22, 13))
    ax = fig.add_axes([0.06, 0.12, 0.56, 0.8])
    side = fig.add_axes([0.66, 0.12, 0.32, 0.8])
    side.set_axis_off()
    if cl is None or cl.empty:
        ax.text(0.5, 0.5, "no predictors survived screening", ha="center", transform=ax.transAxes)
    else:
        order = cl.sort_values("order")["feature"].tolist()
        c = res["corr"].loc[order, order].abs()
        im = ax.imshow(c.to_numpy(), cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(range(len(order)))
        ax.set_yticks(range(len(order)))
        fs = 5.5 if len(order) > 50 else 7.5
        ax.set_xticklabels(order, rotation=90, fontsize=fs)
        ax.set_yticklabels([f"{f} *" if r else f for f, r in zip(order, cl.sort_values("order")["representative"])],
                           fontsize=fs)
        ax.grid(False)
        pos = 0
        for cid, g in cl.sort_values("order").groupby("cluster_id", sort=False):
            n = len(g)
            if n > 1:
                ax.add_patch(Rectangle((pos - 0.5, pos - 0.5), n, n, fill=False, edgecolor=SERIES[1], lw=1.4))
            pos += n
        fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01, label="|Spearman rho| (training rows of all development wells)")
        big = cl[cl["cluster_size"] > 1].groupby("cluster_id").agg(rep=("represented_by", "first"),
                                                                   n=("feature", "size"),
                                                                   members=("feature", lambda s: ", ".join(s)))
        big = big.sort_values("n", ascending=False)
        lines = [f"{len(cl)} predictors -> {cl['cluster_id'].nunique()} clusters "
                 f"(complete linkage, every pair |rho| >= threshold)", ""]
        for _, b in big.head(14).iterrows():
            lines.append(f"[{b['n']}] keep {b['rep']}")
            mem = b["members"].replace(b["rep"] + ", ", "").replace(", " + b["rep"], "")
            lines.append("     drop " + (mem[:95] + "..." if len(mem) > 95 else mem))
        side.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=7.6, family="monospace")
    fig.suptitle(f"06_feature_selection: collinearity among {group} predictors ({base} ms, all development wells)",
                 fontsize=13, fontweight="bold")
    footer(fig, "Orange boxes = redundancy clusters; * = representative kept for testing (chosen WITHOUT labels: "
                "coverage, then centrality). Correlation computed on training rows only.")
    ctx.save_figure(fig, "collinearity_clusters.png", "Rank-correlation matrix ordered by redundancy cluster",
                    required=True)


def _figure_distributions(ctx, df, res, base, group, tasks):
    import matplotlib.pyplot as plt

    t = "lithology" if "lithology" in tasks else next(iter(tasks))
    target, mask, kind, meaning = tasks[t]
    st = res["stats"].get((group, t), pd.DataFrame())
    fig, axs = plt.subplots(2, 3, figsize=(20, 10.5))
    if st is None or st.empty:
        for ax in axs.ravel():
            ax.set_axis_off()
        axs[0, 1].text(0.5, 0.5, "no statistics", ha="center", transform=axs[0, 1].transAxes)
    else:
        s = st[st["candidate"]].dropna(subset=["re_mean_effect"])
        s = s.assign(_o=lambda d: (~d["selected"]).astype(int), _m=lambda d: -d["re_mean_effect"].abs())
        s = s.sort_values(["_o", "_m"]).head(6)
        X = apply_transform(df, res["params"], s["feature"].tolist(), add_indicators=False)
        m = df[mask].astype(bool)
        for ax, (_, r) in zip(axs.ravel(), s.iterrows()):
            f = r["feature"]
            if kind == "binary":
                for val, name, col in ((1, "sand" if t == "lithology" else "HC-potential",
                                        CLASS_COLORS["undetermined_sand"]),
                                       (0, "shale" if t == "lithology" else "water-bearing", CLASS_COLORS["shale"])):
                    v = X.loc[m & (df[target] == val) & df[f].notna(), f]
                    if len(v):
                        ax.hist(v, bins=40, density=True, histtype="step", lw=2, color=col, label=f"{name} (n={len(v)})")
            else:
                ok = m & df[f].notna()
                ax.scatter(X.loc[ok, f], df.loc[ok, target], s=4, alpha=0.4, color=SERIES[0])
                ax.set_ylabel(target)
            ax.set_xlabel(f"{f} (fold transform: (x - median) / IQR)", fontsize=8.5)
            tag = "SELECTED" if r["selected"] else "not selected"
            ax.set_title(f"{f}\n{tag}: r = {r['re_mean_effect']:+.2f} [{r['re_ci_lo']:+.2f}, {r['re_ci_hi']:+.2f}], "
                         f"{int(r['k_wells'])} wells, q = {r['q_value']:.2g}", fontsize=9)
            ax.legend(fontsize=8)
        for ax in axs.ravel()[len(s):]:
            ax.set_axis_off()
    fig.suptitle(f"06_feature_selection: distributions of the top predictors for {TASK_TITLES.get(t, t)} "
                 f"({base} ms, development wells, all-development transform)", fontsize=13, fontweight="bold")
    footer(fig, "Values after the frozen all-development transform (imputed values excluded). Pooled histograms mix "
                "wells; the selection itself used within-well effects (see per_well_effects_heatmap.png).")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    ctx.save_figure(fig, "selected_feature_distributions.png", "Distributions of the top predictors", required=True,
                    multipanel=True)


def _figure_well_effects(ctx, effects, hyp, base, group, tasks, dev):
    import matplotlib.pyplot as plt

    show = [t for t in ("lithology", "vsh", "phit", "sand_fluid") if t in tasks][:2] or list(tasks)[:2]
    fig, axs = plt.subplots(1, len(show), figsize=(11 * len(show), 11), squeeze=False)
    fig.subplots_adjust(left=0.05, right=0.91, bottom=0.24, top=0.9, wspace=0.12)
    im = None
    for ax, t in zip(axs[0], show):
        h = hyp[(hyp["resolution_ms"] == base) & (hyp["group"] == group) & (hyp["task"] == t)
                & (hyp["outer_fold"] == ALL_DEV) & hyp["candidate"]] if len(hyp) else pd.DataFrame()
        if h.empty:
            ax.set_axis_off()
            continue
        h = h.dropna(subset=["re_mean_effect"]).assign(_m=lambda d: d["re_mean_effect"].abs()).nlargest(20, "_m")
        feats = h["feature"].tolist()
        e = effects[(base, t)]
        e = e[e["feature"].isin(feats) & e["eligible"]]
        M = e.pivot(index="well", columns="feature", values="effect").reindex(index=dev, columns=feats)
        im = ax.imshow(np.ma.masked_invalid(M.to_numpy()), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
                       interpolation="nearest")
        ax.set_facecolor("#e4e3df")
        ax.set_xticks(range(len(feats)))
        sel = dict(zip(h["feature"], h["selected"]))
        ax.set_xticklabels([f"{f} *" if sel.get(f) else f for f in feats], rotation=90, fontsize=7.5)
        ax.set_yticks(range(len(dev)))
        ax.set_yticklabels(dev, fontsize=9)
        ax.grid(False)
        for i_ in range(M.shape[0]):
            for j_ in range(M.shape[1]):
                v = M.iat[i_, j_]
                if np.isfinite(v):
                    ax.text(j_, i_, f"{v:+.2f}", ha="center", va="center", fontsize=5.8,
                            color="white" if abs(v) > 0.55 else "#0b0b0b")
        stat = "rank-biserial r" if tasks[t][2] == "binary" else "Spearman rho"
        ax.set_title(f"{TASK_TITLES.get(t, t)}: per-well {stat} (+ = {tasks[t][3]})", fontsize=10)
    if im is not None:
        fig.colorbar(im, cax=fig.add_axes([0.93, 0.3, 0.012, 0.55]), label="within-well effect")
    fig.suptitle(f"06_feature_selection: does each predictor behave the same in every well? ({base} ms, {group})",
                 fontsize=13, fontweight="bold")
    footer(fig, "One row per development well (reserved well never computed). Grey = well not eligible (too few rows "
                "of a class or no variation). * = selected for the all-development model. Top 20 candidates by "
                "|random-effects mean|.")
    ctx.save_figure(fig, "per_well_effects_heatmap.png", "Within-well effects per development well", required=True,
                    multipanel=True)


def _figure_stability(ctx, stab, sel, base, group, tasks, n_outer):
    import matplotlib.pyplot as plt

    show = [t for t in ("lithology", "vsh", "phit") if t in tasks] or list(tasks)[:3]
    fig, axs = plt.subplots(1, len(show), figsize=(7.5 * len(show), 11), squeeze=False)
    cols = [f"F{f}" for f in range(n_outer)] + ["all dev"]
    for ax, t in zip(axs[0], show):
        s = stab[(stab["resolution_ms"] == base) & (stab["group"] == group) & (stab["task"] == t)] if len(stab) else stab
        if s is None or s.empty:
            ax.text(0.5, 0.5, f"{t}: nothing selected in any fold", ha="center", transform=ax.transAxes,
                    color=STATUS["critical"])
            ax.set_axis_off()
            continue
        feats = (s.groupby("feature")["inner_selection_frequency"].mean().sort_values(ascending=False).index.tolist())[:30]
        M = np.full((len(feats), len(cols)), np.nan)
        dots = []
        for j, f in enumerate(list(range(n_outer)) + [ALL_DEV]):
            d = s[s["outer_fold"] == f].set_index("feature")
            for i, x in enumerate(feats):
                if x in d.index:
                    M[i, j] = d.loc[x, "inner_selection_frequency"]
                    if d.loc[x, "selected_at_outer_level"]:
                        dots.append((j, i))
                else:
                    M[i, j] = 0.0
        im = ax.imshow(M, cmap="Greens", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j + 0.18, i, f"{M[i, j]:.2f}", va="center", fontsize=7,
                            color="white" if M[i, j] > 0.6 else "#0b0b0b")
        for j, i in dots:
            ax.plot(j - 0.18, i, "o", color="#0b0b0b" if M[i, j] <= 0.6 else "white", ms=4)
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats, fontsize=7.5)
        ax.grid(False)
        ax.set_title(f"{TASK_TITLES.get(t, t)}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="selection frequency")
    fig.suptitle(f"06_feature_selection: selection stability ({base} ms, {group})", fontsize=13, fontweight="bold")
    footer(fig, "Columns F0-F3: share of the inner splits of that outer fold that selected the predictor (dot = also "
                "selected with all 9 training wells of the fold). 'all dev': share of the 4 outer folds that selected "
                "it (dot = selected for the final model).")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    ctx.save_figure(fig, "selection_stability.png", "Selection frequency across inner splits and folds",
                    multipanel=True)


def _figure_replication(ctx, rep, base, group, tasks, n_outer):
    import matplotlib.pyplot as plt

    fig, (a, b) = plt.subplots(1, 2, figsize=(19, 7.5))
    d = rep[(rep["resolution_ms"] == base) & (rep["group"] == group)] if len(rep) else rep
    tl = list(tasks)
    width = 0.8 / max(1, n_outer)
    for f in range(n_outer):
        x = np.arange(len(tl)) + (f - (n_outer - 1) / 2) * width
        vals, rat, ns = [], [], []
        for t in tl:
            g = d[(d["outer_fold"] == f) & (d["task"] == t)] if len(d) else d
            n = g["n_feature_well_pairs"].sum() if len(g) else 0
            w = g["n_feature_well_pairs"].to_numpy(float) if len(g) else np.zeros(0)
            v = g["sign_replication_rate"].to_numpy(float) if len(g) else np.zeros(0)
            ok = np.isfinite(v) & (w > 0)
            vals.append(float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan)
            rr = g["median_heldout_to_training_effect"].to_numpy(float) if len(g) else np.zeros(0)
            rat.append(float(np.nanmedian(rr)) if np.isfinite(rr).any() else np.nan)
            ns.append(int(n))
        a.bar(x, vals, width, color=FOLD_COLORS[f % 6], label=f"outer fold {f}")
        b.bar(x, rat, width, color=FOLD_COLORS[f % 6], label=f"outer fold {f}")
        for xi, v, n in zip(x, vals, ns):
            if np.isfinite(v):
                a.text(xi, v + 0.01, str(n), ha="center", fontsize=6.5, color=TEXT_2)
    a.axhline(0.5, color=STATUS["critical"], lw=1, ls="--", label="chance (random direction)")
    a.set_ylim(0, 1.05)
    a.set_xticks(range(len(tl)))
    a.set_xticklabels([TASK_TITLES.get(t, t) for t in tl])
    a.set_ylabel("share of (selected predictor, held-out well) pairs\nwith the same direction")
    a.set_title("Direction replicates in inner-validation wells? (numbers = pairs)", fontsize=10)
    a.legend(fontsize=8)
    b.axhline(1.0, color=NEUTRAL, lw=1, ls="--")
    b.axhline(0.0, color=NEUTRAL, lw=0.8)
    b.set_xticks(range(len(tl)))
    b.set_xticklabels([TASK_TITLES.get(t, t) for t in tl])
    b.set_ylabel("median held-out effect / training effect")
    b.set_title("Effect shrinkage in unseen wells (1 = no shrinkage)", fontsize=10)
    fig.suptitle(f"06_feature_selection: inner-fold replication of the selection ({base} ms, {group})",
                 fontsize=13, fontweight="bold")
    footer(fig, "For every outer fold, the selection is repeated on 2 of its 3 inner well groups and checked in the "
                "wells of the 3rd group. Outer test wells are not used. Empty bars: nothing was selected.")
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    ctx.save_figure(fig, "inner_replication.png", "Inner-fold replication of selected predictors", required=True,
                    multipanel=True)


# =============================================================================== summary
def _summarise(ctx, out, rep, miss, base, group, tasks, groups, transforms, tables, fit_sets):
    hyp, sel = out["hyp"], out["sel"]
    # transform sanity: applied to its own fit rows, the scaled median is 0 (independent mode)
    p = transforms[base]["all_development"]["params"]
    train = tables[base][tables[base]["well"].isin(fit_sets[ALL_DEV])]
    X = apply_transform(train, p, add_indicators=False)
    off = {f: float(np.median(X[f])) for f in X.columns if p[f]["n_finite"] >= 10 and abs(np.median(X[f])) > 1e-6}
    ctx.check("transform_round_trip_median_zero", not off, "scaled training median = 0 for every predictor"
              if not off else f"non-zero medians: {list(off.items())[:5]}")
    bad = [f for r in transforms for k, v in transforms[r].items() for f, q in v["params"].items()
           if not (np.isfinite(q["centre"]) and np.isfinite(q["scale"]) and q["scale"] > 0)]
    ctx.check("transforms_complete_and_finite", not bad, f"{sum(len(v['params']) for r in transforms for v in transforms[r].values())} "
              "parameter sets, all finite" if not bad else f"bad: {bad[:5]}")
    ctx.check("every_supported_task_tested", len(hyp) > 0 and set(tasks) <= set(hyp["task"]),
              f"tasks tested: {sorted(set(hyp['task'])) if len(hyp) else []}")
    final = sel[(sel["resolution_ms"] == base) & (sel["outer_fold"] == ALL_DEV) & (sel["outcome"] == "selected")] \
        if len(sel) else sel
    n_sel = {g: {t: int(((final["group"] == g) & (final["task"] == t)).sum()) for t in tasks} for g in groups} \
        if len(final) else {g: {t: 0 for t in tasks} for g in groups}
    ctx.metric("n_selected_all_development", n_sel)
    none = [f"{g}/{t}" for g in groups for t in tasks if n_sel[g][t] == 0]
    if none:
        ctx.limit(f"{base} ms, all development wells: no predictor passed the well-level rule for {none} - for "
                  "those, 07 must use the label-free 'screened' set and say so")
    top = {}
    for t in tasks:
        d = final[(final["group"] == group) & (final["task"] == t)] if len(final) else final
        top[t] = [f"{r.feature} ({r.re_mean_effect:+.2f})" for r in d.head(5).itertuples()] if len(d) else []
    ctx.metric("top_selected_all_development", top)
    if len(rep):
        r = rep[(rep["resolution_ms"] == base) & (rep["group"] == group) & (rep["n_feature_well_pairs"] > 0)]
        repl = {}
        for t in tasks:
            g = r[r["task"] == t]
            w = g["n_feature_well_pairs"].to_numpy(float)
            v = g["sign_replication_rate"].to_numpy(float)
            ok = np.isfinite(v)
            repl[t] = round(float((v[ok] * w[ok]).sum() / w[ok].sum()), 3) if ok.any() and w[ok].sum() else None
        ctx.metric("inner_sign_replication_rate", repl)
        weak = [t for t, v in repl.items() if v is not None and v < 0.7]
        if weak:
            ctx.warn(f"Selected predictors replicate their direction in < 70% of inner-validation wells for {weak}: "
                     "treat those selections as unstable")
    if len(miss):
        inf = miss[(miss["outer_fold"] == ALL_DEV) & miss["informative"]]
        ctx.metric("informative_missingness_all_development",
                   [f"{r.resolution_ms} ms {r.indicator} -> {r.task} ({r.re_mean_effect:+.2f})" for r in inf.itertuples()])
        if len(inf):
            ctx.warn("Missing-value indicators predict a target (q <= fdr_q, |effect| >= 0.2): " +
                     "; ".join(f"{r.indicator} -> {r.task} {r.re_mean_effect:+.2f}" for r in inf[inf["resolution_ms"] == base].itertuples()) +
                     " - the NaN pattern (e.g. inversion coverage) carries information; models with indicators may "
                     "learn coverage rather than rock properties: compare with and without indicators in 07")
    if out["screen"] is not None and len(out["screen"]):
        s = out["screen"]
        s = s[(s["resolution_ms"] == base) & (s["outer_fold"] == ALL_DEV) & (s["group"] == group)]
        ctx.metric("clusters_all_development", {"n_predictors": int(len(s)), "n_kept_after_screening":
                                                int(s["kept_screen"].sum()),
                                                "n_clusters": int(s["cluster_id"].nunique())})
        hi = s[s["between_well_rank_variance_share"] > 0.5]["feature"].tolist()
        if hi:
            ctx.info("predictors whose variance is mostly BETWEEN wells (rank eta^2 > 0.5): %s", hi[:12])
    ctx.limit("Targets are provisional log interpretations tied with a provisional T-Z; statistics describe "
              "associations in the training wells, not validated predictive skill (that is 07/08)")
    ctx.limit("Within-well rows are autocorrelated: p-values come from the across-well HKSJ test (k-1 df); "
              "row-level pooled effects are descriptive only")
    ctx.readme("How to use the outputs", f"""
* `transforms/transforms_<res>ms.json` - frozen parameters per fold key (`0`..`{len(fit_sets) - 2}`,
  `<f>/inner<i>`, `all_development`). Apply with
  `geogg_fresh.preprocess.apply_transform(df, params, features, mode="independent" | "common")`.
* `feature_sets.json` - per resolution / fold / group / task: `screened` (label-free representatives) and
  `selected` (passed: BH q <= fdr_q, |effect| >= min_abs_effect, sign consistency >= min_sign_consistency).
* `tables/hypothesis_test_results.csv` - every test, with the decision text for every predictor.
* Outer test wells of a fold were never read for that fold; the reserved well was never loaded.
""")
