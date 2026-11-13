"""07_models - candidate models trained and tuned with nested whole-well cross-validation.

What this stage answers
-----------------------
"Which learner, on which inputs, predicts each target best in wells it was not tuned on -
and is it better than burial depth alone?"  Every choice (hyperparameters, probability
threshold, number of epochs) is made on INNER validation wells.  The outer test wells of a
fold are used exactly once: to store the sealed predictions that stage 08 evaluates.

Steps (progress bars in the terminal)
-------------------------------------
1. Upstream     - 05 table + split plan, 06 frozen transforms + feature sets, 03 patches (deep models).
2. Data         - development rows only (the reserved well is dropped unread); per task the rows
                  with a valid target; inputs:
                  * ``stacks``          stack predictors, all rows                (primary benchmark)
                  * ``stacks@matched``  stack predictors, rows with every inversion input
                  * ``fused@matched``   stacks + inversion predictors, the same rows
                  so Stacks-vs-Fused is compared on identical rows (no coverage effect).
3. Tabular      - prior, depth_trend (TWT only), linear, ExtraTrees, HistGBM, XGBoost, MLP.
                  For each outer fold: grid search on the 3 inner splits (06 inner transforms
                  and feature sets), threshold for sand/HC-potential from the inner predictions,
                  refit on the 9 training wells, predict the 3 test wells (sealed).
4. Deep         - TCN and 3-D CNN (multi-task lithology + VSH + PHIT) on the 03 patches; epochs
                  from early stopping on the inner wells; refit; predict the test wells (sealed).
5. Final models - every candidate refitted on all 12 development wells (hyperparameters = the
                  most frequent choice of the outer folds; threshold = median) and saved.
6. Write, 7. Figures, 8. Summary (champions declared from INNER scores only).

Nothing here computes a metric on an outer test well; that is stage 08.
"""

from __future__ import annotations

import json
import time
import traceback
import warnings
from collections import Counter

import numpy as np
import pandas as pd

from .. import models_deep as mdeep
from .. import models_tabular as mtab
from ..stage_compat import StageBlocked, SpecialistStageContext as StageContext
from ..config import cfg_get
from ..metrics import PRIMARY, apply_bounds, best_threshold, is_better, task_metrics
from ..plotting import NEUTRAL, SERIES, STATUS, TEXT_2, footer, not_available_figure, setup_style
from ..preprocess import apply_transform
from ..progress import StepTracker, progress
from .s06_feature_selection import TASKS

STEPS = ["upstream", "data", "tabular models", "deep models", "write", "figures", "summary"]
DEFAULTS = {"resolution_ms": None, "tasks": ["lithology", "sand_fluid", "vsh", "phit", "swt"],
            "inputs": ["stacks", "stacks@matched", "fused@matched"],
            "tabular": ["prior", "depth_trend", "linear", "extratrees", "hist_gbm", "xgboost", "mlp"],
            "deep": ["tcn", "cnn3d_multitask"], "feature_choice": "selected", "scaling_modes": ["independent"],
            "class_weight": "balanced", "n_jobs": -1, "seed": 2026, "save_models": "final", "grids": {},
            "deep_epochs": 60, "deep_batch_size": 128, "deep_patience": 10, "deep_device": "auto",
            "deep_scaling": "common", "deep_dropout": 0.2, "deep_max_nan_fraction": 0.2}
ALL_DEV = -1
DEEP_INPUT = "patch_stacks"


# =============================================================================== stage
def run(ctx: StageContext) -> None:
    import matplotlib

    matplotlib.use("Agg")
    setup_style()
    steps = StepTracker(STEPS, ctx.stage)
    cfg = ctx.cfg
    sp = {**DEFAULTS, **(cfg.get("models") or {})}
    ctx.parameters.update(sp)
    reserved = [w.lower() for w in (cfg_get(cfg, "wells.reserved_reviewed", []) or [])]
    run_dir = ctx.run.run_dir

    # ---------------------------------------------------------------- 1 upstream
    steps.next("upstream")
    tt, fs = run_dir / "05_training_tables", run_dir / "06_feature_selection"
    schema_p, split_p, sets_p = tt / "feature_schema.json", tt / "tables" / "split_plan_wells.csv", fs / "feature_sets.json"
    for p in (schema_p, split_p, sets_p):
        if not p.exists():
            raise StageBlocked(f"Missing {p}: run 05_training_tables and 06_feature_selection first")
        ctx.add_input(p, f"upstream:{p.parent.name}/{p.name}")
    schema = json.loads(schema_p.read_text(encoding="utf-8"))
    sets = json.loads(sets_p.read_text(encoding="utf-8"))
    split = pd.read_csv(split_p)
    datasets = {int(k.replace("ms", "")): v for k, v in schema["datasets"].items()}
    res = int(sp["resolution_ms"] or min(datasets))
    if f"{res}ms" not in sets["resolutions"]:
        raise StageBlocked(f"06 has no feature sets for {res} ms (has {list(sets['resolutions'])})")
    rs = sets["resolutions"][f"{res}ms"]
    tr_p = fs / rs["transforms_file"]
    tab_p = tt / "tables" / datasets[res]
    for p in (tr_p, tab_p):
        if not p.exists():
            raise StageBlocked(f"Missing {p}")
        ctx.add_input(p, f"upstream:{p.name}")
    transforms = json.loads(tr_p.read_text(encoding="utf-8"))["folds"]
    elastic_cols = list(schema["predictor_groups"].get("elastic", []))

    # ---------------------------------------------------------------- 2 data
    steps.next("data")
    df = pd.read_csv(tab_p)
    n_res = int(df["well"].isin(reserved).sum())
    dev = [w for w in split.loc[split["role"] == "development", "well"] if w not in reserved]
    df = df[df["well"].isin(dev)].reset_index(drop=True)
    ctx.info("%d ms table: %d development rows; %d reserved-well rows dropped unread", res, len(df), n_res)
    ctx.check("reserved_well_never_loaded", not df["well"].isin(reserved).any(), f"reserved {reserved}: 0 rows used")
    n_outer = int(split.loc[split["well"].isin(dev), "outer_fold"].max()) + 1
    outer = {f: sorted(split.loc[split["outer_fold"] == f, "well"]) for f in range(n_outer)}
    inner = {}
    for f in range(n_outer):
        col = f"inner_fold_when_outer_{f}"
        inner[f] = ([sorted(g) for _, g in split[split[col] >= 0].groupby(col)["well"]] if col in split else [])
    # consistency: 06 was fitted on exactly these folds
    for f in range(n_outer):
        key = f"outer_{f}"
        if key not in rs["folds"] or sorted(rs["folds"][key]["held_out_wells"]) != outer[f]:
            raise StageBlocked(f"06 fold {key} does not match the 05 split plan - rerun 06_feature_selection")
    support = schema.get("task_support_from_02") or {}
    tasks = {}
    for t in sp["tasks"]:
        if t not in TASKS:
            ctx.warn(f"unknown task '{t}' ignored")
            continue
        target, mask, kind, key, meaning = TASKS[t]
        if support.get(key) == "TASK_UNSUPPORTED":
            ctx.limit(f"task {t} is TASK_UNSUPPORTED in 02 - not modelled")
            continue
        tasks[t] = (target, mask, kind, meaning)
    matched = df[elastic_cols].notna().all(axis=1).to_numpy() if elastic_cols else np.ones(len(df), bool)
    ctx.metric("fraction_rows_matched", round(float(matched.mean()), 4))
    env = {"df": df, "matched": matched, "outer": outer, "inner": inner, "sets": rs["folds"],
           "transforms": transforms, "sp": sp, "dev": dev, "res": res, "reserved": reserved}

    # ---------------------------------------------------------------- 3 tabular
    steps.next("tabular models")
    store = {"registry": [], "hyper": [], "inner": [], "preds": [], "history": [], "audit": [], "arch": {}}
    variants = []
    for m in sp["tabular"]:
        ok, why = mtab.available(m)
        modes = sp["scaling_modes"] if m in ("linear", "mlp") else ["independent"]
        for mode in modes:
            variants.append((m if mode == "independent" else f"{m}[{mode}]", m, mode, ok, why))
    combos = [(t, inp, v) for t in tasks for inp in sp["inputs"] for v in variants]
    bar = progress(combos, desc="tabular candidates", unit="fit-set")
    for t, inp, (label, base_name, mode, ok, why) in bar:
        if hasattr(bar, "set_postfix_str"):
            bar.set_postfix_str(f"{t} | {inp} | {label}")
        if not ok:
            store["registry"].append(_reg_row(t, inp, label, "SKIPPED", why, base_name in mtab.BASELINES))
            continue
        try:
            _nested_tabular(ctx, env, store, t, tasks[t], inp, label, base_name, mode)
        except Exception as exc:  # noqa: BLE001 - one failing candidate must not stop the others
            ctx.warn(f"{t} / {inp} / {label} FAILED: {exc}")
            store["registry"].append(_reg_row(t, inp, label, "FAILED", f"{type(exc).__name__}: {exc}",
                                              base_name in mtab.BASELINES, detail=traceback.format_exc()[-1500:]))

    # ---------------------------------------------------------------- 4 deep
    steps.next("deep models")
    deep_ok, deep_why = mdeep.available()
    patches = _load_patches(ctx, run_dir, cfg) if (deep_ok and sp["deep"]) else None
    deep_tasks = [t for t in mdeep.HEADS if t in tasks]
    for name in sp["deep"]:
        if not deep_ok or patches is None or not deep_tasks:
            why = deep_why or ("03 patches not found" if patches is None else "no deep task active")
            for t in deep_tasks or mdeep.HEADS:
                store["registry"].append(_reg_row(t, DEEP_INPUT, name, "SKIPPED", why, False))
            continue
        try:
            _nested_deep(ctx, env, store, name, patches, deep_tasks)
        except Exception as exc:  # noqa: BLE001
            ctx.warn(f"deep model {name} FAILED: {exc}")
            for t in deep_tasks:
                store["registry"].append(_reg_row(t, DEEP_INPUT, name, "FAILED", f"{type(exc).__name__}: {exc}",
                                                  False, detail=traceback.format_exc()[-1500:]))
    if not deep_ok:
        ctx.limit(f"Deep models skipped: {deep_why}")

    # ---------------------------------------------------------------- 5 write
    steps.next("write")
    out = _write(ctx, store, tasks, env)

    # ---------------------------------------------------------------- 6 figures
    steps.next("figures")
    _figure_comparison(ctx, out["registry"], tasks, sp)
    _figure_tuning(ctx, out["hyper"], tasks, n_outer)
    _figure_learning_curves(ctx, out["history"], deep_ok, deep_why, sp)
    _figure_architecture(ctx, store["arch"], sp)

    # ---------------------------------------------------------------- 7 summary
    steps.next("summary")
    _summarise(ctx, out, tasks, env)
    steps.close()


# =============================================================================== helpers
def _parse_input(inp: str) -> tuple[str, str]:
    g, _, rows = inp.partition("@")
    return g, (rows or "all")


def _features(env, key: str, group: str, task: str) -> tuple[list[str], str]:
    entry = env["sets"][key]["groups"][group]
    sel = entry["selected"].get(task) or []
    if env["sp"]["feature_choice"] == "selected" and sel:
        return list(sel), "selected"
    if not entry["screened"]:
        raise ValueError(f"no screened predictors for {key}/{group}")
    return list(entry["screened"]), ("screened (06 selected none)" if env["sp"]["feature_choice"] == "selected"
                                     else "screened")


def _rows(env, t_spec, inp) -> pd.DataFrame:
    target, mask, kind, _ = t_spec
    df = env["df"]
    m = df[mask].astype(bool).to_numpy() & np.isfinite(df[target].to_numpy(float))
    if kind == "binary":
        m &= df[target].to_numpy(float) >= 0
    if _parse_input(inp)[1] == "matched":
        m &= env["matched"]
    return df[m]


def _design(rows, env, base_name, tkey, feats, mode, twt_ref=None):
    """Model matrix for rows: frozen 06 transform of the features, or [z, z^2] of TWT for depth_trend."""
    if base_name in mtab.BASELINES:
        c, s = twt_ref
        return mtab.depth_features(rows["desired_twt_ms"].to_numpy(float), c, s)
    params = env["transforms"][tkey]["params"]
    return apply_transform(rows, params, feats, mode=mode, add_indicators=False).to_numpy(float)


def _twt_ref(rows):
    v = rows["desired_twt_ms"].to_numpy(float)
    q25, q50, q75 = np.quantile(v, [0.25, 0.5, 0.75])
    return float(q50), float(q75 - q25) or 1.0


def _fit_predict(base_name, kind, params, Xtr, ytr, Xev, sp):
    """Fit one estimator and predict; a training set with one class falls back to its constant."""
    if kind == "binary" and np.unique(ytr).size < 2:
        return None, np.full(Xev.shape[0], float(ytr.mean()) if ytr.size else 0.5), "one class in training rows"
    pw = float((ytr == 0).sum() / max(1, (ytr == 1).sum())) if kind == "binary" else 1.0
    est = mtab.build(base_name, kind, params, seed=int(sp["seed"]), n_jobs=int(sp["n_jobs"]),
                     class_weight=sp["class_weight"], pos_weight=pw)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est.fit(Xtr, ytr)
    note = "did not fully converge" if any("converge" in str(w.message).lower() for w in caught) else ""
    return est, (mtab.predict(est, kind, Xev) if Xev.shape[0] else np.zeros(0)), note


def _reg_row(task, inp, model, status, reason, baseline, detail="", **kw):
    return {"task": task, "input": inp, "model": model, "status": status, "reason": reason,
            "is_baseline": bool(baseline), "detail": detail, **kw}


def _audit(store, task, inp, model, scope, train_w, eval_w):
    store["audit"].append({"task": task, "input": inp, "model": model, "scope": scope,
                           "train_wells": ";".join(sorted(train_w)), "eval_wells": ";".join(sorted(eval_w)),
                           "overlap": ";".join(sorted(set(train_w) & set(eval_w)))})


def _vote(choices: list[int]) -> int:
    """Most frequent grid index across outer folds (ties -> the smallest index = simplest/first)."""
    if not choices:
        return 0
    c = Counter(choices)
    top = max(c.values())
    return min(i for i, n in c.items() if n == top)


# =============================================================================== nested CV (tabular)
def _nested_tabular(ctx, env, store, task, t_spec, inp, label, base_name, mode):
    target, mask, kind, _ = t_spec
    sp = env["sp"]
    group, _ = _parse_input(inp)
    D = _rows(env, t_spec, inp)
    if len(D) < 20 or D["well"].nunique() < 2:
        store["registry"].append(_reg_row(task, inp, label, "SKIPPED", f"only {len(D)} rows in {D['well'].nunique()} "
                                          "wells for this input (matched rows need every inversion input)",
                                          base_name in mtab.BASELINES, n_rows=int(len(D))))
        return
    G = mtab.grid(base_name, (sp.get("grids") or {}).get(base_name))
    prim, _ = PRIMARY[kind]
    fold_info, t0 = [], time.time()
    feat_src = set()
    for f, test_w in env["outer"].items():
        fit_w = [w for w in env["dev"] if w not in test_w]
        scores = {pi: [] for pi in range(len(G))}
        oof = {pi: ([], []) for pi in range(len(G))}
        if not env["inner"][f]:
            store["inner"].append({"task": task, "input": inp, "model": label, "outer_fold": f, "inner_split": -1,
                                   "param_idx": 0, "params": mtab.params_key(G[0]),
                                   "note": "no inner splits (fewer than 2 training wells): first grid point, "
                                           "threshold 0.5"})
        for i, val_w in enumerate(env["inner"][f]):
            tr_w = [w for w in fit_w if w not in val_w]
            key, tkey = f"outer_{f}/inner_{i}", f"{f}/inner{i}"
            feats, src = _features(env, key, group, task) if base_name not in mtab.BASELINES else ([], "TWT only")
            feat_src.add(src)
            Rtr, Rva = D[D["well"].isin(tr_w)], D[D["well"].isin(val_w)]
            if Rtr.empty or Rva.empty:
                continue
            ref = _twt_ref(Rtr)
            Xtr, Xva = (_design(Rtr, env, base_name, tkey, feats, mode, ref), _design(Rva, env, base_name, tkey,
                                                                                    feats, mode, ref))
            ytr, yva = Rtr[target].to_numpy(float), Rva[target].to_numpy(float)
            if kind == "binary":
                ytr, yva = ytr.astype(int), yva.astype(int)
            _audit(store, task, inp, label, f"outer{f}/inner{i}", tr_w, val_w)
            for pi, params in enumerate(G):
                _, p, note = _fit_predict(base_name, kind, params, Xtr, ytr, Xva, sp)
                m = task_metrics(kind, yva, p)
                scores[pi].append(m[prim])
                oof[pi][0].append(yva)
                oof[pi][1].append(p)
                store["inner"].append({"task": task, "input": inp, "model": label, "outer_fold": f, "inner_split": i,
                                       "param_idx": pi, "params": mtab.params_key(params), "n_features": len(feats),
                                       "note": note, **m})
        means = {pi: (np.nanmean(v) if np.isfinite(v).any() else np.nan) for pi, v in scores.items()}
        best = 0
        for pi in range(1, len(G)):
            if is_better(kind, means[pi], means[best]):
                best = pi
        for pi in range(len(G)):
            v = np.asarray(scores[pi], float)
            store["hyper"].append({"task": task, "input": inp, "model": label, "outer_fold": f, "param_idx": pi,
                                   "params": mtab.params_key(G[pi]), "params_json": json.dumps(G[pi]),
                                   f"inner_{prim}_mean": means[pi],
                                   f"inner_{prim}_sd": float(np.nanstd(v)) if np.isfinite(v).any() else np.nan,
                                   "n_inner_splits": int(np.isfinite(v).sum()), "chosen": pi == best})
        thr = 0.5
        inner_summary = {}
        if oof[best][0]:
            yy, pp = np.concatenate(oof[best][0]), np.concatenate(oof[best][1])
            if kind == "binary":
                thr = best_threshold(yy, pp)
            per_split = [task_metrics(kind, a, b, thr) for a, b in zip(*oof[best])]
            for k in per_split[0]:
                vals = np.asarray([d.get(k, np.nan) for d in per_split], float)
                inner_summary[f"inner_{k}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
        # ---- refit on all training wells of the fold, predict the sealed test wells
        feats, src = _features(env, f"outer_{f}", group, task) if base_name not in mtab.BASELINES else ([], "TWT only")
        Rfit, Rte = D[D["well"].isin(fit_w)], D[D["well"].isin(test_w)]
        ref = _twt_ref(Rfit)
        Xfit = _design(Rfit, env, base_name, str(f), feats, mode, ref)
        yfit = Rfit[target].to_numpy(float).astype(int) if kind == "binary" else Rfit[target].to_numpy(float)
        _audit(store, task, inp, label, f"outer{f}/refit", fit_w, test_w)
        Xte = _design(Rte, env, base_name, str(f), feats, mode, ref) if len(Rte) else np.zeros((0, Xfit.shape[1]))
        est, p, note = _fit_predict(base_name, kind, G[best], Xfit, yfit, Xte, sp)
        _store_predictions(store, task, inp, label, f, Rte, target, kind, p, thr)
        if sp["save_models"] == "all" and est is not None:
            _save_tabular(ctx, env, est, f"outer_{f}", task, inp, label, base_name, kind, mode, feats, str(f), G[best],
                          thr, ref)
        fold_info.append({"fold": f, "best": best, "threshold": thr, "n_features": len(feats), "feature_source": src,
                          "inner_primary": means[best], **inner_summary})
    # ---- final model on all development wells
    best_final = _vote([fi["best"] for fi in fold_info])
    thr_final = float(np.median([fi["threshold"] for fi in fold_info])) if fold_info else 0.5
    feats, src = _features(env, "all_development", group, task) if base_name not in mtab.BASELINES else ([], "TWT only")
    ref = _twt_ref(D)
    Xall = _design(D, env, base_name, "all_development", feats, mode, ref)
    yall = D[target].to_numpy(float).astype(int) if kind == "binary" else D[target].to_numpy(float)
    _audit(store, task, inp, label, "final", env["dev"], [])
    est, _, note = _fit_predict(base_name, kind, G[best_final], Xall, yall, Xall[:1], sp)
    path = _save_tabular(ctx, env, est, "final", task, inp, label, base_name, kind, mode, feats, "all_development",
                         G[best_final], thr_final, ref) if est is not None else ""
    fi = pd.DataFrame(fold_info)
    row = _reg_row(task, inp, label, "OK", note, base_name in mtab.BASELINES,
                   kind=kind, primary_metric=prim, n_rows=int(len(D)), n_wells=int(D["well"].nunique()),
                   n_features_by_fold=";".join(str(int(x)) for x in fi["n_features"]) if len(fi) else "",
                   feature_source=";".join(sorted(feat_src | {src})), scaling_mode=mode,
                   chosen_params_by_fold=json.dumps([mtab.params_key(G[b]) for b in fi["best"]]) if len(fi) else "[]",
                   final_params=mtab.params_key(G[best_final]),
                   threshold_by_fold=json.dumps([round(x, 4) for x in fi["threshold"]]) if len(fi) else "[]",
                   final_threshold=thr_final, final_model_file=path, fit_seconds=round(time.time() - t0, 1))
    if len(fi):
        for c in [c for c in fi.columns if c.startswith("inner_")]:
            vals = fi[c].to_numpy(float)
            row[f"{c}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            row[f"{c}_sd_folds"] = float(np.nanstd(vals)) if np.isfinite(vals).any() else np.nan
        row["inner_primary_by_fold"] = json.dumps([None if not np.isfinite(x) else round(float(x), 4)
                                                   for x in fi["inner_primary"]])
        if not len(env["inner"][0]):
            row["reason"] = (row["reason"] + "; " if row["reason"] else "") + "untuned: no inner splits"
    store["registry"].append(row)


def _store_predictions(store, task, inp, model, f, R, target, kind, p, thr):
    if R is None or not len(R):
        return
    d = pd.DataFrame({"task": task, "input": inp, "model": model, "outer_fold": f, "sample_id": R["sample_id"].to_numpy(),
                      "well": R["well"].to_numpy(), "desired_twt_ms": R["desired_twt_ms"].to_numpy(),
                      "y_true": R[target].to_numpy(float)})
    if kind == "binary":
        d["y_pred"] = np.asarray(p, float)
        d["threshold"] = thr
        d["y_pred_class"] = (d["y_pred"] >= thr).astype(int)
        d["y_pred_raw"] = d["y_pred"]
        d["clipped"] = False
    else:
        b, _ = apply_bounds(task, p)
        d["y_pred_raw"] = np.asarray(p, float)
        d["y_pred"] = b
        d["clipped"] = d["y_pred_raw"] != d["y_pred"]
        d["threshold"] = np.nan
        d["y_pred_class"] = -1
    store["preds"].append(d)


def _safe(s: str) -> str:
    return s.replace("@", "_at_").replace("[", "_").replace("]", "").replace("/", "_")


def _save_tabular(ctx, env, est, scope, task, inp, label, base_name, kind, mode, feats, tkey, params, thr, ref):
    import joblib

    rel = f"models/{scope}/{task}__{_safe(inp)}__{_safe(label)}.joblib"
    path = ctx.stage_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"estimator": est, "task": task, "kind": kind, "input": inp, "model": label, "base_model": base_name,
                 "features": feats, "transform_file": f"06_feature_selection/transforms/transforms_{env['res']}ms.json",
                 "transform_key": tkey, "scaling_mode": mode, "params": params, "threshold": thr,
                 "depth_trend_centre_scale": list(ref) if base_name in mtab.BASELINES else None,
                 "bounds_applied_at_prediction": kind == "regression", "code_version": ctx.run.code_version},
                path, compress=3)
    ctx._register(path, "model", f"{label} for {task} on {inp} ({scope})")
    return rel


# =============================================================================== deep models
def _load_patches(ctx, run_dir, cfg):
    ex = run_dir / "03_extraction" / "patches"
    version = str(cfg_get(cfg, "extraction.version", "v1"))
    man_p = ex / f"patch_manifest_{version}.json"
    if not man_p.exists():
        ctx.warn(f"{man_p} not found: deep models need the 03 patches")
        return None
    man = json.loads(man_p.read_text(encoding="utf-8"))
    arr_p = ex / man["file"]
    if not arr_p.exists():
        ctx.warn(f"{arr_p} not found")
        return None
    ctx.add_input(arr_p, "upstream:03 patches (deep models)")
    stacks = [s for s in ((cfg.get("features") or {}).get("stacks") or ["full", "near", "mid", "far", "ultrafar"])
              if s in man["channels"]]
    return {"array": np.load(arr_p, mmap_mode="r"), "channels": man["channels"],
            "stack_idx": [man["channels"].index(s) for s in stacks], "stacks": stacks}


def _nested_deep(ctx, env, store, name, patches, heads):
    sp = env["sp"]
    df = env["df"]
    lab = (df["y_lithology"].to_numpy(float) >= 0) | df["mask_vsh"].to_numpy(bool) | df["mask_phit"].to_numpy(bool)
    rows = df[lab & (df["patch_row"].to_numpy() >= 0)].reset_index(drop=True)
    P = np.asarray(patches["array"][rows["patch_row"].to_numpy(int)][:, patches["stack_idx"]], dtype=np.float32)
    nanfrac = 1.0 - np.isfinite(P).reshape(len(P), -1).mean(1)
    keep = nanfrac <= float(sp["deep_max_nan_fraction"])
    rows, P = rows[keep].reset_index(drop=True), P[keep]
    T = mdeep.make_targets(rows)
    dev = mdeep.device_name(sp["deep_device"])
    kw = dict(epochs=int(sp["deep_epochs"]), batch_size=int(sp["deep_batch_size"]), patience=int(sp["deep_patience"]),
              dropout=float(sp["deep_dropout"]), seed=int(sp["seed"]), device=dev)
    ctx.info("deep %s: %d rows, input %s, device %s", name, len(rows), list(P.shape[1:]), dev)
    wells = rows["well"].to_numpy()
    sub = (lambda idx: {k: v[idx] for k, v in T.items()})
    fold_info, t0 = [], time.time()
    for f, test_w in env["outer"].items():
        fit_w = [w for w in env["dev"] if w not in test_w]
        epochs_best, oof = [], {h: ([], []) for h in heads}
        for i, val_w in enumerate(env["inner"][f]):
            tr = np.flatnonzero(np.isin(wells, [w for w in fit_w if w not in val_w]))
            va = np.flatnonzero(np.isin(wells, val_w))
            if not tr.size or not va.size:
                continue
            sc = mdeep.fit_patch_scaling(P[tr], sp["deep_scaling"])
            r = mdeep.train(name, mdeep.apply_patch_scaling(P[tr], sc), sub(tr), mdeep.apply_patch_scaling(P[va], sc),
                            sub(va), **kw)
            _audit(store, "multi-task", DEEP_INPUT, name, f"outer{f}/inner{i}", [w for w in fit_w if w not in val_w],
                   val_w)
            epochs_best.append(r["best_epoch"])
            for h in r["history"]:
                store["history"].append({"model": name, "outer_fold": f, "inner_split": i, **h,
                                         "best_epoch": r["best_epoch"]})
            pred = mdeep.predict(r["model"], mdeep.apply_patch_scaling(P[va], sc), dev)
            for h in heads:
                m = T[f"m_{h}"][va]
                y = rows[TASKS[h][0]].to_numpy(float)[va][m]
                oof[h][0].append(y.astype(int) if h == "lithology" else y)
                oof[h][1].append(pred[h][m])
                kind = TASKS[h][2]
                store["inner"].append({"task": h, "input": DEEP_INPUT, "model": name, "outer_fold": f, "inner_split": i,
                                       "param_idx": 0, "params": f"best_epoch={r['best_epoch']}",
                                       "n_features": int(np.prod(P.shape[1:])), "note": "",
                                       **task_metrics(kind, oof[h][0][-1], oof[h][1][-1])})
        n_ep = int(np.median(epochs_best)) if epochs_best else int(sp["deep_epochs"])
        thr = best_threshold(np.concatenate(oof["lithology"][0]), np.concatenate(oof["lithology"][1])) \
            if "lithology" in heads and oof["lithology"][0] else 0.5
        inner_by_head = {}
        for h in heads:
            if oof[h][0]:
                kind = TASKS[h][2]
                ms = [task_metrics(kind, a, b, thr) for a, b in zip(*oof[h])]
                inner_by_head[h] = {f"inner_{k}": float(np.nanmean([d.get(k, np.nan) for d in ms])) for k in ms[0]}
        # ---- refit on the fold's training wells, predict the sealed test wells
        fi_idx = np.flatnonzero(np.isin(wells, fit_w))
        te_idx = np.flatnonzero(np.isin(wells, test_w))
        sc = mdeep.fit_patch_scaling(P[fi_idx], sp["deep_scaling"])
        r = mdeep.train(name, mdeep.apply_patch_scaling(P[fi_idx], sc), sub(fi_idx), **{**kw, "epochs": n_ep})
        _audit(store, "multi-task", DEEP_INPUT, name, f"outer{f}/refit", fit_w, test_w)
        if te_idx.size:
            pred = mdeep.predict(r["model"], mdeep.apply_patch_scaling(P[te_idx], sc), dev)
            for h in heads:
                m = T[f"m_{h}"][te_idx]
                R = rows.iloc[te_idx[m]]
                _store_predictions(store, h, DEEP_INPUT, name, f, R, TASKS[h][0], TASKS[h][2], pred[h][m], thr)
        if sp["save_models"] == "all":
            _save_deep(ctx, r["model"], f"outer_{f}", name, sc, n_ep, thr, patches)
        fold_info.append({"fold": f, "epochs": n_ep, "threshold": thr, "inner": inner_by_head})
    # ---- final model on all development wells
    n_final = int(np.median([fi["epochs"] for fi in fold_info])) if fold_info else int(sp["deep_epochs"])
    thr_final = float(np.median([fi["threshold"] for fi in fold_info])) if fold_info else 0.5
    sc = mdeep.fit_patch_scaling(P, sp["deep_scaling"])
    r = mdeep.train(name, mdeep.apply_patch_scaling(P, sc), T, **{**kw, "epochs": n_final})
    _audit(store, "multi-task", DEEP_INPUT, name, "final", env["dev"], [])
    path = _save_deep(ctx, r["model"], "final", name, sc, n_final, thr_final, patches)
    store["arch"][name] = mdeep.describe(r["model"])
    for h in heads:
        kind = TASKS[h][2]
        prim = PRIMARY[kind][0]
        row = _reg_row(h, DEEP_INPUT, name, "OK", "", False, kind=kind, primary_metric=prim, n_rows=int(T[f"m_{h}"].sum()),
                       n_wells=int(rows.loc[T[f"m_{h}"], "well"].nunique()),
                       n_features_by_fold=str(int(np.prod(P.shape[1:]))),
                       feature_source=f"03 patches {patches['stacks']} ({sp['deep_scaling']} amplitude scaling)",
                       scaling_mode=sp["deep_scaling"],
                       chosen_params_by_fold=json.dumps([f"epochs={fi['epochs']}" for fi in fold_info]),
                       final_params=f"epochs={n_final}", final_threshold=thr_final if h == "lithology" else np.nan,
                       threshold_by_fold=json.dumps([round(fi["threshold"], 4) for fi in fold_info]),
                       final_model_file=path, fit_seconds=round(time.time() - t0, 1),
                       n_parameters=store["arch"][name]["n_parameters"])
        vals = [fi["inner"].get(h, {}) for fi in fold_info]
        keys = sorted({k for v in vals for k in v})
        for k in keys:
            arr = np.asarray([v.get(k, np.nan) for v in vals], float)
            row[f"{k}_mean"] = float(np.nanmean(arr)) if np.isfinite(arr).any() else np.nan
            row[f"{k}_sd_folds"] = float(np.nanstd(arr)) if np.isfinite(arr).any() else np.nan
        row["inner_primary_by_fold"] = json.dumps([None if not np.isfinite(v.get(f"inner_{prim}", np.nan)) else
                                                   round(v[f"inner_{prim}"], 4) for v in vals])
        store["registry"].append(row)
        for fi in fold_info:
            store["hyper"].append({"task": h, "input": DEEP_INPUT, "model": name, "outer_fold": fi["fold"],
                                   "param_idx": 0, "params": f"epochs={fi['epochs']}",
                                   "params_json": json.dumps({"epochs": fi["epochs"]}),
                                   f"inner_{prim}_mean": fi["inner"].get(h, {}).get(f"inner_{prim}", np.nan),
                                   "chosen": True})


def _save_deep(ctx, model, scope, name, sc, epochs, thr, patches):
    import torch

    rel = f"models/{scope}/{name}.pt"
    path = ctx.stage_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": mdeep.state_for_saving(model), "model": name, "channels": patches["stacks"],
                "scaling": sc, "epochs": epochs, "lithology_threshold": thr, "heads": list(mdeep.HEADS),
                "upper_bounds": mdeep.UPPER, "code_version": ctx.run.code_version}, path)
    ctx._register(path, "model", f"{name} weights ({scope})")
    return rel


# =============================================================================== write
def _write(ctx, store, tasks, env) -> dict:
    reg = pd.DataFrame(store["registry"])
    if len(reg):
        reg["champion_inner_cv"] = False
        reg["beats_depth_trend_inner"] = pd.Series([None] * len(reg), index=reg.index, dtype=object)
        for (t, inp), g in reg.groupby(["task", "input"]):
            kind = tasks[t][2] if t in tasks else "binary"
            prim, direction = PRIMARY[kind]
            col = f"inner_{prim}_mean"
            if col not in g:
                continue
            cand = g[(g["status"] == "OK") & ~g["is_baseline"] & g[col].notna()]
            if len(cand):
                best = cand[col].idxmax() if direction == "max" else cand[col].idxmin()
                reg.loc[best, "champion_inner_cv"] = True
        for t in reg["task"].unique():
            kind = tasks[t][2] if t in tasks else "binary"
            prim, _ = PRIMARY[kind]
            col = f"inner_{prim}_mean"
            if col not in reg:
                continue
            ref = reg[(reg["task"] == t) & (reg["model"] == "depth_trend")].set_index("input")[col]
            for idx in reg.index[(reg["task"] == t) & (reg["status"] == "OK")]:
                inp = reg.at[idx, "input"]
                base = ref.get(inp if inp in ref.index else "stacks", np.nan)
                v = reg.at[idx, col]
                if np.isfinite(v) and np.isfinite(base):
                    reg.at[idx, "beats_depth_trend_inner"] = bool(is_better(kind, v, base))
    hyper = pd.DataFrame(store["hyper"])
    inner = pd.DataFrame(store["inner"])
    preds = pd.concat(store["preds"], ignore_index=True) if store["preds"] else pd.DataFrame()
    hist = pd.DataFrame(store["history"])
    audit = pd.DataFrame(store["audit"])
    ctx.save_table(reg, "model_registry.csv", "One row per task / input / model: status, rows, features, chosen "
                   "hyperparameters per fold, final hyperparameters and threshold, INNER-CV metrics (mean and sd over "
                   "outer folds), model file, champion flag (declared from inner scores only)", required=True)
    ctx.save_table(hyper if len(hyper) else pd.DataFrame(columns=["task"]), "hyperparameters.csv",
                   "Every grid point of every model per outer fold with its inner-CV score; 'chosen' marks the "
                   "selected point", required=True)
    ctx.save_table(inner if len(inner) else pd.DataFrame(columns=["task"]), "inner_cv_scores.csv",
                   "Metrics on each inner-validation well group (per grid point)", required=True)
    ctx.save_table(preds if len(preds) else pd.DataFrame(columns=["task"]), "outer_test_predictions.csv",
                   "SEALED predictions for the outer test wells of each fold (evaluated in 08); regression values "
                   "clipped to physical bounds (raw value and 'clipped' flag kept)", required=True)
    ctx.save_table(hist if len(hist) else pd.DataFrame(columns=["model"]), "deep_training_history.csv",
                   "Train / validation loss per epoch for the deep models (inner splits)")
    ctx.save_table(audit, "fit_audit.csv", "Every fit: training wells and the wells it was evaluated on (must never "
                   "overlap)", required=True)
    overlap = audit[audit["overlap"] != ""] if len(audit) else audit
    bad_res = audit[audit["train_wells"].str.split(";").apply(lambda w: bool(set(w) & set(env["reserved"])))] \
        if len(audit) else audit
    ctx.check("no_held_out_well_in_any_fit", overlap.empty and bad_res.empty,
              f"{len(audit)} fits audited; overlaps {len(overlap)}, reserved-well fits {len(bad_res)}")
    if len(preds):
        dup = preds.groupby(["task", "input", "model", "sample_id"]).size()
        ctx.check("each_row_predicted_once_per_candidate", bool((dup == 1).all()),
                  f"{len(preds)} sealed predictions; duplicates {int((dup > 1).sum())}")
        wrong = 0
        for f, test_w in env["outer"].items():
            wrong += int((~preds.loc[preds["outer_fold"] == f, "well"].isin(test_w)).sum())
        ctx.check("predictions_only_on_own_test_wells", wrong == 0, f"{wrong} predictions outside their fold's test wells")
    arch = {"tabular": {m: {"grid": mtab.GRIDS.get(m.split("[")[0], {})} for m in env["sp"]["tabular"]},
            "deep": store["arch"], "transform_policy": "06 frozen fold-local transforms; no missing-value indicators "
            "(06 found the inversion-coverage pattern predicts targets); matched rows for Stacks vs Fused",
            "selection_rule": "grid point with the best mean inner primary metric (ROC-AUC / RMSE); threshold "
                              "maximising balanced accuracy on the inner predictions; final = most frequent choice"}
    ctx.save_json(arch, "architectures/model_architectures.json", "Grids, deep architectures and parameter counts",
                  required=True)
    for name, d in store["arch"].items():
        ctx.save_text(d["architecture"] + f"\n\nparameters: {d['n_parameters']}\n", f"architectures/{name}.txt",
                      f"{name} layer list")
    return {"registry": reg, "hyper": hyper, "inner": inner, "preds": preds, "history": hist, "audit": audit}


# =============================================================================== figures
INPUT_COLORS = {"stacks": SERIES[0], "stacks@matched": SERIES[2], "fused@matched": SERIES[1], DEEP_INPUT: SERIES[6]}
TASK_TITLES = {"lithology": "sand vs shale", "sand_fluid": "water vs HC-potential sand", "vsh": "VSH",
               "phit": "PHIT", "swt": "SWT"}


def _figure_comparison(ctx, reg, tasks, sp):
    import matplotlib.pyplot as plt

    tl = list(tasks)
    fig, axs = plt.subplots(2, 3, figsize=(21, 13), squeeze=False)
    for ax, t in zip(axs.ravel(), tl):
        kind = tasks[t][2]
        prim, direction = PRIMARY[kind]
        col = f"inner_{prim}_mean"
        g = reg[(reg["task"] == t)] if len(reg) else reg
        models = list(dict.fromkeys(g["model"])) if len(g) else []
        if not models or col not in g:
            ax.text(0.5, 0.5, f"{t}: no candidate ran", ha="center", transform=ax.transAxes, color=STATUS["critical"])
            ax.set_axis_off()
            continue
        inputs = [i for i in list(sp["inputs"]) + [DEEP_INPUT] if i in set(g["input"])]
        labelled = set()
        for yi, m in enumerate(models):
            for k, inp in enumerate(inputs):
                r = g[(g["model"] == m) & (g["input"] == inp)]
                if r.empty:
                    continue
                r = r.iloc[0]
                y = yi + (k - (len(inputs) - 1) / 2) * 0.22
                val = r.get(col, np.nan)
                if r["status"] != "OK" or not np.isfinite(val if val is not None else np.nan):
                    ax.text(0.01, y, f"{inp}: {r['status']}", fontsize=6.5, color=TEXT_2, va="center",
                            transform=ax.get_yaxis_transform())
                    continue
                raw = r.get("inner_primary_by_fold")
                folds = [v for v in (json.loads(raw) if isinstance(raw, str) else []) if v is not None]
                if folds:
                    ax.plot([min(folds), max(folds)], [y, y], color=INPUT_COLORS.get(inp, NEUTRAL), lw=1.4, alpha=0.7)
                champ = bool(r["champion_inner_cv"])
                ax.scatter(r[col], y, s=110 if champ else 30, zorder=3, marker="*" if champ else "o",
                           color=INPUT_COLORS.get(inp, NEUTRAL), edgecolor="#0b0b0b" if champ else "none",
                           label=None if inp in labelled else inp)
                labelled.add(inp)
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([f"{m} (baseline)" if m in mtab.BASELINES else m for m in models], fontsize=8.5)
        ax.set_ylim(len(models) - 0.4, -0.6)
        ax.set_xlabel(f"inner-CV {prim} ({'higher' if direction == 'max' else 'lower'} is better); dot = mean of the "
                      "4 outer folds, bar = range", fontsize=8.5)
        if kind == "binary":
            ax.axvline(0.5, color=NEUTRAL, lw=0.8, ls="--")
        ax.set_title(TASK_TITLES.get(t, t), fontsize=11)
        if labelled:
            ax.legend(fontsize=7.5, loc="best")
    for ax in axs.ravel()[len(tl):]:
        ax.set_axis_off()
    axs.ravel()[-1].text(0, 0.95, "How to read\n\n* every score comes from INNER validation wells\n  (inside each outer "
                         "training set)\n* outer test wells are untouched here:\n  held-out results are stage 08\n"
                         "* star = champion declared from inner scores\n* depth_trend uses TWT only: a seismic model\n"
                         "  must beat it to claim rock information\n* @matched = rows where every inversion\n  input "
                         "exists (fair Stacks vs Fused)", va="top", fontsize=10, family="monospace",
                         transform=axs.ravel()[-1].transAxes)
    fig.suptitle("07_models: candidate comparison by nested inner cross-validation (whole wells)", fontsize=13,
                 fontweight="bold")
    footer(fig, "ROC-AUC for the binary tasks (threshold-free), RMSE (v/v) for VSH / PHIT / SWT. Hyperparameters, "
                "thresholds and epochs were chosen on the same inner splits, so these scores are mildly optimistic; "
                "08 reports the sealed outer-test predictions.")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    ctx.save_figure(fig, "model_comparison_inner_cv.png", "Inner-CV comparison of all candidates", required=True,
                    multipanel=True)


def _figure_tuning(ctx, hyper, tasks, n_outer):
    import matplotlib.pyplot as plt

    show_t = [t for t in ("lithology", "vsh", "phit") if t in tasks] or list(tasks)[:3]
    models = [m for m in ("linear", "extratrees", "hist_gbm", "xgboost", "mlp")
              if len(hyper) and m in set(hyper["model"])]
    if not models or not show_t:
        fig = not_available_figure("07_models: tuning curves", ["no tuned tabular model ran"])
        ctx.save_figure(fig, "tuning_curves.png", "Inner-CV score per grid point")
        return
    fig, axs = plt.subplots(len(models), len(show_t), figsize=(6.5 * len(show_t), 2.6 * len(models) + 1.5),
                            squeeze=False)
    colors = [SERIES[0], SERIES[1], SERIES[2], SERIES[6], SERIES[4]]
    for j, t in enumerate(show_t):
        prim = PRIMARY[tasks[t][2]][0]
        for i, m in enumerate(models):
            ax = axs[i, j]
            d = hyper[(hyper["task"] == t) & (hyper["model"] == m) & (hyper["input"] == "stacks")]
            col = f"inner_{prim}_mean"
            if d.empty or col not in d:
                ax.set_axis_off()
                continue
            for f in range(n_outer):
                g = d[d["outer_fold"] == f].sort_values("param_idx")
                ax.plot(g["param_idx"], g[col], "o-", color=colors[f % 5], lw=1.2, ms=4, label=f"fold {f}")
                c = g[g["chosen"]]
                ax.scatter(c["param_idx"], c[col], s=80, facecolor="none", edgecolor="#0b0b0b", zorder=4)
            labels = d.drop_duplicates("param_idx").sort_values("param_idx")["params"].tolist()
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels([lab.replace(";", "\n") for lab in labels], fontsize=6.5)
            ax.set_ylabel(f"{m}\ninner {prim}", fontsize=8.5)
            if i == 0:
                ax.set_title(f"{TASK_TITLES.get(t, t)} (stacks)", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("07_models: inner-CV score of every grid point (circle = chosen per outer fold)", fontsize=13,
                 fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    ctx.save_figure(fig, "tuning_curves.png", "Validation curves of the tuned tabular models", multipanel=True)


def _figure_learning_curves(ctx, hist, deep_ok, deep_why, sp):
    import matplotlib.pyplot as plt

    if hist is None or hist.empty:
        reasons = [deep_why] if not deep_ok else ["no deep model was trained (see model_registry.csv)"]
        fig = not_available_figure("07_models: learning curves of the deep models", reasons + [
            "The tabular models are unaffected; their validation curves are in tuning_curves.png."])
        ctx.save_figure(fig, "learning_curves_deep.png", "Deep-model learning curves (not available)", required=True)
        return
    models = list(dict.fromkeys(hist["model"]))
    folds = sorted(hist["outer_fold"].unique())
    fig, axs = plt.subplots(len(models), len(folds), figsize=(5.2 * len(folds), 3.6 * len(models) + 1), squeeze=False)
    colors = [SERIES[0], SERIES[1], SERIES[2]]
    for i, m in enumerate(models):
        for j, f in enumerate(folds):
            ax = axs[i, j]
            d = hist[(hist["model"] == m) & (hist["outer_fold"] == f)]
            for k, (s, g) in enumerate(d.groupby("inner_split")):
                ax.plot(g["epoch"], g["train_loss"], color=colors[k % 3], lw=1, ls="--")
                if "val_loss" in g:
                    ax.plot(g["epoch"], g["val_loss"], color=colors[k % 3], lw=1.6, label=f"inner split {s}")
                    be = int(g["best_epoch"].iloc[0])
                    ax.axvline(be, color=colors[k % 3], lw=0.7, ls=":")
            ax.set_title(f"{m} - outer fold {f}", fontsize=9.5)
            ax.set_xlabel("epoch")
            ax.set_ylabel("loss (masked multi-task)")
            if j == 0:
                ax.legend(fontsize=7)
    fig.suptitle("07_models: deep-model learning curves (dashed = training, solid = inner validation wells, dotted = "
                 "best epoch)", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ctx.save_figure(fig, "learning_curves_deep.png", "Train/validation loss per epoch", required=True, multipanel=True)


def _figure_architecture(ctx, arch, sp):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 10)
    ax.set_axis_off()

    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08", facecolor=color, edgecolor=TEXT_2,
                                    alpha=0.25))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "color": TEXT_2, "lw": 1.3})

    ax.text(0.2, 9.6, "Tabular path", fontsize=12, fontweight="bold")
    box(0.2, 7.6, 3.2, 1.5, "05 training table\n82 raw predictors\n(73 stack + 9 inversion)", SERIES[0])
    box(4.0, 7.6, 3.6, 1.5, "06 frozen transform\n(fold-local median/IQR,\nclip, impute)\n+ 06 feature set", SERIES[2])
    box(8.2, 7.6, 5.2, 1.5, "prior | depth_trend (TWT only)\nlinear | ExtraTrees | HistGBM\nXGBoost | MLP (64-32)",
        SERIES[1])
    box(14.0, 7.6, 5.6, 1.5, "lithology / HC-potential: P(class 1)\n+ threshold from inner wells\nVSH / PHIT / SWT: "
        "value clipped\nto physical bounds (flagged)", SERIES[3])
    arrow(3.4, 8.35, 4.0, 8.35)
    arrow(7.6, 8.35, 8.2, 8.35)
    arrow(13.4, 8.35, 14.0, 8.35)
    ax.text(0.2, 6.7, "Patch path (multi-task, PyTorch)", fontsize=12, fontweight="bold")
    box(0.2, 3.3, 3.2, 3.0, "03 patch\n5 stacks x 5 IL x 5 XL\nx 21 samples (80 ms)\n\ncommon amplitude\nscaling "
        "(training wells)", SERIES[0])
    box(4.0, 4.9, 6.0, 1.5, "TCN: centre 3x3 mean trace [5 x 21]\nConv1d 5->32 k5 | dilated k5 d2 | k3 d4\n"
        "(BatchNorm, ReLU, dropout)", SERIES[2])
    box(4.0, 3.0, 6.0, 1.5, "3-D CNN: [5 x 5 x 5 x 21]\nConv3d 5->16 (3x3x5) | 16->32 (3x3x5)\n(BatchNorm, ReLU, "
        "dropout)", SERIES[2])
    box(10.6, 3.6, 3.4, 2.2, "features:\ncentre sample (+/-2)\n+ global mean\n-> Linear 64->32", SERIES[1])
    box(14.6, 4.9, 5.0, 1.0, "lithology: logit -> P(sand) (masked BCE)", SERIES[3])
    box(14.6, 3.6, 5.0, 1.0, "VSH = sigmoid x 1.0   (masked MSE)", SERIES[3])
    box(14.6, 2.3, 5.0, 1.0, "PHIT = sigmoid x 0.476 (masked MSE)", SERIES[3])
    arrow(3.4, 5.6, 4.0, 5.6)
    arrow(3.4, 3.8, 4.0, 3.8)
    arrow(10.0, 5.6, 10.6, 5.0)
    arrow(10.0, 3.8, 10.6, 4.3)
    for y in (5.4, 4.1, 2.8):
        arrow(14.0, 4.7, 14.6, y)
    txt = "; ".join(f"{k}: {v['n_parameters']:,} parameters" for k, v in arch.items()) if arch else \
        "deep models not trained in this run (see learning_curves_deep.png)"
    ax.text(0.2, 1.4, "Validation: 4 outer folds of whole wells (no-1 reserved), 3 inner well groups per fold for every "
            "choice (grid point, threshold, epochs); final models refitted on all 12 development wells.", fontsize=10)
    ax.text(0.2, 0.8, txt, fontsize=10, color=TEXT_2)
    fig.suptitle("07_models: architectures", fontsize=13, fontweight="bold")
    ctx.save_figure(fig, "architecture_diagram.png", "Schematic of the tabular and patch model paths", required=True)


# =============================================================================== summary
def _summarise(ctx, out, tasks, env):
    reg = out["registry"]
    if reg.empty:
        ctx.block("No candidate model ran")
        return
    ok = reg[reg["status"] == "OK"]
    ctx.metric("n_candidates", {s: int((reg["status"] == s).sum()) for s in ("OK", "SKIPPED", "FAILED")})
    failed = reg[reg["status"] == "FAILED"]
    if len(failed):
        ctx.limit(f"{len(failed)} candidate(s) FAILED (see model_registry.csv 'detail'): " +
                  ", ".join(f"{r.task}/{r.input}/{r.model}" for r in failed.itertuples())[:600])
    champs = {}
    for t in tasks:
        prim = PRIMARY[tasks[t][2]][0]
        col = f"inner_{prim}_mean"
        for inp in list(env["sp"]["inputs"]) + [DEEP_INPUT]:
            c = ok[(ok["task"] == t) & (ok["input"] == inp) & ok["champion_inner_cv"]]
            d = ok[(ok["task"] == t) & (ok["input"] == inp) & (ok["model"] == "depth_trend")]
            if len(c):
                champs[f"{t} | {inp}"] = (f"{c.iloc[0]['model']} inner {prim} {c.iloc[0][col]:.3f}"
                                          + (f" (depth_trend {d.iloc[0][col]:.3f})" if len(d) and col in d else ""))
    ctx.metric("champions_by_inner_cv", champs)
    for t in tasks:
        s = ok[(ok["task"] == t) & (ok["input"] == "stacks") & ok["champion_inner_cv"]]
        if len(s) and s.iloc[0].get("beats_depth_trend_inner") is False:
            ctx.warn(f"{t}: the best stacks model does not beat the TWT-only depth trend on inner CV - its apparent "
                     "skill may be burial depth, not rock properties")
    clipped = out["preds"].groupby(["task", "model", "input"])["clipped"].mean() if len(out["preds"]) else pd.Series()
    big = clipped[clipped > 0.01]
    if len(big):
        ctx.warn("Some regression predictions fell outside physical bounds and were clipped (> 1 % of rows): " +
                 "; ".join(f"{a}/{c}/{b} {100 * v:.1f}%" for (a, b, c), v in big.items())[:500])
    ctx.limit("All scores here are INNER-CV scores used for choices (mildly optimistic); held-out-well performance "
              "is evaluated only in 08 from outer_test_predictions.csv")
    ctx.limit("Targets are provisional interpreted-log labels tied with a provisional T-Z; inversion inputs are "
              "conditional (provenance unaudited)")
    ctx.readme("Outputs for 08 and 09", """
* `tables/outer_test_predictions.csv` - sealed predictions of every candidate for the outer test wells (08 evaluates).
* `tables/model_registry.csv` - chosen hyperparameters, thresholds, inner-CV metrics, final model files,
  champions declared from inner scores only.
* `models/final/*.joblib` - dict with estimator, features, 06 transform key, scaling mode, threshold;
  `models/final/*.pt` - deep weights + amplitude scaling + epochs + threshold.
* `tables/fit_audit.csv` - for every fit, the training wells and the wells it was evaluated on (never overlapping).
""")
