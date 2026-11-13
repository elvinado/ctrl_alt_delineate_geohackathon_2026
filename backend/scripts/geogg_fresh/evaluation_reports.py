"""Held-out-well evaluation helpers for stage 08 (pure numpy / pandas, no file access).

Input everywhere: a frame of SEALED outer-test predictions from 07 with columns
``well, y_true, y_pred, y_pred_class`` (binary) or ``well, y_true, y_pred`` (regression).

Why several numbers per model?
------------------------------
* **pooled** metrics use every held-out row at once.  Probabilities from different wells are
  compared with each other, so a model whose scores shift from well to well is penalised.
* **within-well** metrics are computed inside each test well and then summarised.  They answer
  "does the model rank the rows of ONE well correctly?" - what a log-like prediction track needs.
* **well-bootstrap intervals** resample whole WELLS with replacement (the independent units),
  never rows: with 12 wells, honest intervals are wide.
* **paired comparisons** (model vs depth trend, Fused vs Stacks) use exactly the same rows and the
  same bootstrap wells for both sides, so the difference is not blurred by row differences.

References: Efron & Tibshirani 1993 (bootstrap); Field & Welsh 2007 (cluster bootstrap);
Angelopoulos & Bates 2023 and Barber et al. 2021 (conformal intervals); Niculescu-Mizil &
Caruana 2005 (reliability diagrams).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import _auc, _average_precision, regression_metrics


# =============================================================================== point metrics
def class_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    """Balanced accuracy, macro F1 and per-class precision / recall / F1 from hard labels (0/1)."""
    y, yhat = np.asarray(y, int), np.asarray(yhat, int)
    out = {}
    recalls, f1s = [], []
    for c in (0, 1):
        tp = int(((yhat == c) & (y == c)).sum())
        fp = int(((yhat == c) & (y != c)).sum())
        fn = int(((yhat != c) & (y == c)).sum())
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
        out.update({f"precision_{c}": prec, f"recall_{c}": rec, f"f1_{c}": f1, f"support_{c}": int((y == c).sum())})
        recalls.append(rec)
        f1s.append(f1)
    out["balanced_acc"] = float(np.nanmean(recalls)) if np.isfinite(recalls).any() else np.nan
    out["f1_macro"] = float(np.nanmean(f1s)) if np.isfinite(f1s).any() else np.nan
    return out


def binary_scores(d: pd.DataFrame) -> dict:
    """Probability metrics + hard-label metrics (using the stored fold thresholds)."""
    y = d["y_true"].to_numpy(int)
    p = d["y_pred"].to_numpy(float)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"n": int(y.size), "n_pos": int((y == 1).sum()), "prevalence": float(y.mean()) if y.size else np.nan,
           "roc_auc": _auc(y, p), "pr_auc": _average_precision(y, p),
           "brier": float(((p - y) ** 2).mean()) if y.size else np.nan,
           "log_loss": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()) if y.size else np.nan}
    if "y_pred_class" in d:
        cm = class_metrics(y, d["y_pred_class"].to_numpy(int))
        out.update({k: cm[k] for k in ("balanced_acc", "f1_macro")})
    return out


def scores(kind: str, d: pd.DataFrame) -> dict:
    if kind == "binary":
        return binary_scores(d)
    m = regression_metrics(d["y_true"].to_numpy(float), d["y_pred"].to_numpy(float))
    if "clipped" in d:
        m["fraction_clipped"] = float(d["clipped"].astype(bool).mean())
    return m


def per_well(kind: str, d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w, g in d.groupby("well", sort=True):
        rows.append({"well": w, **scores(kind, g)})
    return pd.DataFrame(rows)


# =============================================================================== bootstrap over wells
def _metric_fn(kind: str, metric: str):
    if kind == "binary":
        if metric == "roc_auc":
            return lambda y, p: _auc(y.astype(int), p)
        if metric == "pr_auc":
            return lambda y, p: _average_precision(y.astype(int), p)
        if metric == "brier":
            return lambda y, p: float(((p - y) ** 2).mean())
    if metric == "rmse":
        return lambda y, p: float(np.sqrt(((p - y) ** 2).mean()))
    if metric == "mae":
        return lambda y, p: float(np.abs(p - y).mean())
    if metric == "spearman":
        return lambda y, p: float(stats.spearmanr(y, p).statistic) if np.ptp(p) > 0 and np.ptp(y) > 0 else np.nan
    if metric == "r2":
        return lambda y, p: float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if np.ptp(y) > 0 else np.nan
    raise ValueError(f"no bootstrap function for {kind}/{metric}")


def _well_blocks(d: pd.DataFrame, cols: list[str]) -> tuple[list[str], list[tuple]]:
    wells = sorted(d["well"].unique())
    blocks = [tuple(d.loc[d["well"] == w, c].to_numpy(float) for c in cols) for w in wells]
    return wells, blocks


def well_bootstrap(kind: str, metric: str, d: pd.DataFrame, n_boot: int = 500, seed: int = 0,
                   alpha: float = 0.05) -> dict:
    """Percentile CI of a pooled metric when whole WELLS are resampled with replacement."""
    f = _metric_fn(kind, metric)
    wells, blocks = _well_blocks(d, ["y_true", "y_pred"])
    k = len(wells)
    point = f(d["y_true"].to_numpy(float), d["y_pred"].to_numpy(float))
    if k < 3:
        return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan, "n_wells": k, "n_boot": 0}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        y = np.concatenate([blocks[i][0] for i in pick])
        p = np.concatenate([blocks[i][1] for i in pick])
        v = f(y, p)
        if np.isfinite(v):
            vals.append(v)
    vals = np.asarray(vals)
    if vals.size < 20:
        return {"point": point, "ci_lo": np.nan, "ci_hi": np.nan, "n_wells": k, "n_boot": int(vals.size)}
    return {"point": point, "ci_lo": float(np.quantile(vals, alpha / 2)), "ci_hi": float(np.quantile(vals, 1 - alpha / 2)),
            "n_wells": k, "n_boot": int(vals.size)}


def paired_difference(kind: str, metric: str, a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 500, seed: int = 0,
                      alpha: float = 0.05) -> dict:
    """metric(a) - metric(b) on the rows both predicted (joined on sample_id), well-bootstrap CI."""
    m = a[["sample_id", "well", "y_true", "y_pred"]].merge(b[["sample_id", "y_pred"]], on="sample_id",
                                                           suffixes=("_a", "_b"))
    f = _metric_fn(kind, metric)
    wells = sorted(m["well"].unique())
    blocks = [tuple(m.loc[m["well"] == w, c].to_numpy(float) for c in ("y_true", "y_pred_a", "y_pred_b"))
              for w in wells]
    y, pa, pb = (m["y_true"].to_numpy(float), m["y_pred_a"].to_numpy(float), m["y_pred_b"].to_numpy(float))
    point = f(y, pa) - f(y, pb)
    out = {"n_rows": int(len(m)), "n_wells": len(wells), "metric_a": f(y, pa), "metric_b": f(y, pb),
           "difference": point, "ci_lo": np.nan, "ci_hi": np.nan, "wells_a_better": np.nan}
    if len(wells) < 3:
        return out
    better = 0
    for bl in blocks:
        da, db = f(bl[0], bl[1]), f(bl[0], bl[2])
        if np.isfinite(da) and np.isfinite(db):
            better += int((da > db) if metric in ("roc_auc", "pr_auc", "spearman", "r2") else (da < db))
    out["wells_a_better"] = better
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(wells), len(wells))
        yy = np.concatenate([blocks[i][0] for i in pick])
        aa = np.concatenate([blocks[i][1] for i in pick])
        bb = np.concatenate([blocks[i][2] for i in pick])
        v = f(yy, aa) - f(yy, bb)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) >= 20:
        out["ci_lo"], out["ci_hi"] = float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))
    return out


def verdict(metric: str, diff: dict) -> str:
    """Plain words for a paired difference (higher-is-better metrics vs error metrics)."""
    lo, hi = diff.get("ci_lo", np.nan), diff.get("ci_hi", np.nan)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "not testable (too few wells)"
    higher_better = metric in ("roc_auc", "pr_auc", "spearman", "r2")
    if lo > 0:
        return "A better" if higher_better else "A worse"
    if hi < 0:
        return "A worse" if higher_better else "A better"
    return "no detectable difference"


# =============================================================================== does seismic add to depth?
def _rank(a):
    return stats.rankdata(np.asarray(a, float))


def partial_spearman(x, y, covariates) -> float:
    """Spearman partial correlation of x and y given covariates (all rank-transformed).

    x and y ranks are each regressed (OLS) on the covariate ranks; the result is the Pearson
    correlation of the two residual vectors.  NaN if either residual is constant.
    """
    rx, ry = _rank(x), _rank(y)
    Z = np.column_stack([np.ones(rx.size)] + [_rank(c) for c in covariates])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if ex.std() < 1e-12 or ey.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ex, ey)[0, 1])


def beyond_depth(seis: pd.DataFrame, depth: pd.DataFrame, kind: str, n_boot: int = 500, seed: int = 0) -> dict:
    """Does a seismic model know something about the target that burial depth does not?

    Within each held-out well: partial Spearman correlation between the model's prediction and
    the truth, controlling for the TWT-only depth-trend prediction AND TWT itself (both ranked).
    Both the prediction and the truth are residualised, so a model is not penalised for ALSO
    reproducing the depth trend (a semi-partial "prediction vs depth residual" correlation would
    be).  Per-well values are averaged; the interval resamples whole wells.  > 0 with an interval
    above zero = information beyond depth.  `kind` is accepted for symmetry (ranks handle binary
    truth via mid-ranks).
    """
    cols = ["sample_id", "well", "y_true", "y_pred"] + (["desired_twt_ms"] if "desired_twt_ms" in seis else [])
    m = seis[cols].merge(depth[["sample_id", "y_pred"]], on="sample_id", suffixes=("", "_depth"))
    per = []
    for w, g in m.groupby("well"):
        if len(g) < 15 or np.ptp(g["y_pred"]) <= 0 or g["y_true"].nunique() < 2:
            continue
        cov = [g["y_pred_depth"]] + ([g["desired_twt_ms"]] if "desired_twt_ms" in g else [])
        cov = [c for c in cov if np.ptp(np.asarray(c, float)) > 0]
        r = partial_spearman(g["y_pred"], g["y_true"], cov)
        if np.isfinite(r):
            per.append((w, r, len(g)))
    out = {"n_wells": len(per), "mean_within_well_partial_rho": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
           "wells_positive": int(sum(r > 0 for _, r, _ in per))}
    if not per:
        return out
    r = np.array([x[1] for x in per])
    out["mean_within_well_partial_rho"] = float(r.mean())
    if len(per) >= 3:
        rng = np.random.default_rng(seed)
        boots = [r[rng.integers(0, r.size, r.size)].mean() for _ in range(n_boot)]
        out["ci_lo"], out["ci_hi"] = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    return out


# =============================================================================== curves, calibration, intervals
def roc_curve(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.concatenate([[0], np.cumsum(ys == 1)])
    fp = np.concatenate([[0], np.cumsum(ys == 0)])
    P, N = max(1, (y == 1).sum()), max(1, (y == 0).sum())
    return fp / N, tp / P


def pr_curve(y, p):
    y, p = np.asarray(y, int), np.asarray(p, float)
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    k = np.arange(1, ys.size + 1)
    return tp / max(1, (y == 1).sum()), tp / k


def calibration_bins(d: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Reliability table: quantile bins of predicted probability vs observed frequency."""
    p, y = d["y_pred"].to_numpy(float), d["y_true"].to_numpy(float)
    if p.size < n_bins:
        return pd.DataFrame()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    b = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    rows = []
    for k in range(len(edges) - 1):
        m = b == k
        if m.any():
            rows.append({"bin": k, "p_lo": edges[k], "p_hi": edges[k + 1], "n": int(m.sum()),
                         "mean_predicted": float(p[m].mean()), "observed_frequency": float(y[m].mean())})
    return pd.DataFrame(rows)


def conformal(d: pd.DataFrame, levels=(0.8, 0.9)) -> dict:
    """Split-conformal half-widths from held-out absolute residuals, plus leave-one-well-out coverage.

    The half-width q(level) is the ceil((n+1) level)-th smallest |y - y_hat|.  Coverage is
    checked honestly: for each well, q is computed WITHOUT that well and applied to it.
    """
    res = np.abs(d["y_true"].to_numpy(float) - d["y_pred"].to_numpy(float))
    wells = d["well"].to_numpy()
    out = {}
    for lv in levels:
        out[f"halfwidth_{int(lv * 100)}"] = _cq(res, lv)
        cov = []
        for w in np.unique(wells):
            q = _cq(res[wells != w], lv)
            r = res[wells == w]
            if r.size and np.isfinite(q):
                cov.append(float((r <= q).mean()))
        out[f"lowo_coverage_{int(lv * 100)}_mean"] = float(np.mean(cov)) if cov else np.nan
        out[f"lowo_coverage_{int(lv * 100)}_min"] = float(np.min(cov)) if cov else np.nan
    return out


def _cq(r: np.ndarray, level: float) -> float:
    r = np.sort(r[np.isfinite(r)])
    if r.size == 0:
        return np.nan
    k = int(np.ceil((r.size + 1) * level)) - 1
    return float(r[min(max(k, 0), r.size - 1)])


def interval_errors(kind: str, d: pd.DataFrame, bin_ms: float = 200.0) -> pd.DataFrame:
    """Errors by TWT interval (where in the section does the model fail?)."""
    t = d["desired_twt_ms"].to_numpy(float)
    lo = np.floor(np.nanmin(t) / bin_ms) * bin_ms
    b = np.floor((t - lo) / bin_ms).astype(int)
    rows = []
    for k in np.unique(b):
        g = d[b == k]
        r = {"twt_from_ms": lo + k * bin_ms, "twt_to_ms": lo + (k + 1) * bin_ms, "n": int(len(g)),
             "n_wells": int(g["well"].nunique())}
        if kind == "binary":
            r["error_rate"] = float((g["y_pred_class"] != g["y_true"]).mean()) if "y_pred_class" in g else np.nan
            r["prevalence"] = float(g["y_true"].mean())
            r["roc_auc"] = _auc(g["y_true"].to_numpy(int), g["y_pred"].to_numpy(float)) if len(g) >= 10 else np.nan
        else:
            e = g["y_pred"].to_numpy(float) - g["y_true"].to_numpy(float)
            r["rmse"] = float(np.sqrt((e ** 2).mean()))
            r["bias"] = float(e.mean())
        rows.append(r)
    return pd.DataFrame(rows)
