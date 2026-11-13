"""Metrics and physical bounds shared by stages 07 (inner CV) and 08 (held-out wells).

Classification (binary: 1 = sand / hydrocarbon-potential sand)
    roc_auc        threshold-free ranking quality (0.5 = chance)      <- used to tune
    pr_auc         average precision (chance = prevalence)
    balanced_acc   mean of sensitivity and specificity at a threshold
    f1_macro       F1 averaged over both classes at a threshold
    brier          mean squared error of the probability (calibration + sharpness)
    log_loss       proper scoring rule (probabilities clipped to [1e-6, 1 - 1e-6])

Regression (VSH, PHIT, SWT as fractions)
    rmse  <- used to tune,  mae,  r2,  spearman

Physical bounds: predictions outside [0, 1] (VSH, SWT) or [0, 0.476] (PHIT, cubic packing)
are impossible.  :func:`apply_bounds` clips them and reports how many were clipped, so a
model that needs clipping is visible instead of silently "fixed".
"""

from __future__ import annotations

import numpy as np
from scipy import stats

PHI_MAX = 0.476
BOUNDS = {"vsh": (0.0, 1.0), "phit": (0.0, PHI_MAX), "swt": (0.0, 1.0)}
PRIMARY = {"binary": ("roc_auc", "max"), "regression": ("rmse", "min")}


def apply_bounds(task: str, pred: np.ndarray) -> tuple[np.ndarray, int]:
    """Clip a regression prediction to its physical range; returns (clipped, n_clipped)."""
    p = np.asarray(pred, float)
    if task not in BOUNDS:
        return p, 0
    lo, hi = BOUNDS[task]
    n = int(((p < lo) | (p > hi)).sum())
    return np.clip(p, lo, hi), n


def best_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Probability threshold that maximises balanced accuracy (ties -> closest to 0.5)."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size == 0 or y.min() == y.max():
        return 0.5
    cand = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 99)))
    best, best_t = -1.0, 0.5
    for t in cand:
        ba = _balanced_accuracy(y, (p >= t).astype(int))
        if ba > best + 1e-12 or (abs(ba - best) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best, best_t = ba, float(t)
    return best_t


def _balanced_accuracy(y, yhat):
    pos, neg = y == 1, y == 0
    sens = (yhat[pos] == 1).mean() if pos.any() else np.nan
    spec = (yhat[neg] == 0).mean() if neg.any() else np.nan
    return float(np.nanmean([sens, spec]))


def _auc(y, p):
    pos, neg = y == 1, y == 0
    n1, n0 = int(pos.sum()), int(neg.sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = stats.rankdata(p)
    return float((r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _average_precision(y, p):
    if (y == 1).sum() == 0:
        return np.nan
    order = np.argsort(-p, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys == 1)
    precision = tp / np.arange(1, ys.size + 1)
    return float((precision * (ys == 1)).sum() / (ys == 1).sum())


def _f1(y, yhat, cls):
    tp = int(((yhat == cls) & (y == cls)).sum())
    fp = int(((yhat == cls) & (y != cls)).sum())
    fn = int(((yhat != cls) & (y == cls)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    """All binary metrics for labels ``y`` (0/1) and probabilities ``p`` of class 1."""
    y, p = np.asarray(y, int), np.asarray(p, float)
    ok = np.isfinite(p)
    y, p = y[ok], p[ok]
    out = {"n": int(y.size), "n_pos": int((y == 1).sum()), "prevalence": float(y.mean()) if y.size else np.nan}
    if y.size == 0:
        return out | {k: np.nan for k in ("roc_auc", "pr_auc", "balanced_acc", "f1_macro", "brier", "log_loss")}
    yhat = (p >= threshold).astype(int)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    out.update({"roc_auc": _auc(y, p), "pr_auc": _average_precision(y, p),
                "balanced_acc": _balanced_accuracy(y, yhat),
                "f1_macro": float(np.nanmean([_f1(y, yhat, 1), _f1(y, yhat, 0)])),
                "brier": float(((p - y) ** 2).mean()),
                "log_loss": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()), "threshold": float(threshold)})
    return out


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if y.size < 2:
        return {"n": int(y.size), "rmse": np.nan, "mae": np.nan, "r2": np.nan, "spearman": np.nan, "bias": np.nan}
    res = p - y
    ss = float(((y - y.mean()) ** 2).sum())
    sp = float(stats.spearmanr(y, p).statistic) if np.ptp(p) > 0 and np.ptp(y) > 0 else np.nan
    return {"n": int(y.size), "rmse": float(np.sqrt((res ** 2).mean())), "mae": float(np.abs(res).mean()),
            "r2": float(1 - (res ** 2).sum() / ss) if ss > 0 else np.nan, "spearman": sp, "bias": float(res.mean())}


def task_metrics(kind: str, y, p, threshold: float = 0.5) -> dict:
    return binary_metrics(y, p, threshold) if kind == "binary" else regression_metrics(y, p)


def is_better(kind: str, a: float, b: float) -> bool:
    """True if primary-metric value ``a`` beats ``b`` (NaN never wins)."""
    if not np.isfinite(a):
        return False
    if not np.isfinite(b):
        return True
    return a > b if PRIMARY[kind][1] == "max" else a < b
