"""Fold-local preprocessing: missing indicators, imputation, clipping and robust scaling.

The golden rule
---------------
Every number in a transform is computed from the rows of the TRAINING wells of one fold
(:func:`fit_transform_params`).  The frozen parameters are then applied, unchanged, to any
rows - validation wells, test wells, the reserved well, or a whole seismic volume
(:func:`apply_transform`).  If the test wells helped to choose a median or a scale, the
test score would be optimistic; that is the leak this module prevents.

What one feature goes through (``apply_transform``)
---------------------------------------------------
1. indicator   ``<feature>__isna`` = 1 where the raw value is NaN (only for features that
               had >= ``indicator_min_missing`` NaN in the training rows).
2. impute      NaN -> the training median (raw units).
3. clip        to the training quantiles [q_lo, q_hi] (winsorising: one spike cannot
               dominate a linear model or a neural network).
4. scale       (x - centre) / scale with centre = training median and scale = training IQR
               (fallbacks: 1.4826 x MAD, then std, then 1 - recorded in ``scale_source``).

Two scaling modes (the brief asks to compare them for AVO sensitivity)
--------------------------------------------------------------------
* ``independent`` - every column has its own median / IQR.  near__rms_long and
  far__rms_long both end up with spread ~1, so the far/near amplitude ratio is lost.
* ``common``      - columns that measure the same amplitude quantity on different stacks
  (``full__amp_c``, ``near__amp_c`` ...) share ONE centre and ONE scale, pooled over the
  stacks.  A far stack that is 20 % weaker stays 20 % weaker after scaling.
Contrasts computed before scaling in 04 (normalised differences, slopes vs stack order)
carry the ratios in both modes.

Parameters are plain JSON (no pickles): readable by a judge, loadable without this code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Stack attributes measured in amplitude units: candidates for a shared (common) scale.
AMPLITUDE_ATTRS = ("amp_c", "rms_short", "rms_long", "maxabs", "p2p", "std_t", "env_c", "dadt_c", "spatial_std_c",
                   "grad_xy_c")


def common_groups(features: list[str], stacks: list[str]) -> dict[str, str]:
    """feature -> common-scale group ('amp:<attr>' for amplitude attributes of the stacks, else itself)."""
    out = {}
    for f in features:
        src, _, attr = f.partition("__")
        out[f] = f"amp:{attr}" if src in stacks and attr in AMPLITUDE_ATTRS else f
    return out


def _robust_scale(v: np.ndarray) -> tuple[float, float, str]:
    """(centre, scale, source) of finite values: median and IQR with documented fallbacks."""
    if v.size == 0:
        return 0.0, 1.0, "no_data"
    med = float(np.median(v))
    q25, q75 = np.quantile(v, [0.25, 0.75])
    if q75 - q25 > 0:
        return med, float(q75 - q25), "iqr"
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if mad > 0:
        return med, mad, "mad"
    sd = float(np.std(v))
    if sd > 0:
        return med, sd, "std"
    return med, 1.0, "constant"


def fit_transform_params(train: pd.DataFrame, features: list[str], groups: dict[str, str] | None = None,
                         clip_quantiles: tuple[float, float] = (0.005, 0.995),
                         indicator_min_missing: float = 0.01) -> dict[str, dict]:
    """Fit per-feature parameters on TRAINING rows only.  Returns {feature: params}."""
    groups = groups or {f: f for f in features}
    lo_q, hi_q = float(clip_quantiles[0]), float(clip_quantiles[1])
    params: dict[str, dict] = {}
    for f in features:
        x = train[f].to_numpy(float) if f in train else np.full(len(train), np.nan)
        v = x[np.isfinite(x)]
        miss = float(1.0 - v.size / max(1, x.size))
        centre, scale, source = _robust_scale(v)
        q_lo, q_hi = (np.quantile(v, [lo_q, hi_q]) if v.size else (np.nan, np.nan))
        params[f] = {"n_fit_rows": int(x.size), "n_finite": int(v.size), "missing_fraction": round(miss, 6),
                     "impute_value": centre if v.size else 0.0, "clip_lo": float(q_lo), "clip_hi": float(q_hi),
                     "centre": centre, "scale": scale, "scale_source": source,
                     "indicator": bool(miss >= indicator_min_missing and v.size > 0),
                     "common_group": groups.get(f, f)}
    # ---- common scale: pool the finite, clipped values of every member of a group
    members: dict[str, list[str]] = {}
    for f in features:
        members.setdefault(params[f]["common_group"], []).append(f)
    for g, fs in members.items():
        pooled = []
        for f in fs:
            x = train[f].to_numpy(float) if f in train else np.zeros(0)
            x = x[np.isfinite(x)]
            pooled.append(np.clip(x, params[f]["clip_lo"], params[f]["clip_hi"]) if x.size else x)
        allv = np.concatenate(pooled) if pooled else np.zeros(0)
        c, s, src = _robust_scale(allv)
        for f in fs:
            params[f].update({"common_centre": c, "common_scale": s, "common_scale_source": src,
                              "common_members": len(fs)})
    return params


def apply_transform(df: pd.DataFrame, params: dict[str, dict], features: list[str] | None = None,
                    mode: str = "independent", add_indicators: bool = True) -> pd.DataFrame:
    """Apply frozen parameters to ANY rows.  Returns scaled features (+ ``__isna`` indicators).

    The output never contains NaN: missing values are imputed with the training median
    (and flagged by the indicator column when the feature has one).
    """
    if mode not in ("independent", "common"):
        raise ValueError("mode must be 'independent' or 'common'")
    features = list(features if features is not None else params)
    cols, ind = {}, {}
    for f in features:
        p = params[f]
        x = df[f].to_numpy(float) if f in df else np.full(len(df), np.nan)
        nan = ~np.isfinite(x)
        if add_indicators and p.get("indicator"):
            ind[f"{f}__isna"] = nan.astype(np.float32)
        x = np.where(nan, p["impute_value"], x)
        if np.isfinite(p["clip_lo"]) and np.isfinite(p["clip_hi"]):
            x = np.clip(x, p["clip_lo"], p["clip_hi"])
        c, s = (p["centre"], p["scale"]) if mode == "independent" else (p["common_centre"], p["common_scale"])
        cols[f] = (x - c) / (s if s else 1.0)
    out = pd.DataFrame(cols, index=df.index)
    if ind:
        out = pd.concat([out, pd.DataFrame(ind, index=df.index)], axis=1)
    return out


def params_table(params: dict[str, dict], **keys) -> pd.DataFrame:
    """Long table of parameters (one row per feature) with extra key columns (e.g. fold)."""
    rows = [{**keys, "feature": f, **p} for f, p in params.items()]
    return pd.DataFrame(rows)
