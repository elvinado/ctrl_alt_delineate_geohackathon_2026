"""Feature statistics that respect the unit of independence: the WELL.

Why not a t-test on all rows?
-----------------------------
Neighbouring 4-ms rows of one well share the wavelet, the tie error and the geology, so
3,000 rows from 9 wells are not 3,000 independent observations.  A row-level p-value is
therefore far too small and a "significant" feature may only separate one well from
another.  Here every statistic is built in two levels:

1. **Within each well** an effect size that ignores the amplitude level of that well:
   * classification (sand vs shale, water vs HC-potential): rank-biserial correlation
     r = 2 * AUC - 1 (Mann-Whitney), from -1 to +1; + means higher values in class 1.
   * regression (VSH, PHIT, SWT): Spearman rho.
2. **Across wells** a random-effects meta-analysis (DerSimonian-Laird between-well
   variance tau^2, Hartung-Knapp-Sidik-Jonkman t-interval with k-1 degrees of freedom,
   with the conservative "max" rule of Knapp & Hartung).  The HKSJ standard error is
   estimated from the observed spread of the well effects, so autocorrelation that
   inflates every well's own precision by a similar factor largely cancels out.

Multiple testing: Benjamini-Hochberg across the predictors tested in one fold / task / group.
A simple sign test across wells (how many wells agree on the direction) is reported too.

References: DerSimonian & Laird 1986; Hartung & Knapp 2001; IntHout et al. 2014 (HKSJ);
Hanley & McNeil 1982 (AUC variance); Bonett & Wright 2000 (Spearman variance);
Benjamini & Hochberg 1995; Roberts et al. 2017 (grouped validation).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform


# =============================================================================== within-well effects
def _avg_ranks(x: np.ndarray) -> np.ndarray:
    return stats.rankdata(x, method="average")


def rank_biserial(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int, int]:
    """(r = 2*AUC-1, variance of r, n1, n0) for a binary y (1 / 0); x finite."""
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return np.nan, np.nan, n1, n0
    r = _avg_ranks(x)
    auc = (r[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    a = float(auc)
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    var_auc = (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n0 - 1) * (q2 - a * a)) / (n1 * n0)
    null_var = (n1 + n0 + 1) / (12.0 * n1 * n0)
    var_auc = max(var_auc, 0.25 * null_var)          # floor: a perfectly separating well must not get infinite weight
    return 2 * a - 1, 4 * var_auc, n1, n0


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """(rho, variance of Fisher z (Bonett-Wright), n); x, y finite."""
    n = x.size
    if n < 4 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return np.nan, np.nan, n
    rx, ry = _avg_ranks(x), _avg_ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    return rho, (1 + rho * rho / 2) / (n - 3), n


def per_well_effects(df: pd.DataFrame, features: list[str], target: str, mask: str, kind: str,
                     min_class_rows: int = 5, min_rows: int = 15) -> pd.DataFrame:
    """One row per (well, feature): within-well effect, its variance, counts, eligibility.

    ``kind`` is 'binary' (target 1 / 0) or 'regression'.  Only rows with ``mask`` True and
    a finite feature value are used.  A well is *eligible* for a feature when it has
    >= ``min_class_rows`` rows of each class (binary) or >= ``min_rows`` rows (regression)
    and the feature varies in that well.
    """
    rows = []
    d = df[df[mask].astype(bool)]
    for w, g in d.groupby("well", sort=True):
        y_all = g[target].to_numpy(float)
        for f in features:
            x = g[f].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y_all)
            x, y = x[ok], y_all[ok]
            r = {"well": w, "feature": f, "n": int(ok.sum())}
            if kind == "binary":
                n1, n0 = int((y == 1).sum()), int((y == 0).sum())
                r.update({"n_class1": n1, "n_class0": n0})
                elig = n1 >= min_class_rows and n0 >= min_class_rows and x.size and np.ptp(x) > 0
                eff, var = (rank_biserial(x, y)[:2] if elig else (np.nan, np.nan))
                r.update({"effect": eff, "var": var, "statistic": "rank_biserial", "eligible": bool(elig)})
            else:
                elig = x.size >= min_rows and np.ptp(x) > 0 and np.ptp(y) > 0
                eff, var, _ = spearman(x, y) if elig else (np.nan, np.nan, x.size)
                r.update({"n_class1": np.nan, "n_class0": np.nan, "effect": eff, "var": var,
                          "statistic": "spearman", "eligible": bool(elig and np.isfinite(eff))})
            rows.append(r)
    return pd.DataFrame(rows)


def pooled_effect(df: pd.DataFrame, feature: str, target: str, mask: str, kind: str) -> float:
    """Row-level effect over all given rows (DESCRIPTIVE only: rows are not independent)."""
    d = df[df[mask].astype(bool)]
    return pooled_effect_xy(d[feature].to_numpy(float), d[target].to_numpy(float), kind)


def pooled_effect_xy(x: np.ndarray, y: np.ndarray, kind: str) -> float:
    """:func:`pooled_effect` on arrays (rows already masked)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10:
        return np.nan
    if kind == "binary":
        return rank_biserial(x[ok], y[ok])[0]
    return spearman(x[ok], y[ok])[0]


# =============================================================================== across wells
def random_effects_hksj(y: np.ndarray, v: np.ndarray, alpha: float = 0.05) -> dict:
    """Random-effects mean of per-well effects ``y`` with variances ``v`` (DL tau^2 + HKSJ)."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    ok = np.isfinite(y) & np.isfinite(v) & (v > 0)
    y, v = y[ok], v[ok]
    k = int(y.size)
    out = {"k": k, "mean": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "t": np.nan, "df": k - 1,
           "p": np.nan, "tau2": np.nan, "i2": np.nan}
    if k < 2:
        return out
    w = 1.0 / v
    mu_fe = float((w * y).sum() / w.sum())
    q = float((w * (y - mu_fe) ** 2).sum())
    c = float(w.sum() - (w ** 2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    ws = 1.0 / (v + tau2)
    mu = float((ws * y).sum() / ws.sum())
    q_hk = float((ws * (y - mu) ** 2).sum() / (k - 1))
    se = float(np.sqrt(max(q_hk, 1.0) / ws.sum()))         # Knapp-Hartung "max" rule (never narrower than RE)
    tcrit = float(stats.t.ppf(1 - alpha / 2, k - 1))
    t = mu / se if se > 0 else np.nan
    out.update({"mean": mu, "se": se, "ci_lo": mu - tcrit * se, "ci_hi": mu + tcrit * se, "t": t,
                "p": float(2 * stats.t.sf(abs(t), k - 1)) if np.isfinite(t) else np.nan, "tau2": tau2,
                "i2": max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0})
    return out


def sign_test(effects: np.ndarray) -> tuple[float, float]:
    """(share of wells agreeing with the majority direction, two-sided binomial p) - zeros dropped."""
    e = np.asarray(effects, float)
    e = e[np.isfinite(e) & (e != 0)]
    if e.size == 0:
        return np.nan, np.nan
    n_pos = int((e > 0).sum())
    agree = max(n_pos, e.size - n_pos) / e.size
    return agree, float(stats.binomtest(n_pos, e.size, 0.5).pvalue)


def aggregate_effects(effects: pd.DataFrame, wells: list[str], kind: str, min_wells: int) -> pd.DataFrame:
    """Per feature: random-effects summary over the ELIGIBLE rows of the given wells only."""
    e = effects[effects["well"].isin(wells) & effects["eligible"]]
    rows = []
    for f, g in e.groupby("feature", sort=False):
        y, v = g["effect"].to_numpy(float), g["var"].to_numpy(float)
        if kind == "regression":                     # meta-analyse Fisher z, report on the rho scale
            z = np.arctanh(np.clip(y, -0.999, 0.999))
            re = random_effects_hksj(z, v)
            for key in ("mean", "ci_lo", "ci_hi"):
                re[key] = float(np.tanh(re[key])) if np.isfinite(re[key]) else np.nan
        else:
            re = random_effects_hksj(y, v)
            for key in ("mean", "ci_lo", "ci_hi"):
                re[key] = float(np.clip(re[key], -1, 1)) if np.isfinite(re[key]) else np.nan
        agree, p_sign = sign_test(y)
        mean_sign = np.sign(re["mean"]) if np.isfinite(re["mean"]) else 0
        consistent = float((np.sign(y) == mean_sign).mean()) if y.size and mean_sign else np.nan
        rows.append({"feature": f, "k_wells": int(g["well"].nunique()), "n_rows": int(g["n"].sum()),
                     "wells": ";".join(sorted(g["well"])),
                     "median_well_effect": float(np.median(y)), "min_well_effect": float(np.min(y)),
                     "max_well_effect": float(np.max(y)), "sign_consistency": consistent,
                     "sign_test_p": p_sign, "re_mean_effect": re["mean"], "re_ci_lo": re["ci_lo"],
                     "re_ci_hi": re["ci_hi"], "re_t": re["t"], "re_df": re["df"], "p_value": re["p"],
                     "tau2": re["tau2"], "i2": re["i2"], "enough_wells": bool(g["well"].nunique() >= min_wells)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out.loc[~out["enough_wells"], ["p_value"]] = np.nan
    return out


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH q-values (NaN p-values stay NaN and do not count as tests)."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# =============================================================================== label-free screening
def screen_features(train: pd.DataFrame, features: list[str], max_missing: float,
                    near_constant_fraction: float) -> pd.DataFrame:
    """Label-free screening on training rows: missingness, near-constancy, between-well share, TWT trend."""
    rows = []
    wells = train["well"].to_numpy()
    twt = train["desired_twt_ms"].to_numpy(float) if "desired_twt_ms" in train else None
    for f in features:
        x = train[f].to_numpy(float) if f in train else np.full(len(train), np.nan)
        ok = np.isfinite(x)
        miss = float(1 - ok.mean()) if x.size else 1.0
        v = x[ok]
        if v.size:
            _, counts = np.unique(v, return_counts=True)
            top = float(counts.max() / v.size)
        else:
            top = 1.0
        share = np.nan
        rho_t = np.nan
        if v.size >= 10:
            r = _avg_ranks(v)
            tot = float(((r - r.mean()) ** 2).sum())
            if tot > 0:
                within = 0.0
                for w in np.unique(wells[ok]):
                    rw = r[wells[ok] == w]
                    within += float(((rw - rw.mean()) ** 2).sum())
                share = 1.0 - within / tot
            if twt is not None and np.ptp(v) > 0:
                rho_t = spearman(v, twt[ok])[0]
        reason = ("too_many_missing" if miss > max_missing else
                  "near_constant" if top >= near_constant_fraction else "")
        rows.append({"feature": f, "train_missing_fraction": miss, "top_value_fraction": top,
                     "between_well_rank_variance_share": share, "spearman_with_twt": rho_t,
                     "kept_screen": reason == "", "screen_reason": reason})
    return pd.DataFrame(rows)


def rank_correlation(train: pd.DataFrame, features: list[str], min_overlap: int = 30) -> pd.DataFrame:
    """Spearman-type matrix: ranks per column (NaN ignored), then pairwise-complete Pearson."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return train[features].rank().corr(method="pearson", min_periods=min_overlap)


def redundancy_clusters(corr: pd.DataFrame, coverage: dict[str, float], threshold: float) -> pd.DataFrame:
    """Complete-linkage clusters of |rho| >= threshold (every pair inside a cluster) + one representative.

    The representative is chosen WITHOUT labels: highest training coverage, then the most
    central member (largest mean |rho| to the others), then the name.
    """
    feats = list(corr.columns)
    if len(feats) == 1:
        return pd.DataFrame([{"feature": feats[0], "cluster_id": 1, "cluster_size": 1, "representative": True,
                              "represented_by": feats[0], "order": 0}])
    a = np.array(corr.abs().fillna(0.0).to_numpy(), dtype=float, copy=True)   # writable (pandas copy-on-write)
    np.fill_diagonal(a, 1.0)
    dist = np.clip(1.0 - (a + a.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)
    Z = linkage(squareform(dist, checks=False), method="complete")
    lab = fcluster(Z, t=1.0 - threshold, criterion="distance")
    order = {f: i for i, f in enumerate(np.array(feats)[leaves_list(Z)])}
    rows = []
    for cid in np.unique(lab):
        members = [f for f, l in zip(feats, lab) if l == cid]
        idx = [feats.index(m) for m in members]
        cent = {m: (a[feats.index(m), idx].sum() - 1.0) / max(1, len(idx) - 1) for m in members}
        rep = sorted(members, key=lambda m: (-coverage.get(m, 0.0), -cent[m], m))[0]
        for m in members:
            rows.append({"feature": m, "cluster_id": int(cid), "cluster_size": len(members),
                         "representative": m == rep, "represented_by": rep, "order": order[m]})
    return pd.DataFrame(rows).sort_values("order").reset_index(drop=True)


# =============================================================================== selection rule
def apply_selection_rule(stats_df: pd.DataFrame, candidates: list[str], fdr_q: float, min_abs_effect: float,
                         min_sign_consistency: float, max_selected: int) -> pd.DataFrame:
    """BH over the candidates, then the documented thresholds.  Adds q_value, selected, rank, decision."""
    s = stats_df.copy()
    s["candidate"] = s["feature"].isin(candidates)
    s["q_value"] = np.nan
    cand = s["candidate"] & s["p_value"].notna()
    s.loc[cand, "q_value"] = benjamini_hochberg(s.loc[cand, "p_value"].to_numpy())
    passed = (cand & (s["q_value"] <= fdr_q) & (s["re_mean_effect"].abs() >= min_abs_effect)
              & (s["sign_consistency"] >= min_sign_consistency))
    s["selected"] = False
    order = s[passed].assign(_a=lambda d: -d["re_mean_effect"].abs()).sort_values(["_a", "feature"]).index
    keep = list(order[:max_selected])
    s.loc[keep, "selected"] = True
    s["rank"] = np.nan
    s.loc[keep, "rank"] = np.arange(1, len(keep) + 1)

    def why(r):
        if not r["candidate"]:
            return "not a candidate (screened out or redundant)"
        if not np.isfinite(r["p_value"]):
            return "too few eligible training wells"
        if r["selected"]:
            return "selected"
        bits = []
        if not (r["q_value"] <= fdr_q):
            bits.append(f"q {r['q_value']:.3g} > {fdr_q}")
        if not (abs(r["re_mean_effect"]) >= min_abs_effect):
            bits.append(f"|effect| {abs(r['re_mean_effect']):.2f} < {min_abs_effect}")
        if not (r["sign_consistency"] >= min_sign_consistency):
            bits.append(f"sign consistency {r['sign_consistency']:.2f} < {min_sign_consistency}")
        return "; ".join(bits) or f"passed but beyond max_selected={max_selected}"

    s["decision"] = s.apply(why, axis=1) if len(s) else []
    return s
