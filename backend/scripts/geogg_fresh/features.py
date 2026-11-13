"""Seismic attributes and AVO helpers for stage 04 (pure numpy/scipy, no file access).

Input for every function is the patch array of stage 03, one channel at a time:
``X[N, n_il, n_xl, n_t]`` (float, NaN where no data).  The centre trace is
``X[:, ci, cj, :]`` and the centre sample ``X[:, ci, cj, ck]``.

Attribute definitions (dt in ms; amplitudes in the volume's own, uncalibrated units)
-------------------------------------------------------------------------------
amp_c            centre amplitude, polarity preserved
rms_short        sqrt(mean(a^2)) over the centre trace, +/-2 samples (5 samples = 16 ms span)
rms_long         same over the whole patch length (21 samples = 80 ms span)
maxabs           max |a| over the centre trace window
p2p              max(a) - min(a) over the centre trace window (peak-to-peak)
std_t            temporal standard deviation of the centre trace window
env_c            envelope |a + i H(a)| at the centre (Hilbert on the reflect-padded window:
                 a short-window approximation, documented as such)
cosphase_c       cos(instantaneous phase) at the centre = a / envelope (bounded, polarity aware)
dadt_c           time derivative at the centre, central difference, amplitude per ms
spatial_std_c    standard deviation over the n_il x n_xl traces at the centre sample
grad_xy_c        |lateral gradient| at the centre: hypot of IL and XL central differences (per bin)
semblance        sum_t (sum_traces a)^2 / (N_traces * sum_t sum_traces a^2) over +/-2 samples:
                 1 = laterally identical waveforms, ~1/N = incoherent
spec_centroid_hz power-weighted mean frequency of the Hann-tapered centre-trace window (DC excluded);
                 coarse: 21 samples at 4 ms resolve ~12 Hz steps

AVO (Shuey two-term)  R(theta) = A + B sin^2(theta), fitted only when effective angles are
documented; otherwise amplitude-vs-stack-ORDER proxies are reported and labelled as such.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.signal import hilbert

STACK_ATTRS = ["amp_c", "rms_short", "rms_long", "maxabs", "p2p", "std_t", "env_c", "cosphase_c", "dadt_c",
               "spatial_std_c", "grad_xy_c", "semblance", "spec_centroid_hz"]

ATTR_DOC = {
    "amp_c": ("centre amplitude", "amplitude units", "patch[ci, cj, ck]"),
    "rms_short": ("RMS over 5 samples (16 ms span)", "amplitude units", "centre trace, ck-2..ck+2"),
    "rms_long": ("RMS over the full window (80 ms span)", "amplitude units", "centre trace, all samples"),
    "maxabs": ("max |amplitude|", "amplitude units", "centre trace, all samples"),
    "p2p": ("peak-to-peak", "amplitude units", "centre trace, all samples"),
    "std_t": ("temporal standard deviation", "amplitude units", "centre trace, all samples"),
    "env_c": ("envelope (short-window Hilbert, reflect padded)", "amplitude units", "centre trace"),
    "cosphase_c": ("cos(instantaneous phase)", "dimensionless [-1, 1]", "centre trace"),
    "dadt_c": ("time derivative", "amplitude units per ms", "centre trace, ck-1 and ck+1"),
    "spatial_std_c": ("lateral standard deviation", "amplitude units", "all traces at ck"),
    "grad_xy_c": ("lateral gradient magnitude", "amplitude units per bin", "IL and XL neighbours at ck"),
    "semblance": ("semblance (coherence)", "dimensionless [0, 1]", "all traces, ck-2..ck+2"),
    "spec_centroid_hz": ("spectral centroid (coarse)", "Hz", "centre trace, Hann taper"),
}


def _centre(shape):
    return shape[1] // 2, shape[2] // 2, shape[3] // 2


# ----------------------------------------------------------------------------- exact zeros
def zero_run_lengths(zero: np.ndarray) -> np.ndarray:
    """For every True sample of ``zero`` (boolean, time on the LAST axis), the length of the run
    of consecutive True samples it belongs to; 0 where ``zero`` is False.

    Example along one trace: [F, T, T, F, T] -> [0, 2, 2, 0, 1].
    """
    z = np.asarray(zero, dtype=bool)
    flat = z.reshape(-1, z.shape[-1]) if z.ndim else z.reshape(1, 1)
    out = np.zeros(flat.shape, dtype=np.int32)
    if flat.size == 0 or not flat.any():
        return out.reshape(z.shape)
    prev = np.zeros_like(flat)
    prev[:, 1:] = flat[:, :-1]
    starts = flat & ~prev                                       # a run starts here (never crosses a trace)
    run_id = np.cumsum(starts.ravel()).reshape(flat.shape)      # 1, 2, 3 ... per run, row-major
    lengths = np.bincount(run_id[flat])                         # samples per run id
    out[flat] = lengths[run_id[flat]]
    return out.reshape(z.shape)


def stack_zero_missing(X: np.ndarray, min_run: int) -> tuple[np.ndarray, np.ndarray]:
    """Which exact zeros of an AMPLITUDE stack are no-data, and which are genuine small amplitudes.

    A muted, padded or dead stretch is a run of exact zeros; a live trace can also contain
    isolated exact zeros (e.g. a volume whose small amplitudes were rounded or gated to 0).
    Rule: a zero is missing only when it lies in a run of >= ``min_run`` consecutive zeros
    along time (a whole-zero trace is such a run).  Returns (zero_mask, missing_mask).
    """
    zero = np.asarray(X) == 0.0
    if min_run <= 1:
        return zero, zero.copy()
    runs = zero_run_lengths(zero)
    return zero, runs >= int(min_run)


def stack_attributes(X: np.ndarray, dt_ms: float, min_finite_fraction: float = 0.7) -> dict[str, np.ndarray]:
    """All STACK_ATTRS for one channel.  X: [N, nI, nJ, nT] -> dict of [N] arrays (NaN if not computable)."""
    with warnings.catch_warnings():                 # all-NaN rows are expected (missing volumes): stay quiet
        warnings.simplefilter("ignore", RuntimeWarning)
        return _stack_attributes(X, dt_ms, min_finite_fraction)


def _stack_attributes(X, dt_ms, min_finite_fraction):
    X = np.asarray(X, dtype=np.float64)
    n, ni, nj, nt = X.shape
    ci, cj, ck = _centre(X.shape)
    T = X[:, ci, cj, :]                                         # centre traces [N, nT]
    fin = np.isfinite(T)
    ok_t = fin.mean(1) >= min_finite_fraction                   # enough samples for window statistics
    full_t = fin.all(1)
    out = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["amp_c"] = T[:, ck]
        out["rms_short"] = np.sqrt(np.nanmean(T[:, max(0, ck - 2):ck + 3] ** 2, axis=1))
        out["rms_long"] = np.where(ok_t, np.sqrt(np.nanmean(T ** 2, axis=1)), np.nan)
        out["maxabs"] = np.where(ok_t, np.nanmax(np.abs(np.where(fin, T, 0.0)), axis=1), np.nan)
        out["p2p"] = np.where(ok_t, np.nanmax(np.where(fin, T, -np.inf), 1) - np.nanmin(np.where(fin, T, np.inf), 1),
                              np.nan)
        out["std_t"] = np.where(ok_t, np.nanstd(T, axis=1), np.nan)
        # analytic signal on a reflect-padded window (needs a complete centre trace)
        env = np.full(n, np.nan)
        cph = np.full(n, np.nan)
        if full_t.any():
            padded = np.pad(T[full_t], ((0, 0), (nt, nt)), mode="reflect")
            analytic = hilbert(padded, axis=1)[:, nt + ck]
            env[full_t] = np.abs(analytic)
            cph[full_t] = np.where(env[full_t] > 0, T[full_t, ck] / env[full_t], np.nan)
        out["env_c"], out["cosphase_c"] = env, cph
        out["dadt_c"] = (T[:, ck + 1] - T[:, ck - 1]) / (2.0 * dt_ms) if nt >= 3 else np.full(n, np.nan)
        S = X[:, :, :, ck].reshape(n, -1)
        out["spatial_std_c"] = np.where(np.isfinite(S).mean(1) >= min_finite_fraction, np.nanstd(S, axis=1), np.nan)
        if ni >= 3 and nj >= 3:
            gil = (X[:, ci + 1, cj, ck] - X[:, ci - 1, cj, ck]) / 2.0
            gxl = (X[:, ci, cj + 1, ck] - X[:, ci, cj - 1, ck]) / 2.0
            out["grad_xy_c"] = np.hypot(gil, gxl)
        else:
            out["grad_xy_c"] = np.full(n, np.nan)
        W = X[:, :, :, max(0, ck - 2):ck + 3].reshape(n, ni * nj, -1)   # [N, traces, time]
        complete = np.isfinite(W).all(axis=(1, 2))
        num = (W.sum(axis=1) ** 2).sum(axis=1)
        den = ni * nj * (W ** 2).sum(axis=(1, 2))
        out["semblance"] = np.where(complete & (den > 0), num / den, np.nan)
        cent = np.full(n, np.nan)
        if full_t.any():
            taper = np.hanning(nt)
            spec = np.abs(np.fft.rfft((T[full_t] - T[full_t].mean(1, keepdims=True)) * taper, axis=1)) ** 2
            f = np.fft.rfftfreq(nt, dt_ms / 1000.0)
            p = spec[:, 1:]
            tot = p.sum(1)
            cent[full_t] = np.where(tot > 0, (p * f[1:]).sum(1) / tot, np.nan)
        out["spec_centroid_hz"] = cent
    return out


def local_summaries(X: np.ndarray, half_il: int = 1, half_xl: int = 1, half_t: int = 2) -> dict[str, np.ndarray]:
    """Centre value, local mean and local std in a small (2h+1)^3 box (for inversion volumes)."""
    X = np.asarray(X, dtype=np.float64)
    ci, cj, ck = _centre(X.shape)
    box = X[:, ci - half_il:ci + half_il + 1, cj - half_xl:cj + half_xl + 1, ck - half_t:ck + half_t + 1]
    flat = box.reshape(X.shape[0], -1)
    enough = np.isfinite(flat).mean(1) >= 0.7
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {"c": X[:, ci, cj, ck],
                "local_mean": np.where(enough, np.nanmean(flat, 1), np.nan),
                "local_std": np.where(enough, np.nanstd(flat, 1), np.nan)}


def normalized_difference(a, b) -> np.ndarray:
    """(a - b) / (a + b) for non-negative quantities; NaN where the sum is ~0 (safe denominator)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = a + b
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.abs(s) > 1e-12, (a - b) / s, np.nan)


def trend_vs_order(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares slope/intercept of amplitude against stack ORDER 1..M (a proxy, not AVO)."""
    v = np.asarray(values, float)                               # [N, M]
    x = np.arange(1, v.shape[1] + 1, dtype=float)
    ok = np.isfinite(v).all(1)
    xm = x.mean()
    slope = np.full(v.shape[0], np.nan)
    inter = np.full(v.shape[0], np.nan)
    if ok.any():
        vm = v[ok].mean(1)
        slope[ok] = ((x - xm) * (v[ok] - vm[:, None])).sum(1) / ((x - xm) ** 2).sum()
        inter[ok] = vm - slope[ok] * xm
    return slope, inter


# ----------------------------------------------------------------------------- AVO
def shuey_fit(amplitudes: np.ndarray, angles_deg: np.ndarray) -> dict:
    """Two-term Shuey fit R = A + B sin^2(theta) for one event (>= 3 angles).

    Returns A, B, their standard errors, residual RMS, R^2 and the number of angles used.
    """
    y = np.asarray(amplitudes, float)
    th = np.radians(np.asarray(angles_deg, float))
    ok = np.isfinite(y) & np.isfinite(th)
    if ok.sum() < 3:
        return {"A": np.nan, "B": np.nan, "se_A": np.nan, "se_B": np.nan, "residual_rms": np.nan,
                "r2": np.nan, "n_angles": int(ok.sum())}
    x = np.sin(th[ok]) ** 2
    G = np.column_stack([np.ones(ok.sum()), x])
    coef, *_ = np.linalg.lstsq(G, y[ok], rcond=None)
    res = y[ok] - G @ coef
    dof = max(1, ok.sum() - 2)
    s2 = float((res ** 2).sum() / dof)
    cov = s2 * np.linalg.inv(G.T @ G)
    ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
    return {"A": float(coef[0]), "B": float(coef[1]), "se_A": float(np.sqrt(cov[0, 0])),
            "se_B": float(np.sqrt(cov[1, 1])), "residual_rms": float(np.sqrt((res ** 2).mean())),
            "r2": float(1 - (res ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan, "n_angles": int(ok.sum())}


def avo_class_candidate(A: float, B: float, near_zero: float) -> str:
    """Rutherford-Williams/Castagna style CANDIDATE class of a reflection event (not a fluid label).

    Assumes SEG normal polarity (positive = impedance increase).  ``near_zero`` is the |A|
    below which the intercept counts as ~0 (Class II); it must be set relative to the event
    population because seismic amplitudes are not calibrated reflectivity.
    """
    if not (np.isfinite(A) and np.isfinite(B)):
        return "undetermined"
    if abs(A) <= near_zero:
        return "II (A~0)"
    if A > 0:
        return "I" if B < 0 else "I (B>=0)"
    return "III" if B < 0 else "IV"


def pick_events(t_ms: np.ndarray, amp: np.ndarray, dt_ms: float, quantile: float = 0.75) -> np.ndarray:
    """Indices of local extrema (peaks and troughs) along one path with |amp| >= the given quantile.

    Only samples whose both neighbours exist and are exactly one sample away count.
    """
    t, a = np.asarray(t_ms, float), np.asarray(amp, float)
    if a.size < 3 or not np.isfinite(a).any():
        return np.zeros(0, dtype=int)
    thr = np.nanquantile(np.abs(a), quantile)
    idx = []
    for i in range(1, a.size - 1):
        if not (np.isfinite(a[i - 1:i + 2]).all() and abs(t[i] - t[i - 1] - dt_ms) < 1e-6
                and abs(t[i + 1] - t[i] - dt_ms) < 1e-6):
            continue
        peak = a[i] > a[i - 1] and a[i] >= a[i + 1] and a[i] > 0
        trough = a[i] < a[i - 1] and a[i] <= a[i + 1] and a[i] < 0
        if (peak or trough) and abs(a[i]) >= thr:
            idx.append(i)
    return np.asarray(idx, dtype=int)


def matched_event_amplitude(trace: np.ndarray, ck: int, sign: float, search: int = 1) -> tuple[float, int]:
    """Amplitude of the same-polarity extremum within +/-search samples of ck (registration tolerance)."""
    seg = np.asarray(trace[max(0, ck - search):ck + search + 1], float)
    if not np.isfinite(seg).any():
        return np.nan, 0
    j = int(np.nanargmax(seg * sign))
    return float(seg[j]), j - min(search, ck)
