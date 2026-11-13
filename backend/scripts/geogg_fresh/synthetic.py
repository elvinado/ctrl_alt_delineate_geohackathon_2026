"""Synthetic seismograms and synthetic-to-seismic ties (only where logs permit).

The convolutional model, step by step
-------------------------------------
1. Sonic DT -> P velocity:  us/ft -> Vp = 304800 / DT ;  us/m -> Vp = 1e6 / DT  (m/s)
2. Density RHOB -> kg/m3 (g/cc x 1000 when the values look like g/cc)
3. Acoustic impedance  AI = rho * Vp
4. Put every log sample at its TWT with the well's T-Z function **along the
   deviated path** (MD -> TVDSS -> TWT), then average AI onto a fine 1 ms grid
5. Reflectivity  R(t) = (AI(t+dt) - AI(t)) / (AI(t+dt) + AI(t))
6. Convolve with a wavelet.  Default: zero-phase Ricker at the *dominant
   frequency measured on the real full-stack trace around the well* (a
   documented statistical estimate - not tuned to maximise the tie)
7. Compare with the full-stack trace extracted along the well path: normalised
   cross-correlation for lags in +-max_lag; report best lag, CC at best lag,
   CC at zero lag and the polarity that gave the best match.

What this module never does: shift the T-Z silently to improve the match, or
use any label (Vsh/PHIT/SWT/class) to choose the wavelet.  If DT or RHOB is
missing the stage writes a NOT AVAILABLE panel - never a fake trace.
"""

from __future__ import annotations

import numpy as np


def sonic_to_vp(dt, unit: str | None = None) -> tuple[np.ndarray, str]:
    """Convert sonic slowness to velocity (m/s).  Unit inferred from values if blank."""
    dt = np.asarray(dt, dtype=float)
    u = (unit or "").strip().lower().replace(" ", "")
    if u in ("us/ft", "usec/ft", "us/f", "uspf", "microseconds/foot"):
        used = "us/ft"
    elif u in ("us/m", "usec/m", "uspm"):
        used = "us/m"
    else:
        med = np.nanmedian(dt)
        used = "us/m (inferred from values)" if med > 250 else "us/ft (inferred from values)"
    factor = 1e6 if used.startswith("us/m") else 304800.0
    with np.errstate(divide="ignore", invalid="ignore"):
        vp = np.where(dt > 0, factor / dt, np.nan)
    return vp, used


def density_to_kgm3(rhob, unit: str | None = None) -> tuple[np.ndarray, str]:
    rhob = np.asarray(rhob, dtype=float)
    u = (unit or "").strip().lower()
    if u in ("kg/m3", "kg/m^3") or np.nanmedian(rhob) > 100:
        return rhob, "kg/m3"
    return rhob * 1000.0, "g/cc -> kg/m3"


def ricker(peak_hz: float, dt_ms: float, length_ms: float = 128.0) -> tuple[np.ndarray, np.ndarray]:
    """Zero-phase Ricker wavelet sampled at ``dt_ms`` (unit peak amplitude)."""
    half = length_ms / 2.0
    t = np.arange(-half, half + dt_ms / 2, dt_ms) / 1000.0
    a = (np.pi * peak_hz * t) ** 2
    return t * 1000.0, (1 - 2 * a) * np.exp(-a)


def dominant_frequency(trace, dt_ms: float) -> dict:
    """Peak frequency and -6 dB band of a (tapered) trace's amplitude spectrum."""
    x = np.asarray(trace, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return {"peak_hz": np.nan, "band_low_hz": np.nan, "band_high_hz": np.nan}
    x = (x - x.mean()) * np.hanning(x.size)
    nfft = int(2 ** np.ceil(np.log2(x.size * 4)))
    spec = np.abs(np.fft.rfft(x, nfft))
    freq = np.fft.rfftfreq(nfft, dt_ms / 1000.0)
    kernel = np.ones(5) / 5
    smooth = np.convolve(spec, kernel, mode="same")
    k = int(np.argmax(smooth[1:]) + 1)
    above = np.flatnonzero(smooth >= smooth[k] / 2.0)            # -6 dB
    return {"peak_hz": float(freq[k]), "band_low_hz": float(freq[above.min()]),
            "band_high_hz": float(freq[above.max()])}


def impedance_in_time(twt_ms, ai, dt_ms: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Average impedance onto a regular TWT grid (empty bins filled by interpolation)."""
    t = np.asarray(twt_ms, dtype=float)
    a = np.asarray(ai, dtype=float)
    ok = np.isfinite(t) & np.isfinite(a)
    t, a = t[ok], a[ok]
    if t.size < 10:
        raise ValueError("Too few valid impedance samples for a synthetic")
    grid = np.arange(np.floor(t.min()), np.ceil(t.max()) + dt_ms, dt_ms)
    idx = np.clip(((t - grid[0]) / dt_ms).astype(int), 0, grid.size - 1)
    sums = np.bincount(idx, weights=a, minlength=grid.size)
    counts = np.bincount(idx, minlength=grid.size)
    vals = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    good = np.isfinite(vals)
    vals = np.interp(grid, grid[good], vals[good])
    return grid, vals


def reflectivity(ai_t) -> np.ndarray:
    ai = np.asarray(ai_t, dtype=float)
    rc = np.zeros_like(ai)
    rc[1:] = (ai[1:] - ai[:-1]) / (ai[1:] + ai[:-1])
    return rc


def make_synthetic(twt_ms, ai, wavelet: np.ndarray, dt_fine_ms: float = 1.0):
    """Return (time grid, AI on grid, reflectivity, synthetic) at ``dt_fine_ms``."""
    grid, ai_t = impedance_in_time(twt_ms, ai, dt_fine_ms)
    rc = reflectivity(ai_t)
    syn = np.convolve(rc, wavelet, mode="same")
    return grid, ai_t, rc, syn


def lagged_correlation(synthetic, observed, dt_ms: float, max_lag_ms: float = 40.0) -> dict:
    """Normalised cross-correlation of two equally sampled traces over +-max_lag."""
    s = np.asarray(synthetic, dtype=float)
    o = np.asarray(observed, dtype=float)
    max_k = int(round(max_lag_ms / dt_ms))
    lags, ccs = [], []
    for k in range(-max_k, max_k + 1):
        if k >= 0:
            a, b = s[: s.size - k] if k else s, o[k:]
        else:
            a, b = s[-k:], o[: o.size + k]
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
            cc = np.nan
        else:
            cc = float(np.corrcoef(a[ok], b[ok])[0, 1])
        lags.append(k * dt_ms)
        ccs.append(cc)
    lags, ccs = np.array(lags), np.array(ccs)
    if not np.isfinite(ccs).any():
        return {"lags_ms": lags, "cc": ccs, "best_lag_ms": np.nan, "cc_best": np.nan,
                "cc_zero_lag": np.nan, "polarity": "undetermined"}
    k = int(np.nanargmax(np.abs(ccs)))
    zero = ccs[lags == 0][0] if (lags == 0).any() else np.nan
    return {"lags_ms": lags, "cc": ccs, "best_lag_ms": float(lags[k]), "cc_best": float(ccs[k]),
            "cc_zero_lag": float(zero),
            "polarity": "as_recorded" if ccs[k] >= 0 else "reversed_relative_to_synthetic"}
