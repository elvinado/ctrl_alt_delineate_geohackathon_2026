"""Read-only, memory-mapped SEG-Y access for very large post-stack volumes.

Why not just ``segyio``?
------------------------
``segyio`` is excellent, but (a) it is not installed in your environment and
(b) its fast "geometry" mode needs a perfectly regular, sorted cube.  The
Challenge 1/2 volumes are *cropped*: per the team's dataset summary the stacks
span IL 748-3054 x XL 1181-4981 but hold only ~1.45 M traces, so the footprint
is irregular.  This module therefore:

* memory-maps the file (the OS pages data in on demand - a 5.4 GB volume is
  never loaded into RAM),
* scans only the 240-byte trace headers, in chunks, with a progress bar,
* builds an explicit (inline, crossline) -> trace-index lookup grid (missing bins
  = -1, never padded with fake zeros),
* fits the affine (IL, XL) -> (X, Y) transform and reports its residual, bin
  spacing and line azimuths, and
* decodes IEEE (format 5) and IBM (format 1) floats, plus integer formats.

Byte locations are configurable (``seismic.byte_locations``); defaults follow
SEG-Y rev 1 (inline 189, crossline 193, CDP X 181, CDP Y 185, scalar 71).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .progress import progress

DEFAULT_BYTE_LOCATIONS = {
    "inline": 189, "crossline": 193, "cdp_x": 181, "cdp_y": 185, "coord_scalar": 71,
    "delay_ms": 109, "ns": 115, "dt_us": 117, "cdp": 21, "coord_units": 89,
}
_FIELD_SIZE = {"inline": 4, "crossline": 4, "cdp_x": 4, "cdp_y": 4, "coord_scalar": 2,
               "delay_ms": 2, "ns": 2, "dt_us": 2, "cdp": 4, "coord_units": 2}
_FORMATS = {1: ("ibm", 4), 2: ("i4", 4), 3: ("i2", 2), 5: ("f4", 4), 8: ("i1", 1)}


class SegyError(RuntimeError):
    pass


# =============================================================================
# IBM <-> IEEE conversion
# =============================================================================
def ibm2ieee(words: np.ndarray) -> np.ndarray:
    """Convert IBM System/360 32-bit floats (given as uint32 words) to float32.

    value = (-1)^sign * 0.mantissa(24 bit) * 16^(exponent - 64)
    """
    w = np.asarray(words, dtype=np.uint32)
    sign = np.where((w >> 31) & 1, -1.0, 1.0)
    exponent = ((w >> 24) & 0x7F).astype(np.int32) - 64
    mantissa = (w & 0x00FFFFFF).astype(np.float64) / float(1 << 24)
    return (sign * mantissa * np.power(16.0, exponent)).astype(np.float32)


def ieee2ibm(values: np.ndarray) -> np.ndarray:
    """Convert floats to IBM 32-bit words (uint32).  Used by tests / future writers."""
    x = np.asarray(values, dtype=np.float64)
    out = np.zeros(x.shape, dtype=np.uint32)
    nz = x != 0
    ax = np.abs(x[nz])
    e = np.floor(np.log(ax) / np.log(16.0)).astype(np.int64) + 1
    mant = ax / np.power(16.0, e)
    hi = mant >= 1.0
    e[hi] += 1
    mant[hi] /= 16.0
    lo = mant < 1.0 / 16.0
    e[lo] -= 1
    mant[lo] *= 16.0
    frac = np.round(mant * (1 << 24)).astype(np.int64)
    over = frac >= (1 << 24)
    frac[over] //= 16
    e[over] += 1
    sign = (x[nz] < 0).astype(np.uint64) << 31
    out[nz] = (sign | ((e + 64).astype(np.uint64) << 24) | frac.astype(np.uint64)).astype(np.uint32)
    return out


# =============================================================================
# Geometry
# =============================================================================
@dataclass
class SurveyGeometry:
    """Where every trace sits: IL/XL grid, XY transform and time axis."""

    name: str
    n_traces: int
    il_min: int
    il_max: int
    il_step: int
    xl_min: int
    xl_max: int
    xl_step: int
    grid: np.ndarray                    # [n_il_bins, n_xl_bins] -> trace index or -1
    affine: dict                        # x = a0 + a1*il + a2*xl ; y = b0 + b1*il + b2*xl
    affine_residual_max: float
    affine_residual_rms: float
    t0_ms: float
    dt_ms: float
    ns: int
    n_duplicate_bins: int = 0
    flags: list = field(default_factory=list)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_headers(cls, name, inline, crossline, x, y, t0_ms, dt_ms, ns, max_fit_points=200_000):
        inline = np.asarray(inline, dtype=np.int64)
        crossline = np.asarray(crossline, dtype=np.int64)
        flags = []
        il_u, xl_u = np.unique(inline), np.unique(crossline)
        il_step = _grid_step(il_u)
        xl_step = _grid_step(xl_u)
        il_min, il_max, xl_min, xl_max = int(il_u[0]), int(il_u[-1]), int(xl_u[0]), int(xl_u[-1])
        if np.any((il_u - il_min) % il_step) or np.any((xl_u - xl_min) % xl_step):
            flags.append("inline/crossline numbers not on a regular increment")
        n_il = (il_max - il_min) // il_step + 1
        n_xl = (xl_max - xl_min) // xl_step + 1
        grid = np.full((n_il, n_xl), -1, dtype=np.int64 if len(inline) > 2**31 - 1 else np.int32)
        ii = (inline - il_min) // il_step
        jj = (crossline - xl_min) // xl_step
        flat = ii * n_xl + jj
        _, first_idx, counts = np.unique(flat, return_index=True, return_counts=True)
        n_dup = int((counts > 1).sum())
        if n_dup:
            flags.append(f"{n_dup} duplicate (IL,XL) bins - first trace kept")
        grid.ravel()[flat[first_idx]] = first_idx
        # affine fit on a subsample of traces with non-zero coordinates
        ok = np.flatnonzero((np.asarray(x) != 0) | (np.asarray(y) != 0))
        if ok.size < 3:
            raise SegyError(f"{name}: fewer than 3 traces with non-zero coordinates")
        sel = ok[np.linspace(0, ok.size - 1, min(ok.size, max_fit_points)).astype(int)]
        design = np.column_stack([np.ones(sel.size), inline[sel], crossline[sel]]).astype(float)
        ax, *_ = np.linalg.lstsq(design, np.asarray(x, dtype=float)[sel], rcond=None)
        by, *_ = np.linalg.lstsq(design, np.asarray(y, dtype=float)[sel], rcond=None)
        resid = np.hypot(design @ ax - np.asarray(x, dtype=float)[sel],
                         design @ by - np.asarray(y, dtype=float)[sel])
        geom = cls(name=name, n_traces=int(len(inline)), il_min=il_min, il_max=il_max, il_step=il_step,
                   xl_min=xl_min, xl_max=xl_max, xl_step=xl_step, grid=grid,
                   affine={"a0": ax[0], "a1": ax[1], "a2": ax[2], "b0": by[0], "b1": by[1], "b2": by[2]},
                   affine_residual_max=float(resid.max()), affine_residual_rms=float(np.sqrt(np.mean(resid ** 2))),
                   t0_ms=float(t0_ms), dt_ms=float(dt_ms), ns=int(ns), n_duplicate_bins=n_dup, flags=flags)
        if geom.affine_residual_max > 0.25 * min(geom.bin_spacing()):
            geom.flags.append(f"XY not affine in IL/XL (max residual {geom.affine_residual_max:.2f} > "
                              f"25% of bin) - irregular or curved geometry")
        return geom

    # ------------------------------------------------------------------ properties
    @property
    def n_live(self) -> int:
        return int((self.grid >= 0).sum())

    @property
    def fill_fraction(self) -> float:
        return self.n_live / self.grid.size

    def times_ms(self) -> np.ndarray:
        return self.t0_ms + self.dt_ms * np.arange(self.ns)

    def bin_spacing(self) -> tuple[float, float]:
        """(trace spacing along an inline, line spacing between inlines) in XY units."""
        a = self.affine
        along_inline = np.hypot(a["a2"], a["b2"]) * self.xl_step
        between_inlines = np.hypot(a["a1"], a["b1"]) * self.il_step
        return float(along_inline), float(between_inlines)

    def azimuths_deg(self) -> dict:
        """Grid azimuths (clockwise from +Y) of increasing XL along an inline and increasing IL."""
        a = self.affine
        return {"inline_direction_increasing_xl": float(np.degrees(np.arctan2(a["a2"], a["b2"])) % 360),
                "crossline_direction_increasing_il": float(np.degrees(np.arctan2(a["a1"], a["b1"])) % 360)}

    # ------------------------------------------------------------------ transforms
    def ilxl_to_xy(self, il, xl):
        a = self.affine
        il, xl = np.asarray(il, dtype=float), np.asarray(xl, dtype=float)
        return a["a0"] + a["a1"] * il + a["a2"] * xl, a["b0"] + a["b1"] * il + a["b2"] * xl

    def xy_to_ilxl(self, x, y):
        """Fractional inline/crossline for map coordinates (inverse affine)."""
        a = self.affine
        m = np.array([[a["a1"], a["a2"]], [a["b1"], a["b2"]]])
        inv = np.linalg.inv(m)
        dx = np.asarray(x, dtype=float) - a["a0"]
        dy = np.asarray(y, dtype=float) - a["b0"]
        return inv[0, 0] * dx + inv[0, 1] * dy, inv[1, 0] * dx + inv[1, 1] * dy

    def trace_index(self, il, xl) -> np.ndarray:
        """Trace index for integer IL/XL (-1 where the bin is empty or outside the survey)."""
        il = np.atleast_1d(np.asarray(il, dtype=np.int64))
        xl = np.atleast_1d(np.asarray(xl, dtype=np.int64))
        i = (il - self.il_min) // self.il_step
        j = (xl - self.xl_min) // self.xl_step
        on_grid = ((il - self.il_min) % self.il_step == 0) & ((xl - self.xl_min) % self.xl_step == 0)
        inside = on_grid & (i >= 0) & (i < self.grid.shape[0]) & (j >= 0) & (j < self.grid.shape[1])
        out = np.full(il.shape, -1, dtype=np.int64)
        out[inside] = self.grid[i[inside], j[inside]]
        return out

    def nearest_bin(self, il_f, xl_f):
        """Round fractional IL/XL to the nearest grid line numbers."""
        il = self.il_min + np.round((np.asarray(il_f) - self.il_min) / self.il_step) * self.il_step
        xl = self.xl_min + np.round((np.asarray(xl_f) - self.xl_min) / self.xl_step) * self.xl_step
        return il.astype(np.int64), xl.astype(np.int64)

    # ------------------------------------------------------------------ persistence
    def summary(self) -> dict:
        along, between = self.bin_spacing()
        az = self.azimuths_deg()
        return {
            "volume": self.name, "n_traces": self.n_traces, "n_live_bins": self.n_live,
            "il_min": self.il_min, "il_max": self.il_max, "il_step": self.il_step,
            "xl_min": self.xl_min, "xl_max": self.xl_max, "xl_step": self.xl_step,
            "grid_fill_fraction": round(self.fill_fraction, 4),
            "t0_ms": self.t0_ms, "dt_ms": self.dt_ms, "ns": self.ns,
            "t_max_ms": self.t0_ms + self.dt_ms * (self.ns - 1),
            "trace_spacing_along_inline": along, "inline_spacing": between,
            "azimuth_inline_dir_deg": az["inline_direction_increasing_xl"],
            "azimuth_crossline_dir_deg": az["crossline_direction_increasing_il"],
            "affine_residual_max": self.affine_residual_max, "affine_residual_rms": self.affine_residual_rms,
            "n_duplicate_bins": self.n_duplicate_bins, "geometry_flags": ";".join(self.flags),
            **{f"affine_{k}": float(v) for k, v in self.affine.items()},
        }

    def save_npz(self, path) -> None:
        """Save the lookup grid + metadata (JSON text, no pickle) for later stages."""
        np.savez_compressed(path, grid=self.grid, meta_json=np.array(json.dumps(self._meta())))

    def _meta(self) -> dict:
        d = {k: getattr(self, k) for k in ("name", "n_traces", "il_min", "il_max", "il_step", "xl_min",
                                           "xl_max", "xl_step", "affine_residual_max", "affine_residual_rms",
                                           "t0_ms", "dt_ms", "ns", "n_duplicate_bins")}
        d["affine"] = {k: float(v) for k, v in self.affine.items()}
        d["flags"] = list(self.flags)
        return d

    @classmethod
    def load_npz(cls, path) -> "SurveyGeometry":
        with np.load(path, allow_pickle=False) as z:   # no pickle: safe to open
            meta = json.loads(str(z["meta_json"]))
            grid = z["grid"]
        return cls(grid=grid, **meta)


def _grid_step(unique_sorted: np.ndarray) -> int:
    if unique_sorted.size < 2:
        return 1
    return int(max(1, np.gcd.reduce(np.diff(unique_sorted).astype(np.int64))))


# =============================================================================
# File access
# =============================================================================
class SegyFile:
    """Memory-mapped SEG-Y file.  Always use as a context manager on Windows::

        with SegyFile(path) as sgy:
            headers = sgy.scan_headers()
    """

    def __init__(self, path: str | Path, byte_locations: dict | None = None):
        self.path = Path(path)
        self.byte_locations = {**DEFAULT_BYTE_LOCATIONS, **(byte_locations or {})}
        self.size = self.path.stat().st_size
        with open(self.path, "rb") as fh:
            text = fh.read(3200)
            binary = fh.read(400)
        if len(binary) < 400:
            raise SegyError(f"{self.path.name}: file shorter than SEG-Y headers")
        self.textual_header = _decode_textual(text)
        self.endian = ">"
        self.binary = _parse_binary(binary, ">")
        if self.binary["format_code"] not in _FORMATS:
            alt = _parse_binary(binary, "<")
            if alt["format_code"] in _FORMATS:
                self.endian, self.binary = "<", alt
            else:
                raise SegyError(f"{self.path.name}: unsupported data format code {self.binary['format_code']}")
        self.format_code = self.binary["format_code"]
        self.sample_kind, self.bytes_per_sample = _FORMATS[self.format_code]
        self.ns = int(self.binary["samples_per_trace"])
        self.dt_us = int(self.binary["sample_interval_us"])
        n_ext = max(0, int(self.binary.get("n_extended_headers", 0)))
        self.data_offset = 3600 + 3200 * n_ext
        self.trace_bytes = 240 + self.ns * self.bytes_per_sample
        payload = self.size - self.data_offset
        if self.ns <= 0 or payload % self.trace_bytes != 0:
            raise SegyError(f"{self.path.name}: file size not consistent with ns={self.ns} "
                            f"(variable trace length or wrong binary header)")
        self.n_traces = payload // self.trace_bytes
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.data_offset,
                             shape=(self.n_traces, self.trace_bytes))

    # ------------------------------------------------------------------ lifecycle
    def close(self) -> None:
        mm = getattr(self, "_mm", None)
        if mm is not None:
            handle = getattr(mm, "_mmap", None)
            self._mm = None
            del mm
            if handle is not None:
                try:
                    handle.close()
                except (BufferError, ValueError):   # still referenced by a view - GC will close it
                    pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ headers
    def _field(self, block: np.ndarray, name: str) -> np.ndarray:
        pos = int(self.byte_locations[name]) - 1
        size = _FIELD_SIZE.get(name, 4)
        raw = np.ascontiguousarray(block[:, pos:pos + size])
        dtype = f"{self.endian}i{size}"
        return raw.view(dtype).reshape(-1).astype(np.int64)

    def scan_headers(self, fields=("inline", "crossline", "cdp_x", "cdp_y", "coord_scalar",
                                   "delay_ms", "ns", "dt_us"), chunk_traces: int = 50_000,
                     show_progress: bool = True) -> dict:
        """Read selected trace-header words for every trace (chunked, with a progress bar)."""
        out = {f: np.empty(self.n_traces, dtype=np.int64) for f in fields}
        bar = progress(total=self.n_traces, desc=f"headers {self.path.name}", unit="tr",
                       leave=False, disable=not show_progress)
        for i0 in range(0, self.n_traces, chunk_traces):
            i1 = min(self.n_traces, i0 + chunk_traces)
            block = self._mm[i0:i1, :240]
            for f in fields:
                out[f][i0:i1] = self._field(block, f)
            bar.update(i1 - i0)
        bar.close()
        return out

    @staticmethod
    def apply_scalar(values: np.ndarray, scalar: np.ndarray) -> np.ndarray:
        """SEG-Y coordinate scalar: >0 multiply, <0 divide, 0 = 1."""
        v = np.asarray(values, dtype=float)
        s = np.asarray(scalar, dtype=float)
        factor = np.where(s > 0, s, np.where(s < 0, 1.0 / np.abs(np.where(s == 0, 1, s)), 1.0))
        return v * factor

    def geometry(self, headers: dict | None = None, name: str | None = None) -> SurveyGeometry:
        h = headers or self.scan_headers()
        x = self.apply_scalar(h["cdp_x"], h["coord_scalar"])
        y = self.apply_scalar(h["cdp_y"], h["coord_scalar"])
        t0 = float(np.median(h["delay_ms"])) if "delay_ms" in h else 0.0
        geom = SurveyGeometry.from_headers(name or self.path.stem, h["inline"], h["crossline"], x, y,
                                           t0, self.dt_us / 1000.0, self.ns)
        if "delay_ms" in h and np.ptp(h["delay_ms"]) != 0:
            geom.flags.append("trace delay (t0) varies between traces")
        if "ns" in h and np.any((h["ns"] != self.ns) & (h["ns"] != 0)):
            geom.flags.append("trace-header ns differs from binary header")
        if "dt_us" in h and np.any((h["dt_us"] != self.dt_us) & (h["dt_us"] != 0)):
            geom.flags.append("trace-header dt differs from binary header")
        return geom

    # ------------------------------------------------------------------ samples
    def _decode(self, raw: np.ndarray) -> np.ndarray:
        raw = np.ascontiguousarray(raw)
        if self.sample_kind == "ibm":
            words = raw.view(f"{self.endian}u4")
            return ibm2ieee(words.astype(np.uint32))
        if self.sample_kind == "i1":
            return raw.view(np.int8).astype(np.float32)
        return raw.view(f"{self.endian}{self.sample_kind}").astype(np.float32)

    def read_trace(self, index: int) -> np.ndarray:
        return self._decode(self._mm[int(index), 240:]).reshape(-1)

    def read_traces(self, indices) -> np.ndarray:
        """Decoded samples for many traces -> float32 array [n, ns] (-1 index -> NaN row)."""
        idx = np.asarray(indices, dtype=np.int64)
        out = np.full((idx.size, self.ns), np.nan, dtype=np.float32)
        valid = np.flatnonzero(idx >= 0)
        if valid.size:
            order = valid[np.argsort(idx[valid])]          # sequential I/O is much faster
            raw = self._mm[idx[order], 240:]
            out[order] = self._decode(raw).reshape(order.size, self.ns)
        return out

    def amplitude_statistics(self, n_traces: int = 2000, seed: int = 0, window_ms: float = 250.0,
                             t0_ms: float = 0.0) -> dict:
        """Robust amplitude summary from a random sample of traces (for stack comparisons)."""
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.n_traces, size=min(n_traces, self.n_traces), replace=False))
        data = self.read_traces(idx)
        finite = np.isfinite(data)
        vals = data[finite]
        q = np.percentile(vals, [1, 50, 99]) if vals.size else [np.nan] * 3
        dt = self.dt_us / 1000.0
        nwin = max(1, int(round(window_ms / dt)))
        rms_profile = []
        for k in range(0, self.ns, nwin):
            block = data[:, k:k + nwin]
            rms_profile.append((t0_ms + k * dt, float(np.sqrt(np.nanmean(block.astype(np.float64) ** 2)))))
        return {"n_sampled_traces": int(idx.size), "rms": float(np.sqrt(np.mean(vals.astype(np.float64) ** 2))),
                "p01": float(q[0]), "p50": float(q[1]), "p99": float(q[2]),
                "fraction_exact_zero": float((vals == 0).mean()) if vals.size else np.nan,
                "n_non_finite": int((~finite).sum()), "rms_by_window": rms_profile}

    def extract_along_path(self, geom: SurveyGeometry, x, y, twt_ms) -> dict:
        """Sample the volume along a well path: nearest trace + linear interpolation in time.

        Returns values plus the trace used, its IL/XL and the horizontal offset
        between the requested point and that trace (so nothing is hidden).
        """
        il_f, xl_f = geom.xy_to_ilxl(x, y)
        il_n, xl_n = geom.nearest_bin(il_f, xl_f)
        tr = geom.trace_index(il_n, xl_n)
        tx, ty = geom.ilxl_to_xy(il_n, xl_n)
        offset = np.hypot(np.asarray(x, dtype=float) - tx, np.asarray(y, dtype=float) - ty)
        f = (np.asarray(twt_ms, dtype=float) - geom.t0_ms) / geom.dt_ms
        values = np.full(f.shape, np.nan, dtype=np.float64)
        uniq = np.unique(tr[tr >= 0])
        traces = self.read_traces(uniq)
        lookup = {int(t): k for k, t in enumerate(uniq)}
        for i in range(f.size):
            if tr[i] < 0 or not np.isfinite(f[i]) or f[i] < 0 or f[i] > geom.ns - 1:
                continue
            row = traces[lookup[int(tr[i])]]
            k = int(np.floor(f[i]))
            w = f[i] - k
            values[i] = row[k] if w == 0 or k + 1 >= geom.ns else (1 - w) * row[k] + w * row[k + 1]
        return {"values": values, "il": il_n, "xl": xl_n, "il_frac": il_f, "xl_frac": xl_f,
                "trace_index": tr, "offset_xy": offset}

    def read_inline(self, geom: SurveyGeometry, il: int) -> tuple[np.ndarray, np.ndarray]:
        """Full inline section [n_xl, ns] (NaN for empty bins) and its crossline numbers."""
        xls = np.arange(geom.xl_min, geom.xl_max + 1, geom.xl_step)
        return self.read_traces(geom.trace_index(np.full(xls.shape, il), xls)), xls

    def read_crossline(self, geom: SurveyGeometry, xl: int) -> tuple[np.ndarray, np.ndarray]:
        """Full crossline section [n_il, ns] (NaN for empty bins) and its inline numbers."""
        ils = np.arange(geom.il_min, geom.il_max + 1, geom.il_step)
        return self.read_traces(geom.trace_index(ils, np.full(ils.shape, xl))), ils


def _decode_textual(raw: bytes) -> list[str]:
    printable = sum(32 <= b < 127 for b in raw) / max(1, len(raw))
    text = raw.decode("ascii", errors="replace") if printable > 0.8 else raw.decode("cp500", errors="replace")
    return [text[i:i + 80].rstrip() for i in range(0, len(text), 80)]


def _parse_binary(raw: bytes, endian: str) -> dict:
    def word(pos, size, signed=True):
        start = pos - 3201
        return int.from_bytes(raw[start:start + size], "big" if endian == ">" else "little", signed=signed)

    return {
        "job_id": word(3201, 4), "line_number": word(3205, 4),
        "traces_per_ensemble": word(3213, 2), "sample_interval_us": word(3217, 2, False),
        "samples_per_trace": word(3221, 2, False), "format_code": word(3225, 2),
        "sorting_code": word(3229, 2), "measurement_system": word(3255, 2),
        "segy_revision_raw": word(3501, 2, False), "fixed_length_flag": word(3503, 2),
        "n_extended_headers": word(3505, 2),
    }
