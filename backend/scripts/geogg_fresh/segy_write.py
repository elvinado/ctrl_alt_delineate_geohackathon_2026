"""Write prediction volumes as SEG-Y on the SOURCE geometry, then prove it by reopening them.

Why a small writer of our own: segyio is not installed in the project environment and the
Challenge volumes are cropped / irregular (no full regular cube).  The rules here are simple
and checkable:

* textual (and any extended textual) header, binary header and every 240-byte trace header are
  COPIED from the reference full-stack file, for exactly the exported traces, in source order;
* only these words change: binary samples-per-trace + data format (5 = IEEE float32), trace
  samples-per-trace (115-116) and delay recording time (109-110) when the time axis is cropped;
* provenance goes into textual lines C37-C40 only when they are blank in the source, else into
  the JSON sidecar only (source metadata is never overwritten);
* one explicit no-data value (default -999.25); class 0 (shale) is a class, never "missing";
* files are written to ``<name>.partial`` and renamed after flush.

``reopen_qc`` then reopens every file with the package reader AND an independent byte parser
and returns PASS / WARN / FAIL rows (headers, geometry, sample axis, bit-exact data roundtrip,
value rules).
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .segy import DEFAULT_BYTE_LOCATIONS, SegyFile

NULL_VALUE = -999.25
BIN_NS, BIN_FORMAT, BIN_EXT_NS = 3221, 3225, 3269          # 1-based byte positions in the file


def _bin_put(binh: bytearray, pos: int, value: int, size: int, endian: str, signed=False) -> None:
    o = pos - 3201
    binh[o:o + size] = int(value).to_bytes(size, "big" if endian == ">" else "little", signed=signed)


def _text_encoding(raw: bytes) -> str:
    printable = sum(32 <= b < 127 for b in raw) / max(1, len(raw))
    return "ascii" if printable > 0.8 else "cp500"


def provenance_textual(raw: bytes, lines: list[str]) -> tuple[bytes, bool]:
    """Write up to 4 provenance lines into C37-C40 if those lines are blank; else keep the source text."""
    enc = _text_encoding(raw)
    text = raw.decode(enc, errors="replace")
    rows = [text[i:i + 80] for i in range(0, 3200, 80)]
    target = range(36, 40)
    blank = all(len(rows[i][4:].strip()) == 0 for i in target)
    if not blank or not lines:
        return raw, False
    for n, i in enumerate(target):
        body = lines[n] if n < len(lines) else ""
        rows[i] = f"C{i + 1:02d} {body}"[:80].ljust(80)
    return "".join(rows).encode(enc, errors="replace")[:3200].ljust(3200, b" " if enc == "ascii" else b"\x40"), True


class ReferenceSurvey:
    """The full-stack file whose headers every product inherits."""

    def __init__(self, path: Path, byte_locations: dict | None = None):
        self.path = Path(path)
        self.bl = {**DEFAULT_BYTE_LOCATIONS, **(byte_locations or {})}
        with SegyFile(self.path, byte_locations) as s:
            self.endian, self.ns, self.dt_us = s.endian, s.ns, s.dt_us
            self.data_offset, self.n_traces, self.format_code = s.data_offset, s.n_traces, s.format_code
        with open(self.path, "rb") as fh:
            self.textual = fh.read(3200)
            self.binary = fh.read(400)
            self.extended = fh.read(self.data_offset - 3600)

    def trace_headers(self, idx: np.ndarray) -> np.ndarray:
        with SegyFile(self.path, None) as s:
            return np.asarray(s._mm[np.asarray(idx, np.int64), :240]).copy()

    def header_word(self, headers: np.ndarray, name: str, size: int = 2) -> np.ndarray:
        pos = int(self.bl[name]) - 1
        return np.ascontiguousarray(headers[:, pos:pos + size]).view(f"{self.endian}i{size}").reshape(-1).astype(np.int64)

    def set_word(self, headers: np.ndarray, name: str, values, size: int = 2) -> None:
        pos = int(self.bl[name]) - 1
        kind = "u" if name in ("ns", "dt_us") else "i"
        v = np.broadcast_to(np.asarray(values, np.int64), (len(headers),)).astype(f"{self.endian}{kind}{size}")
        headers[:, pos:pos + size] = np.ascontiguousarray(v).view(np.uint8).reshape(len(headers), size)


def write_product(ref: ReferenceSurvey, out_path: Path, trace_idx: np.ndarray, k0: int, data: np.ndarray,
                  provenance: list[str] | None = None, chunk: int = 20000) -> dict:
    """Write one product: ``data`` [n_traces_out, ns_out] float32 (NULL_VALUE = no data)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trace_idx = np.asarray(trace_idx, np.int64)
    n, ns_out = data.shape
    if n != trace_idx.size:
        raise ValueError("data rows must equal the number of exported traces")
    if np.isnan(np.asarray(data[: min(n, 1000)])).any():
        raise ValueError("NaN in export data: use the explicit no-data value")
    text, wrote_text = provenance_textual(ref.textual, provenance or [])
    binh = bytearray(ref.binary)
    _bin_put(binh, BIN_NS, ns_out, 2, ref.endian)
    _bin_put(binh, BIN_FORMAT, 5, 2, ref.endian, signed=True)
    o = BIN_EXT_NS - 3201
    if any(binh[o:o + 4]):                               # rev-2 extended samples word in use: keep consistent
        _bin_put(binh, BIN_EXT_NS, ns_out, 4, ref.endian)
    dt_ms = ref.dt_us / 1000.0
    if k0 and abs(k0 * dt_ms - round(k0 * dt_ms)) > 1e-9:
        # the trace delay word (bytes 109-110) holds whole milliseconds: a crop that starts between
        # milliseconds cannot be described honestly, so refuse it instead of rounding the time axis
        raise ValueError(f"crop start {k0} x {dt_ms:g} ms is not a whole millisecond: choose another k0")
    tmp = out_path.with_suffix(out_path.suffix + ".partial")
    sha = hashlib.sha256()
    with open(tmp, "wb") as fh:
        for b in (text, bytes(binh), ref.extended):
            fh.write(b)
            sha.update(b)
        for s in range(0, n, chunk):
            idx = trace_idx[s:s + chunk]
            th = ref.trace_headers(idx)
            if k0:
                delay = ref.header_word(th, "delay_ms") + int(round(k0 * dt_ms))
                ref.set_word(th, "delay_ms", delay)
            ref.set_word(th, "ns", np.full(len(idx), ns_out))
            block = np.empty((len(idx), 240 + 4 * ns_out), np.uint8)
            block[:, :240] = th
            block[:, 240:] = np.ascontiguousarray(np.asarray(data[s:s + chunk], np.float32)
                                                  .astype(f"{ref.endian}f4")).view(np.uint8)
            b = block.tobytes()
            fh.write(b)
            sha.update(b)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, out_path)
    return {"path": str(out_path), "sha256": sha.hexdigest(), "bytes": out_path.stat().st_size,
            "n_traces": int(n), "ns": int(ns_out), "k0": int(k0), "provenance_in_textual_header": wrote_text}


# =============================================================================== reopen QC
def _row(product, check, status, detail="", tolerance=""):
    return {"product": product, "check": check, "result": status, "detail": detail, "tolerance": tolerance}


def independent_read(path: Path, endian: str, data_offset: int, ns: int, rows: np.ndarray) -> np.ndarray:
    """Decode selected traces straight from the bytes (no SegyFile): an independent readback."""
    out = np.empty((len(rows), ns), np.float32)
    tb = 240 + 4 * ns
    with open(path, "rb") as fh:
        for n, r in enumerate(np.asarray(rows, np.int64)):
            fh.seek(data_offset + int(r) * tb + 240)
            out[n] = np.frombuffer(fh.read(4 * ns), dtype=f"{endian}f4")
    return out


def reopen_qc(product: str, kind: str, out_path: Path, ref: ReferenceSurvey, trace_idx: np.ndarray, k0: int,
              expected: np.ndarray, valid_values=None, null=NULL_VALUE, n_independent: int = 64) -> list[dict]:
    """Reopen one exported product and check everything; returns PASS / WARN / FAIL rows."""
    rows = []
    trace_idx = np.asarray(trace_idx, np.int64)
    n, ns_out = expected.shape
    try:
        sgy = SegyFile(out_path)
    except Exception as e:  # noqa: BLE001
        return [_row(product, "reopens_with_package_reader", "FAIL", f"{type(e).__name__}: {e}")]
    with sgy:
        rows.append(_row(product, "reopens_with_package_reader", "PASS", out_path.name))
        rows.append(_row(product, "binary_format_ieee_float32", "PASS" if sgy.format_code == 5 else "FAIL",
                         f"format code {sgy.format_code}"))
        rows.append(_row(product, "binary_samples_and_interval", "PASS" if (sgy.ns == ns_out and sgy.dt_us == ref.dt_us)
                         else "FAIL", f"ns {sgy.ns} (expected {ns_out}); dt {sgy.dt_us} us (source {ref.dt_us})"))
        rows.append(_row(product, "endianness_same_as_source", "PASS" if sgy.endian == ref.endian else "FAIL",
                         sgy.endian))
        rows.append(_row(product, "trace_count", "PASS" if sgy.n_traces == n else "FAIL",
                         f"{sgy.n_traces} traces (expected {n})"))
        with open(out_path, "rb") as fh:
            text, binh = fh.read(3200), fh.read(400)
            ext = fh.read(sgy.data_offset - 3600)
        same_text = text[:36 * 80] == ref.textual[:36 * 80]
        rows.append(_row(product, "textual_header_lines_1_36_identical", "PASS" if same_text else "FAIL",
                         "C37-C40 may carry provenance when blank in the source"))
        rows.append(_row(product, "extended_textual_headers_identical", "PASS" if ext == ref.extended else "FAIL",
                         f"{len(ext) // 3200} extended header(s)"))
        mask = np.ones(400, bool)
        for pos, size in ((BIN_NS, 2), (BIN_FORMAT, 2), (BIN_EXT_NS, 4)):
            mask[pos - 3201:pos - 3201 + size] = False
        same_bin = np.array_equal(np.frombuffer(binh, np.uint8)[mask], np.frombuffer(ref.binary, np.uint8)[mask])
        rows.append(_row(product, "binary_header_identical_except_ns_format", "PASS" if same_bin else "FAIL"))
        # ---- trace headers: every trace, byte for byte (except ns / delay)
        bad_hdr, bad_delay, bad_ns = 0, 0, 0
        dt_ms = ref.dt_us / 1000.0
        hmask = np.ones(240, bool)
        for name in ("ns", "delay_ms"):
            pos = int(ref.bl[name]) - 1
            hmask[pos:pos + 2] = False
        for s in range(0, n, 20000):
            got = np.asarray(sgy._mm[s:s + 20000, :240])
            src = ref.trace_headers(trace_idx[s:s + 20000])
            bad_hdr += int((got[:, hmask] != src[:, hmask]).any(1).sum())
            bad_ns += int((ref.header_word(got, "ns") != ns_out).sum())
            bad_delay += int((ref.header_word(got, "delay_ms") != ref.header_word(src, "delay_ms")
                              + int(round(k0 * dt_ms))).sum())
        rows.append(_row(product, "trace_headers_identical_to_source", "PASS" if bad_hdr == 0 else "FAIL",
                         f"{bad_hdr} of {n} traces differ outside the ns / delay words (IL, XL, CDP X/Y, scalars, "
                         "coordinate units, trace order all inside the compared bytes)"))
        rows.append(_row(product, "trace_ns_word", "PASS" if bad_ns == 0 else "FAIL", f"{bad_ns} traces wrong"))
        rows.append(_row(product, "sample_origin_delay_word", "PASS" if bad_delay == 0 else "FAIL",
                         f"delay = source delay + {k0} samples x {dt_ms:g} ms; {bad_delay} traces wrong"))
        # ---- data: bit-exact roundtrip of every sample
        mismatch, nan_count = 0, 0
        h = hashlib.sha256()
        for s in range(0, n, 20000):
            got = sgy.read_traces(np.arange(s, min(n, s + 20000)))
            exp = np.asarray(expected[s:s + 20000], np.float32)
            mismatch += int((got.view(np.uint32) != exp.view(np.uint32)).sum())
            nan_count += int(np.isnan(got).sum())
            h.update(got.tobytes())
        rows.append(_row(product, "data_roundtrip_bit_exact", "PASS" if mismatch == 0 else "FAIL",
                         f"{mismatch} of {n * ns_out} samples differ from the pre-export store", "0 (float32 bits)"))
        rows.append(_row(product, "no_nan_in_file", "PASS" if nan_count == 0 else "FAIL",
                         f"{nan_count} NaN samples (no-data must be {null})"))
        rng = np.random.default_rng(11)
        pick = np.unique(np.r_[0, n - 1, rng.integers(0, n, min(n, n_independent))])
        ind = independent_read(out_path, sgy.endian, sgy.data_offset, sgy.ns, pick)
        same = np.array_equal(ind.view(np.uint32), np.asarray(expected[pick], np.float32).view(np.uint32))
        rows.append(_row(product, "independent_byte_parser_readback", "PASS" if same else "FAIL",
                         f"{len(pick)} traces (first, last, seeded random) decoded without the package reader"))
    # ---- value rules (on the expected store = file, proven identical above)
    rows += value_rules(product, kind, expected, valid_values, null)
    from .provenance import sha256_file

    rows.append(_row(product, "file_sha256", "INFO", sha256_file(out_path)))
    rows.append(_row(product, "data_sha256", "INFO", h.hexdigest()))
    return rows


def value_rules(product: str, kind: str, data: np.ndarray, valid_values=None, null=NULL_VALUE) -> list[dict]:
    rows = []
    n_valid, n_null, bad, lo, hi = 0, 0, 0, np.inf, -np.inf
    for s in range(0, data.shape[0], 20000):
        d = np.asarray(data[s:s + 20000], np.float32)
        isnull = d == np.float32(null)
        v = d[~isnull]
        n_null += int(isnull.sum())
        n_valid += int(v.size)
        if v.size:
            lo, hi = min(lo, float(v.min())), max(hi, float(v.max()))
        if kind in ("probability", "fraction", "diagnostic_fraction"):
            bad += int(((v < 0) | (v > 1) | ~np.isfinite(v)).sum())
        elif kind in ("class_id", "bitmask") and valid_values is not None:
            bad += int((~np.isin(v, np.asarray(valid_values, np.float32))).sum())
    name = {"probability": "probability_in_0_1", "fraction": "fraction_in_0_1", "diagnostic_fraction":
            "diagnostic_in_0_1", "class_id": "class_ids_in_trained_set", "bitmask": "mask_codes_valid"}.get(kind)
    if name:
        rows.append(_row(product, name, "PASS" if bad == 0 else "FAIL",
                         f"{bad} invalid of {n_valid} valid voxels; range [{lo:.4g}, {hi:.4g}]" if n_valid else
                         "no valid voxels", "exact"))
    rows.append(_row(product, "valid_voxel_count", "PASS" if n_valid > 0 else "WARN",
                     f"{n_valid} valid, {n_null} no-data ({n_null / max(1, n_valid + n_null):.1%})"))
    return rows
