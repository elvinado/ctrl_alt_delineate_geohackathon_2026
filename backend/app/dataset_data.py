"""Materialize frontend-friendly data payloads from a dataset's stored source file.

The source file remains authoritative; these payloads are computed on demand and are
not duplicated in the relational database.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import lasio
import segyio

from app.file_headers import _decode_segy_text, _json_value
from app.well_tables import EXPECTED_COLUMNS
from app.well_list import read_excel_well_list

# Safety limits protect the API from attempting to serialize arbitrarily large files.
MAX_SEISMIC_TRACES = 5000
MAX_WELL_ROWS = 100_000
MAX_TABLE_ROWS = 250_000


def _number(value: Any, fallback: float | int | None = None) -> float | int | None:
    value = _json_value(value)
    if value is None or isinstance(value, bool):
        return fallback
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    if not math.isfinite(number):
        return fallback
    return int(number) if number.is_integer() else number


def read_seismic_data(path: Path, *, max_traces: int = 500) -> dict[str, Any]:
    """Read SEG-Y trace samples and requested trace/header metadata.

    ``traceCount`` is the total number of traces in the file. ``scannedTraceCount``
    is the number of trace sample arrays included in this response. If the file has
    more traces than ``max_traces``, the returned traces are the first N traces.
    Inline/crossline ranges are computed over all trace headers, not just returned
    samples.
    """
    if max_traces < 1 or max_traces > MAX_SEISMIC_TRACES:
        raise ValueError(f"max_traces must be between 1 and {MAX_SEISMIC_TRACES}")
    try:
        with segyio.open(str(path), "r", ignore_geometry=True) as segy:
            trace_count = len(segy.trace)
            scanned_count = min(trace_count, max_traces)
            traces: list[list[float]] = []
            trace_headers: list[dict[str, int]] = []
            inline_values: list[int] = []
            crossline_values: list[int] = []
            inline_field = getattr(segyio.TraceField, "INLINE_3D", None)
            crossline_field = getattr(segyio.TraceField, "CROSSLINE_3D", None)

            for index in range(trace_count):
                header = segy.header[index]
                inline = int(header[inline_field]) if inline_field is not None else 0
                crossline = int(header[crossline_field]) if crossline_field is not None else 0
                if inline_field is not None:
                    inline_values.append(inline)
                if crossline_field is not None:
                    crossline_values.append(crossline)
                if index < scanned_count:
                    trace_headers.append({"inline": inline, "crossline": crossline})
                    traces.append([float(v) if math.isfinite(float(v)) else 0.0 for v in segy.trace[index]])

            def binary_field(name: str) -> int:
                field = getattr(segyio.BinField, name, None)
                if field is None:
                    return 0
                try:
                    return int(segy.bin[field])
                except (KeyError, ValueError, TypeError, AttributeError):
                    return 0

            textual_header = ""
            try:
                textual_header = _decode_segy_text(segy.text[0])
            except (IndexError, TypeError, AttributeError):
                pass

            result = {
                "type": "seismic",
                "traces": traces,
                "traceHeaders": trace_headers,
                "traceCount": trace_count,
                "scannedTraceCount": scanned_count,
                "samplesPerTrace": len(segy.samples),
                "sampleIntervalUs": binary_field("Interval"),
                "formatCode": binary_field("Format"),
                "textualHeader": textual_header,
            }
            # Optional TypeScript properties are omitted when unavailable rather than
            # emitted as JSON null, so the payload matches the declared contract.
            if inline_values:
                result["inlineRange"] = [min(inline_values), max(inline_values)]
            if crossline_values:
                result["crosslineRange"] = [min(crossline_values), max(crossline_values)]
            return result
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to read SEG-Y data: {type(exc).__name__}: {exc}") from exc


def read_well_data(path: Path) -> dict[str, Any]:
    """Read LAS curves and numeric rows in source order."""
    try:
        las = lasio.read(str(path), ignore_data=False, engine="normal")
    except Exception as exc:
        raise ValueError(f"Unable to read LAS data: {type(exc).__name__}: {exc}") from exc

    data = las.data
    row_count = int(data.shape[0]) if data is not None and getattr(data, "ndim", 0) == 2 else 0
    if row_count > MAX_WELL_ROWS:
        raise ValueError(
            f"LAS file contains {row_count} rows; full dataset response is limited to {MAX_WELL_ROWS}. Use curve-specific paginated endpoints instead."
        )
    curves = [str(getattr(curve, "mnemonic", "") or "") for curve in las.curves]
    units = [str(getattr(curve, "unit", "") or "") for curve in las.curves]
    null_value = None
    try:
        null_value = _number(las.well.NULL.value)
    except (AttributeError, TypeError, ValueError):
        pass

    rows: list[list[float]] = []
    if data is not None:
        for row in data:
            normalized: list[float] = []
            for value in row:
                number = _number(value, null_value)
                normalized.append(float(number if number is not None else (null_value if null_value is not None else 0.0)))
            rows.append(normalized)

    index_values = [_number(value) for value in getattr(las, "index", [])]
    finite_indices = [float(value) for value in index_values if value is not None]
    result = {
        "type": "well",
        "curves": curves,
        "units": units,
        "rows": rows,
    }
    if finite_indices:
        result["depthStart"] = finite_indices[0]
        result["depthStop"] = finite_indices[-1]
    if null_value is not None:
        result["nullValue"] = null_value
    return result




def read_table_data(path: Path, kind: str) -> dict[str, Any]:
    """Read deviation/checkshot text as the frontend's string-based table shape."""
    if kind not in EXPECTED_COLUMNS:
        raise ValueError(f"Unsupported table dataset kind: {kind}")
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError as exc:
        raise ValueError(f"Unable to read table file: {exc}") from exc

    expected = EXPECTED_COLUMNS[kind]
    headers: list[str] = []
    raw_rows: list[list[str]] = []
    delimiter = " "
    for line_number, raw in enumerate(lines, start=1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            candidate = line.lstrip("#").strip()
            # Skip metadata comments; only accept a comment that names all expected columns.
            fields = re.split(r"\s+", candidate)
            if len(fields) == len(expected) and {f.casefold() for f in fields} == {f.casefold() for f in expected}:
                headers = fields
            continue
        if "," in line:
            fields = [part.strip() for part in line.split(",")]
            delimiter = ","
        elif "\t" in line:
            fields = [part.strip() for part in line.split("\t")]
            delimiter = "\t"
        else:
            fields = re.split(r"\s+", line)
            delimiter = " "
        # Ignore non-data preamble lines, but reject malformed numeric-looking records.
        try:
            [float(value) for value in fields]
        except ValueError:
            continue
        if len(fields) != len(expected):
            raise ValueError(f"Line {line_number} has {len(fields)} columns; expected {len(expected)}")
        raw_rows.append(fields)

    if not raw_rows:
        raise ValueError(f"No data rows found in {kind} table")
    if len(raw_rows) > MAX_TABLE_ROWS:
        raise ValueError(f"Table contains {len(raw_rows)} rows; response limit is {MAX_TABLE_ROWS}")
    if not headers:
        headers = expected.copy()
    return {
        "type": "table",
        "headers": headers,
        "rows": raw_rows,
        "rowCount": len(raw_rows),
        "delimiter": delimiter,
    }
