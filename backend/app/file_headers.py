"""Read metadata headers from SEG-Y and LAS files without modifying originals."""
from __future__ import annotations

from pathlib import Path
import math
from typing import Any

import lasio
import segyio


def _decode_segy_text(raw: Any) -> str:
    if isinstance(raw, str):
        return raw.rstrip()
    if not isinstance(raw, (bytes, bytearray)):
        return str(raw)
    # SEG-Y textual headers are commonly ASCII or EBCDIC (CP500).
    for encoding in ("ascii", "cp500"):
        try:
            decoded = bytes(raw).decode(encoding)
            # Prefer a decoding that looks like printable text.
            printable = sum(ch.isprintable() or ch in "\r\n\t" for ch in decoded)
            if decoded and printable / len(decoded) > 0.8:
                return decoded.rstrip()
        except (UnicodeDecodeError, LookupError):
            continue
    return bytes(raw).decode("ascii", errors="replace").rstrip()


def _json_value(value: Any) -> Any:
    """Convert common numpy/scientific scalar values to JSON-safe values."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_segy_header(path: Path) -> dict[str, Any]:
    """Return SEG-Y textual/binary header details and a small trace-header sample."""
    try:
        with segyio.open(str(path), mode="r", ignore_geometry=True) as segy:
            text_headers: list[str] = []
            try:
                text_headers.append(_decode_segy_text(segy.text[0]))
            except (IndexError, TypeError, AttributeError):
                pass
            ext_count = int(getattr(segy, "ext_headers", 0) or 0)
            for index in range(ext_count):
                try:
                    text_headers.append(_decode_segy_text(segy.text[index + 1]))
                except (IndexError, TypeError, AttributeError):
                    break

            binary_fields: dict[str, Any] = {}
            binary_names = {
                "job_id": "JobID", "line_number": "Line", "reel_number": "Reel",
                "traces_per_ensemble": "Traces", "aux_traces_per_ensemble": "AuxTraces",
                "sample_interval_us": "Interval", "samples_per_trace": "Samples",
                "sample_format_code": "Format", "measurement_system": "MeasurementSystem",
                "revision": "SEGYRevision", "fixed_length_trace_flag": "TraceFlag",
                "extended_text_header_count": "ExtTextHeaders",
            }
            for output_name, field_name in binary_names.items():
                field = getattr(segyio.BinField, field_name, None)
                if field is not None:
                    try:
                        binary_fields[output_name] = _json_value(segy.bin[field])
                    except (KeyError, ValueError, TypeError, AttributeError):
                        binary_fields[output_name] = None

            trace_count = len(segy.trace)
            trace_fields = {
                "trace_sequence_line": segyio.TraceField.TRACE_SEQUENCE_LINE,
                "trace_sequence_file": segyio.TraceField.TRACE_SEQUENCE_FILE,
                "field_record": segyio.TraceField.FieldRecord,
                "trace_number_in_field_record": segyio.TraceField.TraceNumber,
                "source_x": segyio.TraceField.SourceX,
                "source_y": segyio.TraceField.SourceY,
                "group_x": segyio.TraceField.GroupX,
                "group_y": segyio.TraceField.GroupY,
                "coordinate_scalar": segyio.TraceField.SourceGroupScalar,
                "coordinate_units_code": segyio.TraceField.CoordinateUnits,
            }
            first_trace: dict[str, Any] = {}
            if trace_count:
                header = segy.header[0]
                for name, field in trace_fields.items():
                    try:
                        first_trace[name] = _json_value(header[field])
                    except (KeyError, ValueError, TypeError):
                        first_trace[name] = None

            samples = len(segy.samples)
            return {
                "format": "segy",
                "readable": True,
                "textual_headers": text_headers,
                "binary_header": binary_fields,
                "trace_count": trace_count,
                "sample_count_per_trace": samples,
                "sample_values_per_trace": samples,
                "first_trace_header": first_trace,
                "warnings": [],
            }
    except Exception as exc:
        raise ValueError(f"Unable to parse SEG-Y file: {type(exc).__name__}: {exc}") from exc


def read_las_header(path: Path) -> dict[str, Any]:
    """Read LAS well, parameter and curve sections plus basic data dimensions."""
    try:
        las = lasio.read(str(path), ignore_data=False, engine="normal")
        def items(section: Any) -> list[dict[str, Any]]:
            output = []
            for item in section:
                output.append({
                    "mnemonic": str(getattr(item, "mnemonic", "") or ""),
                    "unit": str(getattr(item, "unit", "") or ""),
                    "value": _json_value(getattr(item, "value", None)),
                    "description": str(getattr(item, "descr", "") or ""),
                })
            return output

        curve_info = items(las.curves)
        data_rows = int(len(las.index))
        warnings: list[str] = []
        version = items(las.version)
        if not curve_info:
            warnings.append("No curve definitions were found in the LAS file.")
        return {
            "format": "las",
            "readable": True,
            "version": version,
            "well": items(las.well),
            "parameters": items(las.params),
            "curve_count": len(curve_info),
            "curves": curve_info,
            "data_row_count": data_rows,
            "index_mnemonic": str(getattr(las.index, "name", "") or ""),
            "index_unit": str(getattr(las.index, "unit", "") or ""),
            "warnings": warnings,
        }
    except Exception as exc:
        raise ValueError(f"Unable to parse LAS file: {type(exc).__name__}: {exc}") from exc


def read_file_header(path: Path, file_format: str) -> dict[str, Any]:
    normalized = file_format.lower().lstrip(".")
    if normalized in {"sgy", "segy"}:
        return read_segy_header(path)
    if normalized == "las":
        return read_las_header(path)
    raise ValueError(f"Header parsing is not supported for .{normalized} files; supported formats: sgy, segy, las")
