"""Build concise dataset descriptions from the file's parsed header."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

def _read_file_header(path: Path, file_format: str) -> dict[str, Any]:
    from app.file_headers import read_file_header
    return read_file_header(path, file_format)


def _read_well_table(path: Path, kind: str) -> dict[str, Any]:
    from app.well_tables import read_well_table
    return read_well_table(path, kind, offset=0, limit=0)


DEFAULT_DESCRIPTIONS = {
    "seismic": "Seismic volume data",
    "well_log": "Well log curves and measurements",
    "deviation": "Wellbore trajectory and directional survey",
    "checkshot": "Checkshot time-depth measurements",
}


def dataset_description(kind: str, file_format: str, path: Path | None) -> str:
    """Describe a dataset from its parsed header, falling back safely by kind."""
    if path is not None:
        try:
            if kind == "seismic" and file_format.lower().lstrip(".") in {"sgy", "segy"}:
                description = _describe_segy(_read_file_header(path, file_format))
            elif kind == "well_log" and file_format.lower().lstrip(".") == "las":
                description = _describe_las(_read_file_header(path, file_format))
            elif kind in {"deviation", "checkshot"}:
                description = _describe_well_table(_read_well_table(path, kind))
            else:
                description = None
            if description:
                return description
        except Exception:
            # Inventory should remain available even if an optional header read fails.
            pass
    return DEFAULT_DESCRIPTIONS.get(kind, "Geoscience dataset")


def _describe_segy(header: dict[str, Any]) -> str | None:
    if not header.get("readable"):
        return None

    details: list[str] = []
    trace_count = header.get("trace_count")
    sample_count = header.get("sample_count_per_trace")
    if isinstance(trace_count, int) and trace_count > 0:
        details.append(f"{trace_count:,} traces")
    if isinstance(sample_count, int) and sample_count > 0:
        details.append(f"{sample_count:,} samples per trace")

    interval = (header.get("binary_header") or {}).get("sample_interval_us")
    if isinstance(interval, (int, float)) and interval > 0:
        details.append(f"{interval / 1000:g} ms sample interval")

    textual_summary = _segy_text_summary(header.get("textual_headers") or [])
    if textual_summary:
        details.append(textual_summary)
    if not details:
        return None
    return "SEG-Y seismic data; " + "; ".join(details)


def _segy_text_summary(headers: list[str]) -> str | None:
    candidates: list[str] = []
    for header in headers:
        for raw_line in str(header).splitlines():
            line = re.sub(r"^\s*C\s*\d{1,2}\s*", "", raw_line, flags=re.IGNORECASE).strip()
            line = re.sub(r"\s+", " ", line)
            if len(line) < 5:
                continue
            if line.casefold() in {"end text header", "endtext"}:
                continue
            if line not in candidates:
                candidates.append(line)
    if not candidates:
        return None
    # Header text is source-controlled; keep the inventory concise and single-line.
    summary = " / ".join(candidates[:2])
    return f"header: {summary[:180]}"


def _describe_las(header: dict[str, Any]) -> str | None:
    if not header.get("readable"):
        return None

    well_items = header.get("well") or []
    well_name = _las_value(well_items, {"WELL", "WN"})
    index_name = str(header.get("index_mnemonic") or "").strip()
    index_unit = str(header.get("index_unit") or "").strip()
    curves = header.get("curves") or []
    curve_labels = []
    for curve in curves:
        mnemonic = str(curve.get("mnemonic") or "").strip()
        if not mnemonic:
            continue
        unit = str(curve.get("unit") or "").strip()
        curve_labels.append(f"{mnemonic} ({unit})" if unit else mnemonic)

    parts = ["LAS well log"]
    if well_name:
        parts.append(f"well {well_name}")
    if index_name:
        parts.append(f"index {index_name}" + (f" ({index_unit})" if index_unit else ""))
    if curve_labels:
        shown = ", ".join(curve_labels[:5])
        if len(curve_labels) > 5:
            shown += f", and {len(curve_labels) - 5} more"
        parts.append(f"curves: {shown}")
    row_count = header.get("data_row_count")
    if isinstance(row_count, int) and row_count > 0:
        parts.append(f"{row_count:,} samples")
    if len(parts) == 1:
        return None
    return "; ".join(parts)


def _las_value(items: list[dict[str, Any]], mnemonics: set[str]) -> str | None:
    for item in items:
        if str(item.get("mnemonic") or "").strip().upper() in mnemonics:
            value = str(item.get("value") or "").strip()
            if value:
                return value
    return None


def _describe_well_table(parsed: dict[str, Any]) -> str | None:
    columns = parsed.get("columns") or []
    names = []
    for column in columns:
        name = str(column.get("name") or "").strip()
        unit = str(column.get("unit") or "").strip()
        if name:
            names.append(f"{name} ({unit})" if unit else name)
    if not names:
        return None
    row_count = parsed.get("row_count")
    label = "Deviation survey" if parsed.get("kind") == "deviation" else "Checkshot table"
    description = f"{label}; columns: {', '.join(names)}"
    if isinstance(row_count, int) and row_count > 0:
        description += f"; {row_count:,} rows"
    return description
