"""Random-access SEG-Y geometry and section services for the seismic viewer."""
from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import segyio

GEOMETRY_LAYOUTS = (
    {"label": "SEG-Y Rev 1 standard", "inline_byte": 189, "crossline_byte": 193},
    {"label": "Legacy 181 / 185", "inline_byte": 181, "crossline_byte": 185},
    {"label": "Field record / CDP", "inline_byte": 9, "crossline_byte": 21},
    {"label": "Bytes 9 / 13", "inline_byte": 9, "crossline_byte": 13},
    {"label": "CDP / trace", "inline_byte": 21, "crossline_byte": 25},
)


def _header_int(header: Any, byte: int) -> int:
    try:
        return int(header[byte])
    except (KeyError, ValueError, TypeError, AttributeError, IndexError):
        return 0


def _score_geometry(inlines: list[int], crosslines: list[int], standard: bool) -> float:
    valid = [(i, x) for i, x in zip(inlines, crosslines) if i and x and abs(i) < 100_000_000 and abs(x) < 100_000_000]
    if len(valid) < min(8, len(inlines)):
        return 0.0
    ui, ux, n = len({p[0] for p in valid}), len({p[1] for p in valid}), len(valid)
    if ui < 2 or ux < 2:
        score = 4.5 if ((ui == 1 and ux >= 8) or (ux == 1 and ui >= 8)) else (1.0 if standard else .2)
    else:
        cell_ratio = ui * ux / n
        grid_fit = math.exp(-abs(math.log(max(.0001, cell_ratio))))
        repeated_axes = (.8 if ui < n * .8 else 0) + (.8 if ux < n * .8 else 0)
        score = len(valid) / len(inlines) * 4 + grid_fit * 4 + repeated_axes
    return score + (.25 if standard else 0)


@lru_cache(maxsize=8)
def _scan_geometry(path_text: str, mtime_ns: int, file_size: int, inline_byte: int, crossline_byte: int) -> dict[str, Any]:
    """Cache header index by canonical path, file fingerprint and byte mapping."""
    path = Path(path_text)
    try:
        with segyio.open(str(path), "r", ignore_geometry=True) as segy:
            count = len(segy.trace)
            inline_values = np.empty(count, dtype=np.int32)
            crossline_values = np.empty(count, dtype=np.int32)
            trace_intervals: set[int] = set()
            for index in range(count):
                header = segy.header[index]
                inline_values[index] = _header_int(header, inline_byte)
                crossline_values[index] = _header_int(header, crossline_byte)
                trace_samples = _header_int(header, 115)
                if trace_samples and trace_samples != len(segy.samples):
                    raise ValueError("Variable trace lengths detected; section retrieval requires fixed-length traces")
                trace_interval = _header_int(header, 117)
                if trace_interval:
                    trace_intervals.add(trace_interval)
            inline_positions = sorted(set(inline_values.tolist()) - {0})
            crossline_positions = sorted(set(crossline_values.tolist()) - {0})
            # A single survey-wide robust scale preserves relative amplitudes between traces.
            amplitudes: list[float] = []
            trace_stride = max(1, count // 256)
            for trace_index in range(0, count, trace_stride):
                samples = np.asarray(segy.trace[trace_index], dtype=np.float64)
                if samples.size:
                    sample_stride = max(1, samples.size // 512)
                    amplitudes.extend(np.abs(samples[::sample_stride][np.isfinite(samples[::sample_stride])]).tolist())
            scale = float(np.percentile(np.asarray(amplitudes), 98)) if amplitudes else 1.0
            if not math.isfinite(scale) or scale <= 0:
                scale = 1.0
            def binary_value(name: str) -> int:
                field = getattr(segyio.BinField, name, None)
                if field is None:
                    return 0
                try:
                    return int(segy.bin[field])
                except (KeyError, ValueError, TypeError, AttributeError):
                    return 0

            try:
                delay_ms = _header_int(segy.header[0], 109) if count else 0
            except (KeyError, ValueError, TypeError, AttributeError):
                delay_ms = 0
            binary_interval = binary_value("Interval")
            if len(trace_intervals) > 1 or (binary_interval and trace_intervals and trace_intervals != {binary_interval}):
                raise ValueError("Mixed trace sample intervals detected; section retrieval requires a consistent sample interval")
            if not binary_interval and trace_intervals:
                binary_interval = next(iter(trace_intervals))
            return {
                "trace_count": count,
                "inline": inline_values,
                "crossline": crossline_values,
                "inlines": inline_positions,
                "crosslines": crossline_positions,
                "samples_per_trace": len(segy.samples),
                "sample_interval_us": binary_interval,
                "format_code": binary_value("Format"),
                "delay_ms": delay_ms,
                "amplitude_scale": scale,
            }
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to index SEG-Y geometry: {type(exc).__name__}: {exc}") from exc


def _index(path: Path, inline_byte: int, crossline_byte: int) -> dict[str, Any]:
    if not (1 <= inline_byte <= 240 and 1 <= crossline_byte <= 240):
        raise ValueError("Inline and crossline byte positions must be between 1 and 240")
    stat = path.stat()
    return _scan_geometry(str(path.resolve()), stat.st_mtime_ns, stat.st_size, inline_byte, crossline_byte)


def geometry_metadata(path: Path, inline_byte: int | None = None, crossline_byte: int | None = None) -> dict[str, Any]:
    candidates = []
    for layout in GEOMETRY_LAYOUTS:
        index = _index(path, layout["inline_byte"], layout["crossline_byte"])
        score = _score_geometry(index["inline"], index["crossline"], layout["inline_byte"] == 189)
        candidates.append({**layout, "score": round(score, 3), "inline_range": [min(index["inlines"], default=0), max(index["inlines"], default=0)], "crossline_range": [min(index["crosslines"], default=0), max(index["crosslines"], default=0)]})
    candidates.sort(key=lambda item: item["score"], reverse=True)
    if inline_byte is not None or crossline_byte is not None:
        if inline_byte is None or crossline_byte is None:
            raise ValueError("Both inline_byte and crossline_byte must be supplied together")
        selected = next((item for item in candidates if item["inline_byte"] == inline_byte and item["crossline_byte"] == crossline_byte), None)
        if selected is None:
            manual_index = _index(path, inline_byte, crossline_byte)
            selected = {"label": "Manual mapping", "inline_byte": inline_byte, "crossline_byte": crossline_byte,
                        "score": round(_score_geometry(manual_index["inline"], manual_index["crossline"], False), 3),
                        "inline_range": [min(manual_index["inlines"], default=0), max(manual_index["inlines"], default=0)],
                        "crossline_range": [min(manual_index["crosslines"], default=0), max(manual_index["crosslines"], default=0)]}
            candidates.append(selected)
        best = selected
    else:
        best = candidates[0]
    index = _index(path, best["inline_byte"], best["crossline_byte"])
    return {
        "trace_count": index["trace_count"], "samples_per_trace": index["samples_per_trace"],
        "sample_interval_us": index["sample_interval_us"], "format_code": index["format_code"],
        "delay_ms": index["delay_ms"], "amplitude_scale": index["amplitude_scale"],
        "inlines": index["inlines"], "crosslines": index["crosslines"],
        "inline_range": best["inline_range"], "crossline_range": best["crossline_range"],
        "geometry_confidence": "High" if best["score"] >= 7 else "Medium" if best["score"] >= 4 else "Low",
        "geometry_candidates": candidates,
        "active_inline_byte": best["inline_byte"], "active_crossline_byte": best["crossline_byte"],
    }


def _regular_positions(values: list[int]) -> list[int]:
    if len(values) < 2:
        return values
    import math as _math
    step = 0
    for a, b in zip(values, values[1:]):
        step = _math.gcd(step, abs(b - a))
    if not step:
        return values
    count = (values[-1] - values[0]) // step + 1
    if count > 100_000:
        raise ValueError("Geometry is excessively sparse; verify inline/crossline header bytes")
    return [values[0] + i * step for i in range(count)]


def read_seismic_section(path: Path, mode: str, target: int, inline_byte: int = 189, crossline_byte: int = 193, max_axis: int = 1024, max_samples: int = 2048) -> dict[str, Any]:
    if mode not in {"inline", "crossline", "timeslice"}:
        raise ValueError("mode must be inline, crossline or timeslice")
    index = _index(path, inline_byte, crossline_byte)
    if mode == "timeslice":
        inlines = index["inlines"]
        crosslines = index["crosslines"]
        inlines = inlines[::max(1, math.ceil(len(inlines) / max_axis))]
        crosslines = crosslines[::max(1, math.ceil(len(crosslines) / max_axis))]
        lookup: dict[tuple[int, int], int] = {}
        duplicates = 0
        for trace_id, (il, xl) in enumerate(zip(index["inline"], index["crossline"])):
            if il and xl:
                if (il, xl) in lookup:
                    duplicates += 1
                else:
                    lookup[(il, xl)] = trace_id
        traces: list[list[float | None]] = []
        with segyio.open(str(path), "r", ignore_geometry=True) as segy:
            if target < 0 or target >= index["samples_per_trace"]:
                raise ValueError("Time-slice sample index is outside the trace sample range")
            for xl in crosslines:
                row = []
                for il in inlines:
                    trace_id = lookup.get((il, xl))
                    value = float(segy.trace[trace_id][target]) if trace_id is not None else float("nan")
                    # JSON has no NaN/Infinity values. Null represents an absent or invalid cell.
                    row.append(value / index["amplitude_scale"] if math.isfinite(value) else None)
                traces.append(row)
        return {"mode": mode, "target": target, "traces": traces, "x_positions": crosslines, "y_positions": inlines, "present": sum(1 for row in traces for value in row if value is not None and math.isfinite(value)), "total": len(inlines) * len(crosslines), "duplicates": duplicates, "sample_count": index["samples_per_trace"], "sample_interval_us": index["sample_interval_us"], "start_ms": index["delay_ms"], "amplitude_scale": index["amplitude_scale"]}

    target_axis = index["inlines"] if mode == "inline" else index["crosslines"]
    lateral_axis = index["crosslines"] if mode == "inline" else index["inlines"]
    if target not in target_axis:
        raise ValueError(f"{mode.title()} {target} is not present in the selected geometry")
    positions = _regular_positions(lateral_axis)
    full_position_count = len(positions)
    lateral_stride = max(1, math.ceil(len(positions) / max_axis))
    positions = positions[::lateral_stride]
    if mode == "crossline":
        positions.reverse()
    sample_stride = max(1, math.ceil(index["samples_per_trace"] / max_samples))
    selected: dict[int, int] = {}
    duplicates = 0
    for trace_id, (il, xl) in enumerate(zip(index["inline"], index["crossline"])):
        if (il if mode == "inline" else xl) != target:
            continue
        lateral = xl if mode == "inline" else il
        if lateral in selected:
            duplicates += 1
        else:
            selected[lateral] = trace_id
    sample_indices = list(range(0, index["samples_per_trace"], sample_stride))
    traces: list[list[float | None]] = [[None] * len(sample_indices) for _ in positions]
    delays: set[int] = set()
    with segyio.open(str(path), "r", ignore_geometry=True) as segy:
        for slot, lateral in enumerate(positions):
            trace_id = selected.get(lateral)
            if trace_id is None:
                continue
            raw = np.asarray(segy.trace[trace_id], dtype=np.float64)
            if raw.size != index["samples_per_trace"]:
                raise ValueError("Variable trace lengths detected in the selected section")
            header = segy.header[trace_id]
            try:
                delays.add(int(header[109] or 0))
            except (KeyError, ValueError, TypeError, AttributeError):
                delays.add(0)
            traces[slot] = [
                float(raw[sample] / index["amplitude_scale"])
                if math.isfinite(float(raw[sample])) else None
                for sample in sample_indices
            ]
    if len(delays) > 1:
        raise ValueError("Selected section contains varying recording delays; time alignment is required")
    start_ms = next(iter(delays), index["delay_ms"])
    return {"mode": mode, "target": target, "traces": traces, "x_positions": positions, "present": len(selected), "total": full_position_count, "duplicates": duplicates, "sample_count": len(sample_indices), "sample_stride": sample_stride, "sample_interval_us": index["sample_interval_us"], "start_ms": start_ms, "amplitude_scale": index["amplitude_scale"]}
