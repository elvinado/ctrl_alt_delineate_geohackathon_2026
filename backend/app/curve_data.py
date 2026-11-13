"""Read numerical samples for catalogued LAS curves."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import lasio

from app.models import Curve


def _json_number(value: Any) -> float | int | None:
    """Convert a scientific scalar to a JSON-safe number or null."""
    if hasattr(value, "item"):
        value = value.item()
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def read_curve_samples(
    path: Path,
    curve: Curve,
    *,
    offset: int = 0,
    limit: int = 1000,
    index_min: float | None = None,
    index_max: float | None = None,
) -> dict[str, Any]:
    """Read a bounded page of samples from a LAS curve by its catalogued position.

    The LAS file remains the source of truth; this function does not copy sample
    values into the relational database.
    """
    try:
        las = lasio.read(str(path), ignore_data=False, engine="normal")
    except Exception as exc:
        raise ValueError(f"Unable to read LAS curve data: {type(exc).__name__}: {exc}") from exc

    data = las.data
    if data is None or len(data) == 0:
        rows = []
    else:
        if curve.position < 0 or curve.position >= data.shape[1]:
            raise ValueError(
                f"Curve position {curve.position} is not present in the source LAS file"
            )
        index_values = list(las.index)
        values = data[:, curve.position]
        rows = []
        for index_value, sample_value in zip(index_values, values, strict=False):
            index_number = _json_number(index_value)
            if index_number is None:
                continue
            if index_min is not None and index_number < index_min:
                continue
            if index_max is not None and index_number > index_max:
                continue
            rows.append(
                {
                    "index": index_number,
                    "value": _json_number(sample_value),
                }
            )

    page = rows[offset : offset + limit]

    # lasio exposes ``las.index`` as a NumPy array, not a pandas Series, so it
    # has no ``name`` attribute. In LAS, the index curve is the first curve
    # definition; use its mnemonic and unit for response metadata.
    index_curve = las.curves[0] if las.curves else None
    index_mnemonic = (
        str(getattr(index_curve, "mnemonic", "") or "").strip() or None
    )
    index_unit = str(getattr(index_curve, "unit", "") or "").strip() or None

    return {
        "curve_id": curve.id,
        "dataset_id": curve.dataset_id,
        "mnemonic": curve.mnemonic,
        "unit": curve.unit,
        "index": {
            "mnemonic": index_mnemonic,
            "unit": index_unit,
        },
        "total_samples": len(rows),
        "offset": offset,
        "limit": limit,
        "returned_samples": len(page),
        "samples": page,
    }
