"""Canonicalize common GeoChat domain terms before skill execution."""
from __future__ import annotations

import re
from typing import Any


_DATASET_ALIASES = {
    "seismic": {
        "seismic", "seismic data", "seismic volume", "seismic cube",
        "3d seismic", "2d seismic", "segy", "seg-y", "sgy", "segy file",
        "segy files", "sgy file", "seismic file", "seismic files",
    },
    "well_log": {
        "well_log", "well log", "well logs", "well logging", "wireline log",
        "wireline logs", "log data", "well log data", "las", "las data", "las file", "las files",
        "petrophysical logs",
    },
    "checkshot": {
        "checkshot", "check shot", "check-shot", "checkshots",
        "check shot survey", "checkshot data", "time-depth data",
        "time depth data", "t-d survey",
    },
    "deviation": {
        "deviation", "well deviation", "deviation survey", "directional survey",
        "directional data", "borehole trajectory", "well trajectory",
    },
}


def _normalize_term(value: str) -> str:
    value = value.casefold().strip()
    value = re.sub(r"[_\-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value


_ALIAS_TO_KIND = {
    _normalize_term(alias): kind
    for kind, aliases in _DATASET_ALIASES.items()
    for alias in aliases | {kind}
}


def canonical_dataset_kind(value: str) -> str | None:
    """Return a supported database dataset kind for a known alias."""
    return _ALIAS_TO_KIND.get(_normalize_term(value))


def normalize_skill_parameters(skill_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Normalize known user-facing dataset aliases in skill parameters."""
    normalized = dict(parameters)
    if skill_name != "data_inventory":
        return normalized

    data_type = normalized.get("data_type")
    if isinstance(data_type, str):
        canonical = canonical_dataset_kind(data_type)
        if canonical:
            normalized["data_type"] = canonical
    elif isinstance(data_type, list):
        canonical_types = [
            canonical_dataset_kind(item) if isinstance(item, str) else None
            for item in data_type
        ]
        normalized["data_type"] = list(dict.fromkeys(
            item for item in canonical_types if item is not None
        ))
    return normalized
