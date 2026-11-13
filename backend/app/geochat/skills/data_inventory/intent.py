"""Deterministic interpretation of common project-inventory questions."""
from __future__ import annotations

import re
from typing import Any


_WELL_DATA_TYPES = ["well_log", "deviation", "checkshot"]


def parse_inventory_request(message: str) -> dict[str, Any] | None:
    """Return inventory filters for a clear inventory question, else ``None``.

    Broad ``well data`` means every well-associated dataset type, not only LAS
    logs. A specific type mentioned by the user takes precedence.
    """
    text = re.sub(r"\s+", " ", (message or "").casefold()).strip()
    if not text:
        return None

    action = bool(re.search(
        r"\b(what|which|list|show|display|available|availability|inventory|"
        r"registered|uploaded|have|has|any|exist|exists|are there|is there)\b",
        text,
    ))
    subject = bool(re.search(
        r"\b(data|dataset|datasets|file|files|well|wells|survey|surveys|"
        r"log|logs|las|segy|sgy|check.?shot|deviation)\b",
        text,
    ))
    project_inventory_phrase = bool(re.search(
        r"\b(what('?s| is) in|contents? of) (this|the|my) project\b", text
    ))
    well_inventory_fragment = bool(re.search(r"\bwell[ -]+(data|datasets?|files?)\b", text))
    if not ((action and subject) or project_inventory_phrase or well_inventory_fragment):
        return None
    # "What is well-log data?" is a conceptual question, not an inventory query.
    if re.match(r"^what\s+(?:is|are)\b", text) and not re.search(
        r"\b(available|availability|registered|uploaded|have|exists?|inventory|list|show)\b",
        text,
    ):
        return None

    filters: dict[str, Any] = {}
    # Specific dataset types take precedence over the broader phrase "well data".
    if re.search(r"\b(check.?shots?|time[ -]?depth data|t[ -]?d survey)\b", text):
        filters["data_type"] = "checkshot"
    elif re.search(r"\b(deviation|directional survey|well trajectory|borehole trajectory)\b", text):
        filters["data_type"] = "deviation"
    elif re.search(r"\b(well[ -]?logs?|wireline logs?|petrophysical logs?|las(?: files?| data)?)\b", text):
        filters["data_type"] = "well_log"
    elif re.search(r"\b(seismic|segy|sgy|2d seismic|3d seismic)\b", text):
        filters["data_type"] = "seismic"
    elif re.search(r"\bwell[ -]+(data|datasets?|files?)\b|\b(data|datasets?|files?)\s+for\s+(?:the\s+)?wells?\b", text):
        filters["data_type"] = list(_WELL_DATA_TYPES)
    elif re.search(r"\blogs?\b", text):
        filters["data_type"] = "well_log"

    # Well IDs such as M-1 may be mentioned with or without the word "well".
    well_match = re.search(r"\b(?:well\s+)?([a-z]{1,3}-\d+[a-z]?)\b", text)
    if well_match:
        filters["well_name"] = well_match.group(1)

    survey_match = re.search(r"\bsurvey\s+([a-z0-9][a-z0-9_-]*)\b", text)
    if survey_match and survey_match.group(1) not in {
        "data", "dataset", "datasets", "file", "files", "available", "inventory"
    }:
        filters["survey_name"] = survey_match.group(1)

    return filters


def parse_well_name_filter(message: str) -> str | None:
    """Extract a well identifier, never an arbitrary word after 'well'."""
    text = re.sub(r"\s+", " ", (message or "").casefold()).strip()
    match = re.search(r"\b(?:well\s+)?([a-z]{1,3}-\d+[a-z]?)\b", text)
    return match.group(1) if match else None
