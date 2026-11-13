"""Recognize requests to list registered project wells or surveys."""
from __future__ import annotations

import re


def parse_entity_inventory_request(message: str) -> dict[str, str] | None:
    """Return an entity-list request without capturing dataset-inventory questions."""
    text = re.sub(r"\s+", " ", (message or "").casefold()).strip()
    if not text:
        return None

    # These are dataset questions, not requests to enumerate entity records.
    if re.search(r"\b(data|dataset|datasets|file|files|logs?|las|segy|sgy|check.?shots?|deviation|trajectory|curves?)\b", text):
        return None

    asks_to_list = bool(re.search(
        r"\b(what|which|list|show|display|available|availability|have|has|any|exist|exists|are there|is there)\b",
        text,
    ))
    if not asks_to_list:
        return None

    if re.search(r"\bsurveys?\b", text):
        return {"entity_type": "surveys"}
    if re.search(r"\b(wells?|boreholes?)\b", text):
        return {"entity_type": "wells"}
    return None
