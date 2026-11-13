"""Deterministic Markdown formatting for well and survey inventories."""
from __future__ import annotations

from typing import Any


def render_markdown(result: dict[str, Any]) -> str:
    entity_type = result.get("entity_type")
    items = result.get("items") or []
    count = int(result.get("count", len(items)))
    label = "Wells" if entity_type == "wells" else "Surveys"
    singular = "well" if entity_type == "wells" else "survey"

    if not items:
        return (
            f"**Project {label.lower()}**\n\n"
            f"No {label.lower()} are registered in this project.\n\n"
            f"**Total: 0 {label.lower()}**"
        )

    if entity_type == "wells":
        lines = ["**Project wells**", "", "| Well | UWI |", "|---|---|"]
        for item in items:
            name = _cell(item.get("name"))
            uwi = _cell(item.get("uwi") or "—")
            lines.append(f"| {name} | {uwi} |")
    else:
        lines = ["**Project surveys**", "", "| Survey | Description | CRS |", "|---|---|---|"]
        for item in items:
            name = _cell(item.get("name"))
            description = _cell(item.get("description") or "—")
            crs = _cell(item.get("crs") or "—")
            lines.append(f"| {name} | {description} | {crs} |")

    lines.extend(["", f"**Total: {count} {singular}{'s' if count != 1 else ''}**"])
    return "\n".join(lines)


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()
