"""Stable Markdown presentation for authoritative dataset inventory results."""
from __future__ import annotations
from typing import Any

from app.geochat.skills.data_inventory.descriptions import DEFAULT_DESCRIPTIONS


_KIND_LABELS = {
    "seismic": "Seismic",
    "well_log": "Well Log",
    "deviation": "Deviation",
    "checkshot": "Checkshot",
}


def render_markdown(result: dict[str, Any]) -> str:
    """Render an inventory result in one stable, compact Markdown table."""
    datasets = result.get("datasets") or []
    count = int(result.get("count", len(datasets)))
    if not datasets:
        return (
            "**Dataset inventory**\n\n"
            "No matching datasets are registered in this project.\n\n"
            "**Total: 0 datasets**"
        )

    lines = [
        "**Dataset inventory**",
        "",
        "| Dataset | Type | Well / Survey | Description |",
        "|---|---|---|---|",
    ]
    for dataset in datasets:
        filename = str(dataset.get("filename") or dataset.get("name") or "Unnamed dataset")
        # Preserve the filename as a distinct inline-code token for frontend matching.
        filename = filename.replace("|", "\\|").replace("`", "\\`")
        kind = _KIND_LABELS.get(
            str(dataset.get("kind")), str(dataset.get("kind") or "Unknown")
        )
        association = str(dataset.get("well") or dataset.get("survey") or "—")
        association = association.replace("|", "\\|").replace("\n", " ")
        description = str(dataset.get("description") or "").strip()
        if not description:
            description = DEFAULT_DESCRIPTIONS.get(str(dataset.get("kind")), "Geoscience dataset")
        description = description.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{filename}` | {kind} | {association} | {description} |")
    lines.extend(["", f"**Total: {count} dataset{'s' if count != 1 else ''}**"])
    return "\n".join(lines)
