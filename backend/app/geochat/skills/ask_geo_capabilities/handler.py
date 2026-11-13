"""Build Ask Geo capabilities from the live skill registry."""
from __future__ import annotations

from typing import Any


def execute(*, db, project_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
    from app.geochat.skills.registry import SKILL_REGISTRY

    capabilities = [
        {"name": skill.name, "description": skill.description}
        for skill in SKILL_REGISTRY.values()
        if skill.name != "ask_geo_capabilities"
    ]
    process_skills = [item for item in capabilities if item["name"].startswith("process_")]
    examples = [
        "What datasets and wells are available in this project?",
        "What processing scripts and workflows are available?",
        "Run <process workflow> using <required input>.",
        "Analyze the output from processing job <job ID>.",
        "Explain a geoscience concept relevant to my project.",
    ]
    if not process_skills:
        examples.remove("Run <process workflow> using <required input>.")
    if any(item["name"].endswith("_output_analysis") for item in process_skills):
        # Result readers answer by process name and pick the latest run themselves.
        examples.insert(-1, "Explain the results of <process workflow>, or of each process.")
    return {
        "capabilities": capabilities,
        "frontend_actions": [
            {"name": "upload", "description": "Open the Input Data upload dialog for geological files."},
            {"name": "view_seismic", "description": "Navigate to the Seismic Viewer for supported SEG-Y previews."},
            {"name": "view_well", "description": "Navigate to the Well Viewer for LAS curves and deviation/checkshot TXT tables."},
        ],
        "examples": examples,
        "limitations": [
            "Project-specific data and processing operations require a selected project.",
            "Processing discovery is read-only; discovering a script does not guarantee its dependencies or compute environment are ready.",
            "Only registered skills and explicitly implemented viewer actions are available; arbitrary code execution is not supported.",
        ],
    }
