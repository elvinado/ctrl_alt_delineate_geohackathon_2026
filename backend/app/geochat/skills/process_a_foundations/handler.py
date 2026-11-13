"""Pre-flight validation for Process A (GeoGG foundations, stages 00-07).

This handler checks that the datasets proposed for a run cover what the process needs and returns
a submission plan. It does NOT submit the job: submission must go through the approved backend
processing service after the user confirms. It never executes user-provided code and never opens
the data files; it reasons only over the dataset descriptors it is given.
"""
from __future__ import annotations

import re
from typing import Any

SKILL_NAME = "process_a_foundations"
PYTHON_SCRIPT = "process_A_foundations.py"
OPERATION = "geogg_fresh_stages_00_07"

MIN_WELLS = 4            # four whole-well test groups need at least four wells
RECOMMENDED_WELLS = 8    # below this, test groups hold one well each and scores swing widely
RECOMMENDED_TZ_TABLES = 3

# Seismic roles, recognised from the dataset name. Order matters: 'ultrafar' before 'far'.
SEISMIC_ROLES: list[tuple[str, tuple[str, ...], str]] = [
    ("ultrafar", ("ultrafar",), "partial stack"),
    ("near", ("nearstack", "near"), "partial stack"),
    ("mid", ("midstack", "mid"), "partial stack"),
    ("far", ("farstack", "far"), "partial stack"),
    ("full", ("fullstack", "full"), "full stack"),
    ("pimp", ("pimpedance", "pimp"), "inversion"),
    ("simp", ("simpedance", "simp"), "inversion"),
    ("vpvs", ("vpvs",), "inversion"),
    ("totalsandprob", ("totalsandprob", "sandprob"), "excluded"),
]
PARTIALS = ("near", "mid", "far", "ultrafar")
INVERSION = ("pimp", "simp", "vpvs")

REQUIREMENTS: list[dict[str, str]] = [
    {"input": "Well list table (.xlsx)", "need": "required, exactly one",
     "why": "Gives each well's name, surface X/Y, rotary-table elevation and total depth."},
    {"input": "Well logs (.las), one per well", "need": f"required, at least {MIN_WELLS} wells",
     "why": "Interpreted VSH, PHIT and SWT curves become the answers the models learn from."},
    {"input": "Deviation survey (.txt), one per well", "need": "required for every well that has logs",
     "why": "Turns measured depth into true vertical depth and map position along the well path."},
    {"input": "Time-depth or checkshot tables (.txt)", "need": f"at least 1, {RECOMMENDED_TZ_TABLES} or more recommended",
     "why": "Anchor the depth-to-time conversion. Wells without one use a regional relation fitted to the others."},
    {"input": "Full-stack seismic (.sgy)", "need": "required",
     "why": "Defines the survey grid and is the reference volume for every seismic cut-out."},
    {"input": "Near, mid, far and ultra-far stacks (.sgy)", "need": "recommended",
     "why": "Provide the amplitude-versus-angle contrast attributes."},
    {"input": "Inversion volumes: P-impedance, S-impedance, Vp/Vs (.sgy)", "need": "optional",
     "why": "Used as conditional inputs for the stacks-plus-inversion comparison."},
]

WHAT_HAPPENS: list[str] = [
    "00 Data check-in: confirms every file is present, readable and on one grid.",
    "01 Depth-to-time conversion: traces well paths and places logs in seismic time.",
    "02 Rock and fluid labels: turns interpreted logs into sand, shale and fluid answers.",
    "03 Seismic cut-outs: extracts a small seismic cube at every point along each well.",
    "04 Seismic attributes: measures amplitude, frequency, continuity and contrast.",
    "05 Training tables: joins inputs with answers and assigns whole wells to test groups.",
    "06 Attribute selection: keeps attributes that behave consistently from well to well.",
    "07 Model training: tunes tree models and two small deep networks, and packages a model release.",
]


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(text or "").lower())


def _seismic_role(name: Any) -> str | None:
    key = _norm(name)
    for role, tokens, _ in SEISMIC_ROLES:
        if any(token in key for token in tokens):
            return role
    return None


def _issue(level: str, code: str, message: str, fix: str) -> dict[str, str]:
    """level: error (blocks the run) | warning (run allowed, quality reduced)."""
    return {"level": level, "code": code, "message": message, "how_to_fix": fix}


def validate(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Check dataset descriptors (dataset_id, kind, format, name, well_name, survey_name, metadata)."""
    issues: list[dict[str, str]] = []
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for item in datasets:
        if isinstance(item, dict):
            by_kind.setdefault(str(item.get("kind") or "").lower(), []).append(item)

    def wells(kind: str) -> dict[str, list[str]]:
        found: dict[str, list[str]] = {}
        for item in by_kind.get(kind, []):
            name = str(item.get("well_name") or "").strip()
            if name:
                found.setdefault(name, []).append(str(item.get("dataset_id")))
            else:
                issues.append(_issue("error", f"{kind}_without_well",
                                     f"'{item.get('name')}' is not linked to a well.",
                                     "Associate the dataset with its well in the project, then try again."))
        for name, ids in found.items():
            if len(ids) > 1:
                issues.append(_issue("error", f"duplicate_{kind}",
                                     f"Well {name} has {len(ids)} {kind.replace('_', ' ')} datasets.",
                                     "Select one per well so the process knows which to use."))
        return found

    well_lists = [i for i in by_kind.get("table", [])
                  if str((i.get("metadata") or {}).get("table_type") or "").lower() == "well_list"]
    if len(well_lists) != 1:
        issues.append(_issue("error", "well_list",
                             f"{len(well_lists)} well list tables were selected; exactly one is needed.",
                             "Select the single well list workbook that holds name, RT, TD, X and Y."))
    logs, deviations, tz = wells("well_log"), wells("deviation"), wells("checkshot")
    if len(logs) < MIN_WELLS:
        issues.append(_issue("error", "too_few_wells",
                             f"{len(logs)} wells have logs; the process needs at least {MIN_WELLS}.",
                             "Add more wells with LAS logs."))
    elif len(logs) < RECOMMENDED_WELLS:
        issues.append(_issue("warning", "few_wells",
                             f"Only {len(logs)} wells have logs.",
                             f"The run will work, but with fewer than {RECOMMENDED_WELLS} wells each test group is tiny and scores are unstable."))
    missing_dev = sorted(set(logs) - set(deviations))
    if missing_dev:
        issues.append(_issue("error", "missing_deviation",
                             f"No deviation survey for: {', '.join(missing_dev)}.",
                             "Add a deviation survey for each of these wells, or remove their logs from the selection."))
    orphan_dev = sorted(set(deviations) - set(logs))
    if orphan_dev:
        issues.append(_issue("warning", "deviation_without_logs",
                             f"Deviation survey but no logs for: {', '.join(orphan_dev)}.",
                             "These wells cannot contribute labels. Add their logs or ignore this."))
    if not tz:
        issues.append(_issue("error", "no_time_depth",
                             "No time-depth or checkshot table was selected.",
                             "Add at least one. Without it the logs cannot be placed on the seismic at all."))
    elif len(tz) < RECOMMENDED_TZ_TABLES:
        issues.append(_issue("warning", "few_time_depth",
                             f"Only {len(tz)} time-depth table(s) were selected.",
                             f"With fewer than {RECOMMENDED_TZ_TABLES} the regional depth-to-time relation cannot be cross-checked well against well."))

    roles: dict[str, list[dict[str, Any]]] = {}
    unknown: list[str] = []
    for item in by_kind.get("seismic", []):
        role = _seismic_role(item.get("name"))
        if role:
            roles.setdefault(role, []).append(item)
        else:
            unknown.append(str(item.get("name")))
    for role, items in roles.items():
        if len(items) > 1:
            issues.append(_issue("error", "duplicate_seismic",
                                 f"{len(items)} volumes look like the '{role}' volume: {', '.join(str(i.get('name')) for i in items)}.",
                                 "Select only one volume per role."))
    if "full" not in roles:
        issues.append(_issue("error", "no_full_stack", "No full-stack seismic volume was selected.",
                             "Add the full-stack SEG-Y volume; it defines the survey grid."))
    missing_partials = [r for r in PARTIALS if r not in roles]
    if missing_partials:
        issues.append(_issue("warning", "missing_partial_stacks",
                             f"Partial stacks not selected: {', '.join(missing_partials)}.",
                             "The run will have fewer amplitude-contrast attributes. Add them if they exist."))
    missing_inversion = [r for r in INVERSION if r not in roles]
    if missing_inversion:
        issues.append(_issue("warning", "missing_inversion",
                             f"Inversion volumes not selected: {', '.join(missing_inversion)}.",
                             "The stacks-plus-inversion comparison will be incomplete or skipped."))
    if unknown:
        issues.append(_issue("warning", "unrecognised_seismic",
                             f"Volume name(s) not recognised: {', '.join(unknown)}.",
                             "Rename so the role is clear (for example 'Near_Stack', 'P_Impedance'), or confirm they can be ignored."))
    surveys = {str(i.get("survey_name")) for i in by_kind.get("seismic", []) if i.get("survey_name")}
    if len(surveys) > 1:
        issues.append(_issue("error", "mixed_surveys",
                             f"Seismic volumes come from different surveys: {', '.join(sorted(surveys))}.",
                             "Select volumes from one survey; they must share the same grid."))

    errors = [i for i in issues if i["level"] == "error"]
    return {
        "ready": not errors,
        "issues": issues,
        "selection": {
            "well_list_tables": len(well_lists),
            "wells_with_logs": sorted(logs), "wells_with_deviation": sorted(deviations),
            "wells_with_time_depth": sorted(tz),
            "seismic_roles": {role: str(items[0].get("name")) for role, items in sorted(roles.items())},
            "note_on_sand_probability": (
                "The Total Sand Probability volume is read for the inventory but never used as a model input, "
                "because it is too close to the answer being predicted." if "totalsandprob" in roles else None),
        },
        "dataset_ids": [str(i.get("dataset_id")) for i in datasets if isinstance(i, dict) and i.get("dataset_id")],
    }


def execute(*, project_id: str, parameters: dict[str, Any], db: Any) -> dict[str, Any]:
    """Validate the proposed inputs for Process A and return a plan for the user to confirm.

    parameters["datasets"]: list of dataset descriptors as registered in the project, each with
    dataset_id, kind (table | well_log | deviation | checkshot | seismic), format, name, well_name,
    survey_name and metadata. The process takes no numeric parameters.
    """
    base = {
        "skill": SKILL_NAME, "project_id": project_id, "python_script": PYTHON_SCRIPT, "operation": OPERATION,
        "target_well": None, "submitted": False, "confirmation_required": True,
        "requirements": REQUIREMENTS,
    }
    datasets = parameters.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return {**base, "status": "needs_inputs",
                "summary": "No datasets were provided to check. Process A needs the inputs listed under 'requirements'.",
                "issues": [], "what_will_happen": WHAT_HAPPENS}
    report = validate(datasets)
    errors = sum(1 for i in report["issues"] if i["level"] == "error")
    warnings = len(report["issues"]) - errors
    selection = report["selection"]
    summary = (
        f"{'Ready to run' if report['ready'] else 'Not ready'}: {len(selection['wells_with_logs'])} wells with logs, "
        f"{len(selection['wells_with_time_depth'])} time-depth table(s), {len(selection['seismic_roles'])} seismic volume(s); "
        f"{errors} blocking issue(s), {warnings} warning(s).")
    return {
        **base, "status": "ready_for_confirmation" if report["ready"] else "not_ready", "summary": summary,
        "issues": report["issues"], "selection": selection, "dataset_ids": report["dataset_ids"],
        "what_will_happen": WHAT_HAPPENS,
        "expected_outputs": [
            "geogg_foundations_summary.json: status, self-checks and limitations of every stage",
            "stages/00_inventory ... stages/07_models: tables and figures per stage",
            "provenance/: one record per stage with input and output checksums",
            "a packaged model release for the next process",
        ],
        "next_action": (
            "Show this plan to the user. After they confirm, submit the script and dataset IDs through the approved "
            "processing service and report the returned job ID and status." if report["ready"] else
            "Explain each blocking issue and how to fix it. Do not submit."),
    }
