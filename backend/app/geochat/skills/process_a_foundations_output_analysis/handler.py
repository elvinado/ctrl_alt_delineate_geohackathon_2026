"""Expert, deterministic reading of one Process A (GeoGG foundations, stages 00-07) run folder.

The handler measures and applies fixed, documented reading rules. It never calls an LLM and never
invents a value: every number in the result is read from a file inside the job folder, and every
file it used is listed under ``artifacts``. The GeoChat agent explains the returned evidence.

This module is self-contained on purpose: it does not import the generic
``processing_output_folder_analysis`` handler or any other skill.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

SKILL_NAME = "process_a_foundations_output_analysis"
EXPECTED_OPERATION = "geogg_fresh_stages_00_07"
EXPECTED_SCRIPT = "process_A_foundations.py"
SUMMARY_FILE = "geogg_foundations_summary.json"

MAX_JSON_BYTES = 8_000_000
MAX_CSV_BYTES = 30_000_000
MAX_CSV_ROWS = 60_000
MAX_TEXT = 420
OVERVIEW_FINDINGS = 12

# --------------------------------------------------------------------------------------------
# Static knowledge about the process (what each stage is for, in plain words)
# --------------------------------------------------------------------------------------------
STAGES: dict[str, dict[str, str]] = {
    "00_inventory": {
        "title": "Data check-in",
        "plain": "Confirms every well file and seismic volume is present, readable and on one common grid.",
        "section": "data",
    },
    "01_well_ties": {
        "title": "Depth-to-time conversion",
        "plain": "Traces each well path and converts log depths into seismic two-way time, so logs and seismic line up.",
        "section": "ties",
    },
    "02_labels": {
        "title": "Rock and fluid labels",
        "plain": "Turns the interpreted logs (shale volume, porosity, water saturation) into the answers the models learn from.",
        "section": "labels",
    },
    "03_extraction": {
        "title": "Seismic cut-outs at the wells",
        "plain": "Cuts a small seismic cube around each point of every well path and pairs it with its label.",
        "section": "training",
    },
    "04_features": {
        "title": "Seismic attributes",
        "plain": "Measures descriptors of each cut-out: amplitude, frequency, continuity, near-versus-far contrast, impedance.",
        "section": "features",
    },
    "05_training_tables": {
        "title": "Training tables and fair-test plan",
        "plain": "Joins inputs with answers and assigns whole wells to test groups, so no well is both studied and examined.",
        "section": "training",
    },
    "06_feature_selection": {
        "title": "Which attributes matter",
        "plain": "Tests which attributes relate to the rock consistently from well to well, not just inside one well.",
        "section": "features",
    },
    "07_models": {
        "title": "Model training",
        "plain": "Trains and tunes the candidate models. Scores here are practice scores, not the final exam.",
        "section": "models",
    },
}

FOCUS_ALIASES: dict[str, str] = {
    "": "overview", "overview": "overview", "summary": "overview", "all": "all", "full": "all",
    "data": "data", "inventory": "data", "inputs": "data", "seismic": "data", "00": "data", "0": "data",
    "ties": "ties", "well_ties": "ties", "tie": "ties", "time_depth": "ties", "tz": "ties", "01": "ties", "1": "ties",
    "labels": "labels", "classes": "labels", "petrophysics": "labels", "02": "labels", "2": "labels",
    "training": "training", "extraction": "training", "tables": "training", "folds": "training",
    "03": "training", "3": "training", "05": "training", "5": "training",
    "features": "features", "attributes": "features", "selection": "features",
    "04": "features", "4": "features", "06": "features", "6": "features",
    "models": "models", "model": "models", "scores": "models", "performance": "models", "07": "models", "7": "models",
    "glossary": "glossary", "figures": "figures", "plots": "figures",
}

TASK_NAMES: dict[str, str] = {
    "lithology": "sand versus shale",
    "sand_fluid": "water-bearing versus hydrocarbon-potential sand",
    "vsh": "shale volume (VSH)",
    "phit": "total porosity (PHIT)",
    "swt": "total water saturation (SWT)",
}

INPUT_NAMES: dict[str, str] = {
    "stacks": "seismic stacks only, all rows",
    "stacks@matched": "seismic stacks only, rows that also have inversion data",
    "fused@matched": "stacks plus inversion volumes, rows that have inversion data",
    "patch_stacks": "raw seismic cut-outs (deep learning), all rows",
}

GLOSSARY: dict[str, str] = {
    "CONDITIONAL": "The stage ran and passed its checks, but its results depend on stated assumptions. It is not a failure.",
    "check": "An automatic test the pipeline runs on itself, for example 'no test well was used for training'.",
    "limitation": "A known weakness of the inputs that the software cannot fix, stated so nobody over-trusts the result.",
    "TWT": "Two-way time: how long sound takes to travel down to a layer and back, in milliseconds. Seismic is recorded in TWT.",
    "time-depth (T-Z) relation": "The conversion between depth in a well and seismic time. Without it logs cannot be placed on seismic.",
    "checkshot": "A direct measurement of travel time at known depths in a well. The most reliable basis for a T-Z relation.",
    "well tie": "Proof that a well's logs line up with the seismic, normally by matching a synthetic trace built from sonic and density logs.",
    "LOWO": "Leave-one-well-out: fit using the other wells, then measure the error on the well left out.",
    "VSH": "Shale volume, a fraction from 0 (clean sand) to 1 (pure shale).",
    "PHIT": "Total porosity: the fraction of the rock that is pore space.",
    "SWT": "Total water saturation: the fraction of the pore space filled with water. Low values hint at hydrocarbons.",
    "hydrocarbon potential": "Sand whose interpreted water saturation is low. It is an indication only; it does not say oil or gas.",
    "partial stacks": "Near, mid, far and ultra-far seismic volumes built from different reflection angles.",
    "inversion volumes": "P-impedance, S-impedance and Vp/Vs cubes derived from seismic by a separate modelling step.",
    "matched rows": "The rows where inversion data exist. Stacks-only and stacks-plus-inversion models are compared on these same rows.",
    "outer fold": "A group of whole wells set aside as a test group. Each well is in exactly one outer fold.",
    "inner CV": "Practice testing done only inside the training wells to choose settings. Slightly optimistic by design.",
    "ROC-AUC": "A 0.5-to-1 score for yes/no predictions. 0.5 is a coin flip, 1.0 is perfect.",
    "RMSE": "Typical size of the prediction error, in the units of the property (fractions here).",
    "R2": "Share of the variation explained. 0 means no better than always guessing the average; it can be negative.",
    "depth trend": "Rock properties change steadily with burial depth. A model can score well by learning depth alone.",
    "feature / attribute": "One measured number that describes the seismic at a point, used as a model input.",
    "champion": "The best candidate for one task and input set according to inner CV.",
}

FIGURES: list[tuple[str, str]] = [
    ("stages/00_inventory/figures/well_locations_map.png", "Where the wells sit inside the seismic survey."),
    ("stages/00_inventory/figures/data_availability_matrix.png", "Which data type exists for which well."),
    ("stages/01_well_ties/figures/all_wells_twt_vs_tvdss.png", "Time-depth curves of all wells. They should form one tight, smooth band."),
    ("stages/01_well_ties/figures/measured_vs_provisional_comparison.png", "Supplied time-depth tables against the fitted regional relation."),
    ("stages/01_well_ties/figures/well_trajectories_map.png", "Well paths in map view, showing which wells are deviated."),
    ("stages/02_labels/figures/class_distribution_by_well.png", "How much shale, sand and unlabelled rock each well contributes."),
    ("stages/02_labels/figures/rock_physics_crossplots.png", "VSH, PHIT and SWT against each other, coloured by label."),
    ("stages/03_extraction/figures/label_time_alignment.png", "Labels placed on the seismic trace at each well."),
    ("stages/03_extraction/figures/extraction_coverage_by_well.png", "How many seismic samples were cut out for each well."),
    ("stages/04_features/figures/feature_distributions_by_well.png", "Whether attributes look alike from well to well."),
    ("stages/05_training_tables/figures/train_test_well_map.png", "Which wells form each test group."),
    ("stages/05_training_tables/figures/target_coverage_heatmap.png", "Where each target has usable answers."),
    ("stages/06_feature_selection/figures/feature_selection_ranks.png", "Selected attributes with their well-to-well effect and interval."),
    ("stages/06_feature_selection/figures/per_well_effects_heatmap.png", "Whether an attribute points the same way in every well."),
    ("stages/07_models/figures/model_comparison_inner_cv.png", "All candidates side by side on the practice score."),
    ("stages/07_models/figures/learning_curves_deep.png", "Training against validation loss for the deep models."),
]

SUGGESTED_QUESTIONS: list[str] = [
    "Is this run good enough to move to the next process?",
    "How well can the models tell sand from shale?",
    "Does adding the inversion volumes help?",
    "Why are there no oil or gas labels?",
    "How reliable is the depth-to-time conversion?",
    "Which seismic attributes matter most for porosity?",
    "Tell me about one well, for example its test group and label counts.",
    "What should we fix first to improve the results?",
]

# Order in which findings are surfaced (first match on the id prefix wins).
PRIORITY: list[str] = [
    "ties_validated", "inner_cv_only", "model_", "coverage_gap_", "no_depth_baseline", "depth_trend_features",
    "no_oil_gas_labels", "time_uncertainty", "inversion_coverage", "inversion_", "no_sonic_density",
    "label_policy_provisional", "nothing_selected", "fluid_sample_size", "informative_missingness", "avo_qualitative",
    "physical_clipping", "tz_extrapolation", "label_balance", "data_complete",
]
TAKEAWAY_IDS: list[str] = [
    "ties_validated", "inner_cv_only", "no_depth_baseline", "coverage_gap_", "no_oil_gas_labels", "time_uncertainty",
]

RATING_NOTE = (
    "Ratings are reading conventions, not measurements: ROC-AUC below 0.55 none, below 0.65 weak, below 0.75 modest, "
    "below 0.85 good, otherwise strong; R2 at or below 0.02 none, below 0.20 weak, below 0.50 moderate, otherwise strong."
)


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------
def _num(value: Any) -> float | None:
    """Finite float or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _r(value: Any, digits: int = 3) -> float | None:
    number = _num(value)
    return None if number is None else round(number, digits)


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _clip(text: Any, limit: int = MAX_TEXT) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _pct(value: Any, digits: int = 0) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number * 100:.{digits}f}%"


def _fmt(value: Any, digits: int = 3) -> str:
    number = _num(value)
    return "n/a" if number is None else f"{number:.{digits}f}"


class _Reader:
    """Reads fixed relative paths inside one job folder, safely, and remembers what it touched."""

    def __init__(self, job_dir: Path) -> None:
        self.job_dir = job_dir.resolve()
        self.used: list[dict[str, Any]] = []
        self.skipped: list[dict[str, str]] = []
        self._seen: set[str] = set()

    def _path(self, relative: str, limit: int) -> Path | None:
        path = self.job_dir / relative
        try:
            if path.is_symlink() or not path.is_file():
                return None
            resolved = path.resolve()
            if self.job_dir not in resolved.parents:
                return None
            size = resolved.stat().st_size
        except OSError as exc:
            self._skip(relative, str(exc))
            return None
        if size > limit:
            self._skip(relative, f"file is {size} bytes, above the {limit} byte inspection limit")
            return None
        if relative not in self._seen:
            self._seen.add(relative)
            self.used.append({"name": relative, "size_bytes": size})
        return resolved

    def _skip(self, relative: str, reason: str) -> None:
        self.skipped.append({"name": relative, "reason": _clip(reason, 240)})

    def exists(self, relative: str) -> bool:
        path = self.job_dir / relative
        try:
            return path.is_file() and not path.is_symlink() and self.job_dir in path.resolve().parents
        except OSError:
            return False

    def size(self, relative: str) -> int | None:
        try:
            return (self.job_dir / relative).stat().st_size if self.exists(relative) else None
        except OSError:
            return None

    def json(self, relative: str) -> dict[str, Any]:
        path = self._path(relative, MAX_JSON_BYTES)
        if path is None:
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._skip(relative, f"not readable as JSON: {exc}")
            return {}
        return value if isinstance(value, dict) else {}

    def table(self, relative: str) -> list[dict[str, str]]:
        path = self._path(relative, MAX_CSV_BYTES)
        if path is None:
            return []
        rows: list[dict[str, str]] = []
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    rows.append(row)
                    if len(rows) >= MAX_CSV_ROWS:
                        self._skip(relative, f"only the first {MAX_CSV_ROWS} rows were read")
                        break
        except (OSError, UnicodeDecodeError, csv.Error) as exc:
            self._skip(relative, f"not readable as CSV: {exc}")
            return []
        return rows


# Which focus each finding belongs to (used to filter findings in a focused answer).
_FINDING_SECTION: dict[str, tuple[str, ...]] = {
    "data": ("data_complete", "no_sonic_density"),
    "ties": ("ties_validated", "time_uncertainty", "tz_extrapolation", "no_sonic_density"),
    "labels": ("label_balance", "no_oil_gas_labels", "label_policy_provisional"),
    "training": ("inversion_coverage", "fluid_sample_size", "time_uncertainty"),
    "features": ("avo_qualitative", "nothing_selected", "depth_trend_features", "informative_missingness"),
    "models": ("no_depth_baseline", "inner_cv_only", "physical_clipping", "depth_trend_features", "nothing_selected"),
}
_SECTION_OF: dict[str, str] = {"model": "models", "inversion": "models", "coverage": "models"}


def _finding(fid: str, level: str, observed: str, meaning: str, evidence: list[str]) -> dict[str, Any]:
    """level: good | info | caution | critical."""
    return {"id": fid, "level": level, "observed": observed, "plain_meaning": meaning, "evidence": evidence}


# --------------------------------------------------------------------------------------------
# Stage cards
# --------------------------------------------------------------------------------------------
def _light(status: str, failed: int, blockers: int) -> str:
    status = (status or "").upper()
    if blockers or failed or status in {"FAILED", "BLOCKED", "ERROR"}:
        return "red"
    if status in {"PASS", "PASSED", "OK", "SUCCEEDED"}:
        return "green"
    if status == "CONDITIONAL":
        return "amber"
    return "grey"


def _dedupe(messages: list[Any], limit: int = 6) -> list[str]:
    """Collapse near-identical messages (same text after the first ':') and cap the list."""
    out: list[str] = []
    seen: dict[str, int] = {}
    for message in messages or []:
        text = _clip(message)
        key = text.split(":", 1)[1].strip()[:60] if ":" in text[:24] else text[:80]
        if key in seen:
            seen[key] += 1
            continue
        seen[key] = 1
        out.append(text)
    repeats = sum(count - 1 for count in seen.values())
    out = out[:limit]
    if repeats:
        out.append(f"(+{repeats} similar message(s) for other volumes or wells)")
    return out


def _stage_records(reader: _Reader, summary: dict[str, Any]) -> list[dict[str, Any]]:
    stages = [s for s in summary.get("stages", []) if isinstance(s, dict)]
    if stages:
        return stages
    # Fallback: the per-stage provenance records carry the same fields.
    return [rec for sid in STAGES if (rec := reader.json(f"provenance/{sid}.json"))]


def _stage_cards(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for record in records:
        sid = str(record.get("stage_id") or "")
        info = STAGES.get(sid, {"title": sid, "plain": "", "section": ""})
        checks = record.get("checks") if isinstance(record.get("checks"), dict) else {}
        failed = [
            {"check": name, "detail": _clip(body.get("detail"))}
            for name, body in checks.items()
            if isinstance(body, dict) and body.get("passed") is not True
        ]
        blockers = [_clip(b) for b in (record.get("blockers") or [])]
        outputs = record.get("outputs") or []
        cards.append({
            "stage_id": sid,
            "title": info["title"],
            "what_it_does": info["plain"],
            "status": record.get("status") or "unknown",
            "light": _light(str(record.get("status")), len(failed), len(blockers)),
            "checks_passed": len(checks) - len(failed),
            "checks_total": len(checks),
            "failed_checks": failed,
            "blockers": blockers,
            "limitations": [_clip(x) for x in (record.get("limitations") or [])],
            "warnings": _dedupe(record.get("warnings") or []),
            "n_outputs": len(outputs),
        })
    return cards


def _metrics(records: list[dict[str, Any]], stage_id: str) -> dict[str, Any]:
    for record in records:
        if record.get("stage_id") == stage_id and isinstance(record.get("metrics"), dict):
            return record["metrics"]
    return {}


# --------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------
def _data_section(reader: _Reader, records: list[dict[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    m = _metrics(records, "00_inventory")
    inventory = reader.table("stages/00_inventory/tables/seismic_volume_inventory.csv")
    consistency = reader.table("stages/00_inventory/tables/stack_consistency_check.csv")
    manifest = reader.json("stages/00_inventory/data_manifest.json")
    volumes = [{
        "volume": row.get("volume"), "kind": row.get("kind"), "role": _clip(row.get("role"), 120),
        "n_traces": _num(row.get("n_traces")), "sample_interval_ms": _num(row.get("dt_ms")),
        "samples_per_trace": _num(row.get("ns")), "record_end_ms": _num(row.get("t_max_ms")),
        "inline_range": [_num(row.get("il_min")), _num(row.get("il_max"))],
        "crossline_range": [_num(row.get("xl_min")), _num(row.get("xl_max"))],
        "bin_spacing_m": [_r(row.get("inline_spacing"), 1), _r(row.get("trace_spacing_along_inline"), 1)],
        "geometry_flag": _clip(row.get("geometry_flags"), 120) or None,
    } for row in inventory]
    section = {
        "wells_expected": m.get("n_wells_expected"),
        "wells_with_logs": m.get("n_las_found"),
        "wells_with_deviation_survey": m.get("n_deviation_found"),
        "time_depth_tables_found": m.get("n_tz_tables_found"),
        "wells_with_sonic_and_density": m.get("n_wells_with_DT_and_RHOB"),
        "seismic_volumes_read": m.get("n_seismic_volumes_read"),
        "volumes": volumes,
        "partial_stack_similarity_to_full": [{
            "volume": row.get("volume"), "median_correlation": _r(row.get("median_corr_with_full")),
            "polarity": _clip(row.get("polarity_verdict"), 80),
        } for row in consistency],
        "coordinate_system_note": _clip(manifest.get("crs"), 260) or None,
    }
    wells, las, dev = _num(m.get("n_wells_expected")), _num(m.get("n_las_found")), _num(m.get("n_deviation_found"))
    if wells is not None:
        complete = las == wells and dev == wells
        findings.append(_finding(
            "data_complete", "good" if complete else "caution",
            f"{int(las or 0)} of {int(wells)} wells have logs and {int(dev or 0)} have a deviation survey; "
            f"{m.get('n_seismic_volumes_read', 'n/a')} seismic volumes were read.",
            "Every expected input was found and opened." if complete else "Some wells are missing a basic input; see the data section.",
            ["geogg_foundations_summary.json"]))
    if _num(m.get("n_wells_with_DT_and_RHOB")) == 0 and wells:
        findings.append(_finding(
            "no_sonic_density", "caution",
            "No well has both a sonic (DT) and a density (RHOB) log.",
            "The logs cannot be tied to the seismic by a synthetic trace, and porosity or shale volume cannot be recomputed "
            "independently. The run relies on the provider's interpreted curves and on a depth-to-time conversion that could not be verified.",
            ["stages/00_inventory/tables/well_availability.csv"]))
    return section


def _ties_section(reader: _Reader, records: list[dict[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    m = _metrics(records, "01_well_ties")
    m3 = _metrics(records, "03_extraction")
    ties = reader.table("stages/01_well_ties/tables/tie_summary.csv")
    models = reader.table("stages/01_well_ties/tables/regional_model_selection.csv")
    inventory = reader.table("stages/00_inventory/tables/seismic_volume_inventory.csv")
    dt_ms = next((_num(row.get("dt_ms")) for row in inventory if row.get("volume") == "full"), None)
    selected = next((row for row in models if _truthy(row.get("selected"))), {})
    rms = _num(m3.get("twt_uncertainty_rms_ms")) or _num(selected.get("mean_lowo_rms_ms"))
    samples = rms / dt_ms if rms is not None and dt_ms else None
    strong = [row.get("well") for row in ties if _truthy(row.get("strong_extrapolation"))]
    n_wells = _num(m.get("n_wells_processed")) or len(ties)
    n_valid = _num(m.get("n_wells_validated"))
    section = {
        "wells_processed": n_wells,
        "wells_with_validated_tie": n_valid,
        "tie_status_counts": m.get("tie_status_counts"),
        "regional_model_selected": m.get("regional_model_selected") or selected.get("model"),
        "candidate_models": [{
            "model": row.get("model"), "mean_leave_one_well_out_rms_ms": _r(row.get("mean_lowo_rms_ms"), 1),
            "physically_valid": _truthy(row.get("physically_valid")), "selected": _truthy(row.get("selected")),
        } for row in models],
        "selection_rule": _clip(selected.get("selection_rule"), 160) or None,
        "time_uncertainty_rms_ms": _r(rms, 1),
        "seismic_sample_interval_ms": dt_ms,
        "time_uncertainty_in_samples": _r(samples, 1),
        "wells_with_strong_extrapolation": strong,
        "per_well": [{
            "well": row.get("well"), "time_depth_source": row.get("tz_source"), "tie_status": row.get("tie_status"),
            "extrapolation_m": _r(row.get("max_tz_extrapolation_m"), 0),
            "strong_extrapolation": _truthy(row.get("strong_extrapolation")),
        } for row in ties],
    }
    if n_valid is not None and n_wells:
        ok = n_valid >= n_wells
        findings.append(_finding(
            "ties_validated", "good" if ok else "critical" if n_valid == 0 else "caution",
            f"{int(n_valid)} of {int(n_wells)} wells have a validated seismic tie"
            + (f"; the depth-to-time conversion uses the '{section['regional_model_selected']}' regional model." if section["regional_model_selected"] else "."),
            "Logs are confidently positioned on the seismic." if ok else
            "Where each log sample sits on the seismic is an estimate, not a measurement. Everything downstream is exploratory until measured checkshots or synthetic ties are available.",
            ["stages/01_well_ties/tables/tie_summary.csv"]))
    if rms is not None and samples is not None:
        findings.append(_finding(
            "time_uncertainty", "caution" if samples > 1 else "info",
            f"Depth-to-time uncertainty is about {rms:.1f} ms RMS, which is {samples:.1f} seismic samples at {dt_ms:g} ms.",
            "A label can be misplaced by more than one seismic sample, so thin beds may be paired with the wrong piece of seismic. "
            "Results on the coarser 12 ms and 24 ms tables are less sensitive to this than the 4 ms table." if samples > 1 else
            "Label placement error is within one seismic sample.",
            ["stages/01_well_ties/tables/lowo_validation.csv", "stages/01_well_ties/tables/regional_model_selection.csv"]))
    if strong:
        findings.append(_finding(
            "tz_extrapolation", "caution",
            f"The time-depth relation is extrapolated far beyond its control in: {', '.join(str(w) for w in strong)}.",
            "The deepest samples of these wells are the least trustworthy and are flagged in the tables.",
            ["stages/01_well_ties/tables/tie_summary.csv"]))
    return section


def _labels_section(reader: _Reader, records: list[dict[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    m = _metrics(records, "02_labels")
    policy = reader.json("stages/02_labels/label_policy.json")
    classes = reader.table("stages/02_labels/tables/class_definition_table.csv")
    sensitivity = reader.table("stages/02_labels/tables/cutoff_sensitivity.csv")
    effective = policy.get("effective_policy") if isinstance(policy.get("effective_policy"), dict) else {}
    status = m.get("task_status") if isinstance(m.get("task_status"), dict) else {}
    supported = sorted(k for k, v in status.items() if str(v).startswith("SUPPORTED"))
    unsupported = sorted(k for k, v in status.items() if str(v) == "TASK_UNSUPPORTED")
    thickness = {k[len("dev_md_m_"):]: _r(v, 0) for k, v in m.items() if k.startswith("dev_md_m_")}
    section = {
        "policy_status": policy.get("status"),
        "lithology_rule": effective.get("lithology"),
        "fluid_rule": effective.get("fluid"),
        "labelled_fraction_of_logged_interval": _r(m.get("dev_fraction_labelled")),
        "thickness_m_by_class": thickness,
        "classes": [{
            "class": row.get("class_name"), "meaning": _clip(row.get("geological_meaning"), 90),
            "rule_in_this_run": _clip(row.get("rule_in_this_run"), 140),
            "thickness_m": _r(row.get("md_thickness_m_all_wells"), 0), "task_status": row.get("task_status") or None,
        } for row in classes],
        "tasks_supported": supported,
        "tasks_unsupported": unsupported,
        "cutoff_sensitivity": [{
            "vsh_sand_max": _num(row.get("vsh_sand_max")), "vsh_shale_min": _num(row.get("vsh_shale_min")),
            "sand_m": _r(row.get("sand_md_m"), 0), "shale_m": _r(row.get("shale_md_m"), 0),
            "active": _truthy(row.get("is_active_policy")),
        } for row in sensitivity],
        "provider_pay_flag_note": _clip((m.get("provider_flag_inference") or {}).get("note"), 300) or None,
    }
    sand = sum(v or 0 for k, v in thickness.items() if k.endswith("_sand"))
    shale = thickness.get("shale")
    if shale and sand:
        findings.append(_finding(
            "label_balance", "info",
            f"Labelled rock is {shale:.0f} m of shale and {sand:.0f} m of sand; "
            f"{_pct(m.get('dev_fraction_labelled'))} of the logged interval carries a label.",
            f"Shale outweighs sand by about {shale / sand:.1f} to 1, so a model can look accurate just by answering 'shale'. "
            "Read ROC-AUC and balanced accuracy, not plain accuracy.",
            ["stages/02_labels/tables/class_definition_table.csv"]))
    fluid_unsupported = [t for t in unsupported if "oil" in t or "gas" in t]
    if fluid_unsupported:
        findings.append(_finding(
            "no_oil_gas_labels", "caution",
            "Oil sand and gas sand are not trainable in this run (TASK_UNSUPPORTED).",
            "There is no independent fluid evidence such as tests, samples or production. The run can indicate 'hydrocarbon potential' "
            "from low water saturation, but it cannot say oil or gas, and no output should be read that way.",
            ["stages/02_labels/tables/task_support.csv"]))
    if policy.get("status"):
        findings.append(_finding(
            "label_policy_provisional", "caution" if "PROVISIONAL" in str(policy.get("status")).upper() else "info",
            f"Label policy status: {_clip(policy.get('status'), 120)}.",
            "The sand and shale cutoffs are sensible defaults that a geologist has not yet approved. Changing them changes the labels; "
            "the cutoff sensitivity table shows by how much.",
            ["stages/02_labels/label_policy.json", "stages/02_labels/tables/cutoff_sensitivity.csv"]))
    return section


def _training_section(reader: _Reader, records: list[dict[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    m3, m5 = _metrics(records, "03_extraction"), _metrics(records, "05_training_tables")
    m4 = _metrics(records, "04_features")
    patches = reader.json("stages/03_extraction/patches/patch_manifest_v1.json")
    plan = reader.table("stages/05_training_tables/tables/split_plan_wells.csv")
    resolutions = sorted({int(k[5:-2]) for k in m5 if k.startswith("rows_") and k.endswith("ms") and k[5:-2].isdigit()})
    section = {
        "resolutions_ms": resolutions,
        "rows": {f"{r}ms": m5.get(f"rows_{r}ms") for r in resolutions},
        "usable_targets": {f"{r}ms": m5.get(f"usable_targets_{r}ms") for r in resolutions},
        "class_counts": {f"{r}ms": m3.get(f"class_counts_{r}ms") for r in resolutions},
        "cut_out_shape": patches.get("shape"),
        "cut_out_axes": patches.get("axis_order"),
        "cut_out_channels": patches.get("channels"),
        "excluded_channels": patches.get("excluded_channels"),
        "n_predictors": m5.get("n_predictors"),
        "fraction_rows_with_inversion_data": _r(m5.get("fraction_rows_with_all_elastic_inputs_4ms")),
        "inversion_valid_fraction_at_wells": m4.get("elastic_centre_valid_fraction"),
        "test_groups": m5.get("outer_folds"),
        "per_well": [{
            "well": row.get("well"), "role": row.get("role"), "test_group": _num(row.get("outer_fold")),
            "rows": _num(row.get("n_rows")), "sand_rows": _num(row.get("n_sand")), "shale_rows": _num(row.get("n_shale")),
            "water_sand_rows": _num(row.get("n_water_sand")), "hc_potential_sand_rows": _num(row.get("n_hc_potential_sand")),
            "nearest_training_well_m": _r(row.get("nearest_training_well_path_distance_m"), 0),
        } for row in plan],
    }
    fraction = _num(m5.get("fraction_rows_with_all_elastic_inputs_4ms"))
    if fraction is not None and fraction < 0.98:
        findings.append(_finding(
            "inversion_coverage", "caution",
            f"Only {_pct(fraction)} of training rows have inversion data; the rest fall outside the inversion volumes' coverage.",
            "Stacks-only and stacks-plus-inversion models can only be compared fairly on the rows that have both, "
            "which is why the run reports '@matched' results.",
            ["stages/04_features/tables/zero_fill_summary.csv", "stages/05_training_tables/training_manifest.json"]))
    fine = resolutions[0] if resolutions else None
    targets = m5.get(f"usable_targets_{fine}ms") if fine else None
    if isinstance(targets, dict) and _num(targets.get("y_sand_fluid")) is not None:
        n_fluid, n_lith = _num(targets.get("y_sand_fluid")), _num(targets.get("y_lithology"))
        if n_fluid is not None and n_fluid < 1000:
            findings.append(_finding(
                "fluid_sample_size", "caution",
                f"The fluid task has {int(n_fluid)} usable rows at {fine} ms, against {int(n_lith or 0)} for sand versus shale.",
                "That is a small sample spread over few wells, so fluid scores are noisy and can swing with one well.",
                ["stages/05_training_tables/tables/split_plan_wells.csv"]))
    return section


def _features_section(reader: _Reader, records: list[dict[str, Any]], findings: list[dict[str, Any]]) -> dict[str, Any]:
    m4, m6 = _metrics(records, "04_features"), _metrics(records, "06_feature_selection")
    selected = reader.table("stages/06_feature_selection/tables/selected_features.csv")
    screening = reader.table("stages/06_feature_selection/tables/screening_and_clusters.csv")
    resolutions = sorted({int(r["resolution_ms"]) for r in selected if str(r.get("resolution_ms", "")).isdigit()})
    fine = str(resolutions[0]) if resolutions else ""
    twt = {
        row.get("feature"): _num(row.get("spearman_with_twt")) for row in screening
        if row.get("fold_label") == "all_development" and row.get("resolution_ms") == fine and row.get("group") == "fused"
    }
    top: dict[str, list[dict[str, Any]]] = {}
    for row in selected:
        if (row.get("fold_label") != "all_development" or row.get("resolution_ms") != fine
                or row.get("group") != "fused" or row.get("outcome") != "selected"):
            continue
        top.setdefault(str(row.get("task")), []).append({
            "rank": _num(row.get("rank")), "feature": row.get("feature"),
            "well_level_effect": _r(row.get("re_mean_effect")),
            "interval": [_r(row.get("re_ci_lo")), _r(row.get("re_ci_hi"))],
            "wells": _num(row.get("k_wells")), "sign_consistency": _r(row.get("sign_consistency"), 2),
            "correlation_with_time": _r(twt.get(row.get("feature"))),
        })
    for task in top:
        top[task] = sorted(top[task], key=lambda item: item["rank"] if item["rank"] is not None else 1e9)[:6]
    counts = m6.get("n_selected_all_development") if isinstance(m6.get("n_selected_all_development"), dict) else {}
    none_selected = sorted({
        f"{group}/{task}" for group, tasks in counts.items() if isinstance(tasks, dict)
        for task, n in tasks.items() if _num(n) == 0
    })
    section = {
        "avo_mode": m4.get("avo_mode"),
        "n_stack_attributes": m4.get("n_stack_feature_columns"),
        "n_inversion_attributes": m4.get("n_elastic_feature_columns"),
        "redundancy_clusters": m6.get("clusters_all_development"),
        "n_selected_by_group_and_task": counts,
        "nothing_selected_for": none_selected,
        "sign_replication_rate_by_task": m6.get("inner_sign_replication_rate"),
        "top_attributes_by_task": top,
        "informative_missingness": (m6.get("informative_missingness_all_development") or [])[:8],
        "resolution_ms_used_for_ranking": int(fine) if fine else None,
    }
    if str(m4.get("avo_mode") or "").upper().startswith("QUALITATIVE"):
        findings.append(_finding(
            "avo_qualitative", "caution",
            "The near, mid, far and ultra-far stacks have no documented angle ranges (AVO mode: QUALITATIVE_PROXY_ONLY).",
            "Near-versus-far contrasts are used as descriptive attributes only. They are not calibrated AVO intercept and gradient, "
            "so they must not be read as fluid indicators.",
            ["stages/04_features/tables/feature_dictionary.csv"]))
    if none_selected:
        findings.append(_finding(
            "nothing_selected", "caution",
            f"No attribute passed the well-to-well consistency test for: {', '.join(none_selected)}.",
            "No single seismic attribute relates to that target reliably across wells. Models for it fall back to the unfiltered "
            "attribute set, so any score they reach deserves extra suspicion.",
            ["stages/06_feature_selection/tables/selected_features.csv"]))
    depth_like = []
    for task, items in top.items():
        strong = [i for i in items[:5] if i["correlation_with_time"] is not None and abs(i["correlation_with_time"]) >= 0.6]
        if items and len(strong) * 2 >= min(len(items), 5):
            depth_like.append((task, strong))
    if depth_like:
        text = "; ".join(
            f"{TASK_NAMES.get(task, task)}: " + ", ".join(f"{i['feature']} ({i['correlation_with_time']:+.2f})" for i in strong[:3])
            for task, strong in depth_like)
        findings.append(_finding(
            "depth_trend_features", "caution",
            f"The top-ranked attributes for some targets rise or fall steadily with seismic time. Correlation with time: {text}.",
            "These attributes largely track burial depth, and porosity and saturation also change with depth. Part of the apparent "
            "skill may be the depth trend, not the seismic seeing the rock. A depth-only baseline is the test that settles it.",
            ["stages/06_feature_selection/tables/screening_and_clusters.csv", "stages/06_feature_selection/tables/selected_features.csv"]))
    if m6.get("informative_missingness_all_development"):
        findings.append(_finding(
            "informative_missingness", "caution",
            "Whether inversion data are missing is itself related to the targets: "
            + "; ".join(str(x) for x in m6["informative_missingness_all_development"][:4]) + ".",
            "Where the inversion volumes have data coincides with different rock. A model allowed to see the missing pattern could "
            "learn coverage, not geology. This run trains without missing-value indicators for that reason.",
            ["stages/06_feature_selection/tables/missingness_informativeness.csv"]))
    return section


def _rate(kind: str, auc: float | None, r2: float | None) -> str:
    if kind == "binary":
        if auc is None:
            return "unrated"
        return ("none" if auc < 0.55 else "weak" if auc < 0.65 else "modest" if auc < 0.75
                else "good" if auc < 0.85 else "strong (verify there is no leakage)")
    if r2 is None:
        return "unrated"
    return "none" if r2 <= 0.02 else "weak" if r2 < 0.20 else "moderate" if r2 < 0.50 else "strong"


def _models_section(reader: _Reader, records: list[dict[str, Any]], summary: dict[str, Any],
                    findings: list[dict[str, Any]]) -> dict[str, Any]:
    m7 = _metrics(records, "07_models")
    registry = reader.table("stages/07_models/tables/model_registry.csv")
    board: list[dict[str, Any]] = []
    for row in registry:
        kind = str(row.get("kind") or "")
        metric = str(row.get("primary_metric") or "")
        value = _num(row.get("inner_primary_mean"))
        if value is None:
            value = _num(row.get(f"inner_{metric}_mean"))
        spread = _num(row.get("inner_primary_sd_folds"))
        if spread is None:
            spread = _num(row.get(f"inner_{metric}_sd_folds"))
        auc, r2 = _num(row.get("inner_roc_auc_mean")), _num(row.get("inner_r2_mean"))
        board.append({
            "task": row.get("task"), "input": row.get("input"), "model": row.get("model"),
            "status": row.get("status"), "kind": kind,
            "metric": metric, "value": _r(value, 4), "fold_spread": _r(spread, 4), "rows": _num(row.get("n_rows")),
            "roc_auc": _r(auc), "balanced_accuracy": _r(row.get("inner_balanced_acc_mean")),
            "prevalence": _r(row.get("inner_prevalence_mean")),
            "r2": _r(r2), "spearman": _r(row.get("inner_spearman_mean")),
            "champion": _truthy(row.get("champion_inner_cv")),
            "beats_depth_trend": row.get("beats_depth_trend_inner") or None,
            "rating": _rate(kind, auc, r2),
        })

    def best(task: str, input_name: str) -> dict[str, Any] | None:
        rows = [b for b in board if b["task"] == task and b["input"] == input_name and b["value"] is not None]
        if not rows:
            return None
        return (max if rows[0]["kind"] == "binary" else min)(rows, key=lambda b: b["value"])

    def gain(a: dict[str, Any], b: dict[str, Any]) -> tuple[float, float, str]:
        """Improvement of b over a (positive = b better), the larger fold spread, and a display string."""
        noise = max(a["fold_spread"] or 0.0, b["fold_spread"] or 0.0)
        if a["kind"] == "binary":
            return b["value"] - a["value"], noise, f"ROC-AUC {a['value']:.3f} to {b['value']:.3f}"
        return a["value"] - b["value"], noise, f"RMSE {a['value']:.4f} to {b['value']:.4f}"

    tasks = list(dict.fromkeys(b["task"] for b in board))
    readings: list[dict[str, Any]] = []
    for task in tasks:
        plain = TASK_NAMES.get(str(task), str(task))
        candidates = [b for b in board if b["task"] == task and b["value"] is not None]
        if not candidates:
            continue
        reading: dict[str, Any] = {"task": task, "task_plain": plain, "best_by_input": {}}
        for input_name in dict.fromkeys(b["input"] for b in candidates):
            top = best(task, input_name)
            if top:
                reading["best_by_input"][input_name] = {
                    k: top[k] for k in ("model", "metric", "value", "fold_spread", "rows", "roc_auc", "r2", "spearman", "rating")}
        stacks_m, fused_m = best(task, "stacks@matched"), best(task, "fused@matched")
        if stacks_m and fused_m:
            delta, noise, shown = gain(stacks_m, fused_m)
            verdict = ("helps" if delta > noise and delta > 0 else "hurts" if -delta > noise else "no clear difference")
            reading["inversion_effect"] = {
                "verdict": verdict, "change": shown, "improvement": _r(delta, 4), "fold_spread": _r(noise, 4),
                "still_no_skill": fused_m["rating"] == "none"}
        stacks_all, deep = best(task, "stacks"), best(task, "patch_stacks")
        if stacks_all and deep:
            delta, noise, shown = gain(stacks_all, deep)
            reading["deep_learning_effect"] = {
                "verdict": "helps" if delta > noise and delta > 0 else "hurts" if -delta > noise else "no clear difference",
                "change": shown, "improvement": _r(delta, 4), "fold_spread": _r(noise, 4), "deep_model": deep["model"]}
        if stacks_all and stacks_m:
            if stacks_all["kind"] == "binary":
                drop, shown = (stacks_all["roc_auc"] or 0) - (stacks_m["roc_auc"] or 0), "ROC-AUC"
                a, b, flagged = stacks_all["roc_auc"], stacks_m["roc_auc"], drop >= 0.08
            else:
                a, b = stacks_all["r2"], stacks_m["r2"]
                drop = (a - b) if a is not None and b is not None else 0.0
                shown, flagged = "R2", drop >= 0.15
            reading["all_rows_versus_matched_rows"] = {
                "metric": shown, "all_rows": a, "matched_rows": b, "drop": _r(drop), "flagged": bool(flagged)}
        readings.append(reading)

        gap = reading.get("all_rows_versus_matched_rows") or {}
        pool = reading["best_by_input"]
        if gap.get("flagged"):  # the all-rows score is not a fair basis: read the matched rows instead
            pool = {k: v for k, v in pool.items() if k.endswith("@matched")} or pool
        name, top = max(pool.items(), key=lambda kv: (
            kv[1]["roc_auc"] if kv[1]["roc_auc"] is not None else kv[1]["r2"] if kv[1]["r2"] is not None else -9))
        score = (f"ROC-AUC {top['roc_auc']:.2f}" if top["roc_auc"] is not None
                 else f"R2 {top['r2']:.2f}, RMSE {top['value']:.3f}" if top["r2"] is not None else f"{top['metric']} {top['value']}")
        basis = "matched rows, because the all-rows score is inflated by the coverage boundary" if gap.get("flagged") else "best candidate"
        reading["fair_reading"] = {"input": name, "model": top["model"], "score": score, "rating": top["rating"], "basis": basis}
        level = {"none": "caution", "weak": "caution", "modest": "info", "moderate": "info", "good": "good", "strong": "good"}.get(
            top["rating"].split(" ")[0], "info")
        meaning = {
            "none": "In practice testing the models do no better than a naive guess for this target.",
            "weak": "There is a real but small signal. It is not yet strong enough to rely on.",
            "modest": "There is a usable signal, clearly better than chance but far from decisive.",
            "moderate": "The models explain a meaningful part of the variation, with sizeable error remaining.",
            "good": "The models separate the classes well in practice testing.",
            "strong": "The score is high enough that leakage should be ruled out before it is believed.",
        }.get(top["rating"].split(" ")[0], "")
        findings.append(_finding(
            f"model_{task}", level,
            f"Fairest practice score for {plain}: {score} ({top['model']}, {INPUT_NAMES.get(name, name)}). Rating: {top['rating']}.",
            meaning, ["stages/07_models/tables/model_registry.csv"]))
        effect = reading.get("inversion_effect")
        if effect:
            text = {
                "helps": "Adding the inversion volumes improves the practice score by more than the fold-to-fold spread.",
                "hurts": "Adding the inversion volumes makes the practice score worse.",
                "no clear difference": "Adding the inversion volumes changes the score by less than the fold-to-fold spread, so no benefit is shown.",
            }[effect["verdict"]]
            if effect["verdict"] == "helps" and effect["still_no_skill"]:
                text += " The model still shows no real skill for this target, so the gain has no practical value yet."
            findings.append(_finding(
                f"inversion_{task}", "info",
                f"Inversion volumes for {plain}: {effect['change']} on matched rows (fold-to-fold spread {_fmt(effect['fold_spread'])}).",
                text, ["stages/07_models/tables/model_registry.csv"]))
        if gap.get("flagged"):
            findings.append(_finding(
                f"coverage_gap_{task}", "caution",
                f"For {plain}, the stacks-only {gap['metric']} falls from {gap['all_rows']} on all rows to {gap['matched_rows']} "
                "on the rows that have inversion data.",
                "Much of the all-rows score appears to come from telling apart the zones with and without inversion coverage, "
                "which is largely a depth effect, and not from recognising the rock within a zone. Treat the matched-row score as the fairer one.",
                ["stages/07_models/tables/model_registry.csv"]))

    depth_values = {b["beats_depth_trend"] for b in board}
    depth_tested = bool(depth_values - {None})
    if board and not depth_tested:
        findings.append(_finding(
            "no_depth_baseline", "caution",
            "The registry column 'beats_depth_trend_inner' is empty for every candidate.",
            "This run does not show whether any model beats a simple depth-only trend. Until that comparison exists, "
            "no score here proves that the seismic adds information beyond depth.",
            ["stages/07_models/tables/model_registry.csv"]))
    if board:
        findings.append(_finding(
            "inner_cv_only", "caution",
            "Every score in this run is an inner cross-validation score.",
            "These are practice scores used to choose settings and they run slightly optimistic. The honest held-out-well "
            "performance has not been computed in this process; the sealed predictions are waiting for the evaluation stage.",
            ["stages/07_models/tables/model_registry.csv"]))
    clip_warning = next((w for r in records if r.get("stage_id") == "07_models" for w in (r.get("warnings") or [])
                         if "clipped" in str(w).lower()), None)
    if clip_warning:
        findings.append(_finding(
            "physical_clipping", "caution", _clip(clip_warning, 300),
            "Some raw predictions were physically impossible, such as a saturation outside 0 to 1, and were pulled back to the limit. "
            "It signals a model extrapolating beyond what it learned.",
            ["geogg_foundations_summary.json"]))

    release = summary.get("model_release") if isinstance(summary.get("model_release"), dict) else {}
    sealed = "stages/07_models/tables/outer_test_predictions.csv"
    checks = next((r.get("checks") for r in records if r.get("stage_id") == "07_models"), {}) or {}
    return {
        "candidates": m7.get("n_candidates"),
        "fraction_rows_matched": _r(m7.get("fraction_rows_matched")),
        "scorecard": [{
            "target": r["task_plain"], "task": r["task"], **r.get("fair_reading", {}),
            "inversion_volumes": (r.get("inversion_effect") or {}).get("verdict"),
            "deep_learning": (r.get("deep_learning_effect") or {}).get("verdict"),
        } for r in readings],
        "task_names": TASK_NAMES,
        "input_names": INPUT_NAMES,
        "scoreboard": [{k: v for k, v in b.items() if v is not None and k != "kind"} for b in board],
        "readings_by_task": readings,
        "depth_trend_baseline_evaluated": depth_tested,
        "sealed_held_out_predictions": {
            "file": sealed, "present": reader.exists(sealed), "size_bytes": reader.size(sealed),
            "note": "Not scored by this skill. Held-out-well performance belongs to the evaluation stage (08).",
            "integrity": _clip((checks.get("each_row_predicted_once_per_candidate") or {}).get("detail"), 160) or None,
        },
        "leakage_audit": _clip((checks.get("no_held_out_well_in_any_fit") or {}).get("detail"), 160) or None,
        "model_release": {k: release.get(k) for k in ("model_release_id", "status", "model_count")} if release else None,
        "rating_convention": RATING_NOTE,
    }


def _well_card(reader: _Reader, well: str) -> dict[str, Any]:
    """Everything the run recorded about one well, from the per-well summary tables."""
    wanted = well.strip().lower()
    sources = {
        "tie": "stages/01_well_ties/tables/tie_summary.csv",
        "trajectory": "stages/01_well_ties/tables/trajectory_summary.csv",
        "split": "stages/05_training_tables/tables/split_plan_wells.csv",
        "alignment": "stages/03_extraction/tables/label_alignment_audit.csv",
        "classes": "stages/02_labels/tables/class_counts_by_well.csv",
    }
    tables = {key: reader.table(path) for key, path in sources.items()}
    known = sorted({str(row.get("well")) for row in tables["tie"] or tables["split"] if row.get("well")})

    def one(key: str) -> dict[str, str]:
        return next((row for row in tables[key] if str(row.get("well", "")).strip().lower() == wanted), {})

    tie, trajectory, split, alignment = one("tie"), one("trajectory"), one("split"), one("alignment")
    if not (tie or split):
        return {"well": well, "found": False, "available_wells": known}
    return {
        "well": tie.get("well") or split.get("well"), "found": True,
        "total_depth_md_m": _r(trajectory.get("md_td_m"), 0), "max_inclination_deg": _r(trajectory.get("max_incl_deg"), 1),
        "horizontal_reach_m": _r(trajectory.get("max_horizontal_displacement_m"), 0),
        "time_depth_source": tie.get("tz_source"), "tie_status": tie.get("tie_status"),
        "time_depth_extrapolation_m": _r(tie.get("max_tz_extrapolation_m"), 0),
        "strong_extrapolation": _truthy(tie.get("strong_extrapolation")),
        "why_no_synthetic_tie": _clip(tie.get("synthetic_reason"), 200) or None,
        "role": split.get("role"), "test_group": _num(split.get("outer_fold")),
        "training_rows": _num(split.get("n_rows")), "sand_rows": _num(split.get("n_sand")), "shale_rows": _num(split.get("n_shale")),
        "water_sand_rows": _num(split.get("n_water_sand")), "hc_potential_sand_rows": _num(split.get("n_hc_potential_sand")),
        "nearest_training_well_m": _r(split.get("nearest_training_well_path_distance_m"), 0),
        "seismic_time_window_ms": [_num(alignment.get("twt_top_ms")), _num(alignment.get("twt_base_ms"))],
        "seismic_cut_outs": _num(alignment.get("n_patch_rows")),
        "thickness_m_by_class": {
            str(row.get("class_name")): _r(row.get("md_thickness_m"), 0) for row in tables["classes"]
            if str(row.get("well", "")).strip().lower() == wanted and (_num(row.get("md_thickness_m")) or 0) > 0},
        "figures": [p for p in (
            f"stages/01_well_ties/figures/per_well/{tie.get('well')}_tz_velocity.png",
            f"stages/02_labels/figures/per_well/{tie.get('well')}_petrophysical_panel.png",
        ) if reader.exists(p)],
    }


# --------------------------------------------------------------------------------------------
# Verdict and next steps
# --------------------------------------------------------------------------------------------
def _verdict(status: str, cards: list[dict[str, Any]]) -> dict[str, Any]:
    missing = [sid for sid in STAGES if sid not in {c["stage_id"] for c in cards}]
    failed = sum(len(c["failed_checks"]) for c in cards)
    blockers = sum(len(c["blockers"]) for c in cards)
    total = sum(c["checks_total"] for c in cards)
    limitations = sum(len(c["limitations"]) for c in cards)
    if str(status).lower() not in {"succeeded", "success", "completed"}:
        level, plain = "did_not_finish", f"The job status is '{status}', so this is not a completed result."
    elif missing:
        level, plain = "incomplete", f"The run finished but {len(missing)} stage(s) have no record: {', '.join(missing)}."
    elif failed or blockers:
        level, plain = "needs_attention", f"{failed} self-check(s) failed and {blockers} blocker(s) were raised. Resolve these before using the outputs."
    elif limitations or any(c["light"] == "amber" for c in cards):
        level = "usable_with_caveats"
        plain = (f"The run completed and passed all {total} of its self-checks, with no blockers. It is technically sound. "
                 f"It also states {limitations} limitation(s) about the input data, so results are exploratory, not final.")
    else:
        level, plain = "clean", f"The run completed, passed all {total} self-checks and states no limitations."
    return {"level": level, "plain": plain, "checks_passed": total - failed, "checks_total": total,
            "blockers": blockers, "limitations": limitations, "stages_recorded": len(cards), "stages_missing": missing}


def _next_steps(findings: list[dict[str, Any]], verdict: dict[str, Any]) -> list[str]:
    ids = {f["id"] for f in findings}
    levels = {f["id"]: f["level"] for f in findings}
    steps: list[str] = []
    if verdict["level"] in {"did_not_finish", "incomplete", "needs_attention"}:
        steps.append("Resolve the failed checks, blockers or missing stages first; the scientific reading below is provisional until then.")
    if levels.get("ties_validated") in {"critical", "caution"}:
        steps.append("Obtain measured checkshots, or sonic and density logs for synthetic ties. This is the single biggest lever on label placement.")
    if "inner_cv_only" in ids:
        steps.append("Run the held-out-well evaluation (the next process) before quoting any performance number outside the team.")
    if "no_depth_baseline" in ids or "depth_trend_features" in ids:
        steps.append("Compare every model with a depth-only (TWT) baseline to show what the seismic adds beyond burial depth.")
    if any(i.startswith("coverage_gap_") for i in ids):
        steps.append("Quote the matched-row scores for targets where the all-rows score drops sharply, and investigate the coverage boundary.")
    if "time_uncertainty" in ids and levels["time_uncertainty"] == "caution":
        steps.append("Cross-check conclusions on the 12 ms and 24 ms tables, which tolerate the depth-to-time uncertainty better than 4 ms.")
    if "no_oil_gas_labels" in ids:
        steps.append("To train oil or gas classes, supply fluid evidence (tests, samples) or have a named reviewer accept the provider pay flags.")
    if "label_policy_provisional" in ids and levels["label_policy_provisional"] == "caution":
        steps.append("Ask a geologist to approve or adjust the sand and shale cutoffs, using the cutoff sensitivity table.")
    if "avo_qualitative" in ids:
        steps.append("Obtain the angle ranges of the partial stacks so near-versus-far attributes can be used as calibrated AVO.")
    return steps


# --------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------
def analyse_job_dir(job_dir: Path, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyse one already-located job folder. Pure file inspection; no backend access."""
    parameters = parameters or {}
    reader = _Reader(job_dir)
    metadata, result = reader.json("job.json"), reader.json("result.json")
    summary = reader.json(SUMMARY_FILE)
    job_id = str(parameters.get("job_id") or metadata.get("job_id") or job_dir.name)
    script = str(metadata.get("python_script") or result.get("python_script") or "")
    operation = str(summary.get("operation") or metadata.get("operation") or result.get("operation") or "")
    status = str(result.get("status") or summary.get("status") or metadata.get("status") or "unknown")
    base = {
        "skill": SKILL_NAME, "job_id": job_id, "python_script": script or None, "operation": operation or None,
        "target_well": (str(metadata.get("target_well") or "").strip() or None), "status": status,
        "created_at": metadata.get("created_at") or result.get("created_at"),
    }
    records = _stage_records(reader, summary)
    recognised = operation == EXPECTED_OPERATION or script == EXPECTED_SCRIPT
    if not recognised or not records:
        reason = ("This folder was not produced by Process A (expected operation "
                  f"'{EXPECTED_OPERATION}' or script '{EXPECTED_SCRIPT}')." if not recognised else
                  f"No stage records were found ('{SUMMARY_FILE}' and 'provenance/' are missing or unreadable).")
        if status.lower() not in {"succeeded", "success", "completed"} and recognised:
            reason += f" The job status is '{status}', so the run may have stopped before writing them."
        return {**base, "applicable": False, "summary": reason, "findings": [reason], "artifacts": reader.used,
                "skipped": reader.skipped,
                "limitations": ["Use the generic processing_output_folder_analysis skill to inventory this folder."]}

    cards = _stage_cards(records)
    verdict = _verdict(status, cards)
    findings: list[dict[str, Any]] = []
    sections = {
        "data": _data_section(reader, records, findings),
        "ties": _ties_section(reader, records, findings),
        "labels": _labels_section(reader, records, findings),
        "training": _training_section(reader, records, findings),
        "features": _features_section(reader, records, findings),
        "models": _models_section(reader, records, summary, findings),
    }
    def rank(finding: dict[str, Any]) -> int:
        return next((i for i, prefix in enumerate(PRIORITY) if finding["id"].startswith(prefix)), len(PRIORITY))

    findings.sort(key=rank)
    takeaways: list[dict[str, Any]] = []
    for prefix in TAKEAWAY_IDS:
        match = next((f for f in findings if f["id"].startswith(prefix)), None)
        if match and match["level"] in {"critical", "caution"}:
            takeaways.append(match)
    takeaways = takeaways[:5]

    focus_raw = str(parameters.get("focus") or "").strip().lower().replace(" ", "_").replace("-", "_")
    focus = FOCUS_ALIASES.get(focus_raw) or FOCUS_ALIASES.get(focus_raw.replace("stage_", "").replace("stage", ""), "overview")
    details: dict[str, Any] = {}
    if focus == "all":
        details = dict(sections)
    elif focus in sections:
        details = {focus: sections[focus]}
    # overview keeps the payload small for the language model: ask again with a focus for the tables
    if focus in sections:
        wanted = {sid for sid, info in STAGES.items() if info["section"] == focus}
        shown_cards = [c for c in cards if c["stage_id"] in wanted]
        shown_findings = [f for f in findings if _SECTION_OF.get(f["id"].split("_")[0], "") == focus or f["id"] in _FINDING_SECTION.get(focus, ())] or findings
    elif focus == "all":
        shown_cards, shown_findings = cards, findings
    elif focus in {"glossary", "figures"}:
        shown_cards, shown_findings = [], []
    else:  # overview: one line per stage and the most important findings only
        shown_cards = [{k: c[k] for k in ("stage_id", "title", "what_it_does", "status", "light", "checks_passed", "checks_total")}
                       | {"n_limitations": len(c["limitations"]), "n_warnings": len(c["warnings"]),
                          "failed_checks": c["failed_checks"], "blockers": c["blockers"]} for c in cards]
        shown_findings = findings[:OVERVIEW_FINDINGS]

    response: dict[str, Any] = {
        **base, "applicable": True, "focus": focus,
        "created_at": base["created_at"] or reader.json("stages/00_inventory/data_manifest.json").get("created_utc"),
        "verdict": verdict,
        "summary": verdict["plain"],
        "takeaways": takeaways,
        "scorecard": sections["models"]["scorecard"],
        "stage_cards": shown_cards,
        "key_findings": shown_findings,
        "findings": [f["observed"] for f in shown_findings],
        "next_steps": _next_steps(findings, verdict),
        "details": details,
        "findings_not_shown": len(findings) - len(shown_findings) if focus == "overview" else 0,
        "more_detail": "Call again with focus = data, ties, labels, training, features, models or all, or with well = <name>.",
    }
    well = str(parameters.get("well") or "").strip()
    if well:
        response["well_card"] = _well_card(reader, well)
    if focus in {"all", "glossary"}:
        response["glossary"] = GLOSSARY
    if focus in {"all", "figures"}:
        response["figures_guide"] = [{"file": path, "shows": caption} for path, caption in FIGURES if reader.exists(path)]
    if focus == "overview":
        response["suggested_questions"] = SUGGESTED_QUESTIONS
    response["artifacts"] = [{**item, "role": "evidence read by this analysis"} for item in reader.used]
    response["skipped"] = reader.skipped
    response["limitations"] = [
        "Numbers are read from the run's own tables and summary; this skill does not recompute the science or re-run any model.",
        "All model scores are inner cross-validation scores. Held-out-well performance is not part of Process A.",
        RATING_NOTE,
        "Plain-language meanings are fixed expert reading rules applied to the observed numbers. They state what a result "
        "usually implies, and are not proof of a geological cause.",
        "Figures are listed with what they show; their pixels are not inspected.",
        "Binary artifacts (model files, seismic cut-outs, geometry arrays) are not opened.",
    ]
    return response


def execute(*, db, project_id: str | None, parameters: dict[str, Any]) -> dict[str, Any]:
    """Locate the processing job folder through the configured storage roots, then analyse it."""
    from app.storage import configured_storage_roots

    job_id = str(parameters.get("job_id") or "").strip()
    if not job_id or Path(job_id).name != job_id or job_id in {".", ".."}:
        raise ValueError("A valid processing job ID is required.")
    job_dir = None
    for root in configured_storage_roots(db):
        jobs_root = (root / "processing_jobs").resolve()
        candidate_path = root / "processing_jobs" / job_id
        if candidate_path.is_symlink():
            continue
        candidate = candidate_path.resolve()
        if jobs_root in candidate.parents and candidate.is_dir():
            job_dir = candidate
            break
    if job_dir is None:
        raise ValueError(f"Processing run {job_id} was not found.")
    return analyse_job_dir(job_dir, {**parameters, "job_id": job_id})


def supports_job(metadata: dict[str, Any]) -> bool:
    """True when a job's ``job.json`` metadata names this process. Used for process-agnostic discovery."""
    metadata = metadata if isinstance(metadata, dict) else {}
    script = str(metadata.get("python_script") or "").replace("\\", "/").rsplit("/", 1)[-1]
    return script == EXPECTED_SCRIPT or str(metadata.get("operation") or "") == EXPECTED_OPERATION
