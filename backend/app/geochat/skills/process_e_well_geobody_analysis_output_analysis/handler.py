"""Expert, deterministic reading of one Process E (GeoGG well-geobody analysis, stage 13) run folder.

Process E walks along each well path and records where it passes through a candidate geobody,
then tries to describe the largest body each well meets: its depth and its predicted porosity
and water saturation. This handler reads the run's own summary, preflight and tables and applies
fixed, documented reading rules. It never calls an LLM, never invents a value and never opens a
volume. Every file it used is listed under ``artifacts``.

This module is self-contained on purpose: it imports no other skill.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

SKILL_NAME = "process_e_well_geobody_analysis_output_analysis"
EXPECTED_SCRIPT = "process_E_well_geobody_analysis.py"
EXPECTED_STAGE = "13_well_geobody_analysis"
SUMMARY_FILE = "geogg_well_geobody_analysis_summary.json"
PREFLIGHT_FILE = "geogg_well_geobody_analysis_preflight.json"
S13 = "stages/13_well_geobody_analysis"
TABLES = f"{S13}/tables"

MAX_JSON_BYTES = 8_000_000
MAX_CSV_BYTES = 30_000_000
MAX_CSV_ROWS = 60_000
MAX_TEXT = 420
OVERVIEW_FINDINGS = 8
DOMINANT_SHARE = 0.90
BACKGROUND_TIME_SPAN_MS = 2000.0

FOCUS_ALIASES: dict[str, str] = {
    "": "overview", "overview": "overview", "summary": "overview", "all": "all", "full": "all",
    "wells": "wells", "well": "wells", "intersections": "wells", "paths": "wells",
    "depth": "depth", "depths": "depth", "time_depth": "depth", "tz": "depth", "thickness": "depth",
    "properties": "properties", "property": "properties", "phit": "properties", "swt": "properties", "porosity": "properties",
    "provenance": "provenance", "preflight": "provenance", "products": "provenance", "files": "provenance", "checks": "provenance",
    "glossary": "glossary", "13": "overview",
}

GLOSSARY: dict[str, str] = {
    "geobody": "A connected 3-D group of voxels with a high sand score. A shape, not a proven reservoir.",
    "intersection": "A point on a well path that lies inside a geobody.",
    "path sample": "One recorded position along a well, here every few metres of measured depth.",
    "path interval": "A continuous stretch of a well path inside one body.",
    "selected body": "For each well, the largest body it passes through. Largest by voxel count, not best by quality.",
    "measured depth (MD)": "Distance along the well from the surface, in metres.",
    "TVDSS": "True vertical depth below sea level, in metres.",
    "TWT": "Two-way time in milliseconds, the vertical axis of seismic.",
    "time-depth (T-Z) model": "The conversion between seismic time and depth. Several candidate models exist here.",
    "model support": "The time range over which a T-Z model was fitted. Outside it the model is not used.",
    "model spread": "How far the candidate T-Z models disagree. It indicates uncertainty but is not a statistical interval.",
    "PHIT": "Total porosity: the fraction of the rock that is pore space.",
    "SWT": "Total water saturation: the fraction of the pore space filled with water.",
    "base rate": "How often something happens by chance alone. Here, the share of the whole cube that lies inside a body.",
}

SUGGESTED_QUESTIONS: list[str] = [
    "Did any well confirm a geobody?",
    "Why does every well meet the same body?",
    "How deep is the body, and how thick?",
    "Why are there no porosity or saturation numbers?",
    "Tell me about one well.",
    "What should we fix to make this analysis meaningful?",
]

# Order in which findings are surfaced (first match on the id prefix wins).
PRIORITY: list[str] = [
    "same_background_body", "base_rate", "fragmented_paths", "small_bodies_untested", "properties_not_computed",
    "depth_not_available", "depth_model_spread", "ties_provisional", "no_log_comparison", "count_is_per_well",
    "exploratory", "checks_passed", "files_missing",
]
TAKEAWAY_IDS: tuple[str, ...] = ("same_background_body", "base_rate", "properties_not_computed", "depth_not_available",
                                 "no_log_comparison", "ties_provisional")
_FINDING_FOCUS: dict[str, tuple[str, ...]] = {
    "wells": ("same_background_body", "base_rate", "fragmented_paths", "small_bodies_untested", "count_is_per_well", "no_log_comparison"),
    "depth": ("depth_not_available", "depth_model_spread", "ties_provisional"),
    "properties": ("properties_not_computed",),
    "provenance": ("checks_passed", "exploratory", "files_missing", "ties_provisional"),
}


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
    if number is None:
        return "n/a"
    text = f"{number:.{digits}f}"
    return text[1:] if text.startswith("-") and float(text) == 0 else text


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


def _finding(fid: str, level: str, observed: str, meaning: str, evidence: list[str]) -> dict[str, Any]:
    """level: good | info | caution | critical."""
    return {"id": fid, "level": level, "observed": observed, "plain_meaning": meaning, "evidence": evidence}


def _int(value: Any) -> int | None:
    number = _num(value)
    return None if number is None else int(round(number))


def _median(values: list[float]) -> float | None:
    values = sorted(values)
    if not values:
        return None
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _geobody_context(reader: _Reader, geobody_job_id: str) -> dict[str, Any]:
    """Totals of the geobody extraction, recorded only in the geobody job next to this one."""
    if not geobody_job_id or Path(geobody_job_id).name != geobody_job_id or geobody_job_id in {".", ".."}:
        return {}
    jobs_root = reader.job_dir.parent
    relative = "geogg_geobody_summary.json"
    path = jobs_root / geobody_job_id / relative
    try:
        if path.is_symlink() or (jobs_root / geobody_job_id).is_symlink() or not path.is_file():
            return {}
        if jobs_root not in path.resolve().parents or path.stat().st_size > MAX_JSON_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    counts = value.get("counts") if isinstance(value, dict) and isinstance(value.get("counts"), dict) else {}
    if not counts:
        return {}
    reader.used.append({"name": f"../{geobody_job_id}/{relative}", "size_bytes": path.stat().st_size})
    return {"voxels_in_bodies": _num(counts.get("candidate_voxels")), "bodies_kept": _int(counts.get("retained_geobodies")),
            "share_of_cube_in_bodies": _num(counts.get("candidate_fraction_of_valid"))}


# --------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------
def _wells_section(reader: _Reader, summary: dict[str, Any], context: dict[str, Any],
                   findings: list[dict[str, Any]]) -> dict[str, Any]:
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    per_well = [w for w in summary.get("per_well", []) if isinstance(w, dict)]
    selected = [s for s in summary.get("selected_geobodies", []) if isinstance(s, dict)]
    intervals = reader.table(f"{TABLES}/well_geobody_path_intervals.csv")
    hits = {str(r.get("well")): r for r in reader.table(f"{TABLES}/well_geobody_intersections.csv")}
    rows, shares, lengths = [], [], []
    for well in per_well:
        name = str(well.get("well"))
        c = well.get("counts") if isinstance(well.get("counts"), dict) else {}
        total, inside = _num(c.get("trajectory_samples")), _num(c.get("samples_intersecting_geobodies"))
        share = inside / total if total and inside is not None else None
        if share is not None:
            shares.append(share)
        mine = [r for r in intervals if r.get("well") == name]
        spans = [b - a for r in mine if (a := _num(r.get("md_top_m"))) is not None and (b := _num(r.get("md_base_m"))) is not None]
        lengths += spans
        hit = hits.get(name, {})
        rows.append({
            "well": name, "status": well.get("status"), "path_samples": _int(total), "samples_inside_a_body": _int(inside),
            "share_of_path_inside_a_body": _r(share, 2), "bodies_met": well.get("intersected_geobody_ids"),
            "selected_body": well.get("selected_geobody_id"), "separate_stretches": _int(c.get("path_intervals")),
            "typical_stretch_m": _r(_median(spans), 0),
            "inside_from_md_m": _r(hit.get("md_top_m"), 0), "inside_to_md_m": _r(hit.get("md_base_m"), 0),
        })
    unique = sorted({i for w in per_well for i in (w.get("intersected_geobody_ids") or [])})
    chosen = sorted({w.get("selected_geobody_id") for w in per_well if w.get("selected_geobody_id") is not None})
    n_wells = len(per_well)
    total_samples, inside_samples = _num(counts.get("trajectory_samples")), _num(counts.get("samples_intersecting_geobodies"))
    overall = inside_samples / total_samples if total_samples and inside_samples is not None else None
    body = selected[0] if selected else {}
    body_voxels = _num(body.get("catalogue_voxel_count"))
    body_share = body_voxels / context["voxels_in_bodies"] if body_voxels and context.get("voxels_in_bodies") else None
    one_body = n_wells > 1 and len(chosen) == 1
    # Without the extraction totals, fall back on height: no sedimentary body is two seconds of record tall.
    tall = (_num(body.get("gross_time_span_ms")) or 0) >= BACKGROUND_TIME_SPAN_MS
    background = one_body and (body_share >= DOMINANT_SHARE if body_share is not None else tall)
    if one_body:
        text = f"All {n_wells} wells meet the same body (body {chosen[0]}), and no well meets any other body."
        if body_share is not None:
            text += f" That body holds {body_share * 100:.2f}% of all body voxels in the survey."
        elif body.get("gross_time_span_ms") is not None:
            text += f" It spans {_num(body.get('gross_time_span_ms')):.0f} ms of the record."
        findings.append(_finding(
            "same_background_body", "critical" if background else "caution", text,
            "That body is the joined-up background of the sand cube, not a sand body. Passing through it tells you nothing about "
            "the well, so this run has not tied any well to a geobody." if background else
            "Every well being tied to one body means the analysis does not tell the wells apart.",
            [SUMMARY_FILE]))
    base = context.get("share_of_cube_in_bodies")
    if overall is not None and base is not None:
        close = abs(overall - base) <= 0.10
        findings.append(_finding(
            "base_rate", "caution" if close else "info",
            f"{_pct(overall)} of well-path samples lie inside a body, against {_pct(base)} of the whole cube"
            + (f"; per well the share runs from {_pct(min(shares))} to {_pct(max(shares))}." if shares else "."),
            "The wells are inside a body about as often as a randomly placed point would be. The bodies do not concentrate where "
            "the wells are, so the intersections carry no information." if close else
            "The wells meet bodies at a different rate from chance, which is worth examining against the logs.",
            [SUMMARY_FILE, "geogg_geobody_summary.json of the source geobody job"]))
    stretches = [r["separate_stretches"] for r in rows if r["separate_stretches"] is not None]
    typical = _median(lengths)
    if one_body and stretches and typical is not None:
        findings.append(_finding(
            "fragmented_paths", "info",
            f"Each well enters and leaves the body between {min(stretches)} and {max(stretches)} times; a typical stretch inside is "
            f"{typical:.0f} m of measured depth.",
            "Along a well the body is a patchwork of short pieces, which is what a noisy score crossing a threshold looks like. "
            "It is not one continuous unit.",
            [f"{TABLES}/well_geobody_path_intervals.csv"]))
    kept = context.get("bodies_kept")
    if kept and len(unique) < kept:
        findings.append(_finding(
            "small_bodies_untested", "caution",
            f"Only {len(unique)} of the {kept} bodies in the catalogue is met by any well." if len(unique) == 1 else
            f"Only {len(unique)} of the {kept} bodies in the catalogue are met by any well.",
            "The other bodies, including every small candidate that the extraction actually separated, have no well control at all.",
            [SUMMARY_FILE, "geogg_geobody_summary.json of the source geobody job"]))
    reported = _int(counts.get("distinct_intersected_geobodies"))
    if reported is not None and reported != len(unique) and unique:
        findings.append(_finding(
            "count_is_per_well", "info",
            f"The summary field 'distinct_intersected_geobodies' reads {reported}, but only {len(unique)} different body "
            f"ID{'s are' if len(unique) != 1 else ' is'} involved.",
            "The field adds up the per-well counts. Do not quote it as the number of different bodies drilled.",
            [SUMMARY_FILE]))
    findings.append(_finding(
        "no_log_comparison", "caution",
        "This process records where well paths cross bodies. It does not compare the bodies with what the logs found.",
        "An intersection is geometry only. Whether the rock inside the body is sand at the well has not been checked here; that "
        "needs the log labels from Process A laid against these path samples.",
        [SUMMARY_FILE]))
    return {
        "wells_analysed": n_wells, "wells_with_intersections": counts.get("wells_with_intersections"),
        "different_bodies_met": unique, "selected_bodies": chosen, "all_wells_same_body": one_body,
        "selected_body_is_background": background,
        "share_of_path_samples_inside_a_body": _r(overall, 3), "share_of_cube_inside_a_body": _r(base, 3),
        "selection_rule": "For each well the largest body it meets is selected, by voxel count. It is not a quality ranking.",
        "per_well": rows,
        "how_measured": "Intersections are read at recorded path samples, not by tracing the wellbore continuously.",
    }


def _depth_section(reader: _Reader, summary: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    depth = summary.get("depth_characterisation") if isinstance(summary.get("depth_characterisation"), dict) else {}
    selected = [s for s in summary.get("selected_geobodies", []) if isinstance(s, dict)]
    comparison = reader.table(f"{TABLES}/depth_model_comparison.csv")
    samples = reader.table(f"{TABLES}/well_geobody_path_samples.csv")
    models = [m for m in depth.get("available_models", []) if isinstance(m, dict)]
    supports = [s[1] for m in models if isinstance(s := m.get("twt_support_ms"), list) and len(s) == 2 and _num(s[1]) is not None]
    without = [s for s in selected if not (_num(s.get("columns_with_primary_depth")) or 0)]
    if selected and without:
        base = max((_num(s.get("twt_base_ms")) or 0 for s in without), default=0)
        text = f"No depth could be assigned to the selected body for {len(without)} of {len(selected)} wells."
        if supports and base:
            text += (f" The body reaches {base:.0f} ms, while the time-depth models are fitted only down to "
                     f"{min(supports):.0f} to {max(supports):.0f} ms.")
        findings.append(_finding(
            "depth_not_available", "caution", text,
            "There is no top, base or thickness in metres for any body. The process refuses to extend the time-depth models beyond "
            "where they were fitted, which is the correct choice. A body confined to the supported time range would get a depth.",
            [SUMMARY_FILE, f"{TABLES}/selected_well_geobody_analysis.csv"]))
    others = [r for r in comparison if not _truthy(r.get("is_primary_model_for_well"))]
    by_model: dict[str, list[float]] = {}
    for row in others:
        if (value := _num(row.get("depth_residual_rmse_m"))) is not None:
            by_model.setdefault(str(row.get("model_name")), []).append(value)
    spread = [{"model": name, "wells": len(values), "median_difference_m": _r(_median(values), 1), "largest_m": _r(max(values), 1)}
              for name, values in sorted(by_model.items())]
    worst = max((_num(r.get("depth_residual_max_abs_m")) or 0 for r in others), default=0)
    typical = _median([v for values in by_model.values() for v in values])
    if typical is not None:
        findings.append(_finding(
            "depth_model_spread", "caution" if typical >= 5 else "info",
            f"Along the well paths, the alternative time-depth models differ from the one in use by about {typical:.0f} m "
            f"(typical RMS difference), and by up to {worst:.0f} m.",
            "That is the scale of the depth uncertainty from the choice of model alone. It is a spread between models, not a "
            "statistical interval, and the true error could be larger because none of the models is validated.",
            [f"{TABLES}/depth_model_comparison.csv"]))
    ties: dict[str, int] = {}
    strong = 0
    for row in samples:
        ties[str(row.get("trajectory_tie_status"))] = ties.get(str(row.get("trajectory_tie_status")), 0) + 1
        strong += _truthy(row.get("tz_strong_extrapolation"))
    validated = sum(n for k, n in ties.items() if "validated" in k.lower() and not k.lower().startswith("unvalidated")
                    or "measured" in k.lower() or "synthetic" in k.lower())
    if ties and not validated:
        findings.append(_finding(
            "ties_provisional", "caution",
            f"None of the {sum(ties.values()):,} path samples comes from a validated well tie; {strong} lie where the time-depth "
            "relation is strongly extrapolated.",
            "Where each well sits in the seismic cube is an estimate. A well could be placed several seismic samples away from its "
            "true position, which is enough to move it in or out of a thin body.",
            [f"{TABLES}/well_geobody_path_samples.csv"]))
    return {
        "regional_model_in_use": depth.get("selected_regional_model"), "model_per_well": depth.get("well_specific_models"),
        "models_available": [{"model": m.get("model_name"), "type": m.get("source_type"),
                              "fitted_down_to_ms": _r((m.get("twt_support_ms") or [None, None])[1], 0)} for m in models],
        "wells_with_a_depth_for_their_body": len(selected) - len(without), "wells_selected": len(selected),
        "difference_from_model_in_use": spread,
        "path_samples_by_tie_status": ties, "path_samples_strongly_extrapolated": strong,
        "uncertainty_method": depth.get("uncertainty_method"),
        "limitations": [_clip(x, 220) for x in depth.get("limitations") or []],
    }


def _properties_section(summary: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    spec = summary.get("specialist_phit_swt_analysis") if isinstance(summary.get("specialist_phit_swt_analysis"), dict) else {}
    arrays = spec.get("prediction_arrays") if isinstance(spec.get("prediction_arrays"), dict) else {}
    missing = [name.replace("prediction_", "").upper() for name, info in arrays.items()
               if not (isinstance(info, dict) and info.get("sha256") and info.get("shape"))]
    selected = [s for s in summary.get("selected_geobodies", []) if isinstance(s, dict)]
    computed = [s for s in selected if not str(s.get("property_status") or "").upper().startswith("NOT_COMPUTED")]
    status = str(spec.get("status") or "")
    if spec and (status.upper() != "COMPLETED" or not computed):
        findings.append(_finding(
            "properties_not_computed", "caution",
            f"Porosity and water-saturation statistics were not computed for any body (status {status or 'unknown'})"
            + (f"; the missing cube is {', '.join(missing)}." if missing else "."),
            ("No water-saturation cube was registered by the earlier steps, so there was nothing to read. "
             if "SWT" in missing else "")
            + "There is therefore no pore-fluid screening of any kind in this run. Even with the cube, porosity times one minus water "
            "saturation would be a model-on-model screening number, not hydrocarbon volume or net pay.",
            [SUMMARY_FILE, f"{TABLES}/specialist_phit_swt_summary.csv"]))
    return {"status": status or None, "cubes_missing": missing, "model_identity": spec.get("model_identity_status"),
            "bodies_with_statistics": len(computed), "bodies_selected": len(selected),
            "planned_statistics": spec.get("statistics"), "coverage_gate": spec.get("coverage_gate"),
            "weighting": spec.get("weighting"), "interpretation_rule": spec.get("interpretation")}


def _provenance_section(reader: _Reader, summary: dict[str, Any], preflight: dict[str, Any],
                        findings: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [c for c in preflight.get("checks", []) if isinstance(c, dict)]
    failed = [str(c.get("check")) for c in checks if str(c.get("status")).upper() != "PASS"]
    source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
    products = [{"product": p.get("product"), "file": p.get("path"), "present": reader.exists(str(p.get("path")))}
                for p in summary.get("products", []) if isinstance(p, dict)]
    if checks:
        findings.append(_finding(
            "checks_passed" if not failed else "checks_failed", "good" if not failed else "critical",
            f"{len(checks) - len(failed)} of {len(checks)} preflight checks passed" + (f"; failed: {', '.join(failed)}." if failed else
            ": the source jobs link up, and the trajectories, label volume and catalogue match their checksums."),
            "The run used the files it claims to have used. That is a statement about bookkeeping, not about meaning."
            if not failed else "The inputs could not be verified. Do not use the outputs.",
            [PREFLIGHT_FILE]))
    status = str(summary.get("interpretation_status") or "")
    if status:
        findings.append(_finding(
            "exploratory", "info", f"The run labels itself {status}.",
            "It is a geometric exercise on exploratory shapes. It makes no reservoir, fluid, net-pay or reserves claim.",
            [SUMMARY_FILE]))
    missing = [p["file"] for p in products if not p["present"]]
    if missing:
        findings.append(_finding(
            "files_missing", "info", f"{len(missing)} of {len(products)} output tables named in the summary are not in this job folder.",
            "The summary describes them, but they could not be found here. Large tables may not have been copied.",
            [SUMMARY_FILE]))
    return {"preflight_status": preflight.get("status"), "checks": [{"check": c.get("check"), "status": c.get("status")} for c in checks],
            "source_jobs": {"foundation": source.get("foundation_job_id"), "inference": source.get("inference_job_id"),
                            "volume_products": source.get("source_volume_job_id"), "geobody": source.get("source_geobody_job_id")},
            "trajectory_source": source.get("trajectory_source"),
            "well_selection": (summary.get("well_selection") or {}).get("mode"),
            "output_tables": products, "limitations": [_clip(x, 220) for x in summary.get("limitations") or []]}


def _well_card(reader: _Reader, wells: dict[str, Any], depth: dict[str, Any], well: str) -> dict[str, Any]:
    wanted = well.strip().lower()
    row = next((r for r in wells["per_well"] if str(r["well"]).lower() == wanted), None)
    if row is None:
        return {"well": well, "found": False, "available_wells": sorted(str(r["well"]) for r in wells["per_well"])}
    name = str(row["well"])
    samples = [r for r in reader.table(f"{TABLES}/well_geobody_path_samples.csv") if r.get("well") == name]
    ties: dict[str, int] = {}
    for sample in samples:
        ties[str(sample.get("trajectory_tie_status"))] = ties.get(str(sample.get("trajectory_tie_status")), 0) + 1
    comparison = [{"model": r.get("model_name"), "in_use": _truthy(r.get("is_primary_model_for_well")),
                   "rms_difference_m": _r(r.get("depth_residual_rmse_m"), 1), "largest_difference_m": _r(r.get("depth_residual_max_abs_m"), 1)}
                  for r in reader.table(f"{TABLES}/depth_model_comparison.csv") if r.get("well") == name]
    reading = ("This well passes through the survey-wide background body, like every other well. The intersection says nothing "
               "specific about it." if wells["selected_body_is_background"] else
               "This well meets the body listed. Check the logs over the stretches inside it before drawing any conclusion.")
    return {**row, "found": True, "reading": reading,
            "time_depth_model_in_use": (depth.get("model_per_well") or {}).get(name),
            "tie_status_of_path_samples": ties,
            "samples_strongly_extrapolated": sum(_truthy(s.get("tz_strong_extrapolation")) for s in samples),
            "time_depth_models_compared": comparison}


# --------------------------------------------------------------------------------------------
# Verdict and next steps
# --------------------------------------------------------------------------------------------
def _verdict(status: str, sections: dict[str, Any], failed_checks: bool) -> dict[str, Any]:
    wells, depth, props = sections["wells"], sections["depth"], sections["properties"]
    if str(status).lower() not in {"succeeded", "success", "completed"}:
        level, plain = "did_not_finish", f"The job status is '{status}', so this is not a completed result."
    elif not wells["wells_analysed"]:
        level, plain = "incomplete", "The run finished but no per-well result was recorded."
    elif failed_checks:
        level, plain = "needs_attention", "A preflight check failed, so the inputs could not be verified. Do not use the outputs."
    else:
        level = "usable_with_caveats"
        if wells["selected_body_is_background"]:
            plain = (f"The run completed for all {wells['wells_analysed']} wells, but it produced no usable well-to-geobody tie. "
                     "Every well meets the same survey-wide background body.")
        else:
            plain = f"The run completed for {wells['wells_analysed']} wells, which meet {len(wells['different_bodies_met'])} different bodies."
        gaps = []
        if depth["wells_selected"] and not depth["wells_with_a_depth_for_their_body"]:
            gaps.append("no depth")
        if props["bodies_selected"] and not props["bodies_with_statistics"]:
            gaps.append("no porosity or saturation statistics")
        if gaps:
            plain += f" It also delivered {' and '.join(gaps)} for any body."
    assessment = ("background_body_only" if wells["selected_body_is_background"] else
                  "single_body" if wells["all_wells_same_body"] else "distinct_bodies")
    return {"level": level, "plain": plain, "assessment": assessment, "wells_analysed": wells["wells_analysed"],
            "note": "A completed run and a meaningful result are separate questions."}


def _next_steps(findings: list[dict[str, Any]], verdict: dict[str, Any]) -> list[str]:
    ids = {f["id"] for f in findings}
    steps: list[str] = []
    if verdict["level"] in {"did_not_finish", "incomplete", "needs_attention"}:
        steps.append("Resolve the failed or missing items first.")
    if "same_background_body" in ids:
        steps.append("Re-run the geobody extraction with a higher low threshold, then repeat this analysis; until then it cannot be informative.")
    if "no_log_comparison" in ids:
        steps.append("Compare the in-body stretches of each well with the sand and shale labels from the logs. That is the real test of a body.")
    if "depth_not_available" in ids:
        steps.append("Restrict bodies to the time range the time-depth models cover, so that depths can be assigned.")
    if "properties_not_computed" in ids:
        steps.append("Before producing a water-saturation cube, check its model's held-out skill in the Process B analysis; a weak model is not worth mapping.")
    if "ties_provisional" in ids:
        steps.append("Obtain measured checkshots or sonic and density logs to firm up where the wells sit in the cube.")
    if "small_bodies_untested" in ids:
        steps.append("Treat the small bodies as untested leads; none of them has a well through it.")
    return steps[:6]


# --------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------
def analyse_job_dir(job_dir: Path, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyse one already-located job folder. Pure file inspection; no backend access."""
    parameters = parameters or {}
    reader = _Reader(job_dir)
    metadata, result = reader.json("job.json"), reader.json("result.json")
    summary = reader.json(SUMMARY_FILE) or reader.json(f"{S13}/well_geobody_analysis_manifest.json")
    preflight = reader.json(PREFLIGHT_FILE)
    job_id = str(parameters.get("job_id") or metadata.get("job_id") or job_dir.name)
    script = str(metadata.get("python_script") or result.get("python_script") or "")
    status = str(result.get("status") or metadata.get("status") or summary.get("status") or "unknown")
    source = summary.get("source") if isinstance(summary.get("source"), dict) else (
        preflight.get("source") if isinstance(preflight.get("source"), dict) else {})
    base = {
        "skill": SKILL_NAME, "job_id": job_id, "python_script": script or None,
        "operation": summary.get("operation") or metadata.get("operation"),
        "target_well": (str(metadata.get("target_well") or "").strip() or None), "status": status,
        "created_at": metadata.get("created_at") or result.get("created_at"),
        "source_geobody_job_id": source.get("source_geobody_job_id"), "source_volume_job_id": source.get("source_volume_job_id"),
        "source_inference_job_id": source.get("inference_job_id"), "foundation_job_id": source.get("foundation_job_id"),
    }
    recognised = script == EXPECTED_SCRIPT or EXPECTED_STAGE in {str(summary.get("stage")), str(preflight.get("stage"))}
    if not recognised or not summary:
        reason = (f"This folder was not produced by Process E (expected script '{EXPECTED_SCRIPT}')." if not recognised else
                  f"'{SUMMARY_FILE}' is missing or unreadable, so there is nothing to analyse.")
        if recognised and not summary:
            progress = reader.json(f"{S13}/progress.json")
            if progress:
                reason += (f" The progress file reports phase '{progress.get('phase')}' with {progress.get('completed_wells')} of "
                           f"{progress.get('total_wells')} wells done.")
        return {**base, "applicable": False, "summary": reason, "findings": [reason], "artifacts": reader.used,
                "skipped": reader.skipped,
                "limitations": ["Use the generic processing_output_folder_analysis skill to inventory this folder."]}

    context = _geobody_context(reader, str(base["source_geobody_job_id"] or ""))
    findings: list[dict[str, Any]] = []
    sections = {
        "wells": _wells_section(reader, summary, context, findings),
        "depth": _depth_section(reader, summary, findings),
        "properties": _properties_section(summary, findings),
        "provenance": _provenance_section(reader, summary, preflight, findings),
    }
    verdict = _verdict(status, sections, any(f["id"] == "checks_failed" for f in findings))

    def rank(finding: dict[str, Any]) -> int:
        return next((i for i, prefix in enumerate(PRIORITY) if finding["id"].startswith(prefix)), len(PRIORITY))

    findings.sort(key=lambda f: (f["id"] != "checks_failed", rank(f)))
    takeaways = [f for f in findings if f["level"] in {"critical", "caution"} and f["id"].startswith((*TAKEAWAY_IDS, "checks_failed"))]
    takeaways = sorted(takeaways, key=lambda f: f["level"] != "critical")[:4]

    focus_raw = str(parameters.get("focus") or "").strip().lower().replace(" ", "_").replace("-", "_")
    focus = FOCUS_ALIASES.get(focus_raw) or FOCUS_ALIASES.get(focus_raw.replace("stage_", "").replace("stage", ""), "overview")
    if focus == "all":
        details, shown = dict(sections), findings
    elif focus in sections:
        details = {focus: sections[focus]}
        shown = [f for f in findings if f["id"].startswith(_FINDING_FOCUS.get(focus, ()))] or findings[:OVERVIEW_FINDINGS]
    elif focus == "glossary":
        details, shown = {}, []
    else:
        details, shown = {}, findings[:OVERVIEW_FINDINGS]

    wells = sections["wells"]
    response: dict[str, Any] = {
        **base, "applicable": True, "focus": focus,
        "verdict": verdict, "summary": verdict["plain"],
        "takeaways": takeaways,
        "at_a_glance": {
            "wells_analysed": wells["wells_analysed"], "different_bodies_met": len(wells["different_bodies_met"]),
            "all_wells_same_body": wells["all_wells_same_body"],
            "share_of_path_inside_a_body": wells["share_of_path_samples_inside_a_body"],
            "share_of_cube_inside_a_body": wells["share_of_cube_inside_a_body"],
            "wells_with_a_depth_for_their_body": sections["depth"]["wells_with_a_depth_for_their_body"],
            "bodies_with_porosity_or_saturation_statistics": sections["properties"]["bodies_with_statistics"],
        },
        "key_findings": shown,
        "findings": [f["observed"] for f in shown],
        "next_steps": _next_steps(findings, verdict),
        "details": details,
        "findings_not_shown": len(findings) - len(shown) if focus == "overview" else 0,
        "more_detail": "Call again with focus = wells, depth, properties, provenance or all, or with well = <name>.",
    }
    well = str(parameters.get("well") or "").strip()
    if well:
        response["well_card"] = _well_card(reader, wells, sections["depth"], well)
    if focus in {"all", "glossary"}:
        response["glossary"] = GLOSSARY
    if focus == "overview":
        response["suggested_questions"] = SUGGESTED_QUESTIONS
    response["artifacts"] = [{**item, "role": "evidence read by this analysis"} for item in reader.used]
    response["skipped"] = reader.skipped
    response["limitations"] = [
        "Numbers are read from the run's summary, preflight and tables; nothing is recomputed and no volume is opened.",
        "Intersections are geometric. This skill does not compare bodies with well logs, because the run does not.",
        "Differences between time-depth models are a spread, not a statistical interval; none of the models is validated.",
        "Plain-language meanings are fixed expert reading rules applied to the observed numbers, not proof of a geological cause.",
    ]
    if not context:
        response["limitations"].append("The totals of the geobody extraction live in the geobody job, which could not be read from "
                                       "here, so the well intersections are not compared with the cube-wide base rate.")
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
    return script == EXPECTED_SCRIPT
