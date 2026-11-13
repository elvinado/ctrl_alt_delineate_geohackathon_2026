"""Expert, deterministic reading of one Process F (GeoGG prospect definition, stage 14) run folder.

Process F turns each stretch of a well inside a candidate geobody into a "prospect": the part of
the body within that time window. It measures each prospect's geometry, tries to compute gross
rock volume, and characterises it with the well logs that cross it (net-to-gross, porosity,
water saturation). Qualification gates decide whether any volume may be reported.

This handler reads the run's own summary, preflight and tables and applies fixed, documented
reading rules. It never calls an LLM, never invents a value and never opens a voxel or point
file. Every file it used is listed under ``artifacts``. It imports no other skill.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

SKILL_NAME = "process_f_prospect_output_analysis"
EXPECTED_SCRIPT = "process_F_prospect.py"
EXPECTED_STAGE = "14_prospect"
SUMMARY_FILE = "geogg_prospect_summary.json"
PREFLIGHT_FILE = "geogg_prospect_preflight.json"
TABLE_DIRS = ("tables", "stages/14_prospect/tables")

MAX_JSON_BYTES = 8_000_000
MAX_CSV_BYTES = 30_000_000
MAX_CSV_ROWS = 60_000
MAX_TEXT = 420
OVERVIEW_FINDINGS = 8
LISTED_PROSPECTS = 12
SURVEY_SCALE_SHARE = 0.90

# Defaults for the pragmatic voxel-based estimate when the run does not expose a vertical cell size.
# A 4 ms sample interval at 2,500 m/s interval velocity gives a 5 m voxel height (two-way time -> depth).
DEFAULT_SAMPLE_INTERVAL_MS = 4.0
DEFAULT_INTERVAL_VELOCITY_MS = 2500.0

FOCUS_ALIASES: dict[str, str] = {
    "": "overview", "overview": "overview", "summary": "overview", "all": "all", "full": "all",
    "prospects": "prospects", "prospect": "prospects", "list": "prospects", "geometry": "prospects", "ranking": "prospects",
    "volumes": "volumes", "volume": "volumes", "grv": "volumes", "hcpv": "volumes", "volumetrics": "volumes", "gates": "volumes",
    "logs": "logs", "wells": "logs", "ntg": "logs", "net_to_gross": "logs", "porosity": "logs", "saturation": "logs",
    "fluid": "logs", "cutoffs": "logs", "uncertainty": "logs",
    "provenance": "provenance", "preflight": "provenance", "products": "provenance", "files": "provenance", "storage": "provenance",
    "glossary": "glossary", "14": "overview",
}

GLOSSARY: dict[str, str] = {
    "prospect": "Here, the part of a candidate geobody inside one time window around a well. Not a drillable prospect in the industry sense.",
    "anchor interval": "A continuous stretch of a well path inside a body. Each one defines one prospect.",
    "GRV": "Gross rock volume: the total volume of rock inside the prospect outline, in cubic metres.",
    "net-to-gross (NTG)": "The share of the logged thickness that counts as reservoir rock under the chosen cutoffs.",
    "cutoffs": "The rules that decide what counts as net rock, here a maximum shale volume and a minimum porosity.",
    "PHIT": "Total porosity: the fraction of the rock that is pore space. It is not effective porosity.",
    "SWT": "Total water saturation: the fraction of the pore space filled with water. High values mean water-bearing rock.",
    "HCPV": "Hydrocarbon pore volume: GRV x NTG x porosity x (1 - water saturation). A scenario number, never a resource estimate.",
    "voxel-based GRV estimate": "A pragmatic gross rock volume from voxels x map-cell area x voxel height, with the map units assumed metres and a sample interval and velocity for the voxel height. An order-of-magnitude screening figure that bypasses the run's gate, not the gated volume.",
    "qualification gate": "A rule that stops a number being reported when its inputs are not good enough.",
    "GEOMETRY_ONLY": "The lowest reporting level: only shape and size in grid units may be reported, no volumes.",
    "no-contact fill": "The assumption that the whole body is filled, with no fluid contact. It is the most optimistic case.",
    "survey-scale object": "A body so wide it covers most of the survey. It is background, not a trap.",
    "block bootstrap": "Re-drawing blocks of log samples many times to see how much a number wobbles. It covers sampling noise only.",
    "P90 / P50 / P10": "Low, middle and high cases: the value exceeded in 90%, 50% and 10% of the re-draws.",
}

SUGGESTED_QUESTIONS: list[str] = [
    "Did this run find any real prospects?",
    "Why are there no volumes?",
    "What do the well logs say about the rock inside the bodies?",
    "Which prospects are compact enough to look at?",
    "Estimate the voxel-based GRV and HCPV for the compact prospects.",
    "Tell me about one prospect by its ID.",
    "What has to be fixed before the gated volumes can be reported?",
]

# Order in which findings are surfaced (first match on the id prefix wins).
PRIORITY: list[str] = [
    "checks_failed", "background_slabs", "overlap", "volumes_withheld", "volumes_reported", "volumes_estimated", "ntg_evidence",
    "fluid_evidence", "compact_prospects",
    "units_unconfirmed", "depth_invalid", "upside_assumptions", "cutoff_sensitivity", "uncertainty_partial", "storage_cost",
    "exploratory",
]
TAKEAWAY_IDS: tuple[str, ...] = ("checks_failed", "background_slabs", "overlap", "volumes_withheld", "volumes_reported", "ntg_evidence",
                                 "fluid_evidence", "units_unconfirmed")
_FINDING_FOCUS: dict[str, tuple[str, ...]] = {
    "prospects": ("background_slabs", "overlap", "compact_prospects", "depth_invalid"),
    "volumes": ("volumes_withheld", "volumes_reported", "volumes_estimated", "units_unconfirmed", "depth_invalid", "upside_assumptions", "overlap"),
    "logs": ("ntg_evidence", "fluid_evidence", "cutoff_sensitivity", "uncertainty_partial"),
    "provenance": ("storage_cost", "exploratory", "units_unconfirmed", "checks_failed"),
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


def _quantile(values: list[float], q: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    position = q * (len(values) - 1)
    low = int(math.floor(position))
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def _table(reader: _Reader, name: str) -> tuple[list[dict[str, str]], str]:
    """Tables sit in 'tables/' or 'stages/14_prospect/tables/' depending on how the job folder was laid out."""
    for folder in TABLE_DIRS:
        if reader.exists(f"{folder}/{name}"):
            return reader.table(f"{folder}/{name}"), f"{folder}/{name}"
    return [], f"{TABLE_DIRS[0]}/{name}"


def _sand_share_at_wells(reader: _Reader, foundation_job_id: str) -> float | None:
    """Share of the logged interval labelled sand, recorded only in the foundation job next to this one."""
    if not foundation_job_id or Path(foundation_job_id).name != foundation_job_id or foundation_job_id in {".", ".."}:
        return None
    jobs_root = reader.job_dir.parent
    relative = "geogg_foundations_summary.json"
    path = jobs_root / foundation_job_id / relative
    try:
        if path.is_symlink() or (jobs_root / foundation_job_id).is_symlink() or not path.is_file():
            return None
        if jobs_root not in path.resolve().parents or path.stat().st_size > MAX_JSON_BYTES:
            return None
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    stage = next((s for s in value.get("stages", []) if isinstance(s, dict) and s.get("stage_id") == "02_labels"), {})
    metrics = stage.get("metrics") if isinstance(stage.get("metrics"), dict) else {}
    parts = {k[len("dev_md_m_"):]: _num(v) or 0.0 for k, v in metrics.items() if k.startswith("dev_md_m_")}
    total = sum(parts.values())
    if not total:
        return None
    reader.used.append({"name": f"../{foundation_job_id}/{relative}", "size_bytes": path.stat().st_size})
    return sum(v for k, v in parts.items() if k.endswith("_sand")) / total


def _prospects(reader: _Reader) -> tuple[list[dict[str, Any]], list[str]]:
    definitions, path_d = _table(reader, "prospect_definitions.csv")
    volumetrics, path_v = _table(reader, "prospect_volumetrics_deterministic.csv")
    flags = {str(r.get("prospect")): r for r in volumetrics}
    out = []
    for row in definitions:
        name = str(row.get("prospect"))
        vol = flags.get(name, {})
        reasons = [x for x in str(row.get("level_reasons") or vol.get("level_reasons") or "").split(";") if x]
        top, base = _num(row.get("twt_top_ms")), _num(row.get("twt_base_ms"))
        out.append({
            "prospect": name, "anchor_well": row.get("anchor_well"), "geobody_id": _int(row.get("geobody_id")),
            "voxels": _int(row.get("n_voxels")) or 0, "map_cells": _int(row.get("n_occupied_columns")),
            "share_of_live_survey": _r(vol.get("footprint_fraction_of_live_survey"), 4),
            "twt_range_ms": [top, base], "time_thickness_ms": (base - top) if None not in (top, base) else None,
            "window_ms": [_num(row.get("window_twt_top_ms")), _num(row.get("window_twt_base_ms"))],
            "centre": {"inline": _r(row.get("centroid_il"), 0), "crossline": _r(row.get("centroid_xl"), 0)},
            "survey_scale": _truthy(vol.get("flag_survey_scale_object")) or "survey_scale_object" in reasons,
            "reporting_level": row.get("reporting_level") or vol.get("reporting_level"), "reasons": reasons,
            "logged_thickness_m": _r(row.get("gross_m"), 1), "net_thickness_m": _r(row.get("net_m"), 1),
            "net_to_gross": _r(row.get("ntg")), "porosity_of_net": _r(row.get("phit_net")),
            "water_saturation_of_net": _r(row.get("swt_net_pv")), "log_samples": _int(row.get("n_log_samples")),
            "wells_with_net": _int(vol.get("n_wells_with_net")),
            "wells_water_bearing": [w for w in str(vol.get("wells_water_bearing") or "").split(";") if w],
            "wells_low_water_saturation": [w for w in str(vol.get("wells_low_swt") or "").split(";") if w],
            "grv_m3": _num(vol.get("grv_m3")), "hcpv_m3": _num(vol.get("hcpv_scenario_uniform_m3")),
            "hcpv_status": row.get("status_hcpv") or vol.get("status_hcpv"),
            "window_truncated": _truthy(row.get("window_truncated")), "touches_survey_edge": _truthy(row.get("touches_survey_edge")),
        })
    return out, [path_d, path_v]


def _apply_voxel_estimates(summary: dict[str, Any], prospects: list[dict[str, Any]],
                           parameters: dict[str, Any]) -> dict[str, Any]:
    """Compute a pragmatic voxel-based GRV (and HCPV screening scenario) for every prospect that has
    the inputs, and record the assumptions used.

    GRV_est = voxels x median map-cell area x voxel height, where voxel height = (sample interval / 1000)
    x (interval velocity / 2) converts two-way time to depth. HCPV_est = GRV_est x NTG x PHIT x (1 - SWT).

    Each input is taken from, in order: an explicit value in ``parameters``, a value the run recorded,
    then a documented default. The returned dict reports which inputs were assumed so the caller can
    label the number honestly. This is a screening figure, not a gated resource number: the run's own
    qualification gates are left untouched and ``grv_m3`` / ``hcpv_m3`` keep reporting what the run wrote.
    """
    qc = summary.get("qc") if isinstance(summary.get("qc"), dict) else {}
    area = qc.get("cell_area") if isinstance(qc.get("cell_area"), dict) else {}
    depth = summary.get("depth_model_selection") if isinstance(summary.get("depth_model_selection"), dict) else {}
    geom = summary.get("cube_geometry") if isinstance(summary.get("cube_geometry"), dict) else {}

    cell_area = _num(parameters.get("cell_area_m2")) or _num(area.get("median_cell_area_m2"))
    units_confirmed = qc.get("xy_coordinate_units_confirmed") is True
    sample_run = _num(geom.get("twt_sample_interval_ms")) or _num(summary.get("twt_sample_interval_ms"))
    sample_ms = _num(parameters.get("sample_interval_ms")) or sample_run or DEFAULT_SAMPLE_INTERVAL_MS
    vel_run = (_num(depth.get("interval_velocity_ms")) or _num(depth.get("velocity_m_s")) or _num(depth.get("v0_m_s")))
    velocity = _num(parameters.get("interval_velocity_ms")) or vel_run or DEFAULT_INTERVAL_VELOCITY_MS
    voxel_height_m = (sample_ms / 1000.0) * (velocity / 2.0) if (sample_ms and velocity) else None

    for p in prospects:
        voxels = p.get("voxels") or 0
        grv = voxels * cell_area * voxel_height_m if (cell_area and voxel_height_m and voxels) else None
        p["grv_voxel_est_m3"] = round(grv) if grv else None
        ntg, phit, swt = p.get("net_to_gross"), p.get("porosity_of_net"), p.get("water_saturation_of_net")
        hcpv = grv * ntg * phit * (1.0 - swt) if (grv is not None and None not in (ntg, phit, swt)) else None
        p["hcpv_voxel_est_m3"] = round(hcpv) if hcpv is not None else None

    return {
        "method": "voxels x map-cell area x voxel height; HCPV = GRV x NTG x PHIT x (1 - SWT)",
        "cell_area_m2": _r(cell_area, 1) if cell_area else None,
        "xy_units_assumed_metres": bool(cell_area) and not units_confirmed,
        "sample_interval_ms": _r(sample_ms, 2),
        "sample_interval_assumed": _num(parameters.get("sample_interval_ms")) is None and sample_run is None,
        "interval_velocity_m_s": _r(velocity, 0),
        "interval_velocity_assumed": _num(parameters.get("interval_velocity_ms")) is None and vel_run is None,
        "voxel_height_m": _r(voxel_height_m, 2) if voxel_height_m else None,
        "caveat": "Unvalidated screening scenario, not a resource estimate. Prospects overlap; never add these together.",
    }


# --------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------
def _prospects_section(prospects: list[dict[str, Any]], evidence: list[str], findings: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(prospects)
    wide = [p for p in prospects if p["survey_scale"]]
    compact = sorted((p for p in prospects if not p["survey_scale"]), key=lambda p: -p["voxels"])
    bodies = sorted({p["geobody_id"] for p in prospects if p["geobody_id"] is not None})
    shares = [p["share_of_live_survey"] for p in prospects if p["share_of_live_survey"] is not None]
    background = bool(n) and len(wide) / n >= SURVEY_SCALE_SHARE
    if n and wide:
        text = f"{len(wide)} of {n} prospects are flagged as survey-scale objects"
        if shares:
            text += f"; the typical prospect covers {_pct(_quantile(shares, 0.5))} of the live survey area"
        if len(bodies) == 1:
            text += f". All {n} are cut from the same body (body {bodies[0]})"
        findings.append(_finding(
            "background_slabs", "critical" if background else "caution", text + ".",
            "These are not prospects. They are time slices of one survey-wide background body, one slice per stretch of well path. "
            "Nothing here outlines a trap or a sand body." if background else
            "Survey-wide objects are background, not traps. Set them aside and look only at the compact ones.",
            evidence))
    windows = sorted((p["window_ms"][0], p["window_ms"][1]) for p in prospects if None not in p["window_ms"])
    covered, current = 0.0, None
    for top, base in windows:
        if current is None or top > current[1]:
            covered += (current[1] - current[0]) if current else 0.0
            current = [top, base]
        else:
            current[1] = max(current[1], base)
    covered += (current[1] - current[0]) if current else 0.0
    stacked = sum(base - top for top, base in windows)
    if covered and stacked / covered >= 1.5:
        findings.append(_finding(
            "overlap", "critical" if background else "caution",
            f"The prospects' time windows add up to {stacked:,.0f} ms but together cover only {covered:,.0f} ms of the record, so "
            f"the same rock is counted about {stacked / covered:.0f} times over.",
            "Prospects overlap heavily. Their voxel counts, thicknesses and any future volumes must never be added together.",
            evidence[:1]))
    if compact:
        findings.append(_finding(
            "compact_prospects", "info",
            f"{len(compact)} prospects are not survey-scale: {', '.join(p['prospect'] for p in compact[:LISTED_PROSPECTS])}"
            + (f" and {len(compact) - LISTED_PROSPECTS} more" if len(compact) > LISTED_PROSPECTS else "")
            + f". They range from {min(p['voxels'] for p in compact):,} to {max(p['voxels'] for p in compact):,} voxels.",
            "These are the only objects of a size a prospect could have. They are still pieces of the same body and carry every "
            "caveat of this run, so treat them as places to look, not as prospects.",
            evidence))
    keep = ("prospect", "anchor_well", "voxels", "map_cells", "share_of_live_survey", "twt_range_ms", "centre", "net_to_gross",
            "porosity_of_net", "water_saturation_of_net", "grv_voxel_est_m3", "hcpv_voxel_est_m3", "wells_with_net", "reasons")
    return {
        "prospects": n, "different_bodies": bodies, "survey_scale_prospects": len(wide), "compact_prospects": len(compact),
        "mostly_background": background,
        "typical_share_of_live_survey": _r(_quantile(shares, 0.5), 3) if shares else None,
        "time_covered_ms": _r(covered, 0), "time_windows_summed_ms": _r(stacked, 0),
        "prospects_per_anchor_well": {w: sum(1 for p in prospects if p["anchor_well"] == w)
                                      for w in sorted({str(p["anchor_well"]) for p in prospects})},
        "compact_list": [{k: p[k] for k in keep} for p in compact[:LISTED_PROSPECTS]],
        "largest_survey_scale": [{k: p[k] for k in keep[:6]} for p in sorted(wide, key=lambda p: -p["voxels"])[:3]],
        "how_defined": "One prospect per continuous stretch of well path inside a body: the part of the body within that time window.",
        "units": "Sizes are in voxels, grid cells and milliseconds. Areas are in unconfirmed coordinate units.",
    }


def _volumes_section(reader: _Reader, summary: dict[str, Any], prospects: list[dict[str, Any]],
                     findings: list[dict[str, Any]], assumptions: dict[str, Any]) -> dict[str, Any]:
    qc = summary.get("qc") if isinstance(summary.get("qc"), dict) else {}
    area = qc.get("cell_area") if isinstance(qc.get("cell_area"), dict) else {}
    grv_rows, path = _table(reader, "prospect_grv_by_depth_model.csv")
    statuses: dict[str, int] = {}
    for row in grv_rows:
        statuses[str(row.get("status"))] = statuses.get(str(row.get("status")), 0) + 1
    n = len(prospects)
    with_grv = sum(1 for p in prospects if p["grv_m3"] is not None)
    with_hcpv = sum(1 for p in prospects if p["hcpv_m3"] is not None)
    levels: dict[str, int] = {}
    reasons: dict[str, int] = {}
    for p in prospects:
        levels[str(p["reporting_level"])] = levels.get(str(p["reporting_level"]), 0) + 1
        for reason in p["reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    if n and not with_grv and not with_hcpv:
        blocked = max(statuses, key=statuses.get) if statuses else str(summary.get("status"))
        findings.append(_finding(
            "volumes_withheld", "caution",
            f"No rock volume and no hydrocarbon pore volume is reported for any of the {n} prospects "
            f"(status {summary.get('status')}; every volume row reads {blocked}). All {levels.get('GEOMETRY_ONLY', 0)} prospects are "
            "at reporting level GEOMETRY_ONLY.",
            "The run's safety gates worked: the official, gated volume stays withheld because map units are unconfirmed and depth is "
            "flagged invalid. A pragmatic voxel-based estimate is provided separately below for screening only; it is not the gated number.",
            [SUMMARY_FILE, path]))
    side = math.sqrt(_num(area.get("median_cell_area_m2")) or 0.0)
    if with_grv or with_hcpv:
        findings.append(_finding(
            "volumes_reported", "caution",
            f"A gross rock volume is reported for {with_grv} of {n} prospects and a hydrocarbon pore volume scenario for {with_hcpv}.",
            "Read these as screening scenarios under the most optimistic fill assumption. They are not resource estimates, they "
            "must not be added across prospects, and they inherit every caveat of the bodies they are cut from.",
            [path]))
    if area and (qc.get("xy_coordinate_units_confirmed") is False or str(area.get("units_status") or "").upper().startswith("UNCONFIRMED")):
        findings.append(_finding(
            "units_unconfirmed", "caution",
            f"The map coordinate units are unconfirmed. A typical grid cell measures {_fmt(area.get('median_cell_area_m2'), 0)} in "
            f"squared coordinate units, which would be a {side:.0f} m by {side:.0f} m cell only if the coordinates are in metres.",
            "Area and volume cannot be stated until someone confirms the units. This is the gate that blocks gross rock volume, and "
            "it needs a statement from the data owner, not a re-run.",
            [SUMMARY_FILE]))
    invalid = reasons.get("depth_invalid", 0)
    if n and invalid:
        findings.append(_finding(
            "depth_invalid", "caution", f"Depth is flagged invalid for {invalid} of {n} prospects.",
            "The time-to-depth conversion is provisional, so the depths listed for prospects are indicative only and cannot support a volume.",
            [path]))
    fills = {str(r.get("fill_assumption")) for r in _table(reader, "prospect_volumetrics_deterministic.csv")[0]}
    truncated = sum(1 for p in prospects if p["window_truncated"])
    if any("NO_CONTACT" in f for f in fills):
        findings.append(_finding(
            "upside_assumptions", "info",
            f"The volume method assumes the whole body is filled, with no fluid contact, and uses total porosity. {truncated} of {n} "
            "prospects are cut off by their time window.",
            "Even when the gates open, the result will be a most-optimistic scenario built on uniform properties, not a resource estimate.",
            [SUMMARY_FILE]))
    estimated = [p for p in prospects if p.get("grv_voxel_est_m3") is not None]
    est_compact = [p for p in estimated if not p["survey_scale"]]
    with_hcpv_est = sum(1 for p in estimated if p.get("hcpv_voxel_est_m3") is not None)
    compact_grv = [p["grv_voxel_est_m3"] for p in est_compact]
    if estimated:
        extra = ""
        if compact_grv:
            extra = (f" For the {len(est_compact)} compact prospects the estimate runs about "
                     f"{min(compact_grv):,.0f} to {max(compact_grv):,.0f} m3 of gross rock.")
        findings.append(_finding(
            "volumes_estimated", "info",
            f"A pragmatic voxel-based GRV was estimated for {len(estimated)} of {n} prospects "
            f"(voxels x map-cell area x voxel height), and a voxel-based HCPV screening scenario for {with_hcpv_est}." + extra,
            "This is the hackathon screening number: it fills the gap the gate leaves. It assumes the map coordinates are in metres"
            + (" and uses an assumed sample interval and velocity for the voxel height" if assumptions.get("sample_interval_assumed")
               or assumptions.get("interval_velocity_assumed") else "")
            + ", so read it as order-of-magnitude only. Survey-scale prospects produce meaningless (survey-wide) numbers; use the "
            "compact ones. Never add estimates across prospects, they overlap.",
            [SUMMARY_FILE, path]))
    return {
        "run_status": summary.get("status"), "prospects_with_rock_volume": with_grv, "prospects_with_hcpv": with_hcpv,
        "hcpv_status": qc.get("hcpv_status"), "reporting_levels": levels, "gate_reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "volume_rows_by_status": statuses,
        "cell_area": {"median_in_squared_coordinate_units": _r(area.get("median_cell_area_m2"), 1), "method": area.get("method"),
                      "units_status": area.get("units_status"), "units_confirmed": qc.get("xy_coordinate_units_confirmed")},
        "depth_model": (summary.get("depth_model_selection") or {}).get("model"),
        "fill_assumption": sorted(fills)[:2], "hcpv_formula": "GRV x net-to-gross x total porosity x (1 - water saturation)",
        "voxel_based_estimate": {
            **assumptions,
            "prospects_with_grv_estimate": len(estimated),
            "compact_prospects_with_estimate": len(est_compact),
            "compact_grv_est_m3_range": [round(min(compact_grv)), round(max(compact_grv))] if compact_grv else None,
            "prospects_with_hcpv_estimate": with_hcpv_est,
            "note": "Per-prospect values are in the compact-prospect table and the prospect card. Do not sum across prospects; they overlap.",
        },
    }


def _logs_section(reader: _Reader, prospects: list[dict[str, Any]], baseline: float | None,
                  findings: list[dict[str, Any]]) -> dict[str, Any]:
    logged = [p for p in prospects if p["net_to_gross"] is not None]
    ntg = [p["net_to_gross"] for p in logged]
    phit = [p["porosity_of_net"] for p in prospects if p["porosity_of_net"] is not None]
    swt = [p["water_saturation_of_net"] for p in prospects if p["water_saturation_of_net"] is not None]
    n = len(prospects)
    median = _quantile(ntg, 0.5)
    _, path = _table(reader, "prospect_definitions.csv")
    if median is not None:
        text = (f"Inside the prospects, the well logs show a typical net-to-gross of {median:.2f} "
                f"(middle 80% of prospects: {_quantile(ntg, 0.1):.2f} to {_quantile(ntg, 0.9):.2f}; {len(logged)} prospects with logs).")
        if baseline is not None:
            text += f" Across all logged rock at the wells, sand makes up {baseline:.2f}."
            lift = median / baseline if baseline else None
            meaning = (f"Rock inside the candidate sand bodies is about {lift:.1f} times as sandy as the wells overall. "
                       + ("That is a real but small enrichment: " if lift and lift >= 1.2 else "That is no meaningful enrichment: ")
                       + f"roughly {_pct(1 - median)} of what the bodies call sand is not net sand at the wells.")
        else:
            meaning = (f"Roughly {_pct(1 - median)} of the rock the bodies call sand is not net sand at the wells. This is the first "
                       "direct check of the bodies against the logs.")
        findings.append(_finding("ntg_evidence", "caution" if median < 0.5 else "info", text, meaning, [path]))
    water = sum(1 for p in prospects if "water_bearing_at_wells" in p["reasons"])
    mixed = sum(1 for p in prospects if "mixed_fluid_wells" in p["reasons"])
    no_net = sum(1 for p in prospects if "no_net_support" in p["reasons"])
    if swt:
        high = sum(1 for v in swt if v >= 0.8)
        findings.append(_finding(
            "fluid_evidence", "caution",
            f"Where logs give a water saturation for the net rock ({len(swt)} prospects), the typical value is "
            f"{_quantile(swt, 0.5):.2f}, and {high} are at or above 0.80. The run flags {water} prospects as water-bearing at the wells, "
            f"{mixed} as mixed, and {no_net} as having no net rock in any well.",
            "On the log evidence most of this rock is water-bearing. Nothing here indicates a hydrocarbon accumulation, and lower "
            "saturations in some wells are an interpreted-log indication only, never proof of oil or gas.",
            [path]))
    sensitivity, path_c = _table(reader, "prospect_cutoff_sensitivity.csv")
    pooled: dict[tuple[float, float], list[float]] = {}
    for row in sensitivity:
        key = (_num(row.get("vsh_max")), _num(row.get("phit_min")))
        if None in key:
            continue
        cell = pooled.setdefault(key, [0.0, 0.0])
        cell[0] += _num(row.get("gross_m")) or 0.0
        cell[1] += _num(row.get("net_m")) or 0.0
    grid = [{"max_shale_volume": k[0], "min_porosity": k[1], "pooled_net_to_gross": round(v[1] / v[0], 3)}
            for k, v in sorted(pooled.items()) if v[0]]
    if len(grid) >= 2:
        lo, hi = min(g["pooled_net_to_gross"] for g in grid), max(g["pooled_net_to_gross"] for g in grid)

        def swing(vary: str, hold: str) -> float:
            spans = []
            for fixed in {g[hold] for g in grid}:
                values = [g["pooled_net_to_gross"] for g in grid if g[hold] == fixed]
                spans.append(max(values) - min(values))
            return max(spans) if spans else 0.0

        by_shale, by_porosity = swing("max_shale_volume", "min_porosity"), swing("min_porosity", "max_shale_volume")
        bigger, smaller = ("shale-volume", "porosity") if by_shale >= by_porosity else ("porosity", "shale-volume")
        findings.append(_finding(
            "cutoff_sensitivity", "info",
            f"Depending on the cutoffs, pooled net-to-gross runs from {lo:.2f} to {hi:.2f}.",
            "Net-to-gross is a choice as much as a measurement. Here the " + bigger + " cutoff moves it more than the " + smaller
            + " cutoff. Agree the cutoffs with a geologist before quoting any single value.",
            [path_c]))
    screening, path_u = _table(reader, "prospect_uncertainty_screening.csv")
    modes: dict[str, int] = {}
    for row in screening:
        modes[str(row.get("status"))] = modes.get(str(row.get("status")), 0) + 1
    excludes = next((str(r.get("excludes")) for r in screening if r.get("excludes")), "")
    if screening:
        findings.append(_finding(
            "uncertainty_partial", "info",
            f"Low, middle and high cases were computed by re-drawing log samples for {modes.get('BLOCK_BOOTSTRAP', 0)} of "
            f"{len(screening)} prospects; no hydrocarbon-volume range was produced.",
            "These ranges cover sampling noise in the logs only. They leave out " + (_clip(excludes, 160) if excludes else "model, "
            "geometry and geological risk") + ", which are the larger uncertainties.",
            [path_u]))
    return {
        "prospects_with_logs": len(logged), "prospects_total": n,
        "net_to_gross": {"median": _r(median), "p10": _r(_quantile(ntg, 0.1)), "p90": _r(_quantile(ntg, 0.9)),
                         "prospects_with_zero_net": sum(1 for v in ntg if v == 0)},
        "sand_share_of_all_logged_rock": _r(baseline),
        "porosity_of_net": {"median": _r(_quantile(phit, 0.5)), "min": _r(min(phit)) if phit else None, "max": _r(max(phit)) if phit else None,
                            "prospects": len(phit), "type": "total porosity, not effective porosity"},
        "water_saturation_of_net": {"median": _r(_quantile(swt, 0.5)), "prospects": len(swt),
                                    "at_or_above_0.8": sum(1 for v in swt if v >= 0.8), "at_or_below_0.7": sum(1 for v in swt if v <= 0.7)},
        "run_flags": {"water_bearing_at_wells": water, "mixed_fluid_wells": mixed, "no_net_support": no_net},
        "cutoff_sensitivity": grid, "uncertainty_screening": {"modes": modes, "excludes": _clip(excludes, 200) or None},
        "how_measured": "Log properties are read only where well paths cross a prospect, then applied to the whole prospect. "
                        "Pooled figures count overlapping prospects more than once.",
    }


def _provenance_section(reader: _Reader, summary: dict[str, Any], preflight: dict[str, Any],
                        findings: list[dict[str, Any]]) -> dict[str, Any]:
    products = [p for p in summary.get("products", []) if isinstance(p, dict)]
    groups: dict[str, list[float]] = {}
    for item in products:
        path = str(item.get("path") or "")
        kind = ("summary tables" if "/tables/" in path else "voxel index arrays" if "prospect_voxels" in path
                else "point clouds" if "/points/" in path else "other")
        cell = groups.setdefault(kind, [0.0, 0.0])
        cell[0] += 1
        cell[1] += _num(item.get("bytes")) or 0.0
    total_gb = sum(v[1] for v in groups.values()) / 1e9
    exports = summary.get("point_exports") if isinstance(summary.get("point_exports"), dict) else {}
    skipped = sum(1 for v in exports.values() if isinstance(v, dict) and str(v.get("status")).upper().startswith("SKIPPED"))
    if total_gb >= 20:
        findings.append(_finding(
            "storage_cost", "caution",
            f"The run wrote about {total_gb:,.0f} GB across {len(products)} files, nearly all of it per-prospect voxel and point files; "
            f"{skipped} of {len(exports)} point exports were skipped for exceeding the row limit.",
            "Most of that storage describes overlapping slices of one background body. It can be removed once the summary tables are kept.",
            [SUMMARY_FILE]))
    status = str(summary.get("interpretation_status") or "")
    if status:
        findings.append(_finding(
            "exploratory", "info", f"The run labels every prospect {status}.",
            "It makes no closure, contact, hydrocarbon, net-pay, resource or reserves claim.", [SUMMARY_FILE]))
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
    return {
        "run_status": summary.get("status"), "preflight_status": preflight.get("status"),
        "phases": summary.get("phase_scope"), "counts": counts,
        "anchor_wells": summary.get("anchor_well_ids"), "anchor_selection": summary.get("anchor_selection_mode"),
        "source_jobs": {"foundation": source.get("foundation_job_id"), "inference": source.get("inference_job_id"),
                        "volume_products": source.get("source_volume_job_id"), "geobody": source.get("source_geobody_job_id")},
        "inputs_checksummed": sorted((summary.get("source_products") or {}).keys()),
        "outputs": {k: {"files": int(v[0]), "size_gb": round(v[1] / 1e9, 1)} for k, v in groups.items()},
        "outputs_total_gb": round(total_gb, 1), "point_exports_skipped": skipped,
        "limitations": [_clip(x, 220) for x in summary.get("limitations") or []],
    }


def _prospect_card(reader: _Reader, prospects: list[dict[str, Any]], wanted: str,
                   assumptions: dict[str, Any]) -> dict[str, Any]:
    key = wanted.strip().upper()
    if key.isdigit():
        key = f"P{int(key):03d}"
    item = next((p for p in prospects if str(p["prospect"]).upper() == key), None)
    if item is None:
        return {"prospect": wanted, "found": False, "example_ids": [p["prospect"] for p in prospects[:5]], "prospects": len(prospects)}
    wells = [{"well": r.get("well"), "logged_m": _r(r.get("gross_m"), 1), "net_m": _r(r.get("net_m"), 1), "net_to_gross": _r(r.get("ntg")),
              "porosity_of_net": _r(r.get("phit_net")), "water_saturation_of_net": _r(r.get("swt_net_pv")), "saturation_status": r.get("swt_status")}
             for r in _table(reader, "prospect_well_properties.csv")[0]
             if r.get("prospect") == item["prospect"] and _truthy(r.get("intersects_prospect"))]
    screen = next((r for r in _table(reader, "prospect_uncertainty_screening.csv")[0] if r.get("prospect") == item["prospect"]), {})
    cutoffs = [{"max_shale_volume": _num(r.get("vsh_max")), "min_porosity": _num(r.get("phit_min")), "net_to_gross": _r(r.get("ntg"))}
               for r in _table(reader, "prospect_cutoff_sensitivity.csv")[0] if r.get("prospect") == item["prospect"]]
    if item["survey_scale"]:
        reading = "A survey-wide time slice of the background body. It is not a prospect; do not rank it or give it a volume."
    elif not item["wells_with_net"]:
        reading = "Compact, but no well found net rock in it. There is no log support for calling it sand."
    else:
        reading = "Compact, with net rock in at least one well. A place to inspect on the seismic, still exploratory."
    # Voxel-based estimate for this prospect. The low/mid/high HCPV combines the log bootstrap percentiles:
    # low uses the low NTG and PHIT with the high SWT (least hydrocarbon), high does the reverse.
    grv_est = item.get("grv_voxel_est_m3")
    ntg_lmh = [_num(screen.get(k)) for k in ("ntg_P90", "ntg_P50", "ntg_P10")]
    phit_lmh = [_num(screen.get(k)) for k in ("phit_net_P90", "phit_net_P50", "phit_net_P10")]
    swt_lmh = [_num(screen.get(k)) for k in ("swt_net_pv_P90", "swt_net_pv_P50", "swt_net_pv_P10")]

    def _hcpv_case(i_ntg: int, i_phit: int, i_swt: int) -> int | None:
        if grv_est is None:
            return None
        ntg_, phit_, swt_ = ntg_lmh[i_ntg], phit_lmh[i_phit], swt_lmh[i_swt]
        if None in (ntg_, phit_, swt_):
            return None
        return round(grv_est * ntg_ * phit_ * (1.0 - swt_))

    estimate = {
        "grv_est_m3": grv_est,
        "hcpv_est_m3": item.get("hcpv_voxel_est_m3"),
        "hcpv_est_m3_low_mid_high": [_hcpv_case(0, 0, 2), _hcpv_case(1, 1, 1), _hcpv_case(2, 2, 0)],
        "inputs_used": {"voxels": item.get("voxels"), "net_to_gross": item.get("net_to_gross"),
                        "porosity_of_net": item.get("porosity_of_net"), "water_saturation_of_net": item.get("water_saturation_of_net"),
                        "cell_area_m2": assumptions.get("cell_area_m2"), "voxel_height_m": assumptions.get("voxel_height_m")},
        "assumptions": {k: assumptions.get(k) for k in ("xy_units_assumed_metres", "sample_interval_ms", "sample_interval_assumed",
                                                        "interval_velocity_m_s", "interval_velocity_assumed")},
        "caveat": ("This prospect is survey-scale background; the estimate spans most of the survey and is meaningless as a prospect volume."
                   if item["survey_scale"] else
                   "Unvalidated voxel-based screening scenario; not a resource estimate. Inspect on the seismic before trusting it."),
    }
    return {**item, "found": True, "reading": reading, "wells_crossing": wells,
            "voxel_based_estimate": estimate,
            "low_middle_high": {"status": screen.get("status"),
                                "net_to_gross": [_r(screen.get(k)) for k in ("ntg_P90", "ntg_P50", "ntg_P10")],
                                "porosity_of_net": [_r(screen.get(k)) for k in ("phit_net_P90", "phit_net_P50", "phit_net_P10")],
                                "water_saturation_of_net": [_r(screen.get(k)) for k in ("swt_net_pv_P90", "swt_net_pv_P50", "swt_net_pv_P10")]},
            "net_to_gross_by_cutoff": cutoffs}


# --------------------------------------------------------------------------------------------
# Verdict and next steps
# --------------------------------------------------------------------------------------------
def _verdict(status: str, summary: dict[str, Any], sections: dict[str, Any]) -> dict[str, Any]:
    prospects, volumes = sections["prospects"], sections["volumes"]
    n = prospects["prospects"]
    if str(status).lower() not in {"succeeded", "success", "completed"}:
        level, plain = "did_not_finish", f"The job status is '{status}', so this is not a completed result."
    elif not n:
        level, plain = "incomplete", "The run finished but no prospect table was found."
    else:
        level = "usable_with_caveats"
        if prospects["mostly_background"]:
            plain = (f"The run completed and defined {n} prospects, but they are not prospects in any useful sense: "
                     f"{prospects['survey_scale_prospects']} are survey-wide slices of one background body.")
        else:
            plain = f"The run completed and defined {n} exploratory prospects, {prospects['compact_prospects']} of them compact."
        if not volumes["prospects_with_rock_volume"] and not volumes["prospects_with_hcpv"]:
            plain += " No volume of any kind is reported, and that is the correct outcome."
    assessment = "background_slices" if prospects["mostly_background"] else "candidate_prospects"
    return {"level": level, "plain": plain, "assessment": assessment, "run_status": summary.get("status"), "prospects": n,
            "volumes_reported": bool(volumes["prospects_with_rock_volume"] or volumes["prospects_with_hcpv"]),
            "note": "A completed run and a meaningful prospect list are separate questions."}


def _next_steps(findings: list[dict[str, Any]], verdict: dict[str, Any]) -> list[str]:
    ids = {f["id"] for f in findings}
    steps: list[str] = []
    if verdict["level"] in {"did_not_finish", "incomplete"}:
        steps.append("Resolve the failed or missing items first.")
    if "background_slabs" in ids:
        steps.append("Re-run the geobody extraction with a higher low threshold, then repeat the well analysis and this step. "
                     "Until the bodies are separated, prospect definition cannot work.")
    if "units_unconfirmed" in ids:
        steps.append("Ask the data owner to confirm the map coordinate units; that single answer unblocks gross rock volume.")
    if "ntg_evidence" in ids:
        steps.append("Use the net-to-gross result as feedback on the sand cube: it measures how sandy the predicted bodies are at the wells.")
    if "overlap" in ids:
        steps.append("Never add prospects together; they overlap.")
    if "compact_prospects" in ids:
        steps.append("Inspect the few compact prospects on the seismic, starting with those that have net rock in a well.")
    if "storage_cost" in ids:
        steps.append("Keep the summary tables and clear the per-prospect voxel and point files to recover storage.")
    return steps[:6]


# --------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------
def analyse_job_dir(job_dir: Path, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyse one already-located job folder. Pure file inspection; no backend access."""
    parameters = parameters or {}
    reader = _Reader(job_dir)
    metadata, result = reader.json("job.json"), reader.json("result.json")
    summary = reader.json(SUMMARY_FILE)
    preflight = reader.json(PREFLIGHT_FILE)
    job_id = str(parameters.get("job_id") or metadata.get("job_id") or job_dir.name)
    script = str(metadata.get("python_script") or result.get("python_script") or "")
    status = str(result.get("status") or metadata.get("status") or "unknown")
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
        reason = (f"This folder was not produced by Process F (expected script '{EXPECTED_SCRIPT}')." if not recognised else
                  f"'{SUMMARY_FILE}' is missing or unreadable, so there is nothing to analyse.")
        if recognised and preflight and not summary:
            reason += f" The preflight reports status '{preflight.get('status')}'."
        return {**base, "applicable": False, "summary": reason, "findings": [reason], "artifacts": reader.used,
                "skipped": reader.skipped,
                "limitations": ["Use the generic processing_output_folder_analysis skill to inventory this folder."]}

    prospects, evidence = _prospects(reader)
    assumptions = _apply_voxel_estimates(summary, prospects, parameters)
    baseline = _sand_share_at_wells(reader, str(base["foundation_job_id"] or ""))
    findings: list[dict[str, Any]] = []
    sections = {
        "prospects": _prospects_section(prospects, evidence, findings),
        "volumes": _volumes_section(reader, summary, prospects, findings, assumptions),
        "logs": _logs_section(reader, prospects, baseline, findings),
        "provenance": _provenance_section(reader, summary, preflight, findings),
    }
    verdict = _verdict(status, summary, sections)

    def rank(finding: dict[str, Any]) -> int:
        return next((i for i, prefix in enumerate(PRIORITY) if finding["id"].startswith(prefix)), len(PRIORITY))

    findings.sort(key=rank)
    takeaways = [f for f in findings if f["level"] in {"critical", "caution"} and f["id"].startswith(TAKEAWAY_IDS)]
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

    p, v, logs = sections["prospects"], sections["volumes"], sections["logs"]
    response: dict[str, Any] = {
        **base, "applicable": True, "focus": focus,
        "verdict": verdict, "summary": verdict["plain"],
        "takeaways": takeaways,
        "at_a_glance": {
            "prospects": p["prospects"], "survey_scale_prospects": p["survey_scale_prospects"], "compact_prospects": p["compact_prospects"],
            "prospects_with_rock_volume": v["prospects_with_rock_volume"], "prospects_with_hcpv": v["prospects_with_hcpv"],
            "prospects_with_voxel_grv_estimate": v["voxel_based_estimate"]["prospects_with_grv_estimate"],
            "typical_net_to_gross_from_logs": logs["net_to_gross"]["median"],
            "sand_share_of_all_logged_rock": logs["sand_share_of_all_logged_rock"],
            "typical_water_saturation_of_net": logs["water_saturation_of_net"]["median"],
            "run_status": summary.get("status"),
        },
        "key_findings": shown,
        "findings": [f["observed"] for f in shown],
        "next_steps": _next_steps(findings, verdict),
        "details": details,
        "findings_not_shown": len(findings) - len(shown) if focus == "overview" else 0,
        "more_detail": "Call again with focus = prospects, volumes, logs, provenance or all, or with prospect = <ID such as P042>.",
    }
    wanted = str(parameters.get("prospect") or "").strip()
    if wanted:
        response["prospect_card"] = _prospect_card(reader, prospects, wanted, assumptions)
    if focus in {"all", "glossary"}:
        response["glossary"] = GLOSSARY
    if focus == "overview":
        response["suggested_questions"] = SUGGESTED_QUESTIONS
    response["artifacts"] = [{**item, "role": "evidence read by this analysis"} for item in reader.used]
    response["skipped"] = reader.skipped
    response["limitations"] = [
        "Numbers are read from the run's summary, preflight and tables; no voxel or point file is opened. The voxel-based GRV and HCPV "
        "estimates are the one exception: they are derived here from voxel counts, map-cell area and a voxel height, and are an "
        "unvalidated screening scenario, not the run's gated volume.",
        "Typical values are medians across prospects. Prospects overlap, so pooled figures count the same rock more than once.",
        "Log properties are interpreted logs read where wells cross a prospect; applying them to the whole prospect is an assumption.",
        "Plain-language meanings are fixed expert reading rules applied to the observed numbers, not proof of a geological cause.",
    ]
    if baseline is None:
        response["limitations"].append("The sand share of all logged rock lives in the foundation job, which could not be read from "
                                       "here, so net-to-gross is not compared with it.")
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
