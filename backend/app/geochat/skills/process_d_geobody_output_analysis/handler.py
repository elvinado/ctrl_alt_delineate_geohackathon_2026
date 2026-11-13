"""Expert, deterministic reading of one Process D (GeoGG geobody extraction, stage 12) run folder.

Process D turns the sand-score cube into candidate geobodies: connected groups of voxels found
by a two-threshold (hysteresis) rule. This handler reads the run's own summary, preflight and
catalogue and applies fixed, documented reading rules. It never calls an LLM, never invents a
value and never opens a volume, mesh or figure. Every file it used is listed under ``artifacts``.

This module is self-contained on purpose: it imports no other skill.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

SKILL_NAME = "process_d_geobody_output_analysis"
EXPECTED_SCRIPT = "process_D_geobody.py"
SUMMARY_FILE = "geogg_geobody_summary.json"
PREFLIGHT_FILE = "geogg_geobody_preflight.json"
S12 = "stages/12_geobody"

MAX_JSON_BYTES = 8_000_000
MAX_CSV_BYTES = 30_000_000
MAX_CSV_ROWS = 60_000
MAX_TEXT = 420
OVERVIEW_FINDINGS = 8
LISTED_BODIES = 12

# Site-percolation threshold of a simple cubic lattice with face (6-) connectivity. Above this
# occupied fraction a single cluster spanning the whole volume forms even in random noise.
PERCOLATION_6 = 0.3116
DOMINANT_SHARE = 0.90

FOCUS_ALIASES: dict[str, str] = {
    "": "overview", "overview": "overview", "summary": "overview", "all": "all", "full": "all",
    "bodies": "bodies", "geobodies": "bodies", "catalogue": "bodies", "catalog": "bodies", "list": "bodies",
    "method": "method", "thresholds": "method", "settings": "method", "segmentation": "method",
    "products": "products", "files": "products", "exports": "products", "segy": "products", "meshes": "products", "api": "products",
    "preflight": "preflight", "checks": "preflight", "provenance": "preflight",
    "glossary": "glossary", "figures": "figures", "plots": "figures", "12": "overview",
}

GLOSSARY: dict[str, str] = {
    "geobody": "A connected 3-D group of voxels that share a property, here a high sand score. A shape, not a proven reservoir.",
    "voxel": "One cell of a 3-D cube: one inline, one crossline, one time sample.",
    "sand score": "The model's 0-to-1 output for sand versus shale. Higher means more sand-like. It is not a calibrated probability.",
    "hysteresis thresholding": "A two-level rule: a body must contain at least one voxel above the high level (a seed), and then grows through neighbours above the low level.",
    "seed": "A voxel above the high threshold. Without one, a group of voxels is discarded.",
    "connectivity 6": "Voxels count as joined only when they share a face: up, down, and the four sides.",
    "percolation": "When enough cells are switched on, they inevitably link into one cluster that spans the whole volume, even if they are random.",
    "candidate fraction": "The share of predicted voxels that ended up inside any body.",
    "column": "One trace position (one inline and crossline). 'Distinct columns' is a body's map-view footprint in cells.",
    "TWT": "Two-way time in milliseconds. Bodies here are measured in time, not depth.",
    "run-length index": "A compact table listing, for each column, the time ranges that belong to each body.",
    "mesh (PLY)": "A 3-D surface file of one body, for viewing in 3-D software.",
    "exploratory candidate": "A result to look at and question, not to rely on.",
}

SUGGESTED_QUESTIONS: list[str] = [
    "Did the geobody extraction find anything meaningful?",
    "Why is one body so much larger than all the others?",
    "Which bodies are worth looking at first?",
    "What thresholds were used, and should we change them?",
    "Tell me about one body by its ID.",
    "What files can I load, and how?",
]

# Order in which findings are surfaced (first match on the id prefix wins).
PRIORITY: list[str] = [
    "export_failed", "reconciliation_failed", "giant_body", "percolation", "low_threshold", "remaining_bodies",
    "stacked_pillars", "weak_seeds", "exploratory", "time_only", "exports_ready", "files_missing", "thresholds_unvalidated",
    "rejected_components", "boundary",
]
TAKEAWAY_IDS: tuple[str, ...] = ("export_failed", "reconciliation_failed", "giant_body", "percolation", "low_threshold",
                                 "stacked_pillars", "exploratory", "time_only")
_FINDING_FOCUS: dict[str, tuple[str, ...]] = {
    "bodies": ("giant_body", "remaining_bodies", "stacked_pillars", "weak_seeds", "boundary"),
    "method": ("percolation", "low_threshold", "thresholds_unvalidated", "rejected_components", "giant_body"),
    "products": ("exports_ready", "export_failed", "files_missing", "reconciliation_failed", "time_only"),
    "preflight": ("exploratory", "time_only", "thresholds_unvalidated"),
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


def _ids(items: list[Any], limit: int = 8) -> str:
    shown = ", ".join(str(i) for i in items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


def _score_stats(reader: _Reader, volume_job_id: str, product: str) -> dict[str, float]:
    """Percentiles of the source score cube, recorded only in the volume-products job next to this one."""
    if not volume_job_id or Path(volume_job_id).name != volume_job_id or volume_job_id in {".", ".."}:
        return {}
    jobs_root = reader.job_dir.parent
    relative = "stages/10_summary/tables/volume_statistics.csv"
    path = jobs_root / volume_job_id / relative
    try:
        if path.is_symlink() or (jobs_root / volume_job_id).is_symlink() or not path.is_file():
            return {}
        if jobs_root not in path.resolve().parents or path.stat().st_size > 1_000_000:
            return {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, UnicodeDecodeError, csv.Error):
        return {}
    wanted = product.replace("prediction_", "")
    row = next((r for r in rows if r.get("attribute") == wanted), None)
    if not row:
        return {}
    reader.used.append({"name": f"../{volume_job_id}/{relative}", "size_bytes": path.stat().st_size})
    return {k: v for k in ("p01", "p05", "p25", "p50", "p75", "p95", "p99") if (v := _num(row.get(k))) is not None}


def _bodies(reader: _Reader, geometry: dict[str, Any]) -> list[dict[str, Any]]:
    il_step = _num(geometry.get("il_step")) or 1
    xl_step = _num(geometry.get("xl_step")) or 1
    dt = _num(geometry.get("dt_ms")) or 1
    rows = reader.table(f"{S12}/tables/geobody_catalog.csv")
    total = sum(_int(r.get("voxel_count")) or 0 for r in rows)
    bodies = []
    for r in rows:
        voxels = _int(r.get("voxel_count")) or 0
        il0, il1, xl0, xl1 = (_int(r.get(k)) for k in ("il_min", "il_max", "xl_min", "xl_max"))
        t0, t1 = _num(r.get("twt_min_ms")), _num(r.get("twt_max_ms"))
        il_cells = int((il1 - il0) / il_step) + 1 if None not in (il0, il1) else None
        xl_cells = int((xl1 - xl0) / xl_step) + 1 if None not in (xl0, xl1) else None
        samples = int((t1 - t0) / dt) + 1 if None not in (t0, t1) else None
        share = voxels / total if total else 0.0
        lateral = max(il_cells or 0, xl_cells or 0)
        if share >= DOMINANT_SHARE:
            shape = "survey-scale"
        elif samples and lateral and samples >= 5 * lateral and samples >= 50:
            shape = "pillar"
        elif samples and lateral >= 3 * samples:
            shape = "sheet"
        else:
            shape = "compact"
        bodies.append({
            "geobody_id": _int(r.get("geobody_id")), "voxels": voxels, "share_of_body_voxels": round(share, 6),
            "map_cells": _int(r.get("distinct_columns")), "mean_score": _r(r.get("mean_score")),
            "seed_voxels": _int(r.get("seed_voxel_count")),
            "inline_range": [il0, il1], "crossline_range": [xl0, xl1], "twt_range_ms": [t0, t1],
            "extent_cells_inline_crossline": [il_cells, xl_cells], "time_thickness_ms": (t1 - t0 + dt) if None not in (t0, t1) else None,
            "centre": {"inline": _r(r.get("centroid_inline"), 0), "crossline": _r(r.get("centroid_crossline"), 0),
                       "twt_ms": _r(r.get("centroid_twt_ms"), 0)},
            "shape": shape,
            "touches_edge": any(_truthy(r.get(k)) for k in r if k.startswith("touches_")),
            "status": r.get("interpretation_status"),
        })
    return sorted(bodies, key=lambda b: -b["voxels"])


# --------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------
def _method_section(summary: dict[str, Any], bodies: list[dict[str, Any]], stats: dict[str, float],
                    findings: list[dict[str, Any]]) -> dict[str, Any]:
    method = summary.get("method") if isinstance(summary.get("method"), dict) else {}
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
    low, high = _num(method.get("tau_low")), _num(method.get("tau_high"))
    fraction = _num(counts.get("candidate_fraction_of_valid"))
    connectivity = _int(method.get("connectivity"))
    top = bodies[0] if bodies else None
    dominant = bool(top and top["share_of_body_voxels"] >= DOMINANT_SHARE)
    if dominant:
        total = sum(b["voxels"] for b in bodies)
        findings.append(_finding(
            "giant_body", "critical",
            f"Body {top['geobody_id']} holds {top['share_of_body_voxels'] * 100:.2f}% of all body voxels "
            f"({top['voxels']:,} of {total:,}). It spans inlines {top['inline_range'][0]} to {top['inline_range'][1]}, "
            f"crosslines {top['crossline_range'][0]} to {top['crossline_range'][1]} and {top['twt_range_ms'][0]:.0f} to "
            f"{top['twt_range_ms'][1]:.0f} ms.",
            "This is not a geobody. It is the background of the cube joined into one piece, so the extraction has not separated "
            "sand bodies from their surroundings. Its size, shape and average score carry no geological meaning.",
            [f"{S12}/tables/geobody_catalog.csv"]))
    if fraction is not None and connectivity == 6 and fraction > PERCOLATION_6:
        findings.append(_finding(
            "percolation", "critical" if dominant else "caution",
            f"{_pct(fraction)} of predicted voxels lie inside a body. With face-to-face connectivity, switched-on cells link "
            f"into one volume-spanning cluster once they exceed about {_pct(PERCOLATION_6)}, even in random noise.",
            "A single giant body was mathematically expected at these settings. It reflects the thresholds, not the geology.",
            ["geogg_geobody_summary.json"]))
    target = None
    if low is not None and stats.get("p50") is not None:
        for name, share_above in (("p75", 0.25), ("p95", 0.05)):
            if stats.get(name) is not None and stats[name] > low:
                target = {"percentile": name, "score": round(stats[name], 2), "share_of_cube_above": share_above}
                break
        if low < stats["p50"]:
            findings.append(_finding(
                "low_threshold", "critical" if dominant else "caution",
                f"The low threshold ({low:g}) is below the cube's median sand score ({stats['p50']:.2f}), so more than half of the "
                "cube passes it.",
                "Bodies are allowed to grow through ordinary background values, which is why everything joins up."
                + (f" A low threshold near {target['score']:.2f} would let only about {_pct(target['share_of_cube_above'])} of the cube through."
                   if target else ""),
                ["geogg_geobody_summary.json", "volume_statistics.csv of the source volume-products job"]))
    raw, kept, rejected = _int(counts.get("raw_low_threshold_components")), _int(counts.get("retained_geobodies")), _int(counts.get("rejected_components"))
    if raw:
        findings.append(_finding(
            "rejected_components", "info",
            f"{rejected:,} of {raw:,} raw voxel groups were discarded and {kept} were kept.",
            "The discarded groups contained no voxel above the high threshold, or were too small. This filter worked as designed.",
            ["geogg_geobody_summary.json"]))
    if str(method.get("threshold_status") or "").upper().startswith("DEVELOPMENT"):
        findings.append(_finding(
            "thresholds_unvalidated", "info",
            f"Threshold status: {method.get('threshold_status')}.",
            "The thresholds rest on evidence from the development (training) folds. They have not been checked on independent data.",
            ["geogg_geobody_summary.json"]))
    return {
        "method": method.get("name"), "source_cube": source.get("product"), "connectivity": connectivity,
        "low_threshold": low, "high_threshold": high,
        "minimum_size": {"voxels": method.get("min_voxels"), "map_cells": method.get("min_distinct_columns"),
                         "lateral_extent_cells": method.get("min_lateral_extent_cells")},
        "threshold_status": method.get("threshold_status"),
        "predicted_voxels": _int(counts.get("valid_voxels")), "voxels_in_bodies": _int(counts.get("candidate_voxels")),
        "share_of_predicted_voxels_in_bodies": _r(fraction), "percolation_threshold_for_connectivity_6": PERCOLATION_6,
        "raw_groups": raw, "bodies_kept": kept, "groups_discarded": rejected,
        "source_cube_score_percentiles": {k: round(v, 3) for k, v in stats.items()} or None,
        "suggested_low_threshold": target, "one_dominant_body": dominant,
        "how_it_works": "A body must contain at least one voxel above the high threshold. It then grows through touching "
                        "voxels above the low threshold. Groups below the minimum size are discarded.",
    }


def _bodies_section(bodies: list[dict[str, Any]], method: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    dominant = bodies[0] if bodies and method["one_dominant_body"] else None
    others = [b for b in bodies if b is not dominant]
    total = sum(b["voxels"] for b in bodies)
    high = method.get("high_threshold")
    if others:
        voxels = sum(b["voxels"] for b in others)
        scores = [b["mean_score"] for b in others if b["mean_score"] is not None]
        below = [b for b in others if high is not None and b["mean_score"] is not None and b["mean_score"] < high]
        text = (f"The other {len(others)} bodies total {voxels:,} voxels ({voxels / total * 100:.3f}% of all body voxels), "
                f"from {min(b['voxels'] for b in others):,} to {max(b['voxels'] for b in others):,} voxels each.") if dominant else (
                f"{len(others)} bodies were kept, from {min(b['voxels'] for b in others):,} to {max(b['voxels'] for b in others):,} voxels each.")
        if scores:
            text += f" Their average sand scores run from {min(scores):.2f} to {max(scores):.2f}."
        findings.append(_finding(
            "remaining_bodies", "caution" if dominant else "info", text,
            ("These are the only features the extraction actually separated. " if dominant else "")
            + (f"All {len(below)} have an average score below the high threshold ({high:g}), so each is mostly low-score fill around "
               "a few strong samples. " if below and len(below) == len(others) else "")
            + "Treat them as leads to inspect on the seismic, not as mapped sands.",
            [f"{S12}/tables/geobody_catalog.csv"]))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for body in others:
        groups.setdefault((*body["inline_range"], *body["crossline_range"]), []).append(body)
    stacked = [g for g in groups.values() if len(g) >= 2]
    pillars = [b for b in others if b["shape"] == "pillar"]
    if stacked or pillars:
        group = max(stacked, key=len) if stacked else pillars
        ids = sorted(b["geobody_id"] for b in group)
        top = min(b["twt_range_ms"][0] for b in group)
        base = max(b["twt_range_ms"][1] for b in group)
        cells = max(b["map_cells"] or 0 for b in group)
        findings.append(_finding(
            "stacked_pillars", "caution",
            f"Bodies {_ids(ids)} sit on the same {cells} trace positions and together run from {top:.0f} to {base:.0f} ms.",
            "A column that narrow and that tall is not a depositional shape. It usually marks a small isolated patch of traces or a "
            "data artefact. Check the seismic there before reading it as sand.",
            [f"{S12}/tables/geobody_catalog.csv"]))
    weak = [b for b in others if b["seed_voxels"] is not None and b["seed_voxels"] <= 3]
    if weak:
        findings.append(_finding(
            "weak_seeds", "caution",
            f"{len(weak)} of the {len(others)} smaller bodies rest on three or fewer voxels above the high threshold "
            f"(bodies {_ids(sorted(b['geobody_id'] for b in weak))}).",
            "A handful of extreme samples triggered these bodies. A small change in the threshold, or noise, would remove them.",
            [f"{S12}/tables/geobody_catalog.csv"]))
    edge = [b["geobody_id"] for b in bodies if b["touches_edge"]]
    if edge:
        findings.append(_finding(
            "boundary", "caution", f"Bodies {_ids(sorted(edge))} touch the edge of the cube.",
            "They are cut off by the data boundary, so their true size is unknown.", [f"{S12}/tables/geobody_catalog.csv"]))
    shapes: dict[str, int] = {}
    for body in bodies:
        shapes[body["shape"]] = shapes.get(body["shape"], 0) + 1
    keep = ("geobody_id", "voxels", "map_cells", "mean_score", "seed_voxels", "twt_range_ms", "extent_cells_inline_crossline", "centre", "shape")
    return {
        "bodies_kept": len(bodies), "shape_counts": shapes,
        "dominant_body": {k: dominant[k] for k in (*keep, "share_of_body_voxels", "inline_range", "crossline_range")} if dominant else None,
        "largest_other_bodies": [{k: b[k] for k in keep} for b in others[:LISTED_BODIES]],
        "other_bodies_not_listed": max(0, len(others) - LISTED_BODIES),
        "shape_rule": "survey-scale: holds 90% or more of all body voxels. pillar: at least 50 samples tall and five times taller "
                      "than wide, in cells. sheet: at least three times wider than tall. compact: everything else.",
        "units": "Sizes are in voxels, grid cells and milliseconds. No depth, thickness in metres, area or volume is available.",
    }


def _products_section(reader: _Reader, summary: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    exports = summary.get("exports") if isinstance(summary.get("exports"), dict) else {}
    spatial = exports.get("spatial") if isinstance(exports.get("spatial"), dict) else {}
    segy, failed = [], []
    for item in exports.get("segy") or []:
        graded = [q for q in (item.get("reopen_qc") or []) if str(q.get("result")).upper() in {"PASS", "FAIL", "WARN"}]
        bad = [q.get("check") for q in graded if str(q.get("result")).upper() != "PASS"]
        if bad or str(item.get("status")).upper() != "PASS":
            failed.append(str(item.get("product")))
        segy.append({"product": item.get("product"), "file": item.get("path"), "present": reader.exists(str(item.get("path"))),
                     "size_gb": _r((_num(item.get("bytes")) or 0) / 1e9, 2), "status": item.get("status"),
                     "checks_passed": len(graded) - len(bad), "checks_total": len(graded), "checks_failed": bad})
    meshes = [m for m in exports.get("meshes") or [] if isinstance(m, dict)]
    mesh_ok = [m for m in meshes if str(m.get("status")).upper() == "PASS"]
    mesh_skipped = [{"geobody_id": m.get("geobody_id"), "reason": m.get("reason")} for m in meshes if str(m.get("status")).upper() != "PASS"]
    volumes = [{"product": p.get("product"), "file": p.get("path"), "dtype": p.get("dtype"), "present": reader.exists(str(p.get("path")))}
               for p in summary.get("products") or [] if isinstance(p, dict) and p.get("format") == "npy"]
    reconciliation = spatial.get("catalogue_reconciliation") if isinstance(spatial.get("catalogue_reconciliation"), dict) else {}
    index = spatial.get("run_length_index") if isinstance(spatial.get("run_length_index"), dict) else {}
    api = summary.get("api") if isinstance(summary.get("api"), dict) else {}
    section = {
        "segy_files": segy,
        "segy_meaning": {"geobody_mask": "1 inside any body, 0 outside", "geobody_id": "the body number at each voxel, 0 outside"},
        "label_volumes": volumes,
        "meshes": {"written": len(mesh_ok), "format": "PLY", "folder": f"{S12}/meshes", "skipped": mesh_skipped,
                   "coordinates": next((m.get("coordinate_system") for m in mesh_ok), None)},
        "run_length_index": {"file": index.get("path"), "rows": index.get("run_rows"), "voxels": index.get("total_exported_voxels"),
                             "present": reader.exists(str(index.get("path")))},
        "map_coordinates": {"status": spatial.get("xy_coordinate_status"), "source": _clip(spatial.get("xy_coordinate_source"), 160)},
        "depth": _clip(spatial.get("depth_conversion"), 160),
        "counts_reconcile": reconciliation.get("all_counts_reconcile"), "bodies_compared": reconciliation.get("geobodies_compared"),
        "tables": [f"{S12}/tables/geobody_catalog.csv", f"{S12}/tables/geobody_component_audit.csv"],
        "api": {"base_path": api.get("base_path"), "endpoints": api.get("endpoints")} if api else None,
    }
    if failed:
        findings.append(_finding("export_failed", "critical", f"Export checks failed for: {', '.join(failed)}.",
                                 "Do not distribute the affected file.", ["geogg_geobody_summary.json"]))
    if reconciliation and reconciliation.get("all_counts_reconcile") is not True:
        findings.append(_finding(
            "reconciliation_failed", "critical", "Voxel counts in the label volume, the catalogue and the run-length export do not agree.",
            "The exports are inconsistent with each other. Do not use them until this is resolved.",
            [f"{S12}/tables/geobody_spatial_export_reconciliation.csv"]))
    if segy and not failed:
        total = sum(s["checks_total"] for s in segy)
        text = f"{len(segy)} SEG-Y volumes ({segy[0]['size_gb']} GB each) passed all {total} export checks; {len(mesh_ok)} meshes were written"
        if mesh_skipped:
            text += f" and {len(mesh_skipped)} skipped (body {_ids([m['geobody_id'] for m in mesh_skipped])}: too large)"
        if reconciliation.get("all_counts_reconcile") is True:
            text += f". Voxel counts agree across label volume, catalogue and index for all {reconciliation.get('geobodies_compared')} bodies"
        findings.append(_finding(
            "exports_ready", "good", text + ".",
            "The files are internally consistent and safe to load. That says nothing about whether the bodies are meaningful.",
            ["geogg_geobody_summary.json"]))
    missing = [s["file"] for s in segy if not s["present"]] + [v["file"] for v in volumes if not v["present"]]
    if missing:
        findings.append(_finding(
            "files_missing", "caution", f"{len(missing)} large output files named in the summary are not in this job folder.",
            "The summary describes them, but they could not be found here. They may have been moved or not copied.",
            ["geogg_geobody_summary.json"]))
    return section


def _preflight_section(preflight: dict[str, Any], summary: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [c for c in preflight.get("checks", []) if isinstance(c, dict)]
    counts: dict[str, int] = {}
    for check in checks:
        counts[str(check.get("status")).upper()] = counts.get(str(check.get("status")).upper(), 0) + 1
    conditional = [{"check": c.get("check"), "detail": _clip(c.get("detail"), 200)} for c in checks if str(c.get("status")).upper() != "PASS"]
    status = str(summary.get("interpretation_status") or "")
    if status or conditional:
        findings.append(_finding(
            "exploratory", "caution",
            f"Every body is labelled {status or 'exploratory'}; the preflight ended {preflight.get('status') or 'unknown'}"
            + (f" because of: {', '.join(str(c['check']) for c in conditional)}." if conditional else "."),
            "The bodies are cut from a sand score that is not a calibrated probability. They are shapes to look at and question, "
            "not interpreted reservoirs.",
            [PREFLIGHT_FILE, "geogg_geobody_summary.json"]))
    if any(c["check"] == "depth_conversion" for c in conditional):
        findings.append(_finding(
            "time_only", "caution", "No qualified time-to-depth conversion is available; bodies are reported in two-way time only.",
            "Thickness in metres, area and rock volume cannot be stated. Voxel counts must not be converted to volumes.",
            [PREFLIGHT_FILE]))
    return {"status": preflight.get("status"), "check_counts": counts, "checks_not_passed": conditional,
            "source_jobs": {"foundation": preflight.get("foundation_job_id"), "inference": preflight.get("inference_job_id"),
                            "volume_products": preflight.get("source_volume_job_id")},
            "notes": [_clip(n, 220) for n in preflight.get("notes") or []],
            "limitations": [_clip(n, 220) for n in summary.get("limitations") or []]}


def _body_card(reader: _Reader, bodies: list[dict[str, Any]], summary: dict[str, Any], wanted: Any) -> dict[str, Any]:
    number = _int(wanted)
    body = next((b for b in bodies if b["geobody_id"] == number), None)
    if body is None:
        return {"geobody_id": wanted, "found": False, "available_ids": sorted(b["geobody_id"] for b in bodies)}
    exports = summary.get("exports") if isinstance(summary.get("exports"), dict) else {}
    mesh = next((m for m in exports.get("meshes") or [] if _int(m.get("geobody_id")) == number), None)
    figure = next((v.get("path") for v in exports.get("visualisations") or [] if _int(v.get("geobody_id")) == number), None)
    readings = {
        "survey-scale": "This body is the joined-up background of the cube, not a geological object.",
        "pillar": "Tall and very narrow: not a depositional shape. Check the seismic for an isolated patch of traces or an artefact.",
        "sheet": "Much wider than it is thick, which is at least the right proportion for a sedimentary layer.",
        "compact": "Small and blob-like. Inspect it on the seismic before giving it any meaning.",
    }
    if body["seed_voxels"] is not None and body["seed_voxels"] <= 3 and body["shape"] != "survey-scale":
        readings[body["shape"]] += " It rests on three or fewer strong samples, so it is fragile."
    return {**body, "found": True, "reading": readings[body["shape"]],
            "mesh": ({"status": mesh.get("status"), "file": mesh.get("path"), "reason": mesh.get("reason"),
                      "present": reader.exists(str(mesh.get("path"))) if mesh.get("path") else False} if mesh else None),
            "figure": {"file": figure, "present": reader.exists(str(figure))} if figure else None}


# --------------------------------------------------------------------------------------------
# Verdict and next steps
# --------------------------------------------------------------------------------------------
def _verdict(status: str, sections: dict[str, Any], n_bodies: int, has_preflight: bool) -> dict[str, Any]:
    method, products = sections["method"], sections["products"]
    failed = [s["product"] for s in products["segy_files"] if s["checks_failed"] or str(s["status"]).upper() != "PASS"]
    dominant = sections["bodies"]["dominant_body"]
    if str(status).lower() not in {"succeeded", "success", "completed"}:
        level, plain = "did_not_finish", f"The job status is '{status}', so this is not a completed result."
    elif not n_bodies and (method["bodies_kept"] or 0) > 0:
        level = "incomplete"
        plain = f"The summary reports {method['bodies_kept']} bodies, but the geobody catalogue is missing or unreadable."
    elif failed or products["counts_reconcile"] is False:
        level, plain = "needs_attention", "An export check or the voxel-count reconciliation failed. Resolve this before using any file."
    else:
        level = "usable_with_caveats"
        if dominant:
            plain = (f"The extraction ran and its files are verified, but it did not separate geobodies. One body holds "
                     f"{dominant['share_of_body_voxels'] * 100:.2f}% of everything, leaving {n_bodies - 1} small candidates to inspect. "
                     "The thresholds need to be raised before the result is useful.")
        else:
            plain = (f"The extraction ran and its files are verified. {n_bodies} exploratory candidate bodies were kept."
                     if n_bodies else "The extraction ran, but no body passed the thresholds and size limits.")
    return {"level": level, "plain": plain, "segmentation": "one_dominant_body" if dominant else "separated_bodies",
            "bodies_kept": n_bodies, "preflight_status": sections["preflight"]["status"] if has_preflight else None,
            "note": "Verified files and meaningful bodies are separate questions."}


def _next_steps(findings: list[dict[str, Any]], verdict: dict[str, Any], method: dict[str, Any]) -> list[str]:
    ids = {f["id"] for f in findings}
    steps: list[str] = []
    if verdict["level"] in {"did_not_finish", "incomplete", "needs_attention"}:
        steps.append("Resolve the failed or missing items first.")
    if "giant_body" in ids or "percolation" in ids:
        target = method.get("suggested_low_threshold")
        steps.append("Re-run with a higher low threshold so that well under 31% of the cube is switched on"
                     + (f" (for example about {target['score']:.2f}, the cube's {target['percentile']})." if target else "."))
        steps.append("Ignore the dominant body, its mesh and its statistics; they describe the cube's background.")
    if "remaining_bodies" in ids:
        steps.append("Inspect the largest smaller bodies on the seismic sections, starting with the sheet-shaped ones.")
    if "stacked_pillars" in ids:
        steps.append("Check the pillar location on the seismic for an isolated patch of traces before reading it as sand.")
    if "exploratory" in ids:
        steps.append("Calibrate the sand score against wells before treating any threshold as a probability cut-off.")
    if "time_only" in ids:
        steps.append("Do not quote thickness, area or volume until a qualified depth conversion exists.")
    return steps[:6]


# --------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------
def analyse_job_dir(job_dir: Path, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Analyse one already-located job folder. Pure file inspection; no backend access."""
    parameters = parameters or {}
    reader = _Reader(job_dir)
    metadata, result = reader.json("job.json"), reader.json("result.json")
    summary = reader.json(SUMMARY_FILE) or reader.json(f"{S12}/geobody_manifest.json")
    preflight = reader.json(PREFLIGHT_FILE)
    job_id = str(parameters.get("job_id") or metadata.get("job_id") or job_dir.name)
    script = str(metadata.get("python_script") or result.get("python_script") or "")
    status = str(result.get("status") or metadata.get("status") or summary.get("status") or "unknown")
    source = summary.get("source") if isinstance(summary.get("source"), dict) else {}
    base = {
        "skill": SKILL_NAME, "job_id": job_id, "python_script": script or None,
        "operation": summary.get("operation") or metadata.get("operation"),
        "target_well": (str(metadata.get("target_well") or "").strip() or None), "status": status,
        "created_at": metadata.get("created_at") or result.get("created_at"),
        "source_volume_job_id": source.get("volume_job_id") or preflight.get("source_volume_job_id"),
        "source_inference_job_id": source.get("inference_job_id") or preflight.get("inference_job_id"),
        "foundation_job_id": source.get("foundation_job_id") or preflight.get("foundation_job_id"),
    }
    recognised = script == EXPECTED_SCRIPT or str(summary.get("stage")) == "12_geobody" or str(preflight.get("check")) == "12_geobody"
    if not recognised or not summary:
        reason = (f"This folder was not produced by Process D (expected script '{EXPECTED_SCRIPT}')." if not recognised else
                  f"'{SUMMARY_FILE}' is missing or unreadable, so there is nothing to analyse.")
        if recognised and preflight and not summary:
            blocked = [str(c.get("check")) for c in preflight.get("checks", []) if str(c.get("status")).upper() in {"FAIL", "FAILED", "BLOCKED"}]
            reason += (f" The preflight ended '{preflight.get('status')}'" + (f" with failed checks: {', '.join(blocked)}." if blocked else "."))
        return {**base, "applicable": False, "summary": reason, "findings": [reason], "artifacts": reader.used,
                "skipped": reader.skipped,
                "limitations": ["Use the generic processing_output_folder_analysis skill to inventory this folder."]}

    geometry = summary.get("geometry") if isinstance(summary.get("geometry"), dict) else {}
    bodies = _bodies(reader, geometry)
    stats = _score_stats(reader, str(base["source_volume_job_id"] or ""), str(source.get("product") or ""))

    findings: list[dict[str, Any]] = []
    method = _method_section(summary, bodies, stats, findings)
    sections = {
        "method": method,
        "bodies": _bodies_section(bodies, method, findings),
        "products": _products_section(reader, summary, findings),
        "preflight": _preflight_section(preflight, summary, findings),
    }
    verdict = _verdict(status, sections, len(bodies), bool(preflight))

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
    elif focus in {"glossary", "figures"}:
        details, shown = {}, []
    else:
        details, shown = {}, findings[:OVERVIEW_FINDINGS]

    response: dict[str, Any] = {
        **base, "applicable": True, "focus": focus,
        "verdict": verdict, "summary": verdict["plain"],
        "takeaways": takeaways,
        "at_a_glance": {
            "bodies_kept": len(bodies), "one_dominant_body": method["one_dominant_body"],
            "share_of_predicted_voxels_in_bodies": method["share_of_predicted_voxels_in_bodies"],
            "low_threshold": method["low_threshold"], "high_threshold": method["high_threshold"],
            "interpretation_status": summary.get("interpretation_status"), "units": "voxels, grid cells and milliseconds only",
        },
        "key_findings": shown,
        "findings": [f["observed"] for f in shown],
        "next_steps": _next_steps(findings, verdict, method),
        "details": details,
        "findings_not_shown": len(findings) - len(shown) if focus == "overview" else 0,
        "more_detail": "Call again with focus = bodies, method, products, preflight, figures or all, or with geobody_id = <number>.",
    }
    wanted = parameters.get("geobody_id")
    if wanted not in (None, ""):
        response["body_card"] = _body_card(reader, bodies, summary, wanted)
    if focus in {"all", "glossary"}:
        response["glossary"] = GLOSSARY
    if focus in {"all", "figures"}:
        exports = summary.get("exports") if isinstance(summary.get("exports"), dict) else {}
        shows = {"overview": "All kept bodies in a 3-D view.", "catalogue": "Body sizes and scores side by side.",
                 "body_slices": "Inline, crossline and time slices through one body."}
        response["figures_guide"] = [
            {"file": v.get("path"), "geobody_id": v.get("geobody_id"), "shows": shows.get(str(v.get("kind")), "Figure.")}
            for v in exports.get("visualisations") or [] if isinstance(v, dict) and reader.exists(str(v.get("path")))]
    if focus == "overview":
        response["suggested_questions"] = SUGGESTED_QUESTIONS
    response["artifacts"] = [{**item, "role": "evidence read by this analysis"} for item in reader.used]
    response["skipped"] = reader.skipped
    response["limitations"] = [
        "Numbers are read from the run's summary, preflight and catalogue; nothing is recomputed, and no volume, mesh or figure is opened.",
        "The per-group audit table is too large to read here; its totals are taken from the summary.",
        "Shape labels and the percolation threshold are reading conventions. The percolation value is exact only for random, "
        "uncorrelated cells; real seismic is smoother, which makes joining up easier, not harder.",
        "Plain-language meanings are fixed expert reading rules applied to the observed numbers, not proof of a geological cause.",
    ]
    if not stats:
        response["limitations"].append("The score percentiles of the source cube live in the volume-products job, which could not be "
                                       "read from here, so the threshold is not compared with the cube's median.")
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
