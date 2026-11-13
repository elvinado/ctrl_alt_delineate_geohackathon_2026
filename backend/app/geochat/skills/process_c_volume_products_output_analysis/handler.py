"""Expert, deterministic reading of one Process C (GeoGG volume products, stages 10-11) run folder.

Process C audits the pipeline, packages the prediction cubes as SEG-Y and sets up blind-well
validation. This handler reads the run's own summaries and tables and applies fixed, documented
reading rules. It never calls an LLM, never invents a value and never opens a cube file. Every
file it used is listed under ``artifacts``. The GeoChat agent explains the returned evidence.

This module is self-contained on purpose: it imports no other skill.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any

SKILL_NAME = "process_c_volume_products_output_analysis"
EXPECTED_SCRIPT = "process_C_volume_products.py"
SUMMARY_FILE = "geogg_volume_products_summary.json"
S10 = "stages/10_summary"
S11 = "stages/11_volume_export"
SRC = f"{S10}/tables/evaluation_source"

MAX_JSON_BYTES = 8_000_000
MAX_CSV_BYTES = 30_000_000
MAX_CSV_ROWS = 60_000
MAX_TEXT = 420
OVERVIEW_FINDINGS = 8

FOCUS_ALIASES: dict[str, str] = {
    "": "overview", "overview": "overview", "summary": "overview", "all": "all", "full": "all",
    "products": "products", "files": "products", "segy": "products", "export": "products", "deliverables": "products",
    "loading": "products", "11": "products",
    "quality": "quality", "trust": "quality", "performance": "quality", "scores": "quality", "cubes": "quality",
    "statistics": "statistics", "stats": "statistics", "values": "statistics", "coverage": "statistics",
    "audit": "audit", "provenance": "audit", "integrity": "audit", "reproducibility": "audit", "10": "audit",
    "validation": "validation", "blind": "validation", "blind_well": "validation", "blind_wells": "validation",
    "wells": "wells", "well": "wells",
    "glossary": "glossary", "figures": "figures", "sections": "figures", "dashboard": "figures", "plots": "figures",
}

TASK_NAMES: dict[str, str] = {
    "lithology": "sand versus shale",
    "vsh": "shale volume (VSH)",
    "phit": "total porosity (PHIT)",
    "sand_fluid": "water-bearing versus hydrocarbon-potential sand",
    "swt": "total water saturation (SWT)",
}

CUBE_ADVICE: dict[str, str] = {
    "none": "Do not interpret this cube. The model that made it showed no skill on unseen wells.",
    "weak": "Qualitative screening only, alongside other evidence. Never the sole basis for a decision.",
    "modest": "A relative indicator of where the property is more or less likely. Not for quantitative work.",
    "moderate": "A relative indicator, to be quoted with its uncertainty band. Not for volumetrics without calibration.",
    "good": "Usable for interpretation. Confirm against a blind well before quantitative use.",
    "strong": "Usable, but first rule out leakage and confirm against a blind well.",
    "unrated": "Reliability unknown: no held-out score could be matched to this cube's model.",
}

GLOSSARY: dict[str, str] = {
    "SEG-Y": "The standard file format for seismic volumes. Interpretation software loads it directly.",
    "audit": "A check that every input and output is the file it claims to be, using checksums.",
    "checksum (SHA-256)": "A fingerprint of a file. If one byte changes, the fingerprint changes.",
    "bit-exact": "Identical down to the last binary digit, with no rounding.",
    "no-data value": "A placeholder number stored where there is no prediction. It must be set as null when loading.",
    "held-out score": "A score on wells kept away from training in cross-validation. Honest, but still the training wells.",
    "blind well": "A well that played no part in building or tuning the model. The strongest test available.",
    "frozen predictions": "Predictions locked with a checksum before any answers are revealed, so the test cannot be influenced.",
    "ROC-AUC": "A 0.5-to-1 score for yes/no predictions. 0.5 is a coin flip, 1.0 is perfect.",
    "R2": "Share of the variation explained. 0 means no better than always guessing the average.",
    "depth trend": "Rock properties change steadily with burial depth. A model can score well by learning depth alone.",
    "tile": "A block of traces predicted together. Poor blending between tiles can leave a grid pattern.",
    "inline / crossline": "The two grid directions of a 3-D seismic survey.",
    "provenance": "The recorded history of where a file came from and what produced it.",
}

SUGGESTED_QUESTIONS: list[str] = [
    "Which cubes can I load and trust, and for what?",
    "How do I load the SEG-Y files correctly?",
    "Has this been tested on a blind well?",
    "What did the audit find?",
    "Why does the shale-volume cube look so smooth?",
    "What should we do before sharing these cubes?",
]

# Order in which findings are surfaced (first match on the id prefix wins).
PRIORITY: list[str] = [
    "segy_qc_failed", "cube_quality_", "files_ready", "qc_is_not_quality", "no_blind_validation", "no_depth_baseline",
    "time_window", "segy_missing", "stats_", "no_data_value", "reproducibility", "tile_seams", "xy_geometry",
    "audit_status", "well_intersections",
]
TAKEAWAY_IDS: tuple[str, ...] = ("segy_qc_failed", "cube_quality_", "qc_is_not_quality", "no_blind_validation",
                                 "no_depth_baseline", "time_window")
_FINDING_FOCUS: dict[str, tuple[str, ...]] = {
    "products": ("files_ready", "segy_qc_failed", "segy_missing", "no_data_value", "qc_is_not_quality", "xy_geometry"),
    "quality": ("cube_quality_", "no_depth_baseline", "time_window", "qc_is_not_quality", "no_blind_validation"),
    "statistics": ("stats_",),
    "audit": ("audit_status", "reproducibility", "xy_geometry"),
    "validation": ("no_blind_validation", "tile_seams", "reproducibility"),
    "wells": ("well_intersections", "no_blind_validation"),
}

RATING_NOTE = (
    "Ratings are reading conventions, not measurements: ROC-AUC below 0.55 none, below 0.65 weak, below 0.75 modest, "
    "below 0.85 good, otherwise strong, and a ROC-AUC below 0.65 whose 95% interval reaches 0.5 is also rated none; "
    "R2 at or below 0.02 none, below 0.20 weak, below 0.50 moderate, otherwise strong."
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


def _rate(kind: str, auc: float | None, ci_lo: float | None, r2: float | None) -> str:
    if kind == "binary":
        if auc is None:
            return "unrated"
        if auc < 0.55 or (ci_lo is not None and ci_lo <= 0.5 and auc < 0.65):
            return "none"
        return "weak" if auc < 0.65 else "modest" if auc < 0.75 else "good" if auc < 0.85 else "strong"
    if r2 is None:
        return "unrated"
    return "none" if r2 <= 0.02 else "weak" if r2 < 0.20 else "moderate" if r2 < 0.50 else "strong"


def _status_counts(items: list[dict[str, Any]], key: str = "status") -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        name = str(item.get(key) or "UNKNOWN").upper()
        counts[name] = counts.get(name, 0) + 1
    return counts


def _cube_model(reader: _Reader, source_job_id: str) -> dict[str, Any]:
    """Which model made the cubes: recorded only in the source inference job, next to this job folder."""
    if not source_job_id or Path(source_job_id).name != source_job_id or source_job_id in {".", ".."}:
        return {}
    jobs_root = reader.job_dir.parent
    path = jobs_root / source_job_id / "stages" / "09_inference" / "volume_inference_summary.json"
    try:
        if path.is_symlink() or (jobs_root / source_job_id).is_symlink() or not path.is_file():
            return {}
        if jobs_root not in path.resolve().parents or path.stat().st_size > MAX_JSON_BYTES:
            return {}
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    model = value.get("model") if isinstance(value, dict) else None
    if not isinstance(model, dict):
        return {}
    reader.used.append({"name": f"../{source_job_id}/stages/09_inference/volume_inference_summary.json",
                        "size_bytes": path.stat().st_size})
    return {"input": model.get("input"), "model": model.get("model")}


# --------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------
def _quality_section(reader: _Reader, cube_model: dict[str, Any], attributes: list[str], geometry: dict[str, Any],
                     findings: list[dict[str, Any]]) -> dict[str, Any]:
    table = reader.table(f"{SRC}/model_performance.csv")
    cubes, prevalence = [], None
    for task in attributes:
        row = next((r for r in table if r.get("task") == task and r.get("input") == cube_model.get("input")
                    and r.get("model") == cube_model.get("model")), None)
        rating, score = "unrated", None
        if row:
            kind = "binary" if str(row.get("kind", "")).startswith("bin") else "regression"
            auc, r2 = _num(row.get("outer_roc_auc")), _num(row.get("outer_r2"))
            lo, hi = _num(row.get("outer_roc_auc_ci_lo")), _num(row.get("outer_roc_auc_ci_hi"))
            rating = _rate(kind, auc, lo, r2)
            if kind == "binary":
                score = f"ROC-AUC {_fmt(auc, 2)}" + (f" (95% interval {_fmt(lo, 2)} to {_fmt(hi, 2)})" if lo is not None else "")
                prevalence = _num(row.get("outer_prevalence"))
            else:
                score = f"R2 {_fmt(r2, 2)}, RMSE {_fmt(row.get('outer_rmse'), 3)}"
        cubes.append({"property": TASK_NAMES.get(task, task), "task": task,
                      "made_by": cube_model.get("model"), "held_out_score": score, "rating": rating,
                      "advice": CUBE_ADVICE[rating]})
        level = {"none": "critical", "weak": "caution", "unrated": "caution", "good": "good", "strong": "good"}.get(rating, "info")
        findings.append(_finding(
            f"cube_quality_{task}", level,
            f"{TASK_NAMES.get(task, task)[:1].upper()}{TASK_NAMES.get(task, task)[1:]} cube: " + (
                f"made by {cube_model.get('model')}, held-out score {score}. Rating: {rating}." if score else
                "no held-out score could be matched to the model that made it."),
            CUBE_ADVICE[rating], [f"{SRC}/model_performance.csv"]))

    depth_rows = reader.table(f"{SRC}/beyond_depth.csv") if reader.exists(f"{SRC}/beyond_depth.csv") else []
    if table and not depth_rows:
        findings.append(_finding(
            "no_depth_baseline", "critical",
            "The imported 'beyond_depth.csv' is empty: no model was compared with a depth-only baseline.",
            "Rock properties change with burial depth, so the cubes may show little more than a depth trend. Nothing in this package rules that out.",
            [f"{SRC}/beyond_depth.csv"]))

    intervals = [r for r in reader.table(f"{SRC}/errors_by_interval.csv") if (_num(r.get("n")) or 0) > 0]
    wells = max((_num(r.get("n_wells")) or 0 for r in table), default=0)
    need = max(3.0, wells / 2)
    solid = [r for r in intervals if (_num(r.get("n_wells")) or 0) >= need and _num(r.get("twt_from_ms")) is not None]
    window = [min(_num(r["twt_from_ms"]) for r in solid), max(_num(r["twt_to_ms"]) for r in solid)] if solid else None
    t0, t1 = _num(geometry.get("t0_ms")), _num(geometry.get("t_max_ms"))
    share = None
    if window and t0 is not None and t1 is not None and t1 > t0:
        share = max(0.0, min(window[1], t1) - max(window[0], t0)) / (t1 - t0)
        findings.append(_finding(
            "time_window", "caution",
            f"The cubes span {t0:.0f} to {t1:.0f} ms, but labels from at least {need:.0f} wells exist only between "
            f"{window[0]:.0f} and {window[1]:.0f} ms ({_pct(share)} of the cube).",
            "Outside that window the model was never trained or tested. Interpret the cubes only inside it.",
            [f"{SRC}/errors_by_interval.csv", f"{S10}/volume_summary.json"]))
    return {"cube_model": cube_model or None, "cube_guide": cubes, "sand_share_at_wells": _r(prevalence),
            "well_supported_window_ms": window, "cube_time_range_ms": [t0, t1], "supported_share_of_cube": _r(share, 2),
            "depth_baseline_evaluated": bool(depth_rows),
            "evidence_status": "Scores are imported unchanged from the stage 08 evaluation; nothing is recalculated here.",
            "rating_convention": RATING_NOTE}


def _products_section(reader: _Reader, summary: dict[str, Any], geometry: dict[str, Any], cubes: list[dict[str, Any]],
                      findings: list[dict[str, Any]]) -> dict[str, Any]:
    export = summary.get("derived_segy_export") if isinstance(summary.get("derived_segy_export"), dict) else {}
    qc = reader.table(f"{S11}/tables/segy_export_qc.csv")
    native = reader.table(f"{S11}/tables/product_inventory.csv")
    no_data = None
    for row in qc:
        match = re.search(r"no-data must be (-?\d+(?:\.\d+)?)", str(row.get("detail")))
        if match:
            no_data = float(match.group(1))
            break
    files, failed_total = [], 0
    for item in export.get("products") or []:
        name = str(item.get("product"))
        mine = [r for r in qc if r.get("product") == name]
        graded = [r for r in mine if str(r.get("result")).upper() in {"PASS", "FAIL", "WARN"}]
        failed = [r.get("check") for r in graded if str(r.get("result")).upper() != "PASS"]
        failed_total += len(failed)
        relative = f"{S11}/{item.get('path')}"
        task = name.replace("prediction_", "")
        files.append({
            "property": TASK_NAMES.get(task, task), "file": relative, "present": reader.exists(relative),
            "size_gb": _r((_num(item.get("bytes")) or 0) / 1e9, 2), "status": item.get("status"),
            "qc_checks_passed": len(graded) - len(failed), "qc_checks_total": len(graded), "qc_failed": failed,
            "sha256_start": str(item.get("sha256") or "")[:12] or None,
            "rating": next((c["rating"] for c in cubes if c["task"] == task), None),
        })
    section = {
        "segy_export_status": export.get("status"), "segy_files": files,
        "native_arrays": [{"product": r.get("product"), "location": r.get("storage_root_relative_path"),
                           "size_gb": _r((_num(r.get("bytes")) or 0) / 1e9, 2), "dtype": r.get("dtype")} for r in native],
        "native_note": "The original NumPy arrays stay in the source inference job; they are referenced, not copied.",
        "how_to_load": {
            "sample_format": "IEEE float32, same byte order as the source full stack",
            "samples_per_trace": geometry.get("ns"), "sample_interval_ms": geometry.get("dt_ms"),
            "start_time_ms": geometry.get("t0_ms"), "live_traces": (geometry.get("region") or {}).get("live_traces"),
            "inline_start_and_step": [geometry.get("il_min"), geometry.get("il_step")],
            "crossline_start_and_step": [geometry.get("xl_min"), geometry.get("xl_step")],
            "no_data_value": no_data,
            "headers": "Trace and binary headers are copied from the source full stack, so use the same byte locations.",
        },
    }
    if files:
        ok = failed_total == 0 and all(str(f["status"]).upper() == "PASS" for f in files)
        total = sum(f["qc_checks_total"] for f in files)
        findings.append(_finding(
            "files_ready" if ok else "segy_qc_failed", "good" if ok else "critical",
            f"{len(files)} SEG-Y cubes were written ({files[0]['size_gb']} GB each); {total - failed_total} of {total} "
            "export checks passed" + (", including a bit-exact comparison with the source arrays and identical trace headers." if ok else "."),
            "The files are faithful copies of the predictions and load with the same geometry as the full-stack seismic."
            if ok else "At least one export check failed. Do not distribute the affected file.",
            [f"{S11}/tables/segy_export_qc.csv"]))
        weak = [f["property"] for f in files if f["rating"] in {"none", "weak", "unrated"}]
        if ok and weak:
            findings.append(_finding(
                "qc_is_not_quality", "caution",
                f"Every export check passed for all cubes, including those whose model is rated none, weak or unrated: {', '.join(weak)}.",
                "A passed export check proves the file matches the prediction. It says nothing about whether the prediction is right.",
                [f"{S11}/tables/segy_export_qc.csv", f"{SRC}/model_performance.csv"]))
        missing = [f["file"] for f in files if not f["present"]]
        if missing:
            findings.append(_finding(
                "segy_missing", "caution", f"{len(missing)} of {len(files)} SEG-Y files named in the summary are not in this job folder.",
                "The summary describes the files, but they could not be found here. They may have been moved or not copied.",
                ["geogg_volume_products_summary.json"]))
        if no_data is not None:
            findings.append(_finding(
                "no_data_value", "info", f"Samples without a prediction are stored as {no_data:g} in the SEG-Y files.",
                "Set this as the null value when loading, otherwise it will distort colour scales and statistics.",
                [f"{S11}/tables/segy_export_qc.csv"]))
    return section


def _statistics_section(reader: _Reader, volume: dict[str, Any], quality: dict[str, Any],
                        findings: list[dict[str, Any]]) -> dict[str, Any]:
    rows = reader.table(f"{S10}/tables/volume_statistics.csv")
    ratings = {c["task"]: c["rating"] for c in quality["cube_guide"]}
    out = []
    for row in rows:
        task = str(row.get("attribute"))
        stats = {k: _r(row.get(k)) for k in ("min", "p05", "p25", "p50", "p75", "p95", "max", "mean", "std")}
        out.append({"property": TASK_NAMES.get(task, task), "task": task, **stats})
        plain = TASK_NAMES.get(task, task)
        if None in (stats["p05"], stats["p95"], stats["p75"]):
            continue
        spread = stats["p95"] - stats["p05"]
        if task == "vsh" and spread < 0.35:
            findings.append(_finding(
                f"stats_{task}", "caution",
                f"The {plain} cube is compressed: 90% of its values lie between {stats['p05']:.2f} and {stats['p95']:.2f}, "
                "although the property can range from 0 to 1.",
                "A model with little skill plays safe by predicting close to the average. The cube therefore shows little real "
                "contrast, and its smooth look is not geological." + (
                    " This matches the model's held-out rating of none." if ratings.get(task) == "none" else ""),
                [f"{S10}/tables/volume_statistics.csv"]))
        elif task == "lithology":
            share = quality.get("sand_share_at_wells")
            if stats["p75"] > 0.5 and share is not None and share < 0.3:
                findings.append(_finding(
                    f"stats_{task}", "caution",
                    f"More than a quarter of the {plain} cube scores above 0.5 (median {stats['p50']:.2f}), while sand makes up "
                    f"{_pct(share)} of labelled samples at the wells.",
                    "The scores run high, so a simple 0.5 cut-off would call too much sand. Read the cube as a ranking, or choose a higher cut-off.",
                    [f"{S10}/tables/volume_statistics.csv", f"{SRC}/model_performance.csv"]))
        elif task == "phit":
            findings.append(_finding(
                f"stats_{task}", "info",
                f"The {plain} cube ranges from {stats['min']:.2f} to {stats['max']:.2f}, with 90% of values between "
                f"{stats['p05']:.2f} and {stats['p95']:.2f}.",
                "These are physically plausible porosities. Note that the automated range check only tests 0 to 1, which is loose for porosity.",
                [f"{S10}/tables/volume_statistics.csv"]))
    valid, total = _num(volume.get("valid_voxels")), _num(volume.get("candidate_voxels"))
    return {"cube_shape_inline_crossline_time": volume.get("shape"), "voxels_with_prediction": valid,
            "share_of_grid_with_prediction": _r(valid / total) if valid and total else None,
            "live_traces": volume.get("live_trace_count"), "per_property": out,
            "how_to_read": "p05 and p95 bracket the middle 90% of values. Cells outside the survey outline have no prediction."}


def _audit_section(reader: _Reader, audit: dict[str, Any], geometry: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    checks = [c for c in audit.get("checks", []) if isinstance(c, dict)]
    counts = _status_counts(checks)
    artifacts = _status_counts(reader.table(f"{S10}/tables/stage_artifact_audit.csv"))
    code = reader.table(f"{S10}/tables/code_currency.csv")
    code_missing = [r.get("stage") for r in code if str(r.get("status")).upper() == "NOT_RECORDED"]
    scorecard = reader.table(f"{S10}/tables/stage_scorecard.csv")
    section = {
        "audit_status": audit.get("status"), "check_counts": counts,
        "checks_not_passed": [{"check": c.get("check"), "status": c.get("status")} for c in checks if str(c.get("status")).upper() != "PASS"],
        "upstream_stages": [{"stage": r.get("stage"), "job": str(r.get("processing_job_id"))[:8], "status": r.get("status"),
                             "artifact_audit": r.get("artifact_audit_status")} for r in scorecard],
        "upstream_artifact_hash_counts": artifacts,
        "stages_without_code_record": code_missing,
        "output_integrity": _status_counts(reader.table(f"{S10}/tables/output_integrity.csv")),
        "input_lineage": _status_counts(reader.table(f"{S10}/tables/lineage_consistency.csv")),
        "limitations": [_clip(x, 260) for x in audit.get("limitations") or []],
    }
    failed = [c for c in checks if str(c.get("status")).upper() in {"FAIL", "FAILED", "ERROR"}]
    if checks:
        findings.append(_finding(
            "audit_status", "critical" if failed else "info",
            f"Audit status is '{audit.get('status')}': " + ", ".join(f"{n} {k}" for k, n in sorted(counts.items())) + " acceptance checks.",
            "At least one acceptance check failed; resolve it before using the package." if failed else
            "No check failed. The warnings come from upstream stages that finished with stated limitations, which this package inherits.",
            [f"{S10}/summary_audit.json"]))
    unhashed, hashed = artifacts.get("NOT_HASHED", 0), artifacts.get("PASS", 0)
    if code_missing or unhashed:
        findings.append(_finding(
            "reproducibility", "caution",
            f"Code identity was not recorded for {len(code_missing)} upstream stage(s), and {unhashed} of {unhashed + hashed} "
            "upstream artifacts carry no checksum.",
            "The cubes and their direct inputs are checksummed and verified, but the run cannot be proven reproducible end to end, "
            "because the exact code that produced it is unknown.",
            [f"{S10}/tables/code_currency.csv", f"{S10}/tables/stage_artifact_audit.csv"]))
    reference = geometry.get("spatial_reference") if isinstance(geometry.get("spatial_reference"), dict) else {}
    if reference and reference.get("xy_mapping_supported") is False:
        findings.append(_finding(
            "xy_geometry", "caution",
            f"Map X/Y cannot be computed from inline and crossline by a simple formula: the worst mismatch is "
            f"{_fmt(reference.get('affine_residual_max_m'), 1)} m (typical {_fmt(reference.get('affine_residual_rms_m'), 2)} m).",
            "Work in inline and crossline. For map positions use the trace coordinates stored in the SEG-Y headers, which are copied from the source.",
            [f"{S10}/volume_summary.json"]))
    return section


def _validation_section(reader: _Reader, findings: list[dict[str, Any]]) -> dict[str, Any]:
    validation = reader.json(f"{S11}/validation_summary.json")
    protocol = reader.json(f"{S11}/validation/blind_well_protocol.json")
    freeze = reader.json(f"{S11}/validation/prediction_freeze_manifest.json")
    repeat = reader.json(f"{S11}/validation/reproducibility_report.json")
    parity = reader.table(f"{S11}/validation/feature_tile_parity.csv")
    checks = [c for c in validation.get("validation_checks", []) if isinstance(c, dict)]
    status = {str(c.get("check")): str(c.get("status")) for c in checks}
    section = {
        "validation_status": validation.get("validation_status"), "product_qc_status": validation.get("native_product_qc_status"),
        "product_checks": _status_counts([c for c in validation.get("checks", []) if isinstance(c, dict)]),
        "validation_checks": [{"check": c.get("check"), "status": c.get("status"),
                               "detail": _clip(c.get("detail"), 160) if isinstance(c.get("detail"), str) else None} for c in checks],
        "blind_well_protocol_state": protocol.get("protocol_state"), "truth_reveal": protocol.get("truth_reveal"),
        "protocol_steps": protocol.get("steps"), "protocol_prohibitions": protocol.get("prohibitions"),
        "predictions_frozen": freeze.get("freeze_state"), "frozen_at_utc": freeze.get("frozen_at_utc"),
        "freeze_sha256_start": str(freeze.get("freeze_sha256") or "")[:12] or None,
        "repeated_inference": repeat.get("repeated_inference"),
        "parity_tests": [{"check": r.get("check"), "status": r.get("status")} for r in parity],
    }
    if status.get("per_well_blind_validation", "").upper() == "NOT_RUN" or status.get("blind_well_registry", "").upper() == "NOT_CONFIGURED":
        frozen = str(freeze.get("freeze_state") or "").upper() == "FROZEN"
        findings.append(_finding(
            "no_blind_validation", "caution",
            "Blind-well validation has not been run: no independent wells are registered"
            + (", though the predictions are frozen and ready for it." if frozen else "."),
            "The cubes have never been checked against a well that played no part in building them. The only accuracy evidence is "
            "the cross-validated scoring of the training wells." + (
                " Freezing first is good practice: once blind wells are supplied, the test cannot be influenced by the answers." if frozen else ""),
            [f"{S11}/validation_summary.json", f"{S11}/validation/blind_well_protocol.json"]))
    untested = [r.get("check") for r in parity if str(r.get("status")).upper() == "NOT_TESTABLE"]
    if "tile_seam_consistency" in untested:
        findings.append(_finding(
            "tile_seams", "caution",
            "Consistency across tile boundaries could not be tested (the cubes were predicted in 32 by 32 trace tiles).",
            "Tile-edge artefacts are not ruled out. On time slices, watch for a faint regular grid pattern; if present, it is processing, not geology.",
            [f"{S11}/validation/feature_tile_parity.csv"]))
    return section


def _wells_section(reader: _Reader, findings: list[dict[str, Any]]) -> dict[str, Any]:
    rows = reader.table(f"{S10}/tables/well_intersection_status.csv")
    sections = reader.json(f"{S10}/seismic_sections.json")
    per_well = [{"well": r.get("well"), "status": r.get("status"), "path_samples": _num(r.get("trajectory_sample_count")),
                 "samples_with_prediction": _num(r.get("samples_with_valid_prediction"))} for r in rows]
    if rows:
        hit = [r for r in rows if str(r.get("status")) == "intersecting_valid_prediction"]
        findings.append(_finding(
            "well_intersections", "info",
            f"{len(hit)} of {len(rows)} well paths pass through predicted parts of the cubes.",
            "These are the wells the models were trained and cross-validated on, so reading cube values at them is not an independent test.",
            [f"{S10}/tables/well_intersection_status.csv"]))
    return {"per_well": per_well, "method": _clip(sections.get("well_intersection_method"), 200) or None,
            "detail_table": f"{S10}/tables/well_volume_intersections.csv (cube values sampled along each well path)",
            "well_points_drawn_on_sections": sections.get("well_section_overlay_count")}


def _figures(reader: _Reader) -> dict[str, Any]:
    table = reader.table(f"{S10}/tables/seismic_sections.csv")
    sections = [{"property": TASK_NAMES.get(str(r.get("attribute")), r.get("attribute")), "type": r.get("section_type"),
                 "at": _num(r.get("coordinate")) if r.get("coordinate") else _num(r.get("time_ms")),
                 "file": f"{S10}/{r.get('path')}"} for r in table if reader.exists(f"{S10}/{r.get('path')}")]
    return {
        "dashboard": {"file": f"{S10}/dashboard.html", "present": reader.exists(f"{S10}/dashboard.html"),
                      "size_mb": _r((reader.size(f"{S10}/dashboard.html") or 0) / 1e6, 0),
                      "shows": "Offline one-page overview of the package. It embeds the figures, so treat it as confidential."},
        "sections": sections,
        "sections_show": "Seismic in grey with the prediction in colour on inlines, crosslines and time slices. Cells without a prediction are light grey.",
        "evaluation_figures": [p for p in (f"{S10}/figures/evaluation/{n}.png" for n in (
            "model_comparison_outer", "per_well_metrics", "calibration", "confusion_matrices", "roc_pr_curves", "true_vs_predicted"))
            if reader.exists(p)],
        "model_cards": f"{S10}/model_cards/ (one card per registered model)",
    }


# --------------------------------------------------------------------------------------------
# Verdict and next steps
# --------------------------------------------------------------------------------------------
def _verdict(status: str, sections: dict[str, Any], has_stage10: bool, has_stage11: bool) -> dict[str, Any]:
    files = sections["products"]["segy_files"]
    audit_failed = [c["check"] for c in sections["audit"]["checks_not_passed"] if str(c["status"]).upper() in {"FAIL", "FAILED", "ERROR"}]
    qc_failed = [f["property"] for f in files if f["qc_failed"] or str(f["status"]).upper() != "PASS"]
    cubes = sections["quality"]["cube_guide"]
    bad = [c["property"] for c in cubes if c["rating"] in {"none", "unrated"}]
    if str(status).lower() not in {"succeeded", "success", "completed"}:
        level, plain = "did_not_finish", f"The job status is '{status}', so this is not a completed result."
    elif not (has_stage10 and has_stage11):
        level, plain = "incomplete", "The run finished but the audit summary or the export validation summary is missing."
    elif audit_failed or qc_failed:
        level = "needs_attention"
        plain = (f"{len(audit_failed)} audit check(s) and {len(qc_failed)} cube export(s) failed. Resolve these before distributing anything.")
    else:
        level = "usable_with_caveats"
        plain = (f"The package is complete: {len(files)} SEG-Y cubes were written and every export check passed, so the files are safe to load. "
                 "Packaging adds no new accuracy evidence.")
        if bad:
            plain += f" {len(bad)} of {len(cubes)} cubes should not be interpreted ({', '.join(bad)})."
        if str(sections["validation"]["validation_status"]).lower() != "completed":
            plain += " Blind-well validation has not been run."
    return {"level": level, "plain": plain, "audit_status": sections["audit"]["audit_status"],
            "audit_checks": sections["audit"]["check_counts"], "validation_status": sections["validation"]["validation_status"],
            "note": "File integrity and scientific quality are separate. See the cube guide for quality."}


def _next_steps(findings: list[dict[str, Any]], verdict: dict[str, Any]) -> list[str]:
    ids = {f["id"]: f["level"] for f in findings}
    steps: list[str] = []
    if verdict["level"] in {"did_not_finish", "incomplete", "needs_attention"}:
        steps.append("Resolve the failed or missing items first; do not distribute the cubes until then.")
    if any(k.startswith("cube_quality_") and v == "critical" for k, v in ids.items()):
        steps.append("Withhold or clearly label any cube rated 'none' before sharing the package.")
    if "no_blind_validation" in ids:
        steps.append("Register one or more independent wells and run the blind-well protocol against the frozen predictions.")
    if "no_depth_baseline" in ids:
        steps.append("Add a depth-only baseline to the evaluation to show what the seismic adds beyond burial depth.")
    if "time_window" in ids:
        steps.append("Crop or mask the cubes to the time window that has well control.")
    if "no_data_value" in ids:
        steps.append("Tell whoever loads the SEG-Y files which no-data value to set.")
    if "reproducibility" in ids:
        steps.append("Record the code version at launch in future runs so the result can be reproduced.")
    if "tile_seams" in ids:
        steps.append("Inspect a few time slices for a regular 32-trace grid pattern before interpreting fine detail.")
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
    job_id = str(parameters.get("job_id") or metadata.get("job_id") or job_dir.name)
    script = str(metadata.get("python_script") or result.get("python_script") or "")
    status = str(result.get("status") or summary.get("status") or metadata.get("status") or "unknown")
    source = str(summary.get("source_inference_job_id") or "")
    base = {
        "skill": SKILL_NAME, "job_id": job_id, "python_script": script or None, "operation": summary.get("operation") or metadata.get("operation"),
        "target_well": (str(metadata.get("target_well") or "").strip() or None), "status": status,
        "created_at": metadata.get("created_at") or result.get("created_at"),
        "source_inference_job_id": source or None,
    }
    recognised = script == EXPECTED_SCRIPT or ("summary_stage" in summary and "volume_export_stage" in summary)
    if not recognised or not summary:
        reason = (f"This folder was not produced by Process C (expected script '{EXPECTED_SCRIPT}')." if not recognised else
                  f"'{SUMMARY_FILE}' is missing or unreadable, so there is nothing to analyse.")
        if status.lower() not in {"succeeded", "success", "completed"} and recognised:
            reason += f" The job status is '{status}', so the run may have stopped before writing it."
        return {**base, "applicable": False, "summary": reason, "findings": [reason], "artifacts": reader.used,
                "skipped": reader.skipped,
                "limitations": ["Use the generic processing_output_folder_analysis skill to inventory this folder."]}

    volume = reader.json(f"{S10}/volume_summary.json")
    audit = reader.json(f"{S10}/summary_audit.json")
    geometry = volume.get("geometry") if isinstance(volume.get("geometry"), dict) else {}
    attributes = [str(a) for a in (volume.get("attributes") or {})]
    cube_model = _cube_model(reader, source)

    findings: list[dict[str, Any]] = []
    quality = _quality_section(reader, cube_model, attributes, geometry, findings)
    sections = {
        "quality": quality,
        "products": _products_section(reader, summary, geometry, quality["cube_guide"], findings),
        "statistics": _statistics_section(reader, volume, quality, findings),
        "audit": _audit_section(reader, audit, geometry, findings),
        "validation": _validation_section(reader, findings),
        "wells": _wells_section(reader, findings),
    }
    has_stage11 = bool(sections["validation"]["validation_checks"] or sections["validation"]["product_checks"])
    verdict = _verdict(status, sections, bool(audit), has_stage11)

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
        "foundation_job_id": geometry.get("foundation_job_id"), "model_release_id": audit.get("model_release_id"),
        "verdict": verdict, "summary": verdict["plain"],
        "takeaways": takeaways,
        "cube_guide": [{**c, **{k: f[k] for k in ("file", "present", "size_gb")}} for c in quality["cube_guide"]
                       for f in sections["products"]["segy_files"] if f["property"] == c["property"]] or quality["cube_guide"],
        "key_findings": shown,
        "findings": [f["observed"] for f in shown],
        "next_steps": _next_steps(findings, verdict),
        "details": details,
        "findings_not_shown": len(findings) - len(shown) if focus == "overview" else 0,
        "more_detail": "Call again with focus = products, quality, statistics, audit, validation, wells, figures or all.",
    }
    if focus in {"all", "glossary"}:
        response["glossary"] = GLOSSARY
    if focus in {"all", "figures"}:
        response["figures_guide"] = _figures(reader)
    if focus == "overview":
        response["suggested_questions"] = SUGGESTED_QUESTIONS
    response["artifacts"] = [{**item, "role": "evidence read by this analysis"} for item in reader.used]
    response["skipped"] = reader.skipped
    response["limitations"] = [
        "Numbers are read from the run's own summaries and tables; nothing is recomputed, and the multi-gigabyte cube files are never opened.",
        "Accuracy scores are imported from the stage 08 evaluation of the training wells. They are not blind-well results.",
        RATING_NOTE,
        "Plain-language meanings are fixed expert reading rules applied to the observed numbers, not proof of a geological cause.",
        "Figures and the dashboard are listed with what they show; their content is not inspected.",
    ]
    if not cube_model:
        response["limitations"].insert(0, "The model that made the cubes is recorded only in the source inference job, which could not "
                                          "be read from here, so the cubes are unrated.")
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
