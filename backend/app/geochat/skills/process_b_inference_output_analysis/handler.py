"""Expert, deterministic reading of one Process B (GeoGG evaluation and inference, stages 08-09) run folder.

The handler measures and applies fixed, documented reading rules. It never calls an LLM and never
invents a value: every number in the result is read from a file inside the job folder, and every
file it used is listed under ``artifacts``. The multi-gigabyte prediction cubes are never opened.
The GeoChat agent explains the returned evidence.

This module is self-contained on purpose: it does not import the generic
``processing_output_folder_analysis`` handler or any other skill.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

SKILL_NAME = "process_b_inference_output_analysis"
EXPECTED_OPERATION = "geogg_stage_f_08_09"
EXPECTED_SCRIPT = "process_B_inference.py"
SUMMARY_FILE = "geogg_inference_summary.json"
EVAL = "stages/08_evaluation"
INFER = "stages/09_inference"

MAX_JSON_BYTES = 8_000_000
MAX_CSV_BYTES = 30_000_000
MAX_CSV_ROWS = 60_000
MAX_TEXT = 420
OVERVIEW_FINDINGS = 12

# --------------------------------------------------------------------------------------------
# Static knowledge about the process (what each stage is for, in plain words)
# --------------------------------------------------------------------------------------------
STAGES: dict[str, dict[str, str]] = {
    "08_evaluation": {
        "title": "Final exam on unseen wells",
        "plain": "Scores every model on wells it never trained on, using predictions that were sealed during training.",
    },
    "09_inference": {
        "title": "Prediction cubes for the whole survey",
        "plain": "Runs the chosen model over every live trace of the seismic survey and writes one cube per predicted property.",
    },
}

FOCUS_ALIASES: dict[str, str] = {
    "": "overview", "overview": "overview", "summary": "overview", "all": "all", "full": "all",
    "performance": "performance", "evaluation": "performance", "models": "performance", "model": "performance",
    "scores": "performance", "accuracy": "performance", "08": "performance", "8": "performance",
    "comparisons": "comparisons", "comparison": "comparisons", "inversion": "comparisons", "paired": "comparisons",
    "uncertainty": "uncertainty", "calibration": "uncertainty", "confidence": "uncertainty", "intervals": "uncertainty",
    "by_time": "by_time", "time": "by_time", "depth": "by_time", "by_depth": "by_time", "errors": "by_time",
    "wells": "wells", "per_well": "wells", "by_well": "wells",
    "volume": "volume", "volumes": "volume", "cube": "volume", "cubes": "volume", "inference": "volume",
    "09": "volume", "9": "volume",
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
    "held-out well": "A well kept completely away from training. Scoring on it shows how the model behaves somewhere new.",
    "sealed predictions": "Predictions for held-out wells written during training and not touched again, so the exam cannot be re-sat.",
    "ROC-AUC": "A 0.5-to-1 score for yes/no predictions. 0.5 is a coin flip, 1.0 is perfect.",
    "RMSE": "Typical size of the prediction error, in the units of the property (fractions here).",
    "R2": "Share of the variation explained. 0 means no better than always guessing the average; it can be negative.",
    "95% interval": "The range the true score plausibly lies in, found by re-drawing whole wells many times (a well bootstrap).",
    "optimism": "Practice score minus held-out score. Near zero means the practice scores were honest.",
    "paired comparison": "Two models scored on exactly the same held-out rows, so the difference is not caused by different data.",
    "calibration": "Whether a predicted 70% comes true about 70% of the time. A model can rank well and still be badly calibrated.",
    "precision": "Of the samples called sand, the share that are sand.",
    "recall": "Of the samples that are sand, the share the model found.",
    "prediction interval": "A band around a predicted value that should contain the true value a stated share of the time.",
    "coverage": "How often the true value fell inside the prediction interval.",
    "depth trend": "Rock properties change steadily with burial depth. A model can score well by learning depth alone.",
    "TWT": "Two-way time: how long sound takes to travel down to a layer and back, in milliseconds. Seismic is recorded in TWT.",
    "matched rows": "The rows where inversion data exist. Stacks-only and stacks-plus-inversion models are compared on these same rows.",
    "voxel": "One cell of a 3-D cube: one inline, one crossline, one time sample.",
    "live trace": "A seismic trace that contains data. Positions outside the survey outline are empty.",
    "tile": "A block of traces processed together. The cube is built tile by tile so a long run can resume.",
    "extrapolation": "Using a model outside the range of conditions it was trained and tested on.",
    "VSH": "Shale volume, a fraction from 0 (clean sand) to 1 (pure shale).",
    "PHIT": "Total porosity: the fraction of the rock that is pore space.",
}

FIGURES: list[tuple[str, str]] = [
    (f"{EVAL}/figures/model_comparison_outer.png", "All candidates side by side on the held-out score, with intervals."),
    (f"{EVAL}/figures/per_well_metrics.png", "Held-out score well by well; look for wells that stand apart."),
    (f"{EVAL}/figures/roc_pr_curves.png", "Trade-off curves for the yes/no tasks."),
    (f"{EVAL}/figures/confusion_matrices.png", "Counts of right and wrong calls for the yes/no tasks."),
    (f"{EVAL}/figures/calibration.png", "Predicted probability against how often it came true; the diagonal is ideal."),
    (f"{EVAL}/figures/true_vs_predicted.png", "Predicted against log-derived values for porosity, shale volume and saturation."),
]

SUGGESTED_QUESTIONS: list[str] = [
    "Which of the prediction cubes can I trust, and for what?",
    "How well do the models work on wells they have never seen?",
    "Does adding the inversion volumes really help?",
    "Is deep learning better than the simpler models here?",
    "Over what time range is the cube reliable?",
    "How uncertain is a porosity value read from the cube?",
    "How did the models do at one particular well?",
    "What should we fix first to improve the results?",
]

# Order in which findings are surfaced (first match on the id prefix wins).
PRIORITY: list[str] = [
    "evaluation_missing", "cube_model_", "no_depth_baseline", "time_window", "cube_complete", "cube_files_missing", "best_", "coverage_gap_",
    "calibration", "sand_hit_rate", "time_", "inversion_", "deep_vs_tabular_", "interval_", "well_spread_",
    "optimism", "no_self_checks", "xy_geometry", "no_fluid_cube",
]
TAKEAWAY_IDS: list[str] = ["evaluation_missing", "cube_complete", "cube_model_", "no_depth_baseline", "time_window", "calibration", "cube_files_missing"]

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


# Which focus each finding belongs to (used to filter findings in a focused answer).
_FINDING_FOCUS: dict[str, tuple[str, ...]] = {
    "performance": ("best_", "cube_model_", "optimism", "coverage_gap_", "no_depth_baseline", "sand_hit_rate", "no_self_checks"),
    "comparisons": ("inversion_", "deep_vs_tabular_", "coverage_gap_"),
    "uncertainty": ("calibration", "interval_", "sand_hit_rate"),
    "by_time": ("time_", "no_depth_baseline"),
    "wells": ("well_spread_",),
    "volume": ("cube_", "time_window", "xy_geometry", "no_fluid_cube", "calibration", "interval_"),
}


def _finding(fid: str, level: str, observed: str, meaning: str, evidence: list[str]) -> dict[str, Any]:
    """level: good | info | caution | critical."""
    return {"id": fid, "level": level, "observed": observed, "plain_meaning": meaning, "evidence": evidence}


def _light(status: str, failed: int, blockers: int, limitations: int) -> str:
    status = (status or "").upper()
    if blockers or failed or status in {"FAILED", "BLOCKED", "ERROR"}:
        return "red"
    if limitations or status == "CONDITIONAL":
        return "amber"
    if status in {"COMPLETED", "PASS", "PASSED", "OK", "SUCCEEDED"}:
        return "green"
    return "grey"


def _stage_cards(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cards = []
    for record in records:
        sid = str(record.get("stage_id") or "")
        info = STAGES.get(sid, {"title": sid, "plain": ""})
        checks = record.get("checks") if isinstance(record.get("checks"), dict) else {}
        failed = [{"check": name, "detail": _clip(body.get("detail"))} for name, body in checks.items()
                  if isinstance(body, dict) and body.get("passed") is not True]
        blockers = [_clip(b) for b in (record.get("blockers") or [])]
        limitations = [_clip(x) for x in (record.get("limitations") or [])]
        cards.append({
            "stage_id": sid, "title": info["title"], "what_it_does": info["plain"],
            "status": record.get("status") or "unknown",
            "light": _light(str(record.get("status")), len(failed), len(blockers), len(limitations)),
            "checks_passed": len(checks) - len(failed), "checks_total": len(checks),
            "passed_checks": [{"check": name, "detail": _clip(body.get("detail"), 200)} for name, body in checks.items()
                              if isinstance(body, dict) and body.get("passed") is True],
            "failed_checks": failed, "blockers": blockers, "limitations": limitations,
            "warnings": [_clip(w) for w in (record.get("warnings") or [])][:6],
            "n_outputs": len(record.get("outputs") or []),
        })
    return cards


# --------------------------------------------------------------------------------------------
# Reading rules
# --------------------------------------------------------------------------------------------
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


RATING_MEANING: dict[str, str] = {
    "none": "On wells it never trained on, the model does no better than a naive guess.",
    "weak": "There is a real but small signal on unseen wells. It is not strong enough to rely on by itself.",
    "modest": "There is a usable signal on unseen wells, clearly better than chance but far from decisive.",
    "moderate": "On unseen wells the model explains a meaningful part of the variation, with sizeable error remaining.",
    "good": "The model separates the classes well on unseen wells.",
    "strong": "The score is high enough that leakage should be ruled out before it is believed.",
    "unrated": "No score was available to rate.",
}
CUBE_ADVICE: dict[str, str] = {
    "none": "Do not interpret this cube. The model that made it showed no skill on unseen wells.",
    "weak": "Use for qualitative screening only, alongside other evidence, never as the sole basis for a decision.",
    "modest": "Use as a relative indicator of where the property is more or less likely. Not for quantitative work.",
    "moderate": "Use as a relative indicator with the stated uncertainty band. Not for volumetrics without calibration.",
    "good": "Usable for interpretation. Confirm against a blind well before quantitative use.",
    "strong": "Usable, but first rule out leakage and confirm against a blind well.",
    "unrated": "No held-out score was found for this cube's model, so its reliability is unknown.",
}


def _score_text(row: dict[str, Any]) -> str:
    if row["kind"] == "binary":
        text = f"ROC-AUC {_fmt(row['roc_auc'], 2)}"
        if row["ci"][0] is not None:
            text += f" (95% interval {_fmt(row['ci'][0], 2)} to {_fmt(row['ci'][1], 2)})"
        return text
    text = f"R2 {_fmt(row['r2'], 2)}, RMSE {_fmt(row['rmse'], 3)}"
    if row["ci"][0] is not None:
        text += f" (95% interval {_fmt(row['ci'][0], 3)} to {_fmt(row['ci'][1], 3)})"
    return text


def _performance(reader: _Reader) -> list[dict[str, Any]]:
    rows = []
    for row in reader.table(f"{EVAL}/tables/model_performance.csv"):
        kind = "binary" if str(row.get("kind", "")).startswith("bin") else "regression"
        auc, r2 = _num(row.get("outer_roc_auc")), _num(row.get("outer_r2"))
        ci = ([_num(row.get("outer_roc_auc_ci_lo")), _num(row.get("outer_roc_auc_ci_hi"))] if kind == "binary"
              else [_num(row.get("outer_rmse_ci_lo")), _num(row.get("outer_rmse_ci_hi"))])
        rows.append({
            "task": row.get("task"), "input": row.get("input"), "model": row.get("model"), "kind": kind,
            "rows": _num(row.get("n_rows")), "wells": _num(row.get("n_wells")),
            "roc_auc": _r(auc), "balanced_accuracy": _r(row.get("outer_balanced_acc")),
            "prevalence": _r(row.get("outer_prevalence")), "brier": _r(row.get("outer_brier")),
            "rmse": _r(row.get("outer_rmse"), 4), "mae": _r(row.get("outer_mae"), 4), "r2": _r(r2),
            "spearman": _r(row.get("outer_spearman")), "bias": _r(row.get("outer_bias"), 4),
            "ci": [_r(ci[0], 4), _r(ci[1], 4)],
            "practice_score": _r(row.get("inner_roc_auc") if kind == "binary" else row.get("inner_rmse"), 4),
            "optimism": _r(row.get("optimism_inner_minus_outer"), 4),
            "per_well_range": ([_r(row.get("within_well_roc_auc_min")), _r(row.get("within_well_roc_auc_max"))] if kind == "binary"
                               else [_r(row.get("within_well_rmse_min"), 4), _r(row.get("within_well_rmse_max"), 4)]),
            "declared_champion": _truthy(row.get("declared_champion")),
            "rating": _rate(kind, auc, ci[0] if kind == "binary" else None, r2),
        })
    return rows


def _skill_value(row: dict[str, Any]) -> float:
    value = row["roc_auc"] if row["kind"] == "binary" else row["r2"]
    return value if value is not None else -9.0


def _best(rows: list[dict[str, Any]], task: Any, input_name: str) -> dict[str, Any] | None:
    pool = [r for r in rows if r["task"] == task and r["input"] == input_name]
    return max(pool, key=_skill_value) if pool else None


# --------------------------------------------------------------------------------------------
# Sections
# --------------------------------------------------------------------------------------------
def _performance_section(reader: _Reader, rows: list[dict[str, Any]], evaluation: dict[str, Any],
                         findings: list[dict[str, Any]]) -> dict[str, Any]:
    scorecard = []
    for task in dict.fromkeys(r["task"] for r in rows):
        plain = TASK_NAMES.get(str(task), str(task))
        stacks_all, stacks_m = _best(rows, task, "stacks"), _best(rows, task, "stacks@matched")
        gap = None
        if stacks_all and stacks_m:
            a, b = _skill_value(stacks_all), _skill_value(stacks_m)
            metric = "ROC-AUC" if stacks_all["kind"] == "binary" else "R2"
            flagged = (a - b) >= (0.08 if metric == "ROC-AUC" else 0.15)
            gap = {"metric": metric, "all_rows": a, "matched_rows": b, "drop": _r(a - b), "flagged": flagged}
        pool = [r for r in rows if r["task"] == task]
        if gap and gap["flagged"]:
            pool = [r for r in pool if str(r["input"]).endswith("@matched")] or pool
        top = max(pool, key=_skill_value)
        scorecard.append({
            "target": plain, "task": task, "input": top["input"], "model": top["model"], "score": _score_text(top),
            "rating": top["rating"], "wells": top["wells"], "rows": top["rows"],
            "basis": ("matched rows, because the all-rows score is inflated by the coverage boundary"
                      if gap and gap["flagged"] else "best candidate on held-out wells"),
        })
        level = {"none": "caution", "weak": "caution", "good": "good", "strong": "good"}.get(top["rating"], "info")
        findings.append(_finding(
            f"best_{task}", level,
            f"Fairest held-out score for {plain}: {_score_text(top)} ({top['model']}, {INPUT_NAMES.get(str(top['input']), top['input'])}). "
            f"Rating: {top['rating']}.",
            RATING_MEANING[top["rating"]], [f"{EVAL}/tables/model_performance.csv"]))
        if gap and gap["flagged"]:
            findings.append(_finding(
                f"coverage_gap_{task}", "caution",
                f"For {plain}, the stacks-only held-out {gap['metric']} falls from {_fmt(gap['all_rows'])} on all rows to "
                f"{_fmt(gap['matched_rows'])} on the rows that have inversion data.",
                "Much of the all-rows score appears to come from telling apart the zones with and without inversion coverage, "
                "which is largely a depth effect, and not from recognising the rock within a zone. The matched-row score is the fairer one.",
                [f"{EVAL}/tables/model_performance.csv"]))

    optimism = [abs(r["optimism"]) for r in rows if r["optimism"] is not None and r["kind"] == "binary"]
    if optimism:
        worst = max(optimism)
        findings.append(_finding(
            "optimism", "good" if worst <= 0.05 else "caution",
            f"Practice scores and held-out scores agree closely: the largest ROC-AUC gap is {worst:.3f}.",
            "The practice scores from Process A were honest predictors of held-out performance, which is a sign of a leak-free workflow."
            if worst <= 0.05 else
            "Some practice scores were noticeably better than the held-out result, so model selection was partly fitting noise.",
            [f"{EVAL}/tables/model_performance.csv"]))

    depth_rows = (evaluation.get("tables") or {}).get("beyond_depth.csv")
    depth_file = reader.table(f"{EVAL}/tables/beyond_depth.csv") if reader.exists(f"{EVAL}/tables/beyond_depth.csv") else []
    depth_tested = bool(depth_file) or bool(_num(depth_rows))
    if rows and not depth_tested:
        findings.append(_finding(
            "no_depth_baseline", "critical",
            "The 'beyond_depth.csv' table is empty: no model was compared with a depth-only baseline.",
            "Rock properties change steadily with burial depth, so a model can score well by learning depth alone. This run does not "
            "show that any model sees more than depth. Until that test exists, treat every score as an upper limit on what the seismic contributes.",
            [f"{EVAL}/tables/beyond_depth.csv"]))
    return {
        "source": evaluation.get("source"), "policy": evaluation.get("evaluation_policy"),
        "prediction_rows": evaluation.get("prediction_rows"), "candidates": evaluation.get("groups"),
        "wells": evaluation.get("well_ids"), "bootstrap": evaluation.get("bootstrap"),
        "depth_baseline_evaluated": depth_tested,
        "task_names": TASK_NAMES, "input_names": INPUT_NAMES,
        "scorecard": scorecard,
        "scoreboard": [{k: v for k, v in r.items() if v is not None and k != "kind"} for r in rows],
        "rating_convention": RATING_NOTE,
    }


def _comparisons_section(reader: _Reader, weak_tasks: dict[str, bool], findings: list[dict[str, Any]]) -> dict[str, Any]:
    table = reader.table(f"{EVAL}/tables/paired_comparisons.csv")
    rows = [{
        "task": r.get("task"), "comparison": r.get("comparison"), "model": r.get("model"), "input": r.get("input"),
        "reference_model": r.get("reference_model"), "reference_input": r.get("reference_input"), "metric": r.get("metric"),
        "value": _r(r.get("metric_a"), 4), "reference_value": _r(r.get("metric_b"), 4), "difference": _r(r.get("difference"), 4),
        "interval": [_r(r.get("ci_lo"), 4), _r(r.get("ci_hi"), 4)],
        "wells_better": _num(r.get("wells_a_better")), "wells": _num(r.get("n_wells")), "verdict": r.get("verdict"),
    } for r in table]
    for prefix, fid, subject in (("fused_vs_stacks", "inversion", "Adding the inversion volumes"),
                                 ("patch_champion_vs_tabular", "deep_vs_tabular", "The deep-learning model")):
        for task in dict.fromkeys(r["task"] for r in rows):
            group = [r for r in rows if r["task"] == task and str(r["comparison"]).startswith(prefix)]
            if not group:
                continue
            better = [r for r in group if str(r["verdict"]).lower() == "a better"]
            worse = [r for r in group if str(r["verdict"]).lower() == "b better"]
            shown = "; ".join(
                f"{r['model']}: {r['metric']} {_fmt(r['reference_value'])} to {_fmt(r['value'])}, "
                f"better in {int(r['wells_better'] or 0)} of {int(r['wells'] or 0)} wells ({r['verdict']})" for r in group)
            plain = TASK_NAMES.get(str(task), str(task))
            if len(better) == len(group):
                meaning = f"{subject} gives a consistent gain for this target; the interval on the difference excludes zero."
            elif better:
                meaning = (f"{subject} helps with one model type but not the other, so the benefit is not robust. "
                           "Treat it as unproven.")
            elif worse:
                meaning = f"{subject} makes the result worse for this target."
            else:
                meaning = f"{subject} makes no detectable difference for this target on held-out wells."
            if better and fid == "inversion" and weak_tasks.get(str(task)):
                meaning += " The models for this target remain weak overall, so the gain has little practical value yet."
            findings.append(_finding(f"{fid}_{task}", "info", f"{subject} for {plain}: {shown}.", meaning,
                                     [f"{EVAL}/tables/paired_comparisons.csv"]))
    return {"paired_comparisons": rows,
            "how_to_read": "Each row compares two candidates on the same held-out rows. 'difference' is value minus reference "
                           "value; for RMSE a negative difference is an improvement. The interval comes from resampling whole wells."}


def _ece(bins: list[dict[str, str]]) -> tuple[float | None, dict[str, Any] | None]:
    total = sum(_num(b.get("n")) or 0 for b in bins)
    if not total:
        return None, None
    ece = sum((_num(b.get("n")) or 0) * abs((_num(b.get("mean_predicted")) or 0) - (_num(b.get("observed_frequency")) or 0))
              for b in bins) / total
    top = max(bins, key=lambda b: _num(b.get("mean_predicted")) or 0)
    return ece, {"mean_predicted": _r(top.get("mean_predicted")), "observed_frequency": _r(top.get("observed_frequency")),
                 "n": _num(top.get("n"))}


def _uncertainty_section(reader: _Reader, deployed: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    key = (deployed.get("input"), deployed.get("model"))
    calibration = reader.table(f"{EVAL}/tables/calibration_bins.csv")
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in calibration:
        groups.setdefault((str(row.get("task")), str(row.get("input")), str(row.get("model"))), []).append(row)
    calib_out = []
    for (task, input_name, model), bins in groups.items():
        ece, top = _ece(bins)
        calib_out.append({"task": task, "input": input_name, "model": model, "expected_calibration_error": _r(ece),
                          "most_confident_bin": top, "for_cube_model": (input_name, model) == key})
        if (input_name, model) == key and ece is not None and top:
            over = (top["mean_predicted"] or 0) - (top["observed_frequency"] or 0)
            findings.append(_finding(
                "calibration", "caution" if ece > 0.05 else "good",
                f"For the cube's {TASK_NAMES.get(task, task)} output, where the model is most confident it predicts "
                f"{_pct(top['mean_predicted'])} on average but the label is true {_pct(top['observed_frequency'])} of the time "
                f"(average calibration gap {ece:.2f}).",
                ("The numbers in this cube overstate the chance of sand. Read them as a ranking score, where higher means more "
                 "sand-like, and not as a probability." if over > 0.05 else
                 "The numbers in this cube understate the chance of sand. Read them as a ranking score, not as a probability."
                 if over < -0.05 else "The cube's values behave like probabilities at the wells."),
                [f"{EVAL}/tables/calibration_bins.csv"]))
    classes = reader.table(f"{EVAL}/tables/per_class_metrics.csv")
    class_out = [{"task": r.get("task"), "input": r.get("input"), "model": r.get("model"), "class_id": _num(r.get("class_id")),
                  "precision": _r(r.get("precision")), "recall": _r(r.get("recall")), "support": _num(r.get("support"))}
                 for r in classes]
    sand = next((c for c in class_out if c["task"] == "lithology" and (c["input"], c["model"]) == key and c["class_id"] == 1), None)
    shale = next((c for c in class_out if c["task"] == "lithology" and (c["input"], c["model"]) == key and c["class_id"] == 0), None)
    if sand and sand["precision"] is not None:
        base = (sand["support"] / (sand["support"] + shale["support"])) if shale and sand["support"] and shale["support"] else None
        findings.append(_finding(
            "sand_hit_rate", "info",
            f"At the model's own threshold, {_pct(sand['precision'])} of the samples the cube's model calls sand are sand at the "
            f"held-out wells, and it finds {_pct(sand['recall'])} of the true sand"
            + (f" (sand makes up {_pct(base)} of labelled samples)." if base else "."),
            "Roughly one in three 'sand' calls is correct. That is better than picking at random, and still wrong more often than right."
            if sand["precision"] < 0.5 else "Most 'sand' calls are correct at the held-out wells.",
            [f"{EVAL}/tables/per_class_metrics.csv"]))
    intervals = reader.table(f"{EVAL}/tables/conformal_intervals.csv")
    interval_out = [{
        "task": r.get("task"), "input": r.get("input"), "model": r.get("model"),
        "halfwidth_80": _r(r.get("halfwidth_80")), "coverage_80_mean": _r(r.get("lowo_coverage_80_mean")),
        "coverage_80_worst_well": _r(r.get("lowo_coverage_80_min")),
        "halfwidth_90": _r(r.get("halfwidth_90")), "coverage_90_mean": _r(r.get("lowo_coverage_90_mean")),
        "coverage_90_worst_well": _r(r.get("lowo_coverage_90_min")),
        "for_cube_model": (r.get("input"), r.get("model")) == key,
    } for r in intervals]
    for item in interval_out:
        if not item["for_cube_model"] or item["halfwidth_80"] is None:
            continue
        wide = item["halfwidth_80"] >= 0.25
        findings.append(_finding(
            f"interval_{item['task']}", "caution" if wide or (item["coverage_80_worst_well"] or 1) < 0.65 else "info",
            f"For the cube's {TASK_NAMES.get(str(item['task']), item['task'])} output, an 80% uncertainty band is plus or minus "
            f"{item['halfwidth_80']:.2f}. It held for {_pct(item['coverage_80_mean'])} of samples on average and for "
            f"{_pct(item['coverage_80_worst_well'])} in the worst well.",
            ("The band spans most of the possible 0-to-1 range, so a single cube value says very little." if wide else
             "Quote cube values with this band.") +
            " The band is reliable on average but not in every well, so it can be too narrow locally.",
            [f"{EVAL}/tables/conformal_intervals.csv"]))
    return {"calibration": calib_out, "per_class": class_out, "prediction_intervals": interval_out,
            "how_to_read": "Calibration compares predicted probability with how often the label was true. Prediction intervals "
                           "are half-widths: value plus or minus the half-width. Coverage is the share of true values inside the band, "
                           "measured leaving one well out at a time."}


def _by_time_section(reader: _Reader, deployed: dict[str, Any], n_wells: float | None,
                     findings: list[dict[str, Any]]) -> dict[str, Any]:
    table = reader.table(f"{EVAL}/tables/errors_by_interval.csv")
    key = (deployed.get("input"), deployed.get("model"))
    rows = [{
        "task": r.get("task"), "input": r.get("input"), "model": r.get("model"),
        "from_ms": _num(r.get("twt_from_ms")), "to_ms": _num(r.get("twt_to_ms")), "n": _num(r.get("n")),
        "wells": _num(r.get("n_wells")), "roc_auc": _r(r.get("roc_auc")), "error_rate": _r(r.get("error_rate")),
        "prevalence": _r(r.get("prevalence")), "rmse": _r(r.get("rmse"), 4), "bias": _r(r.get("bias"), 4),
    } for r in table]
    rows = [r for r in rows if r["from_ms"] is not None and r["to_ms"] is not None and (r["n"] or 0) > 0]
    window = [min(r["from_ms"] for r in rows), max(r["to_ms"] for r in rows)] if rows else None
    need = max(3.0, (n_wells or 0) / 2)
    solid = [r for r in rows if (r["wells"] or 0) >= need]
    core = [min(r["from_ms"] for r in solid), max(r["to_ms"] for r in solid)] if solid else None
    cube_rows = sorted([r for r in rows if (r["input"], r["model"]) == key], key=lambda r: (str(r["task"]), r["from_ms"]))
    for task in dict.fromkeys(r["task"] for r in cube_rows):
        group = [r for r in cube_rows if r["task"] == task]
        plain = TASK_NAMES.get(str(task), str(task))
        scored = [r for r in group if r["roc_auc"] is not None and (r["n"] or 0) >= 50]
        if scored:
            weight = sum(r["n"] for r in scored)
            mean = sum(r["roc_auc"] * r["n"] for r in scored) / weight
            below = [r for r in scored if r["roc_auc"] < 0.5]
            lo, hi = min(scored, key=lambda r: r["roc_auc"]), max(scored, key=lambda r: r["roc_auc"])
            findings.append(_finding(
                f"time_{task}", "caution" if below or mean < 0.6 else "info",
                f"Inside 200 ms time windows, the cube model's ROC-AUC for {plain} ranges from {lo['roc_auc']:.2f} "
                f"({lo['from_ms']:.0f}-{lo['to_ms']:.0f} ms) to {hi['roc_auc']:.2f} ({hi['from_ms']:.0f}-{hi['to_ms']:.0f} ms), "
                f"averaging {mean:.2f}; it is below 0.5 in {len(below)} of {len(scored)} windows.",
                "Within a narrow depth range the model is unreliable, and in some ranges it ranks sand and shale the wrong way round. "
                "Part of the overall score comes from sand being more common at some depths, which the model can pick up without "
                "recognising individual beds." if below else
                "Skill is fairly even across depth ranges.",
                [f"{EVAL}/tables/errors_by_interval.csv"]))
        biased = [r for r in group if r["bias"] is not None and (r["n"] or 0) >= 50]
        if biased:
            worst = max(biased, key=lambda r: abs(r["bias"]))
            scale = 0.03 if task == "phit" else 0.10
            if abs(worst["bias"]) >= scale:
                findings.append(_finding(
                    f"time_bias_{task}", "caution",
                    f"The cube model's {plain} error drifts with time: the largest average bias is {worst['bias']:+.3f} in "
                    f"{worst['from_ms']:.0f}-{worst['to_ms']:.0f} ms.",
                    "The model reads systematically high or low in some depth ranges, so absolute values there are shifted even "
                    "where the pattern is right.",
                    [f"{EVAL}/tables/errors_by_interval.csv"]))
    return {"label_supported_window_ms": window, "well_supported_window_ms": core,
            "well_supported_rule": f"windows with labelled samples from at least {need:.0f} wells",
            "cube_model_by_window": [{k: v for k, v in r.items() if v is not None and k not in ("input", "model")} for r in cube_rows],
            "all_models_by_window": None,
            "_rows": rows}


def _wells_section(reader: _Reader, deployed: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    table = reader.table(f"{EVAL}/tables/per_well_metrics.csv")
    key = (deployed.get("input"), deployed.get("model"))
    rows = [{
        "well": r.get("well"), "task": r.get("task"), "input": r.get("input"), "model": r.get("model"), "n": _num(r.get("n")),
        "roc_auc": _r(r.get("roc_auc")), "balanced_accuracy": _r(r.get("balanced_acc")), "prevalence": _r(r.get("prevalence")),
        "rmse": _r(r.get("rmse"), 4), "r2": _r(r.get("r2")), "bias": _r(r.get("bias"), 4),
        "fraction_clipped": _r(r.get("fraction_clipped")),
    } for r in table]
    cube_rows = [r for r in rows if (r["input"], r["model"]) == key]
    for task in dict.fromkeys(r["task"] for r in cube_rows):
        group = [r for r in cube_rows if r["task"] == task]
        plain = TASK_NAMES.get(str(task), str(task))
        aucs = [r for r in group if r["roc_auc"] is not None]
        r2s = [r for r in group if r["r2"] is not None]
        if aucs:
            lo, hi = min(aucs, key=lambda r: r["roc_auc"]), max(aucs, key=lambda r: r["roc_auc"])
            findings.append(_finding(
                f"well_spread_{task}", "info",
                f"Per held-out well, the cube model's ROC-AUC for {plain} runs from {lo['roc_auc']:.2f} ({lo['well']}) to "
                f"{hi['roc_auc']:.2f} ({hi['well']}).",
                "Performance depends strongly on the well, so expect the cube to be more trustworthy in some areas than others.",
                [f"{EVAL}/tables/per_well_metrics.csv"]))
        elif r2s:
            negative = [r for r in r2s if r["r2"] <= 0]
            lo, hi = min(r2s, key=lambda r: r["r2"]), max(r2s, key=lambda r: r["r2"])
            bias = max(r2s, key=lambda r: abs(r["bias"] or 0))
            findings.append(_finding(
                f"well_spread_{task}", "caution" if negative else "info",
                f"Per held-out well, the cube model's R2 for {plain} runs from {lo['r2']:.2f} ({lo['well']}) to {hi['r2']:.2f} "
                f"({hi['well']}); it is zero or negative in {len(negative)} of {len(r2s)} wells. Largest average bias: "
                f"{bias['bias']:+.2f} at {bias['well']}.",
                "In the wells with zero or negative R2 the model is worse than guessing that well's average. A well-wide bias means "
                "the whole well reads too high or too low, which a local calibration could correct."
                if negative else "The model keeps some skill in every held-out well.",
                [f"{EVAL}/tables/per_well_metrics.csv"]))
    return {"cube_model_by_well": [{k: v for k, v in r.items() if v is not None and k not in ("input", "model")} for r in cube_rows],
            "_rows": rows}


def _find_number(node: Any, names: set[str], depth: int = 0) -> float | None:
    if depth > 4 or not isinstance(node, dict):
        return None
    for key, value in node.items():
        if str(key).lower() in names and _num(value) is not None:
            return _num(value)
    for value in node.values():
        found = _find_number(value, names, depth + 1)
        if found is not None:
            return found
    return None


def _volume_section(reader: _Reader, records: list[dict[str, Any]], volume: dict[str, Any], rows: list[dict[str, Any]],
                    by_time: dict[str, Any], findings: list[dict[str, Any]]) -> dict[str, Any]:
    record = next((r for r in records if r.get("stage_id") == "09_inference"), {})
    listed = {str(o.get("path")): _num(o.get("size_bytes")) for o in (record.get("outputs") or []) if isinstance(o, dict)}
    model = volume.get("model") if isinstance(volume.get("model"), dict) else {}
    tiling = volume.get("tiling") if isinstance(volume.get("tiling"), dict) else {}
    region = volume.get("region") if isinstance(volume.get("region"), dict) else {}
    perf = volume.get("performance") if isinstance(volume.get("performance"), dict) else {}
    shape = volume.get("volume_shape") or []
    predicted, candidate = _num(volume.get("predicted_voxels")), _num(volume.get("candidate_voxels"))
    live = _num(region.get("live_traces"))
    bins = (shape[0] * shape[1]) if len(shape) == 3 else None

    cubes = []
    outputs = volume.get("outputs") if isinstance(volume.get("outputs"), dict) else {}
    for task, path in outputs.items():
        relative = f"{INFER}/{path}"
        score = next((r for r in rows if r["task"] == task and r["input"] == model.get("input") and r["model"] == model.get("model")), None)
        rating = score["rating"] if score else "unrated"
        cubes.append({
            "property": TASK_NAMES.get(str(task), str(task)), "task": task, "file": relative,
            "file_present": reader.exists(relative), "size_gb": _r((listed.get(str(path)) or 0) / 1e9, 1) or None,
            "made_by": f"{model.get('model')} on {INPUT_NAMES.get(str(model.get('input')), model.get('input'))}",
            "held_out_score": _score_text(score) if score else None, "rating": rating, "advice": CUBE_ADVICE[rating],
        })
        level = {"none": "critical", "weak": "caution", "unrated": "caution", "good": "good", "strong": "good"}.get(rating, "info")
        findings.append(_finding(
            f"cube_model_{task}", level,
            f"The {TASK_NAMES.get(str(task), str(task))} cube was made by {model.get('model')}, whose held-out score is "
            f"{_score_text(score) if score else 'not available'}. Rating: {rating}.",
            CUBE_ADVICE[rating], [f"{INFER}/volume_inference_summary.json", f"{EVAL}/tables/model_performance.csv"]))

    done, total = _num(tiling.get("completed_tiles")), _num(tiling.get("tile_count"))
    parity = volume.get("stage03_parity") if isinstance(volume.get("stage03_parity"), dict) else {}
    if total:
        complete = done == total
        text = f"{int(done or 0)} of {int(total)} tiles were processed"
        if predicted and candidate:
            text += f" and {_pct(predicted / candidate, 1)} of the {candidate / 1e9:.2f} billion candidate voxels received a prediction"
        if parity.get("rows_compared"):
            text += (f". A re-extraction test reproduced {parity['rows_compared']} training cut-outs with a maximum difference of "
                     f"{parity.get('max_abs_difference')}")
        findings.append(_finding(
            "cube_complete", "good" if complete else "critical", text + ".",
            "The cube is complete, and the seismic fed to the model during prediction is identical to what it saw in training."
            if complete else "The cube is incomplete: some tiles were never processed. Do not use it until the run is resumed.",
            [f"{INFER}/volume_inference_summary.json"]))
    missing = [c["file"] for c in cubes if not c["file_present"]]
    if cubes and missing:
        findings.append(_finding(
            "cube_files_missing", "caution",
            f"{len(missing)} of {len(cubes)} cube files named in the summary are not in this job folder.",
            "The summary describes the cubes, but the files themselves could not be found here. They may have been moved or not copied.",
            [f"{INFER}/volume_inference_summary.json"]))

    geometry = reader.json(f"{INFER}/volumes/volume_geometry.json")
    dt = _find_number(geometry, {"dt_ms", "sample_interval_ms", "sample_rate_ms"})
    t0 = _find_number(geometry, {"t0_ms", "t_min_ms", "start_time_ms", "time_start_ms"})
    time_range = [t0, t0 + (shape[2] - 1) * dt] if dt and t0 is not None and len(shape) == 3 else None
    window = by_time.get("well_supported_window_ms") or by_time.get("label_supported_window_ms")
    if window:
        if time_range:
            overlap = max(0.0, min(window[1], time_range[1]) - max(window[0], time_range[0]))
            share = overlap / (time_range[1] - time_range[0]) if time_range[1] > time_range[0] else None
            observed = (f"The cube spans {time_range[0]:.0f} to {time_range[1]:.0f} ms, while well labels from enough wells exist only "
                        f"between {window[0]:.0f} and {window[1]:.0f} ms ({_pct(share)} of the cube's time range).")
        else:
            observed = (f"The cube covers all {shape[2] if len(shape) == 3 else 'n/a'} time samples of the survey, while well labels from "
                        f"enough wells exist only between {window[0]:.0f} and {window[1]:.0f} ms.")
        findings.append(_finding(
            "time_window", "caution", observed,
            "Outside that time window the model was never trained or tested, so the cube there is pure extrapolation. "
            "Restrict interpretation to the supported window.",
            [f"{EVAL}/tables/errors_by_interval.csv", f"{INFER}/volume_inference_summary.json"]))
    flags = [q for q in (record.get("quality_flags") or []) if isinstance(q, dict) and q.get("category") == "xy_geometry"]
    if flags:
        findings.append(_finding(
            "xy_geometry", "caution", _clip(flags[0].get("detail"), 260),
            "The cube is indexed by inline and crossline, which is safe. Converting a position to map X/Y needs the original trace "
            "coordinates; a simple grid formula can be off by a large part of a bin.",
            ["geogg_inference_summary.json"]))
    tasks = [str(t) for t in (volume.get("tasks") or [])]
    if tasks and not ({"sand_fluid", "swt"} & set(tasks)):
        findings.append(_finding(
            "no_fluid_cube", "info",
            f"Cubes were produced for: {', '.join(TASK_NAMES.get(t, t) for t in tasks)}. There is no fluid or water-saturation cube.",
            "Nothing in this run maps hydrocarbons. The cubes describe rock type and porosity only.",
            [f"{INFER}/volume_inference_summary.json"]))
    hours = (_num(perf.get("total_elapsed_seconds")) or 0) / 3600
    return {
        "model": model, "release_id": volume.get("release_id"), "foundation_job_id": volume.get("foundation_job_id"),
        "input_channels": volume.get("channels"), "device": volume.get("device"),
        "cube_shape_inline_crossline_time": shape, "inline_range": region.get("inline_range"),
        "crossline_range": region.get("crossline_range"), "time_range_ms": time_range,
        "live_traces": live, "live_fraction_of_grid": _r(live / bins) if live and bins else None,
        "candidate_voxels": candidate, "predicted_voxels": predicted,
        "predicted_fraction": _r(predicted / candidate, 4) if predicted and candidate else None,
        "tiles": {"completed": done, "total": total, "resumed": tiling.get("resume_used")},
        "training_cut_out_parity": parity or None,
        "cubes": cubes,
        "other_files": [{"file": f"{INFER}/{p}", "size_gb": _r((s or 0) / 1e9, 2), "present": reader.exists(f"{INFER}/{p}")}
                        for p, s in listed.items() if p.startswith("volumes/") and p not in outputs.values()],
        "run_time_hours": _r(hours, 1) if hours else None,
        "voxels_per_second": _r(perf.get("predicted_voxels_per_second"), 0),
        "label_independent": volume.get("label_independent"),
        "storage_note": _clip(volume.get("note"), 300) or None,
    }


def _well_card(wells_rows: list[dict[str, Any]], deployed: dict[str, Any], scorecard: list[dict[str, Any]], well: str) -> dict[str, Any]:
    wanted = well.strip().lower()
    known = sorted({str(r["well"]) for r in wells_rows if r.get("well")})
    mine = [r for r in wells_rows if str(r.get("well", "")).strip().lower() == wanted]
    if not mine:
        return {"well": well, "found": False, "available_wells": known}
    key = (deployed.get("input"), deployed.get("model"))
    fair = {(s["task"], s["input"], s["model"]) for s in scorecard}

    def trim(row: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in row.items() if v is not None and k != "well"}

    return {
        "well": mine[0]["well"], "found": True,
        "note": "Every number is a held-out result: the models that predicted this well never trained on it.",
        "cube_model": [trim(r) for r in mine if (r["input"], r["model"]) == key],
        "best_models": [trim(r) for r in mine if (r["task"], r["input"], r["model"]) in fair],
    }


# --------------------------------------------------------------------------------------------
# Verdict and next steps
# --------------------------------------------------------------------------------------------
def _verdict(status: str, cards: list[dict[str, Any]], cubes: list[dict[str, Any]], problems: list[str]) -> dict[str, Any]:
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
    elif problems:
        level, plain = "needs_attention", "The job reports success, but its outputs are not complete: " + " ".join(problems)
    else:
        level = "usable_with_caveats" if limitations or any(c["rating"] in {"none", "weak", "unrated"} for c in cubes) else "clean"
        plain = (f"The run completed: the models were scored on wells they never trained on, and prediction cubes were written. "
                 f"All {total} recorded self-checks passed, with no blockers.")
        usable = [c["property"] for c in cubes if c["rating"] not in {"none", "unrated"}]
        unusable = [c["property"] for c in cubes if c["rating"] in {"none", "unrated"}]
        if cubes:
            plain += (f" Of the {len(cubes)} cubes, {len(usable)} show some held-out skill"
                      + (f" ({', '.join(usable)})" if usable else "")
                      + (f" and {len(unusable)} should not be interpreted ({', '.join(unusable)})." if unusable else "."))
    return {"level": level, "plain": plain, "checks_passed": total - failed, "checks_total": total, "blockers": blockers,
            "limitations": limitations, "stages_recorded": len(cards), "stages_missing": missing,
            "note": "Technical completion and scientific quality are separate. See the scorecard and cube guide for quality."}


def _next_steps(findings: list[dict[str, Any]], verdict: dict[str, Any]) -> list[str]:
    ids = {f["id"]: f["level"] for f in findings}
    steps: list[str] = []
    if verdict["level"] in {"did_not_finish", "incomplete", "needs_attention"}:
        steps.append("Resolve the failed checks, blockers or missing stages first; the scientific reading is provisional until then.")
    if "no_depth_baseline" in ids:
        steps.append("Add a depth-only (TWT) baseline to the evaluation. It is the one test that shows whether the seismic adds anything beyond burial depth.")
    if any(k.startswith("cube_model_") and v == "critical" for k, v in ids.items()):
        steps.append("Withdraw or clearly label the cubes whose model showed no held-out skill, so nobody interprets them by mistake.")
    if "time_window" in ids:
        steps.append("Crop or mask the cubes to the time window that has well control before sharing them.")
    if ids.get("calibration") == "caution":
        steps.append("Recalibrate the sand output on held-out predictions (for example isotonic or Platt scaling) before calling it a probability.")
    if any(k.startswith("inversion_") for k in ids):
        steps.append("Consider producing the cubes with the best held-out candidate for each target, not only the deep model, where the inversion volumes gave a proven gain.")
    if any(k.startswith("well_spread_") and v == "caution" for k, v in ids.items()):
        steps.append("Check the wells with negative R2 or a large bias for tie or log problems before blaming the model.")
    if "xy_geometry" in ids:
        steps.append("Keep working in inline and crossline; use the stored trace coordinates for any map position.")
    steps.append("Read the Process A analysis of the foundation job for the tie and label caveats, which this run inherits without restating.")
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
        "model_release_id": summary.get("model_release_id"),
    }
    records = [s for s in summary.get("stages", []) if isinstance(s, dict)]
    recognised = operation == EXPECTED_OPERATION or script == EXPECTED_SCRIPT
    if not recognised or not records:
        reason = ("This folder was not produced by Process B (expected operation "
                  f"'{EXPECTED_OPERATION}' or script '{EXPECTED_SCRIPT}')." if not recognised else
                  f"No stage records were found ('{SUMMARY_FILE}' is missing or unreadable).")
        if status.lower() not in {"succeeded", "success", "completed"} and recognised:
            reason += f" The job status is '{status}', so the run may have stopped before writing it."
        return {**base, "applicable": False, "summary": reason, "findings": [reason], "artifacts": reader.used,
                "skipped": reader.skipped,
                "limitations": ["Use the generic processing_output_folder_analysis skill to inventory this folder."]}

    cards = _stage_cards(records)
    evaluation = reader.json(f"{EVAL}/evaluation_summary.json")
    volume = reader.json(f"{INFER}/volume_inference_summary.json")
    deployed = volume.get("model") if isinstance(volume.get("model"), dict) else {}
    rows = _performance(reader)
    n_wells = max((r["wells"] or 0 for r in rows), default=0) or None

    findings: list[dict[str, Any]] = []
    by_time = _by_time_section(reader, deployed, n_wells, findings)
    wells = _wells_section(reader, deployed, findings)
    performance = _performance_section(reader, rows, evaluation, findings)
    weak_tasks = {str(s["task"]): s["rating"] in {"none", "weak"} for s in performance["scorecard"]}
    sections = {
        "performance": performance,
        "comparisons": _comparisons_section(reader, weak_tasks, findings),
        "uncertainty": _uncertainty_section(reader, deployed, findings),
        "by_time": by_time, "wells": wells,
        "volume": _volume_section(reader, records, volume, rows, by_time, findings),
    }
    stage8 = next((c for c in cards if c["stage_id"] == "08_evaluation"), None)
    if stage8 and stage8["checks_total"] == 0 and not stage8["limitations"]:
        findings.append(_finding(
            "no_self_checks", "caution",
            "The evaluation stage recorded no self-checks and no limitations of its own.",
            "A status of COMPLETED here means only that the stage ran. It does not restate the provisional well ties and labels "
            "inherited from Process A, which still apply to every score.",
            ["geogg_inference_summary.json"]))
    all_rows_by_time, all_rows_by_well = by_time.pop("_rows"), wells.pop("_rows")
    by_time.pop("all_models_by_window")
    cubes = sections["volume"]["cubes"]
    problems: list[str] = []
    if not rows:
        problems.append("The held-out performance table is missing or empty, so no model score can be reported.")
        findings.append(_finding(
            "evaluation_missing", "critical", f"'{EVAL}/tables/model_performance.csv' is missing, empty or unreadable.",
            "Without it there is no evidence of how any model performs on unseen wells, and the cubes cannot be rated.",
            [f"{EVAL}/tables/model_performance.csv"]))
    tiles = sections["volume"]["tiles"]
    if tiles["total"] and tiles["completed"] != tiles["total"]:
        problems.append(f"Only {int(tiles['completed'] or 0)} of {int(tiles['total'])} cube tiles were processed.")
    if not cubes:
        problems.append("No prediction cube is described in the inference summary.")
    verdict = _verdict(status, cards, cubes, problems)

    def rank(finding: dict[str, Any]) -> int:
        return next((i for i, prefix in enumerate(PRIORITY) if finding["id"].startswith(prefix)), len(PRIORITY))

    findings.sort(key=rank)
    takeaways: list[dict[str, Any]] = []
    for prefix in TAKEAWAY_IDS:
        for match in [f for f in findings if f["id"].startswith(prefix)]:
            if match["level"] in {"critical", "caution"} and match not in takeaways:
                takeaways.append(match)
    takeaways = sorted(takeaways, key=lambda f: f["level"] != "critical")[:6]

    focus_raw = str(parameters.get("focus") or "").strip().lower().replace(" ", "_").replace("-", "_")
    focus = FOCUS_ALIASES.get(focus_raw) or FOCUS_ALIASES.get(focus_raw.replace("stage_", "").replace("stage", ""), "overview")
    if focus == "all":
        details, shown_cards, shown = dict(sections), cards, findings
    elif focus in sections:
        details = {focus: sections[focus]}
        shown_cards = [c for c in cards if c["stage_id"] == ("09_inference" if focus == "volume" else "08_evaluation")]
        shown = [f for f in findings if f["id"].startswith(_FINDING_FOCUS.get(focus, ()))] or findings
    elif focus in {"glossary", "figures"}:
        details, shown_cards, shown = {}, [], []
    else:
        details = {}
        shown_cards = [{k: c[k] for k in ("stage_id", "title", "what_it_does", "status", "light", "checks_passed", "checks_total",
                                           "failed_checks", "blockers")} | {"n_limitations": len(c["limitations"])} for c in cards]
        shown = findings[:OVERVIEW_FINDINGS]

    response: dict[str, Any] = {
        **base, "applicable": True, "focus": focus,
        "foundation_job_id": volume.get("foundation_job_id"),
        "verdict": verdict, "summary": verdict["plain"],
        "takeaways": takeaways,
        "scorecard": sections["performance"]["scorecard"],
        "cube_guide": cubes,
        "stage_cards": shown_cards,
        "key_findings": shown,
        "findings": [f["observed"] for f in shown],
        "next_steps": _next_steps(findings, verdict),
        "details": details,
        "findings_not_shown": len(findings) - len(shown) if focus == "overview" else 0,
        "more_detail": "Call again with focus = performance, comparisons, uncertainty, by_time, wells, volume or all, or with well = <name>.",
    }
    well = str(parameters.get("well") or "").strip()
    if well:
        response["well_card"] = _well_card(all_rows_by_well, deployed, sections["performance"]["scorecard"], well)
    del all_rows_by_time
    if focus in {"all", "glossary"}:
        response["glossary"] = GLOSSARY
    if focus in {"all", "figures"}:
        guide = [{"file": path, "shows": caption} for path, caption in FIGURES if reader.exists(path)]
        folder = f"{EVAL}/figures/errors_by_interval"
        try:
            per_model = sorted(p.name for p in (reader.job_dir / folder).iterdir() if p.suffix == ".png" and not p.is_symlink())
        except OSError:
            per_model = []
        if per_model:
            guide.append({"folder": folder, "count": len(per_model), "pattern": "<task>_<input>_<model>.png",
                          "shows": "Error against seismic time for one candidate; look for depth ranges where it fails."})
        response["figures_guide"] = guide
    if focus == "overview":
        response["suggested_questions"] = SUGGESTED_QUESTIONS
    response["artifacts"] = [{**item, "role": "evidence read by this analysis"} for item in reader.used]
    response["skipped"] = reader.skipped
    response["limitations"] = [
        "Numbers are read from the run's own tables and summaries; this skill does not recompute any metric or re-run any model.",
        "The prediction cubes are several gigabytes each and are never opened. Statements about them come from the run's summary and "
        "from the held-out scores of the model that produced them.",
        "Held-out scores measure agreement with interpreted-log labels placed with a provisional depth-to-time conversion, "
        "not agreement with ground truth.",
        RATING_NOTE,
        "Plain-language meanings are fixed expert reading rules applied to the observed numbers. They state what a result usually "
        "implies, and are not proof of a geological cause.",
        "Figures are listed with what they show; their pixels are not inspected.",
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
