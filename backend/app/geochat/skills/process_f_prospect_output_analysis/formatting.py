"""Stable Markdown presentation for Process F (prospect definition, stage 14) output-analysis results.

The handler measures; this module only lays the evidence out. Every answer uses the same order
(title, verdict, short reading, headline table, what matters most, detail tables, other findings,
next steps, evidence read) so the same question always looks the same in Ask Geo and in the
Evidence tab. Numbers are printed as the handler returned them and are never recomputed here.

Self-contained on purpose: it imports no other skill.
"""
from __future__ import annotations

import re
from typing import Any

TITLE = "Process F · Prospect definition (stage 14)"
# First line of every rendered answer. ``intent.py`` recognises follow-up questions by it.
RESULT_MARKER = TITLE + " — results"

# What the user can ask next, as (word to use, what it shows).
ASK_ABOUT: list[tuple[str, str]] = [
    ("prospects", "the list and geometry"),
    ("volumes", "rock volume and voxel-based GRV"),
    ("logs", "net-to-gross, porosity and saturation from wells"),
    ("provenance", "checks, sources and storage"),
    ("a prospect ID", "one-prospect card, for example P042"),
    ("glossary", "terms in plain words"),
    ("all", "every table"),
]

# Headline tables shown in the overview, as (result key, heading, columns in display order).
HEADLINE_TABLES: list[tuple[str, str, list[str]]] = []

# One-item cards the handler adds when the user names a well, a body or a prospect.
CARDS: list[tuple[str, str]] = [
    ("prospect_card", "Prospect card"),
]

# Column and row labels that read better than the raw key.
LABELS: dict[str, str] = {
    "self_checks_passed": "Self-checks passed",
    "limitations": "Stated limitations",
    "blockers": "Blockers",
    "stages_recorded": "Stages recorded",
    "stages_missing": "Stages missing",
    "prospects": "Prospects defined",
    "survey_scale_prospects": "Survey-scale prospects (background)",
    "compact_prospects": "Compact prospects",
    "prospects_with_rock_volume": "Prospects with a gated GRV",
    "prospects_with_hcpv": "Prospects with a gated HCPV",
    "typical_net_to_gross_from_logs": "Typical net-to-gross from logs (median)",
    "sand_share_of_all_logged_rock": "Sand share of all logged rock",
    "typical_water_saturation_of_net": "Typical water saturation of net (median)",
    "run_status": "Run status",
    "twt_range_ms": "TWT range (ms)",
    "time_thickness_ms": "Time thickness (ms)",
    "grv_m3": "GRV (gated, m³)",
    "hcpv_m3": "HCPV (gated, m³)",
    "volumes_reported": "Volumes reported",
    "hcpv_status": "Volume reporting status",
    "grv_voxel_est_m3": "Voxel-based GRV estimate (m³)",
    "hcpv_voxel_est_m3": "Voxel-based HCPV estimate (m³)",
    "voxel_based_estimate": "Voxel-based estimate (screening)",
    "grv_est_m3": "GRV estimate (m³)",
    "hcpv_est_m3": "HCPV estimate (m³)",
    "hcpv_est_m3_low_mid_high": "HCPV estimate — low / mid / high (m³)",
    "compact_grv_est_m3_range": "Compact GRV estimate range (m³)",
    "prospects_with_grv_estimate": "Prospects with a GRV estimate",
    "prospects_with_voxel_grv_estimate": "Prospects with a GRV estimate",
    "compact_prospects_with_estimate": "Compact prospects with an estimate",
    "prospects_with_hcpv_estimate": "Prospects with an HCPV estimate",
    "cell_area_m2": "Map-cell area (m²)",
    "voxel_height_m": "Voxel height (m)",
    "sample_interval_ms": "Sample interval (ms)",
    "interval_velocity_m_s": "Interval velocity (m/s)",
    "xy_units_assumed_metres": "XY units assumed metres",
    "sample_interval_assumed": "Sample interval assumed",
    "interval_velocity_assumed": "Interval velocity assumed",
    "inputs_used": "Inputs used",
}

# Coded values that read better in words.
VALUE_LABELS: dict[str, str] = {}

# Column order for detail tables, by the key that holds the table. Unlisted columns follow in file order.
TABLE_COLUMNS: dict[str, list[str]] = {
    "compact_list": ["prospect", "anchor_well", "voxels", "share_of_live_survey", "net_to_gross", "porosity_of_net", "water_saturation_of_net", "grv_voxel_est_m3", "hcpv_voxel_est_m3", "wells_with_net"],
    "largest_survey_scale": ["prospect", "anchor_well", "voxels", "share_of_live_survey", "twt_range_ms", "net_to_gross", "water_saturation_of_net", "reporting_level"],
}

# Columns that only repeat a code already shown in words.
HIDDEN_COLUMNS: set[str] = {"centre", "reasons", "window_ms"}

_LEVEL_ICON = {"critical": "🔴", "caution": "🟡", "good": "🟢", "info": "🔵"}
_LEVEL_WORD = {"critical": "Critical", "caution": "Caution", "good": "Good", "info": "Note"}
_LIGHT_ICON = {"green": "🟢", "amber": "🟡", "red": "🔴", "grey": "⚪"}
_VERDICT = {
    "clean": ("🟢", "Clean"),
    "usable_with_caveats": ("🟡", "Usable with caveats"),
    "needs_attention": ("🔴", "Needs attention"),
    "incomplete": ("🔴", "Incomplete"),
    "did_not_finish": ("🔴", "Did not finish"),
}
_RATING_ICON = {"none": "🔴", "weak": "🟡", "modest": "🟡", "moderate": "🟡", "good": "🟢", "strong": "🟢", "unrated": "⚪"}
_STATUS_ICON = {
    "pass": "🟢", "passed": "🟢", "ok": "🟢", "completed": "🟢", "succeeded": "🟢", "written": "🟢", "frozen": "🟢",
    "warn": "🟡", "warning": "🟡", "conditional": "🟡", "in_progress": "🟡", "not_testable": "⚪", "not_run": "⚪",
    "disabled": "⚪", "skipped": "⚪", "fail": "🔴", "failed": "🔴", "blocked": "🔴", "error": "🔴", "missing": "🔴",
}
_ACRONYMS = {
    "twt": "TWT", "md": "MD", "tvdss": "TVDSS", "segy": "SEG-Y", "hcpv": "HCPV", "grv": "GRV", "ntg": "NTG", "qc": "QC",
    "id": "ID", "ids": "IDs", "ms": "ms", "gb": "GB", "mb": "MB", "m": "m", "m3": "m³", "roc": "ROC", "auc": "AUC",
    "rmse": "RMSE", "mae": "MAE", "r2": "R²", "vsh": "VSH", "phit": "PHIT", "swt": "SWT", "xy": "XY", "api": "API",
    "avo": "AVO", "tz": "T-Z", "utc": "UTC", "sha256": "SHA-256", "rms": "RMS", "ci": "CI", "hc": "HC", "p10": "P10",
    "p50": "P50", "p90": "P90", "p05": "P05", "p25": "P25", "p75": "P75", "p95": "P95", "p01": "P01", "p99": "P99",
}
_FILE_SUFFIX = re.compile(r"\.(csv|json|npy|npz|sgy|segy|md|html|yaml|yml|png|svg|ply|pt|pth|las|txt|py|gz|joblib)$", re.IGNORECASE)
_UNITS = {"ms", "m", "m³", "GB", "MB", "hours", "minutes", "deg", "seconds"}
_LEAD, _TRAIL = "([{\"'“‘", ".,;:!?)]}\"'”’"
_MAX_ROWS = 25
_MAX_COLUMNS = 10
_SKIP_KEYS = {"task_names", "input_names", "rating_convention", "_rows"}


# --------------------------------------------------------------------------------------------
# Text and value formatting
# --------------------------------------------------------------------------------------------
def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _empty(value: Any) -> bool:
    if isinstance(value, (list, tuple)) and value and all(item is None for item in value):
        return True
    return value is None or value == "" or value == [] or value == {}


def _code_token(token: str) -> str:
    """Wrap identifiers, statuses and paths in backticks so underscores are not read as italics."""
    start, end = 0, len(token)
    while start < end and token[start] in _LEAD:
        start += 1
    while end > start and token[end - 1] in _TRAIL:
        end -= 1
    core = token[start:end]
    looks_like_code = bool(core) and (
        "_" in core or _FILE_SUFFIX.search(core) is not None or (core.endswith("/") and len(core) > 1)
    )
    if looks_like_code:
        before, after = token[:start], token[end:]
        if before[-1:] in {"'", '"'} and after[:1] == before[-1:]:
            before, after = before[:-1], after[1:]  # the backticks replace the quotes
        return before + "`" + core + "`" + after
    return token.replace("*", "∗")


def _text(value: Any) -> str:
    """One safe line of Markdown: no pipes, no line breaks, identifiers shown as code."""
    text = " ".join(str(value).replace("|", "/").split())
    pieces = re.split(r"(`[^`]*`)", text)
    for index, piece in enumerate(pieces):
        if piece.startswith("`") and piece.endswith("`") and len(piece) > 1:
            continue
        pieces[index] = " ".join(_code_token(token) for token in piece.replace("`", "'").split(" "))
    return "".join(pieces)


def _label(key: Any) -> str:
    key = str(key)
    if key in LABELS:
        return LABELS[key]
    words = [_ACRONYMS.get(word.lower(), word) for word in key.replace("_", " ").split()]
    if not words:
        return key
    if words[0].islower():
        words[0] = words[0].capitalize()
    if len(words) > 1 and words[-1] in _UNITS:
        words[-1] = f"({words[-1]})"
    return " ".join(words)


def _number(value: float | int, key: str = "") -> str:
    key = key.lower()
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        is_identifier = key.endswith("_id") or key in {"rank", "class_id", "geobody_id", "test_group", "stage", "year"}
        return str(value) if is_identifier else f"{value:,}"
    if value != value or value in (float("inf"), float("-inf")):
        return "—"
    is_share = ("share" in key or "fraction" in key or key.startswith("coverage_")) and -1.0 <= value <= 1.0
    if is_share:
        percent = value * 100
        return (f"{percent:.0f}%" if abs(percent - round(percent)) < 0.05 else f"{percent:.1f}%")
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    if abs(value) >= 1000:
        return f"{value:,.1f}"
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _coded(value: str, key: str = "") -> str:
    """A status, rating or other coded word, with its traffic-light icon when it has one."""
    lowered = value.strip().lower()
    key = key.lower()
    if key == "rating" and lowered in _RATING_ICON:
        return f"{_RATING_ICON[lowered]} {lowered}"
    if key == "light" and lowered in _LIGHT_ICON:
        return _LIGHT_ICON[lowered]
    if (key.endswith("status") or key in {"artifact_audit", "predictions_frozen", "repeated_inference", "truth_reveal"}) and lowered in _STATUS_ICON:
        return f"{_STATUS_ICON[lowered]} {lowered.replace('_', ' ')}"
    return VALUE_LABELS.get(value, "") or _text(value)


def _value(value: Any, key: str = "") -> str:
    """Render one table cell."""
    if _empty(value):
        return "—"
    if isinstance(value, bool) or isinstance(value, (int, float)):
        return _number(value, key)
    if isinstance(value, str):
        return _coded(value, key)
    if isinstance(value, (list, tuple)):
        if all(_is_scalar(item) for item in value):
            numeric = [item for item in value if isinstance(item, (int, float)) and not isinstance(item, bool)]
            is_pair = len(value) == 2 and len(numeric) >= 1 and any(
                word in key.lower() for word in ("range", "window", "interval", "ci", "extent", "start_and_step", "low_middle_high"))
            if is_pair and "start_and_step" in key.lower():
                return f"start {_value(value[0])}, step {_value(value[1])}"
            if is_pair:
                return f"{_value(value[0])} to {_value(value[1])}"
            shown = [_value(item, key) for item in value[:12]]
            more = f" (+{len(value) - 12} more)" if len(value) > 12 else ""
            separator = " × " if "shape" in key.lower() or "extent" in key.lower() else ", "
            return separator.join(shown) + more
        return f"{len(value)} item(s)"
    if isinstance(value, dict):
        parts = [f"{_label(name)}: {_value(item, str(name))}" for name, item in value.items() if not _empty(item)]
        return "; ".join(parts[:8]) or "—"
    return _text(value)


def _is_simple(value: Any) -> bool:
    """Values that fit in one cell of a two-column measure table."""
    if _is_scalar(value):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_scalar(item) for item in value) and len(value) <= 12 and not any(
            isinstance(item, str) and len(item) > 60 for item in value)
    if isinstance(value, dict):
        return bool(value) and len(value) <= 6 and all(_is_scalar(item) for item in value.values())
    return False


# --------------------------------------------------------------------------------------------
# Tables and generic sections
# --------------------------------------------------------------------------------------------
def _table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> list[str]:
    """A Markdown table from a list of records. Empty columns are dropped; long tables are capped."""
    rows = [row for row in rows if isinstance(row, dict)]
    if not rows:
        return []
    present: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in present and not _empty(value) and key not in _SKIP_KEYS:
                present.append(key)
    if columns:
        ordered = [key for key in columns if key in present]
        ordered += [key for key in present if key not in columns and key not in HIDDEN_COLUMNS]
    else:
        ordered = [key for key in present if key not in HIDDEN_COLUMNS]
    if "task" in ordered and ("target" in ordered or "property" in ordered):
        ordered.remove("task")  # the plain name is already shown
    ordered = ordered[:_MAX_COLUMNS]
    if not ordered:
        return []
    lines = ["| " + " | ".join(_label(key) for key in ordered) + " |", "|" + "---|" * len(ordered)]
    for row in rows[:_MAX_ROWS]:
        lines.append("| " + " | ".join(_value(row.get(key), key) for key in ordered) + " |")
    if len(rows) > _MAX_ROWS:
        lines.extend(["", f"Showing the first {_MAX_ROWS} of {len(rows):,} rows."])
    return lines


def _measure_table(items: list[tuple[str, Any]]) -> list[str]:
    items = [(key, value) for key, value in items if not _empty(value) and key not in _SKIP_KEYS]
    if not items:
        return []
    lines = ["| Measure | Value |", "|---|---|"]
    lines.extend(f"| {_label(key)} | {_value(value, key)} |" for key, value in items)
    return lines


def _heading(title: str, depth: int) -> str:
    return "#" * max(2, min(depth, 3)) + " " + title


def _section(title: str, node: Any, depth: int = 2, key: str = "") -> list[str]:
    """Render any part of the handler's ``details`` without knowing its shape in advance."""
    if _empty(node):
        return []
    if isinstance(node, (list, tuple)):
        if all(isinstance(item, dict) for item in node):
            table = _table(list(node), TABLE_COLUMNS.get(key))
            return [_heading(title, depth), "", *table] if table else []
        return [_heading(title, depth), "", *[f"- {_value(item, key) if _is_simple(item) else _text(item)}" for item in list(node)[:20]]]
    if not isinstance(node, dict):
        return [_heading(title, depth), "", _value(node, key)]

    values = list(node.values())
    if len(node) > 1 and all(isinstance(item, dict) for item in values) and all(_is_simple(inner) for item in values for inner in item.values()):
        # A dictionary of records (for example one entry per property): show the name as the first column.
        table = _table([{"name": name, **item} for name, item in node.items()])
        return [_heading(title, depth), "", *table] if table else []

    simple, notes, nested = [], [], []
    for name, item in node.items():
        if _empty(item) or name in _SKIP_KEYS:
            continue
        if isinstance(item, str) and len(item) > 110:
            notes.append((name, item))
        elif _is_simple(item):
            simple.append((name, item))
        else:
            nested.append((name, item))
    lines = [_heading(title, depth), ""]
    lines.extend(_measure_table(simple))
    if notes:
        if simple:
            lines.append("")
        lines.extend(f"- **{_label(name)}:** {_text(item)}" for name, item in notes)
    for name, item in nested:
        block = _section(_label(name), item, depth + 1, name)
        if block:
            lines.extend(["", *block])
    return lines if len(lines) > 2 else []


# --------------------------------------------------------------------------------------------
# Fixed parts of every answer
# --------------------------------------------------------------------------------------------
def _title_block(result: dict[str, Any]) -> list[str]:
    parts = [f"Job `{result.get('job_id') or 'unknown'}`"]
    status = str(result.get("status") or "unknown")
    parts.append(f"status **{_text(status)}**")
    focus = str(result.get("focus") or "")
    if focus:
        parts.append(f"view **{_label(focus).lower()}**")
    return ["# " + RESULT_MARKER, "", " · ".join(parts)]


def _verdict_block(result: dict[str, Any]) -> list[str]:
    verdict = result.get("verdict") if isinstance(result.get("verdict"), dict) else {}
    level = str(verdict.get("level") or "")
    icon, word = _VERDICT.get(level, ("⚪", _label(level) if level else "Not assessed"))
    lines = [f"> {icon} **Verdict: {word}.** {_text(verdict.get('plain') or result.get('summary') or '')}"]
    if verdict.get("note"):
        lines.append(f"> {_text(verdict['note'])}")
    return lines


def _glance_block(result: dict[str, Any]) -> list[str]:
    verdict = result.get("verdict") if isinstance(result.get("verdict"), dict) else {}
    items: list[tuple[str, Any]] = []
    passed, total = verdict.get("checks_passed"), verdict.get("checks_total")
    if total not in (None, ""):
        items.append(("self_checks_passed", f"{passed} of {total}"))
    items.extend((key, value) for key, value in verdict.items()
                 if key not in {"level", "plain", "note", "checks_passed", "checks_total"} and _is_simple(value))
    glance = result.get("at_a_glance") if isinstance(result.get("at_a_glance"), dict) else {}
    items.extend((key, value) for key, value in glance.items() if _is_simple(value))
    table = _measure_table(items)
    return ["## At a glance", "", *table] if table else []


def _stage_block(cards: list[dict[str, Any]]) -> list[str]:
    cards = [card for card in cards if isinstance(card, dict)]
    if not cards:
        return []
    lines = ["## Stage by stage", "", "| Stage | What it does | Status | Self-checks | Limitations | Warnings |", "|---|---|---|---|---|---|"]
    extra: list[str] = []
    for card in cards:
        limitations = card.get("n_limitations", len(card.get("limitations") or []))
        warnings = card.get("n_warnings", len(card.get("warnings") or []))
        light = _LIGHT_ICON.get(str(card.get("light") or "").lower(), "⚪")
        stage = _text(card.get("stage_id") or "")
        lines.append(
            f"| {stage} | {_text(card.get('title') or '')} | {light} {_text(card.get('status') or 'unknown')} | "
            f"{card.get('checks_passed', '—')} of {card.get('checks_total', '—')} | {limitations} | {warnings} |")
        for failed in card.get("failed_checks") or []:
            if isinstance(failed, dict):
                extra.append(f"- 🔴 {stage} failed check {_text(failed.get('check') or '')}: {_text(failed.get('detail') or 'no detail recorded')}")
        extra.extend(f"- 🔴 {stage} blocker: {_text(item)}" for item in card.get("blockers") or [])
        extra.extend(f"- 🟡 {stage} limitation: {_text(item)}" for item in card.get("limitations") or [])
        extra.extend(f"- 🟡 {stage} warning: {_text(item)}" for item in card.get("warnings") or [])
    if extra:
        lines.extend(["", *extra[:30]])
    return lines


def _headline_blocks(result: dict[str, Any]) -> list[list[str]]:
    blocks: list[list[str]] = []
    for key, heading, columns in HEADLINE_TABLES:
        rows = result.get(key)
        if not isinstance(rows, list) or not rows:
            continue
        if key == "stage_cards":
            blocks.append(_stage_block(rows))
            continue
        table = _table(rows, columns)
        if table:
            blocks.append([f"## {heading}", "", *table])
    return blocks


def _source(finding: dict[str, Any]) -> str:
    files = [str(item) for item in (finding.get("evidence") or []) if item][:2]
    return "Source: " + ", ".join(f"`{name}`" for name in files) if files else ""


def _takeaway_block(findings: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for finding in findings:
        level = str(finding.get("level") or "info")
        lines.append(f"{_LEVEL_ICON.get(level, '🔵')} **{_LEVEL_WORD.get(level, 'Note')} — {_text(finding.get('observed') or '')}**")
        if finding.get("plain_meaning"):
            lines.append(f"What it usually means: {_text(finding['plain_meaning'])}")
        if _source(finding):
            lines.append(_source(finding))
        lines.append("")
    return ["## What matters most", "", *lines[:-1]] if lines else []


def _finding_list(title: str, findings: list[dict[str, Any]]) -> list[str]:
    lines = []
    for finding in findings:
        level = str(finding.get("level") or "info")
        line = f"- {_LEVEL_ICON.get(level, '🔵')} {_text(finding.get('observed') or '')}"
        if finding.get("plain_meaning"):
            line += f" **Meaning:** {_text(finding['plain_meaning'])}"
        if _source(finding):
            line += f" {_source(finding)}"
        lines.append(line)
    return [f"## {title}", "", *lines] if lines else []


def _card_blocks(result: dict[str, Any]) -> list[list[str]]:
    blocks = []
    for key, heading in CARDS:
        card = result.get(key)
        if not isinstance(card, dict) or not card:
            continue
        name = next((card[field] for field in ("well", "prospect", "geobody_id") if card.get(field) not in (None, "")), "")
        title = f"{heading}: {_text(name)}" if name != "" else heading
        if card.get("found") is False:
            options = card.get("available_wells") or card.get("available_ids") or card.get("example_ids") or []
            lines = [f"## {title}", "", f"🔴 {_text(card.get('note') or 'It was not found in this run.')}"]
            if options:
                lines.extend(["", "Available: " + ", ".join(_text(item) for item in list(options)[:30])])
            blocks.append(lines)
            continue
        block = _section(title, {name_: value for name_, value in card.items() if name_ != "found"}, 2, key)
        if block:
            blocks.append(block)
    return blocks


def _glossary_block(glossary: Any) -> list[str]:
    if not isinstance(glossary, dict) or not glossary:
        return []
    lines = ["## Glossary", "", "| Term | Meaning |", "|---|---|"]
    lines.extend(f"| {_text(term)} | {_text(meaning)} |" for term, meaning in glossary.items())
    return lines


def _footer_block(result: dict[str, Any]) -> list[str]:
    lines = ["---"]
    artifacts = [item for item in (result.get("artifacts") or []) if isinstance(item, dict)]
    skipped = result.get("skipped") or []
    if artifacts or skipped:
        names = ", ".join(f"`{item.get('name')}`" for item in artifacts[:6])
        more = f" and {len(artifacts) - 6} more" if len(artifacts) > 6 else ""
        line = f"**Evidence read:** {len(artifacts)} file(s) in the job folder"
        line += f" ({names}{more})." if names else "."
        if skipped:
            line += f" {len(skipped)} file(s) could not be read."
        lines.append(line)
    sources = [(key, value) for key, value in result.items()
               if (key.startswith("source_") or key == "foundation_job_id" or key == "model_release_id") and value]
    if sources:
        lines.extend(["", "**Built on:** " + "; ".join(f"{_label(key)} `{value}`" for key, value in sources)])
    not_shown = result.get("findings_not_shown") or 0
    asks = ", ".join(f"**{word}** ({what})" for word, what in ASK_ABOUT)
    hint = f"**Ask for more:** {asks}."
    if not_shown:
        hint = f"{not_shown} more finding(s) are held back in this overview. " + hint
    lines.extend(["", hint])
    limits = [str(item) for item in (result.get("limitations") or []) if item][:3]
    volume_note = ("Volumes labelled 'voxel-based GRV' are computed as voxel count × voxel size "
                   "(GRV × NTG × PHIT × (1 − SWT) screening scenario). "
                   "They are not a resource estimate and carry no fluid-contact, closure or FVF correction.")
    lines.extend(["", "**Keep in mind:**", f"- {volume_note}"])
    if limits:
        lines.extend([*[f"- {_text(item)}" for item in limits]])
    return lines


def _not_applicable(result: dict[str, Any]) -> str:
    lines = ["# " + RESULT_MARKER, "", f"Job `{result.get('job_id') or 'unknown'}` · status **{_text(result.get('status') or 'unknown')}**", "",
             f"> ⚪ **Not analysed.** {_text(result.get('summary') or 'This job folder could not be read as a run of this process.')}"]
    limits = [str(item) for item in (result.get("limitations") or []) if item]
    if limits:
        lines.extend(["", *[f"- {_text(item)}" for item in limits]])
    return "\n".join(lines)


def render_markdown(result: dict[str, Any]) -> str:
    """Render one output-analysis result in the fixed layout described in SKILL.md.

    ``result["narrative"]`` is optional: the short plain-language reading written by the language
    model from this same evidence. Everything else comes straight from the handler.
    """
    if not isinstance(result, dict) or not result.get("applicable", False):
        return _not_applicable(result if isinstance(result, dict) else {})

    focus = str(result.get("focus") or "overview")
    overview = focus in {"overview", "all"}
    takeaways = [item for item in (result.get("takeaways") or []) if isinstance(item, dict)]
    findings = [item for item in (result.get("key_findings") or []) if isinstance(item, dict)]
    details = result.get("details") if isinstance(result.get("details"), dict) else {}

    blocks: list[list[str]] = [_title_block(result), _verdict_block(result)]
    narrative = str(result.get("narrative") or "").strip()
    if narrative:
        blocks.append(["**In plain words**", "", narrative])
    blocks.extend(_card_blocks(result))
    if overview:
        blocks.append(_glance_block(result))
        blocks.extend(_headline_blocks(result))
        blocks.append(_takeaway_block(takeaways))
    elif focus not in {"glossary", "figures"}:
        # A focused view still shows the stage cards the handler narrowed down for it.
        blocks.append(_stage_block(result.get("stage_cards") or []))
    for name, node in details.items():
        blocks.append(_section(f"Details: {_label(name).lower()}", node, 2, name))
    if overview:
        shown = {item.get("id") for item in takeaways}
        blocks.append(_finding_list("Other findings", [item for item in findings if item.get("id") not in shown]))
    else:
        blocks.append(_finding_list("Findings for this view", findings))
    steps = [str(item) for item in (result.get("next_steps") or []) if item][:5]
    if steps:
        blocks.append(["## Suggested next steps", "", *[f"{index}. {_text(step)}" for index, step in enumerate(steps, 1)]])
    blocks.append(_glossary_block(result.get("glossary")))
    figures = result.get("figures_guide")
    if figures:
        blocks.append(_section("Figures in this run", figures, 2, "figures_guide"))
    blocks.append(_footer_block(result))
    return "\n\n".join("\n".join(block) for block in blocks if block)
