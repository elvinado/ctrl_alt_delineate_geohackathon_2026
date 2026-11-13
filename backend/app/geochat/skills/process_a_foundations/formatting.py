"""Stable Markdown presentation for the Process A (foundations, stages 00-07) pre-flight check.

The handler validates and plans; this module only lays its answer out, always in the same order:
readiness, issues with their fixes, what was checked, the inputs the process needs, what the run
will do, what comes out, and the next step. Nothing is submitted from here.

Self-contained on purpose: it imports no other skill.
"""
from __future__ import annotations

import re
from typing import Any

TITLE = "Process A · Foundations (stages 00–07)"
# First line of every rendered pre-flight answer.
PREFLIGHT_MARKER = TITLE + " — pre-flight check"

# Column and row labels that read better than the raw key.
LABELS: dict[str, str] = {
    "wells_with_logs": "Wells with logs",
    "wells_with_deviation": "Wells with a deviation survey",
    "wells_with_time_depth": "Wells with a time-depth table",
    "seismic_roles": "Seismic volumes by role",
    "well_list_tables": "Well list tables",
}

# Coded values that read better in words.
VALUE_LABELS: dict[str, str] = {}

# Column order for detail tables, by the key that holds the table.
TABLE_COLUMNS: dict[str, list[str]] = {}

# Columns that only repeat a code already shown in words.
HIDDEN_COLUMNS: set[str] = {"code"}

_READINESS = {
    "ready_for_confirmation": ("🟢", "Ready to run"),
    "not_ready": ("🔴", "Not ready"),
    "needs_inputs": ("⚪", "Inputs needed"),
}
_ISSUE_ICON = {"error": "🔴 Blocks the run", "warning": "🟡 Warning"}
_LIGHT_ICON = {"green": "🟢", "amber": "🟡", "red": "🔴", "grey": "⚪"}
_RATING_ICON = {"none": "🔴", "weak": "🟡", "modest": "🟡", "moderate": "🟡", "good": "🟢", "strong": "🟢", "unrated": "⚪"}
_STATUS_ICON = {
    "pass": "🟢", "passed": "🟢", "ok": "🟢", "completed": "🟢", "succeeded": "🟢", "ready": "🟢",
    "warn": "🟡", "warning": "🟡", "conditional": "🟡", "fail": "🔴", "failed": "🔴", "blocked": "🔴", "missing": "🔴",
}
_ACRONYMS = {
    "twt": "TWT", "md": "MD", "segy": "SEG-Y", "hcpv": "HCPV", "grv": "GRV", "qc": "QC", "id": "ID", "ids": "IDs",
    "ms": "ms", "gb": "GB", "xy": "XY", "tz": "T-Z", "vsh": "VSH", "phit": "PHIT", "swt": "SWT", "gpu": "GPU",
    "p01": "P01", "p05": "P05", "p25": "P25", "p50": "P50", "p75": "P75", "p95": "P95", "p99": "P99",
}
# Keys that are plumbing for the agent or already shown elsewhere in the answer.
_PLUMBING = {
    "skill", "project_id", "python_script", "operation", "target_well", "submitted", "confirmation_required",
    "requirements", "what_will_happen", "expected_outputs", "reference_run", "issues", "status", "summary",
    "next_action", "dataset_ids",
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
# Fixed parts of every pre-flight answer
# --------------------------------------------------------------------------------------------
def _issues_block(issues: list[dict[str, Any]]) -> list[str]:
    issues = [item for item in issues if isinstance(item, dict)]
    if not issues:
        return []
    lines = ["## Issues to resolve", "", "| Level | What was found | How to fix it |", "|---|---|---|"]
    for item in sorted(issues, key=lambda issue: 0 if issue.get("level") == "error" else 1):
        level = _ISSUE_ICON.get(str(item.get("level") or ""), _text(item.get("level") or "Note"))
        lines.append(f"| {level} | {_text(item.get('message') or '')} | {_text(item.get('how_to_fix') or '—')} |")
    return lines


def _requirements_block(requirements: list[dict[str, Any]]) -> list[str]:
    rows = [item for item in requirements if isinstance(item, dict)]
    if not rows:
        return []
    lines = ["## Inputs this process needs", "", "| Input | Need | Why it matters |", "|---|---|---|"]
    lines.extend(f"| {_text(row.get('input') or '')} | {_text(row.get('need') or '')} | {_text(row.get('why') or '')} |" for row in rows)
    return lines


def render_markdown(result: dict[str, Any]) -> str:
    """Render one pre-flight result in the fixed layout described in SKILL.md."""
    result = result if isinstance(result, dict) else {}
    status = str(result.get("status") or "")
    icon, word = _READINESS.get(status, ("⚪", _label(status) if status else "Checked"))
    script = str(result.get("python_script") or "")
    blocks: list[list[str]] = [
        ["# " + PREFLIGHT_MARKER],
        [f"> {icon} **{word}.** {_text(result.get('summary') or '')}"],
        _issues_block(result.get("issues") or []),
    ]
    checked_scalars = [(key, value) for key, value in result.items() if key not in _PLUMBING and _is_simple(value)]
    checked = _measure_table(checked_scalars)
    if checked:
        blocks.append(["## What was checked", "", *checked])
    for key, value in result.items():
        if key not in _PLUMBING and not _is_simple(value):
            blocks.append(_section(_label(key), value, 2, key))
    blocks.append(_requirements_block(result.get("requirements") or []))
    steps = [str(item) for item in (result.get("what_will_happen") or []) if item]
    if steps:
        blocks.append(["## What the run will do", "", *[f"- {_text(step)}" for step in steps]])
    outputs = [str(item) for item in (result.get("expected_outputs") or []) if item]
    if outputs:
        blocks.append(["## What you will get", "", *[f"- {_text(item)}" for item in outputs]])
    reference = result.get("reference_run")
    if isinstance(reference, dict) and reference:
        blocks.append(_section("Cost of the reference run", reference, 2, "reference_run"))
    if status == "ready_for_confirmation":
        next_step = (f"**Next step:** review the plan above, then submit `{script}` from the Processing tab. "
                     "This check did not submit anything.")
    elif status == "not_ready":
        next_step = "**Next step:** resolve each issue marked as blocking, then ask for this check again. Nothing was submitted."
    else:
        next_step = (f"**Next step:** tell me which inputs to check, or open `{script}` in the Processing tab. "
                     "Nothing was submitted.")
    blocks.append(["---", next_step])
    return "\n\n".join("\n".join(block) for block in blocks if block)
