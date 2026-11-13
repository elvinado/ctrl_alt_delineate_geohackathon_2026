"""Parser for tabular well deviation and checkshot text files."""
from __future__ import annotations

from pathlib import Path
import math
import re
from typing import Any

EXPECTED_COLUMNS = {
    "deviation": ["MDKB", "AZIM", "INCL"],
    "checkshot": ["MDKB", "TWT", "TVDSS"],
}
_CHECKSHOT_ALIASES = {
    "MDKB": "MDKB", "MD": "MDKB", "MDRT": "MDKB",
    "TWT": "TWT", "TWTT": "TWT", "TIME": "TWT",
    "TVDSS": "TVDSS", "TVD_SS": "TVDSS",
}


def read_well_table(path: Path, kind: str, offset: int = 0, limit: int = 1000) -> dict[str, Any]:
    """Parse comment-header or legacy numeric well tables into a paginated table."""
    if kind not in EXPECTED_COLUMNS:
        raise ValueError(f"Unsupported well table kind: {kind}")
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ValueError(f"Unable to read text file: {exc}") from exc

    headers: list[str] = []
    rows: list[list[float]] = []
    header_units: dict[str, str] = {}
    header_re = re.compile(r"^\s*#\s*(.*?)\s*$")

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("#"):
            candidate = header_re.match(line).group(1).strip()  # type: ignore[union-attr]
            if kind == "checkshot":
                # This format uses one comment per field, e.g. #TVDSS (m), #TWT (ms).
                match = re.match(r"^([A-Za-z_]+)(?:\s*\(([^)]+)\))?$", candidate)
                if match:
                    alias = _CHECKSHOT_ALIASES.get(match.group(1).upper())
                    if alias:
                        headers.append(alias)
                        if match.group(2):
                            header_units[alias] = match.group(2).strip()
                # Also accept a single comment containing all column names.
                tokens = candidate.split()
                aliases = [_CHECKSHOT_ALIASES.get(token.upper()) for token in tokens]
                if len(tokens) in {2, 3} and all(aliases):
                    headers = [str(value) for value in aliases]
            else:
                tokens = candidate.split()
                expected = {name.casefold() for name in EXPECTED_COLUMNS[kind]}
                if len(tokens) == len(EXPECTED_COLUMNS[kind]) and all(
                    token.casefold() in expected for token in tokens
                ):
                    headers = tokens
            continue

        fields = line.split()
        try:
            values = [float(value) for value in fields]
        except ValueError:
            continue
        allowed_counts = {2, 3} if kind == "checkshot" else {3}
        if len(values) not in allowed_counts:
            raise ValueError(
                f"Line {line_number} has {len(values)} numeric columns; expected "
                + ("2 or 3" if kind == "checkshot" else "3")
            )
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"Line {line_number} contains a non-finite numeric value")
        if rows and len(values) != len(rows[0]):
            raise ValueError(f"Line {line_number} has a different number of columns from earlier rows")
        rows.append(values)

    if not rows:
        raise ValueError(f"No numeric {kind} rows found in text file")

    if kind == "checkshot" and len(rows[0]) == 2:
        # Two-column checkshots are explicitly TVDSS, TWT; MDKB is not present.
        columns = ["TVDSS", "TWT"]
        source_columns = headers if len(headers) == 2 else columns
    else:
        columns = EXPECTED_COLUMNS[kind]
        source_columns = headers if len(headers) == len(columns) else columns

    page = rows[offset : offset + limit]
    return {
        "kind": kind,
        "columns": [
            {
                "name": name,
                "source_name": source_columns[i],
                "unit": header_units.get(source_columns[i]) or _unit_for(kind, name, text),
            }
            for i, name in enumerate(columns)
        ],
        "row_count": len(rows),
        "offset": offset,
        "limit": limit,
        "returned_rows": len(page),
        "rows": [dict(zip(columns, row)) for row in page],
    }


def _unit_for(kind: str, column: str, text: str) -> str | None:
    if kind == "deviation" and column == "MDKB" and re.search(r"MDKB\s*\(m\)", text, re.IGNORECASE):
        return "m"
    return None
