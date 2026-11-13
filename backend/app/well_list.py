"""Excel well-list readers shared by ingestion and dataset viewers."""
from pathlib import Path
from typing import Any

MAX_TABLE_ROWS = 250_000

def read_excel_well_list(path: Path) -> dict[str, Any]:
    """Read the five well-list fields used by the Project well import."""
    try:
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=True)
        sheet = workbook.active
        expected = {"well name": "Well Name", "rt": "RT", "td": "TD", "x": "X", "y": "Y"}
        header_row = None
        indexes: dict[str, int] = {}
        for row_number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
            names = {str(value).strip().casefold(): index for index, value in enumerate(row) if value is not None}
            if all(name in names for name in expected):
                header_row = row_number
                indexes = {label: names[label] for label in expected}
                break
            if row_number >= 50:
                break
        if header_row is None:
            raise ValueError("Excel workbook must contain Well Name, RT, TD, X and Y columns")
        headers = list(expected.values())
        rows = []
        for row in sheet.iter_rows(min_row=header_row + 1, values_only=True):
            values = [row[index] if index < len(row) else None for index in indexes.values()]
            if values[0] is None or not str(values[0]).strip():
                continue
            rows.append([str(value) if value is not None else "" for value in values])
            if len(rows) > MAX_TABLE_ROWS:
                raise ValueError(f"Table contains more than {MAX_TABLE_ROWS} rows")
        workbook.close()
        return {"type": "table", "headers": headers, "rows": rows,
                "rowCount": len(rows), "delimiter": "excel"}
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to read Excel well list: {type(exc).__name__}: {exc}") from exc


