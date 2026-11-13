from pathlib import Path

from openpyxl import Workbook

from app.well_list import read_excel_well_list


def test_read_excel_well_list_finds_header_after_preamble(tmp_path: Path):
    path = tmp_path / "wells.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["Project export"])
    sheet.append([])
    sheet.append(["No.", "Well Name", "RT", "WD", "WB", "TD", "Well Types", "X", "Y"])
    sheet.append([])
    sheet.append([1, "EO-1", 21.95, None, None, 3200.4, "Marine", 1374930.5, 1796501.6])
    workbook.save(path)

    result = read_excel_well_list(path)

    assert result["type"] == "table"
    assert result["headers"] == ["Well Name", "RT", "TD", "X", "Y"]
    assert result["rows"] == [["EO-1", "21.95", "3200.4", "1374930.5", "1796501.6"]]
    assert result["rowCount"] == 1
