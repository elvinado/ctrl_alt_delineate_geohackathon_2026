from pathlib import Path

from app.well_tables import read_well_table


def test_two_column_tvdss_twt_comment_header(tmp_path: Path):
    path = tmp_path / "checkshot.txt"
    path.write_text(
        "#m-1\n#Checkshot\n#TVDSS (m)\n#TWT (ms)\n0 0\n270.57 336.2\n",
        encoding="utf-8",
    )

    result = read_well_table(path, "checkshot")

    assert [column["name"] for column in result["columns"]] == ["TVDSS", "TWT"]
    assert [column["unit"] for column in result["columns"]] == ["m", "ms"]
    assert result["row_count"] == 2
    assert result["rows"][1] == {"TVDSS": 270.57, "TWT": 336.2}
    assert "MDKB" not in result["rows"][1]


def test_legacy_three_column_checkshot_is_preserved(tmp_path: Path):
    path = tmp_path / "legacy.txt"
    path.write_text("#MDKB TWT TVDSS\n100 200 90\n", encoding="utf-8")

    result = read_well_table(path, "checkshot")

    assert [column["name"] for column in result["columns"]] == ["MDKB", "TWT", "TVDSS"]
    assert result["rows"] == [{"MDKB": 100.0, "TWT": 200.0, "TVDSS": 90.0}]
