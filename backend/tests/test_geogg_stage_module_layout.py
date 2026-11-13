from pathlib import Path


def test_stages_10_to_12_have_stage_aligned_modules():
    backend = Path(__file__).resolve().parents[1]
    stages = backend / "scripts" / "geogg_fresh" / "stages"
    for name in ("s10_summary.py", "s11_volume_export.py", "s12_geobody.py"):
        assert (stages / name).is_file()


def test_stage_workers_are_thin_adapters():
    backend = Path(__file__).resolve().parents[1]
    geobody_worker = (backend / "scripts" / "process_D_geobody.py").read_text()
    volume_worker = (backend / "scripts" / "process_C_volume_products.py").read_text()
    assert "geogg_fresh.stages.s12_geobody" in geobody_worker
    assert "geogg_fresh.stages.s10_summary" in volume_worker
    assert "def _extract" not in geobody_worker
    assert "def _write_native_volume_export_stage" not in volume_worker
