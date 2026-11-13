"""Backend worker entry point for GeoGG Stage 12 geobody extraction."""
from scripts_utils import validate_outputs
from geogg_fresh.stages.s12_geobody import (
    OUTPUT, SUMMARY, _catalogue, _check, _extract, _export_segy_products,
    _load_yaml, _safe_product_path, _sha256, _write_mesh,
    _write_visualisations, _load_trace_xy, _write_geobody_run_length_index,
    _write_geobody_voxel_xyz, main as _run_geobody,
)


@validate_outputs(expected_outputs=[OUTPUT, SUMMARY])
def main() -> None:
    """Run Stage 12 and validate its required job-level outputs."""
    _run_geobody()


if __name__ == "__main__":
    main()
