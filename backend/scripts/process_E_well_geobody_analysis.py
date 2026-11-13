"""Backend worker entry point for GeoGG Stage 13 well-to-geobody analysis."""
from scripts_utils import validate_outputs
from geogg_fresh.stages.s13_well_geobody_analysis import main as _run_stage13


PREFLIGHT = "geogg_well_geobody_analysis_preflight.json"
SUMMARY = "geogg_well_geobody_analysis_summary.json"


@validate_outputs(expected_outputs=[PREFLIGHT, SUMMARY])
def main() -> None:
    _run_stage13()


if __name__ == "__main__":
    main()
