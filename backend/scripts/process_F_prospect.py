"""Backend worker entry point for GeoGG Stage 14 prospect screening."""
from scripts_utils import validate_outputs
from geogg_fresh.stages.s14_prospect import main as _run_stage14

PREFLIGHT = "geogg_prospect_preflight.json"
SUMMARY = "geogg_prospect_summary.json"


@validate_outputs(expected_outputs=[PREFLIGHT, SUMMARY])
def main() -> None:
    _run_stage14()


if __name__ == "__main__":
    main()
