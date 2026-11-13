"""Backend worker entry point for GeoGG Stages 10–11."""
import json
import os
from pathlib import Path

from scripts_utils import validate_outputs
from geogg_fresh.stages.s10_summary import run_volume_products


VOLUME_PRODUCTS_SUMMARY = "geogg_volume_products_summary.json"


@validate_outputs(expected_outputs=[VOLUME_PRODUCTS_SUMMARY])
def main():
    result_path = Path(os.environ["APGCE_RESULT_PATH"]).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    config = os.environ.get("APGCE_GEOGG_VOLUME_PRODUCTS_CONFIG")
    if not config:
        config = str(Path(__file__).resolve().parent / "C_volume_products.yaml")

    exit_code = run_volume_products(["--config", config])
    if exit_code != 0:
        summary_path = result_path.parent / VOLUME_PRODUCTS_SUMMARY
        detail = ""
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            detail = str(summary.get("summary") or "")
        except (OSError, ValueError, TypeError):
            pass
        message = f"GeoGG volume-products processing failed with exit code {exit_code}."
        if detail:
            message += f" {detail}"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
