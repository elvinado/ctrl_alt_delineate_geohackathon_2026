"""Run GeoGG stages 00–03 in one APGCE processing job."""
import os
from pathlib import Path

from scripts_utils import validate_outputs
from geogg_fresh.runner import run_foundations


FOUNDATIONS_SUMMARY = "geogg_foundations_summary.json"


@validate_outputs(expected_outputs=[FOUNDATIONS_SUMMARY])
def main():
    result_path = Path(os.environ["APGCE_RESULT_PATH"]).expanduser().resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)

    exit_code = run_foundations()
    if exit_code != 0:
        summary_path = result_path.parent / FOUNDATIONS_SUMMARY
        detail = ""
        try:
            summary = __import__("json").loads(summary_path.read_text(encoding="utf-8"))
            blocked = [
                f"{stage.get('stage_id')}: {', '.join(stage.get('blockers') or []) or stage.get('summary', '')}"
                for stage in summary.get("stages", [])
                if stage.get("status") in {"BLOCKED", "FAILED"}
            ]
            detail = " ".join(part for part in blocked if part)
        except (OSError, ValueError, TypeError):
            pass
        message = f"GeoGG stages 00–03 failed with exit code {exit_code}."
        if detail:
            message += f" {detail}"
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
