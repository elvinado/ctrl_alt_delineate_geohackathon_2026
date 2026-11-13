import json
import os
from functools import wraps
from pathlib import Path


def validate_outputs(expected_outputs: list[str]):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result_path = Path(os.environ["APGCE_RESULT_PATH"]).resolve()
            job_dir = result_path.parent

            try:
                result = func(*args, **kwargs)

                output_files = [
                    (job_dir / filename).resolve()
                    for filename in expected_outputs
                ]

                missing_files = [
                    str(path)
                    for path in output_files
                    if not path.is_file()
                ]

                if missing_files:
                    raise FileNotFoundError(
                        f"Expected output files not found: {missing_files}"
                    )

                result_path.write_text(
                    json.dumps(
                        {
                            "status": "succeeded",
                            "outputs": [str(path) for path in output_files],
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                return result

            except Exception as exc:
                result_path.write_text(
                    json.dumps(
                        {
                            "status": "failed",
                            "error_message": str(exc),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                raise

        return wrapper

    return decorator
