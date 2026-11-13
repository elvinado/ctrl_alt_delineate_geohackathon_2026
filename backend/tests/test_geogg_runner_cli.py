"""CLI and APGCE processing-job integration tests for GeoGG foundations."""
from __future__ import annotations

import os
from pathlib import Path
import unittest
from unittest.mock import patch

from scripts.geogg_fresh import runner


class TestGeoGGRunnableManifestArgument(unittest.TestCase):
    def test_runner_uses_shipped_default_config_and_processing_uid(self):
        job_dir = Path("/tmp/apgce-processing-uid-123")
        manifest = job_dir / "inputs.json"
        with patch.dict(os.environ, {
            "APGCE_RESULT_PATH": str(job_dir / "result.json"),
            "APGCE_JOB_ID": job_dir.name,
        }, clear=True), \
             patch.object(runner, "load_yaml_config", return_value={}) as load_config, \
             patch.object(runner, "StageContext") as stage_context, \
             patch.object(runner, "STAGE_MODULES", {}), \
             patch.object(runner, "_write_foundations_summary"):
            stage_context.from_environment.return_value = None
            result = runner.run_foundations(["--input-manifest", str(manifest)])
            self.assertEqual(os.environ["APGCE_INPUT_MANIFEST"], str(manifest.resolve()))
            load_config.assert_called_once()
            self.assertTrue(Path(load_config.call_args.args[0]).is_file())
            self.assertEqual(load_config.call_args.kwargs["output_root"], job_dir.resolve())

        self.assertEqual(result, 0)

    def test_runner_rejects_run_id_that_does_not_match_processing_uid(self):
        job_dir = Path("/tmp/apgce-processing-uid-123")
        with patch.dict(os.environ, {
            "APGCE_RESULT_PATH": str(job_dir / "result.json"),
            "APGCE_JOB_ID": job_dir.name,
            "APGCE_GEOGG_RUN_ID": "some-other-run",
        }, clear=True):
            with self.assertRaisesRegex(RuntimeError, "must match the APGCE processing job ID"):
                runner.run_foundations([])

    def test_runner_rejects_processing_uid_that_does_not_match_job_directory(self):
        job_dir = Path("/tmp/apgce-processing-uid-123")
        with patch.dict(os.environ, {
            "APGCE_RESULT_PATH": str(job_dir / "result.json"),
            "APGCE_JOB_ID": "different-processing-uid",
        }, clear=True):
            with self.assertRaisesRegex(RuntimeError, "does not match the processing job directory ID"):
                runner.run_foundations([])


if __name__ == "__main__":
    unittest.main()
