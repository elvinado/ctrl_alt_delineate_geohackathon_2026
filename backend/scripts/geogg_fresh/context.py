"""Execution context for a GeoGG stage, adapted to the existing job runner."""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import config_sha256
from .contracts import InputRecord
from .provenance import sha256_file


@dataclass(frozen=True)
class RunLayout:
    root: Path
    run_id: str

    @property
    def stages(self) -> Path:
        return self.root / "stages"

    @property
    def provenance(self) -> Path:
        return self.root / "provenance"


class StageContext:
    """Paths and inputs for one stage within an existing APGCE processing job.

    Does not create or manage jobs, database records, SLURM submissions or
    dataset storage. The host runner owns those lifecycle concerns.
    """

    def __init__(
        self,
        *,
        stage_id: str,
        job_dir: str | Path,
        inputs: list[InputRecord] | None = None,
        config: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", stage_id):
            raise ValueError("stage_id must be a simple identifier")
        selected_run_id = run_id or str(uuid.uuid4())
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", selected_run_id):
            raise ValueError("run_id must be a simple identifier")
        self.stage_id = stage_id
        self.job_dir = Path(job_dir).expanduser().resolve()
        self.run_id = selected_run_id
        self.layout = RunLayout(self.job_dir, self.run_id)
        self.stage_dir = self.layout.stages / stage_id
        self.inputs = list(inputs or [])
        self._input_hashes: dict[str, str | None] = {}
        self.config = config or {}
        self.config_hash = config_sha256(self.config)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.layout.provenance.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_environment(cls, *, stage_id: str, config: dict[str, Any] | None = None, run_id: str | None = None) -> "StageContext":
        """Build context from the environment exported by existing local/SLURM runners."""
        result_path = os.environ.get("APGCE_RESULT_PATH")
        if not result_path:
            raise ValueError("APGCE_RESULT_PATH is required to locate the existing processing job directory")
        manifest_path = os.environ.get("APGCE_INPUT_MANIFEST")
        inputs: list[InputRecord] = []
        if manifest_path:
            manifest_file = Path(manifest_path).expanduser().resolve()
            try:
                payload = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Unable to read APGCE input manifest: {exc}") from exc
            rows = payload.get("inputs", [])
            if not isinstance(rows, list):
                raise ValueError("Input manifest 'inputs' must be a list")
            for row in rows:
                if not isinstance(row, dict) or not row.get("path"):
                    raise ValueError("Each input manifest entry must be an object with a path")
                inputs.append(InputRecord(
                    dataset_id=row.get("dataset_id"),
                    kind=str(row.get("kind") or "unknown"),
                    format=str(row.get("format") or ""),
                    name=str(row.get("name") or Path(row["path"]).name),
                    path=str(Path(row["path"]).expanduser().resolve()),
                    well_id=row.get("well_id"),
                    well_name=row.get("well_name"),
                    survey_id=row.get("survey_id"),
                    survey_name=row.get("survey_name"),
                    well_x=row.get("well_x"),
                    well_y=row.get("well_y"),
                    well_crs=row.get("well_crs"),
                    well_metadata=row.get("well_metadata") if isinstance(row.get("well_metadata"), dict) else {},
                    metadata=row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
                ))
        return cls(stage_id=stage_id, job_dir=Path(result_path).resolve().parent, inputs=inputs, config=config, run_id=run_id)

    def add_input(self, path: str | Path | None, role: str, hash_it: bool = True) -> str | None:
        """Register an existing input and return its SHA-256 when requested."""
        if path is None:
            return None
        file_path = Path(path).expanduser().resolve()
        if not file_path.is_file():
            self._input_hashes[str(file_path)] = None
            return None
        digest = None
        if hash_it:
            digest = sha256_file(file_path, chunk_size=1024 * 1024)
        self._input_hashes[str(file_path)] = digest
        return digest

    def input_paths(self, *, kind: str | None = None, format: str | None = None) -> list[Path]:
        """Return existing manifest paths, optionally filtered by kind/format."""
        records = [
            item for item in self.inputs
            if (kind is None or item.kind.casefold() == kind.casefold())
            and (format is None or item.format.casefold() == format.casefold())
        ]
        return [Path(item.path) for item in records]

    def resolve_output(self, relative_path: str | Path) -> Path:
        """Resolve a stage output and reject traversal outside its stage directory."""
        target = (self.stage_dir / relative_path).resolve()
        if self.stage_dir.resolve() not in target.parents and target != self.stage_dir.resolve():
            raise ValueError("Stage output path must remain inside the stage output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
