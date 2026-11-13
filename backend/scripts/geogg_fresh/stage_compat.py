"""Small compatibility surface for the migrated specialist Stage 00/01 code.

It deliberately delegates paths and job identity to the APGCE StageContext;
no second pipeline/job/checkpoint manager is introduced.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import logging

import pandas as pd


class StageBlocked(RuntimeError):
    """An upstream prerequisite or essential input is unavailable."""


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _RunView:
    def __init__(self, run_id: str, run_dir: Path):
        self.run_id = run_id
        self.run_dir = run_dir


class SpecialistStageContext:
    """Expose the specialist stage API over the existing APGCE execution context."""
    def __init__(self, base, cfg: dict[str, Any], stage_name: str):
        self.base = base
        self.cfg = cfg
        self.stage = stage_name
        self.run = _RunView(base.run_id, base.layout.stages)
        self.run.code_version = "apgce-geogg-migrated"
        self.stage_dir = base.stage_dir
        self.tables_dir = self.stage_dir / "tables"
        self.figures_dir = self.stage_dir / "figures"
        # Keep only directories used by stage artifacts. Runtime logs go to
        # the job runner's console/logging system, not a per-stage placeholder.
        for folder in (self.tables_dir, self.figures_dir):
            folder.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"apgce.geogg.{base.run_id}.{stage_name}")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(f"[{stage_name}] %(levelname)s %(message)s"))
            self.logger.addHandler(handler)
        self.logger.propagate = False
        self.warnings: list[str] = []
        self.limitations: list[str] = []
        self.quality_flags: list[dict[str, Any]] = []
        self.blockers: list[str] = []
        self.checks: dict[str, dict[str, Any]] = {}
        self.metrics: dict[str, Any] = {}
        self.extra: dict[str, Any] = {}
        self.parameters: dict[str, Any] = {}
        self.readme_sections: list[tuple[str, str]] = []
        self.inputs: dict[str, dict[str, Any]] = {}
        # Manifest-backed APGCE dataset records stay separate from dynamically
        # registered file inputs so add_input() always writes to the metadata map.
        self.input_records = list(getattr(base, "inputs", []) or [])
        # Specialist-stage output metadata is retained for compatibility.
        # The APGCE runner remains responsible for final provenance scanning.
        self.registered_outputs: dict[str, dict[str, Any]] = {}

    def table_path(self, name: str) -> Path:
        """Return a safe path beneath this stage's tables directory."""
        path = (self.tables_dir / name).resolve()
        root = self.tables_dir.resolve()
        if root not in path.parents and path != root:
            raise ValueError("Table output path must remain inside the stage tables directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def figure_path(self, name: str) -> Path:
        """Return a safe path beneath this stage's figures directory."""
        path = (self.figures_dir / name).resolve()
        root = self.figures_dir.resolve()
        if root not in path.parents and path != root:
            raise ValueError("Figure output path must remain inside the stage figures directory")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _register(self, path: str | Path, kind: str, description: str,
                  units=None, required: bool = False, multipanel: bool = False) -> None:
        """Retain specialist output metadata without creating a second provenance system."""
        output = Path(path).expanduser().resolve()
        stage_root = self.stage_dir.resolve()
        if stage_root not in output.parents:
            raise ValueError("Registered output must remain inside the stage directory")
        relative = output.relative_to(stage_root).as_posix()
        self.registered_outputs[relative] = {
            "kind": kind,
            "description": description,
            "units": units,
            "required": required,
            "multipanel": multipanel,
        }

    def info(self, msg: str, *args: Any) -> None:
        self.logger.info(msg, *args)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)
        self.logger.warning(msg)

    def limit(self, msg: str, *, category: str | None = None) -> None:
        """Record a limitation and, when classified, its downstream quality scope."""
        self.limitations.append(msg)
        if category:
            self.quality_flags.append({"category": category, "detail": msg})
        self.logger.warning("LIMITATION: %s", msg)

    def block(self, msg: str) -> None:
        self.blockers.append(msg)
        self.logger.error("BLOCKER: %s", msg)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        passed = bool(ok)
        self.checks[name] = {"passed": passed, "detail": detail}
        log = self.logger.info if passed else self.logger.warning
        log("CHECK %s: %s %s", name, "PASS" if passed else "FAIL", detail)
        return passed

    def metric(self, key: str, value: Any) -> None:
        self.metrics[key] = value

    def readme(self, title: str, body: str) -> None:
        self.readme_sections.append((title, body.strip()))

    def add_input(self, path, role: str, hash_it: bool = True):
        if path is None:
            return None
        file_path = Path(path).expanduser().resolve()
        digest = self.base.add_input(file_path, role, hash_it=hash_it)
        self.inputs[str(file_path)] = {
            "role": role, "exists": file_path.is_file(),
            "sha256": digest, "bytes": file_path.stat().st_size if file_path.is_file() else None,
        }
        return digest
    def save_table(
        self,
        df: pd.DataFrame,
        name: str,
        description: str,
        units=None,
        required: bool = False,
    ) -> Path:
        path = self.table_path(name)
        df.to_csv(path, index=False)
        return path

    def save_json(
        self,
        obj: Any,
        relpath: str,
        description: str,
        required: bool = False,
    ) -> Path:
        path = self.base.resolve_output(relpath)
        path.write_text(
            json.dumps(obj, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def save_text(
        self,
        text: str,
        relpath: str,
        description: str,
        required: bool = False,
    ) -> Path:
        path = self.base.resolve_output(relpath)
        path.write_text(text, encoding="utf-8")
        return path

    def save_figure(
        self,
        fig,
        name: str,
        description: str,
        required: bool = False,
        multipanel: bool = False,
        dpi: int | None = None,
    ) -> Path:
        import matplotlib.pyplot as plt

        path = self.figure_path(name)
        figure_dpi = dpi or int(self.cfg.get("figures", {}).get("dpi", 150))
        fig.savefig(path, dpi=figure_dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        return path
