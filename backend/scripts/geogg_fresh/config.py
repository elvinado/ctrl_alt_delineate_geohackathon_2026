"""Configuration loading and compatibility mapping for GeoGG Fresh.

The legacy specialist YAML remains the scientific source of truth. This adapter
normalizes its project/output roots and accepts an already-resolved mapping from
the host application. YAML support is imported lazily and is therefore optional.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


class GeoGGConfigError(ValueError):
    """Invalid or incomplete GeoGG Fresh configuration."""


def _absolute_path(value: Any, *, base_dir: Path) -> Path | None:
    if value is None or not str(value).strip():
        return None
    raw = os.path.expandvars(os.path.expanduser(str(value).strip()))
    path = Path(raw)
    return path if path.is_absolute() else (base_dir / path).resolve()


def normalize_config(
    source: Mapping[str, Any],
    *,
    config_path: str | Path | None = None,
    project_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    """Return a detached config with normalized roots and stable runtime metadata.

    ``source`` may be the legacy GeoGG YAML structure or a host-generated mapping.
    It is not mutated. No dataset location is guessed from filenames.
    """
    cfg = json.loads(json.dumps(dict(source), default=str))
    base_dir = Path(config_path).expanduser().resolve().parent if config_path else Path.cwd()
    paths = cfg.setdefault("paths", {})
    project = cfg.setdefault("project", {})

    configured_project = project_root or paths.get("project_root")
    configured_output = output_root or project.get("output_root")
    if configured_project is not None:
        resolved_project = _absolute_path(configured_project, base_dir=base_dir)
        paths["project_root"] = str(resolved_project)
    if configured_output is not None:
        resolved_output = _absolute_path(configured_output, base_dir=base_dir)
        project["output_root"] = str(resolved_output)

    if not paths.get("project_root"):
        raise GeoGGConfigError("Missing required configuration: paths.project_root")
    if not project.get("output_root"):
        raise GeoGGConfigError("Missing required configuration: project.output_root")

    wells = cfg.get("wells")
    if isinstance(wells, dict) and isinstance(wells.get("expected"), list):
        wells["expected"] = [str(well).strip().lower() for well in wells["expected"]]

    cfg["_runtime"] = {
        "config_path": str(Path(config_path).expanduser().resolve()) if config_path else None,
        "config_sha256": config_sha256(cfg),
    }
    return cfg


def load_yaml_config(path: str | Path, **overrides: Any) -> dict[str, Any]:
    """Load legacy YAML without making PyYAML a backend-wide dependency."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise GeoGGConfigError(f"GeoGG config file not found: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise GeoGGConfigError("YAML config loading requires the optional PyYAML dependency") from exc
    try:
        source = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise GeoGGConfigError(f"Unable to read GeoGG YAML config: {exc}") from exc
    if not isinstance(source, dict):
        raise GeoGGConfigError("GeoGG YAML root must be a mapping")
    return normalize_config(source, config_path=config_path, **overrides)


def config_sha256(config: Mapping[str, Any]) -> str:
    """Hash public configuration deterministically, excluding runtime metadata."""
    public = {key: value for key, value in config.items() if not str(key).startswith("_")}
    payload = json.dumps(public, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Read a nested configuration value using a dotted key."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def resolve_path(cfg: dict, value: Any) -> Path | None:
    """Resolve a configured scientific path relative to the configured project root."""
    if value is None or not str(value).strip():
        return None
    raw = Path(os.path.expandvars(os.path.expanduser(str(value).strip())))
    if raw.is_absolute():
        return raw
    root = Path(str(cfg.get("paths", {}).get("project_root", "."))).expanduser()
    if not root.is_absolute():
        root = Path(cfg.get("_runtime", {}).get("config_path") or Path.cwd()).resolve().parent / root
    return (root / raw).resolve()


def resolve_expected_wells(config: Mapping[str, Any], *, inventory=None, labels=None) -> list[str]:
    """Resolve the run's well population without letting an empty optional config erase it.

    Priority: explicit configured wells, Stage 00 inventory, then Stage 02 label rows.
    Values are normalized and de-duplicated while preserving source order.
    """
    configured = cfg_get(dict(config), "wells.expected", []) or []
    candidates = configured
    if not candidates and inventory is not None and "well" in inventory:
        candidates = inventory["well"].dropna().astype(str).tolist()
    if not candidates and labels is not None and "well" in labels:
        candidates = labels["well"].dropna().astype(str).tolist()
    result: list[str] = []
    seen: set[str] = set()
    for value in candidates:
        well = str(value).strip().casefold()
        if well and well not in seen:
            seen.add(well)
            result.append(well)
    return result
