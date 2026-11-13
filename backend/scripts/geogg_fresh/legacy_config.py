"""Compatibility helpers for the original GeoGG scientific configuration."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any


def cfg_get(cfg: dict, dotted: str, default: Any = None):
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node: return default
        node = node[part]
    return node


def resolve_path(cfg: dict, value: Any):
    if value is None: return None
    text = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if not text: return None
    path = Path(text)
    if path.is_absolute() or (len(text) > 1 and text[1] == ":"): return path
    root_text = os.path.expandvars(os.path.expanduser(str(cfg.get("paths", {}).get("project_root", ""))))
    root = Path(root_text) if root_text else Path.cwd()
    if not root.is_absolute(): root = (Path(cfg.get("_config_dir", Path.cwd())) / root).resolve()
    return root / path


def public_config(cfg: dict):
    return {k: v for k, v in cfg.items() if not str(k).startswith("_")}
