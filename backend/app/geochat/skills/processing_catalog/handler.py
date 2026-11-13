"""Read-only discovery of processing workers and model assets."""
from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.storage import ensure_storage_root

SUPPORT_FILES = {"scripts_utils.py"}

def _source_metadata(path: Path) -> dict[str, Any]:
    """Extract lightweight metadata without importing or executing the script."""
    try:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=path.name)
    except (OSError, UnicodeDecodeError, SyntaxError) as exc:
        return {"description": "Source metadata unavailable", "entry_point": False, "metadata_error": str(exc)}

    docstring = ast.get_docstring(tree)
    has_main_guard = any(
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
        for node in ast.walk(tree)
    )
    argparse_used = any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "ArgumentParser"
        for node in ast.walk(tree)
    )
    description = docstring.splitlines()[0].strip() if docstring else None
    if not description:
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "ArgumentParser":
                for keyword in node.keywords:
                    if keyword.arg == "description" and isinstance(keyword.value, ast.Constant):
                        description = str(keyword.value.value).strip()
                        break
            if description:
                break
    return {
        "description": description or "No module description found",
        "entry_point": has_main_guard or argparse_used,
        "has_main_guard": has_main_guard,
        "uses_argparse": argparse_used,
    }


def _discover_models(scripts_dir: Path) -> list[dict[str, Any]]:
    models = []
    for directory in sorted((p for p in scripts_dir.iterdir() if p.is_dir() and p.name.lower().startswith("model")), key=lambda p: p.name.casefold()):
        checkpoint = directory / "checkpoint"
        assets = sorted(path.name for path in checkpoint.iterdir() if path.is_file()) if checkpoint.is_dir() else []
        models.append({
            "name": directory.name,
            "checkpoint_directory": str(checkpoint) if checkpoint.is_dir() else None,
            "assets": assets,
            "asset_count": len(assets),
            "asset_status": "assets discovered" if assets else "no checkpoint assets found",
        })
    return models


def execute(*, db: Session, project_id: str, parameters: dict[str, Any]) -> dict[str, Any]:
    """Scan the active storage root and classify scripts without running them."""
    scripts_dir = ensure_storage_root(db) / "scripts"
    if not scripts_dir.is_dir():
        return {"scripts_directory": str(scripts_dir), "directory_exists": False, "workers": [], "scripts": [], "support_files": [], "runners": [], "models": []}

    workers: list[dict[str, Any]] = []
    support_files: list[str] = []
    runners = sorted(path.name for path in scripts_dir.glob("*.sh") if path.is_file())
    for path in sorted(scripts_dir.glob("*.py"), key=lambda p: p.name.casefold()):
        if not path.is_file():
            continue
        if path.name in SUPPORT_FILES:
            support_files.append(path.name)
            continue
        metadata = _source_metadata(path)
        workers.append({
            "id": path.name,
            "name": path.name,
            "path": str(path),
            **metadata,
            "runnable_candidate": bool(metadata.get("entry_point")),
        })

    return {
        "scripts_directory": str(scripts_dir),
        "directory_exists": True,
        "workers": workers,
        # Stable, process-agnostic catalogue contract for Ask Geo and other clients.
        "scripts": [
            {
                "id": item["id"],
                "name": item["name"],
                "path": item["path"],
                "description": item["description"],
            }
            for item in workers
        ],
        "runnable_count": sum(1 for item in workers if item["runnable_candidate"]),
        "support_files": support_files,
        "runners": runners,
        "models": _discover_models(scripts_dir),
        "scan_is_read_only": True,
    }
