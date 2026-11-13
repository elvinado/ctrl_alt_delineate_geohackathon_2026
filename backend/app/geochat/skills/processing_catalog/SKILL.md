# Processing Catalog

## Description
Discover Python processing scripts in the active project's configured `scripts/` directory without importing or executing them.

## Use this skill when
- The user asks which Python scripts or processing workflows are available.
- The user asks to scan or list scripts that can be selected in the Processing tab.

## Instructions
1. Scan the active storage root's `scripts/` directory at request time; never use a hardcoded script list.
2. Exclude shared helper modules such as `scripts_utils.py` from the selectable script catalogue.
3. Extract a concise description from each script's module docstring or argparse description. Do not invent a purpose when metadata is absent.
4. Never import or execute discovered scripts. Discovery is read-only and does not establish runtime readiness.
5. Keep the catalogue generic. Do not hardcode workflow names, model names, or process-specific behavior.

## Result contract
Return `scripts` as a list of objects with `id`, `name`, `path`, and `description`. The `id` and `name` identify the selectable Python filename. Keep `workers` as a compatibility field for existing backend consumers.
