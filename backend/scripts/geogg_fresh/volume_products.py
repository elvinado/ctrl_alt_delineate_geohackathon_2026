"""Compatibility facade for the migrated GeoGG Stage 10/11 modules.

New code should import Stage 10 from ``geogg_fresh.stages.s10_summary`` and
Stage 11 from ``geogg_fresh.stages.s11_volume_export``.
"""
from .stages.s10_summary import (
    run_volume_products, _resolve_recorded_path, _normalize_volume_axis_order,
    _build_seismic_sections, _import_evaluation_evidence, _png_data_uri,
    _is_unavailable_linked_input, _recorded_dataset_id,
)
from .stages.s11_volume_export import (
    _write_stage11_segy_products, _write_native_volume_export_stage,
    _write_stage11_validation_protocol, write_native_volume_export_stage,
)

__all__ = ["run_volume_products"]
