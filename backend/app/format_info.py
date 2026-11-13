"""Expose supported formats and the implementation status of pipeline stages."""

from fastapi import APIRouter

router = APIRouter()

PIPELINE_DOMAINS = [
    {
        "id": "INGEST",
        "name": "Data Ingestion",
        "status": "available",
        "description": "Register projects, surveys, wells, and datasets; upload, store, and inspect files through resource-oriented routes.",
    },
    {
        "id": "QC",
        "name": "Quality Control and Harmonization",
        "status": "placeholder",
        "description": "Dataset-scoped quality-check endpoint is registered but returns HTTP 501 until implemented.",
    },
    {
        "id": "FEATGEN",
        "name": "Feature Generation",
        "status": "placeholder",
        "description": "Dataset-scoped feature-generation endpoint is registered but returns HTTP 501 until implemented.",
    },
    {
        "id": "MLPROC",
        "name": "Machine Learning Processing",
        "status": "available",
        "description": "A basic GPU availability test can be submitted through SLURM.",
    },
]


@router.get("/capabilities", tags=["capabilities"])
def capabilities() -> dict[str, object]:
    """Return implemented formats and pipeline-domain readiness."""
    return {
        "pipeline_domains": PIPELINE_DOMAINS,
        "dataset_kinds": ["seismic", "well_log", "checkshot", "deviation", "table"],
        "formats_supported_for_upload": ["las", "segy", "sgy", "txt", "xlsx", "xlsm"],
        "parsers": {
            "las": "basic LAS header parsing and curve cataloguing available",
            "segy": "basic SEG-Y header parsing available",
            "txt:deviation": "comment-header and numeric-row parsing available; use /datasets/{dataset_id}/table",
            "txt:checkshot": "comment-header and numeric-row parsing available; use /datasets/{dataset_id}/table",
            "xlsx:well_list": "Excel well-list import supports Well Name, RT, TD, X and Y columns",
        },
        "processing_status": {
            "QC": "not implemented",
            "FEATGEN": "not implemented",
            "MLPROC": "basic SLURM GPU test available",
        },
    }
