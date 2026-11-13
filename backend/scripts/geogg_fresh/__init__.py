"""GeoGG Fresh integration foundation.

This package intentionally imports no scientific stack at module import time.
Stage implementations are migrated incrementally in later roadmap stages.
"""
from .contracts import StageResult, StageRunner, StageStatus
from .context import StageContext

__all__ = ["StageContext", "StageResult", "StageRunner", "StageStatus"]
__version__ = "0.1.0"
