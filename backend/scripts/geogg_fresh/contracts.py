"""Stable, lightweight contracts shared by migrated GeoGG stages."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol


class StageStatus(str, Enum):
    COMPLETED = "COMPLETED"
    CONDITIONAL = "CONDITIONAL"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class InputRecord:
    dataset_id: str | None
    kind: str
    format: str
    name: str
    path: str
    well_id: str | None = None
    well_name: str | None = None
    survey_id: str | None = None
    survey_name: str | None = None
    well_x: float | None = None
    well_y: float | None = None
    well_crs: str | None = None
    well_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputRecord:
    path: str
    role: str
    media_type: str
    size_bytes: int
    sha256: str


@dataclass
class StageResult:
    """A stage's explicit result; scientific limitations are not hidden as warnings."""

    stage_id: str
    status: StageStatus
    summary: str = ""
    outputs: list[OutputRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    quality_flags: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["status"] = self.status.value
        return result


@dataclass(frozen=True)
class StageSpec:
    """Declarative stage identity and dependencies; execution remains in wrappers."""

    stage_id: str
    title: str
    depends_on: tuple[str, ...] = ()
    input_roles: tuple[str, ...] = ()
    output_roles: tuple[str, ...] = ()



class StageRunner(Protocol):
    """Interface implemented by each migrated stage module."""

    def run(self, context: "StageContext") -> StageResult:
        """Execute one stage and return its explicit scientific outcome."""
        ...
