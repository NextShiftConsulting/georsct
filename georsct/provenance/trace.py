"""Provenance trace schema for GeoRSCT-X execution harness.

The key differentiator from TerraBench: the trace records not merely
tool use, but the admissibility rationale, certificate deltas, and
artifacts for each expert activation.

Artifacts are referenced by URI (S3 or local), not stored inline.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Verdict vocabulary
# ---------------------------------------------------------------------------

class Verdict(str, Enum):
    """Certificate verdict after harness execution."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    SUPPRESS = "suppress"
    ESCALATE = "escalate"


# ---------------------------------------------------------------------------
# Gate (per-gate outcome within a certificate)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Gate:
    """Single gate evaluation result."""

    name: str
    threshold: float
    observed: float
    passed: bool
    explanation: str = ""


# ---------------------------------------------------------------------------
# Weakness (ranked, capped at 3)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Weakness:
    """A diagnosed certificate weakness, ranked by severity."""

    weakness_type: str
    severity: float  # [0, 1], higher = more severe

    # Valid weakness types
    VALID_TYPES = frozenset({
        "low_target_coverage",
        "under_supported_geometry",
        "residual_spatial_structure",
        "leakage",
        "none",
    })

    def __post_init__(self):
        if self.weakness_type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown weakness type: {self.weakness_type!r}"
            )


# ---------------------------------------------------------------------------
# Certificate (RSCT/YRSN simplex + geospatial metrics)
# ---------------------------------------------------------------------------

@dataclass
class ExecutionCertificate:
    """Certificate state during harness execution.

    Bridges the existing domain types (ReadinessCertificate, GeometryKappa,
    ConstructCertificate) into a single execution-layer view.
    """

    geometry: str

    # YRSN simplex (R + S_sup + N = 1)
    R: float
    S_sup: float
    N: float

    # Compatibility
    kappa_geom: float
    kappa_req: float  # Oobleck threshold

    # Geospatial quality
    leakage_score: float
    fold_stability: float
    residual_moran: float  # lower = better

    # Gates
    gates: list[Gate] = field(default_factory=list)

    # Verdict
    verdict: Verdict = Verdict.WARN

    def is_admissible(self) -> bool:
        return self.verdict in (Verdict.PASS, Verdict.WARN)

    def weakness_vector(self, max_weaknesses: int = 3) -> list[Weakness]:
        """Ranked weakness vector, capped at max_weaknesses.

        If more than max_weaknesses are active, the certificate should
        FAIL outright -- the representation is too degraded for
        incremental expert enrichment.
        """
        weaknesses: list[Weakness] = []
        if self.N > 0.5:
            weaknesses.append(Weakness("low_target_coverage", min(self.N, 1.0)))
        if self.residual_moran > 0.3:
            weaknesses.append(Weakness(
                "residual_spatial_structure", min(self.residual_moran, 1.0),
            ))
        if self.kappa_geom < self.kappa_req:
            gap = self.kappa_req - self.kappa_geom
            weaknesses.append(Weakness(
                "under_supported_geometry", min(gap / self.kappa_req, 1.0),
            ))
        if self.leakage_score > 0.2:
            weaknesses.append(Weakness("leakage", min(self.leakage_score, 1.0)))

        # Rank by severity descending
        weaknesses.sort(key=lambda w: w.severity, reverse=True)

        if len(weaknesses) > max_weaknesses:
            # Too degraded for incremental enrichment
            self.verdict = Verdict.FAIL

        return weaknesses[:max_weaknesses]

    def primary_weakness(self) -> str:
        """Top weakness type (for backward compat with gearbox)."""
        wv = self.weakness_vector()
        return wv[0].weakness_type if wv else "none"


# ---------------------------------------------------------------------------
# Artifact (provenance-bearing, URI-referenced)
# ---------------------------------------------------------------------------

@dataclass
class Artifact:
    """Evidence object with checksum, referenced by URI."""

    artifact_id: str
    artifact_type: str  # geotiff | netcdf | csv | parquet | png | json
    uri: str  # s3://... or file://... (no inline payload)
    checksum: str = ""
    spatial_extent: Optional[dict[str, Any]] = None
    crs: Optional[str] = None
    created_by_step: int = -1
    source_version: str = ""

    @staticmethod
    def compute_checksum(payload: bytes) -> str:
        """SHA-256 checksum (first 16 hex chars)."""
        return hashlib.sha256(payload).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Step (admissibility-provenance log entry)
# ---------------------------------------------------------------------------

@dataclass
class Step:
    """One step in the execution trace.

    This is what distinguishes GeoRSCT-X from TerraBench:
    each step records WHY an expert was admitted, not just
    which tool was called.
    """

    index: int
    tool_name: str
    tool_group: str
    args: dict[str, Any] = field(default_factory=dict)

    # Validity
    valid: bool = True  # passed schema validation
    success: bool = True  # executed without error

    # Observation
    observation: dict[str, Any] = field(default_factory=dict)
    artifact_ids: list[str] = field(default_factory=list)

    # --- Admissibility provenance (GeoRSCT-X differentiator) ---
    admission_reason: str = ""
    gear: str = ""
    certificate_before: Optional[dict[str, Any]] = None
    certificate_after: Optional[dict[str, Any]] = None
    compatibility_delta: float = 0.0


# ---------------------------------------------------------------------------
# Trace (full execution provenance log)
# ---------------------------------------------------------------------------

@dataclass
class Trace:
    """Complete execution trace for one benchmark task.

    The trace is an admissibility log, not just a tool-use log.
    """

    task_id: str
    steps: list[Step] = field(default_factory=list)
    final_json: dict[str, float] = field(default_factory=dict)
    certificate: Optional[ExecutionCertificate] = None

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d, indent=2, default=str)

    @property
    def expert_ids(self) -> list[str]:
        return [s.tool_name for s in self.steps]

    @property
    def tool_groups(self) -> list[str]:
        return [s.tool_group for s in self.steps]
