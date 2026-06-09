"""Harness and patch schemas for R5 harness evolution.

Each harness version is a versioned JSON artifact with editable and frozen
components. Patches are allowlisted JSON Patch operations that modify only
editable components.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Editable components (evolve across steps)
# ---------------------------------------------------------------------------

@dataclass
class EvidenceTemplate:
    """Template controlling how ZCTA features are rendered as text for VLM."""
    preamble: str = ""
    feature_order: list[str] = field(default_factory=list)
    include_uncertainty_note: bool = True
    max_features: int = 20


@dataclass
class FeaturePolicy:
    """Which features from event_df enter the VLM prompt."""
    include: dict[str, bool] = field(default_factory=dict)
    exclude: list[str] = field(default_factory=list)


@dataclass
class Rubric:
    """Scoring rubric provided to the VLM."""
    risk_levels: list[str] = field(
        default_factory=lambda: ["low", "medium", "high", "unknown"]
    )
    instructions: str = ""
    require_uncertainty: bool = True
    require_evidence_citation: bool = True


@dataclass
class ScenarioMemory:
    """Lessons learned from prior evolution steps."""
    entries: list[dict[str, str]] = field(default_factory=list)


@dataclass
class EditableComponents:
    evidence_template: EvidenceTemplate = field(default_factory=EvidenceTemplate)
    feature_policy: FeaturePolicy = field(default_factory=FeaturePolicy)
    rubric: Rubric = field(default_factory=Rubric)
    scenario_memory: ScenarioMemory = field(default_factory=ScenarioMemory)

    ALLOWED_FIELDS: ClassVar[set[str]] = {
        "evidence_template", "feature_policy", "rubric", "scenario_memory",
    }


from typing import ClassVar  # noqa: E402 (deferred for ClassVar)


# ---------------------------------------------------------------------------
# Frozen components (fixed for entire experiment)
# ---------------------------------------------------------------------------

@dataclass
class FrozenComponents:
    map_renderer_version: str = "render_v1"
    legend_version: str = "legend_v1"
    color_scale_version: str = "floodrisk_v1"
    image_size: tuple[int, int] = (1024, 1024)


# ---------------------------------------------------------------------------
# Harness version
# ---------------------------------------------------------------------------

@dataclass
class HarnessVersion:
    """A single versioned snapshot of the full harness state."""
    harness_id: str
    parent_harness_id: str | None
    experiment_id: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    created_by: dict[str, str] = field(default_factory=dict)
    agent_context: dict[str, Any] = field(default_factory=dict)
    editable: EditableComponents = field(default_factory=EditableComponents)
    frozen: FrozenComponents = field(default_factory=FrozenComponents)
    lineage: dict[str, Any] = field(default_factory=dict)
    change_summary: str = ""
    validation: dict[str, bool] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dict for JSON storage."""
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Harness patch
# ---------------------------------------------------------------------------

@dataclass
class PatchOperation:
    """A single JSON Patch operation."""
    op: str          # add | replace | remove
    path: str        # JSON Pointer into editable_components
    value: Any = None

    def target_component(self) -> str:
        """Extract the top-level editable component from the path."""
        parts = self.path.strip("/").split("/")
        return parts[0] if parts else ""


@dataclass
class HarnessPatch:
    """A proposed set of changes from one harness version to the next."""
    patch_id: str
    from_harness: str
    to_harness: str
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    evolver_model: str = ""
    operations: list[PatchOperation] = field(default_factory=list)
    failure_evidence_used: list[dict[str, Any]] = field(default_factory=list)
    evolver_rationale: str = ""
    predicted_effect: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Certificate trajectory step
# ---------------------------------------------------------------------------

@dataclass
class RsctBlock:
    """RSCT certificate signals for one trajectory step (ADR-020 D8.8)."""
    R: float | None = None
    S_sup: float | None = None
    N: float | None = None
    alpha: float | None = None
    kappa_compat: float | None = None
    kappa_difficulty: float | None = None
    active_metric: str | None = None  # "kappa_compat" | "kappa_difficulty" | "kappa_modal_min"
    kappa_source: str | None = None  # "rsn_simplex" | "difficulty_theory" | etc.
    kappa_formula: str | None = None  # "R*(1-N)" | "D*/D_actual" | etc.
    kappa_authority: str | None = None  # "canonical_simplex_estimate" | etc.
    sigma: float | None = None
    kappa_req: float | None = None
    passed: bool | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class SpatialDiagnostics:
    """Spatial diagnostic signals (ADR-020 D8.6: diagnostic-only, never enforcement)."""
    kappa_geom: float | None = None
    morans_i_residual: float | None = None
    gearys_c_residual: float | None = None
    lisa_cluster_rate: float | None = None
    inter_vlm_kappa: float | None = None

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)


@dataclass
class TrajectoryStep:
    """One step in the certificate trajectory."""
    step: int
    harness_id: str
    split: str  # "train" | "validation" | "test"
    primary_score: float = 0.0
    activation_rate: float = 0.0
    adherence_rate: float = 0.0
    grounding_rate: float = 0.0
    schema_validity_rate: float = 0.0
    unprovided_claim_rate: float = 0.0
    rsct: RsctBlock = field(default_factory=RsctBlock)
    spatial_diagnostics: SpatialDiagnostics = field(default_factory=SpatialDiagnostics)
    patch_decision: str = ""  # "accepted" | "rejected" | "baseline"
    n_zctas: int = 0


@dataclass
class CertificateTrajectory:
    """Full trajectory across all evolution steps."""
    experiment_id: str
    agent_vlm: str
    evolver_model: str
    primary_metric: str
    steps: list[TrajectoryStep] = field(default_factory=list)

    def append_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)
