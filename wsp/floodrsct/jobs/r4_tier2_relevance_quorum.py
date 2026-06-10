"""
r4_tier2_relevance_quorum.py

Tier-2 semantic relevance judging for R4 VLM quality scoring.

This module splits Tier-1 verified / unverifiable VLM claims into:

    R_claim
        Source-supported or inferable AND operative for the flood risk assessment.

    S_sup_claim
        Source-supported or inferable BUT non-operative filler.

    None
        Judge abstention / missing value for reliability calculation.

Production labels are separate:

    R_claim
    S_sup_claim
    unresolved

Design invariants
-----------------
INV-1 SIDECAR ONLY.
    Nothing in this module enters kappa_geom, gate decisions, RSCT certificates,
    RsctBlock, SpatialDiagnostics, or any field that determines pass/fail.

INV-2 ALPHA SEES RAW A/B ONLY.
    Krippendorff alpha is computed only from independent Judge A and Judge B
    labels stored in ReliabilityRecord. Judge C and ProductionRecord are excluded.

INV-3 INDEPENDENT MODEL FAMILIES.
    Calibration independence is enforced from provenance stored in the
    ReliabilityRecord itself, not from caller-supplied Judge objects.

INV-4 MISSING IS MISSING.
    Abstentions are None, not an "ambiguous" class and not a disagreement.

INV-5 EXPLORATORY STATUS.
    Tier 2 post-dates DOE v1.4 and must be emitted as EXPLORATORY until
    replicated with a pre-registered rubric.

INV-6 STRUCTURAL FAIL-CLOSED ONLY.
    No model confidence gating. Parse failures, validation failures, dual
    abstention, C abstention, and unresolved A/B/C patterns produce unresolved.

INV-7 ABLATION PROVENANCE.
    Calibration must know whether the claim had a real leave-one-claim-out
    route ablation. By default the gate requires ablation for calibration.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Literal, Optional, Sequence


# ---------------------------------------------------------------------------
# Frozen rubric constants
# ---------------------------------------------------------------------------

ALPHA_THRESHOLD = 0.67
MIN_PAIRABLE = 100
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_LOWER_Q = 0.05

RUBRIC_VERSION = "tier2-relevance-quorum-v3-route-only"

Tier2Label = Literal["R_claim", "S_sup_claim"]
ProductionLabel = Literal["R_claim", "S_sup_claim", "unresolved"]
Tier1Label = Literal["verified", "unverifiable"]

DecisionAxis = Literal[
    "risk_magnitude",
    "risk_spatial",
    "risk_temporal",
    "vulnerability_load",
    "confidence_bound",
    "none",
]

AXES: tuple[DecisionAxis, ...] = (
    "risk_magnitude",
    "risk_spatial",
    "risk_temporal",
    "vulnerability_load",
    "confidence_bound",
    "none",
)

RELEVANCE_RULE = """
A claim is R_claim if BOTH conditions hold:

  1. The claim is supported by, or directly inferable from, the source evidence
     for this ZCTA and scenario.

  2. Removing the claim would change the Floodcaster assessment route:
       Trust / Review / Escalate / Suppress.

A claim is S_sup_claim if it is supported or inferable, but removing it would
not change the Floodcaster route.

Allowed explanatory axes:

  risk_magnitude
      Changes estimated flood probability, severity, depth, loss, or intensity.

  risk_spatial
      Changes which places, zones, neighborhoods, or ZCTA subareas are at risk.

  risk_temporal
      Changes timing, recurrence, duration, or event-to-event risk behavior.

  vulnerability_load
      Changes who or what is exposed or fragile.

  confidence_bound
      Changes certainty, uncertainty, limitation, or reliability of the
      assessment.

  none
      No assessment-relevant axis. Usually S_sup_claim.

Important scope rule:

  Do not reward downstream crisis-response invention. Evacuation priority,
  public messaging, shelter routing, operational command, and resource
  allocation are outside this Tier-2 task unless explicitly present in the
  VLM prompt/evidence as part of the flood risk assessment.

Ablation rule:

  When route_without_claim is provided, use the actual route-change
  counterfactual. When it is not provided, the label is a semantic judgment
  rather than a real route ablation. Calibration records must flag this via
  ablation_available. By default, calibration requires ablation_available=True.

Abstention rule:

  If the judge cannot determine R_claim vs S_sup_claim from the available
  context, return null. Do not invent an "ambiguous" label.
"""


def rubric_hash() -> str:
    blob = "|".join(
        [
            RUBRIC_VERSION,
            str(ALPHA_THRESHOLD),
            str(MIN_PAIRABLE),
            RELEVANCE_RULE,
            "|".join(AXES),
        ]
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClaimContext:
    """
    Context provided to Judge A, Judge B, and Judge C.

    Tier-1 fabricated / N_claim items must never reach this object.
    They should be finalized as N_claim upstream and skipped.
    """

    claim_id: str
    claim_text: str
    tier1_label: Tier1Label

    zcta: str
    scenario: str
    vlm_model_id: str

    source_evidence: str

    route_with_claim: str
    route_without_claim: Optional[str] = None

    def __post_init__(self) -> None:
        if self.tier1_label not in ("verified", "unverifiable"):
            raise ValueError(
                "Tier 2 only accepts tier1_label in {'verified', 'unverifiable'}; "
                f"got {self.tier1_label!r}"
            )

    @property
    def ablation_available(self) -> bool:
        return self.route_without_claim is not None


@dataclass(frozen=True)
class JudgeOutput:
    """
    Raw output from one independent judge.

    label=None means missing / abstain.
    It is not a third nominal category.
    """

    label: Optional[Tier2Label]
    decision_axis: DecisionAxis
    rationale_code: str

    def __post_init__(self) -> None:
        if self.label not in ("R_claim", "S_sup_claim", None):
            raise ValueError(
                f"bad label {self.label!r}; expected R_claim, S_sup_claim, or None"
            )

        if self.decision_axis not in AXES:
            raise ValueError(
                f"bad decision_axis {self.decision_axis!r}; expected one of {AXES}"
            )

        if self.label is None and self.decision_axis != "none":
            raise ValueError("abstention label=None must use decision_axis='none'")


@dataclass(frozen=True)
class ReliabilityRecord:
    """
    Alpha is computed from this record only.

    Judge C never appears here.
    Production labels never appear here.

    Model/family provenance travels with the data so INV-3 is structural.
    """

    claim_id: str

    judge_a_label: Optional[Tier2Label]
    judge_b_label: Optional[Tier2Label]

    judge_a_model_id: str
    judge_b_model_id: str
    judge_a_family: str
    judge_b_family: str

    judge_a_axis: Optional[DecisionAxis]
    judge_b_axis: Optional[DecisionAxis]

    ablation_available: bool

    rubric_version: str = RUBRIC_VERSION
    rubric_hash: str = field(default_factory=rubric_hash)


@dataclass(frozen=True)
class ProductionRecord:
    """
    Final production-side label.

    This record is never used to compute Krippendorff alpha.
    """

    claim_id: str
    tier1_label: Tier1Label

    final_label: ProductionLabel
    decision_axis: DecisionAxis

    adjudication_used: bool
    adjudication_status: Literal[
        "accepted_ab",
        "adjudicated_by_c",
        "unresolved_dual_abstain",
        "unresolved_c_abstain",
        "unresolved_three_way",
        "unresolved_no_adjudicator",
    ]

    judge_a_label: Optional[Tier2Label]
    judge_b_label: Optional[Tier2Label]
    judge_c_label: Optional[Tier2Label]

    judge_a_model_id: str
    judge_b_model_id: str
    judge_c_model_id: Optional[str]

    ablation_available: bool

    rationale_code: str

    tier2_status: Literal["EXPLORATORY"] = "EXPLORATORY"
    trusted_after_alpha_gate: bool = False

    rubric_version: str = RUBRIC_VERSION
    rubric_hash: str = field(default_factory=rubric_hash)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


@dataclass(frozen=True)
class Tier2Result:
    reliability: ReliabilityRecord
    production: ProductionRecord


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_prompt(ctx: ClaimContext) -> str:
    if ctx.route_without_claim is None:
        ablation_text = (
            "\nRoute without this claim: NOT_AVAILABLE\n"
            "Because no actual ablation is available, judge only whether the "
            "claim appears operative for the flood risk assessment from the "
            "provided evidence. This record will be flagged as "
            "ablation_available=false."
        )
    else:
        ablation_text = f"\nRoute without this claim: {ctx.route_without_claim}"

    return f"""
{RELEVANCE_RULE}

ZCTA: {ctx.zcta}
Scenario: {ctx.scenario}
VLM model: {ctx.vlm_model_id}
Tier-1 label: {ctx.tier1_label}

Route with this claim: {ctx.route_with_claim}
{ablation_text}

Source evidence given to the VLM:
{ctx.source_evidence}

Claim to judge:
"{ctx.claim_text}"

Return only JSON with this schema:

{{
  "label": "R_claim" | "S_sup_claim" | null,
  "decision_axis": "risk_magnitude" | "risk_spatial" | "risk_temporal" |
                   "vulnerability_load" | "confidence_bound" | "none",
  "rationale_code": "short_snake_case_reason"
}}

No prose. No markdown. No confidence score.
""".strip()


# ---------------------------------------------------------------------------
# Judge adapter abstraction and safe parsing
# ---------------------------------------------------------------------------

class Judge:
    """
    Thin adapter around any completion backend.

    In production, complete_fn calls Gemini / GPT / Claude.
    In tests, complete_fn can be a stub.
    """

    def __init__(
        self,
        model_id: str,
        family: str,
        complete_fn: Callable[[str], JudgeOutput],
    ):
        if not model_id:
            raise ValueError("model_id is required")
        if not family:
            raise ValueError("family is required")
        self.model_id = model_id
        self.family = family
        self.complete_fn = complete_fn

    def judge(self, ctx: ClaimContext) -> JudgeOutput:
        return self.complete_fn(build_prompt(ctx))


def _normalize_label(value: object) -> Optional[Tier2Label]:
    if value is None:
        return None

    if isinstance(value, str):
        normalized = value.strip()

        if normalized.lower() in {"", "null", "none", "na", "n/a", "abstain"}:
            return None

        if normalized in {"R_claim", "S_sup_claim"}:
            return normalized  # type: ignore[return-value]

    raise ValueError(
        f"bad judge label {value!r}; expected R_claim, S_sup_claim, or null"
    )


def _normalize_axis(value: object, label: Optional[Tier2Label]) -> DecisionAxis:
    if label is None:
        return "none"

    if not isinstance(value, str):
        raise ValueError(f"bad axis {value!r}; expected one of {AXES}")

    normalized = value.strip()

    if normalized in AXES:
        return normalized  # type: ignore[return-value]

    raise ValueError(f"bad axis {value!r}; expected one of {AXES}")


def parse_judge_json(text: str) -> JudgeOutput:
    """
    Parse a model response into JudgeOutput.

    Accepts bare JSON or fenced JSON.
    Normalizes string "null" / "none" / "" to None.
    Coerces any abstention axis to "none".
    """

    cleaned = text.strip()

    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").strip()

    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").strip()

    if cleaned.endswith("```"):
        cleaned = cleaned.removesuffix("```").strip()

    payload = json.loads(cleaned)

    label = _normalize_label(payload.get("label"))
    axis = _normalize_axis(payload.get("decision_axis", "none"), label)

    rationale = payload.get("rationale_code") or "missing_rationale"

    if not isinstance(rationale, str):
        rationale = "invalid_rationale_type"

    return JudgeOutput(
        label=label,
        decision_axis=axis,
        rationale_code=rationale,
    )


def safe_judge(judge: Judge, ctx: ClaimContext) -> JudgeOutput:
    """
    Run a judge with structural fail-closed behavior.

    Any parse, validation, transport, or adapter failure becomes abstention.
    No exception from a model call should kill the claim-level pipeline.
    """

    try:
        out = judge.judge(ctx)
    except Exception:
        return JudgeOutput(
            label=None,
            decision_axis="none",
            rationale_code="parse_or_validation_failure",
        )

    if out.label is None:
        return JudgeOutput(
            label=None,
            decision_axis="none",
            rationale_code=out.rationale_code or "judge_abstained",
        )

    return out


def make_openai_like_judge(
    model_id: str,
    family: str,
    client,
) -> Judge:
    """
    Generic adapter for OpenAI-style clients.

    Expected client interface:
        client.responses.create(model=..., input=...)
    """

    def complete_fn(prompt: str) -> JudgeOutput:
        response = client.responses.create(
            model=model_id,
            input=prompt,
        )

        text = getattr(response, "output_text", None)
        if text is None:
            raise RuntimeError("OpenAI-like response did not contain output_text")

        return parse_judge_json(text)

    return Judge(model_id=model_id, family=family, complete_fn=complete_fn)


def make_anthropic_judge(
    model_id: str,
    family: str,
    client,
) -> Judge:
    """
    Adapter for Anthropic-style clients.
    """

    def complete_fn(prompt: str) -> JudgeOutput:
        response = client.messages.create(
            model=model_id,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )

        chunks: list[str] = []

        for block in response.content:
            if getattr(block, "type", None) == "text":
                chunks.append(block.text)

        return parse_judge_json("".join(chunks))

    return Judge(model_id=model_id, family=family, complete_fn=complete_fn)


# ---------------------------------------------------------------------------
# Krippendorff alpha, nominal, missing-aware
# ---------------------------------------------------------------------------

def krippendorff_alpha_nominal(
    units: Sequence[Sequence[Optional[str]]],
) -> float:
    """
    Nominal Krippendorff alpha.

    units:
        One row per claim.
        One column per rater.
        None means missing and is excluded pairwise.

    Returns:
        float alpha, or nan if undefined.
    """

    coincidence: dict[tuple[str, str], float] = defaultdict(float)
    categories: set[str] = set()

    for unit in units:
        values = [value for value in unit if value is not None]
        m = len(values)

        if m < 2:
            continue

        categories.update(values)

        for i in range(m):
            for j in range(m):
                if i != j:
                    coincidence[(values[i], values[j])] += 1.0 / (m - 1)

    cats = sorted(categories)

    if not cats:
        return float("nan")

    n_c = {
        c: sum(coincidence[(c, k)] for k in cats)
        for c in cats
    }

    n = sum(n_c.values())

    if n <= 1:
        return float("nan")

    observed_diag = sum(coincidence[(c, c)] for c in cats)
    observed_disagreement_num = n - observed_diag

    expected_disagreement_num = n * n - sum(v * v for v in n_c.values())

    if expected_disagreement_num == 0:
        return 1.0 if observed_disagreement_num == 0 else float("nan")

    return 1.0 - ((n - 1) * observed_disagreement_num / expected_disagreement_num)


def _units_from_reliability(
    records: Sequence[ReliabilityRecord],
) -> list[list[Optional[str]]]:
    return [
        [record.judge_a_label, record.judge_b_label]
        for record in records
    ]


def _pairable_count(units: Sequence[Sequence[Optional[str]]]) -> int:
    return sum(
        1
        for unit in units
        if sum(value is not None for value in unit) >= 2
    )


def bootstrap_alpha_lower(
    units: Sequence[Sequence[Optional[str]]],
    iterations: int = BOOTSTRAP_ITERATIONS,
    lower_q: float = BOOTSTRAP_LOWER_Q,
    seed: int = 42,
) -> float:
    """
    Nonparametric bootstrap lower quantile for nominal alpha.

    Resamples claim units with replacement. Undefined alpha samples are dropped.
    Returns nan if no bootstrap sample is defined.
    """

    if not units:
        return float("nan")

    rng = random.Random(seed)
    estimates: list[float] = []

    n = len(units)

    for _ in range(iterations):
        sample = [units[rng.randrange(n)] for _ in range(n)]
        estimate = krippendorff_alpha_nominal(sample)

        if math.isfinite(estimate):
            estimates.append(estimate)

    if not estimates:
        return float("nan")

    estimates.sort()
    index = max(0, min(len(estimates) - 1, int(math.floor(lower_q * len(estimates)))))
    return estimates[index]


# ---------------------------------------------------------------------------
# Quorum adjudication
# ---------------------------------------------------------------------------

def adjudicate_claim(
    ctx: ClaimContext,
    judge_a: Judge,
    judge_b: Judge,
    judge_c: Optional[Judge],
) -> Tier2Result:
    """
    Run A/B independent judging and optional C adjudication.

    Returns both:
      - ReliabilityRecord for alpha
      - ProductionRecord for downstream sidecar metrics
    """

    a = safe_judge(judge_a, ctx)
    b = safe_judge(judge_b, ctx)

    reliability = ReliabilityRecord(
        claim_id=ctx.claim_id,
        judge_a_label=a.label,
        judge_b_label=b.label,
        judge_a_model_id=judge_a.model_id,
        judge_b_model_id=judge_b.model_id,
        judge_a_family=judge_a.family,
        judge_b_family=judge_b.family,
        judge_a_axis=a.decision_axis if a.label is not None else None,
        judge_b_axis=b.decision_axis if b.label is not None else None,
        ablation_available=ctx.ablation_available,
    )

    base = dict(
        claim_id=ctx.claim_id,
        tier1_label=ctx.tier1_label,
        judge_a_label=a.label,
        judge_b_label=b.label,
        judge_a_model_id=judge_a.model_id,
        judge_b_model_id=judge_b.model_id,
        ablation_available=ctx.ablation_available,
    )

    # Case 1: dual abstain
    if a.label is None and b.label is None:
        production = ProductionRecord(
            **base,
            final_label="unresolved",
            decision_axis="none",
            adjudication_used=False,
            adjudication_status="unresolved_dual_abstain",
            judge_c_label=None,
            judge_c_model_id=None,
            rationale_code="dual_abstain",
        )
        return Tier2Result(reliability=reliability, production=production)

    # Case 2: A/B agree on real label
    if a.label is not None and a.label == b.label:
        production = ProductionRecord(
            **base,
            final_label=a.label,
            decision_axis=a.decision_axis,
            adjudication_used=False,
            adjudication_status="accepted_ab",
            judge_c_label=None,
            judge_c_model_id=None,
            rationale_code=a.rationale_code,
        )
        return Tier2Result(reliability=reliability, production=production)

    # Case 3: A/B disagreement or one abstention
    if judge_c is None:
        production = ProductionRecord(
            **base,
            final_label="unresolved",
            decision_axis="none",
            adjudication_used=False,
            adjudication_status="unresolved_no_adjudicator",
            judge_c_label=None,
            judge_c_model_id=None,
            rationale_code="no_adjudicator_available",
        )
        return Tier2Result(reliability=reliability, production=production)

    c = safe_judge(judge_c, ctx)

    # C abstains
    if c.label is None:
        production = ProductionRecord(
            **base,
            final_label="unresolved",
            decision_axis="none",
            adjudication_used=True,
            adjudication_status="unresolved_c_abstain",
            judge_c_label=None,
            judge_c_model_id=judge_c.model_id,
            rationale_code="adjudicator_abstained",
        )
        return Tier2Result(reliability=reliability, production=production)

    real_ab_labels = {label for label in (a.label, b.label) if label is not None}

    # If one judge abstained, C must match the one real A/B label.
    if len(real_ab_labels) == 1 and c.label in real_ab_labels:
        production = ProductionRecord(
            **base,
            final_label=c.label,
            decision_axis=c.decision_axis,
            adjudication_used=True,
            adjudication_status="adjudicated_by_c",
            judge_c_label=c.label,
            judge_c_model_id=judge_c.model_id,
            rationale_code=c.rationale_code,
        )
        return Tier2Result(reliability=reliability, production=production)

    # If A and B gave opposite real labels, C must pick one of them.
    if len(real_ab_labels) == 2 and c.label in real_ab_labels:
        production = ProductionRecord(
            **base,
            final_label=c.label,
            decision_axis=c.decision_axis,
            adjudication_used=True,
            adjudication_status="adjudicated_by_c",
            judge_c_label=c.label,
            judge_c_model_id=judge_c.model_id,
            rationale_code=c.rationale_code,
        )
        return Tier2Result(reliability=reliability, production=production)

    # Structural fail-closed
    production = ProductionRecord(
        **base,
        final_label="unresolved",
        decision_axis="none",
        adjudication_used=True,
        adjudication_status="unresolved_three_way",
        judge_c_label=c.label,
        judge_c_model_id=judge_c.model_id,
        rationale_code="three_way_unresolved",
    )
    return Tier2Result(reliability=reliability, production=production)


# ---------------------------------------------------------------------------
# Calibration gate
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GateResult:
    alpha: float
    alpha_bootstrap_lower: float
    n_records: int
    n_pairable: int
    n_missing_a: int
    n_missing_b: int
    n_ablation_available: int
    passed: bool
    threshold: float = ALPHA_THRESHOLD
    min_pairable: int = MIN_PAIRABLE
    tier2_status: Literal["EXPLORATORY"] = "EXPLORATORY"

    @property
    def status(self) -> str:
        return "activated" if self.passed else "provisional"


def _single_value(records: Sequence[ReliabilityRecord], field_name: str) -> str:
    values = {getattr(record, field_name) for record in records}

    if len(values) != 1:
        raise ValueError(
            f"inconsistent {field_name} across reliability records: {sorted(values)}"
        )

    return next(iter(values))


def calibration_gate(
    reliability_records: Sequence[ReliabilityRecord],
    allow_same_family: bool = False,
    require_ablation: bool = True,
    min_pairable: int = MIN_PAIRABLE,
    bootstrap_iterations: int = BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = 42,
) -> GateResult:
    """
    Compute the activation gate from raw A/B labels only.

    Independence is derived from ReliabilityRecord provenance, not from
    caller-supplied Judge objects. This closes the INV-3 loophole where
    correlated labels could be generated first and laundered through different
    Judge objects at gate time.
    """

    if not reliability_records:
        return GateResult(
            alpha=float("nan"),
            alpha_bootstrap_lower=float("nan"),
            n_records=0,
            n_pairable=0,
            n_missing_a=0,
            n_missing_b=0,
            n_ablation_available=0,
            passed=False,
            min_pairable=min_pairable,
        )

    family_a = _single_value(reliability_records, "judge_a_family")
    family_b = _single_value(reliability_records, "judge_b_family")
    model_a = _single_value(reliability_records, "judge_a_model_id")
    model_b = _single_value(reliability_records, "judge_b_model_id")

    if family_a == family_b and not allow_same_family:
        raise ValueError(
            f"Calibration graders share family {family_a!r}. "
            "Alpha would measure self-consistency/correlation, not "
            "inter-rater reliability."
        )

    if model_a == model_b and not allow_same_family:
        raise ValueError(
            f"Calibration graders share model_id {model_a!r}. "
            "Use independent model families or explicitly override."
        )

    n_ablation_available = sum(record.ablation_available for record in reliability_records)

    if require_ablation and n_ablation_available != len(reliability_records):
        raise ValueError(
            "Calibration requires ablation_available=True for every record by "
            "default. Either generate route_without_claim for the calibration "
            "sample or call calibration_gate(..., require_ablation=False) and "
            "report stratified/provisional results."
        )

    units = _units_from_reliability(reliability_records)
    n_pairable = _pairable_count(units)

    alpha = krippendorff_alpha_nominal(units)
    alpha_lower = bootstrap_alpha_lower(
        units=units,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )

    n_missing_a = sum(record.judge_a_label is None for record in reliability_records)
    n_missing_b = sum(record.judge_b_label is None for record in reliability_records)

    passed = (
        n_pairable >= min_pairable
        and math.isfinite(alpha)
        and math.isfinite(alpha_lower)
        and alpha >= ALPHA_THRESHOLD
        and alpha_lower >= ALPHA_THRESHOLD
    )

    return GateResult(
        alpha=alpha,
        alpha_bootstrap_lower=alpha_lower,
        n_records=len(reliability_records),
        n_pairable=n_pairable,
        n_missing_a=n_missing_a,
        n_missing_b=n_missing_b,
        n_ablation_available=n_ablation_available,
        passed=passed,
        min_pairable=min_pairable,
    )


def apply_gate_to_production(
    production_records: Sequence[ProductionRecord],
    gate: GateResult,
) -> list[ProductionRecord]:
    """
    Return new production records with trusted_after_alpha_gate set.

    Records remain EXPLORATORY either way.
    """

    return [
        replace(record, trusted_after_alpha_gate=gate.passed)
        for record in production_records
    ]


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Tier2Summary:
    tier2_status: Literal["EXPLORATORY"]
    tier2_note: str

    alpha: float
    alpha_bootstrap_lower: float
    alpha_threshold: float
    alpha_min_pairable: int
    alpha_passed: bool

    n_records: int
    n_pairable: int
    n_ablation_available: int

    grounded_signal_rate: float
    filler_rate: float
    unresolved_rate: float
    tier2_coverage_rate: float

    adjudication_rate: float
    adjudication_resolved_rate: float
    unresolved_after_adjudication_rate: float
    dual_abstain_rate: float

    axis_distribution_status: Literal["DESCRIPTIVE_ONLY"]
    axis_distribution: dict[str, int]
    label_distribution: dict[str, int]
    adjudication_status_distribution: dict[str, int]

    rubric_version: str = RUBRIC_VERSION
    rubric_hash: str = field(default_factory=rubric_hash)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)


def summarize_tier2(
    production_records: Sequence[ProductionRecord],
    gate: GateResult,
) -> Tier2Summary:
    n = len(production_records)

    if n == 0:
        return Tier2Summary(
            tier2_status="EXPLORATORY",
            tier2_note=_tier2_note(),
            alpha=gate.alpha,
            alpha_bootstrap_lower=gate.alpha_bootstrap_lower,
            alpha_threshold=gate.threshold,
            alpha_min_pairable=gate.min_pairable,
            alpha_passed=gate.passed,
            n_records=0,
            n_pairable=gate.n_pairable,
            n_ablation_available=gate.n_ablation_available,
            grounded_signal_rate=float("nan"),
            filler_rate=float("nan"),
            unresolved_rate=float("nan"),
            tier2_coverage_rate=float("nan"),
            adjudication_rate=float("nan"),
            adjudication_resolved_rate=float("nan"),
            unresolved_after_adjudication_rate=float("nan"),
            dual_abstain_rate=float("nan"),
            axis_distribution_status="DESCRIPTIVE_ONLY",
            axis_distribution={},
            label_distribution={},
            adjudication_status_distribution={},
        )

    labels = Counter(record.final_label for record in production_records)
    axes = Counter(record.decision_axis for record in production_records)
    statuses = Counter(record.adjudication_status for record in production_records)

    resolved = labels["R_claim"] + labels["S_sup_claim"]
    unresolved = labels["unresolved"]

    adjudicated_total = sum(record.adjudication_used for record in production_records)
    adjudicated_resolved = statuses["adjudicated_by_c"]
    unresolved_after_adjudication = (
        statuses["unresolved_c_abstain"] + statuses["unresolved_three_way"]
    )

    grounded_signal_rate = (
        labels["R_claim"] / resolved
        if resolved > 0
        else float("nan")
    )

    filler_rate = (
        labels["S_sup_claim"] / resolved
        if resolved > 0
        else float("nan")
    )

    return Tier2Summary(
        tier2_status="EXPLORATORY",
        tier2_note=_tier2_note(),
        alpha=gate.alpha,
        alpha_bootstrap_lower=gate.alpha_bootstrap_lower,
        alpha_threshold=gate.threshold,
        alpha_min_pairable=gate.min_pairable,
        alpha_passed=gate.passed,
        n_records=n,
        n_pairable=gate.n_pairable,
        n_ablation_available=gate.n_ablation_available,
        grounded_signal_rate=grounded_signal_rate,
        filler_rate=filler_rate,
        unresolved_rate=unresolved / n,
        tier2_coverage_rate=resolved / n,
        adjudication_rate=adjudicated_total / n,
        adjudication_resolved_rate=adjudicated_resolved / n,
        unresolved_after_adjudication_rate=unresolved_after_adjudication / n,
        dual_abstain_rate=statuses["unresolved_dual_abstain"] / n,
        axis_distribution_status="DESCRIPTIVE_ONLY",
        axis_distribution=dict(axes),
        label_distribution=dict(labels),
        adjudication_status_distribution=dict(statuses),
    )


def _tier2_note() -> str:
    return (
        "Decision-axis taxonomy and route-counterfactual relevance definition "
        "post-date DOE v1.4 pre-registration. Treat Tier-2 relevance quorum "
        "as exploratory until replicated with a pre-registered rubric. "
        "Tier-2 is sidecar-only and never enters RSCT gate or certificate fields. "
        "Axis distribution is descriptive-only unless separately reliability-tested."
    )


# ---------------------------------------------------------------------------
# Tier-1 handoff helper
# ---------------------------------------------------------------------------

def tier1_eligible(tier1_label: str) -> bool:
    """
    Only verified and unverifiable claims go to Tier 2.

    Tier-1 fabricated / N_claim claims should be finalized upstream as N_claim
    and skipped.
    """

    return tier1_label in {"verified", "unverifiable"}


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------

def _stub_judge(
    model_id: str,
    family: str,
    output: JudgeOutput,
) -> Judge:
    return Judge(
        model_id=model_id,
        family=family,
        complete_fn=lambda prompt: output,
    )


def _bad_judge(
    model_id: str,
    family: str,
) -> Judge:
    def boom(prompt: str) -> JudgeOutput:
        raise RuntimeError("simulated parse failure")

    return Judge(
        model_id=model_id,
        family=family,
        complete_fn=boom,
    )


def _ctx(
    claim_id: str = "c1",
    route_with: str = "Review",
    route_without: Optional[str] = "Trust",
) -> ClaimContext:
    return ClaimContext(
        claim_id=claim_id,
        claim_text="45.2% of the ZCTA is in Zone AE.",
        tier1_label="verified",
        zcta="77001",
        scenario="houston_harvey",
        vlm_model_id="qwen",
        source_evidence="zone_ae_pct=45.2; population=32000; svi=0.82",
        route_with_claim=route_with,
        route_without_claim=route_without,
    )


def _rr(
    claim_id: str,
    a: Optional[Tier2Label],
    b: Optional[Tier2Label],
    fam_a: str = "google",
    fam_b: str = "openai",
    model_a: str = "gemini-2.5-pro",
    model_b: str = "gpt-4o",
    ablation_available: bool = True,
) -> ReliabilityRecord:
    return ReliabilityRecord(
        claim_id=claim_id,
        judge_a_label=a,
        judge_b_label=b,
        judge_a_model_id=model_a,
        judge_b_model_id=model_b,
        judge_a_family=fam_a,
        judge_b_family=fam_b,
        judge_a_axis="risk_magnitude" if a == "R_claim" else "none" if a else None,
        judge_b_axis="risk_magnitude" if b == "R_claim" else "none" if b else None,
        ablation_available=ablation_available,
    )


if __name__ == "__main__":
    # Alpha: perfect agreement
    units = [["R_claim", "R_claim"]] * 5 + [["S_sup_claim", "S_sup_claim"]] * 5
    assert math.isclose(krippendorff_alpha_nominal(units), 1.0)

    # Alpha: balanced disagreement should be below chance
    units = [["R_claim", "S_sup_claim"]] * 5 + [["S_sup_claim", "R_claim"]] * 5
    assert krippendorff_alpha_nominal(units) < 0

    # Alpha: hand-check 0.62
    units = [["A", "A"]] * 4 + [["B", "B"]] * 4 + [["A", "B"], ["B", "A"]]
    assert math.isclose(krippendorff_alpha_nominal(units), 0.62, abs_tol=1e-6)

    # Missing is dropped, not treated as a category
    base = [["R_claim", "R_claim"]] * 6 + [["S_sup_claim", "S_sup_claim"]] * 4
    with_missing = base + [["R_claim", None]]
    assert math.isclose(
        krippendorff_alpha_nominal(base),
        krippendorff_alpha_nominal(with_missing),
    )

    # parse_judge_json normalizes string null and coerces axis to none
    parsed = parse_judge_json(
        '{"label": "null", "decision_axis": "risk_magnitude", '
        '"rationale_code": "cannot_determine"}'
    )
    assert parsed.label is None
    assert parsed.decision_axis == "none"

    # safe_judge converts exceptions to abstention
    bad = _bad_judge("bad-model", "bad-family")
    out = safe_judge(bad, _ctx())
    assert out.label is None
    assert out.decision_axis == "none"

    # ClaimContext rejects fabricated / N labels at runtime
    try:
        ClaimContext(
            claim_id="bad",
            claim_text="fake",
            tier1_label="fabricated",  # type: ignore[arg-type]
            zcta="77001",
            scenario="houston",
            vlm_model_id="qwen",
            source_evidence="x",
            route_with_claim="Review",
            route_without_claim="Trust",
        )
        raise AssertionError("fabricated tier1 label reached Tier 2")
    except ValueError:
        pass

    # A/B agreement accepted
    ja = _stub_judge(
        "gemini-2.5-pro",
        "google",
        JudgeOutput("R_claim", "risk_magnitude", "changes_route"),
    )

    jb = _stub_judge(
        "gpt-4o",
        "openai",
        JudgeOutput("R_claim", "risk_magnitude", "changes_route"),
    )

    result = adjudicate_claim(_ctx(), ja, jb, None)
    assert result.production.final_label == "R_claim"
    assert result.production.adjudication_status == "accepted_ab"
    assert result.reliability.judge_a_family == "google"
    assert result.reliability.ablation_available is True

    # Dual abstain unresolved
    ja_null = _stub_judge(
        "gemini-2.5-pro",
        "google",
        JudgeOutput(None, "none", "cannot_determine"),
    )

    jb_null = _stub_judge(
        "gpt-4o",
        "openai",
        JudgeOutput(None, "none", "cannot_determine"),
    )

    result = adjudicate_claim(_ctx(), ja_null, jb_null, None)
    assert result.production.final_label == "unresolved"
    assert result.production.adjudication_status == "unresolved_dual_abstain"

    # A/B disagreement resolved by C
    jb_s = _stub_judge(
        "gpt-4o",
        "openai",
        JudgeOutput("S_sup_claim", "none", "decorative"),
    )

    jc = _stub_judge(
        "claude-opus",
        "anthropic",
        JudgeOutput("R_claim", "risk_magnitude", "route_changes"),
    )

    result = adjudicate_claim(_ctx(), ja, jb_s, jc)
    assert result.production.final_label == "R_claim"
    assert result.production.adjudication_status == "adjudicated_by_c"

    # A/B disagreement with C abstain unresolved
    jc_null = _stub_judge(
        "claude-opus",
        "anthropic",
        JudgeOutput(None, "none", "cannot_determine"),
    )

    result = adjudicate_claim(_ctx(), ja, jb_s, jc_null)
    assert result.production.final_label == "unresolved"
    assert result.production.adjudication_status == "unresolved_c_abstain"

    # Provenance guard derives from records, not Judge objects
    same_family_records = [
        _rr("c1", "R_claim", "R_claim", fam_a="google", fam_b="google"),
        _rr("c2", "S_sup_claim", "S_sup_claim", fam_a="google", fam_b="google"),
    ]

    try:
        calibration_gate(
            same_family_records,
            min_pairable=2,
            bootstrap_iterations=50,
        )
        raise AssertionError("same-family gate did not fail from record provenance")
    except ValueError:
        pass

    # Inconsistent provenance is rejected
    inconsistent = [
        _rr("c1", "R_claim", "R_claim", fam_a="google", fam_b="openai"),
        _rr("c2", "S_sup_claim", "S_sup_claim", fam_a="anthropic", fam_b="openai"),
    ]

    try:
        calibration_gate(
            inconsistent,
            min_pairable=2,
            bootstrap_iterations=50,
        )
        raise AssertionError("inconsistent provenance did not fail")
    except ValueError:
        pass

    # Ablation required by default for calibration
    no_ablation = [
        _rr("c1", "R_claim", "R_claim", ablation_available=False),
        _rr("c2", "S_sup_claim", "S_sup_claim", ablation_available=False),
    ]

    try:
        calibration_gate(
            no_ablation,
            min_pairable=2,
            bootstrap_iterations=50,
        )
        raise AssertionError("missing ablation did not fail")
    except ValueError:
        pass

    # Pairable floor blocks tiny samples even with perfect agreement
    tiny = [
        _rr("c1", "R_claim", "R_claim"),
        _rr("c2", "S_sup_claim", "S_sup_claim"),
    ]

    gate = calibration_gate(
        tiny,
        min_pairable=3,
        bootstrap_iterations=50,
    )
    assert not gate.passed
    assert gate.n_pairable == 2

    # Larger perfect calibration passes when floor and lower CI clear
    enough = (
        [_rr(f"r{i}", "R_claim", "R_claim") for i in range(60)]
        + [_rr(f"s{i}", "S_sup_claim", "S_sup_claim") for i in range(60)]
    )

    gate = calibration_gate(
        enough,
        min_pairable=100,
        bootstrap_iterations=100,
    )
    assert gate.passed
    assert math.isclose(gate.alpha, 1.0)
    assert math.isclose(gate.alpha_bootstrap_lower, 1.0)

    updated = apply_gate_to_production([result.production], gate)
    assert updated[0].trusted_after_alpha_gate is True

    summary = summarize_tier2(updated, gate)
    assert summary.tier2_status == "EXPLORATORY"
    assert summary.axis_distribution_status == "DESCRIPTIVE_ONLY"

    print("rubric_hash:", rubric_hash())
    print("all self-tests passed")
