"""
stress_mast.py - MAST Failure Taxonomy Stress Testing via YRSN Signals

Maps MAST (Multi-Agent System Failure Taxonomy) from NeurIPS 2025
to YRSN collapse detection for stress testing multi-agent systems.

References:
    - MAST Paper: "Why Do Multi-Agent LLM Systems Fail?" (arXiv:2503.13657)
    - GitHub: https://github.com/multi-agent-systems-failure-taxonomy/MAST
    - UC Berkeley Sky Lab: https://sky.cs.berkeley.edu/project/mast/

MAST 14 Failure Modes → YRSN 16 Collapse Types Mapping:

    FC1: Specification & System Design
    ├── FM-1.1 Disobey Task Spec     → DISTRACTION, O_POISONING
    ├── FM-1.2 Disobey Role Spec     → DISTRACTION
    ├── FM-1.3 Step Repetition       → MODE_COLLAPSE
    ├── FM-1.4 Loss of History       → DRIFT, DISTRIBUTIONAL_SHIFT
    └── FM-1.5 Unaware Termination   → CONFLICT, OVERCONFIDENCE

    FC2: Inter-Agent Misalignment
    ├── FM-2.1 Conversation Reset    → DRIFT, RSN_COLLAPSE
    ├── FM-2.2 Fail to Clarify       → OVERCONFIDENCE, ALEATORIC_DOMINANCE
    ├── FM-2.3 Task Derailment       → DISTRACTION, CONFLICT
    ├── FM-2.4 Info Withholding      → CLASH
    ├── FM-2.5 Ignored Input         → POISONING, CLASH
    └── FM-2.6 Reasoning Mismatch    → HALLUCINATION

    FC3: Task Verification & Termination
    ├── FM-3.1 Premature Termination → EPISTEMIC_SPIKE
    ├── FM-3.2 Incomplete Verify     → OVERCONFIDENCE
    └── FM-3.3 Incorrect Verify      → HALLUCINATION

Usage:
    from yrsn.framework.apps import stress_mast_app

    # Run specific MAST failure scenario
    result = stress_mast_app.run("inject_mast_failure", failure_mode="FM-2.3")

    # Detect MAST failure from agent trace
    diagnosis = stress_mast_app.run("diagnose_mast_failure", r=0.2, s=0.6, n=0.2)

    # Run full stress test suite
    report = stress_mast_app.run("run_mast_stress_suite")
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from yrsn.app import YRSNApp
from yrsn.utils.tools import ToolCategory


class MASTCategory(str, Enum):
    """MAST failure categories (3 top-level)."""

    FC1_SPECIFICATION = "FC1"  # Specification and System Design
    FC2_MISALIGNMENT = "FC2"   # Inter-Agent Misalignment
    FC3_VERIFICATION = "FC3"   # Task Verification and Termination


class MASTFailureMode(str, Enum):
    """MAST 14 failure modes."""

    # FC1: Specification and System Design Failures
    FM_1_1_DISOBEY_TASK = "FM-1.1"
    FM_1_2_DISOBEY_ROLE = "FM-1.2"
    FM_1_3_STEP_REPETITION = "FM-1.3"
    FM_1_4_LOSS_HISTORY = "FM-1.4"
    FM_1_5_UNAWARE_TERMINATION = "FM-1.5"

    # FC2: Inter-Agent Misalignment
    FM_2_1_CONVERSATION_RESET = "FM-2.1"
    FM_2_2_FAIL_CLARIFY = "FM-2.2"
    FM_2_3_TASK_DERAILMENT = "FM-2.3"
    FM_2_4_INFO_WITHHOLDING = "FM-2.4"
    FM_2_5_IGNORED_INPUT = "FM-2.5"
    FM_2_6_REASONING_MISMATCH = "FM-2.6"

    # FC3: Task Verification and Termination
    FM_3_1_PREMATURE_TERMINATION = "FM-3.1"
    FM_3_2_INCOMPLETE_VERIFY = "FM-3.2"
    FM_3_3_INCORRECT_VERIFY = "FM-3.3"


@dataclass
class MASTFailureSpec:
    """Specification for a MAST failure mode and its YRSN mapping."""

    mode: MASTFailureMode
    category: MASTCategory
    name: str
    description: str
    yrsn_collapse_types: List[str]  # Primary YRSN collapse types
    yrsn_signal_pattern: Dict[str, Tuple[float, float]]  # R/S/N/omega ranges
    detection_priority: int  # 1=highest priority
    example_scenario: str


# Complete MAST → YRSN Mapping
MAST_FAILURE_SPECS: Dict[MASTFailureMode, MASTFailureSpec] = {
    # ==========================================================================
    # FC1: Specification and System Design Failures
    # ==========================================================================
    MASTFailureMode.FM_1_1_DISOBEY_TASK: MASTFailureSpec(
        mode=MASTFailureMode.FM_1_1_DISOBEY_TASK,
        category=MASTCategory.FC1_SPECIFICATION,
        name="Disobey Task Specification",
        description="Agent fails to adhere to specified constraints, producing outcomes that don't meet task goals",
        yrsn_collapse_types=["DISTRACTION", "O_POISONING"],
        yrsn_signal_pattern={
            "R": (0.1, 0.3),   # Low relevance (off-task)
            "S": (0.4, 0.7),   # High superfluous (wrong content)
            "N": (0.1, 0.3),   # Variable noise
            "omega": (0.3, 0.6),  # Moderate reliability (thinks it's right)
        },
        detection_priority=1,
        example_scenario="Agent asked to write Python but produces JavaScript",
    ),
    MASTFailureMode.FM_1_2_DISOBEY_ROLE: MASTFailureSpec(
        mode=MASTFailureMode.FM_1_2_DISOBEY_ROLE,
        category=MASTCategory.FC1_SPECIFICATION,
        name="Disobey Role Specification",
        description="Agent oversteps assigned responsibilities, behaves like other roles",
        yrsn_collapse_types=["DISTRACTION"],
        yrsn_signal_pattern={
            "R": (0.2, 0.4),   # Partial relevance
            "S": (0.5, 0.8),   # High superfluous (wrong role behavior)
            "N": (0.0, 0.2),   # Low noise
            "omega": (0.5, 0.8),  # High reliability (confident in wrong role)
        },
        detection_priority=2,
        example_scenario="Reviewer agent starts implementing instead of reviewing",
    ),
    MASTFailureMode.FM_1_3_STEP_REPETITION: MASTFailureSpec(
        mode=MASTFailureMode.FM_1_3_STEP_REPETITION,
        category=MASTCategory.FC1_SPECIFICATION,
        name="Step Repetition",
        description="Agent unnecessarily reiterates completed steps, wasting resources",
        yrsn_collapse_types=["MODE_COLLAPSE"],
        yrsn_signal_pattern={
            "R": (0.3, 0.5),   # Moderate R (content is valid but repeated)
            "S": (0.3, 0.5),   # Moderate S (redundant)
            "N": (0.1, 0.2),   # Low N
            "omega": (0.6, 0.9),  # High omega (in-distribution but stuck)
        },
        detection_priority=3,
        example_scenario="Agent re-runs completed unit tests in infinite loop",
    ),
    MASTFailureMode.FM_1_4_LOSS_HISTORY: MASTFailureSpec(
        mode=MASTFailureMode.FM_1_4_LOSS_HISTORY,
        category=MASTCategory.FC1_SPECIFICATION,
        name="Loss of Conversation History",
        description="Context truncation causes agent to disregard recent interactions",
        yrsn_collapse_types=["DRIFT", "DISTRIBUTIONAL_SHIFT"],
        yrsn_signal_pattern={
            "R": (0.2, 0.4),   # Dropping R (losing context)
            "S": (0.3, 0.5),   # Variable S
            "N": (0.2, 0.4),   # Rising N (old/stale info)
            "omega": (0.2, 0.5),  # Low omega (distribution shift)
        },
        detection_priority=1,
        example_scenario="Agent forgets user requirements from 5 turns ago",
    ),
    MASTFailureMode.FM_1_5_UNAWARE_TERMINATION: MASTFailureSpec(
        mode=MASTFailureMode.FM_1_5_UNAWARE_TERMINATION,
        category=MASTCategory.FC1_SPECIFICATION,
        name="Unaware of Termination Conditions",
        description="Agent fails to recognize criteria for stopping, continues unnecessarily",
        yrsn_collapse_types=["CONFLICT", "OVERCONFIDENCE"],
        yrsn_signal_pattern={
            "R": (0.3, 0.5),   # Moderate R
            "S": (0.3, 0.5),   # High S (unnecessary continuation)
            "N": (0.1, 0.3),   # Variable N
            "omega": (0.6, 0.9),  # High omega (thinks it should continue)
        },
        detection_priority=2,
        example_scenario="Agent keeps optimizing after target metric achieved",
    ),

    # ==========================================================================
    # FC2: Inter-Agent Misalignment
    # ==========================================================================
    MASTFailureMode.FM_2_1_CONVERSATION_RESET: MASTFailureSpec(
        mode=MASTFailureMode.FM_2_1_CONVERSATION_RESET,
        category=MASTCategory.FC2_MISALIGNMENT,
        name="Conversation Reset",
        description="Unwarranted dialogue restarts, loss of context and progress",
        yrsn_collapse_types=["DRIFT", "RSN_COLLAPSE"],
        yrsn_signal_pattern={
            "R": (0.25, 0.40),  # Near RSN_COLLAPSE territory
            "S": (0.25, 0.40),
            "N": (0.25, 0.40),
            "omega": (0.1, 0.4),  # Low omega (major distribution shift)
        },
        detection_priority=1,
        example_scenario="Mid-conversation agent restarts with 'Hello, how can I help?'",
    ),
    MASTFailureMode.FM_2_2_FAIL_CLARIFY: MASTFailureSpec(
        mode=MASTFailureMode.FM_2_2_FAIL_CLARIFY,
        category=MASTCategory.FC2_MISALIGNMENT,
        name="Fail to Ask for Clarification",
        description="Agent cannot request additional info when facing unclear data",
        yrsn_collapse_types=["OVERCONFIDENCE", "ALEATORIC_DOMINANCE"],
        yrsn_signal_pattern={
            "R": (0.3, 0.5),   # Moderate R (partial understanding)
            "S": (0.2, 0.4),   # Moderate S
            "N": (0.2, 0.4),   # Elevated N (ambiguity)
            "omega": (0.6, 0.9),  # HIGH omega (overconfident despite ambiguity)
        },
        detection_priority=2,
        example_scenario="Agent guesses file format instead of asking user",
    ),
    MASTFailureMode.FM_2_3_TASK_DERAILMENT: MASTFailureSpec(
        mode=MASTFailureMode.FM_2_3_TASK_DERAILMENT,
        category=MASTCategory.FC2_MISALIGNMENT,
        name="Task Derailment",
        description="Agent deviates from intended objectives, pursues irrelevant directions",
        yrsn_collapse_types=["DISTRACTION", "CONFLICT"],
        yrsn_signal_pattern={
            "R": (0.1, 0.3),   # LOW R (off-task)
            "S": (0.5, 0.8),   # HIGH S (rabbit-holing)
            "N": (0.1, 0.3),   # Variable N
            "omega": (0.4, 0.7),  # Moderate omega
        },
        detection_priority=1,
        example_scenario="Agent discussing API design when asked to fix a typo",
    ),
    MASTFailureMode.FM_2_4_INFO_WITHHOLDING: MASTFailureSpec(
        mode=MASTFailureMode.FM_2_4_INFO_WITHHOLDING,
        category=MASTCategory.FC2_MISALIGNMENT,
        name="Information Withholding",
        description="Agent has crucial data but fails to share with collaborators",
        yrsn_collapse_types=["CLASH"],
        yrsn_signal_pattern={
            "R": (0.4, 0.6),   # Moderate R (has info)
            "S": (0.3, 0.5),   # Variable S
            "N": (0.1, 0.3),   # Low N
            "omega": (0.5, 0.8),  # High omega (info is valid but not shared)
            "S_variance": (0.2, 0.4),  # HIGH variance across agents
        },
        detection_priority=2,
        example_scenario="Planner agent doesn't share constraints with coder agent",
    ),
    MASTFailureMode.FM_2_5_IGNORED_INPUT: MASTFailureSpec(
        mode=MASTFailureMode.FM_2_5_IGNORED_INPUT,
        category=MASTCategory.FC2_MISALIGNMENT,
        name="Ignored Other Agent's Input",
        description="Agent disregards recommendations from collaborating agents",
        yrsn_collapse_types=["POISONING", "CLASH"],
        yrsn_signal_pattern={
            "R": (0.3, 0.5),   # Reduced R (missing input)
            "S": (0.2, 0.4),   # Variable S
            "N": (0.2, 0.5),   # HIGH N (ignored input = noise)
            "omega": (0.4, 0.7),
            "S_variance": (0.15, 0.35),  # Source disagreement
        },
        detection_priority=2,
        example_scenario="Coder ignores reviewer's security concern feedback",
    ),
    MASTFailureMode.FM_2_6_REASONING_MISMATCH: MASTFailureSpec(
        mode=MASTFailureMode.FM_2_6_REASONING_MISMATCH,
        category=MASTCategory.FC2_MISALIGNMENT,
        name="Reasoning-Action Mismatch",
        description="Discrepancies between logical reasoning and actual actions",
        yrsn_collapse_types=["HALLUCINATION"],
        yrsn_signal_pattern={
            "R": (0.5, 0.8),   # HIGH R (reasoning looks good)
            "S": (0.1, 0.3),   # Low S
            "N": (0.1, 0.3),   # Low N
            "omega": (0.1, 0.4),  # LOW omega (action doesn't match)
        },
        detection_priority=1,
        example_scenario="Agent says 'I will use pytest' but runs unittest",
    ),

    # ==========================================================================
    # FC3: Task Verification and Termination
    # ==========================================================================
    MASTFailureMode.FM_3_1_PREMATURE_TERMINATION: MASTFailureSpec(
        mode=MASTFailureMode.FM_3_1_PREMATURE_TERMINATION,
        category=MASTCategory.FC3_VERIFICATION,
        name="Premature Termination",
        description="Task ends before objectives are met, yielding incomplete outcomes",
        yrsn_collapse_types=["EPISTEMIC_SPIKE"],
        yrsn_signal_pattern={
            "R": (0.3, 0.5),   # Incomplete R
            "S": (0.2, 0.4),   # Some S
            "N": (0.2, 0.4),   # Variable N
            "omega": (0.3, 0.6),  # Dropping omega
            "epistemic": (0.4, 0.8),  # HIGH epistemic (should continue)
        },
        detection_priority=1,
        example_scenario="Agent stops after implementing 2 of 5 required functions",
    ),
    MASTFailureMode.FM_3_2_INCOMPLETE_VERIFY: MASTFailureSpec(
        mode=MASTFailureMode.FM_3_2_INCOMPLETE_VERIFY,
        category=MASTCategory.FC3_VERIFICATION,
        name="No or Incomplete Verification",
        description="Proper checking of task outcomes is partially or fully omitted",
        yrsn_collapse_types=["OVERCONFIDENCE"],
        yrsn_signal_pattern={
            "R": (0.4, 0.7),   # Moderate-high R (work done)
            "S": (0.2, 0.4),   # Some S
            "N": (0.1, 0.3),   # Variable N (uncaught errors)
            "omega": (0.7, 1.0),  # HIGH omega (overconfident)
            "epistemic": (0.0, 0.2),  # LOW epistemic (should be higher)
        },
        detection_priority=2,
        example_scenario="Agent claims code works but didn't run tests",
    ),
    MASTFailureMode.FM_3_3_INCORRECT_VERIFY: MASTFailureSpec(
        mode=MASTFailureMode.FM_3_3_INCORRECT_VERIFY,
        category=MASTCategory.FC3_VERIFICATION,
        name="Incorrect Verification",
        description="Inadequate validation permits errors to persist undetected",
        yrsn_collapse_types=["HALLUCINATION"],
        yrsn_signal_pattern={
            "R": (0.5, 0.8),   # HIGH R (thinks it verified)
            "S": (0.1, 0.3),   # Low S
            "N": (0.1, 0.3),   # Hidden N (undetected errors)
            "omega": (0.1, 0.4),  # LOW omega (verification unreliable)
        },
        detection_priority=1,
        example_scenario="Agent says 'all tests pass' when tests actually fail",
    ),
}


@dataclass
class MASTDiagnosis:
    """Diagnosis result mapping YRSN signals to MAST failure modes."""

    detected_collapse: str
    likely_mast_failures: List[Tuple[MASTFailureMode, float]]  # (mode, confidence)
    yrsn_metrics: Dict[str, float]
    recommendations: List[str]
    severity: str

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "detected_collapse": self.detected_collapse,
            "likely_mast_failures": [
                {"mode": m.value, "confidence": c}
                for m, c in self.likely_mast_failures
            ],
            "yrsn_metrics": self.yrsn_metrics,
            "recommendations": self.recommendations,
            "severity": self.severity,
        }


@dataclass
class MASTStressScenario:
    """A stress test scenario that triggers specific MAST failure modes."""

    failure_mode: MASTFailureMode
    input_signals: Dict[str, float]  # R, S, N, omega, etc.
    expected_collapse: str
    description: str


def create_stress_mast_app() -> YRSNApp:
    """Create the MAST stress testing app."""
    app = YRSNApp(
        name="yrsn-stress-mast",
        version="1.0.0",
        description="MAST Failure Taxonomy Stress Testing via YRSN Signals",
    )

    @app.tool(
        name="diagnose_mast_failure",
        description="Diagnose potential MAST failure modes from YRSN signals",
        category=ToolCategory.QUALITY,
        tags=["mast", "diagnosis", "multi-agent"],
    )
    def diagnose_mast_failure(
        r: float,
        s: float,
        n: float,
        omega: float = 0.5,
        epistemic: Optional[float] = None,
        s_variance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Diagnose potential MAST failure modes from YRSN signal pattern.

        Args:
            r: Relevant signal ratio [0, 1]
            s: Superfluous signal ratio [0, 1]
            n: Noise signal ratio [0, 1]
            omega: Reliability weight [0, 1]
            epistemic: Epistemic uncertainty (optional)
            s_variance: S variance across agents (optional)

        Returns:
            MASTDiagnosis with likely failure modes and recommendations
        """
        from yrsn.core.decomposition.collapse import detect_collapse

        # Get YRSN collapse type
        analysis = detect_collapse(r, s, n)
        collapse_type = analysis.collapse_type.value

        # Score each MAST failure mode
        candidates: List[Tuple[MASTFailureMode, float]] = []

        for mode, spec in MAST_FAILURE_SPECS.items():
            score = _score_mast_match(
                r, s, n, omega, epistemic, s_variance, spec
            )
            if score > 0.3:  # Threshold for relevance
                candidates.append((mode, score))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:3]

        # Generate recommendations
        recommendations = []
        if top_candidates:
            top_mode = top_candidates[0][0]
            spec = MAST_FAILURE_SPECS[top_mode]
            recommendations = _get_recommendations(spec, r, s, n, omega)

        diagnosis = MASTDiagnosis(
            detected_collapse=collapse_type,
            likely_mast_failures=top_candidates,
            yrsn_metrics={
                "R": r, "S": s, "N": n,
                "omega": omega,
                "alpha": r / (r + s + n) if (r + s + n) > 0 else 0,
            },
            recommendations=recommendations,
            severity=analysis.severity.value,
        )

        return diagnosis.to_dict()

    @app.tool(
        name="inject_mast_failure",
        description="Generate YRSN signals that would trigger a specific MAST failure",
        category=ToolCategory.QUALITY,
        tags=["mast", "injection", "stress-test"],
    )
    def inject_mast_failure(
        failure_mode: str,
        intensity: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate YRSN signal patterns that simulate a MAST failure mode.

        Used for stress testing multi-agent systems.

        Args:
            failure_mode: MAST failure mode (e.g., "FM-2.3" or "TASK_DERAILMENT")
            intensity: How severe the failure (0.0-1.0)

        Returns:
            Dict with R/S/N/omega values and expected collapse type
        """
        import random

        # Resolve failure mode
        mode = _resolve_failure_mode(failure_mode)
        if mode is None:
            return {"error": f"Unknown failure mode: {failure_mode}"}

        spec = MAST_FAILURE_SPECS[mode]

        # Generate signals within the spec's range, biased by intensity
        def sample_range(range_tuple: Tuple[float, float]) -> float:
            low, high = range_tuple
            # Intensity pushes toward more extreme (failure-like) values
            mid = (low + high) / 2
            return mid + (high - mid) * intensity * random.uniform(0.8, 1.0)

        signals = {
            "R": sample_range(spec.yrsn_signal_pattern["R"]),
            "S": sample_range(spec.yrsn_signal_pattern["S"]),
            "N": sample_range(spec.yrsn_signal_pattern["N"]),
            "omega": sample_range(spec.yrsn_signal_pattern["omega"]),
        }

        # Normalize R/S/N to sum to 1
        total = signals["R"] + signals["S"] + signals["N"]
        if total > 0:
            signals["R"] /= total
            signals["S"] /= total
            signals["N"] /= total

        return {
            "failure_mode": mode.value,
            "failure_name": spec.name,
            "category": spec.category.value,
            "signals": signals,
            "expected_collapse": spec.yrsn_collapse_types[0],
            "intensity": intensity,
            "description": spec.description,
            "example": spec.example_scenario,
        }

    @app.tool(
        name="run_mast_stress_suite",
        description="Run full MAST failure taxonomy stress test suite",
        category=ToolCategory.QUALITY,
        tags=["mast", "stress-test", "suite"],
    )
    def run_mast_stress_suite(
        intensity: float = 0.7,
        categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Run stress tests for all MAST failure modes.

        Args:
            intensity: Failure intensity (0.0-1.0)
            categories: Filter to specific categories (FC1, FC2, FC3)

        Returns:
            Full stress test report with detection accuracy
        """
        from yrsn.core.decomposition.collapse import detect_collapse

        results = []
        correct_detections = 0
        total_tests = 0

        for mode, spec in MAST_FAILURE_SPECS.items():
            # Filter by category if specified
            if categories and spec.category.value not in categories:
                continue

            # Inject the failure
            injection = inject_mast_failure(mode.value, intensity)
            signals = injection["signals"]

            # Detect collapse
            analysis = detect_collapse(
                signals["R"], signals["S"], signals["N"]
            )

            # Check if detected collapse matches expected
            detected = analysis.collapse_type.value
            expected = spec.yrsn_collapse_types
            match = detected in expected

            if match:
                correct_detections += 1
            total_tests += 1

            results.append({
                "failure_mode": mode.value,
                "failure_name": spec.name,
                "category": spec.category.value,
                "injected_signals": signals,
                "expected_collapse": expected,
                "detected_collapse": detected,
                "match": match,
                "severity": analysis.severity.value,
            })

        accuracy = correct_detections / total_tests if total_tests > 0 else 0

        return {
            "total_tests": total_tests,
            "correct_detections": correct_detections,
            "accuracy": round(accuracy, 3),
            "intensity": intensity,
            "results": results,
            "summary": {
                "FC1_results": len([r for r in results if r["category"] == "FC1"]),
                "FC2_results": len([r for r in results if r["category"] == "FC2"]),
                "FC3_results": len([r for r in results if r["category"] == "FC3"]),
            },
        }

    @app.tool(
        name="get_mast_taxonomy",
        description="Get the complete MAST failure taxonomy with YRSN mappings",
        category=ToolCategory.QUALITY,
        tags=["mast", "taxonomy", "reference"],
    )
    def get_mast_taxonomy() -> Dict[str, Any]:
        """
        Get the complete MAST failure taxonomy.

        Returns:
            Full taxonomy with all 14 failure modes and YRSN mappings
        """
        taxonomy = {
            "version": "1.0.0",
            "source": "arXiv:2503.13657",
            "categories": {
                "FC1": {
                    "name": "Specification and System Design",
                    "description": "Failures from poor system design, not model capability",
                    "modes": [],
                },
                "FC2": {
                    "name": "Inter-Agent Misalignment",
                    "description": "Failures in agent-to-agent communication and coordination",
                    "modes": [],
                },
                "FC3": {
                    "name": "Task Verification and Termination",
                    "description": "Failures in validating and completing work",
                    "modes": [],
                },
            },
        }

        for mode, spec in MAST_FAILURE_SPECS.items():
            cat = spec.category.value
            taxonomy["categories"][cat]["modes"].append({
                "id": mode.value,
                "name": spec.name,
                "description": spec.description,
                "yrsn_collapse_types": spec.yrsn_collapse_types,
                "example": spec.example_scenario,
                "priority": spec.detection_priority,
            })

        return taxonomy

    return app


def _score_mast_match(
    r: float,
    s: float,
    n: float,
    omega: float,
    epistemic: Optional[float],
    s_variance: Optional[float],
    spec: MASTFailureSpec,
) -> float:
    """Score how well YRSN signals match a MAST failure spec."""
    score = 0.0
    count = 0

    def in_range(value: float, range_tuple: Tuple[float, float]) -> float:
        low, high = range_tuple
        if low <= value <= high:
            return 1.0
        elif value < low:
            return max(0, 1 - (low - value) * 2)
        else:
            return max(0, 1 - (value - high) * 2)

    # Score R/S/N/omega
    for key, value in [("R", r), ("S", s), ("N", n), ("omega", omega)]:
        if key in spec.yrsn_signal_pattern:
            score += in_range(value, spec.yrsn_signal_pattern[key])
            count += 1

    # Score optional signals
    if epistemic is not None and "epistemic" in spec.yrsn_signal_pattern:
        score += in_range(epistemic, spec.yrsn_signal_pattern["epistemic"])
        count += 1

    if s_variance is not None and "S_variance" in spec.yrsn_signal_pattern:
        score += in_range(s_variance, spec.yrsn_signal_pattern["S_variance"])
        count += 1

    return score / count if count > 0 else 0.0


def _resolve_failure_mode(mode_str: str) -> Optional[MASTFailureMode]:
    """Resolve a failure mode string to enum."""
    # Try direct match
    for mode in MASTFailureMode:
        if mode.value == mode_str or mode.name == mode_str:
            return mode

    # Try partial match
    mode_upper = mode_str.upper().replace("-", "_").replace(" ", "_")
    for mode in MASTFailureMode:
        if mode_upper in mode.name:
            return mode

    return None


def _get_recommendations(
    spec: MASTFailureSpec,
    r: float,
    s: float,
    n: float,
    omega: float,
) -> List[str]:
    """Generate recommendations based on detected failure."""
    recs = []

    if spec.category == MASTCategory.FC1_SPECIFICATION:
        recs.append("Review agent task/role specifications for clarity")
        if "MODE_COLLAPSE" in spec.yrsn_collapse_types:
            recs.append("Add loop detection and termination checks")
        if "DRIFT" in spec.yrsn_collapse_types:
            recs.append("Increase context window or add summarization")

    elif spec.category == MASTCategory.FC2_MISALIGNMENT:
        recs.append("Improve inter-agent communication protocols")
        if "CLASH" in spec.yrsn_collapse_types:
            recs.append("Add explicit state sharing between agents")
        if "HALLUCINATION" in spec.yrsn_collapse_types:
            recs.append("Add reasoning-action consistency checks")
        if r < 0.3:
            recs.append("Strengthen task grounding in agent prompts")

    elif spec.category == MASTCategory.FC3_VERIFICATION:
        recs.append("Add verification steps before task completion")
        if "OVERCONFIDENCE" in spec.yrsn_collapse_types:
            recs.append("Require explicit test execution, not just claims")
        if "EPISTEMIC_SPIKE" in spec.yrsn_collapse_types:
            recs.append("Check completion criteria before termination")

    return recs


# Create singleton instance
stress_mast_app = create_stress_mast_app()


__all__ = [
    "MASTCategory",
    "MASTFailureMode",
    "MASTFailureSpec",
    "MASTDiagnosis",
    "MAST_FAILURE_SPECS",
    "stress_mast_app",
    "create_stress_mast_app",
]
