"""Contract parser for EXPERIMENT_CONTRACT.yaml files."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import CellKey


class ContractParseError(Exception):
    """Raised when a contract YAML cannot be parsed or is invalid."""


@dataclass
class OutputSpec:
    """Specification for a single declared output."""

    key_template: str
    per_scenario: bool = False
    description: str = ""

    def resolve_keys(self, scenarios: list[str]) -> list[str]:
        """Expand {scenario} placeholders if per_scenario, else return as-is."""
        if self.per_scenario:
            return [
                self.key_template.replace("{scenario}", s) for s in scenarios
            ]
        return [self.key_template]


@dataclass
class ArtifactSpec:
    """Specification for an S3 artifact dependency."""

    key_template: str
    per_scenario: bool = False
    scenarios: list[str] | None = None
    optional: bool = False
    prefix: bool = False


@dataclass
class PhaseSpec:
    """Specification for a single experiment phase."""

    phase_id: str
    script: str
    per_scenario: bool
    packages: list[str]
    depends_on: list[str]
    declared_outputs: list[OutputSpec]
    s3_artifacts: list[ArtifactSpec]
    description: str = ""

    @property
    def has_declared_outputs(self) -> bool:
        """Return True if the phase has any declared outputs."""
        return len(self.declared_outputs) > 0

    def resolve_output_keys(self, scenarios: list[str]) -> list[str]:
        """Resolve all output key templates against the given scenarios."""
        keys: list[str] = []
        for output in self.declared_outputs:
            keys.extend(output.resolve_keys(scenarios))
        return keys


@dataclass
class ParsedContract:
    """Fully parsed experiment contract."""

    experiment: str
    bucket: str
    version: str
    scenarios: list[str]
    cell_matrix: frozenset[CellKey]
    targets: list[dict]
    phases: list[PhaseSpec]
    raw: dict

    def get_phase(self, phase_id: str) -> PhaseSpec | None:
        """Look up a phase by its ID."""
        for phase in self.phases:
            if phase.phase_id == phase_id:
                return phase
        return None

    def find_unresolved_templates(self) -> list[str]:
        """Find template variables that are not {scenario} or {scenario_key}.

        Returns a list of human-readable strings describing where
        unresolved templates were found.
        """
        allowed = {"{scenario}", "{scenario_key}"}
        pattern = re.compile(r"\{[a-zA-Z_][a-zA-Z0-9_]*\}")
        issues: list[str] = []

        for phase in self.phases:
            # Check declared outputs
            for output in phase.declared_outputs:
                for match in pattern.findall(output.key_template):
                    if match not in allowed:
                        issues.append(
                            f"phase={phase.phase_id} output key "
                            f"'{output.key_template}' has unresolved "
                            f"template var {match}"
                        )
            # Check s3 artifacts
            for artifact in phase.s3_artifacts:
                for match in pattern.findall(artifact.key_template):
                    if match not in allowed:
                        issues.append(
                            f"phase={phase.phase_id} s3_artifact key "
                            f"'{artifact.key_template}' has unresolved "
                            f"template var {match}"
                        )
        return issues


def _parse_outputs(raw_outputs: list[dict] | None) -> list[OutputSpec]:
    """Parse the outputs section of a phase."""
    if not raw_outputs:
        return []
    specs: list[OutputSpec] = []
    for entry in raw_outputs:
        specs.append(
            OutputSpec(
                key_template=entry["key"],
                per_scenario=entry.get("per_scenario", False),
                description=entry.get("description", ""),
            )
        )
    return specs


def _parse_artifacts(raw_artifacts: list[dict] | None) -> list[ArtifactSpec]:
    """Parse the s3_artifacts section of a phase."""
    if not raw_artifacts:
        return []
    specs: list[ArtifactSpec] = []
    for entry in raw_artifacts:
        scenarios_raw = entry.get("scenarios")
        scenarios = list(scenarios_raw) if scenarios_raw else None
        specs.append(
            ArtifactSpec(
                key_template=entry["key"],
                per_scenario=entry.get("per_scenario", False),
                scenarios=scenarios,
                optional=entry.get("optional", False),
                prefix=entry.get("prefix", False),
            )
        )
    return specs


def _parse_phase(raw_phase: dict) -> PhaseSpec:
    """Parse a single phase entry from the contract."""
    return PhaseSpec(
        phase_id=raw_phase["phase_id"],
        script=raw_phase["script"],
        per_scenario=raw_phase.get("per_scenario", False),
        packages=raw_phase.get("packages", []),
        depends_on=raw_phase.get("depends_on", []),
        declared_outputs=_parse_outputs(raw_phase.get("outputs")),
        s3_artifacts=_parse_artifacts(raw_phase.get("s3_artifacts")),
        description=raw_phase.get("description", ""),
    )


def _build_cell_matrix(targets: list[dict]) -> frozenset[CellKey]:
    """Build cell matrix as cross product of each target with its scenarios."""
    cells: set[CellKey] = set()
    for target in targets:
        column = target["column"]
        for scenario in target.get("scenarios", []):
            cells.add(CellKey(scenario=scenario, target=column))
    return frozenset(cells)


def _collect_scenarios(targets: list[dict]) -> list[str]:
    """Collect unique scenarios from targets, preserving first-seen order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for target in targets:
        for scenario in target.get("scenarios", []):
            if scenario not in seen:
                seen.add(scenario)
                ordered.append(scenario)
    return ordered


def parse_contract(path: str | Path) -> ParsedContract:
    """Read a YAML contract file and return a ParsedContract.

    Args:
        path: Path to the EXPERIMENT_CONTRACT.yaml file.

    Returns:
        A fully resolved ParsedContract instance.

    Raises:
        ContractParseError: If the YAML is missing required fields.
    """
    path = Path(path)
    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ContractParseError(f"Failed to read contract: {exc}") from exc

    if not isinstance(raw, dict):
        raise ContractParseError("Contract YAML must be a mapping at top level")

    for required in ("experiment", "bucket", "targets", "phases"):
        if required not in raw:
            raise ContractParseError(f"Missing required field: {required}")

    targets = raw["targets"]
    scenarios = _collect_scenarios(targets)
    cell_matrix = _build_cell_matrix(targets)
    phases = [_parse_phase(p) for p in raw["phases"]]

    return ParsedContract(
        experiment=raw["experiment"],
        bucket=raw["bucket"],
        version=str(raw.get("version", "")),
        scenarios=scenarios,
        cell_matrix=cell_matrix,
        targets=targets,
        phases=phases,
        raw=raw,
    )
