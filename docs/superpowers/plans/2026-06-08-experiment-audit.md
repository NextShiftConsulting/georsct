# Experiment Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a post-flight experiment audit tool that verifies contracted cells exist, metrics are non-null/non-degenerate, timestamp ordering is correct, and scope violations are flagged — producing a machine-readable audit report and completion manifest.

**Architecture:** Hybrid contract-driven audit. Phases with declared `outputs:` are audited authoritatively; phases without are audited from known S3 conventions but flagged as `WARN_CONTRACT_GAP`. All S3 access uses `swarm_auth`. Aggregate JSONs (geometry_kappa, diagnostics, certificates, money_table, verdicts) are content-inspected, not just existence-checked.

**Tech Stack:** Python 3.11+, pyyaml, boto3 (via swarm_auth), dataclasses, argparse, json

**Locked Design Decisions (from PAR review):**
- **D1 (Contract supplement):** Hybrid mode. Declared outputs = authoritative. Missing outputs = `WARN_CONTRACT_GAP`. Undeclared real artifacts = `WARN_UNDECLARED_ARTIFACT`.
- **D2 (Shared mappings):** Temporary duplicate in `constants.py` with `# TODO(post-submission)` comment. Do not touch `_coverage_common.py` or training scripts.
- **D3 (Timestamps):** Internal JSON timestamp primary, S3 `LastModified` fallback, missing = `WARN_TIMESTAMP_UNVERIFIABLE`. Never hard-fail on S3 `LastModified` alone.

---

## File Structure

```
georsct/experiment_audit/
  __init__.py              # Public API: run_audit()
  __main__.py              # python -m georsct.experiment_audit entry point
  cli.py                   # argparse CLI
  constants.py             # Temporary mirror of OUTPUT_KEYS, SCENARIO_KEYS, level prefixes
  contract.py              # Parse EXPERIMENT_CONTRACT.yaml, resolve cell matrix + templates
  s3_inventory.py          # List S3 artifacts, download JSONs, get timestamps
  content_readers.py       # Parse result JSONs: money_table, certificates, diagnostics, etc.
  validators.py            # 6 audit gates: contract parse, output existence, convention supplement,
                           #   cell matrix, aggregate content, ordering
  timestamp_ordering.py    # Three-tier timestamp logic + ordering rules
  models.py                # Dataclasses: AuditResult, CellKey, ArtifactRecord, CheckResult, Manifest
  report.py                # Generate JSON + Markdown audit reports
  manifest.py              # Write completion manifest (provenance chain)
tests/
  experiment_audit/
    __init__.py
    test_contract.py       # Contract parsing + cell matrix resolution
    test_constants.py      # Verify constants match _coverage_common.py
    test_content_readers.py # JSON schema parsing
    test_validators.py     # All 6 gates
    test_timestamp_ordering.py  # Three-tier logic
    test_models.py         # Severity enum, dataclass serialization
    test_report.py         # Report generation
    fixtures/
      mini_contract.yaml   # Minimal valid contract for testing
      sample_r0.json       # Sample R0 result with runs[]
      sample_money_table.json  # Sample with 8 cells (missing new_orleans/nfip)
      sample_geometry_kappa.json  # Sample with cells[] array
```

**Existing files read but NOT modified:**
- `wsp/floodrsct/exp/s035-model-ladder/EXPERIMENT_CONTRACT.yaml` (input)
- `wsp/floodrsct/jobs/validate_experiment_readiness.py` (reference only)
- `wsp/floodrsct/jobs/_coverage_common.py` (constants source, lines 30-46)
- `georsct/healthcheck/models.py` (pattern reference)

**Modified:**
- `pyproject.toml` — add `[project.optional-dependencies].audit`

---

## Severity Vocabulary

Every check result uses exactly one of these values:

```python
class Severity(str, Enum):
    PASS = "PASS"
    WARN_CONTRACT_GAP = "WARN_CONTRACT_GAP"
    WARN_UNDECLARED_ARTIFACT = "WARN_UNDECLARED_ARTIFACT"
    WARN_TIMESTAMP_FALLBACK = "WARN_TIMESTAMP_FALLBACK"
    WARN_TIMESTAMP_UNVERIFIABLE = "WARN_TIMESTAMP_UNVERIFIABLE"
    WARN_SCOPE_EXTRA = "WARN_SCOPE_EXTRA"
    FAIL_MISSING_OUTPUT = "FAIL_MISSING_OUTPUT"
    FAIL_UNRESOLVED_TEMPLATE = "FAIL_UNRESOLVED_TEMPLATE"
    FAIL_ORDERING_VIOLATION = "FAIL_ORDERING_VIOLATION"
    FAIL_CONTENT_INCOMPLETE = "FAIL_CONTENT_INCOMPLETE"
    FAIL_CONTENT_DEGENERATE = "FAIL_CONTENT_DEGENERATE"
```

---

## Task 1: Data Models (`models.py`)

**Files:**
- Create: `georsct/experiment_audit/models.py`
- Test: `tests/experiment_audit/test_models.py`

- [ ] **Step 1: Write the failing test for Severity enum**

```python
# tests/experiment_audit/test_models.py
"""Tests for experiment_audit data models."""
from __future__ import annotations

import json

from georsct.experiment_audit.models import (
    ArtifactRecord,
    AuditResult,
    CellKey,
    CheckResult,
    Severity,
)


def test_severity_is_fail():
    assert Severity.FAIL_MISSING_OUTPUT.is_fail()
    assert Severity.FAIL_ORDERING_VIOLATION.is_fail()
    assert not Severity.PASS.is_fail()
    assert not Severity.WARN_CONTRACT_GAP.is_fail()


def test_severity_is_warn():
    assert Severity.WARN_CONTRACT_GAP.is_warn()
    assert Severity.WARN_TIMESTAMP_FALLBACK.is_warn()
    assert not Severity.PASS.is_warn()
    assert not Severity.FAIL_MISSING_OUTPUT.is_warn()


def test_cell_key_str():
    ck = CellKey(scenario="houston", target="obs_nfip_event_claims")
    assert str(ck) == "houston/obs_nfip_event_claims"


def test_cell_key_from_string():
    ck = CellKey.from_string("houston/obs_nfip_event_claims")
    assert ck.scenario == "houston"
    assert ck.target == "obs_nfip_event_claims"


def test_check_result_serialization():
    cr = CheckResult(
        gate="contract_parse",
        name="template_resolution",
        severity=Severity.FAIL_UNRESOLVED_TEMPLATE,
        message="Unresolved {vlm} in r4_{vlm}_{scenario}.parquet",
        phase_id="r4_vlm_comparison",
    )
    d = cr.to_dict()
    assert d["severity"] == "FAIL_UNRESOLVED_TEMPLATE"
    assert d["gate"] == "contract_parse"
    # Round-trip through JSON
    assert json.loads(json.dumps(d)) == d


def test_artifact_record():
    ar = ArtifactRecord(
        s3_key="results/s035/r0_houston.json",
        exists=True,
        size_bytes=12345,
        last_modified="2026-06-06T12:00:00Z",
        internal_timestamp="2026-06-05T10:30:00Z",
    )
    assert ar.best_timestamp() == "2026-06-05T10:30:00Z"


def test_artifact_record_fallback_timestamp():
    ar = ArtifactRecord(
        s3_key="results/s035/geometry_kappa.json",
        exists=True,
        size_bytes=5000,
        last_modified="2026-06-04T08:00:00Z",
        internal_timestamp=None,
    )
    assert ar.best_timestamp() == "2026-06-04T08:00:00Z"


def test_artifact_record_no_timestamp():
    ar = ArtifactRecord(
        s3_key="results/s035/something.json",
        exists=False,
        size_bytes=None,
        last_modified=None,
        internal_timestamp=None,
    )
    assert ar.best_timestamp() is None


def test_audit_result_summary_counts():
    results = AuditResult(checks=[
        CheckResult("g1", "a", Severity.PASS, "ok"),
        CheckResult("g1", "b", Severity.FAIL_MISSING_OUTPUT, "missing"),
        CheckResult("g2", "c", Severity.WARN_CONTRACT_GAP, "gap"),
        CheckResult("g2", "d", Severity.WARN_CONTRACT_GAP, "gap2"),
    ])
    counts = results.summary_counts()
    assert counts["PASS"] == 1
    assert counts["FAIL"] == 1
    assert counts["WARN"] == 2


def test_audit_result_overall_pass():
    results = AuditResult(checks=[
        CheckResult("g1", "a", Severity.PASS, "ok"),
        CheckResult("g1", "b", Severity.WARN_CONTRACT_GAP, "gap"),
    ])
    assert results.overall_status() == "WARN"


def test_audit_result_overall_fail():
    results = AuditResult(checks=[
        CheckResult("g1", "a", Severity.PASS, "ok"),
        CheckResult("g1", "b", Severity.FAIL_MISSING_OUTPUT, "bad"),
    ])
    assert results.overall_status() == "FAIL"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'georsct.experiment_audit'`

- [ ] **Step 3: Create `__init__.py` files and implement `models.py`**

Create `georsct/experiment_audit/__init__.py`:
```python
"""Experiment audit -- post-flight contract verification for s035."""
```

Create `tests/experiment_audit/__init__.py`:
```python
```

Create `tests/experiment_audit/fixtures/` directory (empty for now).

Create `georsct/experiment_audit/models.py`:
```python
"""Core data models for experiment audit."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Audit check severity — see docs/superpowers/plans/2026-06-08-experiment-audit.md."""

    PASS = "PASS"
    WARN_CONTRACT_GAP = "WARN_CONTRACT_GAP"
    WARN_UNDECLARED_ARTIFACT = "WARN_UNDECLARED_ARTIFACT"
    WARN_TIMESTAMP_FALLBACK = "WARN_TIMESTAMP_FALLBACK"
    WARN_TIMESTAMP_UNVERIFIABLE = "WARN_TIMESTAMP_UNVERIFIABLE"
    WARN_SCOPE_EXTRA = "WARN_SCOPE_EXTRA"
    FAIL_MISSING_OUTPUT = "FAIL_MISSING_OUTPUT"
    FAIL_UNRESOLVED_TEMPLATE = "FAIL_UNRESOLVED_TEMPLATE"
    FAIL_ORDERING_VIOLATION = "FAIL_ORDERING_VIOLATION"
    FAIL_CONTENT_INCOMPLETE = "FAIL_CONTENT_INCOMPLETE"
    FAIL_CONTENT_DEGENERATE = "FAIL_CONTENT_DEGENERATE"

    def is_fail(self) -> bool:
        return self.value.startswith("FAIL_")

    def is_warn(self) -> bool:
        return self.value.startswith("WARN_")


@dataclass(frozen=True)
class CellKey:
    """A (scenario, target) pair — the fundamental audit unit."""

    scenario: str
    target: str

    def __str__(self) -> str:
        return f"{self.scenario}/{self.target}"

    @classmethod
    def from_string(cls, s: str) -> CellKey:
        scenario, target = s.split("/", 1)
        return cls(scenario=scenario, target=target)


@dataclass
class ArtifactRecord:
    """Metadata for a single S3 artifact."""

    s3_key: str
    exists: bool
    size_bytes: int | None = None
    last_modified: str | None = None
    internal_timestamp: str | None = None
    content: dict[str, Any] | None = None

    def best_timestamp(self) -> str | None:
        """Internal timestamp if available, else S3 LastModified."""
        return self.internal_timestamp or self.last_modified


@dataclass
class CheckResult:
    """One audit check result."""

    gate: str
    name: str
    severity: Severity
    message: str
    phase_id: str | None = None
    cell: CellKey | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "gate": self.gate,
            "name": self.name,
            "severity": self.severity.value,
            "message": self.message,
        }
        if self.phase_id is not None:
            d["phase_id"] = self.phase_id
        if self.cell is not None:
            d["cell"] = str(self.cell)
        return d


@dataclass
class AuditResult:
    """Collection of all check results from a full audit run."""

    checks: list[CheckResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_counts(self) -> dict[str, int]:
        counts = {"PASS": 0, "FAIL": 0, "WARN": 0}
        for c in self.checks:
            if c.severity == Severity.PASS:
                counts["PASS"] += 1
            elif c.severity.is_fail():
                counts["FAIL"] += 1
            elif c.severity.is_warn():
                counts["WARN"] += 1
        return counts

    def overall_status(self) -> str:
        if any(c.severity.is_fail() for c in self.checks):
            return "FAIL"
        if any(c.severity.is_warn() for c in self.checks):
            return "WARN"
        return "PASS"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_models.py -v`
Expected: All 10 tests PASS

- [ ] **Step 5: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add georsct/experiment_audit/__init__.py georsct/experiment_audit/models.py tests/experiment_audit/__init__.py tests/experiment_audit/test_models.py
git commit -m "feat(experiment_audit): add core data models and severity vocabulary"
```

---

## Task 2: Constants (`constants.py`)

**Files:**
- Create: `georsct/experiment_audit/constants.py`
- Read: `wsp/floodrsct/jobs/_coverage_common.py:30-46` (source of truth)
- Test: `tests/experiment_audit/test_constants.py`

- [ ] **Step 1: Read `_coverage_common.py` to get canonical values**

Run: `head -50 wsp/floodrsct/jobs/_coverage_common.py`

- [ ] **Step 2: Write the failing test**

```python
# tests/experiment_audit/test_constants.py
"""Verify constants mirror _coverage_common.py."""
from __future__ import annotations

from georsct.experiment_audit.constants import (
    BUCKET,
    LEVEL_PREFIXES,
    OUTPUT_KEYS,
    SCENARIO_KEYS,
    CONTRACTED_CELLS,
)


def test_output_keys_all_five_scenarios():
    assert set(OUTPUT_KEYS.keys()) == {
        "houston", "southwest_florida", "nyc",
        "riverside_coachella", "new_orleans",
    }


def test_scenario_keys_derived():
    # new_orleans -> "no", riverside_coachella -> "rc", etc.
    assert SCENARIO_KEYS["new_orleans"] == "no"
    assert SCENARIO_KEYS["riverside_coachella"] == "rc"
    assert SCENARIO_KEYS["houston"] == "houston"


def test_level_prefixes():
    assert LEVEL_PREFIXES["r0"] == "r0"
    assert LEVEL_PREFIXES["r1"] == "r1_hydrology"
    assert LEVEL_PREFIXES["r2"] == "r2"


def test_bucket():
    assert BUCKET == "swarm-floodrsct-data"


def test_contracted_cells_count():
    # 5 nfip + 2 has_311 + 2 has_hwm = 9
    assert len(CONTRACTED_CELLS) == 9


def test_contracted_cells_includes_new_orleans_nfip():
    from georsct.experiment_audit.models import CellKey
    assert CellKey("new_orleans", "obs_nfip_event_claims") in CONTRACTED_CELLS


def test_contracted_cells_excludes_swfl_hwm():
    from georsct.experiment_audit.models import CellKey
    assert CellKey("southwest_florida", "obs_has_hwm") not in CONTRACTED_CELLS
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_constants.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 4: Implement `constants.py`**

```python
# georsct/experiment_audit/constants.py
"""Shared constants for experiment audit.

TODO(post-submission): move to georsct.experiment_audit.constants as canonical
location. Other consumers (wsp/floodrsct/jobs/_coverage_common.py,
wsp/floodrsct/jobs/validate_experiment_readiness.py) should import from here.
Currently mirrors wsp/floodrsct/jobs/_coverage_common.py lines 30-46.
"""
from __future__ import annotations

from pathlib import Path

from .models import CellKey

BUCKET = "swarm-floodrsct-data"

# scenario -> assembled parquet S3 key
OUTPUT_KEYS: dict[str, str] = {
    "houston": "processed/houston/houston_event_features.parquet",
    "new_orleans": "processed/new_orleans/no_event_features.parquet",
    "nyc": "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
}

# scenario -> short key used in supplement filenames
SCENARIO_KEYS: dict[str, str] = {
    sc: Path(path).stem.replace("_event_features", "")
    for sc, path in OUTPUT_KEYS.items()
}

# logical level -> S3 prefix used in result filenames
LEVEL_PREFIXES: dict[str, str] = {
    "r0": "r0",
    "r1": "r1_hydrology",
    "r2": "r2",
    "r3": "r3",
}

# Known VLM providers used in R4 phases.
# The contract uses {vlm} template but does not enumerate providers.
# This is the canonical list until the contract YAML is fixed.
VLM_PROVIDERS: list[str] = [
    "claude_sonnet",
    "gemini_flash",
    "gpt4o",
    "gemini_pro",
    "qwen_vl",
    "claude_haiku",
]

# The contracted cell matrix: (scenario, target) pairs.
# Derived from EXPERIMENT_CONTRACT.yaml lines 44-66.
CONTRACTED_CELLS: frozenset[CellKey] = frozenset([
    # obs_nfip_event_claims: all 5 scenarios
    CellKey("houston", "obs_nfip_event_claims"),
    CellKey("southwest_florida", "obs_nfip_event_claims"),
    CellKey("nyc", "obs_nfip_event_claims"),
    CellKey("riverside_coachella", "obs_nfip_event_claims"),
    CellKey("new_orleans", "obs_nfip_event_claims"),
    # obs_has_311: houston, nyc
    CellKey("houston", "obs_has_311"),
    CellKey("nyc", "obs_has_311"),
    # obs_has_hwm: houston, new_orleans
    CellKey("houston", "obs_has_hwm"),
    CellKey("new_orleans", "obs_has_hwm"),
])

# Target metadata from the contract
TARGET_SPECS: dict[str, dict] = {
    "obs_nfip_event_claims": {
        "task": "regression",
        "metric": "r2",
        "scenarios": ["houston", "southwest_florida", "nyc", "riverside_coachella", "new_orleans"],
    },
    "obs_has_311": {
        "task": "binary_classification",
        "metric": "roc_auc",
        "scenarios": ["houston", "nyc"],
    },
    "obs_has_hwm": {
        "task": "binary_classification",
        "metric": "roc_auc",
        "scenarios": ["houston", "new_orleans"],
    },
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_constants.py -v`
Expected: All 7 tests PASS

- [ ] **Step 6: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add georsct/experiment_audit/constants.py tests/experiment_audit/test_constants.py
git commit -m "feat(experiment_audit): add constants mirroring _coverage_common.py"
```

---

## Task 3: Contract Parser (`contract.py`)

**Files:**
- Create: `georsct/experiment_audit/contract.py`
- Create: `tests/experiment_audit/fixtures/mini_contract.yaml`
- Test: `tests/experiment_audit/test_contract.py`

- [ ] **Step 1: Create the minimal test fixture**

```yaml
# tests/experiment_audit/fixtures/mini_contract.yaml
experiment: test-audit
bucket: test-bucket
version: "1.0"

modelable_scenarios: &modelable
  - houston
  - nyc

targets:
  - column: obs_nfip_event_claims
    task: regression
    metric: r2
    scenarios: [houston, nyc]
  - column: obs_has_311
    task: binary_classification
    metric: roc_auc
    scenarios: [houston]

phases:
  - phase_id: geometry_kappa
    script: compute_geometry_kappa.py
    per_scenario: false
    packages: [numpy]
    s3_artifacts:
      - key: assembled_parquet
        per_scenario: true
        scenarios: *modelable
    outputs:
      - key: "results/test/geometry_kappa.json"

  - phase_id: r0_baseline
    script: train_r0_baseline.py
    per_scenario: true
    packages: [numpy, pandas]
    s3_artifacts:
      - key: assembled_parquet
    outputs:
      - key: "results/test/r0_{scenario}.json"
        per_scenario: true
      - key: "results/test/r0_{scenario}_predictions.parquet"
        per_scenario: true

  - phase_id: diagnostics_r0
    script: compute_diagnostics.py
    per_scenario: false
    packages: [numpy]
    s3_artifacts:
      - key: "results/test/r0_{scenario}.json"
        per_scenario: true
        scenarios: *modelable
    # No outputs: block — intentionally missing

  - phase_id: r4_vlm_comparison
    script: compute_vlm_comparison.py
    per_scenario: false
    packages: [numpy]
    s3_artifacts:
      - key: "results/test/r4_{vlm}_{scenario}.parquet"
        per_scenario: true
        scenarios: *modelable
```

- [ ] **Step 2: Write the failing test**

```python
# tests/experiment_audit/test_contract.py
"""Tests for contract parsing and cell matrix resolution."""
from __future__ import annotations

from pathlib import Path

import pytest

from georsct.experiment_audit.contract import (
    ContractParseError,
    ParsedContract,
    parse_contract,
)
from georsct.experiment_audit.models import CellKey

FIXTURES = Path(__file__).parent / "fixtures"


def test_parse_contract_loads():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    assert contract.experiment == "test-audit"
    assert contract.bucket == "test-bucket"


def test_parse_contract_cell_matrix():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    # 2 nfip + 1 has_311 = 3 cells
    assert len(contract.cell_matrix) == 3
    assert CellKey("houston", "obs_nfip_event_claims") in contract.cell_matrix
    assert CellKey("houston", "obs_has_311") in contract.cell_matrix
    assert CellKey("nyc", "obs_nfip_event_claims") in contract.cell_matrix
    # nyc/obs_has_311 NOT contracted
    assert CellKey("nyc", "obs_has_311") not in contract.cell_matrix


def test_parse_contract_phases():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    phase_ids = [p.phase_id for p in contract.phases]
    assert "geometry_kappa" in phase_ids
    assert "r0_baseline" in phase_ids
    assert "diagnostics_r0" in phase_ids


def test_parse_contract_phase_has_outputs():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    gk = contract.get_phase("geometry_kappa")
    assert gk is not None
    assert len(gk.declared_outputs) == 1
    assert gk.declared_outputs[0].key_template == "results/test/geometry_kappa.json"


def test_parse_contract_phase_missing_outputs():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    diag = contract.get_phase("diagnostics_r0")
    assert diag is not None
    assert len(diag.declared_outputs) == 0
    assert diag.has_declared_outputs is False


def test_resolve_output_keys_per_scenario():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    r0 = contract.get_phase("r0_baseline")
    keys = r0.resolve_output_keys(contract.scenarios)
    assert "results/test/r0_houston.json" in keys
    assert "results/test/r0_nyc.json" in keys
    assert "results/test/r0_houston_predictions.parquet" in keys
    assert len(keys) == 4  # 2 scenarios x 2 outputs


def test_resolve_output_keys_non_per_scenario():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    gk = contract.get_phase("geometry_kappa")
    keys = gk.resolve_output_keys(contract.scenarios)
    assert keys == ["results/test/geometry_kappa.json"]


def test_unresolved_template_detected():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    vlm = contract.get_phase("r4_vlm_comparison")
    # s3_artifacts has {vlm} which is not resolvable from contract alone
    unresolved = contract.find_unresolved_templates()
    assert any("{vlm}" in u for u in unresolved)


def test_parse_contract_scenarios():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    assert contract.scenarios == ["houston", "nyc"]


def test_parse_contract_depends_on():
    # Our mini fixture doesn't have depends_on, but test the field exists
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    gk = contract.get_phase("geometry_kappa")
    assert gk.depends_on == []
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_contract.py -v`
Expected: FAIL — `ImportError: cannot import name 'parse_contract'`

- [ ] **Step 4: Implement `contract.py`**

```python
# georsct/experiment_audit/contract.py
"""Parse EXPERIMENT_CONTRACT.yaml and resolve the cell matrix."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .models import CellKey


class ContractParseError(Exception):
    """Raised when the contract YAML is malformed."""


@dataclass
class OutputSpec:
    """One declared output from a contract phase."""

    key_template: str
    per_scenario: bool = False
    description: str = ""

    def resolve_keys(self, scenarios: list[str]) -> list[str]:
        if not self.per_scenario:
            return [self.key_template]
        return [
            self.key_template.replace("{scenario}", sc)
            for sc in scenarios
        ]


@dataclass
class ArtifactSpec:
    """One declared input artifact from a contract phase."""

    key_template: str
    per_scenario: bool = False
    scenarios: list[str] | None = None
    optional: bool = False
    prefix: bool = False


@dataclass
class PhaseSpec:
    """One phase from the contract."""

    phase_id: str
    script: str
    per_scenario: bool
    packages: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    declared_outputs: list[OutputSpec] = field(default_factory=list)
    s3_artifacts: list[ArtifactSpec] = field(default_factory=list)
    description: str = ""

    @property
    def has_declared_outputs(self) -> bool:
        return len(self.declared_outputs) > 0

    def resolve_output_keys(self, scenarios: list[str]) -> list[str]:
        keys: list[str] = []
        for out in self.declared_outputs:
            keys.extend(out.resolve_keys(scenarios))
        return keys


@dataclass
class ParsedContract:
    """Fully parsed experiment contract."""

    experiment: str
    bucket: str
    version: str
    scenarios: list[str]
    cell_matrix: frozenset[CellKey]
    targets: list[dict[str, Any]]
    phases: list[PhaseSpec]
    raw: dict[str, Any]

    def get_phase(self, phase_id: str) -> PhaseSpec | None:
        for p in self.phases:
            if p.phase_id == phase_id:
                return p
        return None

    def find_unresolved_templates(self) -> list[str]:
        """Find template variables in s3_artifacts that cannot be resolved."""
        known_vars = {"{scenario}", "{scenario_key}"}
        unresolved: list[str] = []
        for phase in self.phases:
            for art in phase.s3_artifacts:
                template_vars = set(re.findall(r"\{[^}]+\}", art.key_template))
                bad = template_vars - known_vars
                if bad:
                    unresolved.append(
                        f"{phase.phase_id}: {art.key_template} has unresolved {bad}"
                    )
            for out in phase.declared_outputs:
                template_vars = set(re.findall(r"\{[^}]+\}", out.key_template))
                bad = template_vars - known_vars
                if bad:
                    unresolved.append(
                        f"{phase.phase_id}: {out.key_template} has unresolved {bad}"
                    )
        return unresolved


def _parse_outputs(raw_outputs: list[dict] | None) -> list[OutputSpec]:
    if not raw_outputs:
        return []
    result: list[OutputSpec] = []
    for out in raw_outputs:
        result.append(OutputSpec(
            key_template=out["key"],
            per_scenario=out.get("per_scenario", False),
            description=out.get("description", ""),
        ))
    return result


def _parse_artifacts(raw_artifacts: list[dict] | None) -> list[ArtifactSpec]:
    if not raw_artifacts:
        return []
    result: list[ArtifactSpec] = []
    for art in raw_artifacts:
        result.append(ArtifactSpec(
            key_template=art["key"],
            per_scenario=art.get("per_scenario", False),
            scenarios=art.get("scenarios"),
            optional=art.get("optional", False),
            prefix=art.get("prefix", False),
        ))
    return result


def _build_cell_matrix(targets: list[dict]) -> frozenset[CellKey]:
    cells: set[CellKey] = set()
    for t in targets:
        column = t["column"]
        for sc in t["scenarios"]:
            cells.add(CellKey(scenario=sc, target=column))
    return frozenset(cells)


def parse_contract(path: Path) -> ParsedContract:
    """Parse an EXPERIMENT_CONTRACT.yaml file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raise ContractParseError(f"Empty YAML: {path}")

    experiment = raw.get("experiment", "")
    bucket = raw.get("bucket", "")
    version = raw.get("version", "")
    scenarios = raw.get("modelable_scenarios", [])
    targets = raw.get("targets", [])

    cell_matrix = _build_cell_matrix(targets)

    phases: list[PhaseSpec] = []
    for p in raw.get("phases", []):
        phases.append(PhaseSpec(
            phase_id=p["phase_id"],
            script=p.get("script", ""),
            per_scenario=p.get("per_scenario", True),
            packages=p.get("packages", []),
            depends_on=p.get("depends_on", []),
            declared_outputs=_parse_outputs(p.get("outputs")),
            s3_artifacts=_parse_artifacts(p.get("s3_artifacts")),
            description=p.get("description", ""),
        ))

    return ParsedContract(
        experiment=experiment,
        bucket=bucket,
        version=version,
        scenarios=scenarios,
        cell_matrix=cell_matrix,
        targets=targets,
        phases=phases,
        raw=raw,
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_contract.py -v`
Expected: All 10 tests PASS

- [ ] **Step 6: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add georsct/experiment_audit/contract.py tests/experiment_audit/test_contract.py tests/experiment_audit/fixtures/mini_contract.yaml
git commit -m "feat(experiment_audit): add contract parser with cell matrix resolution"
```

---

## Task 4: S3 Inventory (`s3_inventory.py`)

**Files:**
- Create: `georsct/experiment_audit/s3_inventory.py`
- Test: `tests/experiment_audit/test_s3_inventory.py`

Note: Tests use a stub S3 client (dict-based fake) to avoid real AWS calls. The module uses `swarm_auth` for production.

- [ ] **Step 1: Write the failing test**

```python
# tests/experiment_audit/test_s3_inventory.py
"""Tests for S3 inventory (using stub client)."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from georsct.experiment_audit.models import ArtifactRecord
from georsct.experiment_audit.s3_inventory import (
    S3Inventory,
    S3ClientProtocol,
)


class StubS3Client:
    """Minimal S3 client stub for testing."""

    def __init__(self, objects: dict[str, dict[str, Any]]):
        self._objects = objects  # key -> {"size": int, "last_modified": str, "body": bytes}

    def head_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self._objects:
            from botocore.exceptions import ClientError
            raise ClientError(
                {"Error": {"Code": "404", "Message": "Not Found"}},
                "HeadObject",
            )
        obj = self._objects[Key]
        return {
            "ContentLength": obj["size"],
            "LastModified": datetime.fromisoformat(obj["last_modified"]),
        }

    def get_object(self, Bucket: str, Key: str) -> dict:
        if Key not in self._objects:
            from botocore.exceptions import ClientError
            raise ClientError(
                {"Error": {"Code": "NoSuchKey", "Message": "Not Found"}},
                "GetObject",
            )
        import io
        return {"Body": io.BytesIO(self._objects[Key]["body"])}

    def list_objects_v2(self, Bucket: str, Prefix: str, **kwargs) -> dict:
        matches = [k for k in self._objects if k.startswith(Prefix)]
        contents = [
            {
                "Key": k,
                "Size": self._objects[k]["size"],
                "LastModified": datetime.fromisoformat(self._objects[k]["last_modified"]),
            }
            for k in sorted(matches)
        ]
        return {"Contents": contents} if contents else {}


@pytest.fixture
def stub_inventory():
    import json
    objects = {
        "results/s035/geometry_kappa.json": {
            "size": 5000,
            "last_modified": "2026-06-04T08:00:00+00:00",
            "body": json.dumps({"timestamp": "2026-06-04T07:50:00Z", "cells": []}).encode(),
        },
        "results/s035/r0_houston.json": {
            "size": 12000,
            "last_modified": "2026-06-05T10:00:00+00:00",
            "body": json.dumps({"timestamp": "2026-06-05T09:45:00Z", "runs": []}).encode(),
        },
    }
    client = StubS3Client(objects)
    return S3Inventory(client=client, bucket="test-bucket")


def test_check_exists(stub_inventory):
    record = stub_inventory.check_key("results/s035/geometry_kappa.json")
    assert record.exists is True
    assert record.size_bytes == 5000


def test_check_not_exists(stub_inventory):
    record = stub_inventory.check_key("results/s035/nonexistent.json")
    assert record.exists is False
    assert record.size_bytes is None


def test_download_json(stub_inventory):
    record = stub_inventory.download_json("results/s035/geometry_kappa.json")
    assert record.exists is True
    assert record.content is not None
    assert record.content["timestamp"] == "2026-06-04T07:50:00Z"
    assert record.internal_timestamp == "2026-06-04T07:50:00Z"


def test_download_json_not_found(stub_inventory):
    record = stub_inventory.download_json("results/s035/nope.json")
    assert record.exists is False
    assert record.content is None


def test_list_prefix(stub_inventory):
    keys = stub_inventory.list_prefix("results/s035/")
    assert len(keys) == 2
    assert "results/s035/geometry_kappa.json" in keys


def test_best_timestamp_uses_internal(stub_inventory):
    record = stub_inventory.download_json("results/s035/geometry_kappa.json")
    assert record.best_timestamp() == "2026-06-04T07:50:00Z"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_s3_inventory.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `s3_inventory.py`**

```python
# georsct/experiment_audit/s3_inventory.py
"""S3 artifact inventory for experiment audit.

All S3 access goes through swarm_auth. Never use bare boto3.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Protocol, runtime_checkable

from botocore.exceptions import ClientError

from .models import ArtifactRecord

log = logging.getLogger(__name__)

# Known timestamp field names in result JSONs, checked in priority order.
_TIMESTAMP_FIELDS = ("timestamp", "created_at", "generated_at", "run_started_at")


@runtime_checkable
class S3ClientProtocol(Protocol):
    """Minimal S3 client interface for testing."""

    def head_object(self, Bucket: str, Key: str) -> dict: ...
    def get_object(self, Bucket: str, Key: str) -> dict: ...
    def list_objects_v2(self, Bucket: str, Prefix: str, **kwargs: Any) -> dict: ...


def _make_default_client() -> Any:
    """Create an S3 client using swarm_auth."""
    from swarm_auth import get_aws_credentials
    import boto3
    creds = get_aws_credentials()
    return boto3.client("s3", **creds)


def _extract_internal_timestamp(data: dict[str, Any]) -> str | None:
    """Extract the best internal timestamp from a result JSON."""
    for field_name in _TIMESTAMP_FIELDS:
        val = data.get(field_name)
        if val is not None:
            return str(val)
    return None


class S3Inventory:
    """S3 artifact inventory with metadata extraction."""

    def __init__(
        self,
        client: Any | None = None,
        bucket: str = "swarm-floodrsct-data",
    ):
        self._client = client or _make_default_client()
        self._bucket = bucket

    def check_key(self, key: str) -> ArtifactRecord:
        """Check if an S3 key exists and get metadata."""
        try:
            resp = self._client.head_object(Bucket=self._bucket, Key=key)
            lm = resp.get("LastModified")
            return ArtifactRecord(
                s3_key=key,
                exists=True,
                size_bytes=resp.get("ContentLength"),
                last_modified=lm.isoformat() if lm else None,
            )
        except ClientError:
            return ArtifactRecord(s3_key=key, exists=False)

    def download_json(self, key: str) -> ArtifactRecord:
        """Download a JSON file and extract internal timestamp."""
        try:
            head = self._client.head_object(Bucket=self._bucket, Key=key)
            resp = self._client.get_object(Bucket=self._bucket, Key=key)
            body = resp["Body"].read()
            data = json.loads(body)
            lm = head.get("LastModified")
            internal_ts = _extract_internal_timestamp(data)
            return ArtifactRecord(
                s3_key=key,
                exists=True,
                size_bytes=head.get("ContentLength"),
                last_modified=lm.isoformat() if lm else None,
                internal_timestamp=internal_ts,
                content=data,
            )
        except ClientError:
            return ArtifactRecord(s3_key=key, exists=False)

    def list_prefix(self, prefix: str) -> list[str]:
        """List all S3 keys under a prefix."""
        keys: list[str] = []
        kwargs: dict[str, Any] = {"Bucket": self._bucket, "Prefix": prefix}
        while True:
            resp = self._client.list_objects_v2(**kwargs)
            for obj in resp.get("Contents", []):
                keys.append(obj["Key"])
            if not resp.get("IsTruncated"):
                break
            kwargs["ContinuationToken"] = resp["NextContinuationToken"]
        return keys
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_s3_inventory.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add georsct/experiment_audit/s3_inventory.py tests/experiment_audit/test_s3_inventory.py
git commit -m "feat(experiment_audit): add S3 inventory with swarm_auth and timestamp extraction"
```

---

## Task 5: Content Readers (`content_readers.py`)

**Files:**
- Create: `georsct/experiment_audit/content_readers.py`
- Create: `tests/experiment_audit/fixtures/sample_r0.json`
- Create: `tests/experiment_audit/fixtures/sample_money_table.json`
- Create: `tests/experiment_audit/fixtures/sample_geometry_kappa.json`
- Test: `tests/experiment_audit/test_content_readers.py`

- [ ] **Step 1: Create test fixtures**

Create `tests/experiment_audit/fixtures/sample_r0.json`:
```json
{
  "experiment": "s035-model-ladder",
  "phase": "r0_baseline",
  "scenario": "houston",
  "timestamp": "2026-06-05T09:45:00Z",
  "runs": [
    {
      "scenario": "houston",
      "target": "obs_nfip_event_claims",
      "task": "regression",
      "solver": "histgbdt",
      "split": "spatial_blocked",
      "fold": "fold_0",
      "metrics": {"r2": 0.42, "rmse": 1.23}
    },
    {
      "scenario": "houston",
      "target": "obs_nfip_event_claims",
      "task": "regression",
      "solver": "ridge",
      "split": "spatial_blocked",
      "fold": "fold_0",
      "metrics": {"r2": 0.35, "rmse": 1.45}
    },
    {
      "scenario": "houston",
      "target": "obs_has_311",
      "task": "binary",
      "solver": "histgbdt",
      "split": "spatial_blocked",
      "fold": "fold_0",
      "metrics": {"roc_auc": 0.78}
    },
    {
      "scenario": "houston",
      "target": "obs_has_hwm",
      "task": "binary",
      "solver": "histgbdt",
      "split": "spatial_blocked",
      "fold": "fold_0",
      "metrics": {"roc_auc": null}
    }
  ]
}
```

Create `tests/experiment_audit/fixtures/sample_money_table.json`:
```json
{
  "timestamp": "2026-06-06T12:00:00Z",
  "cells": [
    {
      "scenario": "houston",
      "target": "obs_nfip_event_claims",
      "r0_metric": 0.42,
      "r1_metric": 0.51,
      "r2_metric": 0.55
    },
    {
      "scenario": "houston",
      "target": "obs_has_311",
      "r0_metric": 0.78,
      "r1_metric": 0.80,
      "r2_metric": 0.82
    },
    {
      "scenario": "nyc",
      "target": "obs_nfip_event_claims",
      "r0_metric": null,
      "r1_metric": null,
      "r2_metric": null
    }
  ]
}
```

Create `tests/experiment_audit/fixtures/sample_geometry_kappa.json`:
```json
{
  "timestamp": "2026-06-04T07:50:00Z",
  "cells": [
    {"scenario": "houston", "target": "obs_nfip_event_claims", "kappa_geom": 0.65},
    {"scenario": "houston", "target": "obs_has_311", "kappa_geom": 0.58},
    {"scenario": "nyc", "target": "obs_nfip_event_claims", "kappa_geom": null}
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/experiment_audit/test_content_readers.py
"""Tests for JSON content readers."""
from __future__ import annotations

import json
from pathlib import Path

from georsct.experiment_audit.content_readers import (
    extract_cells_from_runs,
    extract_money_table_cells,
    extract_aggregate_cells,
    CellMetric,
    MetricStatus,
)
from georsct.experiment_audit.models import CellKey

FIXTURES = Path(__file__).parent / "fixtures"


def test_extract_cells_from_runs():
    data = json.loads((FIXTURES / "sample_r0.json").read_text())
    cells = extract_cells_from_runs(data)
    keys = {c.cell for c in cells}
    assert CellKey("houston", "obs_nfip_event_claims") in keys
    assert CellKey("houston", "obs_has_311") in keys
    assert CellKey("houston", "obs_has_hwm") in keys


def test_extract_runs_metric_status():
    data = json.loads((FIXTURES / "sample_r0.json").read_text())
    cells = extract_cells_from_runs(data)
    # nfip has valid r2
    nfip = [c for c in cells if c.cell.target == "obs_nfip_event_claims"][0]
    assert nfip.status == MetricStatus.VALID

    # hwm has null metric
    hwm = [c for c in cells if c.cell.target == "obs_has_hwm"][0]
    assert hwm.status == MetricStatus.NULL


def test_extract_runs_best_metric():
    data = json.loads((FIXTURES / "sample_r0.json").read_text())
    cells = extract_cells_from_runs(data)
    nfip = [c for c in cells if c.cell.target == "obs_nfip_event_claims"][0]
    # Best solver r2 should be 0.42 (histgbdt)
    assert nfip.best_metric == pytest.approx(0.42)


def test_extract_money_table_cells():
    data = json.loads((FIXTURES / "sample_money_table.json").read_text())
    cells = extract_money_table_cells(data)
    assert len(cells) == 3
    # NYC nfip has null metrics
    nyc = [c for c in cells if c.cell.scenario == "nyc"][0]
    assert nyc.status == MetricStatus.NULL


def test_extract_aggregate_cells():
    data = json.loads((FIXTURES / "sample_geometry_kappa.json").read_text())
    cells = extract_aggregate_cells(data, value_field="kappa_geom")
    assert len(cells) == 3
    nyc = [c for c in cells if c.cell.scenario == "nyc"][0]
    assert nyc.status == MetricStatus.NULL
    houston_nfip = [c for c in cells
                    if c.cell.scenario == "houston"
                    and c.cell.target == "obs_nfip_event_claims"][0]
    assert houston_nfip.best_metric == pytest.approx(0.65)


import pytest
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_content_readers.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 4: Implement `content_readers.py`**

```python
# georsct/experiment_audit/content_readers.py
"""Parse result JSONs to extract per-cell metric status.

Handles three JSON shapes:
  1. Per-scenario run results (r0_houston.json) — has runs[] array
  2. Aggregate money table (money_table.json) — has cells[] array with rX_metric
  3. Aggregate cell arrays (geometry_kappa.json, diagnostics) — has cells[] with a value field
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .models import CellKey


class MetricStatus(str, Enum):
    VALID = "VALID"
    NULL = "NULL"
    DEGENERATE = "DEGENERATE"


@dataclass
class CellMetric:
    """Extracted metric for one cell from a result JSON."""

    cell: CellKey
    best_metric: float | None
    status: MetricStatus
    solver: str | None = None
    n_folds: int = 0
    n_null_folds: int = 0


def _classify_metric(value: float | None) -> MetricStatus:
    if value is None:
        return MetricStatus.NULL
    return MetricStatus.VALID


def extract_cells_from_runs(data: dict[str, Any]) -> list[CellMetric]:
    """Extract cell metrics from a per-scenario result JSON with runs[] array.

    Groups by (scenario, target), takes the best metric across solvers.
    """
    runs = data.get("runs", [])
    if not runs:
        return []

    # Group runs by (scenario, target)
    grouped: dict[tuple[str, str], list[dict]] = {}
    for run in runs:
        key = (run["scenario"], run["target"])
        grouped.setdefault(key, []).append(run)

    results: list[CellMetric] = []
    for (scenario, target), cell_runs in grouped.items():
        cell = CellKey(scenario=scenario, target=target)

        # Determine primary metric from task type
        best_metric: float | None = None
        best_solver: str | None = None
        n_folds = len(cell_runs)
        n_null = 0

        for run in cell_runs:
            metrics = run.get("metrics", {})
            # Try r2 for regression, roc_auc for classification
            val = metrics.get("r2") if metrics.get("r2") is not None else metrics.get("roc_auc")
            if val is None:
                n_null += 1
                continue
            if best_metric is None or val > best_metric:
                best_metric = val
                best_solver = run.get("solver")

        status = _classify_metric(best_metric)
        results.append(CellMetric(
            cell=cell,
            best_metric=best_metric,
            status=status,
            solver=best_solver,
            n_folds=n_folds,
            n_null_folds=n_null,
        ))

    return results


def extract_money_table_cells(data: dict[str, Any]) -> list[CellMetric]:
    """Extract cell metrics from money_table.json.

    Checks r0_metric, r1_metric, r2_metric. A cell is NULL if all are None.
    """
    cells_data = data.get("cells", [])
    results: list[CellMetric] = []

    for cell_data in cells_data:
        cell = CellKey(
            scenario=cell_data["scenario"],
            target=cell_data["target"],
        )
        # Take the highest-level non-null metric
        best = None
        for level in ("r2_metric", "r1_metric", "r0_metric"):
            val = cell_data.get(level)
            if val is not None:
                best = val
                break

        results.append(CellMetric(
            cell=cell,
            best_metric=best,
            status=_classify_metric(best),
        ))

    return results


def extract_aggregate_cells(
    data: dict[str, Any],
    value_field: str,
) -> list[CellMetric]:
    """Extract cell metrics from an aggregate JSON with cells[] array.

    Used for geometry_kappa.json, diagnostics, certificates, etc.
    """
    cells_data = data.get("cells", [])
    results: list[CellMetric] = []

    for cell_data in cells_data:
        cell = CellKey(
            scenario=cell_data["scenario"],
            target=cell_data["target"],
        )
        val = cell_data.get(value_field)
        results.append(CellMetric(
            cell=cell,
            best_metric=val,
            status=_classify_metric(val),
        ))

    return results
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_content_readers.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add georsct/experiment_audit/content_readers.py tests/experiment_audit/test_content_readers.py tests/experiment_audit/fixtures/
git commit -m "feat(experiment_audit): add content readers for run results, money table, and aggregates"
```

---

## Task 6: Timestamp Ordering (`timestamp_ordering.py`)

**Files:**
- Create: `georsct/experiment_audit/timestamp_ordering.py`
- Test: `tests/experiment_audit/test_timestamp_ordering.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiment_audit/test_timestamp_ordering.py
"""Tests for three-tier timestamp ordering logic."""
from __future__ import annotations

import pytest

from georsct.experiment_audit.models import ArtifactRecord, CheckResult, Severity
from georsct.experiment_audit.timestamp_ordering import (
    OrderingRule,
    check_ordering,
)


def _make_record(key: str, internal_ts: str | None, s3_ts: str | None) -> ArtifactRecord:
    return ArtifactRecord(
        s3_key=key,
        exists=True,
        internal_timestamp=internal_ts,
        last_modified=s3_ts,
    )


def test_pass_internal_timestamps():
    """geometry_kappa computed before r0 — internal timestamps prove it."""
    before = _make_record("geometry_kappa.json", "2026-06-04T07:00:00Z", "2026-06-04T08:00:00Z")
    after = _make_record("r0_houston.json", "2026-06-05T09:00:00Z", "2026-06-05T10:00:00Z")
    rule = OrderingRule(before_key="geometry_kappa.json", after_key="r0_houston.json",
                        description="geometry_kappa before r0")
    result = check_ordering(rule, before, after)
    assert result.severity == Severity.PASS


def test_fail_internal_timestamps_contradict():
    """Internal timestamps prove wrong ordering."""
    before = _make_record("geometry_kappa.json", "2026-06-06T09:00:00Z", None)
    after = _make_record("r0_houston.json", "2026-06-05T08:00:00Z", None)
    rule = OrderingRule(before_key="geometry_kappa.json", after_key="r0_houston.json",
                        description="geometry_kappa before r0")
    result = check_ordering(rule, before, after)
    assert result.severity == Severity.FAIL_ORDERING_VIOLATION


def test_warn_fallback_s3_timestamps_ok():
    """No internal timestamps; S3 timestamps look correct."""
    before = _make_record("geometry_kappa.json", None, "2026-06-04T08:00:00Z")
    after = _make_record("r0_houston.json", None, "2026-06-05T10:00:00Z")
    rule = OrderingRule(before_key="geometry_kappa.json", after_key="r0_houston.json",
                        description="geometry_kappa before r0")
    result = check_ordering(rule, before, after)
    assert result.severity == Severity.WARN_TIMESTAMP_FALLBACK


def test_warn_review_s3_timestamps_wrong():
    """No internal timestamps; S3 timestamps look wrong (possible rerun)."""
    before = _make_record("geometry_kappa.json", None, "2026-06-06T08:00:00Z")
    after = _make_record("r0_houston.json", None, "2026-06-05T10:00:00Z")
    rule = OrderingRule(before_key="geometry_kappa.json", after_key="r0_houston.json",
                        description="geometry_kappa before r0")
    result = check_ordering(rule, before, after)
    # S3 wrong but not definitive — WARN, not FAIL
    assert result.severity == Severity.WARN_TIMESTAMP_FALLBACK


def test_warn_no_timestamps():
    """No timestamps available at all."""
    before = _make_record("geometry_kappa.json", None, None)
    after = _make_record("r0_houston.json", None, None)
    rule = OrderingRule(before_key="geometry_kappa.json", after_key="r0_houston.json",
                        description="geometry_kappa before r0")
    result = check_ordering(rule, before, after)
    assert result.severity == Severity.WARN_TIMESTAMP_UNVERIFIABLE


def test_warn_one_missing():
    """One artifact doesn't exist — can't verify ordering."""
    before = _make_record("geometry_kappa.json", "2026-06-04T07:00:00Z", None)
    after = ArtifactRecord(s3_key="r0_houston.json", exists=False)
    rule = OrderingRule(before_key="geometry_kappa.json", after_key="r0_houston.json",
                        description="geometry_kappa before r0")
    result = check_ordering(rule, before, after)
    assert result.severity == Severity.WARN_TIMESTAMP_UNVERIFIABLE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_timestamp_ordering.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `timestamp_ordering.py`**

```python
# georsct/experiment_audit/timestamp_ordering.py
"""Three-tier timestamp ordering verification.

Tier 1: Internal JSON timestamp (authoritative)
Tier 2: S3 LastModified (fallback — rerun-sensitive)
Tier 3: Missing timestamps (unverifiable)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from .models import ArtifactRecord, CheckResult, Severity


@dataclass(frozen=True)
class OrderingRule:
    """A rule stating that before_key must have been computed before after_key."""

    before_key: str
    after_key: str
    description: str


# Canonical ordering rules for s035.
CANONICAL_ORDERING_RULES: list[OrderingRule] = [
    OrderingRule(
        before_key="results/s035/geometry_kappa.json",
        after_key="results/s035/r0_{scenario}.json",
        description="geometry_kappa computed before earliest model training",
    ),
    OrderingRule(
        before_key="results/s035/gearbox_warmup.json",
        after_key="results/s035/certificates_r0.json",
        description="gearbox_warmup before certificates",
    ),
    OrderingRule(
        before_key="results/s035/diagnostics_r0.json",
        after_key="results/s035/certificates_r0.json",
        description="diagnostics before certificates at each level",
    ),
    OrderingRule(
        before_key="results/s035/diagnostics_r1.json",
        after_key="results/s035/certificates_r1.json",
        description="diagnostics before certificates at each level",
    ),
    OrderingRule(
        before_key="results/s035/diagnostics_r2.json",
        after_key="results/s035/certificates_r2.json",
        description="diagnostics before certificates at each level",
    ),
]


def _parse_ts(ts_str: str) -> datetime:
    """Parse an ISO timestamp string, tolerating Z suffix."""
    s = ts_str.replace("Z", "+00:00")
    return datetime.fromisoformat(s)


def check_ordering(
    rule: OrderingRule,
    before: ArtifactRecord,
    after: ArtifactRecord,
) -> CheckResult:
    """Check that `before` was computed before `after` using three-tier logic."""

    # If either artifact doesn't exist, ordering is unverifiable
    if not before.exists or not after.exists:
        return CheckResult(
            gate="ordering",
            name=rule.description,
            severity=Severity.WARN_TIMESTAMP_UNVERIFIABLE,
            message=f"Cannot verify ordering: {'before' if not before.exists else 'after'} artifact missing",
        )

    # Tier 1: Both have internal timestamps
    before_internal = before.internal_timestamp
    after_internal = after.internal_timestamp
    if before_internal and after_internal:
        before_dt = _parse_ts(before_internal)
        after_dt = _parse_ts(after_internal)
        if before_dt <= after_dt:
            return CheckResult(
                gate="ordering",
                name=rule.description,
                severity=Severity.PASS,
                message=f"Internal timestamps confirm ordering: {before_internal} <= {after_internal}",
            )
        else:
            return CheckResult(
                gate="ordering",
                name=rule.description,
                severity=Severity.FAIL_ORDERING_VIOLATION,
                message=f"Internal timestamps contradict ordering: {before_internal} > {after_internal}",
            )

    # Tier 2: Fall back to S3 LastModified
    before_s3 = before.last_modified
    after_s3 = after.last_modified
    if before_s3 and after_s3:
        before_dt = _parse_ts(before_s3)
        after_dt = _parse_ts(after_s3)
        if before_dt <= after_dt:
            return CheckResult(
                gate="ordering",
                name=rule.description,
                severity=Severity.WARN_TIMESTAMP_FALLBACK,
                message=f"S3 timestamps suggest correct ordering ({before_s3} <= {after_s3}), but reruns may have inverted LastModified",
            )
        else:
            return CheckResult(
                gate="ordering",
                name=rule.description,
                severity=Severity.WARN_TIMESTAMP_FALLBACK,
                message=f"S3 timestamps suggest wrong ordering ({before_s3} > {after_s3}), possible rerun or backfill",
            )

    # Tier 3: No usable timestamps
    return CheckResult(
        gate="ordering",
        name=rule.description,
        severity=Severity.WARN_TIMESTAMP_UNVERIFIABLE,
        message="No usable timestamps found for either artifact",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_timestamp_ordering.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add georsct/experiment_audit/timestamp_ordering.py tests/experiment_audit/test_timestamp_ordering.py
git commit -m "feat(experiment_audit): add three-tier timestamp ordering verification"
```

---

## Task 7: Validators (`validators.py`) — The 6 Audit Gates

**Files:**
- Create: `georsct/experiment_audit/validators.py`
- Test: `tests/experiment_audit/test_validators.py`

This is the largest module. It implements six gates that run in sequence:
1. Contract parse gate — validates all template variables resolve
2. Declared output existence gate — checks files the contract declares
3. Known convention supplement gate — checks R1/R2/diagnostics/etc. using known patterns
4. Cell matrix gate — verifies contracted cells in results
5. Aggregate JSON content gate — parses money table, geometry kappa, etc.
6. Ordering gate — verifies timestamp ordering

- [ ] **Step 1: Write the failing test**

```python
# tests/experiment_audit/test_validators.py
"""Tests for the 6 audit gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from georsct.experiment_audit.contract import parse_contract
from georsct.experiment_audit.models import ArtifactRecord, CellKey, Severity
from georsct.experiment_audit.validators import (
    gate_1_contract_parse,
    gate_2_declared_outputs,
    gate_3_convention_supplement,
    gate_4_cell_matrix,
    gate_5_aggregate_content,
    gate_6_ordering,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ── Gate 1: Contract Parse ──────────────────────────────────────────────

def test_gate1_detects_unresolved_vlm():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    results = gate_1_contract_parse(contract)
    fails = [r for r in results if r.severity == Severity.FAIL_UNRESOLVED_TEMPLATE]
    assert len(fails) >= 1
    assert any("{vlm}" in r.message for r in fails)


def test_gate1_passes_clean_contract():
    # Build a contract without {vlm}
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    # Remove the vlm phase for this test
    contract.phases = [p for p in contract.phases if p.phase_id != "r4_vlm_comparison"]
    results = gate_1_contract_parse(contract)
    fails = [r for r in results if r.severity.is_fail()]
    assert len(fails) == 0


# ── Gate 2: Declared Output Existence ────────────────────────────────────

def test_gate2_pass_when_outputs_exist():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    # Stub inventory where all declared outputs exist
    inventory = {
        "results/test/geometry_kappa.json": ArtifactRecord(
            s3_key="results/test/geometry_kappa.json", exists=True, size_bytes=5000,
        ),
        "results/test/r0_houston.json": ArtifactRecord(
            s3_key="results/test/r0_houston.json", exists=True, size_bytes=12000,
        ),
        "results/test/r0_nyc.json": ArtifactRecord(
            s3_key="results/test/r0_nyc.json", exists=True, size_bytes=11000,
        ),
        "results/test/r0_houston_predictions.parquet": ArtifactRecord(
            s3_key="results/test/r0_houston_predictions.parquet", exists=True, size_bytes=50000,
        ),
        "results/test/r0_nyc_predictions.parquet": ArtifactRecord(
            s3_key="results/test/r0_nyc_predictions.parquet", exists=True, size_bytes=48000,
        ),
    }
    results = gate_2_declared_outputs(contract, inventory)
    fails = [r for r in results if r.severity.is_fail()]
    assert len(fails) == 0


def test_gate2_fail_when_output_missing():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    inventory = {
        "results/test/geometry_kappa.json": ArtifactRecord(
            s3_key="results/test/geometry_kappa.json", exists=True, size_bytes=5000,
        ),
        # r0_houston.json exists
        "results/test/r0_houston.json": ArtifactRecord(
            s3_key="results/test/r0_houston.json", exists=True, size_bytes=12000,
        ),
        # r0_nyc.json MISSING
        "results/test/r0_nyc.json": ArtifactRecord(
            s3_key="results/test/r0_nyc.json", exists=False,
        ),
        "results/test/r0_houston_predictions.parquet": ArtifactRecord(
            s3_key="results/test/r0_houston_predictions.parquet", exists=True, size_bytes=50000,
        ),
        "results/test/r0_nyc_predictions.parquet": ArtifactRecord(
            s3_key="results/test/r0_nyc_predictions.parquet", exists=False,
        ),
    }
    results = gate_2_declared_outputs(contract, inventory)
    fails = [r for r in results if r.severity == Severity.FAIL_MISSING_OUTPUT]
    assert len(fails) == 2  # r0_nyc.json + r0_nyc_predictions.parquet


def test_gate2_warns_on_undeclared_phase():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    diag = contract.get_phase("diagnostics_r0")
    # diagnostics_r0 has no outputs block
    assert not diag.has_declared_outputs
    inventory = {}
    results = gate_2_declared_outputs(contract, inventory)
    gaps = [r for r in results if r.severity == Severity.WARN_CONTRACT_GAP]
    # diagnostics_r0 should show as WARN_CONTRACT_GAP
    assert any("diagnostics_r0" in r.message for r in gaps)


# ── Gate 4: Cell Matrix ──────────────────────────────────────────────────

def test_gate4_detects_extra_cells():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    # Simulated: result contains nyc/obs_has_311 which is NOT in contract
    actual_cells = {
        CellKey("houston", "obs_nfip_event_claims"),
        CellKey("houston", "obs_has_311"),
        CellKey("nyc", "obs_nfip_event_claims"),
        CellKey("nyc", "obs_has_311"),  # NOT contracted
    }
    results = gate_4_cell_matrix(contract, actual_cells)
    extras = [r for r in results if r.severity == Severity.WARN_SCOPE_EXTRA]
    assert len(extras) == 1
    assert "nyc/obs_has_311" in extras[0].message


def test_gate4_detects_missing_cells():
    contract = parse_contract(FIXTURES / "mini_contract.yaml")
    # Missing houston/obs_has_311
    actual_cells = {
        CellKey("houston", "obs_nfip_event_claims"),
        CellKey("nyc", "obs_nfip_event_claims"),
    }
    results = gate_4_cell_matrix(contract, actual_cells)
    missing = [r for r in results if r.severity == Severity.FAIL_MISSING_OUTPUT]
    assert len(missing) == 1
    assert "houston/obs_has_311" in missing[0].message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_validators.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `validators.py`**

```python
# georsct/experiment_audit/validators.py
"""Six audit gates for post-flight experiment verification.

Gate 1: Contract parse — validates template variables resolve
Gate 2: Declared output existence — checks files the contract declares
Gate 3: Known convention supplement — checks undeclared outputs using S3 patterns
Gate 4: Cell matrix — verifies contracted cells in actual results
Gate 5: Aggregate JSON content — parses money table, geometry kappa, etc.
Gate 6: Ordering — verifies timestamp ordering
"""
from __future__ import annotations

from typing import Any

from .constants import CONTRACTED_CELLS, LEVEL_PREFIXES
from .contract import ParsedContract
from .models import ArtifactRecord, CellKey, CheckResult, Severity
from .timestamp_ordering import CANONICAL_ORDERING_RULES, check_ordering


# ── Gate 1: Contract Parse ──────────────────────────────────────────────


def gate_1_contract_parse(contract: ParsedContract) -> list[CheckResult]:
    """Validate all template variables resolve."""
    results: list[CheckResult] = []

    unresolved = contract.find_unresolved_templates()
    for u in unresolved:
        results.append(CheckResult(
            gate="contract_parse",
            name="template_resolution",
            severity=Severity.FAIL_UNRESOLVED_TEMPLATE,
            message=u,
        ))

    if not unresolved:
        results.append(CheckResult(
            gate="contract_parse",
            name="template_resolution",
            severity=Severity.PASS,
            message="All template variables resolve",
        ))

    return results


# ── Gate 2: Declared Output Existence ────────────────────────────────────


def gate_2_declared_outputs(
    contract: ParsedContract,
    inventory: dict[str, ArtifactRecord],
) -> list[CheckResult]:
    """Check files that the contract actually declares as outputs."""
    results: list[CheckResult] = []

    for phase in contract.phases:
        if not phase.has_declared_outputs:
            results.append(CheckResult(
                gate="declared_outputs",
                name="contract_gap",
                severity=Severity.WARN_CONTRACT_GAP,
                message=f"Phase {phase.phase_id} has no declared outputs -- cannot verify post-flight",
                phase_id=phase.phase_id,
            ))
            continue

        expected_keys = phase.resolve_output_keys(contract.scenarios)
        for key in expected_keys:
            record = inventory.get(key)
            if record is None or not record.exists:
                results.append(CheckResult(
                    gate="declared_outputs",
                    name="output_existence",
                    severity=Severity.FAIL_MISSING_OUTPUT,
                    message=f"Declared output missing: {key}",
                    phase_id=phase.phase_id,
                ))
            else:
                results.append(CheckResult(
                    gate="declared_outputs",
                    name="output_existence",
                    severity=Severity.PASS,
                    message=f"Output exists: {key} ({record.size_bytes} bytes)",
                    phase_id=phase.phase_id,
                ))

    return results


# ── Gate 3: Convention Supplement ────────────────────────────────────────


def gate_3_convention_supplement(
    contract: ParsedContract,
    s3_keys: list[str],
    results_prefix: str = "results/s035/",
) -> list[CheckResult]:
    """Check undeclared outputs using known S3 naming conventions.

    These phases have no outputs: in the contract, but we know what they produce.
    """
    results: list[CheckResult] = []

    # Known conventions for phases without declared outputs
    conventions: dict[str, list[str]] = {
        "diagnostics_r0": [f"{results_prefix}diagnostics_r0.json"],
        "diagnostics_r1": [f"{results_prefix}diagnostics_r1.json"],
        "diagnostics_r2": [f"{results_prefix}diagnostics_r2.json"],
        "certificates_r0": [
            f"{results_prefix}certificates_r0.json",
            f"{results_prefix}certificates_r0.parquet",
        ],
        "certificates_r1": [
            f"{results_prefix}certificates_r1.json",
            f"{results_prefix}certificates_r1.parquet",
        ],
        "certificates_r2": [
            f"{results_prefix}certificates_r2.json",
            f"{results_prefix}certificates_r2.parquet",
        ],
        "uplift_table": [f"{results_prefix}money_table.json"],
        "dgm_routing": [f"{results_prefix}dgm_routing.json"],
    }

    # R1/R2 training phases also lack outputs
    for sc in contract.scenarios:
        prefix = LEVEL_PREFIXES["r1"]
        conventions.setdefault("r1_hydrology", []).extend([
            f"{results_prefix}{prefix}_{sc}.json",
            f"{results_prefix}{prefix}_{sc}_predictions.parquet",
        ])
        prefix = LEVEL_PREFIXES["r2"]
        conventions.setdefault("r2_temporal", []).extend([
            f"{results_prefix}{prefix}_{sc}.json",
            f"{results_prefix}{prefix}_{sc}_predictions.parquet",
        ])

    s3_key_set = set(s3_keys)

    for phase_id, expected_keys in conventions.items():
        phase = contract.get_phase(phase_id)
        if phase is None:
            continue
        if phase.has_declared_outputs:
            continue  # Already checked in gate 2

        for key in expected_keys:
            if key in s3_key_set:
                results.append(CheckResult(
                    gate="convention_supplement",
                    name="undeclared_output_found",
                    severity=Severity.WARN_CONTRACT_GAP,
                    message=f"Phase {phase_id} produced {key}, but contract has no outputs declaration",
                    phase_id=phase_id,
                ))
            else:
                results.append(CheckResult(
                    gate="convention_supplement",
                    name="undeclared_output_missing",
                    severity=Severity.FAIL_MISSING_OUTPUT,
                    message=f"Expected {key} from convention for {phase_id}, but not found on S3",
                    phase_id=phase_id,
                ))

    # Check for undeclared real artifacts (not matching any phase)
    all_expected: set[str] = set()
    for keys in conventions.values():
        all_expected.update(keys)
    # Also include declared outputs
    for phase in contract.phases:
        all_expected.update(phase.resolve_output_keys(contract.scenarios))

    for key in s3_keys:
        if key.startswith(results_prefix) and key not in all_expected:
            results.append(CheckResult(
                gate="convention_supplement",
                name="undeclared_artifact",
                severity=Severity.WARN_UNDECLARED_ARTIFACT,
                message=f"Artifact exists but matches no contract phase: {key}",
            ))

    return results


# ── Gate 4: Cell Matrix ──────────────────────────────────────────────────


def gate_4_cell_matrix(
    contract: ParsedContract,
    actual_cells: set[CellKey],
) -> list[CheckResult]:
    """Compare contracted cells against actual cells found in results."""
    results: list[CheckResult] = []
    contracted = contract.cell_matrix

    # Missing from actuals
    for cell in contracted:
        if cell not in actual_cells:
            results.append(CheckResult(
                gate="cell_matrix",
                name="missing_cell",
                severity=Severity.FAIL_MISSING_OUTPUT,
                message=f"Contracted cell {cell} not found in results",
                cell=cell,
            ))

    # Extra in actuals
    for cell in actual_cells:
        if cell not in contracted:
            results.append(CheckResult(
                gate="cell_matrix",
                name="extra_cell",
                severity=Severity.WARN_SCOPE_EXTRA,
                message=f"Cell {cell} found in results but not in contract",
                cell=cell,
            ))

    # Present in both
    for cell in contracted & actual_cells:
        results.append(CheckResult(
            gate="cell_matrix",
            name="cell_present",
            severity=Severity.PASS,
            message=f"Cell {cell} present in results",
            cell=cell,
        ))

    return results


# ── Gate 5: Aggregate JSON Content ──────────────────────────────────────


def gate_5_aggregate_content(
    contract: ParsedContract,
    artifacts: dict[str, ArtifactRecord],
) -> list[CheckResult]:
    """Parse aggregate JSONs and check for null/degenerate cells."""
    from .content_readers import (
        MetricStatus,
        extract_aggregate_cells,
        extract_money_table_cells,
    )

    results: list[CheckResult] = []

    # Money table
    mt_key = "results/s035/money_table.json"
    mt_record = artifacts.get(mt_key)
    if mt_record and mt_record.content:
        mt_cells = extract_money_table_cells(mt_record.content)
        mt_cell_keys = {c.cell for c in mt_cells}

        # Check for contracted cells missing from money table
        for cell in contract.cell_matrix:
            if cell not in mt_cell_keys:
                results.append(CheckResult(
                    gate="aggregate_content",
                    name="money_table_missing_cell",
                    severity=Severity.FAIL_CONTENT_INCOMPLETE,
                    message=f"Contracted cell {cell} missing from money table (silent drop)",
                    cell=cell,
                ))

        # Check for null metrics
        for mc in mt_cells:
            if mc.status == MetricStatus.NULL:
                results.append(CheckResult(
                    gate="aggregate_content",
                    name="money_table_null_metric",
                    severity=Severity.FAIL_CONTENT_DEGENERATE,
                    message=f"Cell {mc.cell} has null metrics in money table",
                    cell=mc.cell,
                ))

    # Geometry kappa
    gk_key = "results/s035/geometry_kappa.json"
    gk_record = artifacts.get(gk_key)
    if gk_record and gk_record.content:
        gk_cells = extract_aggregate_cells(gk_record.content, "kappa_geom")
        for gc in gk_cells:
            if gc.status == MetricStatus.NULL:
                results.append(CheckResult(
                    gate="aggregate_content",
                    name="geometry_kappa_null",
                    severity=Severity.WARN_CONTRACT_GAP,
                    message=f"Cell {gc.cell} has null kappa_geom",
                    cell=gc.cell,
                ))

    return results


# ── Gate 6: Ordering ────────────────────────────────────────────────────


def gate_6_ordering(
    artifacts: dict[str, ArtifactRecord],
    scenarios: list[str] | None = None,
) -> list[CheckResult]:
    """Verify timestamp ordering using three-tier logic."""
    results: list[CheckResult] = []

    for rule in CANONICAL_ORDERING_RULES:
        if "{scenario}" in rule.before_key or "{scenario}" in rule.after_key:
            # Expand per-scenario rules
            for sc in (scenarios or []):
                expanded_rule = type(rule)(
                    before_key=rule.before_key.replace("{scenario}", sc),
                    after_key=rule.after_key.replace("{scenario}", sc),
                    description=f"{rule.description} ({sc})",
                )
                before = artifacts.get(expanded_rule.before_key,
                                       ArtifactRecord(s3_key=expanded_rule.before_key, exists=False))
                after = artifacts.get(expanded_rule.after_key,
                                      ArtifactRecord(s3_key=expanded_rule.after_key, exists=False))
                results.append(check_ordering(expanded_rule, before, after))
        else:
            before = artifacts.get(rule.before_key,
                                   ArtifactRecord(s3_key=rule.before_key, exists=False))
            after = artifacts.get(rule.after_key,
                                  ArtifactRecord(s3_key=rule.after_key, exists=False))
            results.append(check_ordering(rule, before, after))

    return results
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_validators.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add georsct/experiment_audit/validators.py tests/experiment_audit/test_validators.py
git commit -m "feat(experiment_audit): add 6 audit gates (contract parse, outputs, convention, cell matrix, content, ordering)"
```

---

## Task 8: Report Generator (`report.py`)

**Files:**
- Create: `georsct/experiment_audit/report.py`
- Test: `tests/experiment_audit/test_report.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiment_audit/test_report.py
"""Tests for report generation."""
from __future__ import annotations

import json

from georsct.experiment_audit.models import AuditResult, CheckResult, CellKey, Severity
from georsct.experiment_audit.report import render_json, render_markdown


def _make_sample_result() -> AuditResult:
    return AuditResult(
        checks=[
            CheckResult("contract_parse", "template_resolution", Severity.PASS,
                         "All templates resolve"),
            CheckResult("declared_outputs", "output_existence", Severity.FAIL_MISSING_OUTPUT,
                         "Missing: results/s035/r0_nyc.json", phase_id="r0_baseline"),
            CheckResult("cell_matrix", "extra_cell", Severity.WARN_SCOPE_EXTRA,
                         "Cell southwest_florida/obs_has_hwm not in contract",
                         cell=CellKey("southwest_florida", "obs_has_hwm")),
            CheckResult("convention_supplement", "contract_gap", Severity.WARN_CONTRACT_GAP,
                         "Phase diagnostics_r0 has no declared outputs", phase_id="diagnostics_r0"),
        ],
        metadata={"experiment": "s035-model-ladder", "bucket": "swarm-floodrsct-data"},
    )


def test_render_json_parseable():
    result = _make_sample_result()
    output = render_json(result)
    parsed = json.loads(output)
    assert parsed["overall_status"] == "FAIL"
    assert parsed["summary"]["FAIL"] == 1
    assert parsed["summary"]["WARN"] == 2
    assert parsed["summary"]["PASS"] == 1
    assert len(parsed["checks"]) == 4


def test_render_json_includes_metadata():
    result = _make_sample_result()
    output = render_json(result)
    parsed = json.loads(output)
    assert parsed["metadata"]["experiment"] == "s035-model-ladder"


def test_render_markdown_has_summary():
    result = _make_sample_result()
    md = render_markdown(result)
    assert "## Summary" in md
    assert "FAIL" in md
    assert "WARN" in md
    assert "PASS" in md


def test_render_markdown_has_failures_section():
    result = _make_sample_result()
    md = render_markdown(result)
    assert "## Failures" in md
    assert "r0_nyc.json" in md


def test_render_markdown_has_warnings_section():
    result = _make_sample_result()
    md = render_markdown(result)
    assert "## Warnings" in md
    assert "obs_has_hwm" in md
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_report.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `report.py`**

```python
# georsct/experiment_audit/report.py
"""Generate JSON and Markdown audit reports."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .models import AuditResult, Severity


def render_json(result: AuditResult) -> str:
    """Render audit result as JSON string."""
    output = {
        "overall_status": result.overall_status(),
        "summary": result.summary_counts(),
        "metadata": result.metadata,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checks": [c.to_dict() for c in result.checks],
    }
    return json.dumps(output, indent=2)


def render_markdown(result: AuditResult) -> str:
    """Render audit result as Markdown string."""
    lines: list[str] = []
    counts = result.summary_counts()

    lines.append("# Experiment Audit Report")
    lines.append("")
    exp = result.metadata.get("experiment", "unknown")
    lines.append(f"**Experiment:** {exp}")
    lines.append(f"**Overall:** {result.overall_status()}")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"| Status | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| PASS   | {counts['PASS']}     |")
    lines.append(f"| WARN   | {counts['WARN']}     |")
    lines.append(f"| FAIL   | {counts['FAIL']}     |")
    lines.append("")

    # Failures
    fails = [c for c in result.checks if c.severity.is_fail()]
    if fails:
        lines.append("## Failures")
        lines.append("")
        for c in fails:
            phase = f" [{c.phase_id}]" if c.phase_id else ""
            cell = f" ({c.cell})" if c.cell else ""
            lines.append(f"- **{c.severity.value}**{phase}{cell}: {c.message}")
        lines.append("")

    # Warnings
    warns = [c for c in result.checks if c.severity.is_warn()]
    if warns:
        lines.append("## Warnings")
        lines.append("")
        for c in warns:
            phase = f" [{c.phase_id}]" if c.phase_id else ""
            cell = f" ({c.cell})" if c.cell else ""
            lines.append(f"- **{c.severity.value}**{phase}{cell}: {c.message}")
        lines.append("")

    # Passes (collapsed)
    passes = [c for c in result.checks if c.severity == Severity.PASS]
    if passes:
        lines.append("## Passed Checks")
        lines.append("")
        lines.append(f"{len(passes)} checks passed.")
        lines.append("")

    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_report.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add georsct/experiment_audit/report.py tests/experiment_audit/test_report.py
git commit -m "feat(experiment_audit): add JSON and Markdown report generators"
```

---

## Task 9: Manifest Writer (`manifest.py`)

**Files:**
- Create: `georsct/experiment_audit/manifest.py`
- Test: `tests/experiment_audit/test_manifest.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiment_audit/test_manifest.py
"""Tests for completion manifest writer."""
from __future__ import annotations

import json

from georsct.experiment_audit.manifest import build_manifest
from georsct.experiment_audit.models import ArtifactRecord, CellKey


def test_build_manifest_structure():
    artifacts = {
        "results/s035/r0_houston.json": ArtifactRecord(
            s3_key="results/s035/r0_houston.json",
            exists=True,
            size_bytes=12000,
            last_modified="2026-06-05T10:00:00Z",
            internal_timestamp="2026-06-05T09:45:00Z",
        ),
        "results/s035/geometry_kappa.json": ArtifactRecord(
            s3_key="results/s035/geometry_kappa.json",
            exists=True,
            size_bytes=5000,
            last_modified="2026-06-04T08:00:00Z",
            internal_timestamp="2026-06-04T07:50:00Z",
        ),
    }
    cells = frozenset({CellKey("houston", "obs_nfip_event_claims")})

    manifest = build_manifest(
        experiment="s035-model-ladder",
        bucket="swarm-floodrsct-data",
        artifacts=artifacts,
        contracted_cells=cells,
    )

    assert manifest["experiment"] == "s035-model-ladder"
    assert len(manifest["artifacts"]) == 2
    assert manifest["contracted_cells"] == ["houston/obs_nfip_event_claims"]


def test_build_manifest_json_serializable():
    artifacts = {
        "results/s035/r0_houston.json": ArtifactRecord(
            s3_key="results/s035/r0_houston.json",
            exists=True, size_bytes=12000,
        ),
    }
    cells = frozenset({CellKey("houston", "obs_nfip_event_claims")})
    manifest = build_manifest("test", "test-bucket", artifacts, cells)
    # Must be JSON-serializable
    output = json.dumps(manifest, indent=2)
    assert "test" in output


def test_build_manifest_tracks_provenance():
    artifacts = {
        "results/s035/r0_houston.json": ArtifactRecord(
            s3_key="results/s035/r0_houston.json",
            exists=True,
            size_bytes=12000,
            internal_timestamp="2026-06-05T09:45:00Z",
        ),
    }
    cells = frozenset()
    manifest = build_manifest("test", "bucket", artifacts, cells)
    art = manifest["artifacts"][0]
    assert art["s3_key"] == "results/s035/r0_houston.json"
    assert art["internal_timestamp"] == "2026-06-05T09:45:00Z"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_manifest.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `manifest.py`**

```python
# georsct/experiment_audit/manifest.py
"""Write completion manifests with provenance chains."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .models import ArtifactRecord, CellKey


def build_manifest(
    experiment: str,
    bucket: str,
    artifacts: dict[str, ArtifactRecord],
    contracted_cells: frozenset[CellKey],
) -> dict[str, Any]:
    """Build a completion manifest linking artifacts to contracted cells."""
    artifact_entries: list[dict[str, Any]] = []
    for key, record in sorted(artifacts.items()):
        entry: dict[str, Any] = {
            "s3_key": record.s3_key,
            "exists": record.exists,
            "size_bytes": record.size_bytes,
        }
        if record.internal_timestamp:
            entry["internal_timestamp"] = record.internal_timestamp
        if record.last_modified:
            entry["s3_last_modified"] = record.last_modified
        artifact_entries.append(entry)

    return {
        "experiment": experiment,
        "bucket": bucket,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "contracted_cells": sorted(str(c) for c in contracted_cells),
        "artifact_count": len(artifact_entries),
        "artifacts": artifact_entries,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_manifest.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add georsct/experiment_audit/manifest.py tests/experiment_audit/test_manifest.py
git commit -m "feat(experiment_audit): add completion manifest writer"
```

---

## Task 10: CLI Entry Point (`cli.py` + `__main__.py`)

**Files:**
- Create: `georsct/experiment_audit/cli.py`
- Create: `georsct/experiment_audit/__main__.py`
- Modify: `georsct/experiment_audit/__init__.py`
- Test: `tests/experiment_audit/test_cli.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/experiment_audit/test_cli.py
"""Tests for CLI argument parsing (no S3 calls)."""
from __future__ import annotations

from georsct.experiment_audit.cli import build_parser


def test_parser_required_args():
    parser = build_parser()
    args = parser.parse_args([
        "--contract", "path/to/contract.yaml",
        "--s3-prefix", "results/s035/",
        "--out", "audit_output/",
    ])
    assert args.contract == "path/to/contract.yaml"
    assert args.s3_prefix == "results/s035/"
    assert args.out == "audit_output/"


def test_parser_optional_args():
    parser = build_parser()
    args = parser.parse_args([
        "--contract", "c.yaml",
        "--s3-prefix", "results/s035/",
        "--out", "out/",
        "--bucket", "my-bucket",
        "--json",
    ])
    assert args.bucket == "my-bucket"
    assert args.json_output is True


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([
        "--contract", "c.yaml",
        "--s3-prefix", "results/s035/",
        "--out", "out/",
    ])
    assert args.bucket == "swarm-floodrsct-data"
    assert args.json_output is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_cli.py -v`
Expected: FAIL — `ImportError`

- [ ] **Step 3: Implement `cli.py`**

```python
# georsct/experiment_audit/cli.py
"""CLI entry point for experiment audit."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .constants import BUCKET

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="experiment-audit",
        description="Post-flight experiment contract verification",
    )
    parser.add_argument(
        "--contract",
        required=True,
        help="Path to EXPERIMENT_CONTRACT.yaml",
    )
    parser.add_argument(
        "--s3-prefix",
        required=True,
        help="S3 prefix for results (e.g. results/s035/)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for audit report and manifest",
    )
    parser.add_argument(
        "--bucket",
        default=BUCKET,
        help=f"S3 bucket (default: {BUCKET})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output JSON only (no Markdown)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    parser = build_parser()
    args = parser.parse_args(argv)
    contract_path = Path(args.contract)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Parse contract
    from .contract import parse_contract
    log.info("Parsing contract: %s", contract_path)
    contract = parse_contract(contract_path)
    log.info("Experiment: %s, %d phases, %d contracted cells",
             contract.experiment, len(contract.phases), len(contract.cell_matrix))

    # Step 2: Build S3 inventory
    from .s3_inventory import S3Inventory
    log.info("Building S3 inventory from s3://%s/%s", args.bucket, args.s3_prefix)
    s3 = S3Inventory(bucket=args.bucket)
    s3_keys = s3.list_prefix(args.s3_prefix)
    log.info("Found %d S3 keys", len(s3_keys))

    # Download key JSONs for content inspection
    json_keys = [k for k in s3_keys if k.endswith(".json")]
    artifacts: dict = {}
    for key in json_keys:
        artifacts[key] = s3.download_json(key)
    # Also check declared output keys that might be parquets
    for phase in contract.phases:
        for out_key in phase.resolve_output_keys(contract.scenarios):
            if out_key not in artifacts:
                artifacts[out_key] = s3.check_key(out_key)

    # Step 3: Run all 6 gates
    from .models import AuditResult, CellKey
    from .validators import (
        gate_1_contract_parse,
        gate_2_declared_outputs,
        gate_3_convention_supplement,
        gate_4_cell_matrix,
        gate_5_aggregate_content,
        gate_6_ordering,
    )
    from .content_readers import extract_cells_from_runs

    all_checks = []

    log.info("Gate 1: Contract parse")
    all_checks.extend(gate_1_contract_parse(contract))

    log.info("Gate 2: Declared output existence")
    all_checks.extend(gate_2_declared_outputs(contract, artifacts))

    log.info("Gate 3: Convention supplement")
    all_checks.extend(gate_3_convention_supplement(contract, s3_keys, args.s3_prefix))

    log.info("Gate 4: Cell matrix")
    # Collect actual cells from per-scenario result JSONs
    actual_cells: set[CellKey] = set()
    for key, record in artifacts.items():
        if record.content and "runs" in record.content:
            for cm in extract_cells_from_runs(record.content):
                actual_cells.add(cm.cell)
    all_checks.extend(gate_4_cell_matrix(contract, actual_cells))

    log.info("Gate 5: Aggregate content")
    all_checks.extend(gate_5_aggregate_content(contract, artifacts))

    log.info("Gate 6: Ordering")
    all_checks.extend(gate_6_ordering(artifacts, contract.scenarios))

    audit = AuditResult(
        checks=all_checks,
        metadata={
            "experiment": contract.experiment,
            "bucket": args.bucket,
            "s3_prefix": args.s3_prefix,
            "contract_path": str(contract_path),
            "n_phases": len(contract.phases),
            "n_contracted_cells": len(contract.cell_matrix),
            "n_s3_keys": len(s3_keys),
        },
    )

    # Step 4: Write outputs
    from .report import render_json, render_markdown
    from .manifest import build_manifest

    json_path = out_dir / "experiment_audit.json"
    json_path.write_text(render_json(audit))
    log.info("Wrote %s", json_path)

    if not args.json_output:
        md_path = out_dir / "experiment_audit.md"
        md_path.write_text(render_markdown(audit))
        log.info("Wrote %s", md_path)

    manifest = build_manifest(
        experiment=contract.experiment,
        bucket=args.bucket,
        artifacts=artifacts,
        contracted_cells=contract.cell_matrix,
    )
    manifest_path = out_dir / "completion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("Wrote %s", manifest_path)

    # Print summary
    counts = audit.summary_counts()
    log.info("Audit complete: %s  PASS=%d  WARN=%d  FAIL=%d",
             audit.overall_status(), counts["PASS"], counts["WARN"], counts["FAIL"])

    return 1 if audit.overall_status() == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Create `__main__.py`**

```python
# georsct/experiment_audit/__main__.py
"""Allow running as: python -m georsct.experiment_audit"""
import sys

from .cli import main

sys.exit(main())
```

- [ ] **Step 5: Update `__init__.py`**

```python
# georsct/experiment_audit/__init__.py
"""Experiment audit -- post-flight contract verification for s035."""

from .models import AuditResult, CheckResult, Severity

__all__ = ["AuditResult", "CheckResult", "Severity"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/test_cli.py -v`
Expected: All 3 tests PASS

- [ ] **Step 7: Run full test suite**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/ -v`
Expected: All tests PASS (models: 10, constants: 7, contract: 10, s3_inventory: 6, content_readers: 6, timestamp_ordering: 6, validators: 7, report: 5, manifest: 3, cli: 3 = ~63 tests)

- [ ] **Step 8: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add georsct/experiment_audit/cli.py georsct/experiment_audit/__main__.py georsct/experiment_audit/__init__.py tests/experiment_audit/test_cli.py
git commit -m "feat(experiment_audit): add CLI entry point and __main__ runner"
```

---

## Task 11: Update `pyproject.toml`

**Files:**
- Modify: `pyproject.toml:14-20`

- [ ] **Step 1: Add optional dependency group**

Add after the existing `[project.optional-dependencies]` entries:

```toml
audit = [
    "pyyaml>=6.0",
    "boto3>=1.26",
    "swarm-auth",
]
```

- [ ] **Step 2: Verify package discovery includes experiment_audit**

The existing `[tool.setuptools.packages.find]` has `include = ["rsct*", "georsct*"]`, which will match `georsct.experiment_audit` automatically.

- [ ] **Step 3: Commit**

```bash
cd C:/Users/marti/github/rsct-geocert
git add pyproject.toml
git commit -m "feat(experiment_audit): add audit optional dependency group to pyproject.toml"
```

---

## Task 12: Integration Smoke Test

- [ ] **Step 1: Run the full test suite one final time**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m pytest tests/experiment_audit/ -v --tb=short`
Expected: All tests PASS

- [ ] **Step 2: Verify CLI help works**

Run: `cd C:/Users/marti/github/rsct-geocert && python -m georsct.experiment_audit --help`
Expected: Shows argparse help with `--contract`, `--s3-prefix`, `--out` args

- [ ] **Step 3: Verify imports work**

Run: `cd C:/Users/marti/github/rsct-geocert && python -c "from georsct.experiment_audit import AuditResult, Severity; print('OK:', list(Severity))"`
Expected: Prints `OK:` followed by all severity values

- [ ] **Step 4: Final commit if any fixups needed**

```bash
cd C:/Users/marti/github/rsct-geocert
git add -A
git commit -m "fix(experiment_audit): integration fixups from smoke test"
```
