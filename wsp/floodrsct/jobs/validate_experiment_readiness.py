"""
validate_experiment_readiness.py -- Pre-flight contract validation for s035.

Reads EXPERIMENT_CONTRACT.yaml and validates three layers per phase:
  Layer 1: Packages      -- pip packages that must be importable
  Layer 2: Features      -- columns that must exist in the input parquet
  Layer 3: S3 Artifacts  -- S3 keys that must exist before launch

Importable by _launcher_base.py (via preflight()) and standalone:
    python validate_experiment_readiness.py --phase r0_baseline --scenario houston
    python validate_experiment_readiness.py --all
"""

import importlib.util
import logging
import sys
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path

import yaml
from botocore.exceptions import ClientError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BUCKET = "swarm-floodrsct-data"

CONTRACT_PATH = (
    Path(__file__).resolve().parent.parent
    / "exp" / "s035-model-ladder" / "EXPERIMENT_CONTRACT.yaml"
)

# Duplicated from _coverage_common.py to keep import chain simple (this module
# is imported by _launcher_base.py in scripts/, not jobs/).
OUTPUT_KEYS = {
    "houston": "processed/houston/houston_event_features.parquet",
    "new_orleans": "processed/new_orleans/no_event_features.parquet",
    "nyc": "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
}

# Short keys used in supplement filenames (derived from OUTPUT_KEYS filenames).
SCENARIO_KEYS = {
    sc: Path(path).stem.replace("_event_features", "")
    for sc, path in OUTPUT_KEYS.items()
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class Status(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    layer: str          # "packages", "features", "s3_artifacts"
    name: str
    status: Status
    message: str

    def __str__(self) -> str:
        return f"[{self.layer:12s}] [{self.status.value:4s}] {self.name}: {self.message}"


# ---------------------------------------------------------------------------
# Contract helpers
# ---------------------------------------------------------------------------

def load_contract(path: Path | None = None) -> dict:
    p = path or CONTRACT_PATH
    if not p.exists():
        raise FileNotFoundError(f"Contract not found: {p}")
    with open(p) as f:
        return yaml.safe_load(f)


def get_phase(contract: dict, phase_id: str) -> dict:
    for phase in contract["phases"]:
        if phase["phase_id"] == phase_id:
            return phase
    available = [p["phase_id"] for p in contract["phases"]]
    raise ValueError(f"Unknown phase_id: {phase_id}. Available: {available}")


def resolve_s3_key(key: str, scenario: str) -> str:
    """Expand template variables in an S3 key."""
    if key == "assembled_parquet":
        return OUTPUT_KEYS[scenario]
    result = key.replace("{scenario}", scenario)
    if "{scenario_key}" in result:
        result = result.replace("{scenario_key}", SCENARIO_KEYS.get(scenario, scenario))
    return result


# ---------------------------------------------------------------------------
# Layer 1: Package validation
# ---------------------------------------------------------------------------

def validate_packages(phase: dict) -> list[CheckResult]:
    results = []
    for pkg in phase.get("packages") or []:
        import_name = pkg.replace("-", "_")
        if import_name == "scikit_learn":
            import_name = "sklearn"
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            results.append(CheckResult("packages", pkg, Status.PASS, "importable"))
        else:
            results.append(CheckResult("packages", pkg, Status.WARN,
                "not importable locally (pip-installed on SageMaker)"))
    return results


# ---------------------------------------------------------------------------
# Layer 2: Feature validation
# ---------------------------------------------------------------------------

def validate_features(phase: dict, scenario: str, s3) -> list[CheckResult]:
    features_spec = phase.get("features")
    if not features_spec:
        return [CheckResult("features", "(none)", Status.SKIP, "no feature requirements")]

    source = features_spec.get("source")
    if source != "assembled_parquet":
        return [CheckResult("features", source or "(null)", Status.SKIP, "unknown source type")]

    s3_key = OUTPUT_KEYS.get(scenario)
    if not s3_key:
        return [CheckResult("features", source, Status.FAIL, f"unknown scenario: {scenario}")]

    try:
        import pyarrow.parquet as pq
        obj = s3.get_object(Bucket=BUCKET, Key=s3_key)
        schema = pq.read_schema(BytesIO(obj["Body"].read()))
        cols = set(schema.names)
    except ClientError:
        return [CheckResult("features", s3_key, Status.FAIL, "parquet not found on S3")]
    except Exception as e:
        return [CheckResult("features", s3_key, Status.FAIL, f"cannot read schema: {e}")]

    results = [CheckResult("features", s3_key, Status.PASS, f"{len(cols)} columns")]
    for col in features_spec.get("required", []):
        st = Status.PASS if col in cols else Status.FAIL
        msg = "present" if col in cols else "MISSING"
        results.append(CheckResult("features", col, st, msg))
    for col in features_spec.get("targets", []):
        st = Status.PASS if col in cols else Status.FAIL
        msg = "present" if col in cols else "MISSING"
        results.append(CheckResult("features", f"target:{col}", st, msg))
    return results


# ---------------------------------------------------------------------------
# Layer 3: S3 artifact validation
# ---------------------------------------------------------------------------

def _check_s3_key(s3, key: str, optional: bool, prefix: bool) -> CheckResult:
    """Check a single S3 key or prefix."""
    try:
        if prefix:
            resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=key, MaxKeys=1)
            if resp.get("Contents"):
                return CheckResult("s3_artifacts", key, Status.PASS, "prefix exists")
            st = Status.WARN if optional else Status.FAIL
            return CheckResult("s3_artifacts", key, st,
                "prefix empty" + (" (optional)" if optional else ""))
        else:
            s3.head_object(Bucket=BUCKET, Key=key)
            return CheckResult("s3_artifacts", key, Status.PASS, "exists")
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("404", "NoSuchKey"):
            st = Status.WARN if optional else Status.FAIL
            return CheckResult("s3_artifacts", key, st,
                "not found" + (" (optional)" if optional else ""))
        return CheckResult("s3_artifacts", key, Status.FAIL, f"S3 error: {code}")


def validate_s3_artifacts(
    phase: dict, scenario: str | None, s3, modelable: list[str],
) -> list[CheckResult]:
    results = []
    for artifact in phase.get("s3_artifacts") or []:
        key_template = artifact["key"]
        optional = artifact.get("optional", False)
        is_prefix = artifact.get("prefix", False)
        per_scenario = artifact.get("per_scenario", False)

        if per_scenario:
            scen_list = artifact.get("scenarios", modelable)
            if scenario and scenario in scen_list:
                targets = [scenario]
            elif scenario and scenario not in scen_list:
                results.append(CheckResult("s3_artifacts", key_template, Status.SKIP,
                    f"not required for {scenario}"))
                continue
            else:
                targets = scen_list
            for sc in targets:
                resolved = resolve_s3_key(key_template, sc)
                results.append(_check_s3_key(s3, resolved, optional, is_prefix))
        else:
            resolved = resolve_s3_key(key_template, scenario or "")
            results.append(_check_s3_key(s3, resolved, optional, is_prefix))

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    results: list[CheckResult], phase_id: str, scenario: str | None,
) -> tuple[int, int, int]:
    pass_n = sum(1 for r in results if r.status == Status.PASS)
    fail_n = sum(1 for r in results if r.status == Status.FAIL)
    warn_n = sum(1 for r in results if r.status == Status.WARN)
    skip_n = sum(1 for r in results if r.status == Status.SKIP)

    label = phase_id + (f" / {scenario}" if scenario else "")
    print(f"\n{'=' * 72}")
    print(f"Experiment Readiness: {label}")
    print(f"{'=' * 72}")

    for layer in ("packages", "features", "s3_artifacts"):
        layer_results = [r for r in results if r.layer == layer]
        if not layer_results:
            continue
        print(f"\n--- {layer} ---")
        for r in layer_results:
            print(f"  {r}")

    print(f"\n{'=' * 72}")
    print(f"PASS: {pass_n}  FAIL: {fail_n}  WARN: {warn_n}  SKIP: {skip_n}")
    if fail_n > 0:
        print("VERDICT: BLOCKED -- fix FAILs before launching")
    elif warn_n > 0:
        print("VERDICT: CONDITIONAL PASS -- review WARNs")
    else:
        print("VERDICT: CLEAR")
    print(f"{'=' * 72}\n")

    return pass_n, fail_n, warn_n


# ---------------------------------------------------------------------------
# Orchestrator (importable by launchers)
# ---------------------------------------------------------------------------

def preflight(
    phase_id: str, scenario: str | None = None, s3=None,
) -> tuple[int, int, int]:
    """Run pre-flight validation for a phase. Returns (pass, fail, warn)."""
    contract = load_contract()
    phase = get_phase(contract, phase_id)
    modelable = contract.get("modelable_scenarios", [])

    if s3 is None:
        import boto3
        from swarm_auth import get_aws_credentials
        aws = get_aws_credentials()
        aws.pop("region_name", None)
        s3 = boto3.client("s3", region_name="us-east-1", **aws)

    results: list[CheckResult] = []
    results.extend(validate_packages(phase))

    if scenario:
        results.extend(validate_features(phase, scenario, s3))
    elif not phase.get("per_scenario"):
        results.extend(validate_features(phase, modelable[0] if modelable else "houston", s3))
    else:
        results.append(CheckResult("features", "(skipped)", Status.SKIP,
            "per-scenario phase requires --scenario"))

    results.extend(validate_s3_artifacts(phase, scenario, s3, modelable))

    return print_report(results, phase_id, scenario)


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    import boto3
    from swarm_auth import get_aws_credentials

    parser = argparse.ArgumentParser(description="Pre-flight validation for s035 phases")
    parser.add_argument("--phase", help="Phase ID (e.g. r0_baseline)")
    parser.add_argument("--scenario", help="Scenario (e.g. houston)")
    parser.add_argument("--all", action="store_true", help="Validate all phases")
    parser.add_argument("--contract", type=str, default=None)
    args = parser.parse_args()

    if not args.phase and not args.all:
        parser.error("specify --phase or --all")

    aws = get_aws_credentials()
    aws.pop("region_name", None)
    s3 = boto3.client("s3", region_name="us-east-1", **aws)

    total_fails = 0
    if args.all:
        contract = load_contract(Path(args.contract) if args.contract else None)
        for phase in contract["phases"]:
            pid = phase["phase_id"]
            modelable = contract.get("modelable_scenarios", [])
            if phase.get("per_scenario"):
                for sc in (modelable if not args.scenario else [args.scenario]):
                    _, fails, _ = preflight(pid, sc, s3)
                    total_fails += fails
            else:
                _, fails, _ = preflight(pid, args.scenario, s3)
                total_fails += fails
    else:
        _, fails, _ = preflight(args.phase, args.scenario, s3)
        total_fails += fails

    sys.exit(1 if total_fails > 0 else 0)


if __name__ == "__main__":
    main()
