#!/usr/bin/env python3
"""
verify_experiment_status.py -- Audit EXPERIMENT_STATUS.yaml against S3.

Reads the experiment registry and checks every artifact_key against S3.
Reports mismatches: COMPLETED but missing, DESIGNED but artifact exists.

Usage:
    python verify_experiment_status.py              # audit S3 vs status
    python verify_experiment_status.py --fix         # update status in-place
    python verify_experiment_status.py --next-steps  # show unblocked phases
    python verify_experiment_status.py --json        # machine-readable output
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import boto3
import yaml

FLOODRSCT_ROOT = Path(__file__).resolve().parents[1]
STATUS_FILE = FLOODRSCT_ROOT / "exp" / "s035-model-ladder" / "EXPERIMENT_STATUS.yaml"
CONTRACT_FILE = FLOODRSCT_ROOT / "exp" / "s035-model-ladder" / "EXPERIMENT_CONTRACT.yaml"


def get_s3_client():
    session = boto3.Session(profile_name="nsc-swarm")
    return session.client("s3")


def check_key_exists(s3, bucket: str, key: str) -> bool:
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


def collect_artifact_checks(phases: dict, prefix: str = "") -> list[dict]:
    """Recursively collect all artifact_keys with their status and path."""
    checks = []
    for name, value in phases.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            if "artifact_keys" in value:
                status = value.get("status", "UNKNOWN")
                for key in value["artifact_keys"]:
                    checks.append({
                        "phase": path,
                        "status": status,
                        "artifact_key": key,
                    })
            # Recurse into nested phases
            for k, v in value.items():
                if isinstance(v, dict) and k not in (
                    "artifact_keys", "notes", "paper_claims"
                ):
                    checks.extend(
                        collect_artifact_checks({k: v}, prefix=path)
                    )
    return checks


def get_phase_statuses(phases: dict, prefix: str = "") -> dict[str, str]:
    """Flatten nested phases dict into {phase_path: status}."""
    result = {}
    for name, value in phases.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            status = value.get("status")
            if status:
                result[path] = status
            for k, v in value.items():
                if isinstance(v, dict) and k not in (
                    "artifact_keys", "notes", "paper_claims"
                ):
                    result.update(get_phase_statuses({k: v}, prefix=path))
    return result


def get_phase_details(phases: dict, prefix: str = "") -> dict[str, dict]:
    """Flatten nested phases dict into {phase_key: full_detail_dict}."""
    result = {}
    for name, value in phases.items():
        path = f"{prefix}.{name}" if prefix else name
        if isinstance(value, dict):
            if value.get("status"):
                result[name] = value
                result[path] = value
            for k, v in value.items():
                if isinstance(v, dict) and k not in (
                    "artifact_keys", "notes", "paper_claims"
                ):
                    result.update(get_phase_details({k: v}, path))
    return result


def propagate_staleness(flat_status: dict[str, str],
                        flat_details: dict[str, dict],
                        contract_map: dict[str, dict]) -> dict[str, list[str]]:
    """Walk the dependency graph and propagate STALE downstream.

    If a phase is STALE or was re-run (verified date newer than downstream),
    all downstream phases that depend on it become STALE too.

    Returns: {phase_id: [list of upstream causes]} for newly-stale phases.
    """
    # Build reverse map: phase_id -> list of phases that depend on it
    downstream_of: dict[str, list[str]] = {}
    for pid, phase in contract_map.items():
        for dep in phase.get("depends_on", []):
            downstream_of.setdefault(dep, []).append(pid)

    # Collect root stale phases (explicitly marked STALE or FAILED)
    stale_roots = set()
    for pid in contract_map:
        status = flat_status.get(pid, "UNKNOWN")
        if status in ("STALE", "FAILED"):
            stale_roots.add(pid)

    # Also detect timestamp-based staleness: if upstream verified > downstream
    # verified, downstream is implicitly stale
    for pid, phase in contract_map.items():
        if flat_status.get(pid) != "COMPLETED":
            continue
        downstream_detail = flat_details.get(pid, {})
        downstream_verified = downstream_detail.get("verified", "")

        for dep in phase.get("depends_on", []):
            if flat_status.get(dep) != "COMPLETED":
                continue
            upstream_detail = flat_details.get(dep, {})
            upstream_verified = upstream_detail.get("verified", "")

            # If upstream was verified MORE RECENTLY than downstream,
            # downstream may be stale (upstream was re-run after downstream)
            if upstream_verified and downstream_verified:
                if str(upstream_verified) > str(downstream_verified):
                    stale_roots.add(pid)

    # BFS: propagate staleness downstream through dependency edges
    stale_causes: dict[str, list[str]] = {}
    queue = list(stale_roots)
    visited = set()

    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        for child in downstream_of.get(current, []):
            child_status = flat_status.get(child, "UNKNOWN")
            # Only propagate to phases that think they're COMPLETED
            if child_status == "COMPLETED":
                if child not in stale_causes:
                    stale_causes[child] = []
                stale_causes[child].append(current)
                flat_status[child] = "STALE"
                queue.append(child)
            elif child_status == "STALE":
                # Already stale, but track the cause chain
                if child not in stale_causes:
                    stale_causes[child] = []
                if current not in stale_causes[child]:
                    stale_causes[child].append(current)
                queue.append(child)

    return stale_causes


def compute_next_steps(status_phases: dict[str, str],
                       contract_phases: list[dict],
                       phase_details: dict[str, dict] | None = None,
                       ) -> tuple[list[dict], dict[str, list[str]]]:
    """Determine which phases are unblocked but not yet completed.

    A phase is unblocked when:
    - All depends_on phases have status COMPLETED (not STALE)
    - The phase itself is NOT COMPLETED

    Also propagates staleness: if an upstream phase is STALE or was re-run
    more recently than a downstream phase, the downstream is implicitly STALE.

    Returns: (steps, stale_causes)
    """
    # Build a map from phase_id -> contract entry
    contract_map = {}
    for phase in contract_phases:
        pid = phase.get("phase_id")
        if pid:
            contract_map[pid] = phase

    # Flatten status to simple name -> status (strip nested prefixes)
    # Also normalize: dots to underscores, so "certificates.r0" matches
    # "certificates_r0" from the contract.
    flat_status = {}
    for path, status in status_phases.items():
        # Use the last component as a simple key
        simple = path.split(".")[-1]
        flat_status[simple] = status
        # Also keep full path
        flat_status[path] = status
        # Normalize dots to underscores for cross-file matching
        normalized = path.replace(".", "_")
        flat_status[normalized] = status

    # Known aliases: STATUS names that differ from CONTRACT phase_ids
    _ALIASES = {
        "r1_spatial": "r1_hydrology",
        "r2_temporal": "r2_temporal",
        "diagnostics": "diagnostics_r0",  # single "diagnostics" = all done
    }
    for alias, canonical in _ALIASES.items():
        if alias in flat_status and canonical not in flat_status:
            flat_status[canonical] = flat_status[alias]

    # If "diagnostics" is COMPLETED, propagate to all levels
    if flat_status.get("diagnostics") == "COMPLETED":
        for level in ("diagnostics_r0", "diagnostics_r1", "diagnostics_r2"):
            flat_status.setdefault(level, "COMPLETED")

    # Propagate staleness through the dependency graph
    flat_details = phase_details or {}
    stale_causes = propagate_staleness(
        flat_status, flat_details, contract_map
    )

    unblocked = []
    for pid, phase in contract_map.items():
        # Skip if completed AND not stale
        current = flat_status.get(pid, "UNKNOWN")
        if current == "COMPLETED":
            continue

        deps = phase.get("depends_on", [])
        all_deps_met = True
        missing_deps = []
        for dep in deps:
            dep_status = flat_status.get(dep, "UNKNOWN")
            if dep_status != "COMPLETED":
                all_deps_met = False
                reason = dep_status
                if dep in stale_causes:
                    chain = " <- ".join(stale_causes[dep])
                    reason = f"STALE via {chain}"
                missing_deps.append(f"{dep} ({reason})")

        prereqs = phase.get("prerequisites", [])

        unblocked.append({
            "phase_id": pid,
            "description": phase.get("description", ""),
            "current_status": current,
            "deps_met": all_deps_met,
            "missing_deps": missing_deps,
            "per_scenario": phase.get("per_scenario", False),
            "prerequisites": prereqs,
            "script": phase.get("script", ""),
            "actionable": all_deps_met and len(missing_deps) == 0,
            "stale_cause": stale_causes.get(pid, []),
        })

    # Sort: actionable first, then by phase_id
    unblocked.sort(key=lambda x: (not x["actionable"], x["phase_id"]))
    return unblocked, stale_causes


def print_next_steps(steps: list[dict],
                     stale_causes: dict[str, list[str]]) -> None:
    """Human-readable next-steps output."""
    actionable = [s for s in steps if s["actionable"]]
    blocked = [s for s in steps if not s["actionable"]]

    # Show staleness cascade first if any auto-detected
    if stale_causes:
        print(f"\n{'='*70}")
        print("STALENESS CASCADE (auto-detected from dependency graph)")
        print(f"{'='*70}")
        for phase, causes in sorted(stale_causes.items()):
            chain = " <- ".join(causes)
            print(f"  [!] {phase} is STALE because: {chain}")

    print(f"\n{'='*70}")
    print("UNBLOCKED PHASES (ready to launch)")
    print(f"{'='*70}")

    if not actionable:
        print("  (none -- all phases either COMPLETED or BLOCKED)")
    else:
        for step in actionable:
            scenario_tag = " [per-scenario]" if step["per_scenario"] else ""
            stale_tag = ""
            if step["stale_cause"]:
                stale_tag = f" [STALE: re-run needed]"
            print(f"\n  >> {step['phase_id']}{scenario_tag}{stale_tag}")
            print(f"     {step['description']}")
            print(f"     status: {step['current_status']}")
            print(f"     script: {step['script']}")
            if step["stale_cause"]:
                chain = " <- ".join(step["stale_cause"])
                print(f"     stale because: {chain}")
            if step["prerequisites"]:
                for p in step["prerequisites"]:
                    print(f"     prereq: {p}")

    if blocked:
        print(f"\n{'='*70}")
        print("BLOCKED PHASES (dependencies not met)")
        print(f"{'='*70}")
        for step in blocked:
            print(f"\n  -- {step['phase_id']} ({step['current_status']})")
            for dep in step["missing_deps"]:
                print(f"     waiting on: {dep}")


def main():
    parser = argparse.ArgumentParser(
        description="Verify experiment status against S3 artifacts"
    )
    parser.add_argument("--fix", action="store_true",
                        help="Update EXPERIMENT_STATUS.yaml with findings")
    parser.add_argument("--next-steps", action="store_true",
                        help="Show unblocked phases ready to launch")
    parser.add_argument("--json", action="store_true",
                        help="Machine-readable JSON output")
    args = parser.parse_args()

    if not STATUS_FILE.exists():
        print(f"ERROR: {STATUS_FILE} not found")
        sys.exit(1)

    with open(STATUS_FILE) as f:
        registry = yaml.safe_load(f)

    bucket = registry["s3_bucket"]
    phases = registry.get("phases", {})

    # --next-steps mode: compute unblocked phases from dependency graph
    if args.next_steps:
        phase_statuses = get_phase_statuses(phases)
        phase_details = get_phase_details(phases)
        contract_phases = []
        if CONTRACT_FILE.exists():
            with open(CONTRACT_FILE) as f:
                contract = yaml.safe_load(f)
            contract_phases = contract.get("phases", [])
            if not isinstance(contract_phases, list):
                contract_phases = []

        steps, stale_causes = compute_next_steps(
            phase_statuses, contract_phases, phase_details
        )

        if args.json:
            output = {
                "experiment": registry["experiment"],
                "last_audit": registry.get("last_audit"),
                "total_incomplete": len(steps),
                "actionable": [s for s in steps if s["actionable"]],
                "blocked": [s for s in steps if not s["actionable"]],
                "stale_cascade": {
                    k: v for k, v in stale_causes.items()
                } if stale_causes else {},
            }
            print(json.dumps(output, indent=2, default=str))
        else:
            print(f"Experiment: {registry['experiment']}")
            print(f"Last audit: {registry.get('last_audit', 'never')}")
            print_next_steps(steps, stale_causes)
        return

    s3 = get_s3_client()

    checks = collect_artifact_checks(phases)

    print(f"Experiment: {registry['experiment']}")
    print(f"Last audit: {registry.get('last_audit', 'never')}")
    print(f"Checking {len(checks)} artifact keys against s3://{bucket}/")
    print(f"{'='*70}")

    ok = 0
    mismatch = 0
    missing = 0
    surprise = 0

    for check in checks:
        exists = check_key_exists(s3, bucket, check["artifact_key"])
        status = check["status"]

        if status == "COMPLETED" and exists:
            ok += 1
            marker = "OK"
        elif status == "COMPLETED" and not exists:
            missing += 1
            marker = "MISSING"
        elif status == "DESIGNED" and exists:
            surprise += 1
            marker = "SURPRISE"
        elif status == "DESIGNED" and not exists:
            ok += 1
            marker = "OK"
        else:
            mismatch += 1
            marker = f"CHECK ({status}, exists={exists})"

        icon = {
            "OK": "[+]", "MISSING": "[!]", "SURPRISE": "[?]",
        }.get(marker, "[~]")

        if marker != "OK":
            print(f"  {icon} {check['phase']}")
            print(f"      status={status}, s3_exists={exists}")
            print(f"      key={check['artifact_key']}")

    print(f"\n{'='*70}")
    print(f"  OK:       {ok}")
    print(f"  MISSING:  {missing}  (status=COMPLETED but artifact gone)")
    print(f"  SURPRISE: {surprise}  (status=DESIGNED but artifact exists)")
    print(f"  OTHER:    {mismatch}")
    print(f"{'='*70}")

    if missing == 0 and surprise == 0 and mismatch == 0:
        print("  ALL CLEAR -- registry matches S3")
    else:
        print("  ACTION NEEDED -- update EXPERIMENT_STATUS.yaml")

    # Also check paper claims
    claims = registry.get("paper_claims", {})
    if claims:
        print(f"\nPaper Claim Status:")
        for section, info in claims.items():
            status = info.get("status", "UNKNOWN")
            icon = {"GREEN": "[+]", "YELLOW": "[~]", "RED": "[!]"}.get(
                status, "[?]"
            )
            print(f"  {icon} {section}: {status}")
            if status in ("RED", "YELLOW"):
                print(f"      {info.get('reason', '')}")


if __name__ == "__main__":
    main()
