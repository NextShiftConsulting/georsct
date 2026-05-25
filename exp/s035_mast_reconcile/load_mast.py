"""
load_mast.py — Load all MAST annotated traces from the cloned repo.

MAST dataset: https://github.com/multi-agent-systems-failure-taxonomy/MAST
Paper: Cemri et al. 2025, "Why Do Multi-Agent LLM Systems Fail?"

Returns a list of dicts, each with:
  - instance_id: str
  - source: "AG2" | "HyperAgent"
  - problem_statement: str
  - trajectory: list[dict] (normalized to {content, role, name} format)
  - annotations: dict[str, bool] (22 failure-mode labels)
  - n_steps: int
"""

import json
import subprocess
from pathlib import Path

MAST_REPO = Path(__file__).parent / "mast_repo"


def _git_show(path: str) -> str:
    """Read file content from git HEAD (bypasses checkout issues on Windows)."""
    result = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        capture_output=True, text=True, cwd=MAST_REPO,
        encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise FileNotFoundError(f"git show failed: {path}")
    return result.stdout


def _list_human_traces(prefix: str) -> list:
    """List all *_human.json files under a traces/ subdirectory."""
    result = subprocess.run(
        ["git", "ls-tree", "--name-only", "HEAD", f"{prefix}/"],
        capture_output=True, text=True, cwd=MAST_REPO
    )
    return [f for f in result.stdout.strip().split("\n")
            if f.endswith("_human.json")]


def _normalize_trajectory(raw_traj: list, source: str) -> list:
    """
    Normalize trajectory to consistent format.

    AG2: list of {content: str|list, role: str, name: str}
    HyperAgent: list of str (log lines)
    """
    if not raw_traj:
        return []

    if isinstance(raw_traj[0], dict):
        # AG2 format — already structured
        steps = []
        for s in raw_traj:
            content = s.get("content", "")
            if isinstance(content, list):
                content = "\n".join(str(x) for x in content)
            steps.append({
                "content": str(content),
                "role": s.get("role", "unknown"),
                "name": s.get("name", ""),
            })
        return steps
    else:
        # HyperAgent format — flat log lines, group into logical steps
        # Each line is typically: "AgentName_instance - LEVEL - message"
        steps = []
        for line in raw_traj:
            line = str(line)
            steps.append({
                "content": line,
                "role": "log",
                "name": "",  # parsed downstream if needed
            })
        return steps


def load_all_traces() -> list:
    """
    Load all 61 annotated MAST traces.

    Returns list of normalized trace dicts.
    """
    traces = []
    errors = []

    for source in ["AG2", "HyperAgent"]:
        prefix = f"traces/{source}"
        files = _list_human_traces(prefix)

        for fpath in files:
            try:
                raw = _git_show(fpath)
                data = json.loads(raw)

                raw_traj = data.get("trajectory", [])
                trajectory = _normalize_trajectory(raw_traj, source)

                # Parse annotations (22 binary failure-mode labels)
                options = data.get("note", {}).get("options", {})
                annotations = {k: (v == "yes") for k, v in options.items()}

                traces.append({
                    "instance_id": data.get("instance_id", fpath),
                    "source": source,
                    "problem_statement": data.get("problem_statement", ""),
                    "trajectory": trajectory,
                    "annotations": annotations,
                    "n_steps": len(trajectory),
                })
            except Exception as e:
                errors.append((fpath, str(e)))

    if errors:
        print(f"WARNING: {len(errors)} traces failed to load:")
        for path, err in errors[:5]:
            print(f"  {path}: {err}")

    return traces


# MAST failure-mode categories (from paper Table 1)
MAST_CATEGORIES = {
    "FC1_Specification": [
        "Fail to detect ambiguities/contradictions",
        "Proceed with incorrect assumptions",
        "Fail to elicit clarification",
        "Tendency to overachieve",
        "Underperform by waiting on instructions",
    ],
    "FC2_Misalignment": [
        "Withholding relevant information",
        "Ignoring good suggestions from other agent",
        "Misalignment between internal thoughts and response message",
        "Blurring role",
        "Derailing from task objectives",
    ],
    "FC3_Verification": [
        "No attempt to verify outcome",
        "Evaluator agent fails to be critical",
        "Invented content",
        "Step repetition",
        "Trajectory restart",
        "Discontinued reasoning",
        "Unaware of stopping conditions",
        "Claiming that a task is done while it is not true.",
        "Redundant actions",
        "Inadequate tool selection",
        "Incorrect tool usage",
        "Poor adherence to specified constraints",
    ],
}

# Invert for lookup
MODE_TO_CATEGORY = {}
for cat, modes in MAST_CATEGORIES.items():
    for mode in modes:
        MODE_TO_CATEGORY[mode] = cat


def get_active_categories(annotations: dict) -> set:
    """Return which MAST categories are active for a trace."""
    cats = set()
    for mode, active in annotations.items():
        if active and mode in MODE_TO_CATEGORY:
            cats.add(MODE_TO_CATEGORY[mode])
    return cats


if __name__ == "__main__":
    traces = load_all_traces()
    print(f"Loaded {len(traces)} traces")
    print(f"  AG2: {sum(1 for t in traces if t['source'] == 'AG2')}")
    print(f"  HyperAgent: {sum(1 for t in traces if t['source'] == 'HyperAgent')}")
    print(f"  Total steps: {sum(t['n_steps'] for t in traces)}")
    print(f"  Annotation keys: {len(traces[0]['annotations'])}")

    # Show annotation frequencies
    from collections import Counter
    freq = Counter()
    for t in traces:
        for mode, active in t["annotations"].items():
            if active:
                freq[mode] += 1

    print(f"\nFailure mode frequencies (n={len(traces)}):")
    for mode, count in freq.most_common():
        print(f"  {count:>3} ({count/len(traces)*100:4.0f}%) {mode}")
