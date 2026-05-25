"""
signals.py — Extract structural signals from MAST multi-agent traces.

Each signal is a scalar computed directly from the trajectory structure.
No labels, no classification — just measurement.
"""

import re
from collections import Counter
import numpy as np


def extract_signals(trace: dict) -> dict:
    """
    Compute structural signals from a normalized trace.

    Args:
        trace: dict with 'trajectory' (list of {content, role, name}),
               'source', 'n_steps'

    Returns:
        dict of signal_name -> float
    """
    traj = trace["trajectory"]
    n_steps = trace["n_steps"]

    if n_steps == 0:
        return _empty_signals()

    contents = [s["content"] for s in traj]
    names = [s["name"] for s in traj]
    roles = [s["role"] for s in traj]
    lengths = [len(c) for c in contents]

    total_content = "\n".join(contents)
    total_chars = sum(lengths)

    # ------------------------------------------------------------------
    # 1. REPETITION: How much of the trajectory is repeated content?
    # ------------------------------------------------------------------
    # Hash first 500 chars of each step; measure unique ratio
    hashes = [hash(c[:500]) for c in contents]
    unique_steps = len(set(hashes))
    repetition_ratio = 1.0 - (unique_steps / n_steps)

    # Also: consecutive duplicate detection (stronger signal for loops)
    consecutive_dupes = sum(
        1 for i in range(1, len(hashes)) if hashes[i] == hashes[i-1]
    )
    consecutive_dupe_rate = consecutive_dupes / max(n_steps - 1, 1)

    # ------------------------------------------------------------------
    # 2. AGENT STRUCTURE: How many agents, who dominates?
    # ------------------------------------------------------------------
    active_names = [n for n in names if n]
    unique_agents = set(active_names)
    n_agents = max(len(unique_agents), 1)

    name_freq = Counter(active_names)
    if name_freq:
        dominance = max(name_freq.values()) / sum(name_freq.values())
        entropy_agents = -sum(
            (c / sum(name_freq.values())) * np.log2(c / sum(name_freq.values()))
            for c in name_freq.values()
        )
    else:
        dominance = 1.0
        entropy_agents = 0.0

    # ------------------------------------------------------------------
    # 3. HANDOFF: How often does control pass between agents?
    # ------------------------------------------------------------------
    handoffs = sum(
        1 for i in range(1, len(active_names))
        if active_names[i] != active_names[i-1]
    ) if len(active_names) > 1 else 0
    handoff_rate = handoffs / max(len(active_names) - 1, 1)

    # ------------------------------------------------------------------
    # 4. ERROR DENSITY: Explicit error/failure markers per step
    # ------------------------------------------------------------------
    error_pattern = re.compile(
        r'\b(error|exception|traceback|failed|failure|crash|bug|broken)\b',
        re.IGNORECASE
    )
    error_count = len(error_pattern.findall(total_content))
    error_density = error_count / n_steps

    # ------------------------------------------------------------------
    # 5. VERIFICATION: Checking/validation language per step
    # ------------------------------------------------------------------
    verify_pattern = re.compile(
        r'\b(verify|check|confirm|validate|test|assert|review|ensure|correct)\b',
        re.IGNORECASE
    )
    verify_count = len(verify_pattern.findall(total_content))
    verify_density = verify_count / n_steps

    # ------------------------------------------------------------------
    # 6. CODE DENSITY: Structured output markers per step
    # ------------------------------------------------------------------
    code_pattern = re.compile(
        r'(```|def \w+|class \w+|import \w+|from \w+|print\(|return )'
    )
    code_count = len(code_pattern.findall(total_content))
    code_density = code_count / n_steps

    # ------------------------------------------------------------------
    # 7. TRAJECTORY SHAPE: Length distribution characteristics
    # ------------------------------------------------------------------
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    length_cv = std_length / max(mean_length, 1)

    # Growth: does the trajectory expand or contract?
    if n_steps >= 4:
        first_half = sum(lengths[:n_steps // 2])
        second_half = sum(lengths[n_steps // 2:])
        growth_ratio = second_half / max(first_half, 1)
    else:
        growth_ratio = 1.0

    # ------------------------------------------------------------------
    # 8. TERMINATION: Does the trace reach a clear endpoint?
    # ------------------------------------------------------------------
    last_content = contents[-1] if contents else ""
    termination_pattern = re.compile(
        r'\b(TERMINATE|DONE|final answer|solution is|result:|concluded|finished)\b',
        re.IGNORECASE
    )
    has_termination = float(bool(termination_pattern.search(last_content)))

    # Also check if last 3 steps have termination signals
    tail = " ".join(contents[-3:]) if len(contents) >= 3 else last_content
    tail_termination = float(bool(termination_pattern.search(tail)))

    # ------------------------------------------------------------------
    # 9. ROLE PATTERN: How structured is the role sequence?
    # ------------------------------------------------------------------
    role_bigrams = [(roles[i], roles[i+1]) for i in range(len(roles) - 1)]
    if role_bigrams:
        bigram_freq = Counter(role_bigrams)
        max_bigram = max(bigram_freq.values()) / len(role_bigrams)
    else:
        max_bigram = 0.0

    return {
        # Repetition / looping
        "repetition_ratio": repetition_ratio,
        "consecutive_dupe_rate": consecutive_dupe_rate,
        # Agent structure
        "n_agents": n_agents,
        "dominance": dominance,
        "entropy_agents": entropy_agents,
        # Coordination
        "handoff_rate": handoff_rate,
        # Content markers
        "error_density": error_density,
        "verify_density": verify_density,
        "code_density": code_density,
        # Shape
        "n_steps": n_steps,
        "total_chars": total_chars,
        "mean_step_length": mean_length,
        "length_cv": length_cv,
        "growth_ratio": growth_ratio,
        # Termination
        "has_termination": has_termination,
        "tail_termination": tail_termination,
        # Role structure
        "max_role_bigram": max_bigram,
    }


def _empty_signals() -> dict:
    return {k: 0.0 for k in [
        "repetition_ratio", "consecutive_dupe_rate",
        "n_agents", "dominance", "entropy_agents",
        "handoff_rate", "error_density", "verify_density", "code_density",
        "n_steps", "total_chars", "mean_step_length", "length_cv",
        "growth_ratio", "has_termination", "tail_termination", "max_role_bigram",
    ]}
