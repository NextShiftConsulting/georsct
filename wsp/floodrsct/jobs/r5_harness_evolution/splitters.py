"""Train/validation/test split strategies for R5 harness evolution.

Supports spatial-block holdout, leave-scenario-out, and leave-event-out
splits. Neighboring ZCTAs leak spatial patterns, so random splits are
insufficient -- use spatial blocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class EvolutionSplit:
    """Train/validation/test split for harness evolution."""
    train_ids: list[str] = field(default_factory=list)
    validation_ids: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    split_method: str = ""
    metadata: dict = field(default_factory=dict)


def leave_scenario_out(
    scenario_zctas: dict[str, list[str]],
    train_scenarios: list[str],
    test_scenarios: list[str],
    validation_frac: float = 0.2,
    seed: int = 42,
) -> EvolutionSplit:
    """Split by scenario: train on some cities, test on others.

    Within train scenarios, hold out validation_frac of ZCTAs.
    """
    rng = np.random.default_rng(seed)

    train_all = []
    for s in train_scenarios:
        train_all.extend(scenario_zctas.get(s, []))

    test_ids = []
    for s in test_scenarios:
        test_ids.extend(scenario_zctas.get(s, []))

    # Split train into train/validation
    rng.shuffle(train_all)
    n_val = max(1, int(len(train_all) * validation_frac))
    validation_ids = train_all[:n_val]
    train_ids = train_all[n_val:]

    return EvolutionSplit(
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        split_method="leave_scenario_out",
        metadata={
            "train_scenarios": train_scenarios,
            "test_scenarios": test_scenarios,
            "validation_frac": validation_frac,
            "seed": seed,
        },
    )


def spatial_block_holdout(
    zcta_ids: list[str],
    adjacency: dict[str, set[str]],
    n_blocks: int = 5,
    test_blocks: int = 1,
    validation_blocks: int = 1,
    seed: int = 42,
) -> EvolutionSplit:
    """Split ZCTAs into spatial blocks using connected components.

    Assigns ZCTAs to blocks via BFS from random seeds, then holds out
    entire blocks for validation and test. This prevents spatial leakage
    from neighboring ZCTAs.
    """
    rng = np.random.default_rng(seed)
    unassigned = set(zcta_ids)
    blocks: list[list[str]] = [[] for _ in range(n_blocks)]

    # Seed each block with a random ZCTA
    seeds = rng.choice(list(unassigned), size=n_blocks, replace=False)
    for i, s in enumerate(seeds):
        blocks[i].append(s)
        unassigned.discard(s)

    # BFS expansion
    for _ in range(len(zcta_ids)):
        if not unassigned:
            break
        for i in range(n_blocks):
            if not unassigned:
                break
            frontier = set()
            for z in blocks[i]:
                frontier.update(adjacency.get(z, set()) & unassigned)
            if frontier:
                pick = rng.choice(list(frontier))
                blocks[i].append(pick)
                unassigned.discard(pick)

    # Assign remaining (disconnected) ZCTAs
    for z in list(unassigned):
        smallest = min(range(n_blocks), key=lambda i: len(blocks[i]))
        blocks[smallest].append(z)

    # Assign blocks to splits
    block_order = list(range(n_blocks))
    rng.shuffle(block_order)

    test_idx = block_order[:test_blocks]
    val_idx = block_order[test_blocks:test_blocks + validation_blocks]
    train_idx = block_order[test_blocks + validation_blocks:]

    test_ids = [z for i in test_idx for z in blocks[i]]
    validation_ids = [z for i in val_idx for z in blocks[i]]
    train_ids = [z for i in train_idx for z in blocks[i]]

    return EvolutionSplit(
        train_ids=train_ids,
        validation_ids=validation_ids,
        test_ids=test_ids,
        split_method="spatial_block_holdout",
        metadata={
            "n_blocks": n_blocks,
            "test_blocks": test_blocks,
            "validation_blocks": validation_blocks,
            "seed": seed,
            "block_sizes": [len(b) for b in blocks],
        },
    )
