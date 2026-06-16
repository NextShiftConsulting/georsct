"""GeoRSCT-X scoring — process, outcome, and GeoCert composite.

Three scoring layers:
  1. Process: TerraBench-faithful tool/process metrics.
  2. Outcome: Tolerance-aware numeric accuracy.
  3. GeoCert: Certificate-gated outcome (the differentiator).

Import rule: depends ONLY on contracts and provenance.
Never imports domain, execution, or experts.
The scorer is a leaf -- it receives Trace and TaskGold, nothing else.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from georsct.contracts.task_gold import TaskGold
from georsct.provenance.trace import Trace, Verdict


# ---------------------------------------------------------------------------
# Process scoring (TerraBench-faithful)
# ---------------------------------------------------------------------------

def _lcs_len(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if a[i] == b[j]:
                dp[i][j] = dp[i + 1][j + 1] + 1
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
    return dp[0][0]


def _multiset_overlap(a: list[str], b: list[str]) -> float:
    """Multiset intersection / max(len(a), len(b))."""
    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    denom = max(len(a), len(b), 1)
    return inter / denom


@dataclass(frozen=True)
class ProcessScore:
    """TerraBench-style process metrics."""

    inst_acc: float
    tool_call_success_rate: float
    tool_acc: float
    category_f1: float
    arg_acc: float
    order_score: float
    tool_use_score: float  # weighted composite


def score_process(pred: Trace, gold: Trace) -> ProcessScore:
    """Score the predicted trace against a gold (canonical) trace.

    Metrics per TerraBench:
      - InstAcc: fraction of valid steps
      - ToolCallSuccessRate: fraction of successful steps
      - ToolAcc: relaxed group-level match
      - CategoryF1: multiset F1 over tool-group labels
      - ArgAcc: key+value match over name-aligned pairs
      - OrderScore: (Unique + AnyOrder + SameOrder) / 3
      - ToolUseScore: weighted composite
    """
    P = pred.steps
    G = gold.steps
    p_len = len(P) or 1

    inst_acc = sum(s.valid for s in P) / p_len
    succ_rate = sum(s.success for s in P) / p_len

    g_groups = [s.tool_group for s in G]
    p_groups = [s.tool_group for s in P]

    # ToolAcc: relaxed group-level match
    relaxed = sum((Counter(p_groups) & Counter(g_groups)).values())
    tool_acc = relaxed / (len(G) or 1)

    # CategoryF1: multiset F1
    inter = sum((Counter(p_groups) & Counter(g_groups)).values())
    prec = inter / (len(p_groups) or 1) if p_groups else 0.0
    rec = inter / (len(g_groups) or 1) if g_groups else 0.0
    cat_f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    # ArgAcc: key+value match over name-aligned pairs
    arg_scores: list[float] = []
    used: set[int] = set()
    for sp in P:
        for j, sg in enumerate(G):
            if j in used or sg.tool_name != sp.tool_name:
                continue
            keys = set(sp.args) & set(sg.args)
            denom = set(sp.args) | set(sg.args)
            kscore = len(keys) / (len(denom) or 1)
            vscore = sum(
                1 for k in keys if sp.args[k] == sg.args[k]
            ) / (len(keys) or 1)
            arg_scores.append((kscore + vscore) / 2)
            used.add(j)
            break
    arg_acc = sum(arg_scores) / (len(arg_scores) or 1)

    # OrderScore: (Unique + AnyOrder + SameOrder) / 3
    unique = (
        len(set(p_groups) & set(g_groups)) / (len(set(g_groups)) or 1)
    )
    any_order = _multiset_overlap(p_groups, g_groups)
    same_order = _lcs_len(p_groups, g_groups) / (len(g_groups) or 1)
    order_score = (unique + any_order + same_order) / 3

    # Weighted composite
    tool_use = (
        0.30 * tool_acc
        + 0.15 * inst_acc
        + 0.20 * arg_acc
        + 0.15 * cat_f1
        + 0.15 * order_score
        + 0.05 * succ_rate
    )

    return ProcessScore(
        inst_acc=inst_acc,
        tool_call_success_rate=succ_rate,
        tool_acc=tool_acc,
        category_f1=cat_f1,
        arg_acc=arg_acc,
        order_score=order_score,
        tool_use_score=tool_use,
    )


# ---------------------------------------------------------------------------
# Outcome scoring (tolerance-aware numeric accuracy)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OutcomeScore:
    """Tolerance-aware numeric accuracy."""

    hit_at_tol: float  # fraction of fields within tolerance
    num_score: float  # soft score with exponential decay


def score_outcome(pred: Trace, gold: TaskGold) -> OutcomeScore:
    """Score predicted numeric outputs against ground truth.

    For each output field:
      - lambda = max(abs_tol, rel_tol * |y|, floor_scale)
      - hit = 1 if |y_hat - y| <= lambda, else 0
      - num = 1 if |y_hat - y| <= lambda, else 2^(-(n-1)) where n = |y_hat - y| / lambda
    """
    gold_dict = gold.as_dict()
    hits: list[float] = []
    nums: list[float] = []

    for key, gf in gold_dict.items():
        yhat = pred.final_json.get(key)
        if yhat is None:
            hits.append(0.0)
            nums.append(0.0)
            continue

        lam = max(gf.abs_tol, gf.rel_tol * abs(gf.value), gf.floor_scale)
        lam = lam or 1e-9
        ae = abs(yhat - gf.value)
        n = ae / lam

        hits.append(1.0 if ae <= lam else 0.0)
        nums.append(1.0 if n <= 1 else 2.0 ** (-(n - 1)))

    n_fields = len(hits) or 1
    return OutcomeScore(
        hit_at_tol=sum(hits) / n_fields,
        num_score=sum(nums) / n_fields,
    )


# ---------------------------------------------------------------------------
# GeoCert scoring (the GeoRSCT-X differentiator)
# ---------------------------------------------------------------------------

# Verdict-to-factor mapping for certificate gating.
_VERDICT_FACTOR: dict[Verdict, float] = {
    Verdict.PASS: 1.0,
    Verdict.WARN: 0.8,
    Verdict.ESCALATE: 0.5,
    Verdict.FAIL: 0.0,
    Verdict.SUPPRESS: 0.0,
}


@dataclass(frozen=True)
class GeoCertScore:
    """Certificate-gated outcome score.

    The GeoRSCT differentiator: gates the numeric score by the
    certificate verdict and flags "lucky hits" — cases where the
    numeric answer is correct but the evidence pathway is not certified.
    """

    geocert_score: float  # num_score * verdict_factor
    lucky_hit: bool  # numeric pass under non-admissible certificate
    verdict: str


def score_geocert(
    pred: Trace,
    outcome: OutcomeScore,
) -> GeoCertScore:
    """Compute certificate-gated outcome score.

    A correct answer under an invalid certificate is a "lucky hit" —
    not trustworthy because the evidence pathway was not admissible.
    """
    v = pred.certificate.verdict if pred.certificate else Verdict.FAIL
    factor = _VERDICT_FACTOR[v]
    lucky = outcome.hit_at_tol > 0 and factor == 0.0

    return GeoCertScore(
        geocert_score=outcome.num_score * factor,
        lucky_hit=lucky,
        verdict=v.value,
    )
