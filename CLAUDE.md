# georsct Dev Instructions

## Floodcaster Status: FROZEN at v1.7

Status: Floodcaster side is complete through v1.7. Do not add new Floodcaster runtime architecture, storage, adapters, AI behavior, figures, or claims unless explicitly requested.

We are **not ready to submit the full paper yet**, but the blocker is no longer in the Floodcaster repo.

## Floodcaster completion state

Floodcaster has completed its assigned contribution:

* v1.0 governed demo architecture
* v1.0.1 release hardening and runbook
* v1.1 demo-to-paper bridge
* v1.2 paper figure/table pack
* v1.3 paper section draft and condensed V7 text
* v1.4 paper claim audit
* v1.5 citations and bibliography pass
* v1.6 V7 paper assembly contribution
* v1.7 MMAR / adversarial review pass

Current Floodcaster evidence:

* Governed demo pipeline is tested.
* 457 tests passing.
* Floodcaster sections are drafted, QC'd, claim-audited, cited, and reviewer-reviewed.
* 69 claims classified.
* 0 unresolved red-flag claims.
* Language discipline is clean: no unsupported "proves," "ensures," or "guarantees."
* Demo claims are bounded to mechanism, traceability, replay, governed querying, and public-message constraints.
* No statistical claims are made from the demo.

## Important boundary

Floodcaster's job is now done for this paper cycle.

Do **not** continue adding Floodcaster features to solve paper submission gaps. The remaining blockers depend on other repos and experiment artifacts.

## Submission blockers outside Floodcaster

The full paper is still blocked by 11 placeholder sections that require content from `rsct-governance`, `georsct`, and the `s035` experiment pipeline.

| Gap                                                                 | Source repo / artifact   | Paper sections |
| ------------------------------------------------------------------- | ------------------------ | -------------- |
| Related Work                                                        | V5/V6 papers             | 2.1–2.4        |
| R/S/N decomposition, gates, adversarial setup, topology definitions | rsct-governance, georsct | 4.2–4.5        |
| Divergence matrix and interpretation                                | georsct                  | 5.2–5.3        |
| Per-construct certificate, divergence, topology recovery            | s035 pipeline            | 6.2–6.4        |
| Experiment results                                                  | s035 runs                | 7.1–7.4        |
| Discussion: divergence interpretation and construct leakage         | s035 results             | 8.1–8.2        |
| Additional figures 5–8 and tables 5–8                               | s035 experiments         | various        |

## Next work should happen outside Floodcaster

Proceed to the other repos in this order:

1. `s035` pipeline
   Extract final experiment results, per-construct certificates, divergence outputs, topology recovery outputs, and paper-ready tables/figures.

2. `georsct`
   Supply divergence matrix interpretation, topology framing, and GeoRSCT construct-level discussion.

3. `rsct-governance`
   Supply R/S/N decomposition, gates, adversarial framing, and governance-theory sections.

4. V5/V6 paper materials
   Supply Related Work and prior-paper continuity.

## Floodcaster repo allowed work only if needed

Only allow the following Floodcaster changes:

* Typo fixes.
* Broken link fixes.
* Reference-path corrections.
* Redaction of sensitive hashes, paths, endpoints, or credentials if needed.
* Minor wording edits to align with final paper phrasing.
* No new architecture.
* No new runtime behavior.
* No new empirical claims.
* No new demo features.

## Final instruction

Treat Floodcaster as frozen at the v1.7 boundary for this paper cycle. The next milestone is not a Floodcaster milestone. The next milestone is cross-repo paper completion using `s035`, `georsct`, `rsct-governance`, and V5/V6 materials.

The important line for the team is:

```text
Ready on the Floodcaster side: yes.
Ready to submit the full paper: no.
Remaining blockers are cross-repo paper assembly, not Floodcaster engineering.
```
