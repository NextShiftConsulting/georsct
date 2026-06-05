# FloodRSCT + Floodcaster: SIGSPATIAL 2026 Dual-Track Schedule

**Document date:** Wednesday, May 27, 2026
**Conference:** ACM SIGSPATIAL 2026, Riverside CA, November 3–6, 2026

---

## Hard deadlines (Pacific Time, 11:59 PM)

| Date | Days from today | Deliverable |
|---|---|---|
| **Fri Jun 5, 2026** | 9 | Applications Track **abstract** submission |
| **Fri Jun 12, 2026** | 16 | Applications Track **full paper** (10 pp + refs) |
| **Sun Jun 28, 2026** | 32 | **Demo Track** paper (4 pp incl. refs) |
| Fri Jul 31, 2026 | 65 | Applications Track accept/reject notification |
| Thu Aug 21, 2026 | 86 | Camera-ready (both tracks) |
| Nov 3–6, 2026 | ~160 | Conference, Riverside CA |

**Submission portals:**
- Applications: `easychair.org/conferences/?conf=sigspatial2026app`
- Demo: `easychair.org/conferences/?conf=acmsigspatial2026` (or demo-specific link from CFP)

**Title format requirements:**
- Applications paper: `... [Applications]` suffix
- Demo paper: `... [Demo]` suffix
- Both suffixes are removed at camera-ready.

**Other gotchas:**
- Single-blind review — author block visible, no need to anonymize.
- Two-column `acmart` `sigconf` template — already configured in both `.tex` drafts.
- ACM Open Access APC ($250 if author is SIG member, $350 otherwise) — required to budget.
- Each accepted paper needs a separately paid registration.

---

## The honest reality check before any schedule

A 10-page Applications paper in 16 days is aggressive but achievable **if and only if**:
- The empirical results in §6 can be produced from a working pipeline by Jun 10 (gives 2 days of writing-against-results).
- At least 2 of the 5 scenarios have real data and produce real certificate outputs by Jun 8.
- The remaining 3 scenarios can ship as illustrative analyses with one figure each rather than full RQ1–RQ6 coverage.

If those preconditions are not on track by **Jun 1 (Mon)**, the right move is to shift the application paper from full 10pp to the **Short Papers track (4pp)** as a position-paper-with-prototype, and put the full 10pp version on the SIGSPATIAL 2027 calendar. The Short Papers track has the same Jun 12 deadline and accepts works that "report on early-stage research." This is the prudent fallback, not a failure mode.

---

## Track 1 — Writing (paper deliverables)

### Week 1 — Now through Jun 5 (abstract deadline)

| Day | Deliverable | Owner | Status |
|---|---|---|---|
| Wed May 27 | Front matter `.tex` drafted, four-layer figure compiles | — | **DONE** |
| Thu May 28 | §2 Related Work drafted (1.0 pp), citation discipline applied | writing | open |
| Fri May 29 | §3 Problem Formulation drafted (0.75 pp) | writing | open |
| Sat–Sun May 30–31 | §4 Method body drafted (1.5 pp) — including §5.3a RAG architecture and §5.4 rationale layer | writing | open |
| Mon Jun 1 | **Go/no-go checkpoint:** is the system producing real certificate outputs on ≥1 scenario? If no → consider Short Papers fallback. | both tracks | **decision point** |
| Tue Jun 2 | §5 Scenarios drafted (0.85 pp), Appendix A+B drafted (2.0 pp) | writing | open |
| Wed Jun 3 | §7 Experimental Design drafted (1.25 pp) with all 6 RQs specified | writing | open |
| Thu Jun 4 | **Abstract finalized** (250 words, matches the .tex abstract block) | writing | open |
| **Fri Jun 5** | **Submit abstract by 11:59 PM PT** | writing | **HARD DEADLINE** |

### Week 2 — Jun 6 through Jun 12 (paper deadline)

| Day | Deliverable | Owner | Status |
|---|---|---|---|
| Sat–Sun Jun 6–7 | §6 Results body skeleton with figure stubs; ingest experiment outputs as they land | writing + build | open |
| Mon Jun 8 | **Last day for new experimental data** to land in the paper. Lock results scope. | both tracks | **deadline-2** |
| Tue Jun 9 | §6 Results drafted in full (1.5 pp) against locked data; Figure 1 (four-layer) finalized; Figure 2 (action drift) drafted; Figure 3 (residual map for 1 scenario) drafted | writing | open |
| Wed Jun 10 | §8 Floodcaster Demo section drafted (0.5 pp; the full demo lives in the Demo paper); §9 Governance drafted (1.0 pp) | writing | open |
| Thu Jun 11 | §10 Limitations + §11 Conclusion drafted; Appendix C (N-ceiling for flood tasks) drafted (0.75 pp); full read-through; reference checking | writing | open |
| **Fri Jun 12** | **Submit full paper by 11:59 PM PT** | writing | **HARD DEADLINE** |

### Weeks 3–4 — Jun 13 through Jun 28 (demo paper)

| Day | Deliverable | Owner | Status |
|---|---|---|---|
| Sat–Mon Jun 13–15 | Decompress. Pull the demo paper skeleton into a working draft. Identify which figures need to be remade for the demo paper (the mobile app screen flow). | writing | open |
| Tue–Thu Jun 16–18 | Demo paper §2 System Overview, §3 Mobile Operator Surfaces, §4 Audit Trace drafted in full | writing | open |
| Fri–Sat Jun 19–20 | Demo paper §5 Demonstration Plan drafted with the four-act script tested against the mobile artifact | writing + build | open |
| Sun–Mon Jun 21–22 | Mobile-app screen mockups or screenshots finalized for figures | build | open |
| Tue–Thu Jun 23–25 | Demo paper full read-through; cross-reference check against the Applications paper (citation key `martin2026floodrsct` must resolve) | writing | open |
| Fri–Sat Jun 26–27 | Demo paper polish, final review | writing | open |
| **Sun Jun 28** | **Submit demo paper by 11:59 PM PT** | writing | **HARD DEADLINE** |

---

## Track 2 — Build (system deliverables)

The build track does not have its own external deadline — it serves the writing track. But the build milestones below are the **enablers** for the writing track to meet its deadlines.

### Week 1 — Now through Jun 5

| Date | Build milestone | Enables which paper section |
|---|---|---|
| Thu May 28 | Postgres schema with R/S/N columns + snapshot-key composite index in place | §4 Method (RAG layer specification) |
| Fri May 29 | One scenario's evidence ingested (Houston is the natural first scenario — most data sources public) | §5 Scenarios (Houston subsection), §6 Results |
| Sat May 30 | Scenario router registry for Houston: allowed/required/forbidden per gate question | §4 Method, Appendix E |
| Sun May 31 | Certificate computation pipeline runs end-to-end on Houston: ingest → α/κ/σ/TRF → gate → action | §6 Results (RQ1, RQ2 for Houston) |
| **Mon Jun 1** | **Go/no-go checkpoint with Track 1.** If this works on Houston: proceed. If not: invoke Short Papers fallback. | (decision point) |
| Tue Jun 2 | Second scenario online — recommend Riverside–Coachella (sharpest contrast with Houston, local SIGSPATIAL relevance) | §6 Results (RQ2 cross-scenario contrast) |
| Wed Jun 3 | Stress-test recipes (sensor dropout, rainfall shift, infrastructure failure) implemented over single ablation engine | §6 Results (RQ3) |
| Thu Jun 4 | Geometry flags wired for both scenarios | §6 Results (RQ4) |
| Fri Jun 5 | Rationale generation hooked to certificate outputs (no human eval yet) | §6 Results (RQ5 setup) |

### Week 2 — Jun 6 through Jun 12

| Date | Build milestone | Enables which paper section |
|---|---|---|
| Sat–Sun Jun 6–7 | Scenarios 3–4 (New Orleans, SW Florida) ingested with at least basic certificate runs | §6 Results coverage breadth |
| Mon Jun 8 | **Data lock for paper.** Final experiment runs complete. | §6 Results final tables |
| Tue Jun 9 | Tier-1 compliance auditing implemented (RQ6 first column) | §6 Results §8.6 |
| Wed Jun 10 | Snapshot reproducibility check (RQ6 second column) | §6 Results §8.6 |
| Thu Jun 11 | Optional: scenario 5 (NYC/NJ) ingested as illustrative; if not ready, marked as "future work" with one scenario stub | §5 Scenarios |
| **Fri Jun 12** | (Paper submission — build supports if any last-minute number-check is needed) | — |

### Weeks 3–4 — Jun 13 through Jun 28 (mobile app focus)

| Date | Build milestone | Enables which paper section |
|---|---|---|
| Mon Jun 15 | Mobile app v0 wireframes: action queue, drill hierarchy, evidence drawer, provenance panel, compliance badge, audit footer | Demo §3 |
| Wed Jun 17 | Action queue + drill hierarchy working on Houston snapshot | Demo §3 |
| Fri Jun 19 | Evidence drawer + compliance badge wired to backend | Demo §3 |
| Sun Jun 21 | Audit trace export + replay working end-to-end (one-tap export, laptop-side replay command produces bit-identical action) | Demo §4 |
| Wed Jun 24 | All 5 scenarios swappable from home screen | Demo §2 |
| Fri Jun 26 | Stress-test recipe invocable from drill view | Demo §5 (Act 3) |
| Sat Jun 27 | Final demo dress rehearsal — run the 4-act script end-to-end | Demo §5 |

---

## Critical integration points

Three places where Track 1 and Track 2 must synchronize:

1. **Mon Jun 1 (go/no-go).** If the certificate pipeline does not produce real outputs on Houston by EOD, switch the paper plan to Short Papers (4pp). This decision is irreversible after Jun 2 because the writing track diverges.

2. **Mon Jun 8 (data lock).** All experimental numbers used in §6 must be finalized by EOD. Anything not landed by this point becomes future work, not a paper claim.

3. **Sat Jun 27 (demo dress rehearsal).** The 4-act demo script must work end-to-end. Any act that doesn't work cleanly gets rewritten in the demo paper before Jun 28 submission, or dropped from the script.

---

## Fallback paths (rank-ordered by preference)

If the schedule slips, take fallbacks in this order:

1. **Reduce scenario coverage from 5 to 3.** Keep Houston + Riverside–Coachella + one of {New Orleans, SW Florida}. Mark the omitted scenarios as future work. Cost: 0.5 pages in §5, reduced cross-scenario evidence in §6.

2. **Compress §6 Results from 6 RQs to 4.** Drop RQ4 (geometry flag evaluation) and RQ6 (RAG grounding evaluation) into the appendix as "preliminary results." Keep RQ1 (accuracy vs readiness), RQ2 (scenario-specific failure profiles), RQ3 (stress-test action drift), RQ5 (rationale acceptability). Cost: weaker contribution claims, but defensible.

3. **Switch from Applications Track (10pp) to Short Papers Track (4pp).** Position the paper as a position-paper-with-prototype. Keep front matter, compressed method (0.75 pp), one scenario (Houston) with a representative figure, rationale evaluation as a contribution sketch, governance and conclusion. Cost: smaller venue footprint, but credible standalone artifact. Demo paper proceeds unchanged on the Jun 28 timeline.

4. **Submit only the Demo paper for SIGSPATIAL 2026, save the Applications paper for 2027.** Demo paper is 4pp and depends only on the mobile artifact, which has a longer development runway. Cost: one-year delay on the methods publication, but the demo paper becomes the planting flag for the framework.

The fallbacks are listed in order of decreasing scope, not in order of likelihood. Take the smallest fallback that resolves the constraint.

---

## What I produce next, if asked

In priority order, the next writing deliverables I can produce on request:

1. **§2 Related Work** drafted in the same style as the front matter (1.0 pp, due Thu May 28).
2. **§3 Problem Formulation** drafted (0.75 pp, due Fri May 29).
3. **§4 Method** drafted in full (1.5 pp, due weekend May 30–31).
4. **Appendix A (Certificate Architecture Reference)** drafted with the reproduce-vs-cite discipline applied (1.0 pp, useful any time).
5. **Demo paper §3 Mobile Operator Surfaces** expanded into a full draft once wireframes exist.

Each of these is a single writing pass that fits in one session. The build-side milestones (Track 2) are on you; the writing-side milestones I can move quickly on whenever the build context unblocks them.
