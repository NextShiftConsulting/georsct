# Hypothesis Verdicts -- S035 Model Ladder

Last updated: 2026-06-25
Source: EXPERIMENT_STATUS.yaml (verdicts phase, r3_money_table phase), per_target_h2_breakdown.json, alpha_uplift_threshold.json

## Core Hypotheses (DOE_LOCKED.md)

| ID | Hypothesis | Expected | Actual | Verdict | Evidence |
|----|-----------|----------|--------|---------|----------|
| H1 | R0 baseline establishes measurable skill (RMSE < naive on >= 2/3 targets) | RMSE/baseline < 1.0 on >= 2/3 targets | 7/8 evaluable cells show skill above naive baseline | SUPPORTED | results/s035/r0_{scenario}.json, results/s035/certificates_r0.json |
| H2a | R1 improves over R0 (fold-level Wilcoxon, p<0.05, d>0.2) | Significant paired improvement | Pooled: p=0.114, d=0.27 (INCONCLUSIVE). Classification subset: p=0.004, d=0.50 (PASS) | PROVISIONAL | results/s035/money_table.json, per_target_h2_breakdown.json |
| H2b | kappa_geom predicts which cells benefit from R1 (exploratory) | Spearman rho(kappa_geom, uplift) positive | n=8 cells too small for reliable correlation; alpha>=0.4 predicts positive uplift (6/6) | PROVISIONAL | alpha_uplift_threshold.json |
| H3 | R2 improves over R1 (temporal features reduce RMSE >3%) | uplift_r2_vs_r1 > 0.03 on primary target | Houston NFIP R2=0.78 vs R1=0.72 (+13%); SWFL NFIP R2=0.50 vs R1=0.21 (+155%) | SUPPORTED | results/s035/r2_{scenario}.json, results/s035/money_table.json |
| H4 | Audit flags predict representation uplift (rho > 0.3) | audit_uplift_correlation rho > 0.3 | Not separately testable at n=5 scenarios; subsumed by H2b exploratory analysis | INCONCLUSIVE | results/s035/money_table.json |

## Six-Geometry Verdicts (compute_verdicts.py, verdicts phase)

| ID | Geometry | Expected | Actual | Verdict | Evidence |
|----|----------|----------|--------|---------|----------|
| V-PRED | Prediction | R1/R2 ladder shows measurable uplift | p=0.024, d=0.44 | SUPPORTED | results/s035/verdicts.json |
| V-REL | Relational | W-matrix contributes to RMSE reduction | 0/4 scenarios show W-matrix effect, d < 0.20 | NOT_LOAD_BEARING | results/s035/verdicts.json, results/s035/r1_no_wlag_{scenario}.json, results/s035/r1_wlag_only_{scenario}.json |
| V-RANK | Ranking | FAST Spearman correlations validate ranking order | Houston Spearman rho: -0.189 to -0.329 | PROVISIONAL | results/s035/fast_validation.json |
| V-CLUST | Clustering | Moran's I / LISA fractions show spatial clustering | Moran's I and LISA computed but insufficient for verdict | INSUFFICIENT | results/s035/sidecar/lisa_results.json |
| V-TRANS | Transfer | Cross-event transfer shows consistent patterns | Event distance matrix computed; transfer signal present | PROVISIONAL | results/s035/sidecar/event_distance_matrix.json |
| V-ALLOC | Allocation | All events covered by representation ladder | 12/15 events covered | PARTIAL | results/s035/verdicts.json |

## R3 Certificate-Gated Admission Hypotheses (DOE_R3)

| ID | Hypothesis | Expected | Actual | Verdict | Evidence |
|----|-----------|----------|--------|---------|----------|
| H5 | Certificate-gated admission (R3) improves or stabilizes over blind enrichment (R2) | R3 R^2 >= R2 or lower sigma | p=0.594, d=-0.249 (no degradation) | NO_DEGRADATION | results/s035/r3_money_table.json, results/s035/r3_hypothesis_evidence.json |
| H6 | Admitted blocks have lower leakage than rejected blocks | Significant Mann-Whitney U | Insufficient data for test | INSUFFICIENT_DATA | results/s035/r3_hypothesis_evidence.json |
| H7 (R3) | Block admission is order-robust (>=80% concordance) | >= 80% concordance across orderings | 4/11 blocks concordant (36%) | FAIL | results/s035/r3_hypothesis_evidence.json, results/s035/r3_order_robustness_{scenario}.json |
| H8 | Gear discriminates admission quality (headline > stabilizer delta) | Headline tier has higher delta_spatial | Insufficient data for test | INSUFFICIENT_DATA | results/s035/r3_hypothesis_evidence.json |

## R4 VLM Hypotheses (DOE_R4)

| ID | Hypothesis | Expected | Actual | Verdict | Evidence |
|----|-----------|----------|--------|---------|----------|
| H7 (R4) | VLMs extract measurable flood risk signal (Spearman rho > 0.3) | rho > 0.3 for at least one VLM | VLM money table completed; 5 providers x 5 scenarios assessed | PENDING (evidence on S3, not in local results/) | results/s035/r4_vlm_comparison on S3 |
| H8 (R4) | VLM choice is not a significant factor (pairwise rho > 0.7) | High inter-rater agreement | VLM comparison completed | PENDING (evidence on S3, not in local results/) | results/s035/r4_vlm_comparison on S3 |
| H9 | R4 VLM scores correlate with R0-R2 kappa diagnostics (exploratory) | VLMs detect same spatial structure as diagnostics | VLM comparison completed | PENDING (evidence on S3, not in local results/) | results/s035/r4_vlm_comparison on S3 |

## Floodcaster Story Hypotheses (DOE-C1, DOE-P1, DOE-C2a/b)

| ID | Hypothesis | Expected | Actual | Verdict | Evidence |
|----|-----------|----------|--------|---------|----------|
| DOE-C1 | Five-construct divergence shows heterogeneous spatial certification | Constructs diverge by geography | Houston anti-correlated (rho=-0.25, PASS); SW Florida positively correlated (rho=+0.50, DEMOTION); NYC insufficient | MIXED | results/s035/doe_c1/five_construct_{scenario}.json |
| DOE-P1 | Topology (HAND/GFI) is necessary for prediction | Removing topology degrades RMSE | 2/5 LOAD_BEARING, 2/5 NOT_LOAD_BEARING, 1/5 INCONCLUSIVE; topology hurts 3/5 due to partial HAND coverage | MIXED | results/s035/story/topology_necessity_{scenario}.json |
| DOE-C2a | Omega bootstrap quantifies per-construct quality | Omega range meaningful, alpha_omega discriminative | Omega range 0.869-1.000; NFIP < FEMA in NOLA (0.909 vs 0.975) | COMPLETED | results/s035/doe_c2a/omega_bootstrap_{scenario}.json |
| DOE-C2b | Temporal prior reduces certificate variance | Sequential < independent variance | Houston vr: 1.6-5.4%; NYC vr: 7.8-9.0%; AC-C2b-1/2 PASS | SUPPORTED | results/s035/doe_c2b/temporal_prior_{scenario}.json |
| DOE-H1 | Multi-RP Deltares depth improves hazard shape | Depth vectors beat scalar depth | -- | PENDING (DESIGNED, not executed) | -- |
| DOE-O1 | SAR anomaly nowcast adds lift | Nowcast arm improves over pre-event | -- | PENDING (DESIGNED, not executed) | -- |
| DOE-R1 | Kappa reconstruction validates taxi-route claim | Reconstructed kappa matches geometry kappa | -- | PENDING (REGISTERED, blocked on kappa=0.0 pin fix) | -- |
| DOE-C3 | Tract resolution activates Gate 3B | Census tract rebuild discriminates gate | -- | PENDING (DESIGNED, not executed) | -- |
| DOE-C4 | Morph dispatch loop improves gate-fail cells | ADR-012 regate loop works | -- | PENDING (DESIGNED, blocked on DOE-C3) | -- |

## Notes

- H7 numbering collision: DOE_R3 and DOE_R4 both define H7. Disambiguated as "H7 (R3)" and "H7 (R4)" above.
- H8 numbering collision: same issue, disambiguated as "H8 (R3 gear)" and "H8 (R4 VLM)".
- R4 VLM verdicts are marked PENDING because evidence is on S3 but not mirrored to local results/. The r4_vlm_comparison phase is COMPLETED per EXPERIMENT_STATUS.yaml.
- Variance stack findings (DOE deployment alignment): ladder NOT flat. Houston calibration gap reduced 41% (0.174 to 0.103). NYC FAIL_COVERAGE (19.8% uncovered). These inform V-PRED but are not separate hypotheses.
