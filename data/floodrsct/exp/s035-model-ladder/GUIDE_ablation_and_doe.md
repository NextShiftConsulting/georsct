# Ablation & DOE Guide for s035-model-ladder

A plain-language guide to the experimental design concepts used in this
experiment. Written for collaborators and reviewers who want to understand
**why** the experiment is structured this way, not just **what** it does.

---

## Part 1: What Is a DOE?

**DOE = Design of Experiments.** It is the plan you write *before* running
anything. It answers three questions:

1. **What am I changing?** (independent variables)
2. **What am I measuring?** (dependent variables)
3. **What am I holding constant?** (controlled variables)

The purpose is to make results **interpretable**. If you change five things
at once and the score goes up, you don't know which change helped. A DOE
forces you to change one thing at a time so that every result has a clear
cause.

### DOE in s035

We have a **representation ladder**: R0 -> R1 -> R2. Each level adds
features to the model while keeping everything else identical.

```
R0:  33 features  (demographics, flood zones, terrain)
R1:  61 features  (R0 + hydrology + spatial lag)
R2:  70 features  (R1 + rainfall timing + storm dynamics)
```

The DOE says:

| What changes | What stays the same |
|-------------|---------------------|
| Feature set (R0/R1/R2) | Solver hyperparameters (max_depth=6, lr=0.1) |
| | Fold assignments (same 5 folds for all levels) |
| | Target variable (same obs_nfip_event_claims) |
| | Random seed (42) |

Because **only the features change**, any improvement from R0 to R1 is
caused by the new features -- not by different folds, different tuning,
or different data splits.

### Why "lock" a DOE?

Once the DOE is written, we **lock** it. No changes after results come in.
This prevents a subtle cheat: if you see the results first, you can
unconsciously adjust the design to make them look better. Locking the
design before results is called **pre-registration**.

Our DOE lives in `DOE_LOCKED.md` (original) and `DOE_AMENDMENT_v1.2.md`
(justified changes, each with a version number and date).

---

## Part 2: What Is an Ablation?

**Ablation = removing one component to see what happens.** The name comes
from medicine (ablating tissue to study its function). In ML experiments,
you remove a feature, a module, or a processing step and measure whether
performance drops.

### The logic

```
Full model:    score = 0.65
Remove X:      score = 0.40
                       ----
Conclusion:    X contributed ~0.25 to the score
```

If removing X doesn't change the score, X was dead weight. If it drops
the score to zero, X was doing all the work.

### Ablation vs. addition

They answer different questions:

| Approach | Question | Risk |
|----------|----------|------|
| **Addition** (R0 -> R1) | "Does adding hydrology help?" | Maybe it helps for the wrong reason (e.g., spatial leakage) |
| **Ablation** (R1 -> R1 minus W-matrix) | "Is the improvement from hydrology or from spatial lag?" | Isolates the mechanism |

**Both are needed.** Addition tells you the total effect. Ablation tells
you which part of the addition matters.

### Ablations in s035

We have three types:

#### Type 1: Feature-group ablation (which features matter?)

At R1, we added 8 W-matrix spatial features AND 16 hydrology features.
To tell them apart:

| Variant | Features | What it tests |
|---------|----------|---------------|
| R1 full | R0 + hydrology + W-matrix (61 features) | Everything |
| R1 no-wlag | R0 + hydrology only (53 features) | Is uplift from point features or spatial structure? |
| R1 no-target-lag | R0 + hydrology + W-matrix minus wlag_nfip_claims (60 features) | Is the target spatial lag doing all the work? |
| R1 wlag-only | R0 + W-matrix only (41 features) | Do we even need hydrology? |

If R1-full beats R0 by 15%, but R1-no-wlag only beats R0 by 2%, the
honest conclusion is: "spatial lag features drive the improvement, not
hydrology."

#### Type 2: Random-features ablation (is there signal at all?)

Replace ALL real features with random noise, N(0,1). Same folds, same
targets, same solver. If the random model scores nearly as well as the
real model, your features carry no signal -- the score is an artifact of
the evaluation setup.

```
Real features R0:    R2 = 0.45
Random features R0:  R2 = 0.02   <-- good, real features have signal
Random features R0:  R2 = 0.40   <-- bad, something is leaking
```

This is the **null baseline**. It catches problems that no other test can
find, like target leakage through the fold structure.

#### Type 3: Prompt ablation for VLMs (what does the model read?)

For the R4 VLM arm, we test three prompt levels:

| Variant | Image | Text | Question answered |
|---------|-------|------|-------------------|
| P0 | Map only | None | Can the VLM read the map? |
| P1 | Map + legend | None | Does the legend help? |
| P2 | Map + legend + text | Full evidence | Does text evidence add signal? |

If P0 ~ P2: VLM reads the map. Text is redundant.
If P0 ~ 0 and P2 is high: VLM reads the text numbers and ignores the map.

---

## Part 3: Controls and Baselines

### What is a control?

A **control** is a known input where you already know what the answer
*should* be. If the model gives the wrong answer on a control, something
is broken.

### Controls in s035

**Null-input controls for VLMs:**

| Control | Input | Expected behavior |
|---------|-------|--------------------|
| Blank image | Solid white PNG, no map | Risk score near 0.5 (uncertain) or refusal |
| Noise image | Random pixel noise | Same -- no information to extract |
| Inverted colormap | Real map with colors flipped | Score should DIFFER from the real map |
| Mismatched | Map from Houston, text from NYC | VLM should flag inconsistency |

If a VLM gives the same score for a blank image and a real flood map, it
is returning its priors (what it "expects" flood risk to be), not reading
the image.

**Baselines in s035:**

| Baseline | Purpose |
|----------|---------|
| R0 (33 features) | The floor -- how well can you do with basic demographics? |
| Random-features R0 | The sub-floor -- what score does pure noise get? |
| Mean predictor | R2 = 0 by definition -- the trivial baseline |

Every result is interpreted relative to these baselines. "R2 improved by
15%" means nothing without knowing "compared to what?"

---

## Part 4: Folds, Splits, and Why They Matter

### What is cross-validation?

Instead of one train/test split, you split the data K ways and rotate
which part is the test set. This gives K measurements instead of one.

```
Fold 1:  [TEST] [train] [train] [train] [train]
Fold 2:  [train] [TEST] [train] [train] [train]
Fold 3:  [train] [train] [TEST] [train] [train]
Fold 4:  [train] [train] [train] [TEST] [train]
Fold 5:  [train] [train] [train] [train] [TEST]
```

### Why spatial blocking?

In geospatial data, nearby locations are similar (spatial autocorrelation).
If you split randomly, the training set contains neighbors of every test
point -- the model can "cheat" by memorizing nearby values.

**Spatial blocking** groups ZCTAs by county and assigns entire counties to
the same fold. No test ZCTA has a training neighbor in the same county.

```
Random split:     ZCTA 77001 (test) <-> ZCTA 77002 (train, 2 km away)
                  Model sees its neighbor's answer. Inflated score.

Spatial blocked:  All of Harris County in fold 2.
                  No neighbor leakage across folds. Honest score.
```

### Three splits, three purposes

| Split | Purpose | Used for |
|-------|---------|----------|
| Random 80/20 | Measure how much spatial leakage inflates scores | diag_leakage computation |
| Spatial-blocked 5-fold | Honest evaluation, no neighbor leakage | **All headline results** |
| Leave-event-out | Hold out entire hurricane, test generalization | diag_transfer computation |

The gap between random and spatial-blocked IS the leakage. If random
gives R2=0.60 and spatial-blocked gives R2=0.35, there is 0.25 worth of
spatial autocorrelation being exploited.

---

## Part 5: Hypothesis Testing in s035

### The problem with small samples

We have 9 (scenario x target) cells. That is a small number to draw
conclusions from. Our hypothesis tests are designed around this reality.

### Fold-level paired test (primary)

Instead of testing at the cell level (n=9), we test at the fold level.
Each cell has 5 folds. Each fold gives a paired observation:

```
Fold 1, Houston, NFIP:  R0 score = 0.32,  R1 score = 0.41  -> delta = +0.09
Fold 2, Houston, NFIP:  R0 score = 0.28,  R1 score = 0.35  -> delta = +0.07
Fold 3, Houston, NFIP:  R0 score = 0.35,  R1 score = 0.44  -> delta = +0.09
...
```

Pooling across all cells: 9 cells x 5 folds = ~45 paired observations.
Now we can run a **Wilcoxon signed-rank test** on the deltas: "Are these
deltas systematically positive, or could they be zero?"

### Why paired?

Each R0 fold is compared to the **same** R1 fold (same ZCTAs, same
target). This cancels out fold-level difficulty -- a hard fold stays hard
in both R0 and R1. The delta isolates the effect of the feature change.

```
Unpaired:  "R1 mean = 0.40, R0 mean = 0.35"
           Maybe R1 got easier folds?

Paired:    "R1 - R0 per fold = [+0.09, +0.07, +0.09, ...]"
           Same folds, so the difference IS the feature effect.
```

### Effect size: Cohen's d

p-values tell you "is this real?" but not "is this big?" Cohen's d
answers the second question:

```
d = mean(deltas) / std(deltas)
```

| d | Interpretation |
|---|---------------|
| 0.2 | Small effect |
| 0.5 | Medium effect |
| 0.8 | Large effect |

We require BOTH: Wilcoxon p < 0.05 AND Cohen's d > 0.2 to declare a
PASS. A statistically significant but tiny effect is not useful.

### Multiple comparisons

We run 8 exploratory tests (4 diagnostics x 2 transitions). If you run
enough tests, something will be "significant" by chance. We use
**Holm-Bonferroni correction**: sort p-values, multiply the smallest by
8, the next by 7, etc. This controls the family-wise error rate.

But only ONE test is the pre-registered primary: the fold-level Wilcoxon
on R0 vs R1 deltas. The cell-level associations (including kappa_geom as
predictor, per v1.8 amendment) are exploratory and labeled as such.

---

## Part 6: The Money Table

The money table is the single most important output. One row per
(scenario, target) cell, showing everything:

```
scenario | target | kappa | R0_R2 | R1_R2 | R2_R2 || r4_ref_r2(zs) || R0->R1_pct | R1->R2_pct | R_cert | S_cert | N_cert
---------|--------|-------|-------|-------|-------||---------------||------------|------------|--------|--------|-------
houston  | nfip   | 0.82  | 0.35  | 0.48  | 0.52  || 0.18          || +37%       | +8%        | 0.48   | 0.05   | 0.47
nyc      | nfip   | 0.71  | 0.22  | 0.31  | 0.34  || 0.15          || +41%       | +10%       | 0.31   | 0.09   | 0.60
...
```

The `r4_ref_r2` column reports a zero-shot, event-invariant VLM baseline
on the shared folds for visual comparison only; it is excluded from the
paired confirmatory tests (see Statistical-Considerations.md §3) and
carries no uplift percentage. The double bars mark it as a fenced-off
reference, not a ladder rung.

Everything the paper claims is visible in this table. No hidden analysis.
A reviewer can verify every number.

---

## Part 7: How It All Fits Together

```
                     BEFORE TRAINING
                     ===============
   Write DOE  -----> Lock DOE  -----> Compute kappa_geom
   (this plan)       (no changes)     (geometric compatibility,
                                       no model involved)

                     TRAINING LADDER
                     ===============
   R0 (baseline) --> Diagnostics --> R1 (+ spatial) --> Diagnostics --> R2 (+ temporal)
        |                                |                                    |
        v                                v                                    v
   Random-features   W-matrix           Coastal-feature
   ablation          ablation           ablation
   (null baseline)   (mechanism)        (mechanism)

                     AFTER TRAINING
                     ==============
   Compute uplift table  -->  Fold-level Wilcoxon  -->  Money table
   Compute certificates  -->  Certificate evolution table
   DGM routing analysis  -->  Routing hit rate

                     PARALLEL ARM
                     ============
   R4 VLM (null controls, prompt ablation, deterministic inference)
   --> r4_ref_r2 reference column in money table (zero-shot, no uplift%)
   --> Inferential results (H7-H9) stay in separate R4 table
```

### The causal chain the paper claims

1. **kappa_geom** (computed before training) predicts which cells are hard
2. **Diagnostics at R0** (computed after R0, before R1) predict which cells
   need spatial features
3. **R1 ablation** confirms whether spatial lag or hydrology drives uplift
4. **Diagnostics at R1** predict which cells need temporal features
5. **R2 results** confirm (or deny) the prediction
6. **Certificates** show R increasing and S_sup decreasing across levels
7. **DGM routing** shows the certificate system can automate level selection

Each step is checkable. Each prediction precedes its confirmation. That
is the point of the entire experimental design.

---

## Glossary

| Term | Plain meaning |
|------|---------------|
| **Ablation** | Remove one thing, measure the damage |
| **Baseline** | The score you get with the simplest approach |
| **Cell** | One (scenario, target) combination -- e.g., Houston x NFIP claims |
| **Control** | An input where you know the right answer |
| **Cohen's d** | How big is the effect? (0.2=small, 0.5=medium, 0.8=large) |
| **DOE** | The experiment plan, written before any results |
| **Fold** | One slice of the data used as the test set |
| **Holm-Bonferroni** | Correction for running many tests at once |
| **kappa_geom** | Geometric compatibility score, computed before training |
| **Money table** | The single table with all results, one row per cell |
| **Null baseline** | Random noise features -- the floor below the floor |
| **Paired test** | Compare R0 and R1 on the same fold, not different folds |
| **Pre-registration** | Declare your hypothesis before seeing results |
| **RSN simplex** | (R, S_sup, N) = signal, leakage, noise -- sums to 1 |
| **Spatial blocking** | Group nearby locations into the same fold |
| **Wilcoxon signed-rank** | "Are the paired deltas systematically positive?" |

---

## Reading Order

If you are new to this experiment, read in this order:

1. **This guide** (you are here)
2. [README.md](README.md) -- experiment architecture and file map
3. [Statistical-Considerations.md](Statistical-Considerations.md) -- tests, power, corrections
4. [DOE_R0_baseline.md](DOE_R0_baseline.md) -- the control arm
5. [DOE_R1_spatial.md](DOE_R1_spatial.md) -- first treatment (spatial features)
6. [DOE_R2_temporal.md](DOE_R2_temporal.md) -- second treatment (event dynamics)
7. [DOE_R4_vlm.md](DOE_R4_vlm.md) -- exploratory VLM arm
8. [DOE_AMENDMENT_v1.2.md](DOE_AMENDMENT_v1.2.md) -- all design changes with rationale
9. [DOE_LOCKED.md](DOE_LOCKED.md) -- original locked design (for audit trail)
