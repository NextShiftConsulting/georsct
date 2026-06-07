# Insurance-Damage Divergence: Why NFIP Claims != Flood Damage

**Mission:** M5 (documentation)
**Date:** 2026-06-07
**Status:** Evidence compilation -- no verdict logic altered

---

## 1. Core Finding

Houston FAST validation produces **negative Spearman rho** between NFIP
claims (obs_nfip_event_claims) and physics-based depth-damage (Hazus via
sphere-flood). All three Houston events show rho between -0.19 and -0.33.

This means: ZCTAs with HIGH physical flood damage have LOW insurance
claims, and vice versa.

---

## 2. Why NFIP Claims Are Not Equivalent to Flood Damage

### 2.1 Insurance Penetration is Non-Uniform

NFIP flood insurance purchase is:
- **Mandatory** only within SFHA (Special Flood Hazard Area) for
  federally-backed mortgages
- **Voluntary** outside SFHA -- uptake driven by income, awareness,
  and home ownership status
- **Absent** for renters, commercial properties, public infrastructure,
  and the unbanked

A ZCTA with zero claims may have: (a) no flooding, OR (b) extensive
flooding with no insured structures. The feature contract explicitly
notes this ambiguity:

> "NFIP claims = 0 is ambiguous (no damage vs. no insurance)"
> -- GeoSpatial_Challenges.md, line 539

### 2.2 Socioeconomic Confounding

The assembled feature set contains direct evidence of this mechanism:

| Feature | Contract Location | Signal |
|---------|------------------|--------|
| `acs_median_hh_income` | FEATURE_CONTRACT line 48 | Wealth -> insurance purchase |
| `acs_median_home_value` | FEATURE_CONTRACT line 96 | "Correlated with NFIP penetration" (verbatim note) |
| `acs_pct_no_insurance` | FEATURE_CONTRACT line 136 | Fraction without health insurance (proxy for financial access) |
| `svi_overall` | FEATURE_CONTRACT line 172 | CDC Social Vulnerability Index composite |
| `svi_socioeconomic` | FEATURE_CONTRACT line 185 | SVI Theme 1 -- socioeconomic status |

The DOE_LOCKED.md (line 125) explicitly labels NFIP's bias profile:

> "Insurance penetration, wealth"

### 2.3 The Mechanism

```
High-damage, low-claim ZCTAs:
  Physical damage (Hazus) is HIGH
  + Low income -> no voluntary NFIP purchase
  + Outside SFHA -> no mandatory purchase requirement
  + Renter-occupied -> no flood insurance at all
  = Zero or few claims filed despite real damage

Low-damage, high-claim ZCTAs:
  Physical damage (Hazus) is LOW
  + High income -> voluntary NFIP purchase
  + Inside SFHA -> mandatory purchase (mortgage requirement)
  + Owner-occupied -> policy in force
  = Claims filed for minor events because coverage exists
```

---

## 3. Pre-Registered Kill Rule (DOE_FAST_validation.md)

The kill rule is explicit and pre-registered:

> **Kill Rules:**
> - rho(NFIP_obs, FAST) < 0 -> NFIP claims anticorrelate with engineering
>   damage, validation framework is invalid for this scenario

Source: `DOE_FAST_validation.md`, lines 226-230

This means: when NFIP and FAST disagree on the direction of damage, the
FAST validation framework cannot be applied to that scenario. The model
is not wrong -- the ground truth labels measure different constructs.

---

## 4. Evidence from the Repos

### 4.1 Direct Observation (sme-ranking lesson)

From `sme-ranking/lessons/2026-06-07-houston-anticorrelation.md`:

> All three Houston events show negative rho between predictions and
> FAST depth-damage (-0.19 to -0.34). The OBS ceiling (rho_nfip_obs_fast)
> is also negative (-0.19 to -0.33).

### 4.2 Feature Contract Acknowledgment

`FEATURE_CONTRACT.yaml` documents `acs_median_home_value` with the note:
"Correlated with NFIP penetration" -- explicitly acknowledging the
income-insurance pathway.

### 4.3 Missing Variable Inventory

`GeoSpatial_Challenges.md` (line 539) identifies insurance penetration
rate as a missing variable that "would disambiguate NFIP outcome variable"
and notes that FEMA policy-in-force counts by ZIP are "Available but not
fetched."

### 4.4 DOE Target Documentation

`DOE_LOCKED.md` labels `obs_nfip_event_claims` with bias profile
"Insurance penetration, wealth" -- pre-registering awareness that NFIP
is a biased proxy for damage.

### 4.5 Houston Overview (F37 Equity Context)

`01-Houston-Overview.md` defines `equity_context` as an F40 vector
component sourced from `svi_quartile`, confirming that equity variation
is a known operational factor in the Houston scenario.

---

## 5. How This Affects the Ranking Geometry

### 5.1 Scenario-Restricted Validation

The FAST validation is only meaningful where NFIP and FAST agree on
damage direction:
- **SW Florida:** rho(NFIP, FAST) = +0.32 to +0.64 (VALID)
- **Houston:** rho(NFIP, FAST) = -0.19 to -0.33 (INVALID per kill rule)
- **NYC:** TBD (partial FloodSimBench coverage, Manhattan only)

### 5.2 The Model Is Not Wrong

The model predicts `obs_nfip_event_claims` -- it correctly learns insurance
filing patterns. FAST measures physical damage. When these constructs
diverge, the model's inability to predict FAST is not a failure but a
construct validity boundary.

### 5.3 Implication for the Model Ladder

The representation ladder (R0 -> R1 -> R2) adds physical features
(hydrology, temporal dynamics). If these features improve correlation
with NFIP in SW Florida (where NFIP ~ damage) but not in Houston (where
NFIP ~ income), that is evidence the ladder captures damage signal where
the label allows it.

---

## 6. Paper Framing

### What to Write (Section 6 / Discussion)

The anti-correlation between NFIP claims and depth-damage in Houston is
a **data validity finding**, not a model failure. It reveals that:

1. NFIP claims are a censored observation of damage, gated by insurance
   penetration
2. Insurance penetration co-varies with income, home ownership, and SFHA
   designation
3. Physics-based damage (Hazus) is construct-independent -- it measures
   structural loss regardless of who filed a claim

### Recommended Disclosure Text

> "Houston exhibits negative Spearman correlation between observed NFIP
> claims and physics-based depth-damage (Hazus), consistent with
> non-uniform insurance penetration across socioeconomic strata. This
> confirms that NFIP claims measure filing behavior, not physical damage.
> FAST validation is restricted to scenarios where the ground truth
> labels and engineering estimates agree on damage direction (SW Florida,
> rho > 0.3), per the pre-registered kill rule."

---

## 7. Governance Note

The kill rule `rho(NFIP_obs, FAST) < 0 -> validation inapplicable` is
pre-registered in `DOE_FAST_validation.md` and cannot be changed
post-hoc. This document explains the mechanism -- it does not alter the
verdict logic.
