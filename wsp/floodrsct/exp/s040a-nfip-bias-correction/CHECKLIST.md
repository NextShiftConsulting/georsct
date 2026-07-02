# S040a NFIP Bias Correction -- Pre-Flight Checklist

**Bucket:** `s3://swarm-floodrsct-data/`
**Job scripts:** `jobs/s040a_bias_correction/`
**Launchers:** `scripts/launch_s040a_*.py`
**DOE version:** v1.0

---

## 1. Git State

- [ ] `git status` -- working tree clean
- [ ] `git log -1 --oneline` -- commit hash: `__________`
- [ ] `git push`

---

## 2. Data Prerequisites

### C0 Baseline (existing raw NFIP claims)

| DR | Event | S3 Key | Status |
|----|-------|--------|--------|
| 4332 | Harvey 2017 | `raw/openfema/nfip_claims_dr4332.parquet` | [x] |
| 4466 | Imelda 2019 | `raw/openfema/nfip_claims_dr4466.parquet` | [x] |
| 4781 | Beryl 2024 | `raw/openfema/nfip_claims_dr4781.parquet` | [x] |
| 1603 | Katrina 2005 | `raw/openfema/nfip_claims_dr1603.parquet` | [x] |
| 4080 | Isaac 2012 | `raw/openfema/nfip_claims_dr4080.parquet` | [x] |
| 4458 | Barry 2019 | `raw/openfema/nfip_claims_dr4458.parquet` | [x] |
| 4611 | Ida 2021 LA | `raw/openfema/nfip_claims_dr4611.parquet` | [x] |
| 4085 | Sandy 2012 | `raw/openfema/nfip_claims_dr4085.parquet` | [x] |
| 4615 | Ida 2021 NY | `raw/openfema/nfip_claims_dr4615.parquet` | [x] |
| 4755 | NYC Sep 2023 | `raw/openfema/nfip_claims_dr4755.parquet` | [x] |
| 4673 | Ian 2022 | `raw/openfema/nfip_claims_dr4673.parquet` | [x] |
| 4828 | Helene 2024 | `raw/openfema/nfip_claims_dr4828.parquet` | [x] |
| 4834 | Milton 2024 | `raw/openfema/nfip_claims_dr4834.parquet` | [x] |
| 4699 | Hilary 2023 | `raw/openfema/nfip_claims_dr4699.parquet` | [x] |

### C2 IPW Data (new pull)

| State | S3 Key | Status |
|-------|--------|--------|
| TX | `raw/openfema/s040a/nfip_policies_TX.parquet` | [ ] |
| LA | `raw/openfema/s040a/nfip_policies_LA.parquet` | [ ] |
| NY | `raw/openfema/s040a/nfip_policies_NY.parquet` | [ ] |
| FL | `raw/openfema/s040a/nfip_policies_FL.parquet` | [ ] |
| CA | `raw/openfema/s040a/nfip_policies_CA.parquet` | [ ] |

### C4 Event Correction (derived from C0)

| DR | S3 Key | Status |
|----|--------|--------|
| All 14 | `processed/s040a/nfip_claims_c4_dr*.parquet` | [ ] |

### C7 Fusion Data (new pull)

| DR | S3 Key | Status |
|----|--------|--------|
| All 14 | `raw/openfema/s040a/ia_registrations_dr*.parquet` | [ ] |

### s035 Assembled Features (shared)

| Scenario | S3 Key | Status |
|----------|--------|--------|
| houston | `processed/houston/houston_event_features.parquet` | [x] |
| new_orleans | `processed/new_orleans/no_event_features.parquet` | [x] |
| southwest_florida | `processed/southwest_florida/swfl_event_features.parquet` | [x] |
| nyc | `processed/nyc/nyc_event_features.parquet` | [x] |
| riverside_coachella | `processed/riverside_coachella/rc_event_features.parquet` | [x] |

---

## 3. Dry Run

- [ ] `python scripts/launch_s040a_fetch_ia.py --dry-run`
- [ ] `python scripts/launch_s040a_fetch_policies.py --dry-run`
- [ ] `python scripts/launch_s040a_build_c4.py --dry-run`

---

## 4. Execution Record

| Phase | Job Name | Commit | Launch Time | Status |
|-------|----------|--------|-------------|--------|
| fetch_ia | | | | |
| fetch_policies | | | | |
| build_c4 | | | | |
| ablation_c0 | | | | |
| ablation_c4 | | | | |
| ablation_c7 | | | | |
