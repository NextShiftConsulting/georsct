# GeoRSCT Architecture — Unified Spec (v2)

> **v2 correction.** v1's load-bearing decision named `georsct_kit` as a
> shared local library imported by both Colab and Lambda. That mechanism
> was wrong — it admits version skew (`pip install` drift between
> tutorial materials and production), and the compute path it described
> is not the path that ships. v2 replaces the shared-library boundary
> with a shared-API boundary: both Colab and production call the same
> `swarm-it-api` endpoints via the `swarm-it-adk` Python client. The
> underlying tutorial-honesty principle is preserved — the student in
> Notebook 5 invokes the same compute the emergency dashboard does —
> but the principle is now enforced *architecturally* (one compute
> implementation, behind one API), not by convention.

---

## 0. Two consumption modes

| Mode             | Client                                  | Compute path                                | Live sensors? | Auth                          | Latency budget         |
|------------------|-----------------------------------------|---------------------------------------------|---------------|-------------------------------|------------------------|
| **Tutorial**     | Colab notebook + `swarm-it-adk`          | `swarm-it-api` (public read-only key)        | No (flag off) | Public rate-limited API key   | Student-paced          |
| **Production**   | Emergency-management dashboard + `swarm-it-adk` | `swarm-it-api` (per-tenant key)             | Yes (flag on) | Per-tenant key with usage quota | ~200 ms p95            |

The branch is at **auth and flag**, not at compute. The `include_live_sensors`
flag and the API key are the only meaningful differences between a
student running Notebook 5 and an emergency manager triaging in real time.
Same endpoint. Same response schema. Same audit log format.

---

## 1. Layer diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                                │
│                                                                     │
│  ┌────────────────────┐         ┌─────────────────────────────┐    │
│  │  Colab notebooks   │         │  Production dashboard       │    │
│  │  (tutorial mode)   │         │  (72-hour decision mode)    │    │
│  └─────────┬──────────┘         └────────────┬────────────────┘    │
│            │                                  │                     │
│            │  from swarm_it import client     │                     │
│            │  adk = client.GeoRSCT(           │                     │
│            │    api_key=PUBLIC_TUTORIAL_KEY)  │                     │
│            │                                  │                     │
│            │              ┌───────────────────┘                     │
│            │              │                                          │
│            ▼              ▼                                          │
│       ┌──────────────────────────────────────┐                      │
│       │   swarm-it-adk (Python client)        │                      │
│       │   ─────────────────────────────       │                      │
│       │   adk.certify_geo(...)                │                      │
│       │   adk.certify_geo_trajectory(...)     │                      │
│       │   adk.certify_geo_ablation(...)       │                      │
│       │   adk.ceiling(...)                    │                      │
│       │   adk.ceiling_result(job_id)          │                      │
│       │                                       │                      │
│       │   * Pure request builder + HTTP       │                      │
│       │   * No compute happens here           │                      │
│       │   * Returns typed response models     │                      │
│       └────────────────┬─────────────────────┘                      │
└────────────────────────┼────────────────────────────────────────────┘
                         │ HTTPS + API key
                         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       swarm-it-api LAYER                            │
│                       API Gateway → Lambda or ECS                   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Endpoints                                                   │  │
│  │  ─────────                                                   │  │
│  │  POST /certify/geo                                           │  │
│  │  POST /certify/geo/trajectory                                │  │
│  │  POST /certify/geo/ablation                                  │  │
│  │  POST /ceiling      (202 Accepted, job_id)                   │  │
│  │  GET  /ceiling/{job_id}                                      │  │
│  │                                                              │  │
│  │  Compute (internal handlers, NOT exposed as a library):      │  │
│  │  ─────────                                                   │  │
│  │  • compute_simplex(R, S_sup, N)                              │  │
│  │  • compute_alpha(R, N)                                       │  │
│  │  • compute_kappa_compat(R, N)      [NOT swarm-it-adk's       │  │
│  │  •   compute_kappa, which is       embedding viability]      │  │
│  │  • compute_sigma(static_modal, live_modal)                   │  │
│  │  • SequentialGatekeeper.decide(certificate)                  │  │
│  │                                                              │  │
│  │  Async path: SQS → SageMaker Processing Job                  │  │
│  │  • n_ceiling_estimator (state-FIPS blocked, 200 bootstraps)  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────┬───────────────────────────┬───────────────────┘
                      │                           │
                      ▼                           ▼
┌────────────────────────┐      ┌────────────────────────────────────┐
│      CACHE LAYER       │      │       SENSOR FEED LAYER (live)     │
│  ElastiCache (Redis)   │      │   USGS waterservices               │
│  ┌──────────────────┐  │      │   NOAA National Water Model        │
│  │ Static tier      │  │      │   NWS api.weather.gov              │
│  │ (ETag-keyed)     │  │      │   HCFCD Harris County gauges       │
│  ├──────────────────┤  │      │                                    │
│  │ Live tier        │  │      │   * 5-min TTL on cached reads      │
│  │ (5-min TTL)      │  │      │   * Provider-side freshness owns   │
│  └──────────────────┘  │      │     the source of truth            │
└────────────┬───────────┘      └────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          DATA LAYER (S3)                            │
│                                                                     │
│  s3://swarm-yrsn-datasets/georsct/v24.002/                          │
│    data/                                                            │
│      georsct_table.parquet             (17 MB, 106 cols)            │
│      noaa_storm_events_long.parquet    (1.1 MB)                     │
│    oof/{embedding_arm}.parquet         (6 arms) [PENDING]           │
│    ceilings/{target}_{protocol}_v1.parquet                          │
│                                                                     │
│  Nightly sync to HF `rudymartin/georsct` v24.002                    │
│  (HF dataset is the canonical public artifact; S3 is the AWS mirror)│
└─────────────────────────────────────────────────────────────────────┘
```

The compute primitives that v1 located inside `georsct_kit` now live
*inside* the API Lambda handlers, not in a separately-installable
package. The names are the same (`compute_simplex`, `compute_alpha`,
…) because the manuscript uses them; the *boundary* is different. A
client never imports them.

**Naming note on `compute_kappa_compat`:** `swarm-it-adk` already
exports a function named `compute_kappa` that measures *embedding
viability* (dim / stable_rank ≥ 50). That is not RSCT certificate
κ = R*(1-N). The internal API handler is named `compute_kappa_compat`
to prevent silent semantic collision.

---

## 2. Data layer (S3)

```
s3://swarm-yrsn-datasets/georsct/v24.002/
├── data/
│   ├── georsct_table.parquet             (17 MB, 31,789 rows × 106 cols)
│   │   NFIP claims, TWI/watershed, ACS, SVI, flood zones,
│   │   HIFLD, drive times, NOAA aggregates are all columns here —
│   │   not separate parquet files.
│   └── noaa_storm_events_long.parquet    (1.1 MB, ~979K rows, 1996-2024)
│       Temporal sidecar for Experiment 1 (κ trajectory).
│       Row schema: zcta_id, year, event_type, damage_k, deaths, injuries
├── oof/
│   ├── pca32_v1.parquet          [PENDING — extract from s018/s019 outputs]
│   ├── spatial_lag_v1.parquet    [PENDING]
│   ├── graphsage_v1.parquet      [PENDING]
│   ├── geo_v1.parquet            [PENDING]
│   ├── domain_v1.parquet         [PENDING]
│   └── noisy_control_v1.parquet  [PENDING]
└── ceilings/
    └── {target}_{protocol}_v1.parquet     (pre-computed where stable)
```

The OOF files are not yet extracted. Current HF dataset has embedding
weights and transformer checkpoints from s018/s019; the per-ZCTA
out-of-fold prediction vectors needed by the notebook experiments
must be extracted in a separate SageMaker job before the tutorial is
ready to publish.

Same versioning, same ETag commitment, same nightly HF mirror.

The HF dataset is *optional* for tutorial attendees. They can
download it for offline exploration if they want, but the notebook
cells call the API rather than loading parquets locally. This is a
deliberate choice — the tutorial honesty contract requires that the
student touches the same compute path as production, and that path
goes through `swarm-it-api`, not through local parquet loads.

---

## 3. Cache layer

Two tiers, two different 5-minute intervals doing two different jobs.

### 3.1 Tier 1 — Static (ETag-invalidated)

ElastiCache Redis. Keys prefixed by S3 ETag. Background task does
`HEAD` on each watched S3 object every 5 minutes; ETag change triggers
key invalidation and an audit event.

Warm-up at cold start: Harris County + Gulf Coast ZCTAs (~3,400 rows),
plus the full `noaa_storm_events_long.parquet` (1.1 MB into /tmp).

### 3.2 Tier 2 — Live (5-min TTL)

ElastiCache Redis. Keys are `sensor:{provider}:{geo_unit}:{rounded_5min_timestamp}`.
Hard TTL = 300 s. No ETag polling — the provider's freshness contract
is the source of truth.

### 3.3 The same number, different operations

| Layer  | What runs every 5 minutes  | Why                                                |
|--------|----------------------------|----------------------------------------------------|
| Static | S3 `HEAD` poll for ETag    | Awareness of FIRM amendments / pipeline reruns     |
| Live   | Cache TTL expiry           | Freshness contract for sensor data during a storm  |

---

## 4. API layer

### 4.1 Endpoint table

| Method | Path                                | Mode  | Latency p95 | Cache hit rate (steady state) |
|--------|-------------------------------------|-------|-------------|-------------------------------|
| POST   | `/certify/geo`                      | sync  | 200 ms      | >95% (post warm-up)           |
| POST   | `/certify/geo/trajectory`           | sync  | 250 ms      | >99% (NOAA historical static) |
| POST   | `/certify/geo/ablation`             | sync  | 250 ms      | >95%                          |
| POST   | `/ceiling`                          | async | 100 ms (enqueue) | n/a                       |
| GET    | `/ceiling/{job_id}`                 | sync  | 50 ms       | >99% (results cached)         |

### 4.2 `/certify/geo` example

```http
POST /certify/geo
Content-Type: application/json
X-ADK-Version: swarm-it-adk==0.4.0

{
  "zcta_id":              "77002",
  "target":               "nfip_total_loss",
  "y_pred":               0.71,
  "y_true":               0.83,
  "snapshot_year":        2018,
  "embedding_arm":        "graphsage_v1",
  "include_live_sensors": false
}
```

```http
200 OK
Content-Type: application/json

{
  "certificate": {
    "R":          0.64,
    "S_sup":      0.18,
    "N":          0.18,
    "alpha":      0.78,
    "kappa":      0.62,
    "sigma":      0.18,
    "N_ceiling":  0.18,
    "decision":   "EXECUTE",
    "gate_reason":  "all gates passed",
    "gate_fired":   null
  },
  "metadata": {
    "zcta_id":       "77002",
    "state_fips":    "48",
    "county_fips":   "201",
    "target":        "nfip_total_loss",
    "snapshot_year": 2018,
    "embedding_arm": "graphsage_v1"
  },
  "audit": {
    "request_id":      "req_a1b2c3d4",
    "timestamp":       "2026-09-15T14:23:11Z",
    "dataset_version": "v24.002",
    "source_etags": {
      "georsct_table":  "\"abc123def\"",
      "oof_graphsage":  "\"ghi456jkl\""
    },
    "live_sensors_used": false,
    "api_version":     "swarm-it-api==1.0.0",
    "adk_version":     "swarm-it-adk==0.4.0"
  }
}
```

**`N_ceiling` in the sync response** is pulled from DynamoDB (written
by the async `/ceiling` job). If no ceiling has been computed for this
target + protocol, the field is `null`. The certificate is still valid
without it — the notebooks treat a null ceiling as "not yet estimated"
and prompt the student to run the async job.

**`decision`** uses the ADK `Certificate` vocabulary (`EXECUTE` /
`CAUTION` / `REFUSE`), matching what `swarm_it.certify()` returns in
non-geo contexts. The existing `/certify` endpoint uses
`ALLOW / REVIEW / BLOCK`; the geo endpoints use the canonical ADK enum.
This inconsistency in the existing spec is noted but not fixed here —
it's a pre-existing issue that predates the geo work.

### 4.3 Live-sensor σ behavior

When `include_live_sensors=true`, `compute_sigma` receives both the
static modal sources (SVI, NFIP aggregate, TWI, NOAA historical) and
the live sensor readings (USGS stream gauge, NOAA NWM). σ = std({κ_i})
over the union of all modal sources. Static sources use Tier 1 cache;
live sources use Tier 2 cache (5-min TTL). During a storm event, live
κ contributions will spike, raising σ and lowering the Gate 3 threshold.

### 4.4 Async `/ceiling` pattern

```http
POST /ceiling
{
  "target":          "nfip_total_loss",
  "protocol":        "imputation_state_blocked",
  "embedding_arms":  ["nystroem_krr", "mlp"],
  "bootstrap_reps":  200
}

→ 202 Accepted
{
  "job_id":            "ceil_nfip_total_loss_imp_v1_a1b2",
  "status":            "PENDING",
  "estimated_seconds": 45,
  "poll_url":          "/ceiling/ceil_nfip_total_loss_imp_v1_a1b2"
}
```

**Common case is fast.** Standard CONUS-27 ceilings for the six
canonical targets are pre-computed and cached in DynamoDB. A student
calling `ceiling(target="nfip_total_loss", ...)` for a target in the
standard suite gets a cache hit; polling returns immediately. The async
pattern is preserved for code-shape uniformity, not because the latency
is always 45 s.

### 4.5 Authentication

Public tutorial key: rate-limited, read-only, issued via HF Spaces.
Per-tenant key: usage quota, issued on request. `include_live_sensors`
is available on both; the tutorial key is simply rate-limited to
prevent abuse during a session.

---

## 5. Two request flows, walked end-to-end

### 5.1 Tutorial flow — Notebook 1, Cell 6 (the N-ceiling estimate)

```
Student runs in Colab:

    from swarm_it import client
    adk = client.GeoRSCT(api_key=PUBLIC_TUTORIAL_KEY)

    # POST /ceiling — returns immediately with a job_id (202 Accepted)
    job = adk.ceiling(
        target="nfip_total_loss",
        protocol="imputation_state_blocked",
        embedding_arms=["nystroem_krr", "mlp"],
        bootstrap_reps=200,
    )
    print(job.job_id, job.estimated_seconds)
    # → "ceil_nfip_total_loss_imp_v1_a1b2", 45

    # Poll until ready
    ceiling, lo, hi = adk.ceiling_result(job.job_id, wait=True)
    print(f"ceiling = {ceiling:.2f}, 95% CI [{lo:.2f}, {hi:.2f}]")
    # → ceiling = 0.18, 95% CI [0.14, 0.22]

[Colab]
   │  adk.ceiling(...)
   ▼
[swarm-it-adk client — request builder]
   │  HTTP POST /ceiling, public API key
   ▼
[API Gateway → Lambda]
   │  Lambda validates request, writes to SQS, returns 202
   ▼
[SQS → SageMaker Processing Job]
   │  Job runs georsct internal compute (Nystroem KRR + MLP,
   │  state-FIPS blocked, 200 bootstraps), ~45 seconds
   │  Writes (ceiling, lo, hi) to DynamoDB keyed by job_id
   ▼
[Student polls via adk.ceiling_result(...)]
   │  adk does GET /ceiling/{job_id} every 5 seconds
   │  until result is ready
   ▼
[Notebook cell output] ceiling = 0.18, CI [0.14, 0.22]
```

The student watches the same async pattern an emergency dashboard
watches: enqueue, poll, render. The job runs in the production
SageMaker fleet, not on the student's Colab VM.

### 5.2 Production flow — POST /certify/geo with live sensors

```
[Dashboard]
   adk.certify_geo(
       zcta_id="77002",
       target="nfip_total_loss",
       y_pred=0.71, y_true=0.83,
       snapshot_year=2018,
       embedding_arm="graphsage_v1",
       include_live_sensors=True,
   )
   │
   ▼
[swarm-it-adk client — same request builder, different auth key]
   │  HTTP POST /certify/geo
   ▼
[API Gateway → Lambda]
   │
   ├── Static features for ZCTA 77002 from Redis (Tier 1)
   │      hit: < 1 ms  /  miss: S3 Select ~50 ms
   │
   ├── OOF prediction for {graphsage_v1, 77002} from Redis (Tier 1)
   │
   ├── ZCTA → HUC8 crosswalk lookup (Tier 1 static)
   │
   ├── Live sensor readings for HUC8 12040104 from Redis (Tier 2)
   │      hit: < 1 ms  /  miss: parallel fetch USGS + NOAA NWM ~150 ms
   │
   ├── Internal: compute_simplex(features, oof_pred)        →  (R, S_sup, N)
   ├── Internal: compute_alpha(R, N)                        →  α = 0.78
   ├── Internal: compute_kappa_compat(R, N)                 →  κ = 0.62
   ├── Internal: compute_sigma(static_modal, live_modal)    →  σ = 0.18 OR spike
   ├── Ceilings lookup from DynamoDB                        →  N_ceil = 0.18 or null
   ├── Internal: SequentialGatekeeper.decide(certificate)   →  decision
   │
   ▼
[Response]  certificate + metadata + audit, ~200 ms p95
```

Same endpoint as the tutorial student calls. Same compute functions
inside the Lambda. The only differences are `include_live_sensors=true`
and a per-tenant auth key.

---

## 6. Cache architecture details

### 6.1 ETag polling

ElastiCache Redis. On cold start, the background poller pre-warms Harris
County + Gulf Coast ZCTAs. The poller runs every 5 minutes per watched
S3 key; a changed ETag triggers cache invalidation and an `audit_event`
write to DynamoDB.

```python
# Pseudocode — internal to swarm-it-api
async def poll_static_etags():
    for key, last_etag in watched_keys.items():
        resp = s3.head_object(Bucket=BUCKET, Key=key)
        new_etag = resp["ETag"]
        if new_etag != last_etag:
            redis.delete_pattern(f"static:{last_etag}:*")
            watched_keys[key] = new_etag
            audit_log.write("etag_invalidated", key=key, old=last_etag, new=new_etag)
```

### 6.2 What never gets cached

- HMAC signatures (recomputed per request from live payload)
- Audit log entries (write-once to DynamoDB)
- N-ceiling job status (DynamoDB, not Redis)

### 6.3 Sizing

| Tier   | Working set                                              | Memory estimate | Redis instance       |
|--------|----------------------------------------------------------|-----------------|----------------------|
| Static | ~32K ZCTAs × 6 OOF arms + features                       | ~80 MB          | cache.t3.small       |
| Live   | ~2,200 HUC8s × 4 providers × 5-min buckets               | ~10 MB          | (same instance)      |

A `cache.t3.small` (~$14/mo) covers both tutorial and production load.
Peak at ~25 req/s during a co-occurring tutorial session + active storm
window, still well inside the t3.small envelope.

---

## 7. v1 vs. v2 scope

### 7.1 In scope for v1

Four read-only endpoints + async `/ceiling`, two-tier cache, USGS +
NOAA NWM live sensors, DynamoDB audit log, public read-only key for
tutorial use.

### 7.2 Deferred to v2

Write endpoints, per-tenant data isolation, WebSocket streaming,
automated re-encoding pipeline, multi-region active-active.

### 7.3 `swarm-it-adk` package scope

The ADK is a thin client. It contains:

- Pydantic request/response models matching the OpenAPI spec.
- An HTTP client (`httpx`) with retry and rate-limit awareness.
- The polling helper used by `adk.ceiling_result(job_id, wait=True)`.
- Type-stub files for IDE autocomplete in the tutorial Colab notebooks.

It does **not** contain:

- Simplex / α / κ / σ / N_ceiling / gatekeeper compute.
- Any data loading from S3 or HF (the API handles all data access).
- Any local caching beyond a per-process LRU on response objects for
  the duration of a single notebook session.

If `pip install swarm-it-adk` takes more than ~30 seconds on a Colab
free-tier instance, something has drifted. The package is intended to
stay light.

---

## 8. Cost estimate (rough, monthly)

| Source                                                  | Marginal cost per 100-student session |
|---------------------------------------------------------|---------------------------------------|
| 15K API calls × $3.50/M (API Gateway)                   | $0.05                                 |
| Lambda compute (100 ms × 256 MB × 15K requests)         | $0.06                                 |
| SageMaker Processing for ceiling cache misses (~50 runs × $0.014) | $0.70                       |
| **Total per tutorial session**                          | **~$0.81**                            |

A monthly active hurricane season with tutorial sessions in the same
month tops out around $250.

---

## 9. Decisions to confirm before OpenAPI build

1. **Base URL.** `api.swarm-it.example` (formerly `api.georsct.example`
   placeholder). Production domain TBC.
2. **API key issuance.** Self-serve via HF Spaces, or manual-by-request?
   Self-serve scales; manual-by-request gives tighter usage tracking
   for the SIGSPATIAL audience.
3. **Live sensor providers in v1.** USGS + NOAA NWM confirmed.
   NWS and HCFCD deferred to v1.1.
4. **HUC8 crosswalk.** Majority-overlap default; per-ZCTA coverage
   fraction recorded as metadata for low-coverage cases.
5. **`swarm-it-adk` versioning policy.** Pin major version at v1 launch;
   minor bumps additive-only within v1.x; the `X-ADK-Version` header
   tracks client versions in production audit logs.

---

## 10. The architecture's single load-bearing decision

> **`swarm-it-api` is the single compute layer, called by both Colab
> and production via `swarm-it-adk`. There is no shared local library
> containing the certificate primitives. The tutorial honesty contract
> is enforced architecturally — same endpoints, same auth model, same
> response schema — not by convention.**

The principle this defends: a student in Notebook 5 should be running
the exact compute path the emergency dashboard runs. v1 protected the
principle through a Python library that *could* drift between
deployments. v2 protects it by making the compute path inaccessible
*except* via the API — there is no parallel path to drift away from.

This change costs the org something real (Lambda + SageMaker minutes
the student VMs previously absorbed for free) and buys something more
valuable: the impossibility of tutorial materials silently diverging
from production behavior across the SIGSPATIAL 2027 → 2028 cycle,
the NeurIPS submission → publication cycle, and any future deployment
with a different tenant.

---

## Appendix v2.A — What ripples out of this change

| Artifact                                               | What changes                                                                       |
|--------------------------------------------------------|------------------------------------------------------------------------------------|
| `NOTEBOOK_DESIGN_FIVE_PRIMITIVES.md`                   | Cells that read `from georsct_kit import ...` become `from swarm_it import client` + `adk.<method>(...)`. ~12 cell-level edits across 5 notebooks. |
| `NOTEBOOK_DESIGN_IMPOSSIBILITY_STACK.md`               | Same. ~10 cell-level edits. Two cells are already pseudocode placeholders. |
| `SIGSPATIAL_2027_TUTORIAL_PROPOSAL.md`                 | §5 "Materials Provided" replaces "the `georsct_kit` Python package (PyPI)" with "the `swarm-it-adk` Python client (PyPI) and the `swarm-it-api` OpenAPI 3.1 spec." §6 reproducibility pin becomes `swarm-it-adk==X` + `swarm-it-api==Y` from the audit block. |
| OpenAPI spec (not yet built)                            | Reflects the corrected architecture from the start. No retrofit needed.        |
| GeoRSCT manuscript                                      | Does not currently name `georsct_kit`. No change needed.                      |

## Appendix v2.B — Pre-existing OpenAPI inconsistencies (not fixed here)

The existing `swarm-it-api/marketplace/openapi.yaml` has these
issues that predate the geo work. They should be fixed in a separate
pass, not mixed into the geo endpoint additions:

| Issue | Location | Notes |
|-------|----------|-------|
| Three naming conventions for R/S/N/κ/α/σ | `CertificationScores` vs `DebugInfo` vs `PodcastCertificate` | Geo endpoints introduce a fourth if not standardized |
| Three decision enums | `ALLOW/REVIEW/BLOCK` vs `EXECUTE/CAUTION/REFUSE` vs `EXECUTE/REVIEW/REJECT` | Geo endpoints use ADK enum |
| `gate_stage` integer with no description | `DebugInfo` | Values 0-4 undocumented |
| `metadata` string-only | `CertifyRequest` | Numeric ZCTA metadata (year, population) requires a separate geo schema |
| `certificate_id` optional | `CertifyResponse` | Required for ceiling audit trail |
