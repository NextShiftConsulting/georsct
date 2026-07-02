#!/usr/bin/env python3
"""
fix_croissant.py -- One-time Croissant repair:
  1. Upload noaa_storm_events_long.parquet sidecar to HF
  2. Prune georsct_simplified_001.geoparquet ghost from distribution
  3. Expand cdc-places-ci stub (1 field) to full 44-field schema
  4. Expand acs-moe stub (1 field) to full 33-field schema
  5. Add noaa-storm-events-long recordSet (7 fields)
"""
import boto3
import json
from datetime import datetime, timezone
from huggingface_hub import HfApi
from swarm_auth import get_aws_credentials, get_credential

_aws = get_aws_credentials()
s3 = boto3.client("s3", **_aws)
token = get_credential("HF_TOKEN")
api = HfApi(token=token)
HF_REPO = "rudymartin/georsct"

# ── 1. Upload noaa_storm_events_long.parquet ──────────────────────────────────
print("=== 1. UPLOADING noaa_storm_events_long.parquet ===")
obj = s3.get_object(
    Bucket="swarm-yrsn-datasets",
    Key="rsct_curriculum/series_018/processed/noaa_storm_events_long.parquet",
)
data = obj["Body"].read()
print(f"  {len(data)/1e6:.1f} MB")
api.upload_file(
    path_or_fileobj=data,
    path_in_repo="noaa_storm_events_long.parquet",
    repo_id=HF_REPO,
    repo_type="dataset",
    commit_message=(
        "feat(sidecar): add noaa_storm_events_long.parquet "
        "-- 979,939 rows x ZCTA x year 1996-2024 for Version C Experiment 1"
    ),
)
print("  -> uploaded")

# ── 2. Load current Croissant ─────────────────────────────────────────────────
print("\n=== 2. LOADING CROISSANT ===")
cp = api.hf_hub_download(HF_REPO, "croissant.json", repo_type="dataset", force_download=True)
with open(cp) as f:
    cr = json.load(f)

# ── 3. Prune ghost fileObject ─────────────────────────────────────────────────
print("\n=== 3. PRUNING GHOST fileObject ===")
before = len(cr["distribution"])
cr["distribution"] = [
    fo for fo in cr["distribution"]
    if fo.get("name") != "georsct_simplified_001.geoparquet"
]
print(f"  distribution: {before} -> {len(cr['distribution'])}")

# ── 4. Add noaa-long fileObject ───────────────────────────────────────────────
existing_fo = {fo["name"] for fo in cr["distribution"]}
if "noaa-storm-events-long" not in existing_fo:
    cr["distribution"].append({
        "@type": "cr:FileObject",
        "@id": "noaa-storm-events-long",
        "name": "noaa-storm-events-long",
        "description": (
            "NOAA Storm Events flood history by ZCTA and year (1996-2024). "
            "Sidecar for Version C Experiment 1 (Invariance Test): "
            "use year=2018 (pre-Florence), 2019 (recovery), 2020 (pre-Isaias) "
            "for temporal certificate trajectories. Joins on zcta_id."
        ),
        "contentUrl": "noaa_storm_events_long.parquet",
        "encodingFormat": "application/x-parquet",
    })
    print("  Added noaa-storm-events-long to distribution")

# ── 5. Build full recordSet schemas ───────────────────────────────────────────
print("\n=== 5. BUILDING FULL SCHEMAS ===")

# cdc-places-ci: 21 targets x 2 bounds + zcta_id + has_cdc_ci = 44 fields
cdc_targets = [
    "arthritis", "binge_drinking", "high_blood_pressure", "bp_medicated", "cancer",
    "asthma", "coronary_heart_disease", "annual_checkup", "cholesterol_screening",
    "copd", "smoking", "dental_visit", "diabetes", "high_cholesterol",
    "chronic_kidney_disease", "physical_inactivity", "mental_health_not_good",
    "obesity", "physical_health_not_good", "sleep_less_7hr", "stroke",
]
cdc_fields = [
    {"@type": "cr:Field", "@id": "cdc-places-ci/zcta_id", "name": "zcta_id",
     "dataType": "sc:Text", "description": "5-digit ZCTA identifier (join key)"},
]
for t in cdc_targets:
    for bound in ("low", "high"):
        col = f"target_{t}_ci_{bound}"
        cdc_fields.append({
            "@type": "cr:Field", "@id": f"cdc-places-ci/{col}", "name": col,
            "dataType": "sc:Float",
            "description": f"CDC PLACES 95% CI {bound}er bound for target_{t} (prevalence %)",
        })
cdc_fields.append({
    "@type": "cr:Field", "@id": "cdc-places-ci/has_cdc_ci", "name": "has_cdc_ci",
    "dataType": "sc:Boolean",
    "description": "True if CDC PLACES CI data is available for this ZCTA",
})

# acs-moe: 31 features + zcta_id + has_acs_moe = 33 fields
acs_features = [
    "total_pop", "median_age", "median_home_value", "median_rent", "median_year_built",
    "median_hh_income", "gini_index", "mean_commute_min", "pct_white", "pct_black",
    "pct_asian", "pct_hispanic", "pct_bachelors", "pct_under_18", "pct_female",
    "pct_veterans", "pct_foreign_born", "pct_english_only", "pct_drive_alone",
    "pct_transit", "pct_wfh", "pct_owner_occupied", "pct_renter_occupied", "pct_vacant",
    "pct_below_poverty", "pct_food_stamps", "unemployment_rate", "pct_graduate",
    "pct_walk_bike", "pct_no_vehicle", "pct_no_insurance",
]
moe_fields = [
    {"@type": "cr:Field", "@id": "acs-moe/zcta_id", "name": "zcta_id",
     "dataType": "sc:Text", "description": "5-digit ZCTA identifier (join key)"},
]
for feat in acs_features:
    col = f"acs_{feat}_moe"
    moe_fields.append({
        "@type": "cr:Field", "@id": f"acs-moe/{col}", "name": col,
        "dataType": "sc:Float",
        "description": f"ACS 5-year margin of error (90% CI) for acs_{feat}",
    })
moe_fields.append({
    "@type": "cr:Field", "@id": "acs-moe/has_acs_moe", "name": "has_acs_moe",
    "dataType": "sc:Boolean",
    "description": "True if ACS MOE data is available for this ZCTA",
})

# noaa-storm-events-long: 7 fields
noaa_fields = [
    {"@type": "cr:Field", "@id": "noaa-long/zcta_id",           "name": "zcta_id",          "dataType": "sc:Text",    "description": "5-digit ZCTA identifier (join key)"},
    {"@type": "cr:Field", "@id": "noaa-long/year",              "name": "year",              "dataType": "sc:Integer", "description": "Event year (1996-2024). Key years: 2018 Florence, 2019 recovery, 2020 Isaias"},
    {"@type": "cr:Field", "@id": "noaa-long/flood_events",      "name": "flood_events",      "dataType": "sc:Integer", "description": "NOAA flood event count (Flash Flood + Flood + Coastal Flood + Lakeshore Flood)"},
    {"@type": "cr:Field", "@id": "noaa-long/deaths",            "name": "deaths",            "dataType": "sc:Integer", "description": "Flood-related deaths (direct + indirect)"},
    {"@type": "cr:Field", "@id": "noaa-long/injuries",          "name": "injuries",          "dataType": "sc:Integer", "description": "Flood-related injuries"},
    {"@type": "cr:Field", "@id": "noaa-long/property_damage_k", "name": "property_damage_k", "dataType": "sc:Float",   "description": "Property damage in $1000s"},
    {"@type": "cr:Field", "@id": "noaa-long/crop_damage_k",     "name": "crop_damage_k",     "dataType": "sc:Float",   "description": "Crop damage in $1000s"},
]

# ── 6. Replace stubs / add new recordSets ─────────────────────────────────────
rs_map = {rs["name"]: i for i, rs in enumerate(cr["recordSet"])}

cr["recordSet"][rs_map["cdc-places-ci"]] = {
    "name": "cdc-places-ci",
    "@type": "cr:RecordSet",
    "description": "CDC PLACES 95% CIs for 21 health targets (42 float cols + has_cdc_ci). 32,409 ZCTAs. Joins to georsct-main on zcta_id.",
    "field": cdc_fields,
}
print(f"  cdc-places-ci: 1 stub -> {len(cdc_fields)} fields")

cr["recordSet"][rs_map["acs-moe"]] = {
    "name": "acs-moe",
    "@type": "cr:RecordSet",
    "description": "ACS 5-year margins of error (90% CI) for 31 ACS features (32 float cols + has_acs_moe). 33,774 ZCTAs. Joins to georsct-main on zcta_id.",
    "field": moe_fields,
}
print(f"  acs-moe: 1 stub -> {len(moe_fields)} fields")

if "noaa-storm-events-long" not in rs_map:
    cr["recordSet"].append({
        "name": "noaa-storm-events-long",
        "@type": "cr:RecordSet",
        "description": (
            "NOAA flood events per ZCTA per year (1996-2024). 979,939 rows. "
            "Version C Experiment 1: filter year in [2018, 2019, 2020] for "
            "Florence/recovery/Isaias temporal snapshots. Joins on zcta_id."
        ),
        "field": noaa_fields,
    })
    print(f"  noaa-storm-events-long: added ({len(noaa_fields)} fields)")

# ── 7. Bump date and push ─────────────────────────────────────────────────────
cr["dateModified"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")

print("\n=== 6. PUSHING CROISSANT ===")
print(f"  recordSets ({len(cr['recordSet'])}): {[rs['name'] for rs in cr['recordSet']]}")
print(f"  distribution ({len(cr['distribution'])}): {[fo['name'] for fo in cr['distribution']]}")

api.upload_file(
    path_or_fileobj=json.dumps(cr, indent=2).encode(),
    path_in_repo="croissant.json",
    repo_id=HF_REPO,
    repo_type="dataset",
    commit_message=(
        "fix(croissant): prune geoparquet ghost; expand cdc-ci + acs-moe stubs to full schemas; "
        "add noaa-storm-events-long sidecar recordSet (Version C Experiment 1)"
    ),
)
print("  -> pushed")

# ── 8. Verify parquet files on HF ─────────────────────────────────────────────
print("\n=== 7. HF PARQUET FILES ===")
for f in sorted(api.list_repo_files(HF_REPO, repo_type="dataset")):
    if f.endswith(".parquet"):
        print(f"  {f}")

print("\nDone.")
