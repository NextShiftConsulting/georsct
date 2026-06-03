"""Tests for NFIP historical temporal gate.

Validates:
  - Claims with dateOfLoss < cutoff are included
  - Claims with dateOfLoss >= cutoff are excluded (strict less-than)
  - Same-day claims are excluded (boundary condition)
  - Timezone-aware and timezone-naive cutoffs both work
  - First-event scenario returns zero frequency/severity
  - Multi-event accumulation is monotonically non-decreasing
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "data" / "floodrsct" / "jobs"))

from build_nfip_historical import compute_historical_features


@pytest.fixture
def sample_claims():
    """Claims spanning three events for two ZCTAs."""
    return pd.DataFrame({
        "zcta_id": ["77001", "77001", "77001", "77002", "77002"],
        "dateOfLoss": pd.to_datetime([
            "2017-08-25",  # Harvey
            "2019-09-18",  # Imelda
            "2024-07-08",  # Beryl
            "2017-08-26",  # Harvey
            "2024-07-09",  # Beryl
        ]),
        "amountPaidOnBuildingClaim": [50000, 30000, 20000, 40000, 15000],
    })


def test_strict_less_than_excludes_same_day(sample_claims):
    """Claims on the cutoff date must be excluded (strict <, not <=)."""
    result = compute_historical_features(
        sample_claims,
        zcta_ids=["77001", "77002"],
        cutoff_date="2017-08-25",
    )
    # 77001's first claim is ON the cutoff -> excluded
    row_77001 = result[result["zcta_id"] == "77001"].iloc[0]
    assert row_77001["nfip_historical_frequency"] == 0


def test_claims_before_cutoff_included(sample_claims):
    """Claims before cutoff are counted."""
    result = compute_historical_features(
        sample_claims,
        zcta_ids=["77001", "77002"],
        cutoff_date="2019-09-18",
    )
    # 77001 has one claim before Imelda (Harvey 2017-08-25)
    row_77001 = result[result["zcta_id"] == "77001"].iloc[0]
    assert row_77001["nfip_historical_frequency"] == 1
    assert row_77001["nfip_historical_severity"] == pytest.approx(50000.0)

    # 77002 has one claim before Imelda (Harvey 2017-08-26)
    row_77002 = result[result["zcta_id"] == "77002"].iloc[0]
    assert row_77002["nfip_historical_frequency"] == 1


def test_claims_after_cutoff_excluded(sample_claims):
    """Claims on or after cutoff are excluded."""
    result = compute_historical_features(
        sample_claims,
        zcta_ids=["77001", "77002"],
        cutoff_date="2019-09-18",
    )
    # 77001's Imelda (same day as cutoff) and Beryl claims must be excluded
    row_77001 = result[result["zcta_id"] == "77001"].iloc[0]
    assert row_77001["nfip_historical_frequency"] == 1  # only Harvey


def test_first_event_returns_zero(sample_claims):
    """Cutoff before any claims -> zero frequency and severity."""
    result = compute_historical_features(
        sample_claims,
        zcta_ids=["77001", "77002"],
        cutoff_date="2017-01-01",
    )
    assert (result["nfip_historical_frequency"] == 0).all()
    assert (result["nfip_historical_severity"] == 0.0).all()


def test_monotonic_accumulation(sample_claims):
    """Later cutoffs must produce >= frequency than earlier cutoffs."""
    cutoffs = ["2017-01-01", "2017-08-26", "2019-09-19", "2024-07-10"]
    prev_total = 0
    for cutoff in cutoffs:
        result = compute_historical_features(
            sample_claims,
            zcta_ids=["77001", "77002"],
            cutoff_date=cutoff,
        )
        total = result["nfip_historical_frequency"].sum()
        assert total >= prev_total, (
            f"Frequency decreased from {prev_total} to {total} at cutoff {cutoff}"
        )
        prev_total = total


def test_tz_aware_claims():
    """Timezone-aware dateOfLoss must work with naive cutoff."""
    claims = pd.DataFrame({
        "zcta_id": ["77001", "77001"],
        "dateOfLoss": pd.to_datetime([
            "2017-08-25T12:00:00",
            "2019-09-18T06:00:00",
        ]).tz_localize("UTC"),
        "amountPaidOnBuildingClaim": [50000, 30000],
    })
    result = compute_historical_features(
        claims,
        zcta_ids=["77001"],
        cutoff_date="2019-09-18",
    )
    # Only Harvey claim (2017-08-25) is before cutoff
    row = result[result["zcta_id"] == "77001"].iloc[0]
    assert row["nfip_historical_frequency"] == 1


def test_missing_zcta_gets_zero(sample_claims):
    """ZCTAs with no claims at all get zero frequency/severity."""
    result = compute_historical_features(
        sample_claims,
        zcta_ids=["77001", "77002", "77099"],
        cutoff_date="2024-07-10",
    )
    row_99 = result[result["zcta_id"] == "77099"].iloc[0]
    assert row_99["nfip_historical_frequency"] == 0
    assert row_99["nfip_historical_severity"] == 0.0


def test_severity_is_mean_not_sum(sample_claims):
    """Severity = mean claim amount, not total."""
    result = compute_historical_features(
        sample_claims,
        zcta_ids=["77001"],
        cutoff_date="2019-09-19",
    )
    # 77001 has two claims before day after Imelda: 50000 (Harvey) + 30000 (Imelda)
    row = result[result["zcta_id"] == "77001"].iloc[0]
    assert row["nfip_historical_frequency"] == 2
    assert row["nfip_historical_severity"] == pytest.approx(40000.0)  # (50k+30k)/2
