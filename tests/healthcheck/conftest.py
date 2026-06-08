"""Shared fixtures for failure taxonomy tests."""

from __future__ import annotations

import pytest

from georsct.healthcheck.thresholds import GEOSPATIAL_CONUS27, UNIVERSAL


@pytest.fixture
def preset():
    return GEOSPATIAL_CONUS27


@pytest.fixture
def universal_preset():
    return UNIVERSAL


def make_cert(**overrides) -> dict:
    """Build a certificate dict with sensible defaults."""
    base = {
        "R": 0.60,
        "S_sup": 0.10,
        "N": 0.30,
        "alpha": 0.60,
        "omega": 1.0,
        "kappa": 0.75,
        "tau": 1.0,
        "sigma": 0.10,
        "scenario": "test_scenario",
        "target": "test_target",
        "level": "r0",
        "task_type": "regression",
        "spatial_metric": 0.65,
        "random_metric": 0.60,
        "n_folds": 5,
    }
    base.update(overrides)
    return base


def make_envelope(phase: str, certs: list[dict] | None = None, **kwargs) -> dict:
    """Build a JSON envelope."""
    env: dict = {
        "experiment": "test-experiment",
        "phase": phase,
        "timestamp": "2026-06-06T12:00:00Z",
    }
    env.update(kwargs)
    if certs is not None:
        if "certificates" in phase:
            env["certificates"] = certs
        elif "diagnostics" in phase:
            env["cells"] = certs
        elif "gearbox" in phase:
            env["cells"] = certs
    return env
