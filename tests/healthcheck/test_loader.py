"""Loader tests: phase classification, level extraction, folder loading."""

import json
import os
import tempfile

import pytest

from georsct.healthcheck.loader import (
    _classify_phase,
    _extract_level,
    get_cells,
    load_folder,
)
from georsct.healthcheck.models import WorkflowSnapshot


class TestClassifyPhase:
    def test_certificates(self):
        assert _classify_phase("certificates_r0") == "certificates"
        assert _classify_phase("certificates_r1") == "certificates"

    def test_diagnostics(self):
        assert _classify_phase("diagnostics_r0") == "diagnostics"

    def test_admission(self):
        assert _classify_phase("4_block_admission_r0") == "admission"

    def test_routing(self):
        assert _classify_phase("6_dgm_routing") == "routing"
        assert _classify_phase("dgm_routing_r0") == "routing"

    def test_gearbox(self):
        assert _classify_phase("gearbox_warmup") == "gearbox"

    def test_money_table(self):
        assert _classify_phase("money_table") == "money_table"
        assert _classify_phase("uplift_table") == "money_table"

    def test_unknown(self):
        assert _classify_phase("summary") is None
        assert _classify_phase("metadata") is None


class TestExtractLevel:
    def test_r0(self):
        assert _extract_level("certificates_r0") == "r0"

    def test_r2(self):
        assert _extract_level("diagnostics_r2") == "r2"

    def test_no_level(self):
        assert _extract_level("gearbox_warmup") is None

    def test_complex_phase(self):
        assert _extract_level("4_block_admission_r1") == "r1"


class TestLoadFolder:
    def _write_json(self, tmpdir, name, data):
        path = os.path.join(tmpdir, name)
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def test_empty_folder(self):
        with tempfile.TemporaryDirectory() as d:
            snap = load_folder(d)
            assert len(snap.loader_warnings) > 0

    def test_empty_folder_strict(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(ValueError):
                load_folder(d, strict=True)

    def test_not_a_directory(self):
        with pytest.raises(FileNotFoundError):
            load_folder("/nonexistent/path")

    def test_malformed_json(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w") as f:
                f.write("{bad json")
            snap = load_folder(d)
            assert any("MALFORMED_JSON" in w for w in snap.loader_warnings)

    def test_malformed_json_strict(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "bad.json")
            with open(path, "w") as f:
                f.write("{bad json")
            with pytest.raises(ValueError):
                load_folder(d, strict=True)

    def test_missing_phase(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_json(d, "nophase.json", {"data": "x"})
            snap = load_folder(d)
            assert any("MISSING_PHASE" in w for w in snap.loader_warnings)

    def test_load_certificates(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_json(d, "certs.json", {
                "phase": "certificates_r0",
                "preset_id": "CONUS27",
                "certificates": [
                    {"scenario": "sc1", "target": "t1", "alpha": 0.60},
                ],
            })
            snap = load_folder(d)
            assert "r0" in snap.certificates
            assert len(snap.certificates["r0"]) == 1
            assert snap.provenance["r0"] == "CONUS27"

    def test_load_diagnostics(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_json(d, "diag.json", {
                "phase": "diagnostics_r0",
                "cells": [
                    {"scenario": "sc1", "target": "t1", "diag_leakage": 0.9},
                ],
            })
            snap = load_folder(d)
            assert "r0" in snap.diagnostics

    def test_load_admission(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_json(d, "adm.json", {
                "phase": "4_block_admission_r0",
                "admission_table": [
                    {"scenario": "sc1", "target": "t1", "enforcement_decision": "EXECUTE"},
                ],
            })
            snap = load_folder(d)
            assert snap.admission_table is not None
            assert len(snap.admission_table) == 1

    def test_load_routing(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_json(d, "route.json", {
                "phase": "6_dgm_routing",
                "strategies": {"default": {"routing_table": []}},
            })
            snap = load_folder(d)
            assert snap.routing_table is not None

    def test_duplicate_phase_warns(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_json(d, "a_certs.json", {
                "phase": "certificates_r0",
                "certificates": [{"scenario": "sc1", "target": "t1"}],
            })
            self._write_json(d, "b_certs.json", {
                "phase": "certificates_r0",
                "certificates": [{"scenario": "sc1", "target": "t1"}],
            })
            snap = load_folder(d)
            assert any("DUPLICATE_PHASE" in w for w in snap.loader_warnings)

    def test_no_certificates_warns(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_json(d, "diag.json", {
                "phase": "diagnostics_r0",
                "cells": [],
            })
            snap = load_folder(d)
            assert any("No certificate" in w for w in snap.loader_warnings)


class TestGetCells:
    def test_dedup(self):
        snap = WorkflowSnapshot()
        snap.certificates = {
            "r0": [
                {"scenario": "a", "target": "x"},
                {"scenario": "a", "target": "x"},
                {"scenario": "b", "target": "y"},
            ]
        }
        cells = get_cells(snap)
        assert cells == [("a", "x"), ("b", "y")]
