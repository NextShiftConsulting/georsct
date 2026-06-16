"""Architecture enforcement tests for GeoRSCT-X.

Verifies hex-arch import boundaries. If any of these tests fail,
someone has introduced a forbidden cross-layer dependency.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GEORSCT_ROOT = Path(__file__).resolve().parent.parent.parent / "georsct"


def _get_imports(module_path: Path) -> set[str]:
    """Extract all import targets from a Python file."""
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def _collect_py_files(subdir: str) -> list[Path]:
    """All .py files in a georsct subpackage."""
    d = _GEORSCT_ROOT / subdir
    if not d.exists():
        return []
    return [p for p in d.rglob("*.py") if p.name != "__init__.py"]


# ---------------------------------------------------------------------------
# Forbidden external imports (hex-arch boundary)
# ---------------------------------------------------------------------------

_FORBIDDEN_EXTERNALS = {"yrsn", "yrsn_controlplane", "swarm_it", "swarm_auth"}


class TestNoExternalImports:
    """No GeoRSCT-X module may import yrsn, controlplane, or swarm."""

    @pytest.mark.parametrize("subdir", [
        "contracts", "provenance", "execution", "evaluation", "experts",
    ])
    def test_no_forbidden_imports(self, subdir: str):
        for py_file in _collect_py_files(subdir):
            imports = _get_imports(py_file)
            violations = imports & _FORBIDDEN_EXTERNALS
            assert not violations, (
                f"{py_file.relative_to(_GEORSCT_ROOT)} imports {violations}"
            )


# ---------------------------------------------------------------------------
# Cross-layer import rules
# ---------------------------------------------------------------------------

class TestLayerBoundaries:

    def test_execution_does_not_import_task_gold(self):
        """Execution layer must never see gold values."""
        for py_file in _collect_py_files("execution"):
            source = py_file.read_text(encoding="utf-8")
            assert "task_gold" not in source, (
                f"{py_file.name} imports task_gold — gold leak!"
            )

    def test_execution_does_not_import_scoring(self):
        """Execution layer must not import evaluation."""
        for py_file in _collect_py_files("execution"):
            imports = _get_imports(py_file)
            full_imports = set()
            tree = ast.parse(
                py_file.read_text(encoding="utf-8"),
                filename=str(py_file),
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    full_imports.add(node.module)
            scoring_imports = [
                m for m in full_imports
                if "evaluation" in m or "scoring" in m
            ]
            assert not scoring_imports, (
                f"{py_file.name} imports {scoring_imports}"
            )

    def test_execution_does_not_import_store(self):
        """Execution layer must not import provenance store."""
        for py_file in _collect_py_files("execution"):
            source = py_file.read_text(encoding="utf-8")
            assert "provenance.store" not in source, (
                f"{py_file.name} imports provenance.store"
            )

    def test_evaluation_does_not_import_execution(self):
        """Scoring is a leaf — no upward imports."""
        for py_file in _collect_py_files("evaluation"):
            tree = ast.parse(
                py_file.read_text(encoding="utf-8"),
                filename=str(py_file),
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    assert "execution" not in node.module, (
                        f"{py_file.name} imports {node.module}"
                    )

    def test_store_does_not_import_scoring(self):
        """Store persists scores, does not compute them."""
        for py_file in _collect_py_files("provenance"):
            if py_file.name == "store.py":
                source = py_file.read_text(encoding="utf-8")
                assert "scoring" not in source, (
                    "store.py imports scoring — store must not compute scores"
                )

    def test_experts_do_not_import_concrete_certifier(self):
        """Expert ABCs and impls must not import domain certifiers."""
        for py_file in _collect_py_files("experts"):
            source = py_file.read_text(encoding="utf-8")
            for forbidden in [
                "construct_certificate",
                "kappa_reconstruct",
                "ReadinessCertificate",
            ]:
                assert forbidden not in source, (
                    f"{py_file.name} references {forbidden}"
                )


# ---------------------------------------------------------------------------
# Runtime import verification
# ---------------------------------------------------------------------------

class TestModulesImportCleanly:
    """All GeoRSCT-X modules must import without error."""

    @pytest.mark.parametrize("module", [
        "georsct.contracts.task_contract",
        "georsct.contracts.task_gold",
        "georsct.provenance.trace",
        "georsct.provenance.store",
        "georsct.experts.base",
        "georsct.experts.hwm_reliability",
        "georsct.experts.jrc_surface_water",
        "georsct.execution.gearbox",
        "georsct.execution.harness",
        "georsct.evaluation.scoring",
    ])
    def test_import(self, module: str):
        importlib.import_module(module)
