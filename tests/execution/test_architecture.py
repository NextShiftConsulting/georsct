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


def _get_full_imports(module_path: Path) -> list[str]:
    """Extract full dotted import paths from a Python file."""
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def _collect_py_files(subdir: str) -> list[Path]:
    """All .py files in a georsct subpackage (excluding __init__)."""
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
        "contracts", "provenance", "application", "evaluation", "ports",
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

    def test_application_does_not_import_task_gold(self):
        """Application layer must never see gold values."""
        for py_file in _collect_py_files("application"):
            source = py_file.read_text(encoding="utf-8")
            assert "task_gold" not in source, (
                f"{py_file.name} imports task_gold -- gold leak!"
            )

    def test_application_does_not_import_scoring(self):
        """Application layer must not import evaluation."""
        for py_file in _collect_py_files("application"):
            for imp in _get_full_imports(py_file):
                assert "evaluation" not in imp and "scoring" not in imp, (
                    f"{py_file.name} imports {imp}"
                )

    def test_application_does_not_import_store(self):
        """Application layer must not import provenance store."""
        for py_file in _collect_py_files("application"):
            source = py_file.read_text(encoding="utf-8")
            assert "provenance.store" not in source, (
                f"{py_file.name} imports provenance.store"
            )

    def test_evaluation_does_not_import_application(self):
        """Scoring is a leaf -- no upward imports."""
        for py_file in _collect_py_files("evaluation"):
            for imp in _get_full_imports(py_file):
                assert "application" not in imp, (
                    f"{py_file.name} imports {imp}"
                )

    def test_store_does_not_import_scoring(self):
        """Store persists scores, does not compute them."""
        store = _GEORSCT_ROOT / "provenance" / "store.py"
        if store.exists():
            source = store.read_text(encoding="utf-8")
            assert "scoring" not in source, (
                "store.py imports scoring -- store must not compute scores"
            )

    def test_ports_do_not_import_application(self):
        """Ports are ABCs -- they cannot depend on application layer."""
        for py_file in _collect_py_files("ports"):
            for imp in _get_full_imports(py_file):
                assert "application" not in imp, (
                    f"{py_file.name} imports {imp}"
                )

    def test_no_experts_package(self):
        """experts/ package should not exist -- ABCs in ports/, demos in tests/."""
        experts_dir = _GEORSCT_ROOT / "experts"
        py_files = list(experts_dir.rglob("*.py")) if experts_dir.exists() else []
        # Only __init__.py is tolerable during transition
        real_files = [p for p in py_files if p.name != "__init__.py"]
        assert not real_files, (
            f"experts/ still has production code: {[p.name for p in real_files]}"
        )

    def test_no_execution_package(self):
        """execution/ package should not exist -- harness+gearbox in application/."""
        exec_dir = _GEORSCT_ROOT / "execution"
        py_files = list(exec_dir.rglob("*.py")) if exec_dir.exists() else []
        real_files = [p for p in py_files if p.name != "__init__.py"]
        assert not real_files, (
            f"execution/ still has production code: {[p.name for p in real_files]}"
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
        "georsct.ports.spatial_expert",
        "georsct.ports.trace_store",
        "georsct.ports.threshold_source",
        "georsct.ports.static_thresholds",
        "georsct.application.gearbox",
        "georsct.application.harness",
        "georsct.evaluation.scoring",
    ])
    def test_import(self, module: str):
        importlib.import_module(module)
