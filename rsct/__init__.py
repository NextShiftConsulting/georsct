"""RSCT Context: R/S/N Decomposition for Agentic Context Quality."""

from rsct.experiment_cert import (
    CoherenceResult,
    ExperimentCertificate,
    certify_experiment_cell,
    compute_coherence,
    compute_sigma,
    compute_tau,
    derive_simplex,
)
from rsct.geometry import GeometryExtractor
from rsct.kappa_compat import KappaCompat

__all__ = [
    "CoherenceResult",
    "GeometryExtractor",
    "KappaCompat",
    "ExperimentCertificate",
    "certify_experiment_cell",
    "compute_coherence",
    "compute_sigma",
    "compute_tau",
    "derive_simplex",
]
__version__ = "0.3.0"
