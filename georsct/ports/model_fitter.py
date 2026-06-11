"""Port: supervised model fitting and embedding extraction.

Abstracts the fit-predict-embed pipeline so domain and application code
never import from adapter job scripts.

ADR-014: implementations must use frozen hyperparameters and frozen fold
assignments.  This port describes the evaluation interface, not training.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FitPredictResult:
    """Result of a single fit-predict pass on frozen folds."""

    predictions: np.ndarray  # (n_obs,) out-of-fold predictions
    forward_score: float     # R2, AUC, or accuracy
    task_type: str           # "regression" | "binary_classification" | ...

    class Config:
        arbitrary_types_allowed = True


@dataclass(frozen=True)
class EmbedResult:
    """Region-level embedding from feature aggregation."""

    embeddings: np.ndarray       # (n_regions, n_features) z-scored
    region_order: tuple[str, ...]

    class Config:
        arbitrary_types_allowed = True


class ModelFitter(ABC):
    """Port for supervised model fitting and embedding extraction.

    Implementations must:
    - Use frozen folds (no data-dependent fold assignment at runtime)
    - Use pre-committed hyperparameters (ADR-014)
    - Return out-of-fold predictions only
    """

    @abstractmethod
    def fit_predict(
        self,
        features: np.ndarray,
        target: np.ndarray,
        fold_ids: np.ndarray,
        task_type: str,
    ) -> FitPredictResult:
        """Train model on frozen folds, return out-of-fold predictions.

        Args:
            features: (n_obs, n_features) feature matrix.
            target: (n_obs,) target vector.
            fold_ids: (n_obs,) integer fold assignment.
            task_type: "regression", "binary_classification",
                or "multiclass_classification".

        Returns:
            FitPredictResult with predictions and forward score.
        """

    @abstractmethod
    def aggregate_embeddings(
        self,
        features: np.ndarray,
        region_ids: np.ndarray,
        region_order: tuple[str, ...],
    ) -> EmbedResult:
        """Compute z-scored mean feature vector per region.

        Args:
            features: (n_obs, n_features) feature matrix.
            region_ids: (n_obs,) string region labels.
            region_order: Canonical region ordering for the output.

        Returns:
            EmbedResult with (n_regions, n_features) z-scored embeddings.
        """
