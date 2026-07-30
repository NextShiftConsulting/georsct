"""Adapter: scikit-learn ModelFitter.

Implements the ``ModelFitter`` port (georsct.ports.model_fitter) using
scikit-learn estimators with frozen hyperparameters and frozen fold
assignments (ADR-014): no data-dependent fold selection and no runtime
hyperparameter search occur here -- this is the evaluation interface, not
training.

``fit_predict`` runs leave-one-fold-out cross-validation over the caller's
frozen ``fold_ids`` and returns out-of-fold predictions plus a forward score
(R2 for regression, ROC-AUC for binary classification).  Feature NaNs are
imputed with the training fold's column medians; target NaNs are excluded
from both fitting and scoring but still receive an OOF prediction so the
returned vector stays row-aligned with the shared feature matrix.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, roc_auc_score

from georsct.ports.model_fitter import EmbedResult, FitPredictResult, ModelFitter

# Frozen hyperparameters (ADR-014). Pre-committed, not tuned at runtime.
_N_ESTIMATORS = 200
_RANDOM_STATE = 0
_N_JOBS = -1  # multi-threaded per house rule; RF is embarrassingly parallel


class SklearnModelFitter(ModelFitter):
    """ModelFitter backed by scikit-learn random forests.

    Args:
        n_estimators: Trees per forest (frozen).
        random_state: Seed for reproducibility (frozen).
        n_jobs: Parallelism for the forest (default all cores).
    """

    def __init__(
        self,
        n_estimators: int = _N_ESTIMATORS,
        random_state: int = _RANDOM_STATE,
        n_jobs: int = _N_JOBS,
    ):
        self._n_estimators = int(n_estimators)
        self._random_state = int(random_state)
        self._n_jobs = int(n_jobs)

    # -- port surface -----------------------------------------------------

    def fit_predict(
        self,
        features: np.ndarray,
        target: np.ndarray,
        fold_ids: np.ndarray,
        task_type: str,
    ) -> FitPredictResult:
        """Leave-one-fold-out OOF predictions on frozen folds."""
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float)
        fold_ids = np.asarray(fold_ids)

        n_obs = features.shape[0]
        if not (len(target) == len(fold_ids) == n_obs):
            raise ValueError(
                f"length mismatch: features={n_obs}, target={len(target)}, "
                f"fold_ids={len(fold_ids)}"
            )

        predictions = np.full(n_obs, np.nan, dtype=float)
        fold_scores: list[float] = []

        unique_folds = [f for f in np.unique(fold_ids)]
        for fold in unique_folds:
            test_mask = fold_ids == fold
            train_mask = ~test_mask

            # Train only on rows whose target is finite (ADR-020 D8: never
            # fit on NaN targets); predict every held-out row so the OOF
            # vector stays aligned with the shared feature matrix.
            train_finite = train_mask & np.isfinite(target)
            if int(train_finite.sum()) < 2 or int(test_mask.sum()) == 0:
                continue

            imputer = SimpleImputer(strategy="median")
            x_train = imputer.fit_transform(features[train_finite])
            x_test = imputer.transform(features[test_mask])
            y_train = target[train_finite]

            model = self._make_model(task_type)

            if task_type == "binary_classification":
                y_train_int = (y_train > 0.5).astype(int)
                if len(np.unique(y_train_int)) < 2:
                    # Degenerate single-class fold: predict the constant rate.
                    predictions[test_mask] = float(y_train_int.mean())
                    continue
                model.fit(x_train, y_train_int)
                proba = model.predict_proba(x_test)[:, 1]
                predictions[test_mask] = proba
            else:
                model.fit(x_train, y_train)
                predictions[test_mask] = model.predict(x_test)

            fold_scores.append(
                self._score(target[test_mask], predictions[test_mask], task_type)
            )

        forward_score = self._score(target, predictions, task_type)

        return FitPredictResult(
            predictions=predictions,
            forward_score=forward_score,
            task_type=task_type,
            fold_scores=tuple(s for s in fold_scores if np.isfinite(s)),
        )

    def aggregate_embeddings(
        self,
        features: np.ndarray,
        region_ids: np.ndarray,
        region_order: tuple[str, ...],
    ) -> EmbedResult:
        """Z-scored mean feature vector per region, aligned to region_order."""
        features = np.asarray(features, dtype=float)
        region_ids = np.asarray(region_ids).astype(str)

        n_regions = len(region_order)
        n_features = features.shape[1]
        sums = np.zeros((n_regions, n_features), dtype=float)
        counts = np.zeros(n_regions, dtype=float)
        idx_map = {r: i for i, r in enumerate(region_order)}

        for k in range(features.shape[0]):
            i = idx_map.get(str(region_ids[k]))
            if i is None:
                continue
            row = features[k]
            finite = np.isfinite(row)
            sums[i, finite] += row[finite]
            counts[i] += 1.0

        means = np.full((n_regions, n_features), np.nan, dtype=float)
        nonzero = counts > 0
        means[nonzero] = sums[nonzero] / counts[nonzero, None]

        # Z-score each feature across regions (std guarded; NaN-tolerant).
        col_mean = np.nanmean(means, axis=0)
        col_std = np.nanstd(means, axis=0)
        col_std = np.where(col_std > 0, col_std, 1.0)
        z = (means - col_mean) / col_std
        z = np.nan_to_num(z, nan=0.0)

        return EmbedResult(embeddings=z, region_order=tuple(region_order))

    # -- internal ---------------------------------------------------------

    def _make_model(self, task_type: str):
        if task_type == "binary_classification":
            return RandomForestClassifier(
                n_estimators=self._n_estimators,
                random_state=self._random_state,
                n_jobs=self._n_jobs,
            )
        return RandomForestRegressor(
            n_estimators=self._n_estimators,
            random_state=self._random_state,
            n_jobs=self._n_jobs,
        )

    @staticmethod
    def _score(target: np.ndarray, predictions: np.ndarray, task_type: str) -> float:
        """Held-out metric over rows with finite target and prediction."""
        mask = np.isfinite(target) & np.isfinite(predictions)
        if int(mask.sum()) < 2:
            return float("nan")
        y = target[mask]
        p = predictions[mask]
        if task_type == "binary_classification":
            y_int = (y > 0.5).astype(int)
            if len(np.unique(y_int)) < 2:
                return float("nan")
            return float(roc_auc_score(y_int, p))
        return float(r2_score(y, p))
