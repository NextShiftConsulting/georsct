"""Tree-based kappa compatibility scorer for S_sup detection."""

import pickle
from typing import Optional, Tuple
import numpy as np


class KappaCompat:
    """LightGBM-based scorer for detecting superfluous (S_sup) content.

    Predicts P(S_sup | geometry features) to enable quality gating
    without neural network inference.
    """

    def __init__(self, model=None, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    @classmethod
    def load(cls, path: str) -> "KappaCompat":
        """Load trained model from pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        kappa_compat = cls()
        if isinstance(data, dict):
            kappa_compat.model = data["model"]
            kappa_compat.threshold = data.get("threshold", 0.5)
        else:
            kappa_compat.model = data

        return kappa_compat

    def save(self, path: str) -> None:
        """Save model to pickle."""
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "threshold": self.threshold
            }, f)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Predict P(S_sup) for geometry features.

        Args:
            features: (263,) single sample or (N, 263) batch

        Returns:
            p_s_sup: Probability of superfluous class
        """
        if self.model is None:
            raise ValueError("No model loaded")

        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)

        # LightGBM returns probabilities for each class
        proba = self.model.predict_proba(features)

        # S_sup is class index 1 (R=0, S_sup=1, N=2)
        p_s_sup = proba[:, 1]

        if single:
            return p_s_sup[0]
        return p_s_sup

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Binary prediction: is this S_sup?

        Args:
            features: (263,) or (N, 263) geometry features

        Returns:
            is_s_sup: Boolean prediction
        """
        p_s_sup = self.predict_proba(features)
        return p_s_sup >= self.threshold

    def gate(self, features: np.ndarray) -> Tuple[str, float]:
        """Apply quality gate.

        Args:
            features: (263,) geometry features for single sample

        Returns:
            action: "PROCEED" or "RE_ENCODE"
            confidence: 1 - P(S_sup) for PROCEED, P(S_sup) for RE_ENCODE
        """
        p_s_sup = self.predict_proba(features)

        if p_s_sup < self.threshold:
            return "PROCEED", 1 - p_s_sup
        else:
            return "RE_ENCODE", p_s_sup

    @staticmethod
    def train(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        **lgb_params
    ) -> "KappaCompat":
        """Train a new kappa compatibility scorer model.

        Args:
            X_train: (N, 263) training features
            y_train: (N,) labels (0=R, 1=S_sup, 2=N)
            X_val: Validation features
            y_val: Validation labels
            **lgb_params: LightGBM parameters

        Returns:
            Trained KappaCompat
        """
        import lightgbm as lgb

        default_params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "verbose": -1
        }
        default_params.update(lgb_params)

        train_data = lgb.Dataset(X_train, label=y_train)

        callbacks = []
        valid_sets = [train_data]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            callbacks.append(lgb.early_stopping(50))

        model = lgb.train(
            default_params,
            train_data,
            num_boost_round=500,
            valid_sets=valid_sets,
            callbacks=callbacks
        )

        # Wrap in sklearn-compatible interface
        class LGBWrapper:
            def __init__(self, booster):
                self.booster = booster

            def predict_proba(self, X):
                return self.booster.predict(X)

        kappa_compat = KappaCompat(model=LGBWrapper(model))
        return kappa_compat
