#!/usr/bin/env python3
"""
train_surrogate.py -- SageMaker training job: fit LSTM + XGBoost surrogate
flood forecast models for each FloodRSCT scenario.

Activated only if MaxFloodCast (Lee et al.) is unavailable by May 30.

Architecture:
  - LSTM: sequence model on (MRMS cumulative rainfall, gauge stage, HRRR QPF)
           → 24-hour ahead peak stage prediction per ZCTA
  - XGBoost: tabular fallback on static features + event-level aggregates
           → flood probability score (0-1) per (ZCTA, event)

Both models output `pred_risk_score` that feeds into `cert_r` / `cert_action`
via the RSCT certification pipeline.

Input (from S3 processed outputs):
  - processed/{scenario}/{scenario}_event_features.parquet

Output:
  - model/surrogate/{scenario}/lstm/  (PyTorch state dict + scaler .pkl)
  - model/surrogate/{scenario}/xgboost/  (model.json)
  - model/surrogate/{scenario}/eval_metrics.json

Usage:
    python train_surrogate.py --scenario houston
    python train_surrogate.py --scenario new_orleans
    python train_surrogate.py --scenario nyc
    python train_surrogate.py --scenario riverside_coachella
    python train_surrogate.py --scenario southwest_florida

All runs: ml.g5.2xlarge (1x A10G GPU, 32 GB RAM).
"""

import argparse
import json
import logging
import sys
import tempfile
from pathlib import Path

import boto3
from swarm_auth import get_aws_credentials
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
log = logging.getLogger(__name__)

SRC_BUCKET = "swarm-floodrsct-data"
MODEL_PREFIX = "model/surrogate"

SCENARIOS = [
    "houston", "new_orleans", "nyc", "riverside_coachella", "southwest_florida"
]

# Static + event-level features used by both models
STATIC_FEATURES = [
    "pct_flood_zone_ae", "pct_flood_zone_x",
    "twi_mean", "slope_mean",
    "svi_overall",
    "median_household_income", "pct_below_poverty",
    "pct_renter_occupied", "housing_units_per_sq_mi",
    "nfip_policy_count", "nfip_claim_count",
]

EVENT_FEATURES = [
    "peak_stage_ft", "peak_flow_cfs",
    "mrms_total_mm", "mrms_max_hourly_mm",
    "obs_gauge_count", "obs_gauge_distance_km",
    "obs_mrms_coverage_pct",
    "storm_distance_km",
]

TARGET_COL = "obs_nfip_event_claims"  # proxy label: event-year DR claim count


def load_event_features(s3, scenario: str) -> pd.DataFrame:
    key = f"processed/{scenario}/{scenario}_event_features.parquet"
    local = f"/tmp/{scenario}_event_features.parquet"
    log.info("Downloading s3://%s/%s", SRC_BUCKET, key)
    s3.download_file(SRC_BUCKET, key, local)
    return pd.read_parquet(local)


def prepare_arrays(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (X, y, feature_names) for tabular training."""
    all_features = [
        f for f in STATIC_FEATURES + EVENT_FEATURES if f in df.columns
    ]
    missing = [f for f in STATIC_FEATURES + EVENT_FEATURES if f not in df.columns]
    if missing:
        log.warning("Missing features (will be zero-filled): %s", missing)
        for f in missing:
            df[f] = 0.0

    X = df[all_features].fillna(0).values.astype(np.float32)
    y = df[TARGET_COL].fillna(0).clip(lower=0).values.astype(np.float32)
    return X, y, all_features


def train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: list[str]
) -> dict:
    """Fit XGBoost regressor. Returns eval metrics dict."""
    try:
        import xgboost as xgb
    except ImportError:
        log.error("xgboost not installed; skipping XGBoost path")
        return {"status": "skipped", "reason": "xgboost_not_installed"}

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if len(X_train) < 10:
        log.warning("Too few samples (%d) for XGBoost; skipping", len(X_train))
        return {"status": "skipped", "reason": "insufficient_samples"}

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "n_jobs": -1,
        "seed": 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=50,
    )

    y_pred = model.predict(dval)
    rmse = float(np.sqrt(np.mean((y_pred - y_val) ** 2)))
    log.info("XGBoost val RMSE=%.4f (best_iteration=%d)", rmse, model.best_iteration)

    return {
        "model": model,
        "val_rmse": rmse,
        "best_iteration": model.best_iteration,
        "n_features": len(feature_names),
    }


def build_lstm_sequences(
    df: pd.DataFrame, feature_cols: list[str], seq_len: int = 24
) -> tuple[np.ndarray, np.ndarray]:
    """Build (N, seq_len, F) sequences grouped by zcta_id + event.

    We use MRMS hourly rainfall as the time dimension where available.
    If hourly data is not embedded in the feature table (current state: it's
    aggregated), we fall back to repeating the scalar features across seq_len
    time steps with a linear ramp on mrms_total_mm. This is a degraded but
    valid surrogate signal until the MRMS per-hour columns are added.
    """
    records_X = []
    records_y = []

    grp_cols = [c for c in ["zcta_id", "event"] if c in df.columns]
    if not grp_cols:
        log.warning("No zcta_id/event columns; building single-group sequences")
        grp_cols = []

    for _, grp in (df.groupby(grp_cols) if grp_cols else [(None, df)]):
        for _, row in grp.iterrows():
            feats = np.array(
                [row.get(f, 0.0) if not pd.isna(row.get(f, np.nan)) else 0.0
                 for f in feature_cols],
                dtype=np.float32,
            )
            # Temporal ramp: ramp mrms linearly over seq_len steps
            seq = np.tile(feats, (seq_len, 1))
            if "mrms_total_mm" in feature_cols:
                idx = feature_cols.index("mrms_total_mm")
                total = feats[idx]
                seq[:, idx] = np.linspace(0, total, seq_len)

            records_X.append(seq)
            target = row.get(TARGET_COL, 0.0)
            records_y.append(float(target) if not pd.isna(target) else 0.0)

    if not records_X:
        return np.zeros((0, seq_len, len(feature_cols)), dtype=np.float32), np.zeros(0)

    return (
        np.stack(records_X, axis=0),
        np.array(records_y, dtype=np.float32),
    )


def train_lstm(
    X_seq: np.ndarray, y: np.ndarray, feature_names: list[str]
) -> dict:
    """Fit a simple LSTM regressor. Returns eval metrics dict."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        log.error("PyTorch not installed; skipping LSTM path")
        return {"status": "skipped", "reason": "torch_not_installed"}

    if len(X_seq) < 10:
        log.warning("Too few samples (%d) for LSTM; skipping", len(X_seq))
        return {"status": "skipped", "reason": "insufficient_samples"}

    n_samples, seq_len, n_features = X_seq.shape
    split = max(1, int(0.8 * n_samples))

    X_tr = torch.tensor(X_seq[:split], dtype=torch.float32)
    y_tr = torch.tensor(y[:split], dtype=torch.float32).unsqueeze(1)
    X_val = torch.tensor(X_seq[split:], dtype=torch.float32)
    y_val = torch.tensor(y[split:], dtype=torch.float32).unsqueeze(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("LSTM training on device=%s", device)

    class FloodLSTM(nn.Module):
        def __init__(self, input_size: int, hidden: int = 64, layers: int = 2) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, layers, batch_first=True, dropout=0.2)
            self.head = nn.Sequential(
                nn.Linear(hidden, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :])

    model = FloodLSTM(input_size=n_features).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=32, shuffle=True)
    best_val_loss = float("inf")
    patience, patience_ctr = 20, 0

    for epoch in range(200):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val.to(device)).cpu()
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                log.info("Early stopping at epoch %d", epoch)
                break

        if epoch % 25 == 0:
            log.info("Epoch %d val_rmse=%.4f", epoch, float(val_loss) ** 0.5)
            sys.stdout.flush()

    model.load_state_dict(best_state)
    val_rmse = float(best_val_loss) ** 0.5
    log.info("LSTM best val RMSE=%.4f", val_rmse)

    return {
        "model": model,
        "state_dict": best_state,
        "val_rmse": val_rmse,
        "n_features": n_features,
        "seq_len": seq_len,
    }


def upload_model_artifacts(
    s3, scenario: str, xgb_result: dict, lstm_result: dict
) -> None:
    import pickle

    metrics: dict = {"scenario": scenario, "xgboost": {}, "lstm": {}}
    prefix = f"{MODEL_PREFIX}/{scenario}"

    # XGBoost
    xgb_model = xgb_result.pop("model", None)
    metrics["xgboost"] = {k: v for k, v in xgb_result.items() if k != "model"}
    if xgb_model is not None:
        model_path = f"/tmp/{scenario}_xgb.json"
        xgb_model.save_model(model_path)
        s3.upload_file(model_path, SRC_BUCKET, f"{prefix}/xgboost/model.json")
        log.info("Uploaded XGBoost model to s3://%s/%s/xgboost/model.json", SRC_BUCKET, prefix)

    # LSTM
    state_dict = lstm_result.pop("state_dict", None)
    lstm_model = lstm_result.pop("model", None)
    metrics["lstm"] = {k: v for k, v in lstm_result.items() if k != "model"}
    if state_dict is not None:
        import torch
        torch_path = f"/tmp/{scenario}_lstm.pt"
        torch.save(state_dict, torch_path)
        s3.upload_file(torch_path, SRC_BUCKET, f"{prefix}/lstm/model.pt")
        log.info("Uploaded LSTM state dict to s3://%s/%s/lstm/model.pt", SRC_BUCKET, prefix)

    # Eval metrics
    metrics_path = f"/tmp/{scenario}_eval_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    s3.upload_file(metrics_path, SRC_BUCKET, f"{prefix}/eval_metrics.json")
    log.info("Uploaded eval metrics to s3://%s/%s/eval_metrics.json", SRC_BUCKET, prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True, choices=SCENARIOS)
    parser.add_argument(
        "--skip-lstm", action="store_true",
        help="Skip LSTM (run only XGBoost — useful if GPU unavailable)"
    )
    args = parser.parse_args()

    scenario = args.scenario
    _aws = get_aws_credentials()
    s3 = boto3.client("s3", region_name="us-east-1", **_aws)

    df = load_event_features(s3, scenario)
    log.info("Loaded %d rows for scenario=%s", len(df), scenario)

    X, y, feature_names = prepare_arrays(df)
    log.info("Feature matrix: %s, target non-zero: %d", X.shape, int((y > 0).sum()))

    # XGBoost
    log.info("--- XGBoost ---")
    xgb_result = train_xgboost(X, y, feature_names)

    # LSTM
    if args.skip_lstm:
        lstm_result = {"status": "skipped", "reason": "skip_lstm_flag"}
    else:
        log.info("--- LSTM ---")
        X_seq, y_seq = build_lstm_sequences(df, feature_names)
        log.info("Sequence tensor shape: %s", X_seq.shape)
        lstm_result = train_lstm(X_seq, y_seq, feature_names)

    upload_model_artifacts(s3, scenario, xgb_result, lstm_result)
    log.info("train_surrogate complete for scenario=%s", scenario)


if __name__ == "__main__":
    main()
