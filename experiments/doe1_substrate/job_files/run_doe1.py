"""
DOE-1: Substrate Independence
Test geometry-based S_sup detection across embedding substrates.

Runs on SageMaker. Deploy code first: ./deploy.sh
"""
import argparse
import json
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# SageMaker paths
INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")


def load_embeddings(model_name: str):
    """Load pre-computed embeddings from S3-mounted input."""
    emb_path = INPUT_DIR / "embeddings" / model_name
    # TODO: Implement based on embedding format
    raise NotImplementedError(f"Load embeddings for {model_name}")


def extract_geometry_features(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Extract 263-dim geometry features from embedding pair."""
    diff = u - v
    prod = u * v

    u_norm = np.linalg.norm(u)
    v_norm = np.linalg.norm(v)
    diff_norm = np.linalg.norm(diff)

    cosine = np.dot(u, v) / (u_norm * v_norm + 1e-8)
    u_proj = np.dot(u, diff) / (diff_norm + 1e-8)
    v_proj = np.dot(v, diff) / (diff_norm + 1e-8)
    asymmetry = u_proj - v_proj

    scalars = np.array([cosine, u_norm, v_norm, diff_norm, u_proj, v_proj, asymmetry])
    return np.concatenate([u, v, np.abs(diff), prod, scalars])


def compute_cosine_auc(u_all: np.ndarray, v_all: np.ndarray, labels: np.ndarray):
    """Compute AUC-ROC using cosine similarity as score."""
    from sklearn.metrics import roc_auc_score

    cosines = np.array([
        np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-8)
        for u, v in zip(u_all, v_all)
    ])
    # For S_sup detection, lower cosine might indicate S_sup
    # Binary: is this S_sup (class 1)?
    binary_labels = (labels == 1).astype(int)
    return roc_auc_score(binary_labels, -cosines)  # Negative because lower cosine = more S_sup


def run_substrate_comparison(args):
    """Main experiment: compare substrates."""
    results = {
        "experiment": "DOE-1",
        "timestamp": datetime.now().isoformat(),
        "substrates": {}
    }

    for model in args.models:
        print(f"\n=== {model} ===")
        # TODO: Load embeddings, run tree, compute metrics
        results["substrates"][model] = {
            "auc_tree": None,
            "auc_cosine": None,
            "delta": None,
            "status": "NOT_IMPLEMENTED"
        }

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "doe1_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="DOE-1: Substrate Independence")
    parser.add_argument("--models", nargs="+", default=["s016c", "minilm", "nemotron"],
                       help="Embedding models to compare")
    parser.add_argument("--dataset", default="MIRACL", help="Evaluation dataset")
    args = parser.parse_args()

    print(f"DOE-1: Substrate Independence")
    print(f"Models: {args.models}")
    print(f"Dataset: {args.dataset}")

    run_substrate_comparison(args)


if __name__ == "__main__":
    main()
