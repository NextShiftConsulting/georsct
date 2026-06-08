"""Upload processed FloodRSCT parquet files from S3 directly to HuggingFace Hub.

Streams S3 objects into HF via in-memory buffers -- nothing touches local disk
except the small local evidence CSVs and configs checked into the repo.

Usage:
    source ~/github/swarm-it-auth/keys/.env
    python upload.py                          # dry-run (default)
    python upload.py --execute                # actually upload
    python upload.py --scenario houston       # single scenario
"""

import argparse
import io
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi
from swarm_auth import get_aws_credentials

BUCKET = "swarm-floodrsct-data"
REPO_ID = "rudymartin/floodrsct"

SCENARIOS = [
    "houston",
    "new_orleans",
    "nyc",
    "riverside_coachella",
    "southwest_florida",
]

# S3 key -> HF repo path
S3_PROCESSED = {
    "houston": "processed/houston/houston_event_features.parquet",
    "new_orleans": "processed/new_orleans/no_event_features.parquet",
    "nyc": "processed/nyc/nyc_event_features.parquet",
    "riverside_coachella": "processed/riverside_coachella/rc_event_features.parquet",
    "southwest_florida": "processed/southwest_florida/swfl_event_features.parquet",
}

# Local files to include (small, checked into repo)
EVIDENCE_DIR = Path(__file__).parent.parent / "evidence"
EVIDENCE_FILES = [
    "no_pump_stations_ida2021.csv",
    "nyc_subway_flooding_ida2021.csv",
]
CONFIGS_DIR = Path(__file__).parent.parent / "configs"


def get_s3_client():
    import boto3

    aws = get_aws_credentials()
    return boto3.client("s3", **aws)


def s3_to_buffer(s3, s3_key: str) -> io.BytesIO | None:
    """Stream an S3 object into an in-memory buffer."""
    try:
        response = s3.get_object(Bucket=BUCKET, Key=s3_key)
        buf = io.BytesIO(response["Body"].read())
        size_mb = buf.getbuffer().nbytes / (1024 * 1024)
        print(f"  Streamed s3://{BUCKET}/{s3_key} ({size_mb:.1f} MB)")
        return buf
    except Exception as e:
        print(f"  SKIP s3://{BUCKET}/{s3_key}: {e}")
        return None


def build_operations(scenarios: list[str], s3) -> list[CommitOperationAdd]:
    """Build HF commit operations: S3 objects streamed to memory + local files read."""
    ops = []

    # Parquets from S3 (streamed, no local disk)
    for scenario in scenarios:
        s3_key = S3_PROCESSED[scenario]
        hf_path = f"data/{scenario}/{scenario}_event_features.parquet"
        buf = s3_to_buffer(s3, s3_key)
        if buf:
            ops.append(CommitOperationAdd(path_in_repo=hf_path, path_or_fileobj=buf))

    # Evidence CSVs (local, tiny)
    for fname in EVIDENCE_FILES:
        src = EVIDENCE_DIR / fname
        if src.exists():
            ops.append(CommitOperationAdd(
                path_in_repo=f"data/evidence/{fname}",
                path_or_fileobj=src.read_bytes(),
            ))
            print(f"  Added local evidence/{fname}")

    # Scenario configs (local, tiny)
    for scenario in scenarios:
        src = CONFIGS_DIR / f"{scenario}.yaml"
        if src.exists():
            ops.append(CommitOperationAdd(
                path_in_repo=f"configs/{scenario}.yaml",
                path_or_fileobj=src.read_bytes(),
            ))
            print(f"  Added local configs/{scenario}.yaml")

    # Dataset card
    readme = Path(__file__).parent / "README.md"
    if readme.exists():
        ops.append(CommitOperationAdd(
            path_in_repo="README.md",
            path_or_fileobj=readme.read_bytes(),
        ))

    return ops


def main():
    parser = argparse.ArgumentParser(description="Upload FloodRSCT S3 -> HuggingFace Hub")
    parser.add_argument("--execute", action="store_true", help="Actually upload (default is dry-run)")
    parser.add_argument("--scenario", choices=SCENARIOS, help="Upload single scenario")
    args = parser.parse_args()

    scenarios = [args.scenario] if args.scenario else SCENARIOS

    print(f"FloodRSCT HuggingFace Upload (S3 -> HF direct)")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}\n")

    s3 = get_s3_client()
    ops = build_operations(scenarios, s3)

    parquet_ops = [o for o in ops if o.path_in_repo.endswith(".parquet")]
    other_ops = [o for o in ops if not o.path_in_repo.endswith(".parquet")]

    print(f"\nCommit plan:")
    print(f"  Parquets from S3:  {len(parquet_ops)}")
    print(f"  Local files:       {len(other_ops)}")
    for op in ops:
        print(f"    {op.path_in_repo}")

    if not args.execute:
        print("\n--- DRY RUN --- Re-run with --execute to upload.")
        return

    api = HfApi()
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)
    api.create_commit(
        repo_id=REPO_ID,
        repo_type="dataset",
        operations=ops,
        commit_message="Upload FloodRSCT processed event features",
    )
    print(f"\nDone: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
