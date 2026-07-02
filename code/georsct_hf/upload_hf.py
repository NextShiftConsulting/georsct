#!/usr/bin/env python3
"""Upload all wsp/georsct-hf/ to HuggingFace dataset repo.

Usage:
    python wsp/georsct-hf/upload_to_hf.py              # upload all files
    python wsp/georsct-hf/upload_to_hf.py --dry-run    # show what would be uploaded
"""

import argparse
import os
from pathlib import Path

REPO_ID = "rudymartin/georsct"
HF_DIR = Path(__file__).parent

# Files to skip (not deployed to HF)
SKIP = {"DEPLOY.md", "upload_to_hf.py", ".gitkeep"}


def collect_files():
    """Walk wsp/georsct-hf/ and build (local_path, hf_path) pairs."""
    pairs = []
    for root, _dirs, files in os.walk(HF_DIR):
        for name in sorted(files):
            if name in SKIP:
                continue
            local = Path(root) / name
            hf_path = str(local.relative_to(HF_DIR)).replace("\\", "/")
            pairs.append((str(local), hf_path))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Upload wsp/georsct-hf/ to HuggingFace")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without uploading")
    args = parser.parse_args()

    pairs = collect_files()

    if args.dry_run:
        print(f"Would upload {len(pairs)} files to {REPO_ID}:\n")
        for local, hf_path in pairs:
            print(f"  {hf_path}")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Uploading {len(pairs)} files to {REPO_ID}...\n")

    for local, hf_path in pairs:
        print(f"  {hf_path} ... ", end="", flush=True)
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=hf_path,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"sync: {hf_path} from GitHub wsp/georsct-hf/",
        )
        print("done")

    print(f"\nAll {len(pairs)} files uploaded.")


if __name__ == "__main__":
    main()
