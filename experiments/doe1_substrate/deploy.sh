#!/bin/bash
# Deploy DOE-1 job files to S3
# Run this once before launching jobs, or after any code changes

S3_PATH="s3://yrsn-datasets/rsct_code/doe1_substrate"

echo "Deploying job_files/ to $S3_PATH"
aws s3 sync job_files/ "$S3_PATH/" --exclude "README.md"

echo ""
echo "Deployed files:"
aws s3 ls "$S3_PATH/"
