#!/bin/bash
S3_PATH="s3://yrsn-datasets/rsct_code/doe5_agent_benchmark"
aws s3 sync job_files/ "$S3_PATH/" --exclude "README.md"
aws s3 ls "$S3_PATH/"
