"""sagemaker_doe6.py - Launch DOE-6 geospatial transfer."""
import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from datetime import datetime
import argparse

REGION = "us-east-1"
ROLE = "arn:aws:iam::188494237500:role/SageMakerExecutionRole"
BUCKET = "yrsn-datasets"
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.8.0-cpu-py312-ubuntu22.04-sagemaker"
CODE_S3_PATH = "s3://yrsn-datasets/rsct_code/doe6_geospatial/run_doe6.py"


def launch_job(target: str, dry_run: bool = False):
    session = sagemaker.Session(boto3.Session(region_name=REGION))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"doe6-geospatial-{timestamp}"

    processor = ScriptProcessor(
        role=ROLE, image_uri=IMAGE_URI, instance_count=1,
        instance_type="ml.m5.xlarge", command=["python3"],
        sagemaker_session=session,
    )

    if dry_run:
        print(f"DRY RUN: {job_name}")
        return None

    processor.run(
        code=CODE_S3_PATH,
        arguments=["--target", target],
        inputs=[ProcessingInput(
            source=f"s3://{BUCKET}/s017/doe6_geospatial/pdfm_conus27.csv",
            destination="/opt/ml/processing/input/pdfm_conus27.csv",
        )],
        outputs=[ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{BUCKET}/s017/doe6_geospatial/results/{timestamp}/",
        )],
        job_name=job_name, wait=False,
    )
    print(f"Launched: {job_name}")
    return job_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="median_income")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    launch_job(args.target, args.dry_run)
