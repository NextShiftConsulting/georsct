"""sagemaker_doe4.py - Launch DOE-4 governance workflow analysis."""
import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from datetime import datetime
import argparse

REGION = "us-east-1"
ROLE = "arn:aws:iam::188494237500:role/SageMakerExecutionRole"
BUCKET = "yrsn-datasets"
IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.8.0-cpu-py312-ubuntu22.04-sagemaker"
CODE_S3_PATH = "s3://yrsn-datasets/rsct_code/doe4_ood_detection/run_doe4.py"


def launch_job(tau_low: float, tau_high: float, dry_run: bool = False):
    session = sagemaker.Session(boto3.Session(region_name=REGION))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"doe4-governance-{timestamp}"

    processor = ScriptProcessor(
        role=ROLE, image_uri=IMAGE_URI, instance_count=1,
        instance_type="ml.m5.large", command=["python3"],
        sagemaker_session=session,
    )

    if dry_run:
        print(f"DRY RUN: {job_name}")
        return None

    processor.run(
        code=CODE_S3_PATH,
        arguments=["--tau-low", str(tau_low), "--tau-high", str(tau_high)],
        inputs=[ProcessingInput(
            source=f"s3://{BUCKET}/s017/doe3_confusion_zone/results/",
            destination="/opt/ml/processing/input/",
        )],
        outputs=[ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{BUCKET}/s017/doe4_ood_detection/results/{timestamp}/",
        )],
        job_name=job_name, wait=False,
    )
    print(f"Launched: {job_name}")
    return job_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau-low", type=float, default=0.3)
    parser.add_argument("--tau-high", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    launch_job(args.tau_low, args.tau_high, args.dry_run)
