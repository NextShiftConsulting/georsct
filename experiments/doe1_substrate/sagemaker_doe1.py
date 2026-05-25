"""
sagemaker_doe1.py - Launch DOE-1 substrate comparison on SageMaker.

Deploy code first: ./deploy.sh
"""
import boto3
import sagemaker
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from datetime import datetime
import argparse

REGION = "us-east-1"
ROLE = "arn:aws:iam::188494237500:role/SageMakerExecutionRole"
BUCKET = "yrsn-datasets"

IMAGE_URI = f"763104351884.dkr.ecr.{REGION}.amazonaws.com/pytorch-training:2.8.0-cpu-py312-ubuntu22.04-sagemaker"
CODE_S3_PATH = "s3://yrsn-datasets/rsct_code/doe1_substrate/run_doe1.py"


def launch_job(models: list, dataset: str, dry_run: bool = False):
    """Launch SageMaker processing job."""
    session = sagemaker.Session(boto3.Session(region_name=REGION))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    job_name = f"doe1-substrate-{timestamp}"

    processor = ScriptProcessor(
        role=ROLE,
        image_uri=IMAGE_URI,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        command=["python3"],
        sagemaker_session=session,
        base_job_name="doe1-substrate",
    )

    arguments = ["--models"] + models + ["--dataset", dataset]

    print(f"=== DOE-1 SUBSTRATE ===")
    print(f"Job: {job_name}")
    print(f"Models: {models}")
    print(f"Dataset: {dataset}")

    if dry_run:
        print("DRY RUN - not launching")
        return None

    processor.run(
        code=CODE_S3_PATH,
        arguments=arguments,
        inputs=[
            ProcessingInput(
                source=f"s3://{BUCKET}/s017/doe1_substrate/embeddings/",
                destination="/opt/ml/processing/input/embeddings/",
                s3_data_type="S3Prefix",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output",
                destination=f"s3://{BUCKET}/s017/doe1_substrate/results/{timestamp}/",
            ),
        ],
        job_name=job_name,
        wait=False,
    )

    print(f"\n=== JOB LAUNCHED ===")
    print(f"Monitor: https://console.aws.amazon.com/cloudwatch/home?region={REGION}#logsV2:log-groups/log-group/$252Faws$252Fsagemaker$252FProcessingJobs/log-events/{job_name}")
    return job_name


def main():
    parser = argparse.ArgumentParser(description="DOE-1 Launcher")
    parser.add_argument("--models", nargs="+", default=["s016c", "minilm", "nemotron"])
    parser.add_argument("--dataset", default="MIRACL")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    launch_job(args.models, args.dataset, args.dry_run)


if __name__ == "__main__":
    main()
