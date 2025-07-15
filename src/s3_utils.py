import logging
import os

import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


def get_s3_client():
    """Create S3 client with proper error handling"""
    try:
        # Check if AWS profile is set
        aws_profile = os.getenv("MY_AWS_PROFILE")
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            s3_client = session.client("s3")
        else:
            # Use default aws credentials 
            s3_client = boto3.client("s3")

        return s3_client
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please configure your credentials.")
        raise
    except Exception as e:
        logger.error(f"Error creating S3 client: {str(e)}")
        raise


def upload_file_to_s3(local_path: str, bucket: str, key: str) -> bool:
    """
    Upload a file to S3 bucket with error handling

    Args:
        local_path: Path to local file
        bucket: S3 bucket name
        key: S3 key (path) for the file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if not os.path.exists(local_path):
            logger.error(f"Local file not found: {local_path}")
            return False

        s3_client = get_s3_client()

        s3_client.upload_file(local_path, bucket, key)
        logger.info(f"Successfully uploaded {local_path} to s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logger.error(f"AWS S3 error uploading file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error uploading file: {str(e)}")
        raise


def download_file_from_s3(bucket: str, key: str, local_path: str) -> bool:
    """
    Download a file from S3 bucket with error handling

    Args:
        bucket: S3 bucket name
        key: S3 key (path) for the file
        local_path: Local path to save the file

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        s3_client = get_s3_client()

        try:
            s3_client.head_object(Bucket=bucket, Key=key)
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                logger.error(f"Object not found: s3://{bucket}/{key}")
                return False
            else:
                raise

        s3_client.download_file(bucket, key, local_path)
        logger.info(f"Successfully downloaded s3://{bucket}/{key} to {local_path}")
        return True

    except ClientError as e:
        logger.error(f"AWS S3 error downloading file: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error downloading file: {str(e)}")
        raise


def check_s3_connection(bucket: str) -> bool:
    """
    Test S3 connection and bucket access

    Args:
        bucket: S3 bucket name to test

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        s3_client = get_s3_client()
        s3_client.head_bucket(Bucket=bucket)
        logger.info(f"Successfully connected to S3 bucket: {bucket}")
        return True
    except ClientError as e:
        logger.error(f"Cannot access S3 bucket {bucket}: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error checking S3 connection: {str(e)}")
        return False
