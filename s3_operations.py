import boto3
from botocore.exceptions import ClientError
import os

# ── Config ────────────────────────────────────────────────────────────────────
REGION = "us-east-1"
BUCKET_NAME = "loan-dataset-models"

# Folder structure inside the bucket
S3_FOLDERS = {
    "dataset": "Dataset/",
    "models":  "models/",
}
# ─────────────────────────────────────────────────────────────────────────────


def _get_client():
    return boto3.client("s3", region_name=REGION)


# ── Upload ────────────────────────────────────────────────────────────────────

def upload_file(local_path: str, s3_folder_key: str, s3_filename: str = None) -> bool:
    """
    Upload a local file to S3.

    Parameters
    ----------
    local_path    : path to the local file
    s3_folder_key : one of the keys in S3_FOLDERS ('dataset' | 'models')
    s3_filename   : name to use in S3; defaults to the basename of local_path
    """
    folder = S3_FOLDERS.get(s3_folder_key, s3_folder_key)
    filename = s3_filename or os.path.basename(local_path)
    s3_key = folder + filename

    client = _get_client()
    try:
        client.upload_file(local_path, BUCKET_NAME, s3_key)
        print(f"Uploaded  '{local_path}'  →  s3://{BUCKET_NAME}/{s3_key}")
        return True
    except ClientError as e:
        print(f"Upload failed: {e}")
        return False


# ── Download ──────────────────────────────────────────────────────────────────

def download_file(s3_folder_key: str, s3_filename: str, local_dir: str = None) -> bool:
    """
    Download a file from S3 to a local directory.

    Parameters
    ----------
    s3_folder_key : one of the keys in S3_FOLDERS ('dataset' | 'models')
    s3_filename   : name of the file inside the S3 folder
    local_dir     : local directory to save the file; defaults to the matching
                    local folder (Dataset/ or models/)
    """
    folder = S3_FOLDERS.get(s3_folder_key, s3_folder_key)
    s3_key = folder + s3_filename

    # Mirror the S3 folder structure locally
    if local_dir is None:
        local_dir = s3_folder_key.capitalize() + "/" if s3_folder_key == "dataset" else s3_folder_key + "/"
        local_dir = {"dataset": "Dataset/", "models": "models/"}.get(s3_folder_key, s3_folder_key + "/")

    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, s3_filename)

    client = _get_client()
    try:
        client.download_file(BUCKET_NAME, s3_key, local_path)
        print(f"Downloaded  s3://{BUCKET_NAME}/{s3_key}  →  '{local_path}'")
        return True
    except ClientError as e:
        print(f"Download failed: {e}")
        return False


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Upload the loan dataset
    upload_file("Dataset/Loan_Dataset.csv", "dataset")

    # Upload models
    upload_file("models/random_forest_model.pkl", "models")
    upload_file("models/scaler.pkl", "models")

    # Download it back
    download_file("dataset", "Loan_Dataset.csv")
