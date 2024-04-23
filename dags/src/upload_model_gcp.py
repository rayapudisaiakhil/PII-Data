import os
import json
from google.cloud import storage
import mlflow
from datasets import load_from_disk
from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
from google.oauth2 import service_account
import re

def get_next_version_number(bucket, prefix):
    # Get the list of blobs in the bucket with the prefix
    blobs = bucket.list_blobs(prefix=prefix, delimiter='/')
    versions = []

    for blob in blobs:
        # Extract the version number from the folder name
        match = re.match(r"models/v(\d+)/", blob.name)
        if match:
            versions.append(int(match.group(1)))

    if versions:
        # Return the next version number
        return max(versions) + 1
    else:
        # Return 1 if no versions found
        return 1

def upload_model_to_gcp(**kwargs):
    KEY_PATH = kwargs['KEY_PATH']
    projectid = kwargs['projectid']
    bucket_name = kwargs['bucket_name']
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = storage.Client(credentials=credentials, project=projectid)
    bucket = client.get_bucket(bucket_name)
    ti = kwargs['ti']
    folder_to_upload = ti.xcom_pull(task_ids='train_new_model')

    # Get the next version number for the new folder
    next_version_number = get_next_version_number(bucket, "models/")
    destination_path = f'models/v{next_version_number}/'

    for root, dirs, files in os.walk(folder_to_upload):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, folder_to_upload)
            if 'runs' not in relative_path.split(os.path.sep):
                cloud_path = os.path.join(destination_path, relative_path)
                blob = bucket.blob(cloud_path)
                blob.upload_from_filename(local_path)
                print(f"Uploaded {local_path} to {cloud_path}")

    print("All files except 'runs' uploaded successfully.")
  
    









