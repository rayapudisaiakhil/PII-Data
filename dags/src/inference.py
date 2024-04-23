# import torch
# import numpy as np
from datasets import load_from_disk
# from transformers import AutoModelForTokenClassification
# from sklearn.metrics import precision_recall_fscore_support
# import mlflow
# import os
# from train import train
# from predict import predict

# PROJECT_DIR = os.getcwd()
# ds_mapped_path = os.path.join(PROJECT_DIR, "dags", "processed", "ds_data")
# trained_model_path = os.path.join(PROJECT_DIR, "deberta1")

# def inference(ds_mapped_path, trained_model_path):
#     mlflow.set_tracking_uri('http://127.0.0.1:8080')
#     mlflow.set_experiment("DeBERTa Training")
#     with open('run_id.txt', 'r') as file:
#         run_id = file.read().strip()

#     print("Received data path at", ds_mapped_path)
#     ds_mapped = load_from_disk(ds_mapped_path)
    
#     input_ids = torch.tensor(ds_mapped["input_ids"], dtype=torch.long)
#     token_type_ids = torch.tensor(ds_mapped["token_type_ids"], dtype=torch.long)
#     attention_mask = torch.tensor(ds_mapped["attention_mask"], dtype=torch.long)
#     og_labels = ds_mapped['labels']
    
#     model = AutoModelForTokenClassification.from_pretrained(trained_model_path)
#     model.eval()

#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

#     predictions = outputs.logits
#     pred_softmax = torch.nn.functional.softmax(predictions, dim=-1).detach().numpy()
#     id2label = model.config.id2label
#     o_index = {v: k for k, v in id2label.items()}['O']

#     preds_without_O = pred_softmax[:, :, :o_index].argmax(-1)
#     O_preds = pred_softmax[:, :, o_index]
#     threshold = 0.9
#     preds_final = np.where(O_preds < threshold, preds_without_O, predictions.argmax(-1))

#     flat_og_labels = [label for sublist in og_labels for label in sublist]
#     flat_pred_labels = [label for sublist in preds_final for label in sublist]

#     precision, recall, f1, _ = precision_recall_fscore_support(flat_og_labels, flat_pred_labels, average='weighted')

#     with mlflow.start_run(run_id=run_id):
#         mlflow.log_metrics({
#             "Precision": precision,
#             "Recall": recall,
#             "F1 Score": f1
#         })

#     return precision, recall, f1

# def train_and_predict_if_needed():
#     precision, recall, f1 = inference(ds_mapped_path, trained_model_path)
#     comparison_threshold = 0.1  # Adjust as needed
    
#     if f1 < comparison_threshold:
#         print("Metrics not up to the mark. Retraining model.")
#         retrain=True
#     else:
#         retrain=False
#     return retrain    

# if __name__ == "__main__":
#     train_and_predict_if_needed()


# Evaluates and stashes model metrics
# from train import train
# from predict_data import predict
import os
from google.cloud import storage
from google.oauth2 import service_account
from datetime import datetime
import os
import csv
import mlflow
import shutil
PROJECT_DIR = os.getcwd()
import json

# key_path='/home/vineshgvk/PII-Data/config/key.json' # CHANGED
# project_id='piidatadetection'
# bucket_name='pii_train_data'
# latest_version_model= '/home/vineshgvk/PII-Data/latest_version' # CHANGED
def predict(test_mapped_path, trained_model_path):
    # mlflow.set_tracking_uri('http://127.0.0.1:8081')
    # mlflow.set_experiment("DeBERTa Training")
    # with open('run_id.txt', 'r') as file:
        # run_id = file.read().strip()

    print("Received test data path at", test_mapped_path)
    test_mapped = load_from_disk(test_mapped_path)

    input_ids = torch.tensor(test_mapped["input_ids"], dtype=torch.long)
    token_type_ids = torch.tensor(test_mapped["token_type_ids"], dtype=torch.long)
    attention_mask = torch.tensor(test_mapped["attention_mask"], dtype=torch.long)
    og_labels = test_mapped['labels']

    model = AutoModelForTokenClassification.from_pretrained(trained_model_path)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

    predictions = outputs.logits
    pred_softmax = torch.nn.functional.softmax(predictions, dim=-1).detach().numpy()
    id2label = model.config.id2label
    o_index = {v: k for k, v in id2label.items()}['O']

    preds_without_O = pred_softmax[:, :, :o_index].argmax(-1)
    O_preds = pred_softmax[:, :, o_index]
    threshold = 0.9
    preds_final = np.where(O_preds < threshold, preds_without_O, predictions.argmax(-1))

    flat_og_labels = [label for sublist in og_labels for label in sublist]
    flat_pred_labels = [label for sublist in preds_final for label in sublist]

    precision, recall, f1, _ = precision_recall_fscore_support(flat_og_labels, flat_pred_labels, average='weighted')

    # with mlflow.start_run(run_id=run_id):
    #     mlflow.log_metrics({
    #         "Precision": precision,
    #         "Recall": recall,
    #         "F1 Score": f1
    #     })

    return precision, recall, f1

def fetch_from_gcp(key_path,project_id,bucket_name,latest_version_model):
        # Load the credentials from the service account key file
        credentials = service_account.Credentials.from_service_account_file(key_path)
        
        # Initialize the client
        client = storage.Client(credentials=credentials, project=project_id)
        
        # Get the bucket
        bucket = client.get_bucket(bucket_name)
        
        # Specify the prefix for the folders
        prefix = 'models/v'
        
        # List all folders in the bucket
        blobs = bucket.list_blobs(prefix=prefix)
        folders = [os.path.dirname(blob.name) + '/' for blob in blobs]
        
        # Remove duplicates
        folders = list(set(folders))
        
        # If no folders are found, print a message and exit
        if not folders:
            print("No model found to fetch.")
            exit()
       
        # Find the latest folder
        latest_folder = max(folders, key=lambda folder: int(folder[len(prefix):-1]))
        
        # Specify the local directory where you want to save the downloaded folder
        local_directory = latest_version_model
        
        # Delete the local directory and all its contents
        if os.path.exists(local_directory):
            shutil.rmtree(local_directory)
        
        # Recreate the local directory
        os.makedirs(local_directory, exist_ok=True)
        
        # Download all files in the latest folder
        for blob in bucket.list_blobs(prefix=latest_folder):
            # Skip blobs that represent directories
            if not blob.name.endswith('/'):
                blob.download_to_filename(os.path.join(local_directory, os.path.basename(blob.name)))
        
        print("Latest model downloaded successfully.")
        return latest_version_model
   


def evaluate_model(**kwargs):
    PROJECT_DIR = os.getcwd()
    
    ti_t= kwargs['ti']
    test_mapped_path,_,_ = ti_t.xcom_pull(task_ids='tokenize_data')
    key_path = kwargs['KEY_PATH']
    project_id = kwargs['projectid']
    bucket_name = kwargs['bucket_name']
    # latest_version_model = kwargs['latest_version_model']
    model_save_at=os.path.join(PROJECT_DIR, 'latest_version')
    latest_version_model=fetch_from_gcp(key_path,project_id,bucket_name,model_save_at)
    
    
    precision, recall, f1 = predict(test_mapped_path = test_mapped_path, trained_model_path = latest_version_model)
    timeNow = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    to_append = [timeNow, precision, recall, f1]

    csv_header = ['time', 'precision', 'recall', 'f1'] # header for the csv file
    metrics_model_decay=os.path.join(PROJECT_DIR,'data','model_metrics.csv')

    # If model_metrics.csv file does not exist, create it and add the header.
    if not os.path.exists(metrics_model_decay):
        with open(metrics_model_decay, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
    
    # Append Model Metrics
    with open(metrics_model_decay, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(to_append)
        
    return precision, recall, f1,metrics_model_decay, latest_version_model
