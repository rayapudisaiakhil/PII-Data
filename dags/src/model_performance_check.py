# import os
# import shutil
# from google.cloud import storage
# from google.oauth2 import service_account
# from predict import predict  # Assuming predict.py contains the predict function

# # Set up Google Cloud Storage credentials
# PROJECT_DIR = os.getcwd()
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(PROJECT_DIR, "config", "key.json")

# # Create a client for interacting with Google Cloud Storage
# client = storage.Client()

# # Get the specified bucket
# bucket = client.get_bucket('pii_train_data')

# # Specify the prefix for the folders
# prefix = 'latest_version/'

# # List all files in the specified prefix
# blobs = bucket.list_blobs(prefix=prefix)

# # If no files are found, exit
# if not blobs:
#     print("No models found in the specified directory.")
#     exit()

# # Specify the local directory where you want to save the downloaded folder
# local_directory = '/Users/jayat/Desktop/PII-Data/latest_version'

# # Delete the local directory and all its contents
# if os.path.exists(local_directory):
#     shutil.rmtree(local_directory)

# # Recreate the local directory
# os.makedirs(local_directory, exist_ok=True)

# # Download all files in the specified prefix
# for blob in blobs:
#     # Download the file
#     blob.download_to_filename(os.path.join(local_directory, os.path.basename(blob.name)))

# print("Latest model downloaded successfully.")

# # Load the test data
# test_mapped_path = os.path.join(PROJECT_DIR, "dags", "processed", "test_data")
# trained_model_path = os.path.join(PROJECT_DIR, "deberta1")

# # Predict using the retrained model
# precision_retrained, recall_retrained, f1_retrained = predict(test_mapped_path, trained_model_path)

# # Predict using the latest model
# precision_latest, recall_latest, f1_latest = predict(test_mapped_path, local_directory)

# # Compare F1 scores
# if f1_retrained > f1_latest:
#     print("Retrained model has a higher F1 score. Uploading as a new version.")
#     # Upload the retrained model as a new version
#     version_number = len(list(bucket.list_blobs(prefix=prefix))) + 1
#     version_prefix = f'{prefix}v{version_number}/'
#     for file_name in os.listdir(local_directory):
#         if os.path.isfile(os.path.join(local_directory, file_name)):
#             blob = bucket.blob(version_prefix + file_name)
#             blob.upload_from_filename(os.path.join(local_directory, file_name))
# else:
#     print("Latest model has a higher or equal F1 score. No upload needed.")
# def check_performance(**kwargs):
#     ti= kwargs['ti']
#     precision, recall, f1,metrics_model_decay = ti.xcom_pull(task_ids='inference')
#     if recall>0.9 and f1> 0.8:
#         retrain=True
#     else:
#         retrain=False
#     return retrain

def check_performance(**kwargs):
    ti = kwargs['ti']
    # Pull the results tuple from the inference task
    results = ti.xcom_pull(task_ids='inference', key='return_value')

    if results:
        precision, recall, f1, metrics_model_decay = results

        # Determine if retraining is necessary based on recall and f1 score
        if recall < 0.9 and f1 < 0.8:
            retrain = True
        else:
            retrain = False

        # Optionally, log these metrics or use them further
        print(f"Precision: {precision}, Recall: {recall}, F1: {f1}, Model Decay Metrics: {metrics_model_decay}")

        # Push this decision to XCom for other tasks
        ti.xcom_push(key='retrain', value=retrain)
        return retrain
    else:
        raise ValueError("No data received from 'inference' task")