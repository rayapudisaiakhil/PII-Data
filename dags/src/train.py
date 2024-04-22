# import os
# import json
# from google.cloud import storage
# import mlflow
# from datasets import load_from_disk
# from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
# import psutil

# from google.cloud import storage
# from google.oauth2 import service_account
# import os
# # def setup_google_cloud_credentials():
# #     """Sets up the environment variable for Google Cloud authentication."""
# #     project_dir = os.getcwd()
# #     credentials_path = os.path.join(project_dir, 'config', 'key.json')
    
# #     if not os.path.exists(credentials_path):
# #         raise FileNotFoundError("Google cloud credentials file not found at {}".format(credentials_path))
    
# #     # Set the GOOGLE_APPLICATION_CREDENTIALS environment variable
# #     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
# #     print("Set Google Cloud credentials from:", credentials_path)

# # def clear_system_caches():pt
# #     if os.name == 'posix':
# #         try:
# #             with open('/proc/sys/vm/drop_caches', 'w') as f:
# #                 f.write('3')
# #             print("System caches cleared.")
# #         except Exception as e:
# #             print(f"Failed to clear system caches: {e}")
# #     else:
# #         print("Cache clearing not supported on this OS.")

# # def upload_directory_to_gcs(bucket_name, source_directory, destination_prefix):
# #     """Uploads a directory to a specified GCS bucket."""
# #     storage_client = storage.Client()
# #     bucket = storage_client.bucket(bucket_name)
    
# #     for root, dirs, files in os.walk(source_directory):
# #         for filename in files:
# #             local_path = os.path.join(root, filename)
# #             relative_path = os.path.relpath(local_path, source_directory)
# #             cloud_path = os.path.join(destination_prefix, relative_path)
# #             blob = bucket.blob(cloud_path)
# #             blob.upload_from_filename(local_path)
# #             print(f"Uploaded {local_path} to {cloud_path}")

# # def train():
# #     # Set up Google Cloud credentials
# #     setup_google_cloud_credentials()
# #     PROJECT_DIR = os.getcwd()
# #     TRAINING_MODEL_PATH = "microsoft/deberta-v3-base"
# #     OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "deberta_model")
# #     LABEL_ENCODE_DATA_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'label_encoder_data.json')
# #     BUCKET_NAME = 'pii_train_data'

# #     mlflow.set_tracking_uri('http://127.0.0.1:8080')
# #     mlflow.set_experiment("DeBERTa Training")

# #     with open(LABEL_ENCODE_DATA_PATH, "r") as f:
# #         labels = json.load(f)

# #     train_mapped = load_from_disk(os.path.join(PROJECT_DIR, "dags", "processed", "train_data"))
# #     tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)
# #     model = AutoModelForTokenClassification.from_pretrained(
# #         TRAINING_MODEL_PATH, num_labels=len(labels["all_labels"]),
# #         id2label=labels["id2label"], label2id=labels["label2id"], ignore_mismatched_sizes=True)
# #     collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
# #     args = TrainingArguments(
# #         output_dir=OUTPUT_DIR, fp16=False, learning_rate=2e-5, num_train_epochs=1,
# #         per_device_train_batch_size=128, gradient_accumulation_steps=10,
# #         report_to="tensorboard", evaluation_strategy="no", do_eval=False,
# #         save_total_limit=1, logging_steps=25, lr_scheduler_type='cosine',
# #         metric_for_best_model="f1", greater_is_better=True, warmup_ratio=0.1, weight_decay=0.01)

# #     with mlflow.start_run(log_system_metrics=True) as run:
# #         run_id = run.info.run_id
# #         print("MLflow Run ID:", run_id)
# #         mlflow.log_params(args.to_dict())

# #         trainer = Trainer(model=model, args=args, train_dataset=train_mapped, data_collator=collator, tokenizer=tokenizer)
# #         trainer.train()

# #         mlflow.pytorch.log_model(model, "model")
# #         tokenizer.save_pretrained(OUTPUT_DIR)
# #         mlflow.log_artifact(OUTPUT_DIR, "tokenizer")

# #         # Upload model and tokenizer to GCS
# #         model_path = os.path.join(OUTPUT_DIR, "pytorch_model.bin")
# #         tokenizer_path = os.path.join(OUTPUT_DIR, "tokenizer_config.json")
# #         upload_to_gcs(BUCKET_NAME, model_path, "models/deberta_v3_base/pytorch_model.bin")
# #         upload_to_gcs(BUCKET_NAME, tokenizer_path, "models/deberta_v3_base/tokenizer_config.json")

# #     with open('run_id.txt', 'w') as f:
# #         f.write(run_id)

# #     print("Training complete. Model and tokenizer saved and uploaded.")

# # if __name__ == "__main__":
# #     train()

# import os
# import json
# from datasets import load_from_disk
# from transformers import Trainer, TrainingArguments, AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification
# import mlflow
# import psutil

# # def log_system_metrics():
# #     """Logs CPU and memory usage."""
# #     cpu_percent = psutil.cpu_percent(interval=1)
# #     memory = psutil.virtual_memory()
# #     mlflow.log_metrics({
# #         'cpu_percent': cpu_percent,
# #         'memory_used': memory.used / (1024 * 1024 * 1024),  # Convert to GB
# #         'memory_percent': memory.percent
# #     })

# def clear_system_caches():
#     """Clear system caches (for Linux systems)."""
#     if os.name == 'posix':
#         try:
#             with open('/proc/sys/vm/drop_caches', 'w') as f:
#                 f.write('3')  # Clears page cache, dentries and inodes.
#             print("System caches cleared.")
#         except Exception as e:
#             print(f"Failed to clear system caches: {e}")
#     else:
#         print("Cache clearing not supported on this OS.")


# # def log_system_metrics_explicitly():
# #     """Logs detailed system metrics under custom tags or names for enhanced visibility."""
# #     cpu_percent = psutil.cpu_percent(interval=1)
# #     memory = psutil.virtual_memory()
# #     mlflow.log_metrics({
# #         'system_cpu_percent': cpu_percent,
# #         'system_memory_used_gb': memory.used / (1024 * 1024 * 1024),  # Convert to GB
# #         'system_memory_percent': memory.percent
# #     })

# # def upload_gcloud(bucket_name,project_id,folder_to_upload,destination_path):
# #     # Load the credentials from the service account key file
# #     project_dir = os.getcwd()
# #     credentials_path = os.path.join(project_dir, 'config', 'key.json')
# #     credentials = service_account.Credentials.from_service_account_file(credentials_path)

# #     # Initialize the client
# #     client = storage.Client(credentials=credentials, project=project_id)

# #     # Get the bucket
# #     bucket = client.get_bucket(bucket_name)

# #     # Specify the local directory you want to upload
# #     local_folder = folder_to_upload

# #     # Specify the folder where you want to upload the files
# #     cloud_folder = destination_path

# #     # List all files in the local directory
# #     files = os.listdir(local_folder)

# #     for file in files:
# #         # Get a blob object for each file
# #         blob = bucket.blob(cloud_folder + file)

# #         # Upload the file
# #         blob.upload_from_filename(os.path.join(local_folder, file))

# #     print("All files uploaded successfully.")

# def train():
#     PROJECT_DIR = os.getcwd()
#     TRAINING_MODEL_PATH = "microsoft/deberta-v3-base"
#     OUTPUT_DIR = os.path.join(PROJECT_DIR, "output", "deberta_model")
#     LABEL_ENCODE_DATA_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'label_encoder_data.json')

#     # Setup MLflow
#     mlflow.set_tracking_uri('http://127.0.0.1:8080')
#     mlflow.set_experiment("DeBERTa Training")

#     # Load label encoding mappings from JSON file
#     with open(LABEL_ENCODE_DATA_PATH, "r") as f:
#         labels = json.load(f)
#         label2id = labels["label2id"]
#         id2label = labels["id2label"]
#         all_labels = labels["all_labels"]

#     # Load the training dataset
#     train_mapped = load_from_disk(os.path.join(PROJECT_DIR, "dags", "processed", "train_data"))
#     tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)
#     model = AutoModelForTokenClassification.from_pretrained(
#         TRAINING_MODEL_PATH,
#         num_labels=len(all_labels),
#         id2label=id2label,
#         label2id=label2id,
#         ignore_mismatched_sizes=True
#     )

#     collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)
#     args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         fp16=False,
#         learning_rate=2e-5,
#         num_train_epochs=1,
#         per_device_train_batch_size=1024,
#         gradient_accumulation_steps=10,
#         report_to="tensorboard",
#         evaluation_strategy="no",
#         do_eval=False,
#         save_total_limit=1,
#         # logging_steps=10,
#         lr_scheduler_type='cosine',
#         metric_for_best_model="f1",
#         greater_is_better=True,
#         warmup_ratio=0.1,
#         weight_decay=0.01,
        
#     )

#     # Start an MLflow run and get the run ID
#     with mlflow.start_run(log_system_metrics=True) as run:
#         run_id = run.info.run_id
#         print("MLflow Run ID:", run_id)

#         mlflow.log_params({
#             "learning_rate": args.learning_rate,
#             "num_epochs": args.num_train_epochs,
#             "train_batch_size": args.per_device_train_batch_size,
#             "gradient_accumulation": args.gradient_accumulation_steps,
#             "warmup_ratio": args.warmup_ratio,
#             "weight_decay": args.weight_decay,
#             "model_name": TRAINING_MODEL_PATH
#         })

#         # Initialize and start the training
#         trainer = Trainer(
#             model=model,
#             args=args,
#             train_dataset=train_mapped,
#             data_collator=collator,
#             tokenizer=tokenizer
#         )
#         trainer.train()

#         # Log the model and tokenizer as artifacts
#         mlflow.pytorch.log_model(model, "model")
#         tokenizer.save_pretrained(OUTPUT_DIR)
#         model.save_pretrained(OUTPUT_DIR)
#         mlflow.log_artifact(OUTPUT_DIR, "tokenizer")
#         # Log system metrics after training
#         # Log custom system metrics
#         # log_system_metrics_explicitly()
#     # Save the run ID for later use by the prediction script
#     with open('run_id.txt', 'w') as f:
#         f.write(run_id)

#         #------------edit here
#     # def upload_gcloud(bucket_name,project_id,folder_to_upload,destination_path):
#     # # Load the credentials from the service account key file
#     #     project_dir = os.getcwd()
#     #     credentials_path = os.path.join(project_dir, 'config', 'key.json')
#     #     credentials = service_account.Credentials.from_service_account_file(credentials_path)

#     #     # Initialize the client
#     #     client = storage.Client(credentials=credentials, project=project_id)

#     #     # Get the bucket
#     #     bucket = client.get_bucket(bucket_name)

#     #     # Specify the local directory you want to upload
#     #     local_folder = folder_to_upload

#     #     # Specify the folder where you want to upload the files
#     #     cloud_folder = destination_path

#     #     # List all files in the local directory
#     #     files = os.listdir(local_folder)

#     #     for file in files:
#     #         # Get a blob object for each file
#     #         blob = bucket.blob(cloud_folder + file)

#     #         # Upload the file
#     #         blob.upload_from_filename(os.path.join(local_folder, file))

#     #     print("All files uploaded successfully.")

#     # bucket_name='pii_train_data'
#     # project_id='piidatadetection'
#     # folder_to_upload=OUTPUT_DIR#'/Users/lahariboni/Desktop/PII-Data/deberta1'
#     # destination_path='models/v1/'
#     # upload_gcloud(bucket_name,project_id,folder_to_upload,destination_path)

#     print("Training complete. Model and tokenizer saved.")

# if __name__ == "__main__":
#     train()


# # bucket_name='pii_train_data'
# # project_id='piidatadetection'
# # folder_to_upload='/Users/lahariboni/Desktop/PII-Data/deberta1'
# # destination_path='models/v1/'
# # upload_gcloud(bucket_name,project_id,folder_to_upload,destination_path)



#------------------------------------------------
import os
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, TrainerCallback
from sklearn.model_selection import ParameterGrid
import mlflow
from mlflow.tracking import MlflowClient
from datasets import load_from_disk
from predict import predict  # Ensure this module and function are defined correctly
from torch.utils.tensorboard import SummaryWriter

class TensorBoardCallback(TrainerCallback):
    """Custom callback for logging metrics to TensorBoard."""
    def __init__(self, writer):
        self.writer = writer

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero:
            if logs is not None:
                step = state.global_step
                for k, v in logs.items():
                    if 'loss' in k or 'f1' in k or 'recall' in k or 'precision' in k:  # Log specific metrics
                        self.writer.add_scalar(k, v, step)
                        

def train_model(**kwargs):
    PROJECT_DIR = os.getcwd()
    TRAINING_MODEL_PATH = "dslim/distilbert-NER" #model for finetuning
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "Models", "distilibert") #directory to save the trained model
    
    ti = kwargs['ti']
    LABEL_ENCODE_DATA_PATH = ti.xcom_pull(task_ids='label_encoder')
    # LABEL_ENCODE_DATA_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'label_encoder_data.json')
    _,train_mapped,TEST_DATA_PATH=ti.xcom_pull(task_ids='tokenize_data')
    # TEST_DATA_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'test_data')
    
    # train_mapped = load_from_disk(os.path.join(PROJECT_DIR, "dags", "processed", "train_data"))

    mlflow.set_tracking_uri('http://127.0.0.1:8080')
    client = MlflowClient()
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_name = f"distilbert-models-{current_time}"
    experiment_id = client.create_experiment(experiment_name)
    print(f"New Experiment Created: {experiment_name} with ID: {experiment_id}")

    tensorboard_logdir = os.path.join(OUTPUT_DIR, "tensorboard_logs")
    writer = SummaryWriter(tensorboard_logdir)

    with open(LABEL_ENCODE_DATA_PATH, "r") as f:
        labels = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(TRAINING_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(
        TRAINING_MODEL_PATH,
        num_labels=len(labels['all_labels']),
        id2label=labels['id2label'],
        label2id=labels['label2id'],
        ignore_mismatched_sizes=True
    )

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    param_grid = {
        'learning_rate': [1e-5, 2e-5, 5e-5],
        'num_train_epochs': [1, 2, 3],
        'per_device_train_batch_size': [16, 32, 64],
        'gradient_accumulation_steps': [16,20,25]
    }
    all_params = list(ParameterGrid(param_grid))

    best_f1 = 0
    best_model_path = None
    best_params = {}

    for param in all_params:
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_params(param)
            args = TrainingArguments(
                output_dir=OUTPUT_DIR,
                fp16=False,
                learning_rate=param['learning_rate'],
                num_train_epochs=param['num_train_epochs'],
                per_device_train_batch_size=param['per_device_train_batch_size'],
                gradient_accumulation_steps=16,
                evaluation_strategy="steps",
                eval_steps=50,
                do_eval=True,
                save_total_limit=1,
                logging_steps=20,
                lr_scheduler_type='cosine',
                metric_for_best_model="f1",
                greater_is_better=True,
                warmup_ratio=0.1,
                weight_decay=0.01,
                report_to="tensorboard"
            )
            trainer = Trainer(
                model=model,
                args=args,
                train_dataset=train_mapped,
                data_collator=DataCollatorForTokenClassification(tokenizer),
                tokenizer=tokenizer,
                callbacks=[TensorBoardCallback(writer)]
            )
            trainer.train()
            mlflow.pytorch.log_model(model, "model")
            tokenizer.save_pretrained(OUTPUT_DIR)
            model.save_pretrained(OUTPUT_DIR)
            mlflow.log_artifact(OUTPUT_DIR, "tokenizer")
            
            precision, recall, f1 = predict(TEST_DATA_PATH, OUTPUT_DIR)  # Ensure predict function is correctly implemented
            
            mlflow.log_metric('recall', recall)
            mlflow.log_metric('precision', precision)
            mlflow.log_metric('f1', f1)
            writer.add_scalar('Precision', precision, trainer.state.global_step)
            writer.add_scalar('Recall', recall, trainer.state.global_step)
            writer.add_scalar('F1 Score', f1, trainer.state.global_step)
            
            if f1 > best_f1:
                best_f1 = f1
                best_params = param
                best_model_path = os.path.join(OUTPUT_DIR, "best_model")
                model.save_pretrained(best_model_path)
                tokenizer.save_pretrained(best_model_path)

    writer.close()
    print(f"Training complete. Best model saved at {best_model_path} with F1 score: {best_f1}.")
    print(f"Best parameters: {best_params}")
    return best_model_path,best_f1

# if __name__ == "__main__":
#     train()
