
import os
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, TrainerCallback
from sklearn.model_selection import ParameterGrid
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import mlflow
from mlflow.tracking import MlflowClient
from datasets import load_from_disk
from sklearn.metrics import precision_recall_fscore_support
import torch
import numpy as np
import accelerate
import logging
import dagshub
from torch.utils.tensorboard import SummaryWriter

def predict(test_mapped_path, trained_model_path):
    # with open('run_id.txt', 'r') as file:
    #     run_id = file.read().strip()

    print("Received test data path at", test_mapped_path)
    logging.info(f"Received test data path at {test_mapped_path}")
    test_mapped = load_from_disk(test_mapped_path)

    input_ids = torch.tensor(test_mapped["input_ids"], dtype=torch.long)
    # token_type_ids = torch.tensor(test_mapped["token_type_ids"], dtype=torch.long)
    attention_mask = torch.tensor(test_mapped["attention_mask"], dtype=torch.long)
    og_labels = test_mapped['labels']
    logging.info(f'fetched input_id, attention_mask, og_labels')

    model = AutoModelForTokenClassification.from_pretrained(trained_model_path)
    model.eval()

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

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

    precision, recall, f1, _ = precision_recall_fscore_support(flat_og_labels, flat_pred_labels, average='weighted', zero_division=0)

    return precision, recall, f1                        

def train_model(**kwargs):
    PROJECT_DIR = os.getcwd()
    TRAINING_MODEL_PATH = "dslim/distilbert-NER" # model for finetuning
    OUTPUT_DIR = os.path.join(PROJECT_DIR, "Models", "distilibert") #directory to save the trained model
    logging.info(f'model loaded: {TRAINING_MODEL_PATH}')
    
    # ti = kwargs['ti'] # For DAG
    # print(ti)
    # LABEL_ENCODE_DATA_PATH = ti.xcom_pull(task_ids='label_encoder') # For DAG
    LABEL_ENCODE_DATA_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'label_encoder_data.json') # For local
    # _,train_mapped,TEST_DATA_PATH=ti.xcom_pull(task_ids='tokenize_data') # For DAG
    TEST_DATA_PATH = os.path.join(PROJECT_DIR, 'dags', 'processed', 'test_data') # For local
    
    train_mapped = os.path.join(PROJECT_DIR, 'dags', 'processed', 'train_data') # For local
    train_mapped = load_from_disk(train_mapped)
    # TEST_DATA_PATH=load_from_disk(TEST_DATA_PATH)

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
        logging.info(f"Created output directory at {OUTPUT_DIR}")

    param_grid = {
        'learning_rate': [1e-5, 2e-5, 5e-5] #,
        # 'num_train_epochs': [1, 2, 3],
        # 'per_device_train_batch_size': [16, 32, 64],
        # 'gradient_accumulation_steps': [16,20,25]
    }
    all_params = list(ParameterGrid(param_grid))

    best_f1 = 0
    best_model_path = None
    best_params = {}

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        fp16=False,
        learning_rate= 1e-5,
        num_train_epochs=1, #param['num_train_epochs'],
        per_device_train_batch_size=64, #param['per_device_train_batch_size'],
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
    print('trainer loaded...')
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_mapped,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer
        # callbacks=[TensorBoardCallback(writer)] # For Tensorboard
    )
    
    trainer.train()
    print('model trained')
    # mlflow.pytorch.log_model(model, "model")
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    logging.info(f"Model saved at {OUTPUT_DIR}")
    # mlflow.log_artifact(OUTPUT_DIR, "tokenizer")
    
    precision, recall, f1 = predict(TEST_DATA_PATH, OUTPUT_DIR)  # Ensure predict function is correctly implemented
    logging.info(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print('model saved')

    a=10
    return OUTPUT_DIR, a

# if __name__ == "__main__":
#     train_model()