
# from predict_data import predict
import os
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, TrainerCallback
from sklearn.model_selection import ParameterGrid
import mlflow
from mlflow.tracking import MlflowClient
from datasets import load_from_disk

def predict(test_mapped_path, trained_model_path):
    # with open('run_id.txt', 'r') as file:
    #     run_id = file.read().strip()

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

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    return precision, recall, f1

def model_version(**kwargs):
    ti= kwargs['ti']
    _,_,_,_,latest_version_model = ti.xcom_pull(task_ids='inference')  
    
    best_model_path,_=ti.xcom_pull(task_ids='train_new_model') 
    
    _,_,test_mapped=ti.xcom_pull(task_ids='tokenize_data') 
    
    precision_o, recall_o, f1_o = predict(test_mapped_path = test_mapped, trained_model_path = latest_version_model)
    precision_n, recall_n, f1_n = predict(test_mapped_path = test_mapped, trained_model_path = best_model_path)

    if recall_o < recall_n and f1_o < f1_n:
        version_retrained_model = True
    else:
        version_retrained_model = False
        
    return version_retrained_model
    