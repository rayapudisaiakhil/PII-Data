import torch
import numpy as np
from datasets import load_from_disk
from transformers import AutoModelForTokenClassification
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os

PROJECT_DIR = os.getcwd()
test_mapped_path = os.path.join(PROJECT_DIR, "dags", "processed", "train_data")
trained_model_path = os.path.join(PROJECT_DIR, "distilbert")

def predict(test_mapped_path, trained_model_path):
    # Set up MLflow
    mlflow.set_tracking_uri('http://127.0.0.1:8080')  # Update this URI to your MLflow server
    mlflow.set_experiment("Token_Classification_Metrics")

    with open('run_id.txt', 'r') as file:
        run_id = file.read().strip()

    print("Received test data path at", test_mapped_path)
    test_mapped = load_from_disk(test_mapped_path)
    
    input_ids = torch.tensor(test_mapped["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(test_mapped["attention_mask"], dtype=torch.long)
    og_labels = test_mapped['labels']
    
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

    precision, recall, f1, _ = precision_recall_fscore_support(flat_og_labels, flat_pred_labels, labels=list(id2label.keys()), average=None)
    
    with mlflow.start_run(run_id=run_id):
        for i, label in enumerate(id2label.keys()):
            mlflow.log_metric(f"{id2label[label]}_Precision", precision[i])
            mlflow.log_metric(f"{id2label[label]}_Recall", recall[i])
            mlflow.log_metric(f"{id2label[label]}_F1", f1[i])

    metrics = pd.DataFrame({'Label': [id2label[label] for label in id2label.keys()],
                            'Precision': precision,
                            'Recall': recall,
                            'F1 Score': f1})

    cm = confusion_matrix(flat_og_labels, flat_pred_labels, labels=list(id2label.keys()))

    df_cm = pd.DataFrame(cm, index=[id2label[i] for i in list(id2label.keys())],
                            columns=[id2label[i] for i in list(id2label.keys())])
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=True, fmt='g')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt_path = 'confusion_matrix.png'
    plt.savefig(plt_path)
    plt.close()

    return metrics

if __name__ == "__main__":
    metrics = predict(test_mapped_path, trained_model_path)
    print("Metrics for each label:")
    print(metrics)
