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
    test_mapped_path = os.path.join(PROJECT_DIR, "dags", "processed", "test_data")
    trained_model_path = os.path.join(PROJECT_DIR, "deberta1")

    def predict(test_mapped_path, trained_model_path):
        mlflow.set_tracking_uri('http://127.0.0.1:8080')
        mlflow.set_experiment("DeBERTa Training")
        with open('run_id.txt', 'r') as file:
            run_id = file.read().strip()

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

    if __name__ == "__main__":
        predict(test_mapped_path, trained_model_path)



# import torch
# import numpy as np
# from datasets import load_from_disk
# from transformers import AutoModelForTokenClassification
# import pandas as pd
# from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# import mlflow
# import os
# PROJECT_DIR = os.getcwd()
# test_mapped_path = os.path.join(PROJECT_DIR, "dags", "processed", "test_data")
# trained_model_path = os.path.join(PROJECT_DIR, "deberta1")
# def predict(test_mapped_path, trained_model_path):
    

#     mlflow.set_tracking_uri('http://127.0.0.1:8080')
#     mlflow.set_experiment("DeBERTa Training")
#     with open('run_id.txt', 'r') as file:
#         run_id = file.read().strip()

#     print("Received test data patat", test_mapped_path)
#     test_mapped = load_from_disk(test_mapped_path)
    
#     input_ids = torch.tensor(test_mapped["input_ids"], dtype=torch.long)
#     token_type_ids = torch.tensor(test_mapped["token_type_ids"], dtype=torch.long)
#     attention_mask = torch.tensor(test_mapped["attention_mask"], dtype=torch.long)
#     og_labels = test_mapped['labels']
    
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

#     # Flatten the arrays to compute global metrics and confusion matrix
#     flat_og_labels = [label for sublist in og_labels for label in sublist]
#     flat_pred_labels = [label for sublist in preds_final for label in sublist]

#     precision, recall, f1, _ = precision_recall_fscore_support(flat_og_labels, flat_pred_labels, average='weighted')
#     cm = confusion_matrix(flat_og_labels, flat_pred_labels, labels=list(id2label.keys()))

#     with mlflow.start_run(run_id=run_id):
#         mlflow.log_metrics({
#             "Precision": precision,
#             "Recall": recall,
#             "F1 Score": f1
#         })

#         # Logging the confusion matrix as an artifact
#         df_cm = pd.DataFrame(cm, index=[id2label[i] for i in list(id2label.keys())],
#                              columns=[id2label[i] for i in list(id2label.keys())])
#         plt.figure(figsize=(10, 7))
#         sns.heatmap(df_cm, annot=True, fmt='g')
#         plt.title('Confusion Matrix')
#         plt.ylabel('Actual')
#         plt.xlabel('Predicted')
#         plt_path = 'confusion_matrix.png'
#         plt.savefig(plt_path)
#         plt.close()
#         mlflow.log_artifact(plt_path)

#     print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

# if __name__ == "__main__":
#     predict(test_mapped_path, trained_model_path)

