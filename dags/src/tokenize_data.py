# import os
# import json
# import numpy as np
# from transformers import AutoTokenizer
# from datasets import Dataset
# import pickle


# def tokenize_data(**kwargs):
#     '''
#     tokenize_data
#     '''
#     PROJECT_DIR = os.getcwd()
#     print("fetched project directory successfully",PROJECT_DIR)    
#     # ti_r = kwargs['ti']
#     # data_path = ti_r.xcom_pull(task_ids='resample_data')
#     data_path = os.path.join(PROJECT_DIR, 'dags', 'processed', 'resampled.json')
#     print("fetched path from resample_data task",data_path)
    
#     # ti_l = kwargs['ti']
#     # label2id_path = ti_l.xcom_pull(task_ids='label_encoder')
#     label2id_path = os.path.join(PROJECT_DIR, 'dags', 'processed', 'label_encoder_data.json')
#     print("fetched path from resample_data task",label2id_path)

#     # Load label2id from JSON file

#     output_tokenized = os.path.join(PROJECT_DIR, 'dags', 'processed', 'tokenized_data.pkl')
#     model_path = 'microsoft/deberta-v3-base'
#     max_inference_length = 2048
#     with open(label2id_path, "r") as file:
#         label_encoder_data = json.load(file)
#         label2id = label_encoder_data["label2id"]
#         print("labelencoder data has been fectched successfully")

#     # Load data from JSON file
#     with open(data_path, "r") as file:
#         data = json.load(file)
#         print("resampled data has been fectched successfully")

#     # Initialize tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     print("Inititated the tokenizer using the model", model_path)

#     # Define tokenization function
#     def tokenize(example):
#         text = []
#         labels = []

#         for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):
#             text.append(t)
#             labels.extend([l] * len(t))

#             if ws:
#                 text.append(" ")
#                 labels.append("O")

#         tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_inference_length)

#         labels = np.array(labels)

#         text = "".join(text)
#         token_labels = []

#         for start_idx, end_idx in tokenized.offset_mapping:
#             if start_idx == 0 and end_idx == 0:
#                 token_labels.append(label2id["O"])
#                 continue

#             if text[start_idx].isspace():
#                 start_idx += 1

#             token_labels.append(label2id[labels[start_idx]])

#         length = len(tokenized.input_ids)

#         return {**tokenized, "labels": token_labels, "length": length}

#     # Create Dataset from data
#     ds = Dataset.from_dict({
#         "full_text": [x["full_text"] for x in data],
#         "document": [str(x["document"]) for x in data],
#         "tokens": [x["tokens"] for x in data],
#         "trailing_whitespace": [x["trailing_whitespace"] for x in data],
#         "provided_labels": [x["labels"] for x in data],
#     })

#     # Tokenize dataset
#     ds = ds.map(tokenize, num_proc=1)

#     # Save tokenized dataset to JSON file
#     with open(output_tokenized,"wb") as f:
#         pickle.dump(ds,f)

#     return output_tokenized
# tokenize_data()

import os
import json
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset
import pickle

def tokenize_data(**kwargs):
    '''
    tokenize_data
    '''
    PROJECT_DIR = os.getcwd()
    print("fetched project directory successfully",PROJECT_DIR)    
    ti_r = kwargs['ti']
    data_path = ti_r.xcom_pull(task_ids='resample_data')
    # ti_r = kwargs['ti']
    # data_path = ti_r.xcom_pull(task_ids='resample_data')
    # data_path = os.path.join(PROJECT_DIR, 'dags', 'processed', 'resampled.json')
    print("fetched path from resample_data task",data_path)
    
    ti_l = kwargs['ti']
    label2id_path = ti_l.xcom_pull(task_ids='label_encoder')
    # label2id_path = os.path.join(PROJECT_DIR, 'dags', 'processed', 'label_encoder_data.json')
    print("fetched path from label_encoder task",label2id_path)

    # Load label2id from JSON file
    # PROJECT_DIR = os.getcwd()    
    # data_path = os.path.join(PROJECT_DIR, 'dags', 'processed', 'resampled.json')
    # output_tokenized_json_path = os.path.join(PROJECT_DIR, 'dags', 'processed', 'tokenized_data.json')
    # label2id_path = os.path.join(PROJECT_DIR, 'dags', 'processed', 'label_encoder_data.json')
    model_path = 'dslim/distilbert-NER'
    # max_inference_length = 2048
    TRAINING_MAX_LENGTH=512
    with open(label2id_path, "r") as file:
        label_encoder_data = json.load(file)
        label2id = label_encoder_data["label2id"]
        print("labelencoder data has been fectched successfully")

    # Load data from JSON file
    with open(data_path, "r") as file:
        data = json.load(file)
        print("resampled data has been fectched successfully")


    
    # Create a dataset from the provided data dictionary
    # Create a dataset from the provided data dictionary
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "labels": [x["labels"] for x in data],
    })
    
    # Split the dataset into training and testing sets
    train_dataset, test_dataset = ds.train_test_split(test_size=0.01, seed=42).values()
    
        # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Inititated the tokenizer using the model", model_path)
    
    def tokenize_train(example, tokenizer, label2id, max_length):

        # Rebuild text from tokens
        text = []
        labels = []

        for t, l, ws in zip(
            example["tokens"], example["labels"], example["trailing_whitespace"]
        ):
            # print("entered for loop")
            text.append(t)
            labels.extend([l] * len(t))

            if ws:
                text.append(" ")
                labels.append("O")

        # Actual tokenization
        tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length, padding="max_length",truncation=True)

        labels = np.array(labels)

        text = "".join(text)
        token_labels = []

        for start_idx, end_idx in tokenized.offset_mapping:
            # CLS token
            if start_idx == 0 and end_idx == 0:
                token_labels.append(label2id["O"])
                continue

            # Case when token starts with whitespace
            if text[start_idx].isspace():
                start_idx += 1

            token_labels.append(label2id[labels[start_idx]])

        length = len(tokenized.input_ids)

        return {**tokenized, "labels": token_labels, "length": length}
    print("starting tokenize ds_mapped")
    ds_mapped = ds.map(tokenize_train, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH}, num_proc=3)
    output_ds_json_path = os.path.join(PROJECT_DIR, 'dags', 'processed', 'ds_data.json')  # Specify your output path for the training dataset
    print("starting to jsonify dsmap")
    ds_mapped.to_json(output_ds_json_path)
    print("finishenify")
    ds_mapped.save_to_disk(os.path.join(PROJECT_DIR, 'dags', 'processed', 'ds_data'))
    print("saved to disk")

    train_mapped = train_dataset.map(tokenize_train, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH}, num_proc=3)
    test_mapped = test_dataset.map(tokenize_train, fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": TRAINING_MAX_LENGTH}, num_proc=3)

    output_train_json_path = os.path.join(PROJECT_DIR, 'dags', 'processed', 'train_data.json')  # Specify your output path for the training dataset
    output_test_json_path = os.path.join(PROJECT_DIR, 'dags', 'processed', 'test_data.json')  # Specify your output path for the testing dataset

    train_mapped.to_json(output_train_json_path)
    test_mapped.to_json(output_test_json_path)
    train_mapped.save_to_disk(os.path.join(PROJECT_DIR, 'dags', 'processed', 'train_data'))
    test_mapped.save_to_disk(os.path.join(PROJECT_DIR, 'dags', 'processed', 'test_data'))
    

    return output_ds_json_path,output_train_json_path,output_test_json_path
    # return ds_mapped,train_mapped,test_mapped
    
    
# tokenize_data()

