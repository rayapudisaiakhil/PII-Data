import os
import json
from itertools import chain

# # Determine the absolute path of the project directory
# PROJECT_DIR = os.getcwd()

# INPUT_DATA_JSON_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'resampled.json')
# OUTPUT_LABELS_JSON_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'label_encoder_data.json')



def target_label_encoder(**kwargs):    
    """
    Load data from the input JSON file, compute label2id and id2label mappings,
    and save them to the output JSON file.

    :param input_json_path: Path to the input JSON file.
    :param output_json_path: Path to the output JSON file.
    """
    PROJECT_DIR = os.getcwd()
    print("fetched project directory successfully",PROJECT_DIR)    
    ti = kwargs['ti']
    inputPath = ti.xcom_pull(task_ids='resample_data')
    # inputPath = os.path.join(PROJECT_DIR, 'dags', 'processed', 'resampled.json')
    print("fetched path from resample_data task",inputPath)
    
    # Check if the input file exists
    if not os.path.exists(inputPath):
        raise FileNotFoundError(f"No data found at the specified path: {inputPath}")

    # Load data from input JSON file
    with open(inputPath, "r") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Loaded data is not in the expected list format.")

    # Assuming each item in the list contains a 'labels' key with a list of labels
    all_labels_list = [item["labels"] for item in data]
    all_labels = sorted(list(set(chain(*all_labels_list))))

    # Compute label2id and id2label mappings
    label2id = {label: i for i, label in enumerate(all_labels)}
    id2label = {i: label for label, i in label2id.items()}

    # Data to be saved
    label_encoder_data = {
        "label2id": label2id,
        "id2label": id2label,
        "all_labels":all_labels
    }
    outputPath = os.path.join(PROJECT_DIR, "dags", "processed", "label_encoder_data.json")
    

    # Save the mappings to the output JSON file
    with open(outputPath, "w") as file:
        json.dump(label_encoder_data, file, indent=4)

    print(f"Data saved to {outputPath}.")

    return outputPath


# target_label_encoder()
