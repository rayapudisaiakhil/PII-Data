import os
import pickle
import pandas as pd

# # Get the current working directory
# PROJECT_DIR = os.getcwd()
# # pklPath is input to function. outPklPath is path after processing.
# pklPath = os.path.join(PROJECT_DIR, "data", "processed", "missing_values.pkl")
# outPklPath = os.path.join(PROJECT_DIR, "data", "processed", "dup_removed.pkl")

def dupeRemoval(**kwargs):
    PROJECT_DIR = os.getcwd()
    print("fetched project directory successfully",PROJECT_DIR)    
    ti = kwargs['ti']
    inputPath = ti.xcom_pull(task_ids='missing_values_removal')
    # inputPath = os.path.join(PROJECT_DIR, "dags", "processed", "missing_values.pkl")
    print("fetched path from missing values task op",inputPath)
    if os.path.exists(inputPath):
        print("Loading data from:", inputPath)
        with open(inputPath, "rb") as file:
            df = pickle.load(file)
    else:
        raise FileNotFoundError(f"FAILED! No such path at {inputPath}")

    # Removes rows i.e. any duplicates in the full_text column
    df.drop_duplicates(subset=['full_text'], inplace=True)
    outputPath = os.path.join(PROJECT_DIR, "dags", "processed", "duplicate_removal.pkl")
    # Pickle dataset and push forward
    # opening file in write-binary mode
    with open(outputPath, "wb") as file:
        pickle.dump(df, file)
    print(f'Data pickled after dupeRemoval at {outputPath}')

    return outputPath
# dupeRemoval()

        
