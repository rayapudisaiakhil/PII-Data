import pandas as pd
import os
import pickle
import json

# Get the current working directory
PROJECT_DIR = os.getcwd()
# jsonPath is input to function. outPklPath is path after processing.
jsonPath = os.path.join(PROJECT_DIR, "data", "processed", "train.json")
outPklPath = os.path.join(PROJECT_DIR, "data", "processed", "missing_values.pkl")

def naHandler(inputPath=jsonPath, outputPath=outPklPath):
    if os.path.exists(inputPath):
        print("Loading data from:", inputPath)
        with open(inputPath, "r") as file:
            data = json.load(file)
            df = pd.DataFrame(data)
    else:
        raise FileNotFoundError(f"FAILED! No such path at {inputPath}")

    # Remove NAs wherever applicable.
    df.dropna(subset=['full_text'], inplace=True)

    nullCount = df.isnull().sum().sum()
    if nullCount > 0:
        nullsPresentError = f'Nulls {nullCount} still present in the dataset'
        print(nullsPresentError)
        raise ValueError(nullsPresentError)

    # Pickle dataset and push forward
    with open(outputPath, "wb") as file:
        pickle.dump(df, file)
    print(f'Data pickled after naHandling at {outputPath}')

    return outputPath

naHandler(inputPath=jsonPath, outputPath=outPklPath)