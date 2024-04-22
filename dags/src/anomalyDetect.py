"""
Load data from the input JSON file, validate datatypes and formats.

:param input_json_path: Path to the input JSON file.

"""

import json
import pandas as pd
def anomalyDetect(**kwargs):
    ti = kwargs['ti']
    input_path,_,_ = ti.xcom_pull(task_ids='data_slicing_batches_task')  # Get the output of the prior task
    with open(input_path, "r") as file:
        data = json.load(file)
        
    df = pd.DataFrame(data)
    results = {
        "stats": {},
        "issues": {}
    }

    # Calculate statistics
    results['stats']['average_text_length'] = df['full_text'].apply(lambda x: len(x.split())).mean()
    results['stats']['average_token_length'] = df['tokens'].apply(len).mean()
    results['stats']['num_rows'] = df.shape[0]
    results['stats']['num_columns'] = df.shape[1]

    # Expected norms and types
    expected_text_length = 10
    expected_token_length = 5
    min_rows = 100

    # Check discrepancies
    if results['stats']['average_text_length'] != expected_text_length:
        results['issues']['text_length_issue'] = f"Expected average text length {expected_text_length}, found {results['stats']['average_text_length']}"

    if results['stats']['average_token_length'] < expected_token_length:
        results['issues']['token_length_issue'] = f"Expected minimum average token length {expected_token_length}, found {results['stats']['average_token_length']}"

    if results['stats']['num_rows'] < min_rows:
        results['issues']['row_count_issue'] = f"Insufficient data rows for analysis, expected at least {min_rows}, found {results['stats']['num_rows']}"

    # Data type and value checks
    expected_dtypes = {'document': 'int64', 'full_text': 'object', 'tokens': 'object',
                       'trailing_whitespace': 'object', 'labels': 'object'}
    allowed_labels = ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS',
                      'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM',
                      'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O']

    for column, expected_dtype in expected_dtypes.items():
        if column in df.columns:
            if df[column].dtype != expected_dtype:
                results['issues'][f'{column}_dtype_issue'] = f"Expected {expected_dtype}, found {df[column].dtype}"
        else:
            results['issues'][f'{column}_missing'] = "Column is missing in the dataset"

    # if 'trailing_whitespace' in df.columns and not df['trailing_whitespace'].isin([True, False]).all():
    #     results['issues']['trailing_whitespace_values'] = "Contains values other than True or False"
    
    if 'trailing_whitespace' in df.columns:
        if not all(df['trailing_whitespace'].map(lambda x: all(isinstance(i, bool) for i in x))):
            results['issues']['trailing_whitespace_values'] = "One or more lists in 'trailing_whitespace' column contains non-boolean values."
    else:
        results['issues']['trailing_whitespace_missing'] = "'trailing_whitespace' column is missing."


    if 'labels' in df.columns:
        flat_labels = df['labels'].explode().unique()
        invalid_labels = [label for label in flat_labels if label not in allowed_labels]
        if invalid_labels:
            results['issues']['invalid_labels'] = f"Invalid labels detected: {invalid_labels}"
    else:
        results['issues']['labels_missing'] = "The 'labels' column is missing"

    return results

# input_path='/home/vineshgvk/PII-Data/dags/src/resampled.json'
# r=anomalyDetect(input_path)
# print(r)

# import os
# import json
# import logging
# import pandas as pd

# # Stash the logs in the data/logs path.
# logsPath = os.path.abspath(os.path.join(os.getcwd(), 'data', 'logs'))
# if not os.path.exists(logsPath):
#     # Create the folder if it doesn't exist
#     os.makedirs(logsPath)
#     print(f"Folder '{logsPath}' created successfully.")

# logging.basicConfig(filename=os.path.join(logsPath, 'logs.log'),  # log filename with today's date.
#                     filemode="w",  # write mode
#                     level=logging.ERROR,  # Set error as the default log level.
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # logging format
#                     datefmt='%Y-%m-%d %H:%M:%S', )  # logging (asctime) date format


# def anomalyDetect(**kwargs):
#     '''
#     anomalyDetect looks for the right data types, and checks for text length below a threshold
#     Args:
#         inputPath: Input JSON path after process_data_and_save.
#         outputPath: Output pickle path after dupeRemoval processing.
#     Returns:
#         outputPath
#     '''
#         # Get the current working directory
#     PROJECT_DIR = os.getcwd()
#     # jsonPath is input to function. outPklPath is path after processing.

#     textThreshold = 25  # Remove records with full_text < 25 words.
#     trainSamples = 100  # Needs at least trainSamples amount of records for training.
#     expectedDtypes = {'document': int,
#                     'full_text': object,
#                     'tokens': object,
#                     'trailing_whitespace': object,
#                     'labels': object
#                     }
#     ti = kwargs['ti']
#     inputPath = ti.xcom_pull(task_ids='load_data_from_gcp')
#     print("fetched path from load_gcp_data task",inputPath)
#     # Open file in read mode if exists
#     if os.path.exists(inputPath):
#         with open(inputPath, "r") as file:
#             data = json.load(file)
#     else:
#         raise FileNotFoundError(f"FAILED! No such path at {inputPath}")

#     # Convert JSON data to DataFrame
#     df = pd.DataFrame(data)

#     # Check for text length
#     rowsRemoved = 0
#     for index, row in df.iterrows():
#         if len(row['full_text'].split()) < textThreshold:
#             rowsRemoved += 1
#             df.drop(index, inplace=True)
#     print(f'Records removed because of text length threshold {textThreshold}: {rowsRemoved} records')

#     # Check for trainSamples threshold for training
#     if df.shape[0] < trainSamples:
#         print(f'Not enough training samples for model to be trained')

#     # Check for appropriate text types
#     for col in df.columns:
#         if df[col].dtype != expectedDtypes[col]:
#             print(f'{col} data type mismatch')
#             print(df[col].dtype)

#     # Check for tokens length to be >25
#     for index, row in df.iterrows():
#         if len(row['tokens']) < 25:
#             logging.error(f"Tokens size less than 25 in row {index}")

#     # Check if trailing_whitespace is int and has only values 1 or 0
#     valid_values = [True,False]
#     if 'trailing_whitespace' in df.columns:
#         if not df['trailing_whitespace'].isin(valid_values).all():
#             logging.error("The 'trailing_whitespace' column contains values other than 1 or 0.")
#             print("The 'trailing_whitespace' column contains values other than 1 or 0.")
#     else:
#         logging.error("The 'trailing_whitespace' column is missing.")
#         print("The 'trailing_whitespace' column is missing.")

#     # Check if labels are one of the 12 unique values
#     allowed_labels = ['B-EMAIL', 'B-ID_NUM', 'B-NAME_STUDENT', 'B-PHONE_NUM', 'B-STREET_ADDRESS',
#                       'B-URL_PERSONAL', 'B-USERNAME', 'I-ID_NUM', 'I-NAME_STUDENT', 'I-PHONE_NUM',
#                       'I-STREET_ADDRESS', 'I-URL_PERSONAL', 'O']

#     if 'labels' in df.columns:
#         # Flatten the list of labels
#         flat_labels = df['labels'].explode()
        
#         # Check for invalid labels
#         invalid_labels = flat_labels[~flat_labels.isin(allowed_labels)]
#         if not invalid_labels.empty:
#             logging.error(f"The 'labels' column contains invalid values: {invalid_labels.unique()}.")
#     else:
#         logging.error("The 'labels' column is missing.")
#     return True

