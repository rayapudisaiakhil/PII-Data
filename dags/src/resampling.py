import json
import os

def resample_data(**kwargs):
    PROJECT_DIR = os.getcwd()
    print("fetched project directory successfully", PROJECT_DIR)
    ti = kwargs['ti']
    inputPath,_,_ = ti.xcom_pull(task_ids='data_slicing_batches_task')    #this would return sliced path, cumulative path, end_index (number data points fetched till now)
    # inputPath = os.path.join(PROJECT_DIR, "dags", "processed", "train.json")
    print("fetched the input file", inputPath) 

    try:
        # Load data from a JSON file
        with open(inputPath, "r") as json_file:
            data = json.load(json_file)

        # Consider only the first 5 records
        # data = data[:5]

        # Downsampling of negative examples
        p = []  # positive samples (contain relevant labels)
        n = []  # negative samples (contain "O" only)
        for d in data:
            if any(label != "O" for label in d["labels"]):
                p.append(d)
            else:
                n.append(d)

        # Combine positive samples with a third of negative samples
        data_n = p + n[:len(n) // 3]

        # Specify the output file path
        output_file_path = os.path.join(PROJECT_DIR, "dags", "processed", "resampled.json")

        # Save the modified data to a new JSON file
        with open(output_file_path, "w") as output_file:
            json.dump(data_n, output_file)

        print("Processed data saved successfully.", output_file_path)
        return output_file_path

    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# resample_data()


# import json
# import os
# import pandas as pd
# import pickle  # Import the pickle module

# def resample_data(**kwargs):
#     PROJECT_DIR = os.getcwd()
#     print("fetched project directory successfully", PROJECT_DIR)    
#     inputPath = os.path.join(PROJECT_DIR,"dags","processed","train.json")
#     print("fetched the input file", inputPath) 
#     try:
#         # Load data from a JSON file
#         with open(inputPath, "r") as json_file:
#             data = json.load(json_file)

#         data = data[:2]


# #             # Save the moified JSON files
# #             with open(os.path.join(destination_dir, 'train.json'), 'w') as f:
# #                 json.dump(train_data, f)

        
#         # Downsampling of negative examples
#         p = []  # positive samples (contain relevant labels)
#         n = []  # negative samples (presumably contain entities that are possibly wrongly classified as entity)
#         for d in data:
#             if any(label != "O" for label in d["labels"]):
#                 p.append(d)
#             else:
#                 n.append(d)

#         # Combine data
#         data_n = data + p + n[:len(n) // 3]

#         # Specify the output file path
#         output_file_path = os.path.join(PROJECT_DIR, "dags", "processed", "resampled.json")

#         # Convert the processed DataFrame to a list of dicts (if not already in this format) and save as JSON
#         with open(output_file_path, "w") as output_file:
#             json.dump(data_n, output_file)

#         print("Processed data saved successfully.", output_file_path)
#         return output_file_path

#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")

# resample_data()