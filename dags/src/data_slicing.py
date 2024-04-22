import os
import json
from google.cloud import storage
# 'data_dir': data_dir, 'num_data_points': num_data_points, 'cumulative': cumulative,'bucket_name':bucket_name,
#                'KEY_PATH':KEY_PATH
# def load_data_from_gcp_and_save_as_json(data_dir, num_data_points=10):
def load_data_from_gcp_and_save_as_json(**kwargs):
    try:
        PROJECT_DIR = os.getcwd()
        print("fetched project directory successfully",PROJECT_DIR)          
        data_dir=kwargs['data_dir']
        num_data_points=kwargs['num_data_points']
        bucket_name=kwargs['bucket_name']
        KEY_PATH=kwargs['KEY_PATH']
        
        # bucket_name = 'pii_train_data'
        # KEY_PATH = os.path.join(PROJECT_DIR, "config", "key.json")
        
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = KEY_PATH
        
        # Updated directory path to include the 'Fetched' folder
        destination_dir = os.path.join(PROJECT_DIR, "dags", "processed", "Fetched")
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            print("Folder created in:", destination_dir)
        
        local_file_path = os.path.join(destination_dir, "train.json")
        
        # Check if local train.json exists
        if not os.path.exists(local_file_path):
            client = storage.Client()
            bucket = client.get_bucket(bucket_name)
            blob = storage.Blob("Data/train.json", bucket)
            blob.download_to_filename(local_file_path)
            print(f"File train.json downloaded from the GCP cloud and saved at {local_file_path}.")
        else:
            print(f"File train.json already exists at {local_file_path}, using existing file.")
        
        # Load train.json
        with open(local_file_path, 'r') as f:
            train_data = json.load(f)
        
        end_index_path = os.path.join(destination_dir, 'end_index.txt')
        start_index = 0
        if os.path.exists(end_index_path):
            with open(end_index_path, 'r') as f:
                start_index = int(f.read().strip())
        
        end_index = start_index + num_data_points
        
        # Handle sliced data
        new_sliced_filename = os.path.join(destination_dir, f'sliced_train_{start_index}_{end_index}.json')
        # Handle cumulative data
        new_cumulative_filename = os.path.join(destination_dir, f'cumulative_train_0_{end_index}.json')
        
        # Delete old files
        for file in os.listdir(destination_dir):
            file_path = os.path.join(destination_dir, file)
            if file_path not in [new_sliced_filename, new_cumulative_filename, local_file_path, end_index_path]:
                os.remove(file_path)
                print(f"Deleted old file: {file}")
        
        # Save new sliced data
        with open(new_sliced_filename, 'w') as f:
            json.dump(train_data[start_index:end_index], f)
        print(f"Sliced data saved as {new_sliced_filename}.")
        
        # Save new cumulative data
        if os.path.exists(new_cumulative_filename):
            with open(new_cumulative_filename, 'r') as f:
                old_data = json.load(f)
            new_cumulative_data = old_data + train_data[start_index:end_index]
        else:
            new_cumulative_data = train_data[:end_index]
        with open(new_cumulative_filename, 'w') as f:
            json.dump(new_cumulative_data, f)
        print(f"Cumulative data updated and saved as {new_cumulative_filename}.")
        
        # Update the end index file
        with open(end_index_path, 'w') as f:
            f.write(str(end_index))
        
        return new_sliced_filename,new_cumulative_filename,end_index
    
        # return os.path.join(destination_dir, new_sliced_filename), end_index
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# # Set your data directory here
# data_dir = '/home/vineshgvk/PII-Data'
# print(load_data_from_gcp_and_save_as_json(data_dir=data_dir, num_data_points=10))
