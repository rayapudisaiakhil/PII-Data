import unittest
from unittest.mock import patch, MagicMock
from google.cloud import storage

# Assuming your function is located in a file called `data_slicing.py` under a module named `dags.src`
from dags.src.data_slicing import load_data_from_gcp_and_save_as_json

class TestDataDownload(unittest.TestCase):
    @patch('dags.src.data_slicing.json.dump')
    @patch('dags.src.data_slicing.json.load', return_value={"data": "value"})
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='0')
    @patch('dags.src.data_slicing.storage.Blob')
    @patch('dags.src.data_slicing.storage.Client')
    def test_load_data_from_gcp_and_save_as_json(self, mock_client, mock_blob_class, mock_open, mock_json_load, mock_json_dump):
        # Mock the storage client and blob
        mock_bucket = MagicMock()
        mock_blob_instance = MagicMock()

        # Setup the mock client to return the mock bucket
        mock_client.return_value.get_bucket.return_value = mock_bucket

        # Set the mock Blob class to return the mock_blob_instance when instantiated
        mock_blob_class.return_value = mock_blob_instance

        # Prepare the mock input parameters
        mock_data_dir = '/path/to/data'
        mock_num_data_points = 10
        mock_bucket_name = 'your-bucket-name'
        mock_key_path = '/path/to/key.json'

        # Mock path.exists to simulate different file existence states
        with patch('os.path.exists', side_effect=lambda path: path.endswith('.json') or path.endswith('.txt')):
            with patch('os.makedirs') as mock_makedirs:
                # Perform the test
                local_sliced_path, local_cumulative_path, end_index = load_data_from_gcp_and_save_as_json(
                    data_dir=mock_data_dir,
                    num_data_points=mock_num_data_points,
                    bucket_name=mock_bucket_name,
                    KEY_PATH=mock_key_path
                )

                # Assertions to check behaviors
                mock_makedirs.assert_not_called()
                mock_blob_instance.download_to_filename.assert_called_once()

                # Ensure the function returns a tuple
                self.assertIsInstance((local_sliced_path, local_cumulative_path, end_index), tuple)

                # Check if the function returned the expected paths
                self.assertIn("dags/processed/Fetched/sliced_train_", local_sliced_path)
                self.assertIn("dags/processed/Fetched/cumulative_train_", local_cumulative_path)

if __name__ == '__main__':
    unittest.main()
