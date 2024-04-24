import os
import unittest
from unittest import mock
from unittest.mock import patch
from google.cloud import storage

from dags.src.data_slicing import load_data_from_gcp_and_save_as_json

class TestDataDownload(unittest.TestCase):
    @patch('dags.src.data_slicing.storage.Client')
    @patch('dags.src.data_slicing.storage.Blob')
    def test_load_data_from_gcp_and_save_as_json(self, mock_blob_class, mock_client):
        # Mock the storage client and bucket
        mock_bucket = mock.MagicMock()
        mock_blob_instance = mock.MagicMock()  # This represents an instance of Blob

        # Set up the mock client to return the mock bucket
        mock_client.return_value.get_bucket.return_value = mock_bucket

        # Set the mock Blob class to return the mock_blob_instance when instantiated
        mock_blob_class.return_value = mock_blob_instance

        # Mock the os.path.exists to return True for train.json and False for other files
        mock_exists_paths = {
            os.path.join('dags', 'processed', 'Fetched', 'train.json'): True,
            os.path.join('dags', 'processed', 'Fetched', 'end_index.txt'): False,
            os.path.join('dags', 'processed', 'Fetched', 'sliced_train_0_10.json'): False,
            os.path.join('dags', 'processed', 'Fetched', 'cumulative_train_0_10.json'): False,
        }
        with mock.patch('os.path.exists', side_effect=lambda path: mock_exists_paths.get(path, False)):
            with mock.patch('os.makedirs'):
                # Perform the test
                kwargs = {
                    'data_dir': 'dags/processed',
                    'num_data_points': 10,
                    'bucket_name': 'pii_train_data',
                    'KEY_PATH': 'config/key.json'
                }
                sliced_path, cumulative_path, end_index = load_data_from_gcp_and_save_as_json(**kwargs)

                # Verify the blob download was called on the mock_blob_instance
                mock_blob_instance.download_to_filename.assert_called_once()

                # Check if the function returned the expected paths
                self.assertEqual(sliced_path, os.path.join('dags', 'processed', 'Fetched', 'sliced_train_0_10.json'))
                self.assertEqual(cumulative_path, os.path.join('dags', 'processed', 'Fetched', 'cumulative_train_0_10.json'))
                self.assertEqual(end_index, 10)

if __name__ == '__main__':
    unittest.main()
