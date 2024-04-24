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

        # Mock the os.path.exists to return True for train.json
        with mock.patch('os.path.exists', return_value=True):
            with mock.patch('os.makedirs'):
                # Perform the test
                kwargs = {
                    'data_dir': 'dags/processed',
                    'num_data_points': 10,
                    'bucket_name': 'pii_train_data',
                    'KEY_PATH': 'config/key.json'
                }
                load_data_from_gcp_and_save_as_json(**kwargs)

                # Verify the blob download was called on the mock_blob_instance
                mock_blob_instance.download_to_filename.assert_called_once()

if __name__ == '__main__':
    unittest.main()
