import os
import unittest
from unittest import mock
from unittest.mock import patch
from google.cloud import storage

from dags.src.data_slicing import load_data_from_gcp_and_save_as_json

class TestDataSlicing(unittest.TestCase):
    @patch('dags.src.data_slicing.storage.Client')
    @patch('dags.src.data_slicing.storage.Blob')
    @patch('dags.src.data_slicing.json.load', return_value={"some": "data"})
    @patch('dags.src.data_slicing.json.dump')
    @patch('builtins.open', mock.mock_open())
    def test_load_data_from_gcp_and_save_as_json(self, mock_open, mock_json_dump, mock_json_load, mock_blob_class, mock_client):
        # Mock the storage client and bucket
        mock_bucket = mock.MagicMock()
        mock_blob_instance = mock.MagicMock()  # This represents an instance of Blob

        # Set up the mock client to return the mock bucket
        mock_client.return_value.get_bucket.return_value = mock_bucket

        # Set the mock Blob class to return the mock_blob_instance when instantiated
        mock_blob_class.return_value = mock_blob_instance

        # Mock the os.path.exists to always return True
        with mock.patch('os.path.exists', return_value=True):
            # Mock the os.makedirs to do nothing
            with mock.patch('os.makedirs') as mock_makedirs:
                # Perform the test
                sliced_path, cumulative_path, end_index = load_data_from_gcp_and_save_as_json(
                    data_dir='/fake/dir',
                    num_data_points=10,
                    bucket_name='fake-bucket',
                    KEY_PATH='/fake/key.json'
                )

                # Verify os.makedirs was not called since the directory exists
                mock_makedirs.assert_not_called()

                # Verify the blob download was called on the mock_blob_instance
                mock_blob_instance.download_to_filename.assert_called_once()

                # Verify the function returned expected outputs
                self.assertTrue(sliced_path.endswith('sliced_train.json'))
                self.assertTrue(cumulative_path.endswith('cumulative_train.json'))
                self.assertIsInstance(end_index, int)

if __name__ == '__main__':
    unittest.main()
