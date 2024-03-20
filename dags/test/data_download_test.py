import os
import unittest
from unittest import mock
from unittest.mock import patch
from google.cloud import storage

# Ensure the correct import path for the function you want to test
from src.data_download import load_data_from_gcp_and_save_as_json

class TestDataDownload(unittest.TestCase):
    @patch('src.data_download.storage.Client')
    def test_load_data_from_gcp_and_save_as_json(self, mock_client):
        # Mock the storage client and bucket
        mock_bucket = mock.MagicMock()
        mock_blob = mock.MagicMock()

        # Set up the mock client to return the mock bucket
        mock_client.return_value.get_bucket.return_value = mock_bucket
        
        # Configure the mock bucket to return the mock blob for a given file name
        mock_bucket.blob.return_value = mock_blob
        
        # Mock the os.path.exists to always return True
        with mock.patch('os.path.exists', return_value=True):
            # Mock the os.makedirs to do nothing
            with mock.patch('os.makedirs') as mock_makedirs:
                # Perform the test
                local_file_path = load_data_from_gcp_and_save_as_json()
                
                # Verify os.makedirs was not called since the directory exists
                mock_makedirs.assert_not_called()
                
                # Verify the blob download was called
                mock_blob.download_to_filename.assert_called_once()
                
                # Check if the function returned the expected path
                self.assertIn("dags/processed/train.json", local_file_path)
                
                # Optionally, check for print output or other side effects

if __name__ == '__main__':
    unittest.main()
