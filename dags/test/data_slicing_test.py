import os
import unittest
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from dags.src.data_slicing import load_data_from_gcp_and_save_as_json

class TestLoadDataFromGCPAndSaveAsJSON(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.mock_bucket_name = "mock-bucket-name"
        self.mock_key_path = "/path/to/mock/key"
        self.mock_num_data_points = 10

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('dags.src.data_slicing.os.environ', new_callable=MagicMock)
    @patch('dags.src.data_slicing.os.makedirs')
    @patch('dags.src.data_slicing.os.path.exists', return_value=False)
    @patch('dags.src.data_slicing.storage.Client')
    @patch('dags.src.data_slicing.storage.Blob')
    def test_load_data_from_gcp_and_save_as_json(self, mock_blob, mock_client, mock_path_exists, mock_makedirs, mock_environ):
        mock_client_instance = mock_client.return_value
        mock_blob_instance = mock_blob.return_value
        mock_environ.__getitem__.return_value = self.mock_key_path

        kwargs = {
            'data_dir': None,
            'num_data_points': self.mock_num_data_points,
            'bucket_name': self.mock_bucket_name,
            'KEY_PATH': self.mock_key_path
        }

        load_data_from_gcp_and_save_as_json(**kwargs)

        mock_client.assert_called_once_with()
        mock_client_instance.get_bucket.assert_called_once_with(self.mock_bucket_name)
        mock_blob.assert_called_once_with("Data/train.json", mock_client_instance.get_bucket.return_value)
        mock_blob_instance.download_to_filename.assert_called_once()
        mock_makedirs.assert_called_once()
        mock_path_exists.assert_called()
        mock_environ.__getitem__.assert_called_with('GOOGLE_APPLICATION_CREDENTIALS')

if __name__ == '__main__':
    unittest.main()
