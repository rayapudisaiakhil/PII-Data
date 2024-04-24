import unittest
from unittest.mock import patch, MagicMock
from dags.src.data_slicing import load_data_from_gcp_and_save_as_json

class TestLoadDataFromGCPAndSaveAsJSON(unittest.TestCase):
    @patch('dags.src.data_slicing.os.makedirs')
    @patch('dags.src.data_slicing.os.path.exists', return_value=False)
    @patch('dags.src.data_slicing.storage.Client')
    @patch('dags.src.data_slicing.storage.Blob')
    @patch('dags.src.data_slicing.json.load', side_effect=[[]])
    def test_load_data_from_gcp_and_save_as_json(self, mock_json_load, mock_blob, mock_client, mock_path_exists, mock_makedirs):
        mock_client_instance = mock_client.return_value
        mock_blob_instance = mock_blob.return_value

        kwargs = {
            'data_dir': None,
            'num_data_points': 10,
            'bucket_name': 'test_bucket',
            'KEY_PATH': 'test_key.json'
        }

        load_data_from_gcp_and_save_as_json(**kwargs)

        mock_client.assert_called_once_with()
        mock_client_instance.get_bucket.assert_called_once_with('test_bucket')
        mock_blob.assert_called_once_with("Data/train.json", mock_client_instance.get_bucket.return_value)
        mock_blob_instance.download_to_filename.assert_called_once()
        mock_makedirs.assert_called_once()
        mock_path_exists.assert_called()

        # Ensure that json.load is called with the correct filename
        mock_json_load.assert_called_once_with('/home/runner/work/PII-Data/PII-Data/dags/processed/Fetched/train.json')

if __name__ == '__main__':
    unittest.main()
