import unittest
import json
from unittest.mock import MagicMock, patch
from data_slicing import load_data_from_gcp_and_save_as_json

class TestDataSlicing(unittest.TestCase):
    @patch('data_slicing.os')
    @patch('data_slicing.storage')
    def test_load_data_from_gcp_and_save_as_json(self, mock_storage, mock_os):
        # Mocking kwargs
        kwargs = {
            'data_dir': 'test_data',
            'num_data_points': 10,
            'bucket_name': 'test_bucket',
            'KEY_PATH': 'test_key.json'
        }
        
        # Mocking os.getcwd()
        mock_os.getcwd.return_value = '/test/project'
        
        # Mocking os.path.exists() to return False for destination_dir
        mock_os.path.exists.return_value = False
        
        # Mocking storage.Client() and its methods
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_blob.download_to_filename.return_value = None
        mock_bucket.get_blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.get_bucket.return_value = mock_bucket
        mock_storage.Client.return_value = mock_client
        
        # Call the function
        sliced_filename, cumulative_filename, end_index = load_data_from_gcp_and_save_as_json(**kwargs)
        
        # Assertions
        with open(sliced_filename, 'r') as f:
            sliced_data = json.load(f)
        with open(cumulative_filename, 'r') as f:
            cumulative_data = json.load(f)
        
        self.assertIsInstance(sliced_data, list)
        self.assertIsInstance(cumulative_data, list)
        self.assertEqual(len(sliced_data), 10)  # Assuming it slices correctly
        self.assertEqual(len(cumulative_data), 10)  # Assuming it slices correctly

if __name__ == '__main__':
    unittest.main()
