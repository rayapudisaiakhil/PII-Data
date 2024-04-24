import unittest
from unittest.mock import patch
from dags.src.data_slicing import load_data_from_gcp_and_save_as_json

class TestDataSlicing(unittest.TestCase):
    @patch('dags.src.data_slicing.storage')
    def test_load_data_from_gcp_and_save_as_json(self, mock_storage):
        # Mocking storage.Client() and storage.Blob() behavior
        mock_client = mock_storage.Client.return_value
        mock_bucket = mock_client.get_bucket.return_value
        mock_blob = mock_bucket.blob.return_value
        
        # Call the function
        load_data_from_gcp_and_save_as_json(data_dir='mock_data_dir',
                                             num_data_points=10,
                                             bucket_name='mock_bucket',
                                             KEY_PATH='mock_key_path')
        
        # Assertions
        self.assertTrue(mock_client.get_bucket.called)  # Ensure get_bucket is called
        self.assertTrue(mock_blob.download_to_filename.called)  # Ensure download_to_filename is called

if __name__ == '__main__':
    unittest.main()
