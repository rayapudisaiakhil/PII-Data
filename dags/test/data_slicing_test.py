import unittest
from unittest.mock import patch, MagicMock
from dags.src.data_slicing import load_data_from_gcp_and_save_as_json

class TestDataSlicing(unittest.TestCase):
    @patch('dags.src.data_slicing.storage.Client')
    @patch('dags.src.data_slicing.storage.Blob')
    @patch('dags.src.data_slicing.os.makedirs')  # Add makedirs mock
    @patch('dags.src.data_slicing.os.path.exists', return_value=False)
    def test_load_data_from_gcp_and_save_as_json(self, mock_exists, mock_makedirs, mock_blob_class, mock_client):
        # Mock the storage client and bucket
        mock_bucket = mock_client.return_value.get_bucket.return_value
        mock_blob_instance = mock_blob_class.return_value
        mock_blob_instance.download_to_filename.return_value = None

        # Perform the test
        sliced_file, cumulative_file, end_index = load_data_from_gcp_and_save_as_json(
            data_dir='/mocked/data/dir',
            num_data_points=10,
            bucket_name='mock-bucket',
            KEY_PATH='/mocked/key/path'
        )

        # Verify that makedirs is called
        mock_makedirs.assert_called_once()

        # Verify that download_to_filename is called
        mock_blob_instance.download_to_filename.assert_called_once()

        # Verify the function returned the expected values
        self.assertIsNotNone(sliced_file)
        self.assertIsNotNone(cumulative_file)
        self.assertIsNotNone(end_index)

if __name__ == '__main__':
    unittest.main()
