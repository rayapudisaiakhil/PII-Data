import unittest
from unittest.mock import patch, MagicMock

from dags.src.data_slicing import load_data_from_gcp_and_save_as_json

class TestDataDownload(unittest.TestCase):
    @patch('dags.src.data_slicing.storage.Client')
    @patch('dags.src.data_slicing.storage.Blob')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('json.load', return_value={"data": "dummy data"})
    @patch('json.dump')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def test_load_data_from_gcp_and_save_as_json(self, mock_makedirs, mock_exists, mock_json_dump, mock_json_load, mock_open, mock_blob_class, mock_client):
        # Prepare the mock client and blob
        mock_bucket = MagicMock()
        mock_blob_instance = MagicMock()
        mock_client.return_value.get_bucket.return_value = mock_bucket
        mock_blob_class.return_value = mock_blob_instance

        # Test inputs
        test_data_dir = '/fake/dir'
        test_num_data_points = 10
        test_bucket_name = 'fake-bucket'
        test_key_path = '/fake/key.json'

        # Perform the test
        result = load_data_from_gcp_and_save_as_json(
            data_dir=test_data_dir,
            num_data_points=test_num_data_points,
            bucket_name=test_bucket_name,
            KEY_PATH=test_key_path
        )

        # Check the result type
        self.assertIsInstance(result, tuple, "Function should return a tuple.")
        # Assuming the function returns paths, check they are strings or None
        self.assertTrue(all(isinstance(x, (str, type(None))) for x in result), "Each item in the tuple should be a string or None.")

        # Assert interactions
        mock_blob_instance.download_to_filename.assert_called_once()  # Ensuring file was attempted to be downloaded
        mock_json_dump.assert_called()  # Ensure data was attempted to be saved

if __name__ == '__main__':
    unittest.main()
