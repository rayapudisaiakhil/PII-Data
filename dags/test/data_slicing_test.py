import unittest
from unittest.mock import patch, MagicMock
import json

from dags.src.data_slicing import load_data_from_gcp_and_save_as_json

class TestDataDownload(unittest.TestCase):
    @patch('dags.src.data_slicing.storage.Client')
    @patch('dags.src.data_slicing.storage.Blob')
    @patch('builtins.open', new_callable=unittest.mock.mock_open)
    @patch('json.load', return_value={"data": "dummy data"})
    @patch('json.dump')
    @patch('os.path.exists', return_value=True)
    @patch('os.makedirs')
    def test_load_data_from_gcp_and_save_as_json(
        self, mock_makedirs, mock_exists, mock_json_dump, mock_json_load, mock_open, mock_blob_class, mock_client
    ):
        # Mock bucket and blob setup
        mock_bucket = MagicMock()
        mock_blob_instance = MagicMock()
        mock_client.return_value.get_bucket.return_value = mock_bucket
        mock_blob_class.return_value = mock_blob_instance

        # Setup test parameters
        test_data_dir = '/fake/dir'
        test_num_data_points = 10
        test_bucket_name = 'fake-bucket'
        test_key_path = '/fake/key.json'

        # Perform the test
        try:
            result = load_data_from_gcp_and_save_as_json(
                data_dir=test_data_dir,
                num_data_points=test_num_data_points,
                bucket_name=test_bucket_name,
                KEY_PATH=test_key_path
            )
            # Verify expected result structure
            self.assertIsInstance(result, tuple)
            self.assertEqual(len(result), 3)
            # Assuming the function returns paths, check they are strings
            self.assertIsInstance(result[0], str)
            self.assertIsInstance(result[1], str)
            # Assuming the function returns an index, check it's an integer
            self.assertIsInstance(result[2], int)
        except Exception as e:
            self.fail(f"Function raised an exception: {e}")

        # Assert interactions
        mock_blob_instance.download_to_filename.assert_called_once()  # Ensuring file was attempted to be downloaded
        mock_json_dump.assert_called()  # Ensure data was attempted to be saved

if __name__ == '__main__':
    unittest.main()
