import unittest
from unittest.mock import MagicMock, patch, mock_open
from dags.src.resampling import resample_data
import os


class TestResampleData(unittest.TestCase):
    @patch('dags.src.resampling.open')
    @patch('dags.src.resampling.json')
    @patch('dags.src.resampling.os.getcwd')
    def test_resample_data_function_call(self, mock_getcwd, mock_json, mock_open):
        # Mock os.getcwd to return a dummy project directory
        mock_getcwd.return_value = '/dummy/project/dir'
        
        # Call the function under test
        resample_data()

        # Assert that open method was called
        mock_open.assert_called_once()

        # Assert that json.dump was called
        mock_open().__enter__().write.assert_called_once()

if __name__ == '__main__':
    unittest.main()
