import unittest
from unittest.mock import MagicMock, patch, mock_open
from dags.src.resampling import resample_data
import os
class TestResampleData(unittest.TestCase):
    @patch('dags.src.resampling.open')
    @patch('dags.src.resampling.json')
    @patch('dags.src.resampling.os.getcwd')
    def test_resample_data_function_call(self, mock_getcwd, mock_json, mock_open):
        # Mock input data
        input_data = [{'text': 'sample text', 'labels': ['O', 'B-PERSON', 'O']}]
        
        # Mock os.getcwd to return a dummy project directory
        mock_getcwd.return_value = '/dummy/project/dir'
        
        # Mock open method to return input data
        mock_open().__enter__().read.return_value = input_data
        
        # Call the function under test
        output_path = resample_data()

        # Assert that the function returns a non-empty output path
        self.assertTrue(output_path)

        # Assert that open method was called
        mock_open.assert_called_once()

        # Assert that json.dump was called with the correct data
        mock_open().__enter__().write.assert_called_once_with(input_data)

if __name__ == '__main__':
    unittest.main()
