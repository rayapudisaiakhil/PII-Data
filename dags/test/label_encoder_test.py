import unittest
from unittest.mock import MagicMock, patch
from dags.src.label_encoder import target_label_encoder

class TestTargetLabelEncoder(unittest.TestCase):
    @patch('dags.src.label_encoder.open')
    @patch('dags.src.label_encoder.json')
    def test_target_label_encoder_function_call(self, mock_json, mock_open):
        # Mock ti object with xcom_pull method
        mock_ti = MagicMock()
        
        # Mock the behavior of open to return a file handle with a valid JSON string
        mock_open().__enter__().read.return_value = '[{"labels": ["label1", "label2"]}]'
        
        # Mock the behavior of json.load to return a list of dictionaries
        mock_json.load.return_value = [{"labels": ["label1", "label2"]}]
        
        # Call the function under test
        output_path = target_label_encoder(ti=mock_ti)
        
        # Assert that the function returns a non-empty output path
        self.assertTrue(output_path)

        # Assert that open method was called with the correct arguments
        mock_open.assert_called_once_with(..., 'w')

if __name__ == '__main__':
    unittest.main()
