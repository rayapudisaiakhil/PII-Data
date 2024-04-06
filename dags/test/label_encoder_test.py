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
        
        # Call the function under test
        output_path = target_label_encoder(ti=mock_ti)
        
        # Assert that the function returns a non-empty output path
        self.assertTrue(output_path)

        # Assert that open method was called with the correct arguments
        mock_open.assert_called_once_with(..., 'r')

        # Assert that json.load was called with the correct file handle
        mock_json.load.assert_called_once_with(mock_open().__enter__())

        # Assert that open method was called with the correct arguments
        mock_open.assert_any_call(..., 'w')

        # Assert that json.dump was called with the correct arguments
        mock_open().__enter__().write.assert_called_once()

if __name__ == '__main__':
    unittest.main()
