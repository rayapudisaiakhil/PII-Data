import unittest
from unittest.mock import MagicMock, patch
from dags.src.label_encoder import target_label_encoder

class TestTargetLabelEncoder(unittest.TestCase):
    @patch('dags.src.label_encoder.open')
    def test_target_label_encoder_function_call(self, mock_open):
        # Call the function under test
        output_path = target_label_encoder(ti=MagicMock())
        
        # Assert that the function returns a non-empty output path
        self.assertTrue(output_path)

        # Assert that open method was called
        mock_open.assert_called()

if __name__ == '__main__':
    unittest.main()
