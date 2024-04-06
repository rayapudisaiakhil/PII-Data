import os
import pickle
import unittest
from unittest.mock import MagicMock, patch, mock_open
from dags.src.duplicates import dupeRemoval

class TestDupeRemoval(unittest.TestCase):
    @patch('dags.src.duplicates.pickle.load')
    @patch('builtins.open')
    def test_dupeRemoval(self, mock_open, mock_pickle_load):
        # Mock DataFrame behavior
        mock_df_instance = MagicMock()
        mock_pickle_load.return_value = mock_df_instance
        
        # Call the function under test
        input_path = 'dummy_input.pkl'  # Provide a dummy input path
        output_path = dupeRemoval(input_path)
        
        # Assert your expectations here
        self.assertIsNotNone(output_path)
        self.assertTrue(os.path.exists(output_path))
        # Add more assertions as needed

if __name__ == "__main__":
    unittest.main()
