import os
import pickle
import unittest
from unittest.mock import MagicMock, patch, mock_open
from dags.src.duplicates import dupeRemoval

class TestDupeRemoval(unittest.TestCase):
    @patch('dags.src.duplicates.pickle')
    @patch('dags.src.duplicates.open')
    def test_dupeRemoval_function_call(self, mock_open, mock_pickle):
        # Call the function under test
        dupeRemoval(ti=None)
        
        # Assertions
        mock_open.assert_called_once()
        mock_pickle.load.assert_called_once()

if __name__ == '__main__':
    unittest.main()
