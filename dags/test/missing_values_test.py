import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, mock_open
import os
import json
import pandas as pd

# Import the function to test
from dags.src.missing_values import naHandler


class TestNaHandler(unittest.TestCase):
    @patch('dags.src.missing_values.pd.DataFrame')
    @patch('dags.src.missing_values.pickle.dump')
    @patch('builtins.open')
    def test_naHandler(self, mock_open, mock_pickle_dump, mock_pd_DataFrame):
        # Mock the content of the input JSON file
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps([{'full_text': 'text1'}, {'full_text': 'text2'}])

        # Call the function under test
        output_path = naHandler(ti=MagicMock())

        # Assertions
        self.assertTrue(mock_pd_DataFrame.called)
        self.assertTrue(mock_pickle_dump.called)
        self.assertTrue(output_path.endswith('missing_values.pkl'))

if __name__ == '__main__':
    unittest.main()
