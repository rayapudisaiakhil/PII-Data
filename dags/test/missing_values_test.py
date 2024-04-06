import unittest
from unittest import mock
from unittest.mock import MagicMock, patch, mock_open
import os
import json
import pandas as pd

# Import the function to test
from dags.src.missing_values import naHandler

class TestNaHandler(unittest.TestCase):
    @patch('missing_value.os.getcwd')
    @patch('missing_value.pd.DataFrame')
    @patch('missing_value.pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_naHandler(self, mock_open, mock_pickle_dump, mock_pd_DataFrame, mock_getcwd):
        # Mock the current working directory
        mock_getcwd.return_value = os.path.dirname(os.path.abspath(__file__))

        # Define a mock input data
        input_data = [{'full_text': 'text1'}, {'full_text': 'text2'}]
        # Mock the content of the input JSON file
        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(input_data)

        # Mock the DataFrame constructor
        mock_df_instance = MagicMock(spec=pd.DataFrame)
        mock_pd_DataFrame.return_value = mock_df_instance

        # Mock the XCom value from the previous task
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = os.path.join(os.path.dirname(__file__), 'input.json')

        # Call the function under test
        output_path = naHandler(ti=mock_ti)

        # Assertions
        mock_pd_DataFrame.assert_called_once_with(input_data)
        mock_df_instance.dropna.assert_called_once_with(subset=['full_text'], inplace=True)
        mock_pickle_dump.assert_called_once()

        self.assertEqual(output_path, os.path.join(os.path.dirname(__file__), 'missing_values.pkl'))

    @patch('missing_value.os.getcwd')
    @patch('missing_value.pd.DataFrame')
    @patch('missing_value.pickle.dump')
    @patch('builtins.open', new_callable=mock_open)
    def test_naHandler_with_missing_input(self, mock_open, mock_pickle_dump, mock_pd_DataFrame, mock_getcwd):
        # Mock the current working directory
        mock_getcwd.return_value = os.path.dirname(os.path.abspath(__file__))

        # Mock the XCom value from the previous task to return None
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = None

        # Call the function under test and expect FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            naHandler(ti=mock_ti)

    # Add more test cases as needed...

if __name__ == '__main__':
    unittest.main()
