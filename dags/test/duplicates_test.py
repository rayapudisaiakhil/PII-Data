import os
import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
import pickle

# Import the function to test
from dags.src.duplicates import dupeRemoval

class TestDupeRemoval(unittest.TestCase):
    @patch('dags.src.duplicates.os.getcwd')
    @patch('dags.src.duplicates.os.path.exists')
    @patch('dags.src.duplicates.open', new_callable=mock_open)
    @patch('dags.src.duplicates.pickle.load')
    @patch('dags.src.duplicates.pickle.dump')
    @patch('dags.src.duplicates.ti.xcom_pull')
    def test_dupe_removal(self, mock_xcom_pull, mock_pickle_load, mock_pickle_dump, mock_file, mock_exists, mock_getcwd):
        # Setup
        mock_getcwd.return_value = '/fake/directory'
        input_path = '/fake/directory/dags/processed/missing_values.pkl'
        output_path = '/fake/directory/dags/processed/duplicate_removal.pkl'
        mock_xcom_pull.return_value = input_path
        mock_exists.return_value = True

        # Create a sample DataFrame
        data = {
            'full_text': ['text1', 'text1', 'text2', 'text3', 'text3'],
            'other_column': [1, 1, 2, 3, 3]
        }
        df = pd.DataFrame(data)
        df_expected = df.drop_duplicates(subset=['full_text'])

        mock_pickle_load.return_value = df

        # Call function
        result_path = dupeRemoval(ti=MagicMock())

        # Assertions
        mock_file.assert_called_with(output_path, 'wb')  # Check if file is opened in write mode
        mock_pickle_dump.assert_called_once()  # Check if pickle.dump was called
        args, kwargs = mock_pickle_dump.call_args
        self.assertTrue((args[0] == df_expected).all().all())  # Compare dataframes
        self.assertEqual(result_path, output_path)  # Check if the correct output path is returned
        print("All duplicate entries are removed and data is saved successfully.")

if __name__ == '__main__':
    unittest.main()
