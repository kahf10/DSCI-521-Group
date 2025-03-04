import pandas as pd
import re

class PreProcessingCards:
    def __init__(self, file_path):
        """
        Initializes the PreProcessingCards class and reads the CSV file.
        """
        self.file_path = file_path
        self.data = None

    def readFile(self):
        """
        Reads the CSV file into a Pandas DataFrame.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("File successfully loaded.")
        except Exception as e:
            print(f"Error loading file: {e}")

    def removeDuplicates(self):
        """
        Removes duplicate rows from the dataset.
        """
        before = len(self.data)
        self.data.drop_duplicates(inplace=True)
        after = len(self.data)
        print(f"Removed {before - after} duplicate rows.")

    def dropColumns(self, columnsNames):
        """
        Drops specified columns from the dataset.

        Args:
            columnsNames (list): List of column names to drop.
        """
        existing_columns = [col for col in columnsNames if col in self.data.columns]
        self.data.drop(columns=existing_columns, inplace=True, errors='ignore')
        print(f"Dropped columns: {existing_columns}" if existing_columns else "No matching columns found to drop.")

    def cleanCreditLimitColumn(self):
        """
        Cleans the 'credit_limit' column by removing non-numeric characters and converting to float.
        """
        if 'credit_limit' in self.data.columns:
            self.data['credit_limit'] = self.data['credit_limit'].astype(str) \
                .apply(lambda x: re.sub(r'[^0-9.-]', '', x)).astype(float)
            print("Cleaned 'credit_limit' column.")
        else:
            print("Column 'credit_limit' not found.")

    def cleanDateColumns(self):
        """
        Converts date columns 'expires' and 'acct_open_date' to datetime objects.
        Expects format MM/YYYY for both columns.
        """
        if 'expires' in self.data.columns:
            self.data['expires'] = pd.to_datetime(self.data['expires'], format='%m/%Y', errors='coerce')
            print("Converted 'expires' column to datetime.")
        else:
            print("Column 'expires' not found.")

        if 'acct_open_date' in self.data.columns:
            self.data['acct_open_date'] = pd.to_datetime(self.data['acct_open_date'], format='%m/%Y', errors='coerce')
            print("Converted 'acct_open_date' column to datetime.")
        else:
            print("Column 'acct_open_date' not found.")

    def normalizeBooleanColumns(self):
        """
        Normalizes boolean columns by converting YES/NO values to True/False.
        Applies to 'has_chip' and 'card_on_dark_web'.
        """
        if 'has_chip' in self.data.columns:
            self.data['has_chip'] = self.data['has_chip'].str.strip().str.upper().map({'YES': True, 'NO': False})
            print("Normalized 'has_chip' column to boolean.")
        else:
            print("Column 'has_chip' not found.")

        if 'card_on_dark_web' in self.data.columns:
            self.data['card_on_dark_web'] = self.data['card_on_dark_web'].str.strip().str.upper().map({'YES': True, 'NO': False})
            print("Normalized 'card_on_dark_web' column to boolean.")
        else:
            print("Column 'card_on_dark_web' not found.")

    def savePreProcessedDataset(self, output_path):
        """
        Saves the preprocessed dataset to a new CSV file.
        """
        try:
            self.data.to_csv(output_path, index=False)
            print(f"Preprocessed dataset successfully saved to: {output_path}")
        except Exception as e:
            print(f"Error saving dataset: {e}")

    def printSummaryOfPreProcessedDataset(self):
        """
        Prints a summary of the preprocessed dataset:
         - Total number of rows and columns
         - Data types of each column
         - Missing values per column
         - Statistical summary for numerical columns
        """
        print("\nDataset Summary:")
        print(f"Total Rows: {self.data.shape[0]}, Total Columns: {self.data.shape[1]}")
        print("\nColumn Information:")
        print(self.data.dtypes)
        print("\nMissing Values Per Column:")
        print(self.data.isnull().sum())
        print("\nStatistical Summary for Numerical Columns:")
        print(self.data.describe())

    def runPipeline(self):
        """
        Executes the full preprocessing pipeline for card data:
          1. Reads the CSV file.
          2. Removes duplicate rows.
          3. (Optional) Drops unnecessary columns. [Commented out for now]
          4. Cleans the 'credit_limit' column.
          5. Converts date columns to datetime.
          6. Normalizes boolean columns.
          7. Saves the preprocessed dataset.
          8. Prints a summary of the preprocessed dataset.
        """
        self.readFile()
        if self.data is None:
            print("Pipeline aborted due to file read error.")
            return

        self.removeDuplicates()

        self.dropColumns(['cvv', 'card_number'])

        self.cleanCreditLimitColumn()
        self.cleanDateColumns()
        self.normalizeBooleanColumns()

        self.savePreProcessedDataset("../data/preprocessed_cards_data.csv")
        self.printSummaryOfPreProcessedDataset()

