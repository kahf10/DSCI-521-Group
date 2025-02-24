import pandas as pd
import re


class PreprocessingTransactions:
    def __init__(self, file_path):
        """
        Initializes the Preprocessing class and reads the CSV file.
        """
        self.file_path = file_path
        self.data = None

    def readFile(self):
        """
        Reads the CSV file into a Pandas DataFrame
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("File successfully loaded.")
        except Exception as e:
            print(f"Error loading file: {e}")

    def removeDuplicates(self):
        """
        Removes exact duplicate rows from the dataset.
        """
        before = len(self.data)
        self.data.drop_duplicates(inplace=True)
        after = len(self.data)

        print(f"Removed {before - after} duplicate rows.")

    def dropColumns(self, columnsNames):
        """
        Drops specified columns from the dataset.

        Args:
        columns_to_drop (list): List of column names to drop.
        """
        existing_columns = [columnName for columnName in columnsNames if columnName in self.data.columns]
        self.data.drop(columns=existing_columns, inplace=True, errors='ignore')

        print(f"Dropped columns: {existing_columns}" if existing_columns else "No matching columns found to drop.")

    def removeRowsWithErrors(self):
        """
        Removes rows where the 'errors' column has any value (not null).
        """
        before = len(self.data)
        self.data = self.data[self.data['errors'].isna()]
        after = len(self.data)

        print(f"Removed {before - after} rows with values in the 'errors' column.")

    def separateDateTime(self):
        """
        Splits the 'date' column into separate 'transaction_date' and 'transaction_time' columns.
        """
        # Ensure 'date' is treated as a string
        self.data['date'] = self.data['date'].astype(str)

        # Regex pattern to extract date and time separately
        date_pattern = r"^(\d{4}-\d{2}-\d{2})"
        time_pattern = r"(\d{2}:\d{2}:\d{2})$"

        self.data['transaction_date'] = self.data['date'].apply(
            lambda x: re.search(date_pattern, x).group(1) if re.search(date_pattern, x) else None)
        self.data['transaction_time'] = self.data['date'].apply(
            lambda x: re.search(time_pattern, x).group(1) if re.search(time_pattern, x) else None)

    def cleanAmountColumn(self):
        self.data['amount'] = self.data['amount'].astype(str).apply(lambda x: re.sub(r'[^0-9.-]', '', x)).astype(float)
        print("Removed '$' from amount column.")

    def savePreProcessedDataset(self, output_path):
        """
        Saves the preprocessed dataset to a new CSV file.
        """
        self.data.to_csv(output_path, index=False)
        print(f"Preprocessed dataset successfully saved to: {output_path}")

    def printSummaryOfPreProcessedDataset(self):
        """
        Prints a summary of the preprocessed dataset:
        - Number of rows and columns
        - Column names and data types
        - Number of missing values per column
        - Basic statistical summary for numerical columns
        """
        print("\n Dataset Summary:")
        print(f"Total Rows: {self.data.shape[0]}, Total Columns: {self.data.shape[1]}")

        print("\n Column Information:")
        print(self.data.dtypes)

        print("\n Missing Values Per Column:")
        print(self.data.isnull().sum())

        print("\n Statistical Summary for Numerical Columns:")
        print(self.data.describe())

    def runPipeline(self):
        self.readFile()

        self.removeDuplicates()

        self.dropColumns(['merchant_city', 'merchant_state'])

        self.removeRowsWithErrors()

        self.dropColumns(['errors'])

        self.separateDateTime()

        self.dropColumns(['date'])

        self.cleanAmountColumn()

        self.savePreProcessedDataset("../data/preprocessed_transactions_data.csv")

        self.printSummaryOfPreProcessedDataset()





