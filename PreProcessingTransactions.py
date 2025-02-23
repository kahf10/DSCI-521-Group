import pandas as pd


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
        if 'errors' in self.data.columns:
            before = len(self.data)
            self.data = self.data[self.data['errors'].isna()]
            after = len(self.data)

            print(f"Removed {before - after} rows with values in the 'errors' column.")
        else:
            print("Column 'errors' not found in dataset.")

    def runPipeline(self):
        self.readFile()

        self.removeDuplicates()

        self.dropColumns(['merchant_city', 'merchant_state'])

        self.removeRowsWithErrors()

