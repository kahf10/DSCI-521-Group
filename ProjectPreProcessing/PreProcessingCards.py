import re
import pandas as pd


class PreProcessingCards:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def readFile(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("FIle successfully loaded.")
        except Exception as e:
            print(f"Error loading file: {e}")

    def removeDuplicates(self):
        before = len(self.data)
        self.data.drop_duplicates(inplace=True)
        after = len(self.data)

        print(f"Removed {before - after} duplicate rows.")

    def dropColumns(self, columnsNames):
        existing_columns = [columnName for columnName in columnsNames if columnName in self.data.columns]
        self.data.drop(columns=existing_columns, inplace=True, errors='ignore')

        print(f"Dropped columns: {existing_columns}" if existing_columns else "No matching columns found to drop.")

    def cleanCreditLimitColumn(self):
        self.data['amount'] = self.data['amount'].astype(str).apply(lambda x: re.sub(r'[^0-9.-]', '', x)).astype(float)
        print("Removed '$' from amount column.")


    def runPipeline(self):
        # Yet to implement
        return
