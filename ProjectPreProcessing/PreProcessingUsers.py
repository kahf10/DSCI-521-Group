import pandas as pd
from geopy.geocoders import Nominatim
import re

class PreProcessingUsers:
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

    def calculateZipCode(self):

        geolocator = Nominatim(user_agent="geoapiExercises")  # Use geopy for reverse geocoding

        def get_zipcode(lat, lon):
            try:
                location = geolocator.reverse((lat, lon), exactly_one=True, timeout=10)
                if location and 'postcode' in location.raw['address']:
                    return location.raw['address']['postcode']
                return None
            except Exception:
                return None

        print("Converting latitude & longitude to ZIP codes...")
        self.data['zipcode'] = self.data.apply(lambda row: get_zipcode(row['latitude'], row['longitude']), axis=1)
        print("ZIP codes assigned, and lat/lon columns dropped.")

    def savePreProcessedDataset(self, output_path):
        """
        Saves the preprocessed dataset to a new CSV file.
        """
        self.data.to_csv(output_path, index=False)
        print(f"Preprocessed dataset successfully saved to: {output_path}")

    def cleanColumnsWithDollarSign(self, columnsNames):
        for columnName in columnsNames:
            self.data[columnName] = self.data[columnName].astype(str).apply(lambda x: re.sub(r'[^0-9.-]', '', x)).astype(float)
        print("Removed '$' from amount column.")

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

        self.dropColumns(['birth_year', 'address', 'birth_month'])

       # self.calculateZipCode()

        self.dropColumns(['latitude', 'longitude'])

        self.cleanColumnsWithDollarSign(['per_capita_income', 'yearly_income', 'total_debt'])

        self.savePreProcessedDataset("../data/preprocessed_users_data.csv")

        self.printSummaryOfPreProcessedDataset()




