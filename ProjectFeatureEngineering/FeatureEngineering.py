import pandas as pd
from TransactionFrequency import TransactionFrequency
from SpendingBehavior import SpendingBehavior


class FeatureEngineering:
    def __init__(self, transactions_file, users_file):
        self.users_file = users_file
        self.transactions_file = transactions_file
        self.users_data = None
        self.transactions_data = None
        self.feature_data = None  # Placeholder for our final dataset

    def readFiles(self):
        """
        Reads the users and transactions CSV files into Pandas DataFrames.
        """
        try:
            self.users_data = pd.read_csv(self.users_file)
            print("Users file successfully loaded.")
        except Exception as e:
            print(f"Error loading users file: {e}")

        try:
            self.transactions_data = pd.read_csv(self.transactions_file)
            print("Transactions file successfully loaded.")
        except Exception as e:
            print(f"Error loading transactions file: {e}")

    def initializeFeatureDataSet(self):
        """
        Creates a new DataFrame using the existing users' data.
        This will serve as the base for adding transaction-based features.
        """
        if self.users_data is not None:
            self.users_data.rename(columns={'id': 'client_id'}, inplace=True)
            self.feature_data = self.users_data.copy()
            print("Feature dataset initialized from users data.")
        else:
            print("Error: Users data not loaded. Run readFiles() first.")

    def runPipeline(self):
        """
        Runs the feature engineering pipeline.
        """
        self.readFiles()
        self.initializeFeatureDataSet()

        # Apply transaction frequency features
        print("Applying transaction frequency features")
        transaction_features = TransactionFrequency(self.feature_data, self.transactions_data)
        self.feature_data = transaction_features.generateFeatures()

        # Apply spending behavior features
        print("Applying spending behavior features")
        spending_features = SpendingBehavior(self.feature_data, self.transactions_data)
        self.feature_data = spending_features.generateFeatures()


        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.max_colwidth', None)  # Remove column width restriction

        print("Feature Engineering Pipeline execution completed.")
        print(self.feature_data.head())



