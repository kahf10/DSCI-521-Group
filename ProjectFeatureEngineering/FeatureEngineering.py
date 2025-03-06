import pandas as pd
from ProjectFeatureEngineering.MerchantBehavior import MerchantBehavior
from ProjectFeatureEngineering.PaymentMethod import PaymentMethod
from TransactionFrequency import TransactionFrequency
from SpendingBehavior import SpendingBehavior
from ChurnLabel import ChurnLabel

class FeatureEngineering:
    def __init__(self, transactions_file, users_file):
        self.users_file = users_file
        self.transactions_file = transactions_file
        self.users_data = None
        self.transactions_data = None
        self.feature_data = None  # Final merged dataset

    def readFiles(self):
        """
        Reads the users and transactions CSV files into Pandas DataFrames.
        """
        try:
            self.users_data = pd.read_csv(self.users_file)
            print("Users file successfully loaded. Shape:", self.users_data.shape)
        except Exception as e:
            print(f"Error loading users file: {e}")

        try:
            self.transactions_data = pd.read_csv(self.transactions_file)
            print("Transactions file successfully loaded. Shape:", self.transactions_data.shape)
        except Exception as e:
            print(f"Error loading transactions file: {e}")

    def initializeFeatureDataSet(self):
        """
        Creates a new DataFrame using the existing users' data.
        """
        if self.users_data is not None:
            # Rename id to client_id for merging purposes
            self.users_data.rename(columns={'id': 'client_id'}, inplace=True)
            self.feature_data = self.users_data.copy()
            print("Feature dataset initialized. Shape:", self.feature_data.shape)
        else:
            print("Error: Users data not loaded. Run readFiles() first.")

    def runPipeline(self):
        """
        Runs the entire feature engineering pipeline.
        """
        self.readFiles()
        self.initializeFeatureDataSet()

        self.applyTransactionFeatures()
        self.logCurrentState("After Transaction Frequency")

        self.applySpendingBehaviorFeatures()
        self.logCurrentState("After Spending Behavior")

        self.applyMerchantBehaviorFeatures()
        self.logCurrentState("After Merchant Behavior")

        self.applyPaymentMethodFeatures()
        self.logCurrentState("After Payment Method")

        # Apply churn label based on days since last transaction
        self.applyChurnLabel(threshold_days=180)
        self.logCurrentState("After Churn Label")

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)
        print("Final Feature Engineering Pipeline execution completed.")
        print(self.feature_data.head())

        # Save final features to CSV
        self.feature_data.to_csv("../data/final_features.csv", index=False)
        print("Final features saved to final_features.csv")

    def logCurrentState(self, stage):
        """
        Logs the current state of the feature data.
        """
        print(f"\n--- {stage} ---")
        print("Shape:", self.feature_data.shape)
        print("Columns:", list(self.feature_data.columns))
        if 'churn' in self.feature_data.columns:
            churn_rate = self.feature_data['churn'].mean()
            print(f"Churn rate: {churn_rate:.2f}")
        print("----------------------------\n")

    def applyMerchantBehaviorFeatures(self):
        """
        Apply merchant behavior features.
        """
        print("Applying Merchant Behavior Features")
        merchant_features = MerchantBehavior(self.feature_data, self.transactions_data)
        self.feature_data = merchant_features.generateFeatures()

    def applySpendingBehaviorFeatures(self):
        """
        Apply spending behavior features.
        """
        print("Applying Spending Behavior Features")
        spending_features = SpendingBehavior(self.feature_data, self.transactions_data)
        self.feature_data = spending_features.generateFeatures()

    def applyTransactionFeatures(self):
        """
        Apply transaction frequency features.
        """
        print("Applying Transaction Frequency Features")
        transaction_features = TransactionFrequency(self.feature_data, self.transactions_data)
        self.feature_data = transaction_features.generateFeatures()

    def applyPaymentMethodFeatures(self):
        """
        Apply payment method features.
        """
        print("Applying Payment Method Features")
        payment_features = PaymentMethod(self.feature_data, self.transactions_data)
        self.feature_data = payment_features.generateFeatures()

    def applyChurnLabel(self, threshold_days=180):
        """
        Moves the churn labeling logic to the ChurnLabel module.
        It adds a churn label based on 'days_since_last_transaction' where a user is churned (1)
        if days since last transaction exceed threshold_days, else active (0).
        """
        churn_labeler = ChurnLabel(self.feature_data, threshold_days=threshold_days)
        self.feature_data = churn_labeler.generateChurnLabel()
