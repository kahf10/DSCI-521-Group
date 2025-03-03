import pandas as pd
from __init__ import BaseFeatureEngineering

class TransactionFrequency(BaseFeatureEngineering):
    def generateFeatures(self):
        """
        Calls all transaction frequency & recency feature functions.
        """
        if self.transactions_data is None or self.feature_data is None:
            return self.feature_data

        # Convert transaction_date to datetime format
        self.transactions_data['transaction_date'] = pd.to_datetime(self.transactions_data['transaction_date'])

        # Generate individual features
        self.calculateTotalTransactions()
        self.calculateTransactionDates()
        self.calculateMonthsActive()
        self.calculateTransactionsPerMonth()
        self.calculateDaysSinceLastTransaction()

        return self.feature_data

    def calculateTotalTransactions(self):
        """
        Computes total transactions per user.
        """
        transaction_counts = self.transactions_data.groupby('client_id')['id'].count().reset_index()
        transaction_counts.columns = ['client_id', 'total_transactions']
        self.feature_data = self.feature_data.merge(transaction_counts, on='client_id', how='left')

    def calculateTransactionDates(self):
        """
        Computes first and last transaction dates per user.
        """
        transaction_dates = self.transactions_data.groupby('client_id')['transaction_date'].agg(['min', 'max']).reset_index()
        transaction_dates.columns = ['client_id', 'first_transaction_date', 'last_transaction_date']
        self.feature_data = self.feature_data.merge(transaction_dates, on='client_id', how='left')

    def calculateMonthsActive(self):
        """
        Computes the number of months a user has been active.
        """
        self.feature_data['months_active'] = (
            (self.feature_data['last_transaction_date'] - self.feature_data['first_transaction_date']).dt.days / 30
        ).clip(lower=1)  # Prevent division by zero

    def calculateTransactionsPerMonth(self):
        """
        Computes average transactions per month per user.
        """
        self.feature_data['transactions_per_month'] = self.feature_data['total_transactions'] / self.feature_data['months_active']

    def calculateDaysSinceLastTransaction(self):
        """
        Computes the number of days since the last transaction.
        """
        max_dataset_date = self.transactions_data['transaction_date'].max()
        self.feature_data['days_since_last_transaction'] = (
            max_dataset_date - self.feature_data['last_transaction_date']
        ).dt.days
