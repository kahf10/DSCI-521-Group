import pandas as pd
from __init__ import BaseFeatureEngineering

class SpendingBehavior(BaseFeatureEngineering):
    def generateFeatures(self):
        """
        Calls all spending behavior feature functions.
        """
        if self.transactions_data is None or self.feature_data is None:
            return self.feature_data

        # Convert amount to numeric format
        self.transactions_data['amount'] = pd.to_numeric(self.transactions_data['amount'], errors='coerce')

        # Generate individual features
        self.calculateTotalSpending()
        self.calculateTotalRefunds()
        self.calculateAverageTransactionAmount()
        self.calculateMedianTransactionAmount()
        self.calculateMaxTransactionAmount()
        self.calculateMinTransactionAmount()
        self.calculateSpendingVariability()

        return self.feature_data

    def calculateTotalSpending(self):
        """
        Computes total amount spent per user.
        """
        total_spending = self.transactions_data.groupby('client_id')['amount'].sum().reset_index()
        total_spending.columns = ['client_id', 'total_spending']
        self.feature_data = self.feature_data.merge(total_spending, on='client_id', how='left')

    def calculateTotalRefunds(self):
        """
        Computes total refunds (negative transactions) per user.
        """
        refunds = self.transactions_data[self.transactions_data['amount'] < 0].groupby('client_id')['amount'].sum().reset_index()
        refunds.columns = ['client_id', 'total_refunds']
        refunds['total_refunds'] = refunds['total_refunds'].abs()  # Convert to positive
        self.feature_data = self.feature_data.merge(refunds, on='client_id', how='left')
        self.feature_data['total_refunds'] = self.feature_data['total_refunds'].fillna(0)

    def calculateAverageTransactionAmount(self):
        """
        Computes the average transaction amount per user.
        """
        avg_spending = self.transactions_data.groupby('client_id')['amount'].mean().reset_index()
        avg_spending.columns = ['client_id', 'avg_transaction_amount']
        self.feature_data = self.feature_data.merge(avg_spending, on='client_id', how='left')

    def calculateMedianTransactionAmount(self):
        """
        Computes the median transaction amount per user.
        """
        median_spending = self.transactions_data.groupby('client_id')['amount'].median().reset_index()
        median_spending.columns = ['client_id', 'median_transaction_amount']
        self.feature_data = self.feature_data.merge(median_spending, on='client_id', how='left')

    def calculateMaxTransactionAmount(self):
        """
        Computes the maximum transaction amount per user.
        """
        max_spending = self.transactions_data.groupby('client_id')['amount'].max().reset_index()
        max_spending.columns = ['client_id', 'max_transaction_amount']
        self.feature_data = self.feature_data.merge(max_spending, on='client_id', how='left')

    def calculateMinTransactionAmount(self):
        """
        Computes the smallest positive transaction amount per user.
        """
        min_spending = self.transactions_data[self.transactions_data['amount'] > 0].groupby('client_id')['amount'].min().reset_index()
        min_spending.columns = ['client_id', 'min_transaction_amount']
        self.feature_data = self.feature_data.merge(min_spending, on='client_id', how='left')

    def calculateSpendingVariability(self):
        """
        Computes the standard deviation of transaction amounts per user.
        """
        spending_std = self.transactions_data.groupby('client_id')['amount'].std().reset_index()
        spending_std.columns = ['client_id', 'spending_variability']
        self.feature_data = self.feature_data.merge(spending_std, on='client_id', how='left')
