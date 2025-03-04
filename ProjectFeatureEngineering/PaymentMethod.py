import pandas as pd
from __init__ import BaseFeatureEngineering

class PaymentMethod(BaseFeatureEngineering):
    def generateFeatures(self):
        """
        Calls all payment method feature functions.
        """
        if self.transactions_data is None or self.feature_data is None:
            return self.feature_data

        # Generate individual features
        self.calculateMostFrequentPaymentMethod()
        self.calculateTotalTransactionsPerMethod()
        self.calculateTotalSpendingPerMethod()
        self.calculatePreferredPaymentMethodByAmount()

        return self.feature_data

    def calculateMostFrequentPaymentMethod(self):
        """
        Identifies the most frequently used payment method per user.
        """
        payment_counts = (
            self.transactions_data.groupby(['client_id', 'use_chip'])
            .size()
            .reset_index(name='count')
            .sort_values(['client_id', 'count'], ascending=[True, False])
            .drop_duplicates(subset=['client_id'], keep='first')
        )

        payment_counts = payment_counts[['client_id', 'use_chip']]
        payment_counts.columns = ['client_id', 'most_frequent_payment_method']

        self.feature_data = self.feature_data.merge(payment_counts, on='client_id', how='left')

    def calculateTotalTransactionsPerMethod(self):
        """
        Computes the number of transactions per payment method for each user.
        """
        transaction_counts = self.transactions_data.groupby(['client_id', 'use_chip']).size().unstack(fill_value=0)
        transaction_counts.columns = [f"total_transactions_{col}" for col in transaction_counts.columns]
        transaction_counts.reset_index(inplace=True)

        self.feature_data = self.feature_data.merge(transaction_counts, on='client_id', how='left')

    def calculateTotalSpendingPerMethod(self):
        """
        Computes the total amount spent per payment method for each user.
        """
        spending_per_method = self.transactions_data.groupby(['client_id', 'use_chip'])['amount'].sum().unstack(fill_value=0)
        spending_per_method.columns = [f"total_spent_{col}" for col in spending_per_method.columns]
        spending_per_method.reset_index(inplace=True)

        self.feature_data = self.feature_data.merge(spending_per_method, on='client_id', how='left')

    def calculatePreferredPaymentMethodByAmount(self):
        """
        Determines the payment method with the highest total spending.
        """
        spending_ranks = (
            self.transactions_data.groupby(['client_id', 'use_chip'])['amount'].sum()
            .reset_index()
            .sort_values(['client_id', 'amount'], ascending=[True, False])
            .drop_duplicates(subset=['client_id'], keep='first')
        )

        spending_ranks = spending_ranks[['client_id', 'use_chip']]
        spending_ranks.columns = ['client_id', 'preferred_payment_method']

        self.feature_data = self.feature_data.merge(spending_ranks, on='client_id', how='left')
