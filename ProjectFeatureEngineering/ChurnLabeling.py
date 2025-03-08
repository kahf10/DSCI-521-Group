import pandas as pd
from __init__ import BaseFeatureEngineering

CHURN_CONDITION_THRESHOLD = 1.0  # Adjust this if needed
CHURN_CONDITION_TIME_THRESHOLD = 6  # In months

class ChurnLabeling(BaseFeatureEngineering):

    def generateFeatures(self):
        """
        Generates churn labels based on a significant decline in user activity over time.
        """
        if self.transactions_data is None or self.feature_data is None:
            print("Error: Missing data. Ensure transactions and user data are loaded.")
            return None

        # Convert transaction_date to datetime
        self.transactions_data['transaction_date'] = pd.to_datetime(self.transactions_data['transaction_date'])

        # Compute transaction counts per month per user
        self.transactions_data['transaction_month'] = self.transactions_data['transaction_date'].dt.to_period('M')
        monthly_activity = self.transactions_data.groupby(['client_id', 'transaction_month']).size().reset_index(name='transaction_count')

        # Compute historical and recent average transactions
        churn_data = self.computeChurnLabels(monthly_activity)

        self.feature_data = churn_data.copy()

        print("Churn labels successfully generated.")

        return self.feature_data

    def computeChurnLabels(self, monthly_activity):
        """
        Computes churn labels based on historical and recent transaction averages.
        """
        latest_date = self.transactions_data['transaction_date'].max()
        churn_cutoff_date = latest_date - pd.DateOffset(months=CHURN_CONDITION_TIME_THRESHOLD)

        # Historical transactions (excluding last 6 months)
        historical_activity = monthly_activity[monthly_activity['transaction_month'] < churn_cutoff_date.to_period('M')]
        historical_avg_transactions = historical_activity.groupby('client_id')['transaction_count'].mean().reset_index()
        historical_avg_transactions.columns = ['client_id', 'historical_avg_transactions']

        # Recent transactions (last 6 months)
        recent_activity = monthly_activity[monthly_activity['transaction_month'] >= churn_cutoff_date.to_period('M')]
        recent_avg_transactions = recent_activity.groupby('client_id')['transaction_count'].mean().reset_index()
        recent_avg_transactions.columns = ['client_id', 'recent_avg_transactions']

        # Merge historical and recent averages
        churn_data = self.feature_data.merge(historical_avg_transactions, on='client_id', how='left')
        churn_data = churn_data.merge(recent_avg_transactions, on='client_id', how='left')

        # Handle missing values (users with no recent transactions are considered churned)
        churn_data['recent_avg_transactions'] = churn_data['recent_avg_transactions'].fillna(0)
        churn_data['historical_avg_transactions'] = churn_data['historical_avg_transactions'].fillna(1)  # Avoid division by zero

        # Define churn condition: recent activity is significantly lower than historical
        churn_data['churn_label'] = (churn_data['recent_avg_transactions'] / churn_data['historical_avg_transactions'] < CHURN_CONDITION_THRESHOLD).astype(int)

        return churn_data  # Return updated dataset
