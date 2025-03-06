import pandas as pd


class ChurnLabel:
    def __init__(self, feature_data, threshold_days=180):
        """
        Initializes the churn labeling process.

        :param feature_data: A DataFrame containing the features, including 'days_since_last_transaction'.
        :param threshold_days: The threshold (in days) beyond which a user is considered churned.
        """
        self.feature_data = feature_data
        self.threshold_days = threshold_days

    def generateChurnLabel(self):
        """
        Adds a churn label to the DataFrame based on the 'days_since_last_transaction' column.
        A user is labeled churned (1) if days_since_last_transaction > threshold_days, else active (0).

        :return: The DataFrame with an added 'churn' column.
        """
        if 'days_since_last_transaction' in self.feature_data.columns:
            self.feature_data['churn'] = self.feature_data['days_since_last_transaction'].apply(
                lambda x: 1 if x > self.threshold_days else 0
            )
            print(f"Churn label applied using threshold_days = {self.threshold_days}")
        else:
            print("Error: 'days_since_last_transaction' not found. Check TransactionFrequency features.")
        return self.feature_data
