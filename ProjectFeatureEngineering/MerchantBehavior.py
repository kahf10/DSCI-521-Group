import pandas as pd
from __init__ import BaseFeatureEngineering
import json

class MerchantBehavior(BaseFeatureEngineering):

    def __init__(self, feature_data, transactions_data, mcc_file="../data/mcc_codes.json"):
        super().__init__(feature_data, transactions_data)
        self.mcc_file = mcc_file
        self.mcc_mapping = self.loadMCCMapping()

    def loadMCCMapping(self):
        """
        Loads the MCC mapping from JSON file.
        """
        try:
            with open(self.mcc_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading MCC codes: {e}")
            return {}

    def generateFeatures(self):
        """
        Calls all merchant behavior feature functions.
        """
        if self.transactions_data is None or self.feature_data is None:
            return self.feature_data

        self.calculateUniqueMerchants()
        self.calculateMostFrequentMerchants()
        self.calculateMostFrequentMerchantCategory()
        self.calculateAvgSpendingPerMerchant()

        return self.feature_data.copy()

    def calculateUniqueMerchants(self):
        """
        Computes the number of unique merchants a user has transacted with
        """
        unique_merchants = self.transactions_data.groupby('client_id')['merchant_id'].nunique().reset_index()
        unique_merchants.columns = ['client_id', 'unique_merchants']
        self.feature_data = self.feature_data.merge(unique_merchants, on='client_id', how='left')

    def calculateMostFrequentMerchants(self):
        """
        Finds the most frequently used merchant for each user
        """
        top_merchants = (
            self.transactions_data.groupby(['client_id', 'merchant_id'])
            .size()
            .reset_index(name='merchant_count')
            .sort_values(['client_id', 'merchant_count'], ascending=[True, False])
            .drop_duplicates(subset=['client_id'], keep='first')
        )
        top_merchants = top_merchants[['client_id', 'merchant_id']]
        top_merchants.columns = ['client_id', 'most_frequent_merchant']
        self.feature_data = self.feature_data.merge(top_merchants, on='client_id', how='left')

    def calculateMostFrequentMerchantCategory(self):
        """
        Get the MCC category of the most frequently used merchant using MCC codes.
        """
        merchant_mcc = self.transactions_data[['merchant_id', 'mcc']].drop_duplicates()
        self.feature_data = self.feature_data.merge(
            merchant_mcc, left_on='most_frequent_merchant', right_on='merchant_id', how='left'
        )

        self.feature_data['mcc'] = self.feature_data['mcc'].astype(str).str.split('.').str[0]
        self.feature_data['most_frequent_merchant_category'] = self.feature_data['mcc'].map(self.mcc_mapping)

        self.feature_data['most_frequent_merchant_category'] = self.feature_data[
            'most_frequent_merchant_category'].fillna("Unknown")
        self.feature_data.drop(columns=['merchant_id', 'mcc'], inplace=True)

    def calculateAvgSpendingPerMerchant(self):
        """
        Computes the average spending per merchant for each user.
        """
        spending_per_merchant = self.transactions_data.groupby(['client_id'])['amount'].sum().reset_index()
        spending_per_merchant.columns = ['client_id', 'total_spending']

        self.feature_data = self.feature_data.merge(spending_per_merchant, on='client_id', how='left',
                                                    suffixes=('', '_new'))

        if 'total_spending_new' in self.feature_data.columns:
            self.feature_data['total_spending'] = self.feature_data['total_spending_new']
            self.feature_data.drop(columns=['total_spending_new'], inplace=True)

        self.feature_data['avg_spending_per_merchant'] = (
                self.feature_data['total_spending'] / self.feature_data['unique_merchants']
        )

        return self.feature_data


