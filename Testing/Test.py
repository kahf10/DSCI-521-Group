import pandas as pd

# transactions = pd.read_csv('../data/preprocessed_transactions_data.csv')
# users = pd.read_csv('../data/preprocessed_users_data.csv')

pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_colwidth', None)  # Remove column width restriction
#
# print(transactions.sample(10))
# print(users.sample(10))
#
# print(transactions['use_chip'].unique())

featureData = pd.read_csv('../data/feature_data.csv')
print(featureData.head())

