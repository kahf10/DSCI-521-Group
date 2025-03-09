import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

CORRELATION_THRESHOLD = 0.8
UNNECESSARY_VARIABLES = ['client_id', 'days_since_last_transaction', 'last_transaction_date', 'transactions_per_month',
                             'historical_avg_transactions', 'recent_avg_transactions'
]

MULTICOLLINEAR_COLUMNS_TO_DROP = [
    'per_capita_income', 'transactions_per_month', 'avg_spending_per_merchant',
    'total_spent_chip Transaction', 'total_spent_Swipe Transaction', 'total_spent_Online Transaction',
    "total_transactions_Chip Transaction",
    "total_transactions_Online Transaction", "total_transactions_Swipe Transaction"
]

NUMERICAL_VARIABLES = [
    "current_age", "retirement_age", "yearly_income", "total_debt", "credit_score",
    "num_credit_cards", "total_refunds", "avg_transaction_amount", "median_transaction_amount",
    "max_transaction_amount", "min_transaction_amount", "spending_variability",
    "unique_merchants", "total_transactions" , "total_spending", "days_since_first_transaction",
]
CATEGORICAL_VARIABLES = [
    "gender", "most_frequent_merchant_category", 'most_frequent_payment_method', 'preferred_payment_method'
]

class FeatureEngineeringProcessing:
    def __init__(self, feature_data):
        """
        Initializes the feature processing class.
        """
        self.processed_feature_data = feature_data
        self.train_data = None
        self.test_data = None

    def dropColumns(self, columns_to_drop):
        """
        Drops variables that are not useful for modeling or are redundant.
        """
        self.processed_feature_data.drop(columns=columns_to_drop, inplace=True, errors='ignore')
        print("Removed Unnecessary Columns")

    def identifyCollinearColumns(self):
        """
        Identifies highly correlated columns
        """
        # Compute correlation matrix
        corr_matrix = self.processed_feature_data.select_dtypes(include=[np.number]).corr().abs()

        # Identify pairs of highly correlated features
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Print feature pairs with correlation greater than the threshold
        for col in upper_tri.columns:
            high_corr = upper_tri[col][upper_tri[col] > CORRELATION_THRESHOLD]
            for index, value in high_corr.items():
                print(f"- {col} is correlated with {index} (Correlation: {value:.2f})")

        print("Identified highly correlated columns")

    def splitDataset(self, test_size = 0.2, random_state = 42):
        """
        Splits the dataset into train and test sets with equal distribution
        """

        self.train_data, self.test_data = train_test_split(
            self.processed_feature_data,
            test_size=test_size,
            random_state=random_state,
            stratify=self.processed_feature_data['churn_label'],
        )

        print("Completed splitting dataset")
        print(f"Train Set size: {self.train_data.shape}")
        print(f"Test Set size: {self.test_data.shape}")

    def standardizeNumericalVariables(self):
        """
        Standardizes numerical variables
        """
        scaler = StandardScaler()

        # Fill missing values before standardization
        self.train_data[NUMERICAL_VARIABLES] = self.train_data[NUMERICAL_VARIABLES].fillna(
            self.train_data[NUMERICAL_VARIABLES].median())
        self.test_data[NUMERICAL_VARIABLES] = self.test_data[NUMERICAL_VARIABLES].fillna(
            self.train_data[NUMERICAL_VARIABLES].median())  # Use train median

        # Fit only on the train dataset
        self.train_data[NUMERICAL_VARIABLES] = scaler.fit_transform(self.train_data[NUMERICAL_VARIABLES])

        # Apply the same transformation on test data without fitting
        self.test_data[NUMERICAL_VARIABLES] = scaler.transform(self.test_data[NUMERICAL_VARIABLES])

        print("Standardization Numerical variables")

    def oneHotEncodeCategoricalVariables(self):
        """
        One-hot encode low cardinality categorical variables
        """
        # Fill in missing values
        for col in CATEGORICAL_VARIABLES:
            self.train_data.loc[:, col] = self.train_data[col].fillna("Unknown")
            self.test_data.loc[:, col] = self.test_data[col].fillna("Unknown")

        # Create OHE
        ohe_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        train_encoded = ohe_encoder.fit_transform(self.train_data[CATEGORICAL_VARIABLES])
        test_encoded = ohe_encoder.transform(self.test_data[CATEGORICAL_VARIABLES])

        # Convert to df
        ohe_columns = ohe_encoder.get_feature_names_out(CATEGORICAL_VARIABLES)
        train_encoded_df = pd.DataFrame(train_encoded, columns=ohe_columns, index = self.train_data.index)
        test_encoded_df = pd.DataFrame(test_encoded, columns=ohe_columns, index = self.test_data.index)

        # Drop original categorical columns
        self.train_data.drop(columns=CATEGORICAL_VARIABLES, inplace=True)
        self.test_data.drop(columns=CATEGORICAL_VARIABLES, inplace=True)

        self.train_data = pd.concat([self.train_data, train_encoded_df], axis=1)
        self.test_data = pd.concat([self.test_data, test_encoded_df], axis=1)

        print("Completed One-Hot Encoding Categorical Variables")

    def frequencyEncodeHighCardinalityCategoricalVariables(self):
        """
        Applies frequency encoding to most_frequent_merchant columns
        """
        merchant_counts = self.train_data["most_frequent_merchant"].value_counts().to_dict()

        self.train_data["most_frequent_merchant"] = self.train_data["most_frequent_merchant"].map(merchant_counts)
        self.test_data["most_frequent_merchant"] = self.test_data["most_frequent_merchant"].map(merchant_counts).fillna(0)

        print("Completed frequency encoding")

    def encodeFirstTransactionDate(self):
        """Converts first transaction date to number of days since first transaction."""
        print("Encoding first transaction date...")

        # Convert to datetime
        self.train_data["first_transaction_date"] = pd.to_datetime(
            self.train_data["first_transaction_date"], errors="coerce"
        )
        self.test_data["first_transaction_date"] = pd.to_datetime(
            self.test_data["first_transaction_date"], errors="coerce"
        )

        # Get the minimum
        min_date = self.train_data["first_transaction_date"].min()

        # Convert to numerical feature (days since first transaction)
        self.train_data["days_since_first_transaction"] = (
                self.train_data["first_transaction_date"] - min_date
        ).dt.days

        self.test_data["days_since_first_transaction"] = (
                self.test_data["first_transaction_date"] - min_date
        ).dt.days

        # Drop the original date column
        self.train_data.drop(columns=["first_transaction_date"], inplace=True)
        self.test_data.drop(columns=["first_transaction_date"], inplace=True)

        print("Completed encoding first transaction date.")

    def saveProcessedData(self):
        train_path = "../data/train_data.csv"
        test_path = "../data/test_data.csv"

        self.train_data.to_csv(train_path, index=False)
        self.test_data.to_csv(test_path, index=False)

        print("Saved final training and test sets")

    def runPipeline(self):
        """
        Executes the feature processing pipeline.
        """
        print("-" * 100)
        # Drop redundant or unnecessary columns or columns used to calculate churn label
        self.dropColumns(UNNECESSARY_VARIABLES)

        print("-" * 100)
        # Identify Collinear Columns
        self.identifyCollinearColumns()

        print("-" * 100)
        # Drop collinear columns
        self.dropColumns(MULTICOLLINEAR_COLUMNS_TO_DROP)

        print("-" * 100)
        # Split dataset into training and test
        self.splitDataset()

        print("-" * 100)
        # Encode first transaction date
        self.encodeFirstTransactionDate()

        print("-" * 100)
        # Standardize numerical variables
        self.standardizeNumericalVariables()

        print("-" * 100)
        # One hot encode categorical variables
        self.oneHotEncodeCategoricalVariables()

        print("-" * 100)
        # Encode most_frequent_merchant_column
        self.frequencyEncodeHighCardinalityCategoricalVariables()

        print("-" * 100)
        # Save training and testing data in files
        self.saveProcessedData()

        print("Completed feature processing pipeline.")

        return self.processed_feature_data  # Returning the processed data for now
