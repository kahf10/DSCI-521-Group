import os
import pandas as pd
import numpy as np

# Define file paths
TRAIN_FILE = "../data/train_data.csv"
TEST_FILE = "../data/test_data.csv"
TARGET_COLUMN = "churn_label"

def check_file_existence():
    """Check if train and test files exist."""
    missing_files = [file for file in [TRAIN_FILE, TEST_FILE] if not os.path.exists(file)]
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    print("✅ Both train and test files exist.")
    return True

def load_data():
    """Load CSV files into pandas DataFrames."""
    try:
        train_data = pd.read_csv(TRAIN_FILE)
        test_data = pd.read_csv(TEST_FILE)
        print("✅ Successfully loaded train and test data.")
        return train_data, test_data
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None

def check_missing_values(data, dataset_name):
    """Check for missing values in the dataset."""
    missing = data.isnull().sum()
    total_missing = missing.sum()
    if total_missing > 0:
        print(f"⚠️ {dataset_name} has {total_missing} missing values:")
        print(missing[missing > 0])
    else:
        print(f"✅ No missing values in {dataset_name}.")

def check_column_consistency(train, test):
    """Ensure train and test have the same columns."""
    if set(train.columns) != set(test.columns):
        print("❌ Train and test columns do not match!")
        print(f"Train columns: {set(train.columns)}")
        print(f"Test columns: {set(test.columns)}")
        return False
    print("✅ Train and test columns match.")
    return True

def check_target_variable(data):
    """Ensure target column exists and contains only valid values (0 or 1)."""
    if TARGET_COLUMN not in data.columns:
        print(f"❌ Target column '{TARGET_COLUMN}' not found in dataset!")
        return False
    unique_values = data[TARGET_COLUMN].unique()
    if set(unique_values) <= {0, 1}:
        print(f"✅ Target column '{TARGET_COLUMN}' is correctly formatted.")
        return True
    else:
        print(f"❌ Target column '{TARGET_COLUMN}' contains invalid values: {unique_values}")
        return False

def check_data_types(data, dataset_name):
    """Check if all feature columns are numeric."""
    non_numeric = data.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"⚠️ {dataset_name} contains non-numeric columns: {list(non_numeric)}")
    else:
        print(f"✅ All columns in {dataset_name} are numeric.")

def check_duplicates(data, dataset_name):
    """Check for duplicate rows in the dataset."""
    duplicates = data.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️ {dataset_name} has {duplicates} duplicate rows.")
    else:
        print(f"✅ No duplicate rows in {dataset_name}.")

def check_class_balance(data):
    """Check if the target variable is imbalanced."""
    counts = data[TARGET_COLUMN].value_counts(normalize=True)
    print("\nClass Distribution in Training Data:")
    print(counts)
    if counts.min() < 0.1:
        print("⚠️ Warning: Dataset is highly imbalanced.")
    else:
        print("✅ Class distribution looks reasonable.")

def check_basic_statistics(data, dataset_name):
    """Display basic statistics of numeric columns."""
    print(f"\n📊 Basic statistics for {dataset_name}:")
    print(data.describe())

def run_tests():
    """Run all dataset validation checks."""
    if not check_file_existence():
        return

    train_data, test_data = load_data()
    if train_data is None or test_data is None:
        return

    print("\n🔎 Running dataset checks...\n")

    check_missing_values(train_data, "Train Data")
    check_missing_values(test_data, "Test Data")
    check_column_consistency(train_data, test_data)
    check_target_variable(train_data)
    check_target_variable(test_data)
    check_data_types(train_data, "Train Data")
    check_data_types(test_data, "Test Data")
    check_duplicates(train_data, "Train Data")
    check_duplicates(test_data, "Test Data")
    check_class_balance(train_data)
    check_basic_statistics(train_data, "Train Data")
    check_basic_statistics(test_data, "Test Data")

    print("\n✅ Dataset validation completed.")

if __name__ == "__main__":
    run_tests()
