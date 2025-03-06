import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class FeatureAnalysisPipeline:
    def __init__(self, user_file_path, card_file_path, transaction_file_path,
                 output_file_path="../data/combined_data.csv"):
        """
        Initializes the pipeline with file paths for preprocessed users, cards, and transactions,
        and an optional path to save the final cleaned dataset.
        """
        self.user_file_path = user_file_path
        self.card_file_path = card_file_path
        self.transaction_file_path = transaction_file_path
        self.output_file_path = output_file_path

        self.df = None  # Original merged DataFrame
        self.cleaned_df = None  # DataFrame after removing unwanted columns

    def load_and_merge_data(self):
        """
        Loads the CSV files and merges them.
        """
        user_data = pd.read_csv(self.user_file_path)
        card_data = pd.read_csv(self.card_file_path)
        transaction_data = pd.read_csv(self.transaction_file_path)

        # Rename user identifier if necessary
        if "client_id" not in user_data.columns and "id" in user_data.columns:
            user_data = user_data.rename(columns={'id': 'client_id'})

        # Aggregate card data
        card_agg = card_data.groupby('client_id').agg(
            total_credit_limit=('credit_limit', 'sum'),
            num_cards=('id', 'count')
        ).reset_index()

        # Aggregate transaction data
        transaction_agg = transaction_data.groupby('client_id').agg(
            total_transaction_amount=('amount', 'sum'),
            num_transactions=('id', 'count')
        ).reset_index()

        # Merge
        merged_df = pd.merge(user_data, card_agg, on="client_id", how="left")
        merged_df = pd.merge(merged_df, transaction_agg, on="client_id", how="left")

        # Fill missing values for aggregated columns
        agg_columns = ["total_credit_limit", "num_cards", "total_transaction_amount", "num_transactions"]
        merged_df[agg_columns] = merged_df[agg_columns].fillna(0)

        # Convert key columns to numeric
        for col in ["total_debt", "yearly_income", "credit_score"]:
            if col in merged_df.columns:
                merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0)

        self.df = merged_df

    def remove_unwanted_columns(self, correlation_threshold=0.85):
        """
        1. Removes ID-like columns (e.g., 'client_id').
        2. Removes near-constant columns (low variance).
        3. Removes one of each pair of columns that are highly correlated above correlation_threshold.
        """
        # 1. Drop ID-like columns
        drop_cols = []
        for col in self.df.columns:
            # If a column is named 'client_id' or obviously an ID field, remove it
            if 'id' in col.lower() and col.lower() != 'credit_score':
                drop_cols.append(col)

        df_cleaned = self.df.drop(columns=drop_cols, errors='ignore')

        # 2. Remove near-constant columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        variances = df_cleaned[numeric_cols].var()
        near_constant = variances[variances < 1e-9].index.tolist()  # Adjust threshold if needed
        if near_constant:
            df_cleaned = df_cleaned.drop(columns=near_constant, errors='ignore')
            print(f"Removed near-constant columns: {near_constant}")

        # 3. Remove highly correlated columns
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        corr_matrix = df_cleaned[numeric_cols].corr().abs()

        cols_to_drop = set()
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    colname1 = numeric_cols[i]
                    colname2 = numeric_cols[j]
                    # Drop the column with the higher average correlation
                    avg_corr1 = corr_matrix[colname1].mean()
                    avg_corr2 = corr_matrix[colname2].mean()
                    if avg_corr1 > avg_corr2:
                        cols_to_drop.add(colname1)
                    else:
                        cols_to_drop.add(colname2)

        if cols_to_drop:
            df_cleaned = df_cleaned.drop(columns=list(cols_to_drop), errors='ignore')
            print(f"Removed highly correlated columns (threshold={correlation_threshold}): {cols_to_drop}")

        self.cleaned_df = df_cleaned

    def plot_correlation_matrix(self, df_to_plot, title="Correlation Matrix"):
        """
        Computes and displays a correlation matrix heatmap for numeric features in df_to_plot.
        """
        numeric_df = df_to_plot.select_dtypes(include=[np.number])
        if numeric_df.empty:
            print("No numeric columns to plot.")
            return

        corr = numeric_df.corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title(title)
        plt.show()

    def perform_pca(self, df_to_use, n_components=None):
        """
        Standardizes numeric features in df_to_use and runs PCA.
        """
        numeric_df = df_to_use.select_dtypes(include=[np.number]).dropna(axis=1)

        if numeric_df.empty:
            print("No numeric columns left to perform PCA.")
            return None, None

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric_df)

        if n_components is None:
            n_components = min(X_scaled.shape[1], 10)

        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(X_scaled)

        explained_variance = pca.explained_variance_ratio_
        components = pca.components_

        # Plot explained variance ratio
        plt.figure(figsize=(8, 6))
        plt.bar(range(1, n_components + 1), explained_variance, alpha=0.7, align='center')
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.show()

        # Create and print loadings table
        loadings = pd.DataFrame(components, columns=numeric_df.columns)
        print("PCA Component Loadings:")
        print(loadings)

        return pca, loadings

    def run_analysis(self, correlation_threshold=0.85):
        """
        1. Loads & merges data.
        2. Shows original correlation matrix.
        3. Removes unwanted columns (ID-like, near-constant, highly correlated).
        4. Shows cleaned correlation matrix.
        5. Performs PCA on the cleaned data.
        6. Saves the cleaned dataset to CSV.
        """
        # 1. Load & merge
        self.load_and_merge_data()

        # 2. Original correlation matrix
        print("Original Merged Data (Sample):")
        print(self.df.head())
        self.plot_correlation_matrix(self.df, title="Original Correlation Matrix")

        # 3. Remove unwanted columns
        self.remove_unwanted_columns(correlation_threshold=correlation_threshold)

        # 4. Cleaned correlation matrix
        print("Cleaned Data (Sample):")
        print(self.cleaned_df.head())
        self.plot_correlation_matrix(self.cleaned_df, title="Cleaned Correlation Matrix")

        # 5. PCA
        pca, loadings = self.perform_pca(self.cleaned_df)

        # 6. Save cleaned dataset
        self.cleaned_df.to_csv(self.output_file_path, index=False)
        print(f"\nSaved combined data to {self.output_file_path}")

        return self.cleaned_df, pca, loadings
