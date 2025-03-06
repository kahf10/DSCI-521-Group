# Classification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class ClassificationPipeline:
    def __init__(self, combined_file_path, output_file_path="../data/combined_data_with_classification.csv"):
        """
        Initializes the pipeline with the combined dataset file path and
        where to save the new CSV.
        """
        self.combined_file_path = combined_file_path
        self.output_file_path = output_file_path
        self.df = None

    def load_and_prepare_data(self):
        """
        Loads the combined CSV and converts the per_capita_income column to numeric.
        """
        self.df = pd.read_csv(self.combined_file_path)
        # Convert per_capita_income from string ($, commas) to float
        if "per_capita_income" in self.df.columns:
            self.df["per_capita_income"] = (
                self.df["per_capita_income"]
                .replace({'\$': '', ',': ''}, regex=True)
                .astype(float)
            )
        print("Data loaded. Sample:")
        print(self.df.head())

    def cluster_income(self):
        """
        Clusters users based on per_capita_income into 3 groups and assigns labels:
        'Lower Class', 'Middle Class', 'Rich'.
        """
        if "per_capita_income" not in self.df.columns:
            print("Column 'per_capita_income' not found. Skipping income clustering.")
            return

        X_income = self.df[['per_capita_income']].values
        kmeans_income = KMeans(n_clusters=3, random_state=42)
        self.df['income_cluster'] = kmeans_income.fit_predict(X_income)

        # Compute average income for each cluster
        income_summary = self.df.groupby('income_cluster')['per_capita_income'].mean().sort_values()
        print("\nIncome Cluster Summary (Mean Income per Cluster):")
        print(income_summary)

        ordered_clusters = income_summary.index.tolist()
        income_label_map = {
            ordered_clusters[0]: "Lower Class",
            ordered_clusters[1]: "Middle Class",
            ordered_clusters[2]: "Rich"
        }
        self.df['income_class'] = self.df['income_cluster'].map(income_label_map)

    def cluster_debt(self):
        """
        Clusters users based on total_transaction_amount (as a proxy for debt) into 3 groups
        and assigns labels: 'Low Debt', 'Medium Debt', 'High Debt'.
        """
        if "total_transaction_amount" not in self.df.columns:
            print("Column 'total_transaction_amount' not found. Skipping debt clustering.")
            return

        X_debt = self.df[['total_transaction_amount']].values
        # Scale if the range is very large
        scaler = StandardScaler()
        X_debt_scaled = scaler.fit_transform(X_debt)

        kmeans_debt = KMeans(n_clusters=3, random_state=42)
        self.df['debt_cluster'] = kmeans_debt.fit_predict(X_debt_scaled)

        # Compute average transaction amount for each cluster
        debt_summary = self.df.groupby('debt_cluster')['total_transaction_amount'].mean().sort_values()
        print("\nDebt (Proxy) Cluster Summary (Mean Transaction Amount):")
        print(debt_summary)

        ordered_clusters = debt_summary.index.tolist()
        debt_label_map = {
            ordered_clusters[0]: "Low Debt",
            ordered_clusters[1]: "Medium Debt",
            ordered_clusters[2]: "High Debt"
        }
        self.df['debt_class'] = self.df['debt_cluster'].map(debt_label_map)

    def plot_classifications(self):
        """
        Plots bar charts showing the distribution of income and debt classifications.
        """
        # Plot Income Classification distribution
        if 'income_class' in self.df.columns:
            income_counts = self.df['income_class'].value_counts().sort_index()
            plt.figure(figsize=(12, 5))

            plt.subplot(1, 2, 1)
            plt.bar(income_counts.index, income_counts.values, color='mediumseagreen')
            plt.xlabel("Income Category")
            plt.ylabel("Number of Users")
            plt.title("User Distribution by Income Category")
            for i, count in enumerate(income_counts.values):
                plt.text(i, count, str(count), ha="center", va="bottom", fontsize=12)
        else:
            # Create a blank subplot to maintain layout
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.title("No 'income_class' column found")

        # Plot Debt Classification distribution
        if 'debt_class' in self.df.columns:
            debt_counts = self.df['debt_class'].value_counts().sort_index()
            plt.subplot(1, 2, 2)
            plt.bar(debt_counts.index, debt_counts.values, color='coral')
            plt.xlabel("Debt Category")
            plt.ylabel("Number of Users")
            plt.title("User Distribution by Debt Category")
            for i, count in enumerate(debt_counts.values):
                plt.text(i, count, str(count), ha="center", va="bottom", fontsize=12)
        else:
            # If debt_class doesn't exist, label the second subplot
            plt.subplot(1, 2, 2)
            plt.title("No 'debt_class' column found")

        plt.tight_layout()
        plt.show()

    def save_classified_data(self):
        """
        Saves the DataFrame with new classification columns to CSV.
        """
        self.df.to_csv(self.output_file_path, index=False)
        print(f"\nClassified dataset saved to {self.output_file_path}")

    def run_classification(self):
        """
        Runs the full classification pipeline:
          1. Load and prepare data
          2. Cluster by income
          3. Cluster by debt
          4. Plot classification distributions
          5. Save the final dataset
        """
        self.load_and_prepare_data()
        self.cluster_income()
        self.cluster_debt()
        self.plot_classifications()
        self.save_classified_data()
        return self.df
