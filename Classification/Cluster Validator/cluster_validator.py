import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class ClusterValidation:
    def __init__(self, data_file_path):
        """
        Initializes the cluster validation with the combined dataset.

        :param data_file_path: Path to the combined_data.csv file.
        """
        self.data_file_path = data_file_path
        self.df = None

    def load_data(self):
        """
        Loads the combined data CSV and cleans key columns.
        """
        self.df = pd.read_csv(self.data_file_path)
        # Clean per_capita_income: remove $ and commas, then convert to float
        if "per_capita_income" in self.df.columns:
            self.df["per_capita_income"] = self.df["per_capita_income"] \
                .replace({'\$': '', ',': ''}, regex=True).astype(float)
        print("Combined data loaded. Columns:", self.df.columns.tolist())

    def validate_feature_clusters(self, feature, k_range=range(1, 11)):
        """
        For a given feature, computes inertia (elbow method) and silhouette scores
        for a range of cluster numbers, then plots the results.

        :param feature: The column name to validate.
        :param k_range: A range of cluster numbers to try.
        """
        if feature not in self.df.columns:
            print(f"Feature '{feature}' not found in the dataset.")
            return

        X = self.df[[feature]].values
        inertias = []
        silhouettes = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            # Silhouette score requires at least 2 clusters
            if k == 1 or len(np.unique(labels)) == 1:
                silhouettes.append(0)
            else:
                silhouettes.append(silhouette_score(X, labels))

        # Plot the elbow method and silhouette scores side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.plot(list(k_range), inertias, marker='o')
        ax1.set_title(f'Elbow Method for {feature}')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')

        ax2.plot(list(k_range), silhouettes, marker='o', color='orange')
        ax2.set_title(f'Silhouette Score for {feature}')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')

        plt.suptitle(f"Cluster Validation for '{feature}'")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()


def main():
    # Path to your combined_data.csv (update as needed)
    data_file_path = "../../data/combined_data.csv"

    validator = ClusterValidation(data_file_path)
    validator.load_data()

    # List the features you want to validate.
    # You can adjust this list based on your interest.
    features = [
        "current_age",
        "retirement_age",
        "per_capita_income",
        "credit_score",
        "num_credit_cards",
        "total_credit_limit",
        "total_transaction_amount"
    ]

    for feature in features:
        print(f"\nValidating clusters for '{feature}':")
        validator.validate_feature_clusters(feature=feature, k_range=range(1, 11))


if __name__ == "__main__":
    main()
