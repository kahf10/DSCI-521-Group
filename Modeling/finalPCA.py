import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def biplot(scores, coeff, labels=None):
    """
    Creates a biplot for the first two principal components.

    Parameters:
        scores : array, shape (n_samples, 2)
            The PCA scores (projected data) for the first two components.
        coeff : array, shape (n_features, 2)
            The loadings for the first two components.
        labels : list
            List of feature names.
    """
    xs = scores[:, 0]
    ys = scores[:, 1]

    plt.figure(figsize=(10, 8))
    plt.scatter(xs, ys, c='grey', alpha=0.5)

    # Scale factor for arrows
    scale_x = xs.max() - xs.min()
    scale_y = ys.max() - ys.min()

    n = coeff.shape[0]
    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0] * scale_x, coeff[i, 1] * scale_y,
                  color='red', width=0.005, head_width=0.05, alpha=0.7)
        if labels is None:
            plt.text(coeff[i, 0] * scale_x * 1.1, coeff[i, 1] * scale_y * 1.1, f"Var{i + 1}", color='blue', ha='center',
                     va='center')
        else:
            plt.text(coeff[i, 0] * scale_x * 1.1, coeff[i, 1] * scale_y * 1.1, labels[i], color='blue', ha='center',
                     va='center')

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Biplot (PC1 vs. PC2)")
    plt.grid(True)
    plt.show()


def main():
    # Path to your final features CSV file
    file_path = "../data/final_features.csv"

    # Load the data
    df = pd.read_csv(file_path)
    print("Data loaded. Shape:", df.shape)
    print("Columns:", df.columns.tolist())

    # Select numeric columns for PCA; adjust if you need to drop certain columns (like IDs or churn labels)
    numeric_df = df.select_dtypes(include=[np.number])

    # Fill missing values with column means
    numeric_df = numeric_df.fillna(numeric_df.mean())

    print("\nNumeric columns used for PCA:", numeric_df.columns.tolist())

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_df)

    # Determine number of components (up to 10 or number of features)
    n_components = min(X_scaled.shape[1], 10)

    # Perform PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    explained_variance = pca.explained_variance_ratio_

    print("\nExplained Variance Ratio per Component:")
    for i, ratio in enumerate(explained_variance, 1):
        print(f"PC{i}: {ratio:.2%}")

    # Plot explained variance ratio (Elbow Chart)
    plt.figure(figsize=(8, 6))
    plt.bar(range(1, n_components + 1), explained_variance, alpha=0.7, align='center')
    plt.xlabel("Principal Component")
    plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance Ratio")
    plt.xticks(range(1, n_components + 1))
    plt.show()

    # Get PCA component loadings
    loadings = pd.DataFrame(pca.components_, columns=numeric_df.columns)
    print("\nPCA Component Loadings:")
    print(loadings)

    # Plot heatmap of loadings
    plt.figure(figsize=(12, 8))
    sns.heatmap(loadings, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("PCA Component Loadings Heatmap")
    plt.show()

    # Create a biplot for PC1 vs PC2
    # Note: We use the transpose of the first two components from pca.components_
    labels = numeric_df.columns.tolist()
    # pca.components_ shape: (n_components, n_features) -> We take first two components and transpose to get (n_features, 2)
    biplot(X_pca, pca.components_[:2].T, labels=labels)


if __name__ == "__main__":
    main()
