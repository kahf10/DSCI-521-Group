from FeatureAnalysis import FeatureAnalysisPipeline


def main():
    user_file_path = "../../data/preprocessed_users_data.csv"
    card_file_path = "../../data/preprocessed_cards_data.csv"
    transaction_file_path = "../../data/preprocessed_transactions_data.csv"

    # Instantiate the pipeline, specifying where to save the combined dataset
    pipeline = FeatureAnalysisPipeline(
        user_file_path=user_file_path,
        card_file_path=card_file_path,
        transaction_file_path=transaction_file_path,
        output_file_path="../../data/combined_data.csv"  # <--- Where the cleaned data is saved
    )

    # Run the analysis
    cleaned_df, pca, loadings = pipeline.run_analysis(correlation_threshold=0.85)

    # Inspect the final cleaned dataset and PCA loadings
    print("\n=== Final Cleaned Data Sample ===")
    print(cleaned_df.head())
    if loadings is not None:
        print("\n=== PCA Loadings ===")
        print(loadings)


if __name__ == "__main__":
    main()
