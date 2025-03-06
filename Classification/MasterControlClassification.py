# MasterControlClassification.py

from Classification import ClassificationPipeline


def main():
    # Path to your combined_data.csv
    combined_file_path = "../data/combined_data.csv"
    # Where to save the final dataset with classification columns
    output_file_path = "../data/combined_data_with_classification.csv"

    # Initialize and run the classification pipeline
    pipeline = ClassificationPipeline(combined_file_path, output_file_path)
    classified_df = pipeline.run_classification()

    print("\nFinal Classified Data Sample:")
    print(classified_df.head())


if __name__ == "__main__":
    main()
