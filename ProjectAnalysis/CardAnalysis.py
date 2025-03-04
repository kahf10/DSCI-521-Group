import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CardAnalysis:
    def __init__(self, file_path):
        """
        Initializes the CardAnalysis class with the provided card CSV file path.
        """
        self.file_path = file_path
        self.data = None

    def readFile(self):
        """
        Reads the card CSV file into a Pandas DataFrame.
        """
        try:
            # Parse date columns if necessary.
            self.data = pd.read_csv(self.file_path, parse_dates=['expires', 'acct_open_date'])
            print("Card data file loaded successfully.")
        except Exception as e:
            print(f"Error loading card file: {e}")

    def analyzeCardBrands(self):
        """
        Analyzes and visualizes the distribution of card brands.
        """
        if 'card_brand' in self.data.columns:
            brand_counts = self.data['card_brand'].value_counts()
            print("Card Brand Counts:")
            print(brand_counts)
            plt.figure(figsize=(8,6))
            sns.barplot(x=brand_counts.index, y=brand_counts.values, palette='viridis')
            plt.title("Card Brand Distribution")
            plt.xlabel("Card Brand")
            plt.ylabel("Count")
            plt.show()
        else:
            print("Column 'card_brand' not found in card data.")

    def analyzeCardTypes(self):
        """
        Analyzes and visualizes the distribution of card types.
        """
        if 'card_type' in self.data.columns:
            type_counts = self.data['card_type'].value_counts()
            print("Card Type Counts:")
            print(type_counts)
            plt.figure(figsize=(8,6))
            sns.barplot(x=type_counts.index, y=type_counts.values, palette='magma')
            plt.title("Card Type Distribution")
            plt.xlabel("Card Type")
            plt.ylabel("Count")
            plt.show()
        else:
            print("Column 'card_type' not found in card data.")

    def analyzeCreditLimit(self):
        """
        Plots the distribution of credit limits.
        """
        if 'credit_limit' in self.data.columns:
            plt.figure(figsize=(8,6))
            sns.histplot(self.data['credit_limit'], bins=10, kde=True)
            plt.title("Credit Limit Distribution")
            plt.xlabel("Credit Limit")
            plt.ylabel("Frequency")
            plt.show()
        else:
            print("Column 'credit_limit' not found in card data.")

    def analyzeChipUsage(self):
        """
        Analyzes and visualizes the usage of chip-enabled cards.
        """
        if 'has_chip' in self.data.columns:
            chip_counts = self.data['has_chip'].value_counts()
            print("Chip Usage Counts:")
            print(chip_counts)
            plt.figure(figsize=(8,6))
            sns.barplot(x=chip_counts.index.astype(str), y=chip_counts.values, palette='Set2')
            plt.title("Chip Usage Distribution")
            plt.xlabel("Has Chip")
            plt.ylabel("Count")
            plt.show()
        else:
            print("Column 'has_chip' not found in card data.")

    def run_analysis(self):
        """
        Executes the complete card analysis pipeline.
        """
        self.readFile()
        if self.data is not None:
            self.analyzeCardBrands()
            self.analyzeCardTypes()
            self.analyzeCreditLimit()
            self.analyzeChipUsage()
            print("Card analysis completed.")
