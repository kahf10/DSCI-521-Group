import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class UserAnalysis:
    def __init__(self, file_path):
        """
        Initializes the UserAnalysis class with the path to the users CSV.
        """
        self.file_path = file_path
        self.data = None

    def read_file(self):
        """
        Loads the users CSV into a DataFrame and preprocesses monetary columns.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            # Preprocess monetary columns: remove '$' and convert to float.
            money_columns = ['per_capita_income', 'yearly_income', 'total_debt']
            for col in money_columns:
                if col in self.data.columns:
                    self.data[col] = self.data[col].replace({'\$': ''}, regex=True).astype(float)
            print("User data loaded and preprocessed successfully.")
        except Exception as e:
            print(f"Error loading user data file: {e}")

    def analyze_demographics(self):
        """
        Analyzes demographic aspects:
        - Current Age Distribution
        - Retirement Age Distribution
        - Years Until Retirement (retirement_age - current_age)
        - Gender Distribution
        """
        # Current Age Distribution
        if 'current_age' in self.data.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.data['current_age'], bins=10, kde=True)
            plt.title("Current Age Distribution")
            plt.xlabel("Current Age")
            plt.ylabel("Frequency")
            plt.show()
        else:
            print("Column 'current_age' not found.")

        # Retirement Age Distribution
        if 'retirement_age' in self.data.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.data['retirement_age'], bins=10, kde=True, color='orange')
            plt.title("Retirement Age Distribution")
            plt.xlabel("Retirement Age")
            plt.ylabel("Frequency")
            plt.show()
        else:
            print("Column 'retirement_age' not found.")

        # Years Until Retirement
        if 'current_age' in self.data.columns and 'retirement_age' in self.data.columns:
            self.data['years_to_retirement'] = self.data['retirement_age'] - self.data['current_age']
            plt.figure(figsize=(8, 5))
            sns.histplot(self.data['years_to_retirement'], bins=10, kde=True, color='green')
            plt.title("Years Until Retirement")
            plt.xlabel("Years to Retirement")
            plt.ylabel("Frequency")
            plt.show()
        else:
            print("Cannot compute years until retirement; necessary columns are missing.")

        # Gender Distribution
        if 'gender' in self.data.columns:
            gender_counts = self.data['gender'].value_counts()
            plt.figure(figsize=(6, 4))
            sns.barplot(x=gender_counts.index, y=gender_counts.values, palette='pastel')
            plt.title("Gender Distribution")
            plt.xlabel("Gender")
            plt.ylabel("Count")
            plt.show()
        else:
            print("Column 'gender' not found.")

        print("Demographic analysis completed.")

    def analyze_financials(self):
        """
        Analyzes financial variables:
        - Distributions of per capita income, yearly income, and total debt.
        - Credit score distribution.
        - Distribution of the number of credit cards.
        - Scatter plot of yearly income vs. credit score (colored by gender).
        - Correlation heatmap of key financial metrics.
        """
        # Income and Debt Distributions
        money_columns = ['per_capita_income', 'yearly_income', 'total_debt']
        for col in money_columns:
            if col in self.data.columns:
                plt.figure(figsize=(8, 5))
                sns.histplot(self.data[col], bins=10, kde=True)
                plt.title(f"{col.replace('_', ' ').title()} Distribution")
                plt.xlabel(col.replace('_', ' ').title())
                plt.ylabel("Frequency")
                plt.show()
            else:
                print(f"Column '{col}' not found.")

        # Credit Score Distribution
        if 'credit_score' in self.data.columns:
            plt.figure(figsize=(8, 5))
            sns.histplot(self.data['credit_score'], bins=10, kde=True, color='purple')
            plt.title("Credit Score Distribution")
            plt.xlabel("Credit Score")
            plt.ylabel("Frequency")
            plt.show()
        else:
            print("Column 'credit_score' not found.")

        # Number of Credit Cards Distribution
        if 'num_credit_cards' in self.data.columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(x=self.data['num_credit_cards'], palette='coolwarm')
            plt.title("Number of Credit Cards Distribution")
            plt.xlabel("Number of Credit Cards")
            plt.ylabel("Count")
            plt.show()
        else:
            print("Column 'num_credit_cards' not found.")

        # Scatter Plot: Yearly Income vs Credit Score (if gender available, use it for hue)
        if 'yearly_income' in self.data.columns and 'credit_score' in self.data.columns:
            plt.figure(figsize=(8, 5))
            if 'gender' in self.data.columns:
                sns.scatterplot(data=self.data, x='yearly_income', y='credit_score', hue='gender')
            else:
                sns.scatterplot(data=self.data, x='yearly_income', y='credit_score')
            plt.title("Yearly Income vs Credit Score")
            plt.xlabel("Yearly Income")
            plt.ylabel("Credit Score")
            plt.show()
        else:
            print("Required columns for scatter plot not found.")

        # Correlation Heatmap for Financial Metrics
        financial_cols = ['per_capita_income', 'yearly_income', 'total_debt', 'credit_score', 'num_credit_cards']
        available_cols = [col for col in financial_cols if col in self.data.columns]
        if available_cols:
            corr_matrix = self.data[available_cols].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
            plt.title("Correlation Heatmap of Financial Variables")
            plt.show()
        else:
            print("No financial columns available for correlation analysis.")

        print("Financial analysis completed.")

    def run_analysis(self):
        """
        Executes the complete user analysis pipeline.
        """
        self.read_file()
        self.analyze_demographics()
        self.analyze_financials()
