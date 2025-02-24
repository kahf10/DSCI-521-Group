import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

class TransactionAnalysis:
    def __init__(self, file_path):
        """
        Initializes the TransactionAnalysis class with the path to the transactions CSV.
        """
        self.file_path = file_path
        self.data = None

    def read_file(self):
        """
        Loads the transactions CSV into a DataFrame and preprocesses key columns.
        - Cleans up the 'amount' column (removes '$' if present and converts to float).
        """
        try:
            self.data = pd.read_csv(self.file_path)
            # Clean up the 'amount' column: remove '$' and convert to float.
            if 'amount' in self.data.columns:
                self.data['amount'] = self.data['amount'].replace({'\$': ''}, regex=True).astype(float)
            print("Transactions file loaded and preprocessed successfully.")
        except Exception as e:
            print(f"Error loading file: {e}")

    def spending_trend_analysis(self):
        """
        Analyzes transaction trends over various time scales:
        - Yearly, Monthly, Weekly, and Daily patterns.
        """
        # Use 'transaction_date' if available; otherwise, fallback to 'date'
        date_col = 'transaction_date' if 'transaction_date' in self.data.columns else 'date'
        if date_col not in self.data.columns:
            print(f"Column '{date_col}' not found in the dataset.")
            return

        # Convert date column to datetime
        self.data[date_col] = pd.to_datetime(self.data[date_col])
        self.data['year'] = self.data[date_col].dt.year
        self.data['month'] = self.data[date_col].dt.month
        self.data['day'] = self.data[date_col].dt.day
        self.data['weekday'] = self.data[date_col].dt.day_name()

        # Yearly spending trend
        yearly_trend = self.data.groupby('year')['amount'].sum().reset_index()
        plt.figure(figsize=(10,6))
        sns.barplot(data=yearly_trend, x='year', y='amount', palette='viridis')
        plt.title("Yearly Spending Trend")
        plt.xlabel("Year")
        plt.ylabel("Total Spending")
        plt.show()

        # Monthly spending trend (aggregated across years)
        monthly_trend = self.data.groupby('month')['amount'].sum().reset_index()
        plt.figure(figsize=(10,6))
        sns.barplot(data=monthly_trend, x='month', y='amount', palette='coolwarm')
        plt.title("Monthly Spending Trend (Aggregated)")
        plt.xlabel("Month")
        plt.ylabel("Total Spending")
        plt.show()

        # Weekly spending trend
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekday_trend = self.data.groupby('weekday')['amount'].sum().reindex(weekday_order).reset_index()
        plt.figure(figsize=(10,6))
        sns.barplot(data=weekday_trend, x='weekday', y='amount', palette='magma')
        plt.title("Weekly Spending Trend")
        plt.xlabel("Weekday")
        plt.ylabel("Total Spending")
        plt.show()

        # Daily spending trend for the latest month (as an example)
        latest_period = self.data[date_col].max().to_period('M')
        latest_month_data = self.data[self.data[date_col].dt.to_period('M') == latest_period]
        daily_trend = latest_month_data.groupby('day')['amount'].sum().reset_index()
        plt.figure(figsize=(10,6))
        sns.lineplot(data=daily_trend, x='day', y='amount', marker='o')
        plt.title(f"Daily Spending Trend for {latest_period}")
        plt.xlabel("Day")
        plt.ylabel("Total Spending")
        plt.show()

        print("Spending trend analysis completed.")

    def merchant_category_analysis(self):
        """
        Analyzes merchant categories:
        - Uses 'merchant_category' if available; otherwise, falls back to 'mcc'.
        - Displays top categories by transaction volume and total spending.
        - If available, analyzes average spending per user (using 'client_id').
        """
        # Use 'merchant_category' if it exists; else use 'mcc'
        if 'merchant_category' in self.data.columns:
            cat_col = 'merchant_category'
        elif 'mcc' in self.data.columns:
            cat_col = 'mcc'
        else:
            print("No merchant category column ('merchant_category' or 'mcc') found in the dataset.")
            return

        # Aggregate volume and spending per merchant category
        category_summary = self.data.groupby(cat_col).agg(
            total_volume=pd.NamedAgg(column='amount', aggfunc='count'),
            total_spending=pd.NamedAgg(column='amount', aggfunc='sum')
        ).reset_index()

        # Top categories by volume
        top_volume = category_summary.sort_values('total_volume', ascending=False).head(10)
        plt.figure(figsize=(10,6))
        sns.barplot(data=top_volume, x='total_volume', y=cat_col, palette='Blues_d')
        plt.title("Top Merchant Categories by Transaction Volume")
        plt.xlabel("Transaction Volume")
        plt.ylabel("Merchant Category")
        plt.show()

        # Top categories by total spending
        top_spending = category_summary.sort_values('total_spending', ascending=False).head(10)
        plt.figure(figsize=(10,6))
        sns.barplot(data=top_spending, x='total_spending', y=cat_col, palette='Greens_d')
        plt.title("Top Merchant Categories by Total Spending")
        plt.xlabel("Total Spending")
        plt.ylabel("Merchant Category")
        plt.show()

        # Average user spending by merchant category (if 'client_id' exists)
        if 'client_id' in self.data.columns:
            user_category = self.data.groupby([cat_col, 'client_id'])['amount'].sum().reset_index()
            avg_spending = user_category.groupby(cat_col)['amount'].mean().reset_index()
            plt.figure(figsize=(10,6))
            sns.barplot(data=avg_spending, x='amount', y=cat_col, palette='Oranges_d')
            plt.title("Average User Spending by Merchant Category")
            plt.xlabel("Average Spending per User")
            plt.ylabel("Merchant Category")
            plt.show()
        else:
            print("Column 'client_id' not found; skipping average user spending analysis.")

        print("Merchant category analysis completed.")

    def method_analysis(self):
        """
        Analyzes transactions based on payment methods:
        - Chip vs. non-chip transactions using 'use_chip'.
        - Online vs. in-person transactions (if 'transaction_mode' exists).
        """
        # Chip analysis using 'use_chip'
        if 'use_chip' in self.data.columns:

            self.data['use_chip'] = self.data['use_chip'].replace({
                "Chip Transaction": "Swipe Transaction"
            })
            chip_summary = self.data['use_chip'].value_counts(dropna=False)
            print("\nChip Transaction Types:")
            print(chip_summary)
            plt.figure(figsize=(6,4))
            sns.countplot(data=self.data, x='use_chip', palette='Set2')
            plt.title("Chip Transaction Analysis")
            plt.xlabel("Transaction Type")
            plt.ylabel("Count")
            plt.show()
        else:
            print("Column 'use_chip' not found; skipping chip analysis.")

    print("Method analysis completed.")

    def run_analysis(self):
        """
        Executes the complete analysis pipeline.
        """
        self.read_file()
        self.spending_trend_analysis()
        self.merchant_category_analysis()
        self.method_analysis()
