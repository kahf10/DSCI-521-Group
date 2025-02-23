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
        Loads the transactions CSV into a DataFrame.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            print("Transactions file loaded successfully.")
        except Exception as e:
            print(f"Error loading file: {e}")



    def run_analysis(self):
        """
        Runs the complete analysis pipeline.
        """
        self.read_file()
