from TransactionAnalysis import TransactionAnalysis

class MasterControlTransactionAnalysis:
    def __init__(self, file_path):
        """
        Initializes the master control with the provided transactions file path.
        """
        self.analysis = TransactionAnalysis(file_path)

    def run(self):
        """
        Executes the transaction analysis pipeline.
        """
        self.analysis.run_analysis()

if __name__ == "__main__":
    file_path = "../data/preprocessed_transactions_data.csv"
    master = MasterControlTransactionAnalysis(file_path)
    master.run()
