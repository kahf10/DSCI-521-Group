from TransactionAnalysis import TransactionAnalysis
from UserAnalysis import UserAnalysis


class MasterControlAnalysis:
    def __init__(self, transaction_file_path, user_file_path):
        """
        Initializes the master control with provided file paths for transactions and users.
        """
        self.transaction_analysis = TransactionAnalysis(transaction_file_path)
        self.user_analysis = UserAnalysis(user_file_path)

    def run(self):
        """
        Executes both the transaction and user analysis pipelines.
        """
        print("Starting Transaction Analysis:")
        self.transaction_analysis.run_analysis()

        print("Starting User Analysis:")
        self.user_analysis.run_analysis()


if __name__ == "__main__":
    transaction_file_path = "../data/preprocessed_transactions_data.csv"
    user_file_path = "../data/preprocessed_users_data.csv"
    master = MasterControlAnalysis(transaction_file_path, user_file_path)
    master.run()
