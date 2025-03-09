from FeatureEngineering import FeatureEngineering


class MasterControlFeatureEngineering:
    def __init__(self):
        self.featureEngineering = None

    def runPFeatureEngineering(self, transactionsFilePath, usersFilePath):
        """
        Runs the preprocessing steps for transactions, users, and cards datasets.
        """
        self.featureEngineering = FeatureEngineering(transactionsFilePath, usersFilePath)
        self.featureEngineering.runPipeline()

if __name__ == "__main__":
    transactionsFilePath = "../data/preprocessed_transactions_data.csv"
    usersFilePath = "../data/preprocessed_users_data.csv"

    master = MasterControlFeatureEngineering()
    master.runPFeatureEngineering(transactionsFilePath, usersFilePath)
