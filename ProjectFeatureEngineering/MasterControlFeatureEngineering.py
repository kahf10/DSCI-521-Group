from FeatureEngineering import FeatureEngineering


class MasterControl:
    def __init__(self):
        self.featureEngineeringPreprocessor = None

    def runPreprocessing(self, transactionsFilePath, usersFilePath):
        """
        Runs the preprocessing steps for transactions, users, and cards datasets.
        """
        self.featureEngineeringPreprocessor = FeatureEngineering(transactionsFilePath, usersFilePath)
        self.featureEngineeringPreprocessor.runPipeline()

if __name__ == "__main__":
    transactionsFilePath = "../data/preprocessed_transactions_data.csv"
    usersFilePath = "../data/preprocessed_users_data.csv"
    cardsFilePath = "../data/cards_data.csv"

    master = MasterControl()
    master.runPreprocessing(transactionsFilePath, usersFilePath)
