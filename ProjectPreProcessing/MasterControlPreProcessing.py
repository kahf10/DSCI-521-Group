from PreProcessingTransactions import PreprocessingTransactions
from PreProcessingUsers import PreProcessingUsers
from PreProcessingCards import PreProcessingCards  # Make sure this import is added

class MasterControl:
    def runPreprocessing(self, transactionsFilePath, usersFilePath, cardsFilePath):
        """
        Runs the preprocessing steps for transactions, users, and cards datasets.
        """
        # Uncomment and implement if needed:
        # self.transactionsPreprocessor = PreprocessingTransactions(transactionsFilePath)
        # self.transactionsPreprocessor.runPipeline()

        self.usersPreprocessor = PreProcessingUsers(usersFilePath)
        self.usersPreprocessor.runPipeline()

        self.cardsPreprocessor = PreProcessingCards(cardsFilePath)
        # Optionally pass a list of columns to drop if required:
        self.cardsPreprocessor.runPipeline()

if __name__ == "__main__":
    transactionsFilePath = "../data/transactions_data.csv"
    usersFilePath = "../data/users_data.csv"
    cardsFilePath = "../data/cards_data.csv"

    master = MasterControl()
    master.runPreprocessing(transactionsFilePath, usersFilePath, cardsFilePath)
