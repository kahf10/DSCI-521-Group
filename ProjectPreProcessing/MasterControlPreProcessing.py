from PreProcessingTransactions import PreprocessingTransactions
from PreProcessingUsers import PreProcessingUsers


class MasterControl:
    def runPreprocessing(self, transactionsFilePath, usersFilePath, cardsFilePath):
        """
        Runs the preprocessing steps for transactions, users, and cards datasets.
        """
        #self.transactionsPreprocessor = PreprocessingTransactions(transactionsFilePath)
        #self.transactionsPreprocessor.runPipeline()

        self.usersPreprocessor = PreProcessingUsers(usersFilePath)
        self.usersPreprocessor.runPipeline()

        #self.cardsPreprocessor = PreProcessingUsers(cardsFilePath)
        #self.cardsPreprocessor.runPipeline()

if __name__ == "__main__":
    transactionsFilePath = "../data/transactions_data.csv"
    usersFilePath = "../data/users_data.csv"
    cardsFilePath = "../data/cards_data.csv"

    master = MasterControl()
    master.runPreprocessing(transactionsFilePath, usersFilePath, cardsFilePath)
