from PreProcessingTransactions import PreprocessingTransactions

class MasterControl:
    def __init__(self, file_path):
        """
        Initializes the MasterControl class and sets up preprocessing.
        """
        self.preprocessor = PreprocessingTransactions(file_path)

    def runPreprocessing(self):
        """
        Runs the preprocessing steps.
        """
        self.preprocessor.runPipeline()

if __name__ == "__main__":
    file_path = "../data/transactions_data.csv"
    master = MasterControl(file_path)
    master.runPreprocessing()
