from Modeling import Modeling

class MasterControlModeling:
    def __init__(self):
        self.modeling = None

    def runModeling(self, trainFilePath, testFilePath):
        """
        Runs the modeling pipeline for the churn prediction task.
        """
        self.modeling = Modeling(trainFilePath, testFilePath)
        self.modeling.runPipeline()

if __name__ == "__main__":
    trainFilePath = "../data/train_data.csv"
    testFilePath = "../data/test_data.csv"

    master = MasterControlModeling()
    master.runModeling(trainFilePath, testFilePath)
