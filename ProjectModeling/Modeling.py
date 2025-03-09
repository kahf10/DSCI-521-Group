import pandas as pd

from LogisticRegressionModel import LogisticRegressionModel
from ProjectModeling.DeepLearningModel import DeepLearningModel
from ProjectModeling.SVMModel import SVMModel


class Modeling:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.train_data = None
        self.test_data = None

    def runPipeline(self):
        """
        Runs the modeling pipeline.
        """
        print("-" * 100)
        self.readFiles()

        print("-" * 100)
        self.initializeModelingDataset()

        # print("-" * 100)
        # self.applyLogisticRegression()

        # print("-" * 100)
        # self.applySVMModeling()

        print("-" * 100)
        self.applyDeepLearning()

        print("-" * 100)
        print("Modeling pipeline execution completed.")

    def readFiles(self):
        """
        Reads the train and test CSV files into Pandas DataFrames.
        """
        try:
            self.train_data = pd.read_csv(self.train_file)
            print("Train data successfully loaded.")
        except Exception as e:
            print(f"Error loading train data: {e}")

        try:
            self.test_data = pd.read_csv(self.test_file)
            print("Test data successfully loaded.")
        except Exception as e:
            print(f"Error loading test data: {e}")

    def initializeModelingDataset(self):
        """
        Prepares the dataset for modeling by handling missing values and splitting features/target.
        """
        if self.train_data is not None and self.test_data is not None:
            # Handle missing values in training & test sets
            self.train_data.fillna(self.train_data.median(numeric_only=True), inplace=True)
            self.train_data.fillna(self.train_data.mode().iloc[0], inplace=True)

            self.test_data.fillna(self.test_data.median(numeric_only=True), inplace=True)
            self.test_data.fillna(self.test_data.mode().iloc[0], inplace=True)

            # Splitting features and target
            self.X_train = self.train_data.drop(columns=['churn_label'])
            self.y_train = self.train_data['churn_label']
            self.X_test = self.test_data.drop(columns=['churn_label'])
            self.y_test = self.test_data['churn_label']

            print("Modeling dataset initialized with missing values handled.")
        else:
            print("Error: Data not loaded. Run readFiles() first.")

    def applyLogisticRegression(self):
        """
        Train and evaluate Logistic Regression model.
        """
        print("Training Logistic Regression model")
        logistic_model = LogisticRegressionModel(self.X_train, self.y_train, self.X_test, self.y_test)
        logistic_model.runPipeline()

    def applySVMModeling(self):
        """
        Train and evaluate SVM model
        """
        print("Training SVM model")
        svm_model = SVMModel(self.X_train, self.y_train, self.X_test, self.y_test)
        svm_model.runPipeline()

    def applyDeepLearning(self):
        """
        Train and evaluate Neural Network model
        """
        print("Training Deep Learning model")
        dl_model = DeepLearningModel(self.X_train, self.y_train, self.X_test, self.y_test)
        dl_model.run_pipeline()

