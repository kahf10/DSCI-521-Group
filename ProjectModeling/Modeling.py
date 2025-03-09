import pandas as pd

from LogisticRegressionModel import LogisticRegressionModel
from ProjectModeling.DeepLearningModel import DeepLearningModel
from ProjectModeling.SVMModel import SVMModel
from ProjectModeling.SimpleLogisticRegression import SimpleLogisticRegression


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

        # print("-" * 100)
        # self.applyDeepLearning()

        print("-" * 100)
        self.applySimpleLogisticRegression()

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
        Prepares the dataset for modeling by handling missing values and ensuring correct data types.
        """
        if self.train_data is not None and self.test_data is not None:
            # Step 1: Handle Missing Values (Fill numeric with median, categorical with mode)
            for column in self.train_data.columns:
                if self.train_data[column].isnull().sum() > 0:
                    if self.train_data[column].dtype == 'O':  # Categorical
                        mode_value = self.train_data[column].mode()[0]
                        self.train_data.loc[:, column] = self.train_data[column].fillna(mode_value)
                        self.test_data.loc[:, column] = self.test_data[column].fillna(mode_value)
                    else:  # Numeric
                        median_value = self.train_data[column].median()
                        self.train_data.loc[:, column] = self.train_data[column].fillna(median_value)
                        self.test_data.loc[:, column] = self.test_data[column].fillna(median_value)

            # Step 2: Splitting Features and Target
            self.X_train = self.train_data.drop(columns=['churn_label'])
            self.y_train = self.train_data['churn_label']
            self.X_test = self.test_data.drop(columns=['churn_label'])
            self.y_test = self.test_data['churn_label']

            print("Dataset initialized with missing values handled.")
        else:
            print("Error: Data not loaded. Run readFiles() first.")

    def applyLogisticRegression(self):
        """
        Train and evaluate Logistic Regression model.
        """
        print("Training Logistic Regression model")
        logistic_model = SimpleLogisticRegression(self.X_train, self.y_train, self.X_test, self.y_test)
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

    def applySimpleLogisticRegression(self):
        """
        Train and evaluate Logistic Regression model
        """
        print("Training simple Logistic Regression model")
        logistic_model = SimpleLogisticRegression(self.X_train, self.y_train, self.X_test, self.y_test)
        logistic_model.runPipeline()

