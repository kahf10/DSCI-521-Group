import numpy as np
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


class SVMModel:
    """
    Support Vector Machine (SVM) Model for Churn Prediction
    - Converts labels {0,1} to {-1,1}
    - Standardizes features
    - Trains and evaluates an SVM model
    - Performs hyperparameter tuning
    - Extracts support vectors & feature importance approximation
    """

    def __init__(self, X_train, y_train, X_test, y_test, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()

        # Convert labels from {0,1} to {-1,1} (SVM requirement)
        self.y_train = np.where(self.y_train == 0, -1, 1)
        self.y_test = np.where(self.y_test == 0, -1, 1)

        # Standardize Features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

    def runPipeline(self, hyperparameter_tuning=False, model_path="../models/svm_model.pkl"):
        """
        Runs the full pipeline from training to evaluation and model saving.

        Parameters:
        hyperparameter_tuning (bool): Whether to perform hyperparameter tuning.
        model_path (str): Path to save the trained model.
        """
        self.train()
        if hyperparameter_tuning:
            self.hyperparameterTuning()
        self.evaluate()
        self.getSupportVectors()
        #self.saveModel(model_path)
        print("SVM Model execution complete.\n")

    def train(self, C=1.0, kernel='rbf', gamma='scale'):
        """Trains an SVM classifier."""
        print("Training SVM model...")

        self.model = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, random_state=self.random_state)
        self.model.fit(self.X_train, self.y_train)

        print("Model training completed.\n")

    def hyperparameterTuning(self, param_grid=None, cv=5):
        """Performs grid search for optimal hyperparameters."""
        print("Performing hyperparameter tuning...")

        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            }

        grid_search = GridSearchCV(SVC(probability=True), param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(self.X_train, self.y_train)

        print(f"Best Parameters: {grid_search.best_params_}")
        print(f"Best ROC AUC Score: {grid_search.best_score_:.4f}")

        self.model = grid_search.best_estimator_
        print("Updated model with best parameters.\n")

    def evaluate(self):
        """Evaluates the model on test data and prints key metrics."""
        print("Evaluating the model...")

        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(self.y_test, y_pred),
            "Recall": recall_score(self.y_test, y_pred),
            "F1 Score": f1_score(self.y_test, y_pred),
            "ROC AUC": roc_auc_score(self.y_test, y_pred_proba),
        }

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nEvaluation complete.\n")

    def getSupportVectors(self):
        """Extracts number of support vectors per class."""
        n_support = self.model.n_support_
        print(f"Support Vectors per class: {n_support}")

    def saveModel(self, model_path="../models/svm_model.pkl"):
        """Saves the trained model to disk."""
        joblib.dump(self.model, model_path)
        print(f"Model saved at: {model_path}\n")