import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np


class LogisticRegressionModel:
    """
    Logistic Regression Model for Churn Prediction
    - Trains and evaluates logistic regression
    - Performs hyperparameter tuning
    - Outputs feature importance
    """

    def __init__(self, X_train, y_train, X_test, y_test, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.model = None

    def runPipeline(self, hyperparameter_tuning=True, model_path="../models/logistic_regression.pkl"):
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
        self.getFeatureImportance()
        #self.saveModel(model_path)
        print("Logistic Regression Model execution complete.\n")

    def train(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=5000):
        """Trains a logistic regression model with more iterations for convergence."""
        print("Training Logistic Regression model...")

        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,  # Back to 'lbfgs' for stability
            max_iter=max_iter,  # Increased iterations
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )

        self.model.fit(self.X_train, self.y_train)
        print("Model training completed.\n")

    def hyperparameterTuning(self, param_grid=None, cv=5):
        """Performs grid search for optimal hyperparameters."""
        print("Performing hyperparameter tuning...")

        if param_grid is None:
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],  # Only solvers that support l1 & l2
                'max_iter': [1000, 2000]
            }

        grid_search = GridSearchCV(
            estimator=LogisticRegression(random_state=self.random_state, class_weight='balanced'),
            param_grid=param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

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

    def getFeatureImportance(self):
        """Returns feature importance based on absolute coefficient values."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")

        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Coefficient': self.model.coef_[0],
            'Absolute_Coefficient': np.abs(self.model.coef_[0])
        }).sort_values(by='Absolute_Coefficient', ascending=False)

        print("\nTop 10 Important Features:")
        print(feature_importance.head(10))

        return feature_importance

    def saveModel(self, model_path="../models/logistic_regression.pkl"):
        """Saves the trained model to disk."""
        joblib.dump(self.model, model_path)
        print(f"Model saved at: {model_path}\n")
       