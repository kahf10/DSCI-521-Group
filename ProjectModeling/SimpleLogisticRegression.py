import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


class SimpleLogisticRegression:
    """
    Simple Logistic Regression Model for Churn Prediction.
    """

    def __init__(self, X_train, y_train, X_test, y_test, max_iter=5000, C=0.5):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.max_iter = max_iter
        self.C = C  # Regularization strength
        self.model = None

    def train(self):
        """Train a simple Logistic Regression model."""
        print("Training Simple Logistic Regression Model...")
        self.model = LogisticRegression(max_iter=self.max_iter, C=self.C)
        self.model.fit(self.X_train, self.y_train)
        print("Training completed.\n")

    def evaluate(self, dataset_name, X, y):
        """Evaluate the model on a given dataset."""
        print(f"Evaluating on {dataset_name} Data...")
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1 Score": f1_score(y, y_pred),
            "ROC AUC": roc_auc_score(y, y_pred_proba),
        }

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print(f"\n{dataset_name} Data Evaluation complete.\n")
        return metrics

    def getFeatureImportance(self, top_n=20):
        """Print the most important features based on Logistic Regression coefficients."""
        if self.model is None:
            print("Model not trained yet. Train the model first.")
            return

        feature_importance = self.model.coef_[0]
        feature_names = self.X_train.columns  # Feature names

        # Sort features by importance
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
        importance_df = importance_df.sort_values(by="Importance", ascending=False).head(top_n)

        print("\nTop Feature Importance in Logistic Regression:")
        for idx, row in importance_df.iterrows():
            print(f"  {row['Feature']}: {row['Importance']:.4f}")

        return importance_df

    def runPipeline(self):
        """Run the full pipeline: training, evaluation, and feature importance."""
        self.train()
        self.evaluate("Training", self.X_train, self.y_train)
        self.evaluate("Test", self.X_test, self.y_test)
        self.getFeatureImportance()

