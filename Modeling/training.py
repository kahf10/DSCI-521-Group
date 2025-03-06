import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)

# Import SMOTE for oversampling
from imblearn.over_sampling import SMOTE

# ---------------------------
# Step 1: Load Final Features with Churn
# ---------------------------
data = pd.read_csv("../data/final_features_with_churn.csv")
print("Original shape:", data.shape)
print(data.head())

# ---------------------------
# Step 2: Drop Low-Impact/Redundant Features
# ---------------------------
drop_columns = [
    "client_id",
    "first_transaction_date",
    "last_transaction_date",
    "retirement_age",
    "most_frequent_merchant",
    "min_transaction_amount"
]

data_reduced = data.drop(columns=drop_columns, errors='ignore')
print("After dropping columns, shape:", data_reduced.shape)

# Quick check of churn distribution
plt.figure(figsize=(6,4))
sns.countplot(x="churn_label", data=data_reduced)
plt.title("Churn Distribution (1 = churned, 0 = active)")
plt.show()

# ---------------------------
# Step 3: Prepare Data for Modeling
# ---------------------------
X = data_reduced.drop("churn_label", axis=1)
y = data_reduced["churn_label"]

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Impute missing values
X = X.fillna(X.median())

print("Feature sample after encoding and imputation:")
print(X.head())

# ---------------------------
# Step 4: Train-Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# ---------------------------
# Step 5: Feature Scaling
# ---------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# PART A: Train & Evaluate Models WITHOUT SMOTE
# ============================================================
def evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, note=""):
    print(f"\n--- Model Evaluation {note}---")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        conf_mat = confusion_matrix(y_test, y_pred)
        cls_report = classification_report(y_test, y_pred, zero_division=0)

        print(f"\n{name} ({note})")
        print("Accuracy:", accuracy)
        print("ROC AUC:", roc_auc)
        print("Confusion Matrix:\n", conf_mat)
        print("Classification Report:\n", cls_report)

        # Precision-Recall for the minority class
        precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
        avg_prec = average_precision_score(y_test, y_proba)
        print(f"Average Precision (AP) Score: {avg_prec:.3f}")

        # Plot Precision-Recall curve
        plt.figure()
        plt.plot(recall, precision, label=f"AP = {avg_prec:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"Precision-Recall Curve: {name} ({note})")
        plt.legend(loc="best")
        plt.show()


# Evaluate models WITHOUT SMOTE
evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test, note="No SMOTE")

# ============================================================
# PART B: Apply SMOTE to address class imbalance
# ============================================================
print("\nApplying SMOTE on the training set to handle class imbalance...")
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train_scaled, y_train)

print("SMOTE done. New training set shape:", X_train_smote.shape)
print("Churn distribution in resampled training set:")
print(pd.Series(y_train_smote).value_counts())

# Evaluate models WITH SMOTE
evaluate_models(X_train_smote, X_test_scaled, y_train_smote, y_test, note="With SMOTE")
