import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os


class DeepLearningModel:
    """
    Deep Learning Model for Churn Prediction.
    - Trains & evaluates a neural network model.
    - Performs hyperparameter tuning.
    - Implements class balancing and proper feature scaling.
    """

    def __init__(self, X_train, y_train, X_test, y_test, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.random_state = random_state
        self.model = None
        self.history = None
        self.model_path = "../models/deep_learning_model.keras"

        # Scale features
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Handle class imbalance
        self.class_weights = self.compute_class_weights()

    def compute_class_weights(self):
        """Computes class weights to handle imbalance."""
        class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(self.y_train), y=self.y_train)
        return {i: class_weights[i] for i in range(len(class_weights))}

    def build_model(self, input_dim, hidden_layers=(128, 64, 32), dropout_rate=0.2, learning_rate=0.0001,
                    optimizer='adam'):
        """Builds and compiles a deep learning model with batch normalization and dropout."""

        model = Sequential()
        model.add(Dense(hidden_layers[0], input_dim=input_dim, kernel_regularizer=l2(0.01)))
        model.add(LeakyReLU(alpha=0.1))  # Avoid vanishing gradient
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

        for units in hidden_layers[1:]:
            model.add(Dense(units, kernel_regularizer=l2(0.01)))
            model.add(LeakyReLU(alpha=0.1))
            model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))

        model.add(Dense(1, activation='sigmoid'))  # Binary classification output

        # Choose optimizer
        optimizer = Adam(learning_rate=learning_rate) if optimizer == 'adam' else RMSprop(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])
        return model

    def train(self, hidden_layers=(128, 64, 32), dropout_rate=0.2, learning_rate=0.0001, batch_size=64, epochs=100):
        """Trains the deep learning model with early stopping and adaptive learning rate."""

        print("Training Deep Learning Model...")

        self.model = self.build_model(input_dim=self.X_train.shape[1], hidden_layers=hidden_layers,
                                      dropout_rate=dropout_rate, learning_rate=learning_rate)

        callbacks = [
            EarlyStopping(monitor="val_auc", patience=15, restore_best_weights=True, verbose=1),
            ModelCheckpoint(filepath=self.model_path, monitor="val_auc", mode="max", save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]

        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=self.class_weights,  # Apply class balancing
            callbacks=callbacks,
            verbose=2
        )

        print("Training Completed.")

    def evaluate(self):
        """Evaluates the model on test data."""
        print("\nEvaluating the Model...")
        y_pred_proba = self.model.predict(self.X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
            confusion_matrix

        metrics = {
            "Accuracy": accuracy_score(self.y_test, y_pred),
            "Precision": precision_score(self.y_test, y_pred),
            "Recall": recall_score(self.y_test, y_pred),
            "F1 Score": f1_score(self.y_test, y_pred),
            "ROC AUC": roc_auc_score(self.y_test, y_pred_proba)
        }

        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

        return metrics

    def plot_training_history(self):
        """Plots training & validation metrics over epochs."""
        if self.history is None:
            print("No training history available to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss plot
        ax1.plot(self.history.history['loss'], label="Training Loss")
        ax1.plot(self.history.history['val_loss'], label="Validation Loss")
        ax1.set_title("Loss Over Epochs")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        # AUC plot
        ax2.plot(self.history.history['auc'], label="Training AUC")
        ax2.plot(self.history.history['val_auc'], label="Validation AUC")
        ax2.set_title("AUC Over Epochs")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    def save_model(self):
        """Saves the trained model."""
        self.model.save(self.model_path)
        print(f"Model saved at {self.model_path}")

    def load_model(self):
        """Loads a saved model."""
        if os.path.exists(self.model_path):
            self.model = tf.keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print("No saved model found.")

    def run_pipeline(self):
        """Runs the full pipeline: training, evaluation, and saving."""
        self.train()
        self.evaluate()
        self.plot_training_history()
        #self.save_model()
        print("Deep Learning Model Execution Complete.")
