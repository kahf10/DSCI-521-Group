# Neural Network implementation using Keras/TensorFlow
from BaselineModeling import BaselineModeling
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from scipy.stats import randint, uniform
import tempfile
import shap

class NeuralNetworkModel(BaselineModeling):
    """
    Inherits from BaselineModeling and implements neural network-specific functionality.
    This class provides methods for building, training, tuning, and evaluating
    neural network models for tabular data classification.
    """

    def __init__(self, path_to_X, target_column):
        # Call the parent class constructor
        super().__init__(path_to_X, target_column)
        self.history = None
        self.best_params = None
        self.temp_model_path = None

    def _create_model(self, hidden_layers=(64, 32), activation='relu', 
                      dropout_rate=0.2, learning_rate=0.001, batch_norm=True):
        """
        Create a neural network model with the specified architecture.
        
        Parameters:
        -----------
        hidden_layers : tuple, default=(64, 32)
            Number of neurons in each hidden layer
        activation : str, default='relu'
            Activation function to use in hidden layers
        dropout_rate : float, default=0.2
            Dropout rate for regularization
        learning_rate : float, default=0.001
            Learning rate for the Adam optimizer
        batch_norm : bool, default=True
            Whether to use batch normalization
            
        Returns:
        --------
        model : tf.keras.Model
            Compiled neural network model
        """
        # Get input shape from training data
        input_dim = self.X_train.shape[1]
        
        # Create sequential model
        model = Sequential()
        
        # Add input layer and first hidden layer
        model.add(Dense(hidden_layers[0], input_dim=input_dim, activation=activation))
        if batch_norm:
            model.add(BatchNormalization())
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        # Add additional hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation=activation))
            if batch_norm:
                model.add(BatchNormalization())
            if dropout_rate > 0:
                model.add(Dropout(dropout_rate))
        
        # Add output layer (sigmoid for binary classification)
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        return model

    def train(self, hidden_layers=(64, 32), activation='relu', dropout_rate=0.2,
              learning_rate=0.001, batch_norm=True, batch_size=32, epochs=100):
        """
        Train a neural network model with specified architecture and parameters.
        
        Parameters:
        -----------
        hidden_layers : tuple, default=(64, 32)
            Number of neurons in each hidden layer
        activation : str, default='relu'
            Activation function to use in hidden layers
        dropout_rate : float, default=0.2
            Dropout rate for regularization
        learning_rate : float, default=0.001
            Learning rate for the Adam optimizer
        batch_norm : bool, default=True
            Whether to use batch normalization
        batch_size : int, default=32
            Number of samples per gradient update
        epochs : int, default=100
            Number of epochs to train the model
        """
        print("Training Neural Network model...")
        
        # Set random seed for reproducibility
        tf.random.set_seed(self.random_state)
        
        # Create a temporary file for model checkpointing
        self.temp_model_path = tempfile.mktemp(suffix='.h5')
        
        # Define callbacks
        callbacks = [
            # Stop training when validation loss stops improving
            EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            # Save the best model during training
            ModelCheckpoint(
                filepath=self.temp_model_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            # Reduce learning rate when validation loss plateaus
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Create and compile the model
        self.model = self._create_model(
            hidden_layers=hidden_layers,
            activation=activation,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_norm=batch_norm
        )
        
        # Convert pandas DataFrame to numpy arrays for Keras
        X_train_array = self.X_train.values
        y_train_array = self.y_train.values
        X_val_array = self.X_val.values
        y_val_array = self.y_val.values
        
        # Train the model
        self.history = self.model.fit(
            X_train_array,
            y_train_array,
            validation_data=(X_val_array, y_val_array),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=2  # Show progress bar and one line per epoch
        )
        
        # Load the best model (if early stopping was triggered)
        if self.temp_model_path:
            try:
                self.model = tf.keras.models.load_model(self.temp_model_path)
                print("Loaded best model from checkpoint.")
            except:
                print("No checkpoint model found or error loading model.")
        
        print("Neural Network model training completed.")
        
        # Plot training history
        self._plot_training_history()

    def _plot_training_history(self):
        """
        Plot the training and validation metrics over epochs.
        """
        if self.history is None:
            print("No training history available to plot.")
            return
            
        # Create a figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot loss
        ax1.plot(self.history.history['loss'], label='Training Loss')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot AUC
        ax2.plot(self.history.history['auc'], label='Training AUC')
        ax2.plot(self.history.history['val_auc'], label='Validation AUC')
        ax2.set_title('Model AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def _create_model_for_tuning(self, hidden_layers=2, neurons=64, 
                               activation='relu', dropout_rate=0.2, 
                               learning_rate=0.001, batch_norm=True):
        """
        Create a model with given parameters for hyperparameter tuning.
        This function is used by KerasClassifier.
        
        Parameters are similar to _create_model but structured to work with RandomizedSearchCV.
        """
        # Determine hidden layer structure
        layer_sizes = [neurons] * hidden_layers
        
        # Create and return model
        return self._create_model(
            hidden_layers=layer_sizes,
            activation=activation,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            batch_norm=batch_norm
        )

    def hyperparameter_tuning(self, param_distributions=None, n_iter=20, cv=3):
        """
        Perform randomized search to find optimal hyperparameters for neural network.
        
        Parameters:
        -----------
        param_distributions : dict, default=None
            Dictionary with parameters names as keys and distributions to sample parameters from
        n_iter : int, default=20
            Number of parameter settings sampled
        cv : int, default=3
            Number of cross-validation folds
        """
        print("Performing hyperparameter tuning with randomized search...")
        
        if param_distributions is None:
            # Default parameter distributions for neural network
            param_distributions = {
                'hidden_layers': [1, 2, 3],
                'neurons': randint(16, 128),
                'activation': ['relu', 'elu', 'selu'],
                'dropout_rate': uniform(0, 0.5),
                'learning_rate': uniform(0.0001, 0.01),
                'batch_norm': [True, False],
                'batch_size': [16, 32, 64, 128],
                'epochs': [50]  # Fixed for tuning, we'll use early stopping
            }
        
        # Set random seed for reproducibility
        tf.random.set_seed(self.random_state)
        
        # Create a KerasClassifier wrapper for the model
        model_estimator = KerasClassifier(
            build_fn=self._create_model_for_tuning,
            verbose=0
        )
        
        # Create randomized search
        random_search = RandomizedSearchCV(
            estimator=model_estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring='roc_auc',
            n_jobs=1,  # TensorFlow can parallelize internally
            verbose=2,
            random_state=self.random_state
        )
        
        # Convert pandas DataFrame to numpy arrays for Keras
        X_val_array = self.X_val.values
        y_val_array = self.y_val.values
        
        # Fit randomized search on validation data
        try:
            random_search.fit(X_val_array, y_val_array)
            
            print(f"Best parameters: {random_search.best_params_}")
            print(f"Best ROC AUC score: {random_search.best_score_:.4f}")
            
            # Store best parameters
            self.best_params = random_search.best_params_
            
            # Train a new model with best parameters
            self.train(
                hidden_layers=[self.best_params['neurons']] * self.best_params['hidden_layers'],
                activation=self.best_params['activation'],
                dropout_rate=self.best_params['dropout_rate'],
                learning_rate=self.best_params['learning_rate'],
                batch_norm=self.best_params['batch_norm'],
                batch_size=self.best_params['batch_size'],
                epochs=100  # Use longer epochs with early stopping
            )
            
            print("Model updated with best parameters")
            
        except Exception as e:
            print(f"Error during hyperparameter tuning: {str(e)}")
            print("Using default parameters instead.")
            self.train()

    def get_feature_importance(self, sample_size=100):
        """
        Calculate feature importance using SHAP values.
        
        Parameters:
        -----------
        sample_size : int, default=100
            Number of samples to use for SHAP calculation
            
        Returns:
        --------
        importance_df : pandas.DataFrame
            DataFrame with feature importance values
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        try:
            # Sample data for SHAP analysis (can be computationally expensive)
            X_sample = self.X_train.sample(min(sample_size, len(self.X_train)), 
                                          random_state=self.random_state)
            
            # Create explainer
            explainer = shap.DeepExplainer(self.model, X_sample.values)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample.values)
            
            # Get mean absolute SHAP values as feature importance
            importance = np.mean(np.abs(shap_values[0]), axis=0)
            
            # Create DataFrame with feature names and importance
            importance_df = pd.DataFrame({
                'Feature': self.X_train.columns,
                'Importance': importance
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Create SHAP summary plot
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values[0], X_sample, feature_names=self.X_train.columns)
            
            return importance_df
            
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
            print("Feature importance calculation failed.")
            return None

    def evaluate(self):
        """
        Evaluate the model on test data with neural network-specific metrics.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        # Convert pandas DataFrame to numpy array for Keras
        X_test_array = self.X_test.values
        y_test = self.y_test.values
        
        # Get predictions
        y_prob = self.model.predict(X_test_array)
        y_pred = (y_prob > 0.5).astype(int)
        
        # Print metrics
        print("\nNeural Network Evaluation:")
        print("\nClassification Report:")
        from sklearn.metrics import classification_report
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        from sklearn.metrics import confusion_matrix
        print(confusion_matrix(y_test, y_pred))
        
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"\nROC AUC Score: {roc_auc:.4f}")
        
        f1 = f1_score(y_test, y_pred)
        print(f"\nF1 Score: {f1:.4f}")
        
        # Model evaluation from Keras
        print("\nDetailed metrics:")
        loss, accuracy, auc = self.model.evaluate(X_test_array, y_test, verbose=0)
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC (from model): {auc:.4f}")
        
        # Plot ROC curve
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_model(self, model_path):
        """
        Save the neural network model to disk using TensorFlow's SavedModel format.
        
        Parameters:
        -----------
        model_path : str
            Path where the model should be saved
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
            
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save model in SavedModel format
        self.model.save(model_path)
        print(f"Neural Network model saved to {model_path}")

    def load_model(self, model_path):
        """
        Load a trained neural network model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path from which to load the model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model directory not found at {model_path}")
            
        self.model = tf.keras.models.load_model(model_path)
        print(f"Neural Network model loaded from {model_path}")

    def run_pipeline(self, numerical_columns, categorical_columns, store_model_path):
        """
        Run the complete modeling pipeline from data loading to model evaluation and saving.
        
        Parameters:
        -----------
        numerical_columns : list
            List of numerical column names to normalize
        categorical_columns : list
            List of categorical column names to encode
        store_model_path : str
            Path where the trained model should be saved
        """
        # Execute the pipeline steps
        self.load_data()
        self.split_data()
        self.normalize_standard(numerical_columns)
        self.encode_categorical(categorical_columns)
        
        # Train with default parameters first
        self.train()
        
        # Tune hyperparameters
        self.hyperparameter_tuning()
        
        # Evaluate final model
        self.evaluate()
        
        # Get and display feature importance
        importance = self.get_feature_importance()
        if importance is not None:
            print("\nFeature Importance (based on SHAP values):")
            print(importance.head(10))  # Show top 10 features
        
        # Save the model
        self.save_model(store_model_path)

        