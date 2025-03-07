# inherit from baseline modeling 
# output feature importance 

from BaselineModeling import BaselineModeling
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class LogisticRegressionModel(BaselineModeling):
    """
    Inherits from BaselineModeling and implements logistic regression-specific functionality.
    This class provides methods for training, hyperparameter tuning, and evaluating
    logistic regression models.
    """

    def __init__(self, path_to_X, target_column):
        # Call the parent class constructor to initialize inherited attributes and methods
        super().__init__(path_to_X, target_column)

    def train(self, C=1.0, penalty='l2', solver='lbfgs', max_iter=1000):
        """
        Train a logistic regression model with specified parameters.
        
        Parameters:
        -----------
        C : float, default=1.0
            Inverse of regularization strength; smaller values specify stronger regularization
        penalty : {'l1', 'l2', 'elasticnet', 'none'}, default='l2'
            Regularization type to use
        solver : str, default='lbfgs'
            Algorithm to use in the optimization problem
        max_iter : int, default=1000
            Maximum number of iterations for solvers to converge
        """
        print("Training Logistic Regression model...")

        # Initialize the model with specified parameters
        self.model = LogisticRegression(
            C=C,
            penalty=penalty,
            solver=solver,
            max_iter=max_iter,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )

        # Train the model on the training data
        self.model.fit(self.X_train, self.y_train)
        print("Logistic Regression model training completed.")

    def hyperparameter_tuning(self, param_grid=None, cv=5):
        """
        Perform grid search to find optimal hyperparameters for logistic regression.
        
        Parameters:
        -----------
        param_grid : dict, default=None
            Dictionary with parameters names as keys and lists of parameter values to try
        cv : int, default=5
            Number of cross-validation folds
        """
        if param_grid is None:
            # Default parameter grid for logistic regression
            # Note: Some combinations may not be valid (e.g., l1 penalty with certain solvers)
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                'penalty': ['l2', 'l1', 'elasticnet', 'none'],
                'solver': ['lbfgs', 'liblinear', 'saga'],
                'max_iter': [1000, 2000]
            }
            
            # Filter out invalid combinations
            # l1 penalty only works with liblinear and saga solvers
            # elasticnet only works with saga
            # none penalty works with all solvers
            valid_param_grid = []
            for C in param_grid['C']:
                for penalty in param_grid['penalty']:
                    for solver in param_grid['solver']:
                        for max_iter in param_grid['max_iter']:
                            # Skip invalid combinations
                            if (penalty == 'l1' and solver not in ['liblinear', 'saga']) or \
                               (penalty == 'elasticnet' and solver != 'saga'):
                                continue
                            valid_param_grid.append({
                                'C': C,
                                'penalty': penalty,
                                'solver': solver,
                                'max_iter': max_iter
                            })
            
        print("Performing hyperparameter tuning with grid search...")

        # Create base Logistic Regression model
        base_model = LogisticRegression(random_state=self.random_state, class_weight='balanced')
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=valid_param_grid if 'valid_param_grid' in locals() else param_grid,
            cv=cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search to validation data
        grid_search.fit(self.X_val, self.y_val)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best ROC AUC score: {grid_search.best_score_:.4f}")
        
        # Update model with best estimator
        self.model = grid_search.best_estimator_
        print("Model updated with best parameters")

    def get_feature_importance(self):
        """
        Return the coefficients of the logistic regression model as feature importance.
        For logistic regression, the absolute value of coefficients can indicate feature importance.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        # Get feature names
        feature_names = self.X_train.columns
        
        # Get coefficients from the model
        coefficients = self.model.coef_[0]
        
        # Create a DataFrame with feature names and their coefficients
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Absolute_Coefficient': np.abs(coefficients)
        })
        
        # Sort by absolute coefficient value (higher means more important)
        importance_df = importance_df.sort_values('Absolute_Coefficient', ascending=False)
        
        return importance_df

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
        self.train()
        self.hyperparameter_tuning()
        self.evaluate()
        
        # Display feature importance
        importance = self.get_feature_importance()
        print("\nFeature Importance (based on coefficients):")
        print(importance.head(10))  # Show top 10 features
        
        # Save the model
        self.save_model(store_model_path)