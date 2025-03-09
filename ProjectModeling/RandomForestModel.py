

# inherit from BaselineModeling class 
# tune hyperparameters (hyper parameter tuning is different for each algorithm?)
# output feature importance 

from BaselineModeling import BaselineModeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class RandomForestModel(BaselineModeling):
    """Inherits from baselinemodeling and makes it a bit more specific for the random forest model algorithm"""

    def __init__(self, path_to_X, target_column):
        # call the parent class constructor to initialize parent methods 
        super().__init__(path_to_X, target_column)

    
    def train(self, n_estimators=100, max_depth=None):
        
        print("Training Random Forest model...")

        # initialize the model 
        self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state= self.random_state,
                class_weight='balanced',
                n_jobs=-1
            )

        # train the model on x train 
        self.model.fit(self.X_train, self.y_train)
        print("Random Forest model training completed.")

    
    # grid search to find the optimal hyper parameters
    def hyperparameter_tuning(self, param_grid=None, cv=5):
            
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
        print("Performing hyperparameter tuning with grid search...")

        # Create base Random Forest model
        base_model = RandomForestClassifier(random_state = self.random_state)
        
        # Create GridSearchCV
        grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=cv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
        
        # training
        grid_search.fit(self.X_val, self.y_val)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best  ROC AUC score: {grid_search.best_score_:.4f}")
        
        # Update model with best estimator
        self.model = grid_search.best_estimator_
        print("Model updated with best parameters")


    # predict and evaluate using parent class methods

    def run_pipeline(self, numerical_columns, categorical_columns, store_model_path):

        # run through the methods
        self.load_data()
        self.split_data()
        self.normalize_standard(numerical_columns)
        self.encode_categorical(categorical_columns)
        self.train()
        self.hyperparameter_tuning()
        self.evaluate()
        self.save_model(store_model_path)

        

    