class BaseFeatureEngineering:
    def __init__(self, feature_data, transactions_data):
        """
        Initializes the feature engineering class with required datasets.
        """
        self.feature_data = feature_data
        self.transactions_data = transactions_data

    def generateFeatures(self):
        raise NotImplementedError("Subclasses must implement generate_features()")
