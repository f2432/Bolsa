import numpy as np
import joblib
from ai import train_utils

class AIPredictor:
    """Loads trained AI models and makes predictions for stock movement and future price."""
    def __init__(self, classifier_model_path, regressor_model_path):
        try:
            self.classifier_model = joblib.load(classifier_model_path)
        except Exception:
            self.classifier_model = None
        try:
            self.regressor_model = joblib.load(regressor_model_path)
        except Exception:
            self.regressor_model = None

    def predict_direction(self, data):
        """
        Predict whether the price will go up or down (returns 1 for up, 0 for down).
        Uses the classifier model (e.g., logistic regression).
        """
        if self.classifier_model is None:
            return None
        # Compute latest features from data
        features_df = train_utils.compute_features(data.copy())
        features_df.dropna(inplace=True)
        if features_df.empty:
            return None
        X_latest = features_df.iloc[[-1]].values  # last row as 2D array
        pred = self.classifier_model.predict(X_latest)
        # Assuming model is binary classifier that outputs 0 or 1
        return int(pred[0])

    def predict_price(self, data):
        """
        Predict the future price (e.g., next day closing price).
        Uses the regressor model (e.g., Random Forest).
        """
        if self.regressor_model is None:
            return None
        features_df = train_utils.compute_features(data.copy())
        features_df.dropna(inplace=True)
        if features_df.empty:
            return None
        X_latest = features_df.iloc[[-1]].values
        pred_price = self.regressor_model.predict(X_latest)
        return float(pred_price[0])
