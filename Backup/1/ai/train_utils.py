import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from data.data_provider import DataProvider
from indicators.ta import sma, rsi, macd

def compute_features(data):
    """
    Compute technical indicator features for a DataFrame with at least a 'Close' column.
    Returns a DataFrame of features (one row per date).
    """
    close = data['Close']
    features = pd.DataFrame(index=data.index)
    # Example features: 10-day SMA, 50-day SMA, RSI(14), MACD (12,26)
    features['SMA10'] = sma(close, window=10)
    features['SMA50'] = sma(close, window=50)
    features['RSI14'] = rsi(close, period=14)
    macd_line, signal_line = macd(close, fast=12, slow=26, signal=9)
    features['MACD_line'] = macd_line
    features['MACD_signal'] = signal_line
    # We could add more features (Bollinger Bands, etc.)
    return features

def train_models(data):
    """
    Train a logistic regression classifier and a random forest regressor using historical data.
    The classifier predicts up/down movement; the regressor predicts next-day price.
    Returns the trained models as a tuple: (classifier, regressor).
    """
    # Compute features
    features_df = compute_features(data.copy())
    # Create target variables
    # Future price for next day
    data['FuturePrice'] = data['Close'].shift(-1)
    # Direction: 1 if next day's close is higher, 0 otherwise
    data['Direction'] = np.where(data['FuturePrice'] > data['Close'], 1, 0)
    data.loc[data.index[-1], 'Direction'] = np.nan  # last day has no future price
    # Combine features and target into one frame, drop NaNs
    dataset = pd.concat([features_df, data[['FuturePrice', 'Direction']]], axis=1)
    dataset.dropna(inplace=True)
    # Define feature matrix X and targets
    feature_cols = features_df.columns
    X = dataset[feature_cols].values
    y_class = dataset['Direction'].values.astype(int)
    y_reg = dataset['FuturePrice'].values
    # Initialize models
    clf = LogisticRegression(max_iter=1000)
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    # Train models
    clf.fit(X, y_class)
    reg.fit(X, y_reg)
    return clf, reg

if __name__ == "__main__":
    # Example usage: train models on default data and save them
    import yaml
    # Load settings to get default ticker and model save paths
    with open("config/settings.yaml", 'r') as f:
        config = yaml.safe_load(f)
    tickers = config.get('default_tickers', [])
    if not tickers:
        raise ValueError("No default tickers specified in config.")
    ticker = tickers[0]
    data_provider = DataProvider(period=config.get('timeframe', '1y'), interval=config.get('interval', '1d'))
    data = data_provider.get_historical_data(ticker)
    if data is None or data.empty:
        raise RuntimeError(f"No data fetched for ticker {ticker}. Cannot train models.")
    # Train models
    clf_model, reg_model = train_models(data)
    # Save models to specified paths
    cls_path = config.get('model_classifier_path', 'ai/logistic_model.pkl')
    reg_path = config.get('model_regressor_path', 'ai/rf_model.pkl')
    joblib.dump(clf_model, cls_path)
    joblib.dump(reg_model, reg_path)
    print(f"Models trained and saved to {cls_path} and {reg_path}.")
