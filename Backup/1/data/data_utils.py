import pandas as pd

def calculate_daily_returns(price_series):
    """Calculate daily returns (percentage change) from a series of prices."""
    return price_series.pct_change().dropna()

def fill_missing_values(df):
    """Fill missing values in a DataFrame by forward-filling, then backward-filling as needed."""
    return df.ffill().bfill()
