import pandas as pd

class Backtester:
    """Backtesting engine that simulates strategy trades on historical data."""
    def __init__(self, strategy, initial_capital=10000):
        self.strategy = strategy
        self.initial_capital = initial_capital

    def run(self, data):
        """
        Run backtest for the given strategy on the provided historical data (DataFrame with 'Close' prices).
        Returns a dictionary with equity curve and final results.
        """
        signals = self.strategy.generate_signals(data)
        cash = self.initial_capital
        shares = 0
        equity_curve = []
        in_position = False

        prices = data['Close']
        for date, price in prices.iteritems():
            signal = signals.loc[date] if isinstance(signals, pd.Series) else signals
            # Buy signal
            if signal == 1 or signal == 1.0:
                if not in_position:
                    # Buy as many shares as possible with available cash
                    shares_to_buy = cash // price
                    if shares_to_buy > 0:
                        cost = shares_to_buy * price
                        cash -= cost
                        shares += shares_to_buy
                        in_position = True
            # Sell signal
            if signal == -1 or signal == -1.0:
                if in_position and shares > 0:
                    # Sell all shares
                    cash += shares * price
                    shares = 0
                    in_position = False
            # Calculate equity value = cash + market value of shares
            equity = cash + shares * price
            equity_curve.append(equity)
        # If still holding shares at end, final equity already accounts for their value
        equity_series = pd.Series(equity_curve, index=data.index)
        result = {
            "equity_curve": equity_series,
            "final_equity": equity_series.iloc[-1]
        }
        return result
