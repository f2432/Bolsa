import numpy as np

def total_return(equity_curve):
    """Total return of the strategy (as a fraction)."""
    if equity_curve is None or len(equity_curve) == 0:
        return 0.0
    initial = equity_curve.iloc[0]
    final = equity_curve.iloc[-1]
    return (final - initial) / initial

def max_drawdown(equity_curve):
    """Maximum drawdown (as a fraction)."""
    if equity_curve is None or len(equity_curve) == 0:
        return 0.0
    values = equity_curve.values
    cum_max = np.maximum.accumulate(values)
    drawdowns = (cum_max - values) / cum_max
    return np.max(drawdowns)

def sharpe_ratio(equity_curve, trading_days=252):
    """Sharpe Ratio (annualized, assuming zero risk-free rate)."""
    if equity_curve is None or len(equity_curve) < 2:
        return 0.0
    # Compute daily returns from equity curve
    returns = np.diff(equity_curve.values) / equity_curve.values[:-1]
    if returns.std() == 0:
        return 0.0
    # Sharpe: mean return / std dev of return * sqrt(trading_days)
    return (returns.mean() / returns.std()) * np.sqrt(trading_days)
