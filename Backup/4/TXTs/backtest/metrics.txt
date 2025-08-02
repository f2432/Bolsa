import numpy as np
import pandas as pd

def calculate_metrics(data, signals, equity_curve, trades_df=None):
    """
    Calcula métricas principais para avaliação do backtest.
    data: DataFrame de preços históricos
    signals: pd.Series com sinais (1, 0, -1)
    equity_curve: pd.Series com a evolução do capital
    trades_df: DataFrame com trades executados (opcional, para win rate)
    """
    if equity_curve is None or len(equity_curve) < 2:
        return {"retorno": "N/A", "drawdown": "N/A", "sharpe": "N/A", "num_trades": 0, "win_rate": "N/A"}

    # Retorno total
    retorno = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Drawdown máximo
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve / rolling_max - 1).min()

    # Sharpe ratio anualizado
    daily_ret = equity_curve.pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252) if daily_ret.std() != 0 else 0

    # Contagem de trades
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    num_trades = int(buy_signals + sell_signals)

    # Win rate: percentagem de trades com lucro
    win_rate = np.nan
    if trades_df is not None and not trades_df.empty and 'Resultado (€)' in trades_df.columns:
        vendas = trades_df[trades_df['Tipo'] == 'Venda']
        if not vendas.empty:
            num_wins = (vendas['Resultado (€)'] > 0).sum()
            win_rate = num_wins / len(vendas) if len(vendas) else np.nan

    return {
        "retorno": f"{retorno:.2%}",
        "drawdown": f"{drawdown:.2%}",
        "sharpe": f"{sharpe:.2f}",
        "num_trades": num_trades,
        "win_rate": f"{win_rate:.1%}" if not pd.isna(win_rate) else "N/A",
    }

