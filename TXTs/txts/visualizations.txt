"""
Funções de visualização para a aplicação de trading.

Este módulo reúne diversos utilitários para plotagem de séries temporais financeiras,
indicadores técnicos, histogramas, matrizes de correlação, importância de features e
autocorrelação. As funções retornam figuras do Matplotlib ou eixos prontos a ser
incorporados em widgets PyQt5 ou guardados como imagens.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Iterable

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False

def _ensure_ax(ax=None, figsize=(8, 4)):
    """Garante que existe um eixo válido; se não, cria figura e eixo."""
    if ax is not None:
        return ax, None
    fig, ax = plt.subplots(figsize=figsize)
    return ax, fig

def plot_price_line(data: pd.DataFrame, ax: Optional[plt.Axes] = None):
    """Desenha um gráfico de linha do preço de fecho."""
    ax, fig = _ensure_ax(ax)
    ax.plot(data.index, data['Close'], label='Preço de Fecho', color='blue')
    ax.set_title('Preço de Fecho')
    ax.set_xlabel('Data')
    ax.set_ylabel('Preço')
    ax.legend()
    if fig:
        return fig
    return ax

def plot_volume(data: pd.DataFrame, ax: Optional[plt.Axes] = None):
    """Desenha um gráfico de barras do volume negociado."""
    ax, fig = _ensure_ax(ax)
    ax.bar(data.index, data['Volume'], color='grey', alpha=0.4)
    ax.set_title('Volume de Negociação')
    ax.set_xlabel('Data')
    ax.set_ylabel('Volume')
    if fig:
        return fig
    return ax

def plot_indicators(data: pd.DataFrame, indicators: Iterable[str], ax: Optional[plt.Axes] = None):
    """
    Desenha indicadores técnicos seleccionados sobrepostos ao preço de fecho.
    `indicators` deve ser uma lista de nomes de colunas existentes em `data`
    (por exemplo, 'sma20', 'bb_upper', 'bb_lower', etc.).
    """
    ax, fig = _ensure_ax(ax)
    # Garante que o preço de fecho é plotado como base
    ax.plot(data.index, data['Close'], label='Close', color='black', linewidth=1.0)
    colors = ['orange', 'green', 'red', 'purple', 'brown', 'pink', 'cyan']
    for idx, name in enumerate(indicators):
        if name in data.columns:
            ax.plot(data.index, data[name], label=name, color=colors[idx % len(colors)], linewidth=0.9)
    ax.set_title('Preço com Indicadores Técnicos')
    ax.set_xlabel('Data')
    ax.set_ylabel('Valor')
    ax.legend()
    if fig:
        return fig
    return ax

def plot_histogram(series: pd.Series, bins: int = 50, ax: Optional[plt.Axes] = None):
    """Plota um histograma (e curva KDE) de uma série numérica."""
    ax, fig = _ensure_ax(ax)
    sns.histplot(series.dropna(), bins=bins, kde=True, ax=ax, color='blue', alpha=0.6)
    ax.set_title('Histograma com KDE')
    ax.set_xlabel('Valor')
    ax.set_ylabel('Frequência')
    if fig:
        return fig
    return ax

def plot_correlation_matrix(df: pd.DataFrame, ax: Optional[plt.Axes] = None):
    """Desenha uma matriz de correlação em forma de heatmap."""
    ax, fig = _ensure_ax(ax)
    corr = df.corr()
    sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=False, fmt='.2f', cbar=True)
    ax.set_title('Matriz de Correlação')
    if fig:
        return fig
    return ax

def plot_feature_importance(importances: Iterable[tuple], top_n: int = 10, ax: Optional[plt.Axes] = None):
    """
    Desenha um gráfico de barras horizontal com as features mais importantes.
    `importances` é uma lista de tuplos (nome, valor), normalmente proveniente de `AIPredictor.get_last_feature_importance()`.
    """
    ax, fig = _ensure_ax(ax, figsize=(6, 4))
    # Selecciona as top_n features
    top_feats = importances[:top_n]
    names = [x[0] for x in top_feats]
    vals = [x[1] for x in top_feats]
    y_pos = np.arange(len(names))
    ax.barh(y_pos, vals, align='center', color='teal')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()  # bar em cima primeiro
    ax.set_title('Importância das Features')
    ax.set_xlabel('Peso')
    if fig:
        return fig
    return ax

def plot_acf_pacf(series: pd.Series, lags: int = 40, ax_acf: Optional[plt.Axes] = None, ax_pacf: Optional[plt.Axes] = None):
    """
    Plota os gráficos de ACF e PACF para uma série temporal, se `statsmodels` estiver disponível.
    Retorna (fig_acf, fig_pacf).
    """
    if not _HAS_STATSMODELS:
        raise ImportError("statsmodels não está instalado. plot_acf_pacf necessita desta biblioteca.")
    # Cria figuras se necessário
    ax_acf, fig1 = _ensure_ax(ax_acf)
    ax_pacf, fig2 = _ensure_ax(ax_pacf)
    plot_acf(series.dropna(), lags=lags, ax=ax_acf)
    plot_pacf(series.dropna(), lags=lags, ax=ax_pacf, method='ywm')
    ax_acf.set_title('Autocorrelação (ACF)')
    ax_pacf.set_title('Autocorrelação Parcial (PACF)')
    return (fig1 if fig1 else ax_acf), (fig2 if fig2 else ax_pacf)