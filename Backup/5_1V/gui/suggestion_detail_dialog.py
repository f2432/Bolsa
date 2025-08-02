from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel, QTextEdit, QGridLayout
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd

# Explicações das features, adaptadas do PredictionDialog
FEATURE_EXPLANATIONS = {
    "sma20": lambda v, last: "Preço acima da SMA20 favorece subida." if last > v else "Preço abaixo da SMA20 favorece descida.",
    "rsi14": lambda v, _: (
        "RSI14 muito alto: sobrecomprado (pode descer)." if v > 70 else
        "RSI14 muito baixo: sobrevendido (pode subir)." if v < 30 else
        "RSI14 neutro."
    ),
    "macd": lambda v, signal: "MACD acima do sinal favorece subida." if v > signal else "MACD abaixo do sinal favorece descida.",
    "macd_signal": lambda v, macd: "",
    "bb_upper": lambda v, last: "Preço junto à banda superior: possível reversão para baixo." if last >= v else "",
    "bb_lower": lambda v, last: "Preço junto à banda inferior: possível reversão para cima." if last <= v else "",
    "adx": lambda v, _: "Tendência forte (ADX>25)." if v > 25 else "Tendência fraca (ADX<=25).",
    "cci": lambda v, _: (
        "CCI > 100: sobrecompra (pode descer)." if v > 100 else
        "CCI < -100: sobrevenda (pode subir)." if v < -100 else
        "CCI neutro."
    ),
    "atr": lambda v, _: f"Volatilidade {'alta' if v > 1 else 'baixa'} (ATR={v:.2f}).",
    "stoch_k": lambda v, _: (
        "Stoch > 80: sobrecomprado." if v > 80 else
        "Stoch < 20: sobrevendido." if v < 20 else
        "Stoch neutro."
    ),
    "obv": lambda v, _: "OBV em subida sugere fluxo comprador." if v > 0 else "OBV a descer sugere fluxo vendedor.",
    "mfi": lambda v, _: (
        "MFI > 80: excesso de compras." if v > 80 else
        "MFI < 20: excesso de vendas." if v < 20 else
        "MFI neutro."
    ),
    "bullish_engulfing": lambda v, _: "Padrão Bullish Engulfing: potencial inversão para cima." if v else "",
    "bearish_engulfing": lambda v, _: "Padrão Bearish Engulfing: potencial inversão para baixo." if v else "",
}

def format_percent(val):
    try:
        if val is not None and not pd.isna(val):
            return f"{float(val):.2%}"
    except Exception:
        pass
    return "N/A"

def format_float(val, prec=2):
    try:
        if val is not None and not pd.isna(val):
            return f"{float(val):.{prec}f}"
    except Exception:
        pass
    return "N/A"

class SuggestionDetailDialog(QDialog):
    def __init__(self, ticker, results_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Análise Detalhada: {ticker}")
        layout = QVBoxLayout()
        tabs = QTabWidget()

        for model, res in results_dict.items():
            for h in [1, 3]:
                tab = QWidget()
                tab_layout = QVBoxLayout()

                # Probabilidades
                proba = res.get(f"proba_{h}d", None)
                if proba is None:
                    prob_up = res.get(f"ProbSubida_{h}d", None)
                    prob_down = 1 - prob_up if prob_up is not None and not pd.isna(prob_up) else None
                    proba = [prob_down, prob_up]
                elif isinstance(proba, (list, tuple)) and len(proba) == 2:
                    prob_up = proba[1]
                    prob_down = proba[0]
                else:
                    prob_up = prob_down = None

                # Preço previsto
                price_pred = res.get(f"PrecoPrev_{h}d", None) or res.get(f"price_pred_{h}d", None)

                # Sugestão
                sug = res.get(f"Sugestao_{h}d", "") or res.get(f"sugestao_{h}d", "")

                # Features
                features = res.get(f"features_{h}d", {}) or res.get("features", {})
                if isinstance(features, str):
                    try:
                        import ast
                        features = ast.literal_eval(features)
                    except Exception:
                        features = {}

                # Header
                direction = "SUBIDA" if prob_up and prob_up > 0.5 else "QUEDA"
                tab_html = f"""
                <b>Previsão de tendência:</b> <span style='color:{'green' if direction=='SUBIDA' else 'red'}'>{direction}</span><br>
                <b>Probabilidade de Subida:</b> {format_percent(prob_up)}<br>
                <b>Probabilidade de Queda:</b> {format_percent(prob_down)}<br>
                <b>Preço previsto (próximo fecho):</b> {format_float(price_pred)}<br>
                """

                # Features e explicação das features
                if features:
                    tab_html += "<b>Features utilizadas:</b><br>"
                    for k, v in features.items():
                        tab_html += f"&nbsp;&nbsp;{k}: {v:.4f}<br>"

                # Explicação das features
                if features:
                    tab_html += "<b>Explicação das features:</b><br>"
                    for k, v in features.items():
                        if k in FEATURE_EXPLANATIONS:
                            context_val = 0
                            if k == "macd":
                                context_val = features.get("macd_signal", 0)
                            elif k == "macd_signal":
                                context_val = features.get("macd", 0)
                            elif k in ["sma20", "bb_upper", "bb_lower"]:
                                context_val = features.get("last_close", v)
                            explicacao = FEATURE_EXPLANATIONS[k](v, context_val)
                            if explicacao:
                                tab_html += f"&nbsp;&nbsp;<i>{k}:</i> {explicacao}<br>"

                # Sugestão IA
                tab_html += f"<b>Sugestão IA:</b> {sug}<br>"

                # Mostra HTML
                label = QLabel()
                label.setTextFormat(1)
                label.setText(tab_html)
                tab_layout.addWidget(label)

                # Gráfico das probabilidades
                if prob_up is not None and prob_down is not None and not pd.isna(prob_up) and not pd.isna(prob_down):
                    fig, ax = plt.subplots(figsize=(3, 2))
                    ax.bar(['Queda', 'Subida'], [prob_down, prob_up], color=['red', 'green'])
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Probabilidade')
                    ax.set_title(f"{model.upper()} - {h} Dias")
                    canvas = FigureCanvas(fig)
                    tab_layout.addWidget(canvas)

                tab.setLayout(tab_layout)
                tabs.addTab(tab, f"{model.upper()} - {h} Dia{'s' if h>1 else ''}")

        layout.addWidget(tabs)
        self.setLayout(layout)

