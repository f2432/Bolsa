from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTabWidget, QWidget, QLabel, QTextEdit
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import pandas as pd
import ast

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

def parse_features(val):
    if isinstance(val, str):
        try:
            return ast.literal_eval(val)
        except Exception:
            return {}
    return val if isinstance(val, dict) else {}

class SuggestionDetailDialog(QDialog):
    def __init__(self, ticker, results_dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Análise Detalhada: {ticker}")
        layout = QVBoxLayout()
        tabs = QTabWidget()

        # Mostra preço atual (média dos modelos, se vários, ou qualquer um)
        preco_atual = None
        for model_data in results_dict.values():
            preco = model_data.get("preco_atual", None)
            if preco is not None and not pd.isna(preco):
                preco_atual = preco
                break
        if preco_atual is not None:
            label_preco = QLabel(f"<b>Preço Atual:</b> {format_float(preco_atual, 4)}")
            label_preco.setTextFormat(1)
            layout.addWidget(label_preco)

        for model_name, result in results_dict.items():
            for horizon, horizon_label in [(1, "1 Dia"), (3, "3 Dias")]:
                tab = QWidget()
                vbox = QVBoxLayout()
                proba = result.get(f"proba_{horizon}d", None)
                price_pred = result.get(f"price_pred_{horizon}d", None)
                sugestao = result.get(f"sugestao_{horizon}d", "")
                features = parse_features(result.get(f"features_{horizon}d", {}))
                class_labels = ["Queda", "Subida"]

                direction = None
                if proba and len(proba) >= 2:
                    if all([not pd.isna(x) for x in proba[:2]]):
                        direction = 1 if proba[1] > 0.5 else 0

                # Título previsão
                if direction is not None:
                    pred_txt = "SUBIDA" if direction in [1,2] else "QUEDA"
                    cor = "green" if direction in [1,2] else "red"
                    msg = f"<b>Previsão de tendência:</b> <span style='color:{cor}'>{pred_txt}</span><br>"
                else:
                    msg = "<b>Previsão:</b> N/A<br>"

                # Probabilidades
                if proba is not None and len(proba) >= 2:
                    msg += f"<b>Probabilidade de Subida:</b> {format_percent(proba[1])}<br>"
                    msg += f"<b>Probabilidade de Queda:</b> {format_percent(proba[0])}<br>"
                else:
                    msg += "<b>Probabilidades:</b> N/A<br>"

                # Preço previsto
                msg += f"<b>Preço previsto ({horizon_label}):</b> {format_float(price_pred)}<br>"

                # Sugestão IA
                if sugestao:
                    msg += f"<b>Sugestão IA:</b> {sugestao}<br>"

                # Features usadas (valores)
                if features:
                    msg += "<b>Features utilizadas:</b><br>"
                    for k, v in features.items():
                        msg += f"&nbsp;&nbsp;{k}: {format_float(v, 4)}<br>"

                # Explicação das features
                if features:
                    msg += "<b>Explicação das features:</b><br>"
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
                                msg += f"&nbsp;&nbsp;<i>{k}:</i> {explicacao}<br>"

                label = QLabel(msg)
                label.setTextFormat(1)
                label.setWordWrap(True)
                vbox.addWidget(label)

                # Gráfico das probabilidades
                if proba is not None and len(proba) >= 2 and all([not pd.isna(x) for x in proba[:2]]):
                    fig, ax = plt.subplots(figsize=(3,2))
                    ax.bar(class_labels, proba[:2], color=['red','green'])
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Probabilidade')
                    ax.set_title(f"{model_name.upper()} - {horizon_label}")
                    canvas = FigureCanvas(fig)
                    vbox.addWidget(canvas)

                tab.setLayout(vbox)
                tabs.addTab(tab, f"{model_name.upper()} - {horizon_label}")

        layout.addWidget(tabs)
        self.setLayout(layout)
