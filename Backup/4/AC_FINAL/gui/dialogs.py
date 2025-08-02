from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from gui.indicator_utils import analyse_indicators_custom
from gui.widgets.chart_widget import ChartWidget

# Importa ou replica este dicionário se não quiseres dependências circulares
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

class PredictionDialog(QDialog):
    def __init__(self, direction, proba, price_pred, features, multiclass=False, parent=None, class_labels=None):
        super().__init__(parent)
        self.setWindowTitle("Resultado da Previsão IA")
        layout = QVBoxLayout()
        # Mensagem textual principal
        if multiclass:
            if class_labels is None:
                class_labels = ["Queda", "Neutro", "Subida"]
            pred_txt = class_labels[direction]
            msg = f"<b>Previsão multi-classe:</b> <span style='color:blue'>{pred_txt}</span><br>"
            for i, lbl in enumerate(class_labels):
                msg += f"<b>Probabilidade {lbl}:</b> {proba[i]:.2%}<br>"
        else:
            pred_txt = "SUBIDA" if direction == 1 else "QUEDA"
            msg = f"<b>Previsão de tendência:</b> <span style='color: {'green' if direction==1 else 'red'}'>{pred_txt}</span><br>"
            msg += f"<b>Probabilidade de Subida:</b> {proba[1]:.2%}<br>"
            msg += f"<b>Probabilidade de Queda:</b> {proba[0]:.2%}<br>"
        msg += f"<b>Preço previsto (próximo fecho):</b> {price_pred:.2f}<br>"
        msg += "<b>Features utilizadas:</b><br>"
        for k, v in features.items():
            msg += f"&nbsp;&nbsp;{k}: {v:.4f}<br>"
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
        layout.addWidget(label)
        # Gráfico de barras das probabilidades
        fig, ax = plt.subplots(figsize=(3,2))
        if multiclass:
            if class_labels is None:
                class_labels = ["Queda", "Neutro", "Subida"]
            ax.bar(class_labels, proba, color=['red','orange','green'])
            ax.set_ylim(0,1)
        else:
            ax.bar(['Queda', 'Subida'], [proba[0], proba[1]], color=['red','green'])
            ax.set_ylim(0,1)
        ax.set_ylabel('Probabilidade')
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.setLayout(layout)
