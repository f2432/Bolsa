"""
FactorAttributionDialog
-----------------------

Este diálogo realiza uma atribuição das importâncias das features
por factores, agrupando indicadores em categorias (tendência, momento,
volatilidade, volume, padrões, etc.). Recebe uma lista de tuples
(feature, importance) e devolve um gráfico de barras ou pizza
mostrando a contribuição percentual de cada factor.

Uso:
    dlg = FactorAttributionDialog(importances)
    dlg.exec_()

O diálogo identifica categorias de acordo com substrings nas chaves
das features. Se uma feature não se enquadrar em nenhuma categoria
predefinida, será colocada em "Outros".
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class FactorAttributionDialog(QDialog):
    def __init__(self, importances: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Atribuição por Factores")
        layout = QVBoxLayout(self)
        if not importances:
            layout.addWidget(QLabel("Sem dados de importâncias para atribuir."))
            return
        # Define categorias e critérios de agrupamento (lowercase)
        categories = {
            'Tendência': ['sma', 'ema', 'trend'],
            'Momento': ['rsi', 'macd', 'stoch', 'mfi'],
            'Volatilidade': ['atr', 'bb', 'vol'],
            'Volume': ['obv', 'volume', 'avg_vol'],
            'Padrões': ['bullish', 'bearish', 'pattern'],
        }
        # Inicializa acumuladores
        factor_totals = {cat: 0.0 for cat in categories}
        factor_totals['Outros'] = 0.0
        total_importance = sum(abs(val) for (_, val) in importances) or 1.0
        # Agrupa importâncias
        for feat, val in importances:
            key = feat.lower()
            assigned = False
            for cat, substrs in categories.items():
                if any(sub in key for sub in substrs):
                    factor_totals[cat] += abs(val)
                    assigned = True
                    break
            if not assigned:
                factor_totals['Outros'] += abs(val)
        # Converte para percentagem
        factor_percent = {k: (v / total_importance) for k, v in factor_totals.items() if v > 0}
        # Cria gráfico
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        labels = list(factor_percent.keys())
        values = [factor_percent[l] for l in labels]
        bars = ax.bar(labels, values, color='skyblue')
        ax.set_ylabel('Proporção da Importância')
        ax.set_title('Atribuição da Importância por Factores')
        ax.set_ylim(0, max(values) * 1.2)
        # Anota valores percentuais
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{val*100:.1f}%", ha='center', va='bottom')
        fig.tight_layout()
        layout.addWidget(canvas)
        # Mostra tabela de mapeamento (opcional)
        mapping_html = "<b>Mapping:</b><br>"
        for cat, substrs in categories.items():
            mapping_html += f"{cat}: {', '.join(substrs)}<br>"
        mapping_html += "Outros: resto das features"
        lbl_mapping = QLabel(mapping_html)
        lbl_mapping.setTextFormat(1)
        layout.addWidget(lbl_mapping)