"""
PlanningGraphDialog
-------------------

Este diálogo apresenta um exemplo de gráfico de planeamento de IA para
um processo de decisão simples no contexto de trading. Mostra os
estados possíveis (inicial, compra, manter, vender, stop-loss e
take-profit) e as transições entre eles. Serve como demonstração
conceptual de como um agente pode planear acções num espaço de
estados.

Não depende de networkx; o grafo é desenhado manualmente com
matplotlib.
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlanningGraphDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Planeamento IA - Exemplo de Grafo")
        layout = QVBoxLayout(self)
        fig = Figure(figsize=(6, 4))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.axis('off')
        # Define posições dos nós
        positions = {
            'Inicial': (0.1, 0.5),
            'Compra': (0.3, 0.5),
            'Manter': (0.5, 0.6),
            'Vender': (0.5, 0.4),
            'Stop-Loss': (0.7, 0.6),
            'Take-Profit': (0.7, 0.4),
        }
        # Desenha nós
        for node, (x, y) in positions.items():
            ax.text(x, y, node, ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3', fc='lightblue', ec='black'))
        # Desenha setas
        def draw_arrow(start, end):
            x1, y1 = positions[start]
            x2, y2 = positions[end]
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', lw=1.5))
        # Transições do grafo
        draw_arrow('Inicial', 'Compra')
        draw_arrow('Compra', 'Manter')
        draw_arrow('Compra', 'Vender')
        draw_arrow('Compra', 'Stop-Loss')
        draw_arrow('Compra', 'Take-Profit')
        draw_arrow('Manter', 'Vender')
        draw_arrow('Manter', 'Stop-Loss')
        draw_arrow('Manter', 'Take-Profit')
        fig.tight_layout()
        layout.addWidget(canvas)
        # Explicação
        explanation = ("Este grafo mostra um exemplo simplificado de planeamento: começando "
                       "no estado Inicial, o agente pode escolher Comprar. Uma vez comprado, "
                       "pode Manter a posição ou executar uma das ações de saída: Vender, "
                       "Stop-Loss ou Take-Profit. O grafo ajuda a visualizar as escolhas "
                       "possíveis e o espaço de estados.")
        lbl = QLabel(explanation)
        lbl.setWordWrap(True)
        layout.addWidget(lbl)