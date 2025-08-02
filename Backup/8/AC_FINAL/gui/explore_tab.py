from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton,
    QMessageBox, QComboBox, QLabel, QHBoxLayout
)
from gui.indicator_utils import analyse_indicators_custom
from gui.universe_utils import UNIVERSE_FUNCS
from gui.plot_utils import plot_signals_with_predictions

class ExploreTab(QWidget):
    def __init__(self, data_provider, predictor, portfolio_manager, indicator_analysis_text, parent=None):
        super().__init__(parent)
        self.data_provider = data_provider
        self.predictor = predictor
        self.portfolio_manager = portfolio_manager
        self.indicator_analysis_text = indicator_analysis_text
        self.parent = parent

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # --- Seletor de Universo (bolsa) ---
        universe_panel = QHBoxLayout()
        self.universo_combo = QComboBox()
        self.universo_combo.addItems(list(UNIVERSE_FUNCS.keys()))
        universe_panel.addWidget(QLabel("Bolsa/Índice:"))
        universe_panel.addWidget(self.universo_combo)
        self.layout.addLayout(universe_panel)

        self.reload_btn = QPushButton("Recarregar Universo (forçar online)")
        self.reload_btn.clicked.connect(self.reload_universe)
        self.layout.addWidget(self.reload_btn)

        # --- Tabela de sugestões ---
        self.suggestions_table = QTableWidget()
        self.suggestions_table.setColumnCount(4)
        self.suggestions_table.setHorizontalHeaderLabels(
            ["Ticker", "Score", "Sugestão", "Ver Detalhe"]
        )
        self.layout.addWidget(self.suggestions_table)
        refresh_sug_btn = QPushButton("Atualizar Sugestões")
        refresh_sug_btn.clicked.connect(self.atualizar_sugestoes)
        self.layout.addWidget(refresh_sug_btn)

        self.atualizar_sugestoes()

    def get_universe(self, nome, force_online=False):
        if nome in UNIVERSE_FUNCS:
            try:
                return UNIVERSE_FUNCS[nome](cache=not force_online)
            except Exception as e:
                QMessageBox.warning(self, "Erro a carregar universo", f"Erro ao carregar {nome}: {e}")
                return []
        return []

    def reload_universe(self):
        universo_nome = self.universo_combo.currentText()
        QMessageBox.information(self, "Recarregar", "A recarregar universo forçando download online.")
        universe = self.get_universe(universo_nome, force_online=True)
        if not universe:
            QMessageBox.warning(self, "Erro", f"Não foi possível recarregar o universo '{universo_nome}'.")
        else:
            QMessageBox.information(self, "Sucesso", f"Universo '{universo_nome}' recarregado!")
        self.atualizar_sugestoes()

    def atualizar_sugestoes(self):
        universo_nome = self.universo_combo.currentText()
        universe = self.get_universe(universo_nome)
        if not universe:
            QMessageBox.warning(self, "Erro", f"Não foi possível carregar o universo '{universo_nome}'.")
            self.suggestions_table.setRowCount(0)
            return

        portfolio_tickers = {pos['ticker'] for pos in self.portfolio_manager.positions}
        tickers_para_analise = [t for t in universe if t not in portfolio_tickers]
        self.suggestions_table.setRowCount(len(tickers_para_analise))
        self.suggestions_table.setColumnCount(5)
        self.suggestions_table.setHorizontalHeaderLabels(["Ticker", "Score", "Sugestão", "Ver Detalhe", "Ver Gráfico"])

                for i, ticker in enumerate(tickers_para_analise):
            btn = QPushButton("Ver Gráfico")
            btn.clicked.connect(lambda _, t=ticker: self.abrir_grafico_indicadores(t))
            self.suggestions_table.setCellWidget(i, 4, btn)
            try:
                print(f"Analisando {ticker}...")
                data = self.data_provider.get_historical_data(ticker)
                if data is None or data.empty or 'Close' not in data.columns:
                    score, sug, explic = "-", "Dados Insuf.", "Sem dados suficientes."
                else:
                    indicadores_txt = analyse_indicators_custom(data)
                    try:
                        self.predictor.train_on_data(data)
                        direction = self.predictor.predict_direction(data)
                        proba = self.predictor.predict_proba(data)
                    except Exception as e:
                        direction = None
                        proba = None
                        print(f"[ERRO IA] {ticker}: {e}")
                    score = 0
                    motivos = []
                    if "subida" in indicadores_txt.lower():
                        score += 1
                        motivos.append("Indicadores sugerem subida.")
                    if direction in [1, 2] and proba is not None and max(proba) > 0.6:
                        score += 2
                        motivos.append("IA prevê subida com confiança.")
                    if "descida" in indicadores_txt.lower():
                        score -= 1
                        motivos.append("Indicadores sugerem descida.")
                    if direction == 0 and proba is not None and proba[0] > 0.6:
                        score -= 2
                        motivos.append("IA prevê descida com confiança.")
                    if score >= 2:
                        sug = "Potencial Compra"
                    elif score <= -2:
                        sug = "Evitar"
                    else:
                        sug = "Neutro"
                    explic = "; ".join(motivos) + f"<br><b>Indicadores:</b> {indicadores_txt}"
                self.suggestions_table.setItem(i, 0, QTableWidgetItem(ticker))
                self.suggestions_table.setItem(i, 1, QTableWidgetItem(str(score)))
                self.suggestions_table.setItem(i, 2, QTableWidgetItem(sug))
                btn = QPushButton("Ver Detalhe")
                btn.clicked.connect(lambda _, t=ticker: self.mostrar_detalhe_acao(t))
                self.suggestions_table.setCellWidget(i, 3, btn)
            except Exception as err:
                print(f"[ERRO SUGESTÃO] {ticker}: {err}")

    def mostrar_detalhe_acao(self, ticker):
        data = self.data_provider.get_historical_data(ticker)
        if data is None or data.empty or 'Close' not in data.columns:
            QMessageBox.warning(self, "Detalhe", "Sem dados para análise.")
            return
        indicadores_txt = analyse_indicators_custom(data)
        self.indicator_analysis_text.setHtml(
            f"<b>Análise detalhada de {ticker}:</b><br>{indicadores_txt}<hr>"
        )
        self.parent.tabs.setCurrentWidget(self.parent.tabs.widget(0))
        self.parent.current_ticker = ticker
        self.parent.current_data = data
        self.parent.on_stock_selected()




    def abrir_grafico_indicadores(self, ticker):
        try:
            data = self.data_provider.get_historical_data(ticker)
            if data is None or data.empty:
                QMessageBox.warning(self, "Erro", f"Não foi possível obter dados para {ticker}")
                return
            signals = self.predictor.predict(data)
            predictions = None
            try:
                predictions = self.predictor.predict_price(data)
            except Exception as e:
                print(f"Erro ao obter previsão de preço: {e}")
            plot_signals_with_predictions(data, signals, predictions)
        except Exception as e:
            QMessageBox.critical(self, "Erro", f"Erro ao gerar gráfico: {e}")
