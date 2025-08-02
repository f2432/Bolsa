import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QAbstractItemView, QMessageBox, QSizePolicy,
    QInputDialog, QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QTextEdit, QCheckBox, QDateEdit, QTabWidget,
    QDialog  # <-- Este importa o QDialog!
)

from PyQt5.QtCore import QDate
from gui.widgets.chart_widget import ChartWidget
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_macd_combo import RSIMACDStrategy
from prediction_log import PredictionLogger
import logging

logger = logging.getLogger(__name__)

class PredictionDialog(QDialog):
    def __init__(self, direction, proba, price_pred, features, multiclass=False, parent=None, class_labels=None):
        super().__init__(parent)
        self.setWindowTitle("Resultado da Previsão IA")
        layout = QVBoxLayout()
        # Mensagem textual
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
        label = QLabel(msg)
        label.setTextFormat(1)
        layout.addWidget(label)
        # Gráfico de barras das probabilidades
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
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

class MainWindow(QMainWindow):
    def __init__(self, data_provider, portfolio_manager, predictor, initial_capital, tickers):
        super().__init__()
        self.data_provider = data_provider
        self.portfolio_manager = portfolio_manager
        self.predictor = predictor
        self.pred_logger = PredictionLogger()
        self.initial_capital = initial_capital
        self.tickers = tickers[:]
        portfolio_tickers = [pos['ticker'] for pos in self.portfolio_manager.positions]
        for t in portfolio_tickers:
            if t not in self.tickers:
                self.tickers.append(t)
        self.current_ticker = None
        self.current_data = None

        self.setWindowTitle("Trading Application")
        self.resize(1600, 950)

        # Tabs
        self.tabs = QTabWidget()
        main_widget = QWidget()
        self.tabs.addTab(main_widget, "Principal")
        self.dashboard = QTextEdit()
        self.dashboard.setReadOnly(True)
        self.tabs.addTab(self.dashboard, "Dashboard")
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Stock table
        self.stock_table = QTableWidget(len(self.tickers), 2)
        self.stock_table.setHorizontalHeaderLabels(["Ticker", "Preço"])
        self.stock_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stock_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.stock_table.setSelectionMode(QAbstractItemView.SingleSelection)
        for i, ticker in enumerate(self.tickers):
            self.stock_table.setItem(i, 0, QTableWidgetItem(ticker))
            price = self.data_provider.get_current_price(ticker)
            price_text = f"{price:.2f}" if price is not None else "N/A"
            self.stock_table.setItem(i, 1, QTableWidgetItem(price_text))
        self.stock_table.resizeColumnsToContents()
        self.stock_table.setFixedWidth(self.stock_table.horizontalHeader().length() + 20)
        self.stock_table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.stock_table.itemSelectionChanged.connect(self.on_stock_selected)

        # Right panel (graph + controls)
        right_panel = QVBoxLayout()
        self.chart_widget = ChartWidget()
        self.chart_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.addWidget(self.chart_widget)
        self.price_label = QLabel("Preço Atual: ")
        right_panel.addWidget(self.price_label)

        # Estratégia
        strategy_box = QGroupBox("Estratégia & Parâmetros")
        strategy_layout = QFormLayout()
        self.strategy_classes = {
            "SMA Crossover": SMACrossoverStrategy,
            "RSI+MACD": RSIMACDStrategy,
        }
        self.strategy_params = {
            "SMA Crossover": {"short_window": 50, "long_window": 200},
            "RSI+MACD": {"rsi_buy_threshold": 30, "rsi_sell_threshold": 70}
        }
        self.selected_strategy = "SMA Crossover"
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(list(self.strategy_classes.keys()))
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_change)
        strategy_layout.addRow("Estratégia:", self.strategy_combo)
        self.param_widgets = {}
        self.build_param_widgets(strategy_layout, self.selected_strategy)
        strategy_box.setLayout(strategy_layout)
        right_panel.addWidget(strategy_box)

        # IA parâmetros
        ia_box = QGroupBox("Opções de IA / Previsão")
        ia_layout = QHBoxLayout()
        self.model_selector = QComboBox()
        self.model_selector.addItems(["Logistic", "RF", "MLP"])
        ia_layout.addWidget(QLabel("Modelo:"))
        ia_layout.addWidget(self.model_selector)
        self.multiclass_checkbox = QCheckBox("Previsão Multi-classe (Queda/Neutro/Subida)")
        ia_layout.addWidget(self.multiclass_checkbox)
        self.n_ahead_spin = QSpinBox()
        self.n_ahead_spin.setMinimum(1)
        self.n_ahead_spin.setMaximum(10)
        self.n_ahead_spin.setValue(1)
        ia_layout.addWidget(QLabel("Dias à frente:"))
        ia_layout.addWidget(self.n_ahead_spin)
        ia_box.setLayout(ia_layout)
        right_panel.addWidget(ia_box)

        # Botões de ação
        button_panel = QHBoxLayout()
        button_defs = [
            ("Atualizar Dados", self.refresh_data),
            ("Backtest Estratégia", self.run_backtest),
            ("Treinar IA", self.train_ia),
            ("Guardar Modelo", self.save_model),
            ("Carregar Modelo", self.load_model),
            ("Prever (IA)", self.run_prediction),
            ("Adicionar Posição", self.add_portfolio_position),
            ("Mostrar Indicadores", self.show_indicators),
            ("Mostrar Portefólio", self.show_portfolio),
            ("Guardar Gráfico", self.save_chart),
            ("Limpar Gráfico", self.clear_chart),
            ("Mostrar Trades", self.show_trades),
            ("Exportar Portefólio", self.export_portfolio),
            ("Mostrar Dashboard", self.update_dashboard),
        ]
        for text, callback in button_defs:
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            button_panel.addWidget(btn)
        help_btn = QPushButton("?")
        help_btn.setToolTip("Explicação dos indicadores técnicos")
        help_btn.clicked.connect(self.show_indicators_explanation)
        button_panel.addWidget(help_btn)
        right_panel.addLayout(button_panel)

        self.prediction_label = QLabel("Previsão: N/A")
        right_panel.addWidget(self.prediction_label)
        self.indicator_analysis_label = QLabel("<b>Interpretação dos Indicadores:</b>")
        self.indicator_analysis_text = QTextEdit()
        self.indicator_analysis_text.setReadOnly(True)
        right_panel.addWidget(self.indicator_analysis_label)
        right_panel.addWidget(self.indicator_analysis_text)
        main_layout.addWidget(self.stock_table)
        main_layout.addLayout(right_panel)
        self.setCentralWidget(self.tabs)
        if self.tickers:
            self.stock_table.selectRow(0)
            self.on_stock_selected()

    def refresh_data(self):
        for i, ticker in enumerate(self.tickers):
            price = self.data_provider.get_current_price(ticker)
            price_text = f"{price:.2f}" if price is not None else "N/A"
            self.stock_table.setItem(i, 1, QTableWidgetItem(price_text))
        if self.current_ticker:
            data = self.data_provider.get_historical_data(self.current_ticker, refresh=True)
            if data is not None and not data.empty:
                canonical_index = pd.bdate_range(start=data.index.min(), end=data.index.max(), freq='C')
                data = data.reindex(canonical_index).ffill()
                data.index.name = 'Date'
                self.current_data = data
                self.on_stock_selected()

    def run_backtest(self):
        if self.current_data is None or self.current_ticker is None:
            QMessageBox.information(self, "Backtest", "Nenhuma ação selecionada para backtesting.")
            return
        from backtest.backtester import Backtester
        try:
            signals = self.strategy.generate_signals(self.current_data)
            backtester = Backtester(initial_capital=self.initial_capital)
            results = backtester.run(self.current_data, signals)
            equity = results["equity_curve"]
            trades_df = results.get("trades", None)
            from backtest.metrics import calculate_metrics
            metrics = calculate_metrics(self.current_data, signals, equity, trades_df)
            msg = (f"<b>Estratégia:</b> {self.strategy.__class__.__name__}<br>"
                   f"<b>Retorno Total:</b> {metrics['retorno']}<br>"
                   f"<b>Max Drawdown:</b> {metrics['drawdown']}<br>"
                   f"<b>Sharpe Ratio:</b> {metrics['sharpe']}<br>"
                   f"<b>Num. Trades:</b> {metrics['num_trades']}<br>"
                   f"<b>Win Rate:</b> {metrics['win_rate']}<br>")
            QMessageBox.information(self, "Resultados do Backtest", msg)
        except Exception as e:
            QMessageBox.critical(self, "Erro de Backtest", str(e))
            logger.error(f"Backtest error: {e}")

    def train_ia(self):
        if self.current_data is None or self.current_data.empty:
            QMessageBox.warning(self, "Treinar IA", "Não existem dados carregados para treinar a IA.")
            return
        try:
            model_type = self.model_selector.currentText().lower()
            multiclass = self.multiclass_checkbox.isChecked()
            n_ahead = self.n_ahead_spin.value()
            self.predictor.model_type = model_type
            self.predictor.multiclass = multiclass
            self.predictor.n_ahead = n_ahead
            self.predictor.train_on_data(self.current_data, model_type=model_type)
            QMessageBox.information(self, "Treinar IA", f"Modelo '{model_type}' {'multi-class' if multiclass else ''} treinado com sucesso!\nAgora podes usar o botão 'Prever (IA)'.")
            score = self.predictor.get_last_cv_score()
            overfit = self.predictor.get_last_overfit_warning()
            feat_imp = self.predictor.get_last_feature_importance()
            self.indicator_analysis_text.append(f"<hr><b>Validação Modelo:</b><br>{score}<br>{overfit}")
            self.indicator_analysis_text.append(f"<hr><b>Features mais importantes:</b><br>{feat_imp}")
        except Exception as e:
            QMessageBox.critical(self, "Erro IA", f"Erro ao treinar IA: {e}")

    def run_prediction(self):
        if self.current_data is None or self.current_ticker is None:
            QMessageBox.information(self, "Prever", "Nenhuma ação selecionada para previsão.")
            return
        try:
            multiclass = getattr(self.predictor, 'multiclass', False)
            direction = self.predictor.predict_direction(self.current_data)
            proba = self.predictor.predict_proba(self.current_data)
            price_pred = self.predictor.predict_price(self.current_data)
            features = self.predictor.get_last_features(self.current_data)
            current_price = self.current_data['Close'].iloc[-1]
            if direction is None or proba is None or price_pred is None:
                self.prediction_label.setText("Previsão: Modelo não treinado.")
                return
            # Guarda a previsão
            self.pred_logger.log(
                data=self.current_data.index[-1], ticker=self.current_ticker,
                modelo=self.predictor.model_type, n_ahead=self.predictor.n_ahead,
                multiclass=self.predictor.multiclass, direcao=direction,
                proba=proba, preco_real=current_price, preco_prev=price_pred
            )
            indicadores_txt = self.analyse_indicators()
            consenso_html = self.mostrar_consenso(direction, indicadores_txt)
            dica_html = self.dica_de_ouro(direction, proba, price_pred, indicadores_txt)
            self.indicator_analysis_text.setHtml(indicadores_txt + "<hr>" + consenso_html + "<hr>" + dica_html)
            self.chart_widget.ax.plot(
                [self.current_data.index[-1], self.current_data.index[-1] + pd.Timedelta(days=1)],
                [current_price, price_pred],
                'ro--', label='Previsão IA'
            )
            self.chart_widget.ax.legend()
            self.chart_widget.canvas.draw()
            class_labels = ["Queda", "Neutro", "Subida"] if multiclass else ["Queda", "Subida"]
            dlg = PredictionDialog(direction, proba, price_pred, features, multiclass=multiclass, class_labels=class_labels)
            dlg.exec_()
        except Exception as e:
            logger.error(f"Erro na previsão IA: {e}")
            QMessageBox.critical(self, "Erro IA", f"Erro na previsão IA: {e}")

    def add_portfolio_position(self):
        ticker, ok1 = QInputDialog.getText(self, "Adicionar Posição", "Símbolo do ticker:")
        if not ok1 or ticker.strip() == "":
            return
        qty, ok2 = QInputDialog.getInt(self, "Adicionar Posição", "Quantidade:", value=0, min=0)
        if not ok2:
            return
        price, ok3 = QInputDialog.getDouble(self, "Adicionar Posição", "Preço de compra:", decimals=2)
        if not ok3:
            return
        date_edit = QDateEdit()
        date_edit.setDate(QDate.currentDate())
        date_edit.setCalendarPopup(True)
        dlg = QDialog(self)
        dlg.setWindowTitle("Data de Compra")
        vbox = QVBoxLayout()
        vbox.addWidget(QLabel("Escolha a data de aquisição:"))
        vbox.addWidget(date_edit)
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(dlg.accept)
        vbox.addWidget(btn_ok)
        dlg.setLayout(vbox)
        if not dlg.exec_():
            return
        buy_date = date_edit.date().toString("yyyy-MM-dd")
        self.portfolio_manager.add_position(ticker.strip().upper(), qty, price, buy_date)
        if ticker.strip().upper() not in self.tickers:
            self.tickers.append(ticker.strip().upper())
            new_row = self.stock_table.rowCount()
            self.stock_table.insertRow(new_row)
            self.stock_table.setItem(new_row, 0, QTableWidgetItem(ticker.strip().upper()))
            self.stock_table.setItem(new_row, 1, QTableWidgetItem(f"{price:.2f}"))
            self.stock_table.resizeColumnsToContents()
        metrics = self.portfolio_manager.calculate_metrics(self.data_provider)
        total_val = metrics['total_value']
        total_profit = metrics['total_profit']
        QMessageBox.information(self, "Portefólio Atualizado",
                                f"Valor Total: ${total_val:,.2f}\nLucro/Prejuízo: ${total_profit:,.2f}")

    def on_stock_selected(self):
        row = self.stock_table.currentRow()
        if row < 0:
            return
        ticker = self.stock_table.item(row, 0).text()
        self.current_ticker = ticker
        data = self.data_provider.get_historical_data(ticker)
        if data is None or not isinstance(data, pd.DataFrame) or 'Close' not in data.columns:
            QMessageBox.warning(self, "Erro de Dados",
                f"Nenhum dado válido para {ticker}.\nColunas disponíveis: {list(data.columns) if isinstance(data, pd.DataFrame) else data}")
            self.current_data = None
            self.indicator_analysis_text.setText("Sem dados para análise.")
            return
        canonical_index = pd.bdate_range(start=data.index.min(), end=data.index.max(), freq='C')
        data = data.reindex(canonical_index).ffill()
        data.index.name = 'Date'
        self.current_data = data
        posicoes = [p for p in self.portfolio_manager.positions if p['ticker'] == ticker]
        buy_dates = [pd.to_datetime(p['buy_date']) for p in posicoes if p.get('buy_date')]
        buy_prices = [p['buy_price'] for p in posicoes]
        df_pred = self.pred_logger.get_log(ticker)
        if not df_pred.empty:
            pred_dates = [pd.to_datetime(row['Data']) for idx, row in df_pred.iterrows() if pd.to_datetime(row['Data']) in data.index]
            pred_prices = [row['Preco_Previsto'] for idx, row in df_pred.iterrows() if pd.to_datetime(row['Data']) in data.index]
            pred_directions = [row['Direcao'] for idx, row in df_pred.iterrows() if pd.to_datetime(row['Data']) in data.index]
        else:
            pred_dates, pred_prices, pred_directions = [], [], []
        self.chart_widget.plot(
            data['Close'], title=f"{ticker} Price",
            buy_dates=buy_dates, buy_prices=buy_prices,
            pred_dates=pred_dates, pred_prices=pred_prices, pred_directions=pred_directions
        )
        current_price = self.data_provider.get_current_price(ticker)
        if current_price is None:
            current_price = data['Close'].iloc[-1]
        self.price_label.setText(f"Preço Atual: {current_price:.2f}")
        interpretacao = self.analyse_indicators()
        self.indicator_analysis_text.setText(interpretacao)

    def show_indicators(self):
        if self.current_data is not None:
            from indicators.ta import sma, rsi, macd, bollinger_bands
            close = self.current_data['Close']
            sma20 = sma(close, 20)
            rsi14 = rsi(close, 14)
            macd_line, macd_signal = macd(close)
            bb_sma, bb_upper, bb_lower = bollinger_bands(close)
            self.chart_widget.ax.clear()
            self.chart_widget.ax.plot(close.index, close.values, label='Preço')
            self.chart_widget.ax.plot(sma20.index, sma20.values, label='SMA20')
            self.chart_widget.ax.plot(rsi14.index, rsi14.values, label='RSI14')
            self.chart_widget.ax.plot(macd_line.index, macd_line.values, label='MACD')
            self.chart_widget.ax.plot(macd_signal.index, macd_signal.values, label='MACD Signal')
            self.chart_widget.ax.plot(bb_upper.index, bb_upper.values, '--', label='BB Upper')
            self.chart_widget.ax.plot(bb_lower.index, bb_lower.values, '--', label='BB Lower')
            self.chart_widget.ax.set_title("Indicadores Técnicos")
            self.chart_widget.ax.legend()
            self.chart_widget.canvas.draw()
            interpretacao = self.analyse_indicators()
            self.indicator_analysis_text.setText(interpretacao)

    def show_portfolio(self):
        metrics = self.portfolio_manager.calculate_metrics(self.data_provider)
        msg = ""
        for pos in metrics['positions']:
            msg += f"{pos['ticker']}: {pos['quantity']} @ {pos['buy_price']} | Atual: {pos['current_price']} | P/L: {pos['profit']:.2f}\n"
        msg += f"\nValor Total: {metrics['total_value']:.2f}\nLucro Total: {metrics['total_profit']:.2f}"
        if not metrics['positions']:
            msg = "Portefólio vazio."
        QMessageBox.information(self, "Portefólio Atual", msg)

    def show_trades(self):
        try:
            from backtest.backtester import Backtester
            backtester = Backtester(initial_capital=self.initial_capital)
            signals = self.strategy.generate_signals(self.current_data)
            results = backtester.run(self.current_data, signals)
            if 'trades' in results:
                trades = results['trades']
                msg = "Data\tTipo\tPreço\tQtd\n"
                for _, tr in trades.iterrows():
                    msg += f"{tr['Data']}\t{tr['Tipo']}\t{tr['Preço']}\t{tr['Quantidade']}\n"
            else:
                msg = "Sem trades registadas."
            QMessageBox.information(self, "Trades", msg)
        except Exception as e:
            QMessageBox.critical(self, "Erro Trades", str(e))

    def save_model(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Modelo IA", "", "Modelos IA (*.pkl)")
        if file_path:
            try:
                self.predictor.save_model(file_path)
                QMessageBox.information(self, "Guardar Modelo", "Modelo IA guardado com sucesso.")
            except Exception as e:
                QMessageBox.critical(self, "Guardar Modelo", f"Erro ao guardar modelo: {e}")

    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Carregar Modelo IA", "", "Modelos IA (*.pkl)")
        if file_path:
            try:
                self.predictor.load_model(file_path)
                QMessageBox.information(self, "Carregar Modelo", "Modelo IA carregado com sucesso.")
            except Exception as e:
                QMessageBox.critical(self, "Carregar Modelo", f"Erro ao carregar modelo: {e}")

    def save_chart(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Guardar Gráfico", "", "PNG Image (*.png)")
        if file_path:
            self.chart_widget.figure.savefig(file_path)
            QMessageBox.information(self, "Gráfico Guardado", f"Gráfico guardado em:\n{file_path}")

    def clear_chart(self):
        self.chart_widget.ax.clear()
        self.chart_widget.canvas.draw()

    def export_portfolio(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Exportar Portefólio", "", "CSV Files (*.csv)")
        if file_path:
            import pandas as pd
            metrics = self.portfolio_manager.calculate_metrics(self.data_provider)
            df = pd.DataFrame(metrics['positions'])
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Exportar Portefólio", f"Portefólio exportado para:\n{file_path}")

    def update_dashboard(self):
        metrics = self.portfolio_manager.calculate_metrics(self.data_provider)
        msg = "<b>Resumo do Portefólio:</b><br>"
        for pos in metrics['positions']:
            msg += f"{pos['ticker']}: {pos['quantity']} @ {pos['buy_price']} ({pos['buy_date']}) | Atual: {pos['current_price']} | P/L: {pos['profit']:.2f}<br>"
        msg += f"<br><b>Valor Total:</b> {metrics['total_value']:.2f}<br><b>Lucro Total:</b> {metrics['total_profit']:.2f}<br><hr>"
        msg += "<b>Últimas Previsões:</b><br>"
        df_pred = self.pred_logger.get_log(self.current_ticker) if self.current_ticker else self.pred_logger.get_log()
        if not df_pred.empty:
            ultimas = df_pred.tail(10)
            msg += ultimas.to_html(index=False)
        else:
            msg += "Sem previsões registadas."
        self.dashboard.setHtml(msg)

    def show_indicators_explanation(self):
        text = '''
        <b>Como interpretar os Indicadores Técnicos:</b><br>
        <ul>
        <li><b>Preço:</b> Linha azul, preço de fecho diário da ação.</li>
        <li><b>SMA20:</b> Linha laranja. Preço acima = subida; abaixo = descida.</li>
        <li><b>RSI14:</b> Linha verde, 0-100. >70: sobrecomprado (venda); <30: sobrevendido (compra).</li>
        <li><b>MACD/MACD Signal:</b> Vermelha/roxa. MACD cruza Signal para cima = compra; para baixo = venda.</li>
        <li><b>Bandas de Bollinger:</b> Tracejadas. Preço junto à superior: sobrecomprado; inferior: sobrevendido.</li>
        </ul>
        <b>Combina sinais para maior robustez. Nenhum indicador garante resultados.</b>
        '''
        QMessageBox.information(self, "Ajuda: Indicadores Técnicos", text)

    def analyse_indicators(self):
        if self.current_data is None or 'Close' not in self.current_data.columns:
            return "Sem dados para análise automática."
        from indicators.ta import sma, rsi, macd, bollinger_bands
        close = self.current_data['Close']
        sma20 = sma(close, 20)
        rsi14 = rsi(close, 14)
        macd_line, macd_signal = macd(close)
        bb_sma, bb_upper, bb_lower = bollinger_bands(close)
        msgs = []
        try:
            if close.iloc[-1] > sma20.iloc[-1]:
                msgs.append("Preço acima da SMA20 — <b>tendência de subida</b>.")
            else:
                msgs.append("Preço abaixo da SMA20 — <b>tendência de descida</b>.")
            if rsi14.iloc[-1] > 70:
                msgs.append("RSI14 acima de 70 — <b>sobrecomprado</b> (pode descer).")
            elif rsi14.iloc[-1] < 30:
                msgs.append("RSI14 abaixo de 30 — <b>sobrevendido</b> (pode subir).")
            else:
                msgs.append("RSI14 em zona neutra.")
            if macd_line.iloc[-1] > macd_signal.iloc[-1]:
                msgs.append("MACD acima do sinal — <b>momentum positivo</b>.")
            else:
                msgs.append("MACD abaixo do sinal — <b>momentum negativo</b>.")
            if close.iloc[-1] >= bb_upper.iloc[-1]:
                msgs.append("Preço tocou na banda superior de Bollinger — <b>potencial reversão para baixo</b>.")
            elif close.iloc[-1] <= bb_lower.iloc[-1]:
                msgs.append("Preço tocou na banda inferior de Bollinger — <b>potencial reversão para cima</b>.")
        except Exception as e:
            msgs.append(f"Erro na análise automática: {e}")
        return "<br>".join(msgs)

    def mostrar_consenso(self, direction, indicadores_texto):
        if direction is None:
            cor = "gray"
            consenso = "Sem previsão IA"
        elif ("subida" in indicadores_texto.lower() and direction in [1, 2]) or \
            ("descida" in indicadores_texto.lower() and direction == 0):
            cor = "green"
            consenso = "CONCORDAM"
        else:
            cor = "red"
            consenso = "DIVERGEM"
        return f'<div style="color:{cor};font-size:18px;"><b>CONSENSO: {consenso}</b></div>'

    def dica_de_ouro(self, direction, proba, price_pred, indicadores_texto):
        msg = "<b>DICA DE OURO (Swing Trading):</b><br>"
        if direction is not None:
            ia_txt = "Subida" if direction in [1, 2] else "Queda"
            if ("subida" in indicadores_texto.lower() and ia_txt == "Subida"):
                msg += "Sinais alinhados: <b>Confirmação de tendência de subida</b>.<br>"
                msg += "Swing: pode manter posição longa, mas atenção ao volume e volatilidade.<br>"
            elif ("descida" in indicadores_texto.lower() and ia_txt == "Queda"):
                msg += "Sinais alinhados: <b>Confirmação de tendência de descida</b>.<br>"
                msg += "Swing: pode manter posição curta ou evitar compra.<br>"
            else:
                msg += "⚠️ Sinais contraditórios: <b>tenha cautela</b>.<br>"
                msg += "Reduza exposição ou espere confirmação do próximo dia.<br>"
        else:
            msg += "Sem previsão clara da IA. Use só análise técnica.<br>"
        return msg

    def build_param_widgets(self, form_layout, strategy_name):
        for widget in self.param_widgets.values():
            form_layout.removeRow(widget)
        self.param_widgets = {}
        params = self.strategy_params[strategy_name]
        for param, value in params.items():
            if isinstance(value, int):
                widget = QSpinBox()
                widget.setMinimum(1)
                widget.setMaximum(500)
                widget.setValue(value)
            else:
                widget = QDoubleSpinBox()
                widget.setMinimum(0)
                widget.setMaximum(100)
                widget.setValue(value)
            widget.valueChanged.connect(self.update_strategy_params)
            self.param_widgets[param] = widget
            form_layout.addRow(param.replace("_", " ").capitalize(), widget)

    def on_strategy_change(self, strategy_name):
        self.selected_strategy = strategy_name
        box = self.sender().parentWidget().layout()
        self.build_param_widgets(box, strategy_name)
        self.create_strategy_instance()

    def update_strategy_params(self):
        params = self.strategy_params[self.selected_strategy]
        for param, widget in self.param_widgets.items():
            params[param] = widget.value()
        self.create_strategy_instance()

    def create_strategy_instance(self):
        cls = self.strategy_classes[self.selected_strategy]
        params = self.strategy_params[self.selected_strategy]
        self.strategy = cls(**params)

# Teste standalone (apenas se correr diretamente este ficheiro)
if __name__ == "__main__":
    from data.data_provider import DataProvider
    from portfolio.portfolio_manager import PortfolioManager
    from ai.predictor import AIPredictor
    app = QApplication(sys.argv)
    window = MainWindow(
        data_provider=DataProvider(),
        portfolio_manager=PortfolioManager(),
        predictor=AIPredictor(),
        initial_capital=10000,
        tickers=["AAPL", "MSFT", "GOOG"]
    )
    window.show()
    sys.exit(app.exec_())
