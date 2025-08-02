import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QAbstractItemView, QMessageBox, QSizePolicy,
    QInputDialog, QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QTextEdit, QCheckBox, QDateEdit, QTabWidget,
    QGridLayout, QToolButton, QMenu,
)
from PyQt5.QtCore import QDate
from gui.widgets.chart_widget import ChartWidget
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_macd_combo import RSIMACDStrategy
from prediction_log import PredictionLogger
from gui.dialogs import PredictionDialog
from gui.explore_tab import ExploreTab
from gui.indicator_utils import analyse_indicators_custom
import logging

logger = logging.getLogger(__name__)

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

        self.setWindowTitle("Aplicação de Trading Profissional")
        self.resize(1600, 980)

        # TABS e Layout principal
        self.tabs = QTabWidget()
        main_widget = QWidget()
        self.tabs.addTab(main_widget, "Principal")
        self.dashboard = QTextEdit()
        self.dashboard.setReadOnly(True)
        self.tabs.addTab(self.dashboard, "Dashboard")
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)

        # Painel esquerdo: Tickers
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

        # Painel central: Gráfico e controlos
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
        # Painel de botões: usa um grid para evitar que o layout seja demasiado largo
        button_panel = QGridLayout()
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
        # Distribui os botões em duas linhas para melhorar a disposição
        max_cols = 10
        for idx, (text, callback) in enumerate(button_defs):
            btn = QPushButton(text)
            btn.clicked.connect(callback)
            row = idx // max_cols
            col = idx % max_cols
            button_panel.addWidget(btn, row, col)
        # Adiciona botão de menu de análises
        analysis_btn = QToolButton()
        analysis_btn.setText("Análises")
        analysis_menu = QMenu(analysis_btn)
        # Opções de análise - mapeadas para os métodos existentes
        analysis_menu.addAction("Estatística", self.show_statistics)
        analysis_menu.addAction("Importância Features", self.show_feature_importances)
        analysis_menu.addAction("Backtest", self.show_backtest_analysis)
        analysis_menu.addAction("Trades", self.show_trade_analysis)
        analysis_menu.addAction("Mapa Parâmetros", self.show_heatmap_optimization)
        analysis_menu.addAction("Análise de Risco", self.show_risk_analysis)
        analysis_menu.addAction("Atribuição PnL", self.show_pnl_attribution)
        analysis_menu.addAction("Atribuição Factores", self.show_factor_attribution)
        analysis_menu.addAction("Stop/Take", self.show_stop_loss_take_profit_optimization)
        analysis_menu.addAction("Planeamento IA", self.show_planning_graph)
        analysis_btn.setMenu(analysis_menu)
        analysis_btn.setPopupMode(QToolButton.InstantPopup)
        # Coloca o botão de análises na grelha
        total_buttons = len(button_defs)
        row = total_buttons // max_cols
        col = total_buttons % max_cols
        button_panel.addWidget(analysis_btn, row, col)
        # Botão de ajuda no final
        help_btn = QPushButton("?")
        help_btn.setToolTip("Explicação dos indicadores técnicos")
        help_btn.clicked.connect(self.show_indicators_explanation)
        # Próxima coluna para o botão de ajuda
        col += 1
        if col >= max_cols:
            col = 0
            row += 1
        button_panel.addWidget(help_btn, row, col)
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

        self.explore_tab = ExploreTab(
            self.data_provider,
            self.predictor,
            self.portfolio_manager,
            self.indicator_analysis_text,
            self
        )
        self.tabs.addTab(self.explore_tab, "Explorar Ações")

        # Adiciona separador de histórico de previsões
        try:
            from history_tab import HistoryTab
        except ImportError:
            try:
                from gui.history_tab import HistoryTab
            except ImportError:
                HistoryTab = None
        if HistoryTab is not None:
            self.history_tab = HistoryTab(self.pred_logger, self)
            self.tabs.addTab(self.history_tab, "Histórico")

    # ---------- (Métodos a partir daqui, todos completos) ----------

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

    def show_statistics(self):
        """
        Abre um diálogo com visualizações estatísticas (histograma, correlação, ACF/PACF)
        da série de preços actualmente carregada.
        """
        try:
            if self.current_data is None:
                QMessageBox.information(self, "Análise Estatística", "Nenhuma série seleccionada para análise.")
                return
            # Importa StatisticsDialog de acordo com a localização do ficheiro.  Se o módulo
            # estiver dentro da pasta gui, ajusta o caminho para gui.statistics_dialog.
            try:
                from statistics_dialog import StatisticsDialog  # tentativa no mesmo nível
            except ImportError:
                from gui.statistics_dialog import StatisticsDialog
            dlg = StatisticsDialog(self.current_data, self)
            dlg.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Erro Análise Estatística", str(e))

    def show_feature_importances(self):
        """Desenha a importância das features do último modelo treinado."""
        # Verifica se o objecto predictor tem importâncias calculadas e suportadas
        if not hasattr(self.predictor, 'last_feature_importance'):
            QMessageBox.information(
                self,
                "Importância das Features",
                "O modelo actual não suporta importâncias de features."
            )
            return
        importances = self.predictor.last_feature_importance
        # Se last_feature_importance for None (ex.: MLP), informa o utilizador
        if importances is None:
            QMessageBox.information(
                self,
                "Importância das Features",
                "O modelo actual não fornece importâncias das features (ex.: MLP). "
                "Selecione outro modelo (Logistic ou RF) e treine novamente para ver as importâncias."
            )
            return
        if not importances:
            QMessageBox.information(
                self,
                "Importância das Features",
                "Não há informações de importância disponíveis. Treina um modelo primeiro."
            )
            return
        # Importa o diálogo de importância das features
        FeatureImportanceDialog = None
        try:
            from feature_importance_dialog import FeatureImportanceDialog  # type: ignore
        except ImportError:
            try:
                from gui.feature_importance_dialog import FeatureImportanceDialog  # type: ignore
            except ImportError:
                FeatureImportanceDialog = None
        # Mostra o gráfico em diálogo se possível
        if FeatureImportanceDialog is not None:
            try:
                dlg = FeatureImportanceDialog(importances, self, top_n=10)
                dlg.exec_()
            except Exception:
                pass
        # Lista textual das importâncias no painel
        try:
            top_n = 10
            top_feats = importances[:top_n]
            html_lines = [f"{name}: {imp:.4f}" for name, imp in top_feats]
            html = "<br>".join(html_lines)
            self.indicator_analysis_text.append(
                f"<hr><b>Importância das Features (top {top_n}):</b><br>{html}"
            )
        except Exception:
            pass

    def show_backtest_analysis(self):
        """
        Executa um backtest da estratégia seleccionada e mostra um diálogo
        com a curva de saldo e a distribuição de retornos.
        """
        try:
            if self.current_data is None or self.current_ticker is None:
                QMessageBox.information(self, "Análise Backtest", "Nenhuma série seleccionada para análise.")
                return
            from backtest.backtester import Backtester
            # Corre a estratégia actual
            try:
                # Assegura que a instância da estratégia existe antes de gerar sinais
                if not hasattr(self, 'strategy') or self.strategy is None:
                    self.create_strategy_instance()
                signals = self.strategy.generate_signals(self.current_data)
            except Exception as e:
                QMessageBox.warning(self, "Análise Backtest", f"Erro a gerar sinais: {e}")
                return
            backtester = Backtester(initial_capital=self.initial_capital)
            results = backtester.run(self.current_data, signals)
            equity_curve = results.get("equity_curve")
            trades_df = results.get("trades")
            # Abre o diálogo de análise
            try:
                from backtest_analysis_dialog import BacktestAnalysisDialog
            except ImportError:
                try:
                    from gui.backtest_analysis_dialog import BacktestAnalysisDialog
                except ImportError:
                    BacktestAnalysisDialog = None
            if BacktestAnalysisDialog is not None:
                dlg = BacktestAnalysisDialog(equity_curve, trades_df, self)
                dlg.exec_()
            # Adiciona resumo na área de interpretação
            try:
                from backtest.metrics import calculate_metrics
                metrics_summary = calculate_metrics(self.current_data, signals, equity_curve, trades_df)
                summary = (f"<b>Backtest:</b> Retorno total: {metrics_summary.get('retorno')} | "
                           f"Max Drawdown: {metrics_summary.get('drawdown')} | "
                           f"Sharpe Ratio: {metrics_summary.get('sharpe')} | "
                           f"Nº Trades: {len(trades_df) if trades_df is not None else 0}")
                self.indicator_analysis_text.append(summary)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Análise Backtest", str(e))

    def show_trade_analysis(self):
        """
        Executa um backtest e apresenta a análise da distribuição dos lucros/perdas das trades.
        """
        try:
            if self.current_data is None or self.current_ticker is None:
                QMessageBox.information(self, "Análise de Trades", "Nenhuma série seleccionada para análise.")
                return
            from backtest.backtester import Backtester
            # Garante que existe instância de estratégia
            if not hasattr(self, 'strategy') or self.strategy is None:
                self.create_strategy_instance()
            # Gera sinais e corre backtest
            try:
                signals = self.strategy.generate_signals(self.current_data)
            except Exception as e:
                QMessageBox.warning(self, "Análise de Trades", f"Erro a gerar sinais: {e}")
                return
            backtester = Backtester(initial_capital=self.initial_capital)
            results = backtester.run(self.current_data, signals)
            trades_df = results.get("trades")
            # Abre o diálogo de análise de trades
            try:
                from trade_analysis_dialog import TradeAnalysisDialog
            except ImportError:
                try:
                    from gui.trade_analysis_dialog import TradeAnalysisDialog
                except ImportError:
                    TradeAnalysisDialog = None
            if TradeAnalysisDialog is not None:
                dlg = TradeAnalysisDialog(trades_df, self)
                dlg.exec_()
            # Resumo para a análise de trades
            if trades_df is not None and not trades_df.empty:
                try:
                    pnl = trades_df['Resultado (€)'].replace('', float('nan')).dropna().astype(float)
                    win_rate = (pnl > 0).mean() if len(pnl) > 0 else 0
                    avg_pnl = pnl.mean() if len(pnl) > 0 else 0
                    summary = (f"<b>Trades:</b> Nº Trades: {len(pnl)} | Win rate: {win_rate:.2%} | "
                               f"PnL médio: {avg_pnl:.2f}€")
                    self.indicator_analysis_text.append(summary)
                except Exception:
                    pass
        except Exception as e:
            QMessageBox.critical(self, "Análise de Trades", str(e))

    def show_heatmap_optimization(self):
        """
        Abre um diálogo que gera um mapa de calor com a métrica de Sharpe
        para diferentes combinações de parâmetros de uma estratégia.  A
        implementação actual utiliza a estratégia SMA Crossover (short e
        long window) e calcula o Sharpe ratio para cada par de valores.
        """
        # Verifica se existem dados carregados
        if self.current_data is None or self.current_ticker is None:
            QMessageBox.information(self, "Mapa de Parâmetros", "Nenhuma série seleccionada para otimização.")
            return
        # Importa o diálogo de optimização de forma dinâmica
        try:
            from heatmap_optimization_dialog import HeatmapOptimizationDialog
        except ImportError:
            try:
                from gui.heatmap_optimization_dialog import HeatmapOptimizationDialog
            except ImportError:
                HeatmapOptimizationDialog = None
        if HeatmapOptimizationDialog is None:
            QMessageBox.critical(self, "Mapa de Parâmetros", "O módulo de otimização não está disponível.")
            return
        # Cria e executa o diálogo passando os dados actuais e o capital inicial
        try:
            dlg = HeatmapOptimizationDialog(self.current_data, self.initial_capital, self)
            dlg.exec_()
            # Resumo da melhor combinação short/long
            try:
                # Calcula heatmap novamente para encontrar melhor Sharpe
                short_vals = [10, 20, 30, 40, 50]
                long_vals = [50, 100, 150, 200]
                best_val = None
                best_pair = None
                from backtest.backtester import Backtester
                from strategies.sma_crossover import SMACrossoverStrategy
                from backtest.metrics import calculate_metrics
                backtester = Backtester(initial_capital=self.initial_capital)
                for sw in short_vals:
                    for lw in long_vals:
                        if sw >= lw:
                            continue
                        try:
                            strategy = SMACrossoverStrategy(short_window=sw, long_window=lw)
                            signals = strategy.generate_signals(self.current_data)
                            results = backtester.run(self.current_data, signals)
                            eq = results['equity_curve']
                            tr = results.get('trades')
                            metrics = calculate_metrics(self.current_data, signals, eq, tr)
                            sharpe = float(metrics.get('sharpe', '0'))
                            if best_val is None or sharpe > best_val:
                                best_val = sharpe
                                best_pair = (sw, lw)
                        except Exception:
                            continue
                if best_pair:
                    summary = (f"<b>Mapa Parâmetros:</b> Melhor Sharpe {best_val:.2f} com short={best_pair[0]} e long={best_pair[1]}")
                    self.indicator_analysis_text.append(summary)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Mapa de Parâmetros", f"Erro ao gerar mapa de parâmetros: {e}")

    def show_risk_analysis(self):
        """
        Executa um backtest e apresenta uma análise de risco baseada no
        drawdown da curva de saldo. Mostra a evolução do drawdown, a
        distribuição dos drawdowns negativos e estatísticas chave.
        """
        try:
            if self.current_data is None or self.current_ticker is None:
                QMessageBox.information(self, "Análise de Risco", "Nenhuma série seleccionada para análise.")
                return
            from backtest.backtester import Backtester
            # Garante que a instância da estratégia existe
            if not hasattr(self, 'strategy') or self.strategy is None:
                self.create_strategy_instance()
            try:
                signals = self.strategy.generate_signals(self.current_data)
            except Exception as e:
                QMessageBox.warning(self, "Análise de Risco", f"Erro a gerar sinais: {e}")
                return
            backtester = Backtester(initial_capital=self.initial_capital)
            results = backtester.run(self.current_data, signals)
            equity_curve = results.get("equity_curve")
            # Importa diálogo de risco
            try:
                from risk_analysis_dialog import RiskAnalysisDialog
            except ImportError:
                try:
                    from gui.risk_analysis_dialog import RiskAnalysisDialog
                except ImportError:
                    RiskAnalysisDialog = None
            if RiskAnalysisDialog is None:
                QMessageBox.critical(self, "Análise de Risco", "O módulo de análise de risco não está disponível.")
                return
            dlg = RiskAnalysisDialog(equity_curve, self)
            dlg.exec_()
            # Adiciona resumo do drawdown
            try:
                import pandas as pd
                running_max = equity_curve.cummax()
                drawdown = (equity_curve / running_max) - 1
                drawdown = drawdown.fillna(0)
                max_dd = drawdown.min() if not drawdown.empty else 0
                avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
                # calcula durações
                underwater = drawdown < 0
                durations = []
                count = 0
                for flag in underwater:
                    if flag:
                        count += 1
                    else:
                        if count > 0:
                            durations.append(count)
                        count = 0
                if count > 0:
                    durations.append(count)
                max_duration = max(durations) if durations else 0
                summary = (f"<b>Risco:</b> Máx Drawdown: {max_dd:.2%} | "
                           f"Média Drawdown: {avg_dd:.2%} | "
                           f"Maior tempo sob água: {max_duration} períodos")
                self.indicator_analysis_text.append(summary)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Análise de Risco", str(e))

    def show_pnl_attribution(self):
        """
        Mostra um gráfico de barras com a contribuição de lucro/prejuízo
        de cada ativo no portefólio actual e um resumo do PnL total.
        """
        try:
            # Calcula métricas do portefólio
            metrics = self.portfolio_manager.calculate_metrics(self.data_provider)
            # Importa o diálogo de atribuição de PnL
            try:
                from pnl_attribution_dialog import PnLAttributionDialog
            except ImportError:
                try:
                    from gui.pnl_attribution_dialog import PnLAttributionDialog
                except ImportError:
                    PnLAttributionDialog = None
            if PnLAttributionDialog is None:
                QMessageBox.critical(self, "Atribuição PnL", "O módulo de atribuição de PnL não está disponível.")
                return
            dlg = PnLAttributionDialog(metrics, self)
            dlg.exec_()
            # Resumo de PnL
            try:
                positions = metrics.get('positions', [])
                if positions:
                    profits = [p.get('profit', 0) for p in positions]
                    tickers = [p.get('ticker') for p in positions]
                    if profits:
                        max_profit = max(profits)
                        min_profit = min(profits)
                        best_idx = profits.index(max_profit)
                        worst_idx = profits.index(min_profit)
                        summary = (f"<b>PNL:</b> Melhor ativo: {tickers[best_idx]} ({max_profit:.2f}€) | "
                                   f"Pior ativo: {tickers[worst_idx]} ({min_profit:.2f}€) | "
                                   f"Total: {metrics.get('total_profit', 0):.2f}€")
                        self.indicator_analysis_text.append(summary)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Atribuição PnL", str(e))

    def show_stop_loss_take_profit_optimization(self):
        """
        Optimiza parâmetros de stop-loss e take-profit apresentando um
        mapa de Sharpe ratio para várias combinações de percentagens.
        """
        try:
            if self.current_data is None or self.current_ticker is None:
                QMessageBox.information(self, "Mapa Stop/Take", "Nenhuma série seleccionada para otimização.")
                return
            # Garante que existe instância de estratégia e sinais
            if not hasattr(self, 'strategy') or self.strategy is None:
                self.create_strategy_instance()
            try:
                signals = self.strategy.generate_signals(self.current_data)
            except Exception as e:
                QMessageBox.warning(self, "Mapa Stop/Take", f"Erro a gerar sinais: {e}")
                return
            # Importa o diálogo
            try:
                from stop_loss_take_profit_dialog import StopLossTakeProfitDialog
            except ImportError:
                try:
                    from gui.stop_loss_take_profit_dialog import StopLossTakeProfitDialog
                except ImportError:
                    StopLossTakeProfitDialog = None
            if StopLossTakeProfitDialog is None:
                QMessageBox.critical(self, "Mapa Stop/Take", "O módulo de otimização Stop/Take não está disponível.")
                return
            dlg = StopLossTakeProfitDialog(self.current_data, signals, self.initial_capital, self)
            dlg.exec_()
            # Resumo da melhor combinação stop/take
            try:
                stop_vals = [-0.02, -0.03, -0.05, -0.1]
                take_vals = [0.02, 0.04, 0.06, 0.1]
                best_val = None
                best_pair = None
                from stop_loss_take_profit_dialog import simulate_with_sl_tp
                from backtest.metrics import calculate_metrics
                for sl in stop_vals:
                    for tp in take_vals:
                        try:
                            eq, tr = simulate_with_sl_tp(self.current_data, signals, sl, tp, self.initial_capital)
                            metrics = calculate_metrics(self.current_data, signals, eq, tr)
                            sharpe = float(metrics.get('sharpe', '0'))
                            if best_val is None or sharpe > best_val:
                                best_val = sharpe
                                best_pair = (sl, tp)
                        except Exception:
                            continue
                if best_pair:
                    summary = (f"<b>Stop/Take:</b> Melhor Sharpe {best_val:.2f} com stop={best_pair[0]*100:.0f}% e take={best_pair[1]*100:.0f}%")
                    self.indicator_analysis_text.append(summary)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Mapa Stop/Take", str(e))

    def show_planning_graph(self):
        """
        Mostra um grafo demonstrativo de planeamento de IA, ilustrando
        estados e transições num processo de decisão simples para trading.
        """
        try:
            # Importa diálogo
            try:
                from planning_graph_dialog import PlanningGraphDialog
            except ImportError:
                try:
                    from gui.planning_graph_dialog import PlanningGraphDialog
                except ImportError:
                    PlanningGraphDialog = None
            if PlanningGraphDialog is None:
                QMessageBox.critical(self, "Planeamento IA", "O módulo de planeamento IA não está disponível.")
                return
            dlg = PlanningGraphDialog(self)
            dlg.exec_()
            # Adiciona resumo do grafo
            try:
                summary = ("<b>Planeamento IA:</b> Grafo demonstra transições simples entre estados de trading: "
                           "Compra, Manter, Stop-Loss, Take-Profit e Vender. Permite visualizar "
                           "possíveis caminhos de decisão.")
                self.indicator_analysis_text.append(summary)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Planeamento IA", str(e))

    def show_factor_attribution(self):
        """
        Mostra um gráfico de barras com a atribuição da importância das
        features por factores (tendência, momento, volatilidade, volume,
        padrões, outros). Baseia-se nas importâncias do último modelo treinado.
        """
        try:
            # Recupera importâncias do último modelo
            importances = getattr(self.predictor, 'last_feature_importance', None)
            if not importances:
                QMessageBox.information(self, "Atribuição Factores", "Sem importâncias disponíveis. Treina um modelo primeiro.")
                return
            # Importa diálogo
            try:
                from factor_attribution_dialog import FactorAttributionDialog
            except ImportError:
                try:
                    from gui.factor_attribution_dialog import FactorAttributionDialog
                except ImportError:
                    FactorAttributionDialog = None
            if FactorAttributionDialog is None:
                QMessageBox.critical(self, "Atribuição Factores", "O módulo de atribuição de factores não está disponível.")
                return
            dlg = FactorAttributionDialog(importances, self)
            dlg.exec_()
            # Resumo por factor
            try:
                # Repetir agrupamento para obter maior categoria
                categories = {
                    'Tendência': ['sma', 'ema', 'trend'],
                    'Momento': ['rsi', 'macd', 'stoch', 'mfi'],
                    'Volatilidade': ['atr', 'bb', 'vol'],
                    'Volume': ['obv', 'volume', 'avg_vol'],
                    'Padrões': ['bullish', 'bearish', 'pattern'],
                }
                factor_totals = {cat: 0.0 for cat in categories}
                factor_totals['Outros'] = 0.0
                total_importance = sum(abs(val) for (_, val) in importances) or 1.0
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
                # Find max category
                max_cat = max(factor_totals, key=factor_totals.get)
                perc = factor_totals[max_cat] / total_importance if total_importance != 0 else 0
                summary = (f"<b>Factores:</b> Categoria dominante: {max_cat} ({perc*100:.1f}% da importância)")
                self.indicator_analysis_text.append(summary)
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self, "Atribuição Factores", str(e))

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

