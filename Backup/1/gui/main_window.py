import sys
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTableWidget, QTableWidgetItem, QAbstractItemView, QMessageBox, QSizePolicy,
    QInputDialog, QComboBox, QSpinBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox
)
from gui.widgets.chart_widget import ChartWidget

# Importar estratégias
from strategies.sma_crossover import SMACrossoverStrategy
from strategies.rsi_macd_combo import RSIMACDStrategy

class MainWindow(QMainWindow):
    def __init__(self, data_provider, portfolio_manager, predictor, initial_capital, tickers):
        super().__init__()
        self.data_provider = data_provider
        self.portfolio_manager = portfolio_manager
        self.predictor = predictor
        self.initial_capital = initial_capital
        self.tickers = tickers
        self.current_ticker = None
        self.current_data = None

        # Estratégias disponíveis
        self.strategy_classes = {
            "SMA Crossover": SMACrossoverStrategy,
            "RSI+MACD": RSIMACDStrategy,
        }
        # Valores default dos parâmetros
        self.strategy_params = {
            "SMA Crossover": {"short_window": 50, "long_window": 200},
            "RSI+MACD": {"rsi_buy_threshold": 30, "rsi_sell_threshold": 70}
        }
        # Estratégia selecionada
        self.selected_strategy = "SMA Crossover"
        self.strategy = SMACrossoverStrategy(
            short_window=self.strategy_params["SMA Crossover"]["short_window"],
            long_window=self.strategy_params["SMA Crossover"]["long_window"]
        )

        self.setWindowTitle("Trading Application")
        self.resize(1400, 900)

        # Layouts
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Esquerda: Tabela de tickers
        self.stock_table = QTableWidget(len(self.tickers), 2)
        self.stock_table.setHorizontalHeaderLabels(["Ticker", "Price"])
        self.stock_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.stock_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.stock_table.setSelectionMode(QAbstractItemView.SingleSelection)
        for i, ticker in enumerate(self.tickers):
            self.stock_table.setItem(i, 0, QTableWidgetItem(ticker))
            price = self.data_provider.get_current_price(ticker)
            price_text = f"{price:.2f}" if price is not None else "N/A"
            self.stock_table.setItem(i, 1, QTableWidgetItem(price_text))
        self.stock_table.resizeColumnsToContents()
        table_width = self.stock_table.horizontalHeader().length() + 20
        self.stock_table.setFixedWidth(table_width)
        self.stock_table.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.stock_table.itemSelectionChanged.connect(self.on_stock_selected)

        # Direita: Chart + Painéis de controlo
        right_panel = QVBoxLayout()
        self.chart_widget = ChartWidget()
        self.chart_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_panel.addWidget(self.chart_widget)
        self.price_label = QLabel("Current Price: ")
        right_panel.addWidget(self.price_label)

        # --- PARÂMETROS DE ESTRATÉGIA E SELEÇÃO ---
        strategy_box = QGroupBox("Strategy & Parameters")
        strategy_layout = QFormLayout()
        # ComboBox de estratégia
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(list(self.strategy_classes.keys()))
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_change)
        strategy_layout.addRow("Strategy:", self.strategy_combo)

        # Widgets para parâmetros dinâmicos
        self.param_widgets = {}
        self.build_param_widgets(strategy_layout, self.selected_strategy)

        strategy_box.setLayout(strategy_layout)
        right_panel.addWidget(strategy_box)

        # --- Botões de ação ---
        button_panel = QHBoxLayout()

        # Botão Refresh Data
        refresh_btn = QPushButton("Refresh Data")
        refresh_btn.clicked.connect(self.refresh_data)
        button_panel.addWidget(refresh_btn)

        # Botão Backtest
        backtest_btn = QPushButton("Backtest Strategy")
        backtest_btn.clicked.connect(self.run_backtest)
        button_panel.addWidget(backtest_btn)

        # Botão Predict (IA)
        predict_btn = QPushButton("Predict")
        predict_btn.clicked.connect(self.run_prediction)
        button_panel.addWidget(predict_btn)

        # Botão Add Position
        add_portfolio_btn = QPushButton("Add Position")
        add_portfolio_btn.clicked.connect(self.add_portfolio_position)
        button_panel.addWidget(add_portfolio_btn)

        # Botão Show Indicators
        show_indicators_btn = QPushButton("Show Indicators")
        show_indicators_btn.clicked.connect(self.show_indicators)
        button_panel.addWidget(show_indicators_btn)

        # Botão Show Portfolio
        show_portfolio_btn = QPushButton("Show Portfolio")
        show_portfolio_btn.clicked.connect(self.show_portfolio)
        button_panel.addWidget(show_portfolio_btn)

        # Botão Save Chart
        save_chart_btn = QPushButton("Save Chart")
        save_chart_btn.clicked.connect(self.save_chart)
        button_panel.addWidget(save_chart_btn)

        # Botão Clear Chart
        clear_chart_btn = QPushButton("Clear Chart")
        clear_chart_btn.clicked.connect(self.clear_chart)
        button_panel.addWidget(clear_chart_btn)

        # Botão Show Trades
        show_trades_btn = QPushButton("Show Trades")
        show_trades_btn.clicked.connect(self.show_trades)
        button_panel.addWidget(show_trades_btn)

        # Botão Export Portfolio
        export_portfolio_btn = QPushButton("Export Portfolio")
        export_portfolio_btn.clicked.connect(self.export_portfolio)
        button_panel.addWidget(export_portfolio_btn)

        right_panel.addLayout(button_panel)

        # Label para mostrar o resultado da previsão
        self.prediction_label = QLabel("Prediction: N/A")
        right_panel.addWidget(self.prediction_label)

        main_layout.addWidget(self.stock_table)
        main_layout.addLayout(right_panel)

        # Seleciona o primeiro ticker por omissão
        if self.tickers:
            self.stock_table.selectRow(0)
            self.on_stock_selected()

    def build_param_widgets(self, form_layout, strategy_name):
        # Limpa widgets antigos
        for widget in self.param_widgets.values():
            form_layout.removeRow(widget)
        self.param_widgets = {}

        # Adiciona widgets consoante a estratégia
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
        # Quando muda a estratégia, muda widgets e a estratégia em si
        self.selected_strategy = strategy_name
        box = self.sender().parentWidget().layout()
        self.build_param_widgets(box, strategy_name)
        self.create_strategy_instance()

    def update_strategy_params(self):
        # Atualiza valores nos parâmetros
        params = self.strategy_params[self.selected_strategy]
        for param, widget in self.param_widgets.items():
            params[param] = widget.value()
        self.create_strategy_instance()

    def create_strategy_instance(self):
        # Cria instância da estratégia com os parâmetros atuais
        cls = self.strategy_classes[self.selected_strategy]
        params = self.strategy_params[self.selected_strategy]
        self.strategy = cls(**params)

    def on_stock_selected(self):
        row = self.stock_table.currentRow()
        if row < 0:
            return
        ticker = self.stock_table.item(row, 0).text()
        self.current_ticker = ticker
        data = self.data_provider.get_historical_data(ticker)
        if data is None or data.empty:
            QMessageBox.warning(self, "Data Error", f"No data for {ticker}.")
            return
        if isinstance(data.index, pd.MultiIndex):
            data = data.xs(ticker, level=0)
            data.index.name = 'Date'
            data = data.sort_index()
        if 'Ticker' in data.columns:
            data = data.drop(columns=['Ticker'])
        self.current_data = data
        self.chart_widget.plot(data['Close'], title=f"{ticker} Price")
        current_price = self.data_provider.get_current_price(ticker)
        if current_price is None:
            current_price = data['Close'].iloc[-1]
        self.price_label.setText(f"Current Price: {current_price:.2f}")

    def refresh_data(self):
        for i, ticker in enumerate(self.tickers):
            price = self.data_provider.get_current_price(ticker)
            price_text = f"{price:.2f}" if price is not None else "N/A"
            self.stock_table.setItem(i, 1, QTableWidgetItem(price_text))
        if self.current_ticker:
            data = self.data_provider.get_historical_data(self.current_ticker, refresh=True)
            if data is not None and not data.empty:
                if isinstance(data.index, pd.MultiIndex):
                    data = data.xs(self.current_ticker, level=0)
                    data.index.name = 'Date'
                    data = data.sort_index()
                if 'Ticker' in data.columns:
                    data = data.drop(columns=['Ticker'])
                self.current_data = data
                self.chart_widget.plot(data['Close'], title=f"{self.current_ticker} Price")
                current_price = self.data_provider.get_current_price(self.current_ticker)
                if current_price is None:
                    current_price = data['Close'].iloc[-1]
                self.price_label.setText(f"Current Price: {current_price:.2f}")

    def run_backtest(self):
        if self.current_data is None or self.current_ticker is None:
            QMessageBox.information(self, "Backtest", "No stock selected for backtesting.")
            return
        from backtest.backtester import Backtester
        try:
            backtester = Backtester(strategy=self.strategy, initial_capital=self.initial_capital)
            results = backtester.run(self.current_data)
        except Exception as e:
            QMessageBox.critical(self, "Backtest Error", str(e))
            return
        equity = results["equity_curve"]
        total_return = (equity.iloc[-1] - equity.iloc[0]) / equity.iloc[0]
        QMessageBox.information(self, "Backtest Results",
            f"Strategy: {self.strategy.name}\nTotal Return: {total_return:.2%}")

    def run_prediction(self):
        if self.current_data is None or self.current_ticker is None:
            QMessageBox.information(self, "Predict", "No stock selected for prediction.")
            return
        direction = self.predictor.predict_direction(self.current_data)
        future_price = self.predictor.predict_price(self.current_data)
        if direction is None or future_price is None:
            self.prediction_label.setText("Prediction: Model not available.")
        else:
            direction_text = "UP" if direction == 1 else "DOWN"
            self.prediction_label.setText(f"Prediction: {direction_text}, Next Price ~ {future_price:.2f}")

    def add_portfolio_position(self):
        ticker, ok1 = QInputDialog.getText(self, "Add Position", "Ticker symbol:")
        if not ok1 or ticker.strip() == "":
            return
        qty, ok2 = QInputDialog.getInt(self, "Add Position", "Quantity:", value=0, min=0)
        if not ok2:
            return
        price, ok3 = QInputDialog.getDouble(self, "Add Position", "Purchase Price:", decimals=2)
        if not ok3:
            return
        print("[DEBUG] Chamei add_position pelo GUI!")  # DEBUG
        self.portfolio_manager.add_position(ticker.strip().upper(), qty, price)
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
        QMessageBox.information(self, "Portfolio Updated",
                                f"Total Portfolio Value: ${total_val:,.2f}\nTotal Profit/Loss: ${total_profit:,.2f}")


    def show_indicators(self):
        # Plota SMA20 e SMA50 (adapta conforme quiseres)
        if self.current_data is not None:
            from indicators.ta import sma
            close = self.current_data['Close']
            sma20 = sma(close, 20)
            sma50 = sma(close, 50)
            self.chart_widget.ax.clear()
            self.chart_widget.ax.plot(close.index, close.values, label='Close')
            self.chart_widget.ax.plot(sma20.index, sma20.values, label='SMA20')
            self.chart_widget.ax.plot(sma50.index, sma50.values, label='SMA50')
            self.chart_widget.ax.set_title(f"{self.current_ticker} + Indicators")
            self.chart_widget.ax.legend()
            self.chart_widget.canvas.draw()

    def show_portfolio(self):
        print("[DEBUG] positions em memória:", self.portfolio_manager.positions)
        metrics = self.portfolio_manager.calculate_metrics(self.data_provider)
        print("[DEBUG] metrics calculadas:", metrics)  # Opcional, para ver se está vazio
        msg = ""
        for pos in metrics['positions']:
            msg += f"{pos['ticker']}: {pos['quantity']} @ {pos['buy_price']} | Atual: {pos['current_price']} | P/L: {pos['profit']:.2f}\n"
        msg += f"\nTotal Value: {metrics['total_value']:.2f}\nTotal Profit: {metrics['total_profit']:.2f}"
        QMessageBox.information(self, "Current Portfolio", msg)
        metrics = self.portfolio_manager.calculate_metrics(self.data_provider)
        msg = ""
        for pos in metrics['positions']:
            msg += f"{pos['ticker']}: {pos['quantity']} @ {pos['buy_price']} | Atual: {pos['current_price']} | P/L: {pos['profit']:.2f}\n"
        msg += f"\nTotal Value: {metrics['total_value']:.2f}\nTotal Profit: {metrics['total_profit']:.2f}"
        QMessageBox.information(self, "Current Portfolio", msg)

    def save_chart(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Chart", "", "PNG Image (*.png)")
        if file_path:
            self.chart_widget.figure.savefig(file_path)
            QMessageBox.information(self, "Chart Saved", f"Chart saved to:\n{file_path}")

    def clear_chart(self):
        self.chart_widget.ax.clear()
        self.chart_widget.canvas.draw()

    def show_trades(self):
        try:
            from backtest.backtester import Backtester
            backtester = Backtester(strategy=self.strategy, initial_capital=self.initial_capital)
            results = backtester.run(self.current_data)
            # Supondo que guardas as trades em results['trades']
            if 'trades' in results:
                trades = results['trades']
                msg = "Date\tType\tPrice\tQty\n"
                for tr in trades:
                    msg += f"{tr['date']}\t{tr['type']}\t{tr['price']}\t{tr['qty']}\n"
            else:
                msg = "No trades found."
            QMessageBox.information(self, "Trades", msg)
        except Exception as e:
            QMessageBox.critical(self, "Trades Error", str(e))

    def export_portfolio(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Export Portfolio", "", "CSV Files (*.csv)")
        if file_path:
            import pandas as pd
            metrics = self.portfolio_manager.calculate_metrics(self.data_provider)
            df = pd.DataFrame(metrics['positions'])
            df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Export Portfolio", f"Portfolio exported to:\n{file_path}")

# Apenas para teste standalone
if __name__ == "__main__":
    from data.data_provider import DataProvider
    from portfolio.portfolio_manager import PortfolioManager
    from ai.predictor import AIPredictor
    app = QApplication(sys.argv)
    window = MainWindow(
        data_provider=DataProvider(),
        portfolio_manager=PortfolioManager(),
        predictor=AIPredictor(None, None),
        initial_capital=10000,
        tickers=["AAPL", "MSFT", "GOOG"]
    )
    window.show()
    sys.exit(app.exec_())
