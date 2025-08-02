
import sys
import yaml
from PyQt5.QtWidgets import QApplication
from data.data_provider import DataProvider
from portfolio.portfolio_manager import PortfolioManager
from ai.predictor import AIPredictor
from gui.main_window import MainWindow

if __name__ == "__main__":
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)

    tickers = config.get("default_tickers", [])
    initial_capital = config.get("initial_capital", 10000)
    strategy_name = config.get("default_strategy", "sma_crossover")

    data_provider = DataProvider()
    portfolio_manager = PortfolioManager(filepath='portfolio.csv')

    predictor = AIPredictor(
        config.get("model_classifier_path", "ai/logistic_model.pkl"),
        config.get("model_regressor_path", "ai/rf_model.pkl")
    )

    # Escolher estratégia
    if strategy_name == "ai_model_strategy":
        from strategies.ai_model_strategy import AIModelStrategy
        strategy = AIModelStrategy(predictor)
    elif strategy_name == "rsi_macd_combo":
        from strategies.rsi_macd_combo import RSIMACDStrategy
        strategy = RSIMACDStrategy(
            rsi_buy_threshold=config.get("rsi_buy_threshold", 30),
            rsi_sell_threshold=config.get("rsi_sell_threshold", 70)
        )
    else:
        from strategies.sma_crossover import SMACrossoverStrategy
        strategy = SMACrossoverStrategy(
            short_window=config.get("sma_short_window", 50),
            long_window=config.get("sma_long_window", 200)
        )

    app = QApplication(sys.argv)
    main_window = MainWindow(data_provider, portfolio_manager, predictor, initial_capital, tickers, strategy)
    main_window.show()
    sys.exit(app.exec_())
