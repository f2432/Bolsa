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
    data_provider = DataProvider()
    # Só UMA instância do PortfolioManager!!
    portfolio_manager = PortfolioManager(filepath='portfolio.csv')
    predictor = AIPredictor(config.get("model_classifier_path", "ai/logistic_model.pkl"),
                            config.get("model_regressor_path", "ai/rf_model.pkl"))
    app = QApplication(sys.argv)
    main_window = MainWindow(data_provider, portfolio_manager, predictor, initial_capital, tickers)
    main_window.show()
    sys.exit(app.exec_())
