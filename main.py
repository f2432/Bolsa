from gui.main_window import MainWindow
from data.data_provider import DataProvider
from portfolio.portfolio_manager import PortfolioManager
from ai.predictor import AIPredictor
from utils.logger import logger

import sys
from PyQt5.QtWidgets import QApplication

def main():
    logger.info("A iniciar aplicação de trading!")
    data_provider = DataProvider()
    portfolio_manager = PortfolioManager()
    predictor = AIPredictor()  # (modelo pode ser carregado no futuro)
    initial_capital = 10000
    tickers = ["AAPL", "MSFT", "GOOG"]
    app = QApplication(sys.argv)
    main_window = MainWindow(data_provider, portfolio_manager, predictor, initial_capital, tickers)
    main_window.show()
    logger.info("Aplicação iniciada com sucesso.")
    sys.exit(app.exec_())
    


if __name__ == "__main__":
    main()
