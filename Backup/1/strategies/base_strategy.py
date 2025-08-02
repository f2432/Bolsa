from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    def __init__(self, name="BaseStrategy"):
        self.name = name

    @abstractmethod
    def generate_signals(self, data):
        """
        Given historical price data (DataFrame), generate buy/sell signals.
        Returns a pandas Series or list with values: 1 for buy, -1 for sell, 0 for hold.
        """
        pass
