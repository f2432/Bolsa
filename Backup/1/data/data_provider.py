import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class DataProvider:
    """Fetches historical and real-time stock data using yfinance."""
    def __init__(self, start_date=None, end_date=None, period='1y', interval='1d'):
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.interval = interval
        self.cache = {}  # cache historical data to avoid repeated downloads

    def get_historical_data(self, ticker, refresh=False):
        """
        Get historical OHLCV data for the given ticker.
        If refresh is False and data is cached, returns cached data.
        """
        ticker = ticker.upper()
        if not refresh and ticker in self.cache:
            return self.cache[ticker]
        try:
            logger.info(f"Fetching historical data for {ticker}")
            if self.start_date and self.end_date:
                data = yf.download(ticker, start=self.start_date, end=self.end_date, interval=self.interval)
            else:
                data = yf.download(ticker, period=self.period, interval=self.interval)
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            return None
        if isinstance(data, pd.DataFrame) and not data.empty:
            # Cache the data
            self.cache[ticker] = data
            return data
        else:
            return None

    def get_current_price(self, ticker):
        """
        Get the latest price for the given ticker.
        This uses yfinance to fetch the most recent trading price (near real-time).
        """
        ticker = ticker.upper()
        try:
            yfticker = yf.Ticker(ticker)
            # Try to get current price from ticker info
            info = yfticker.info
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            if price is None:
                # Fallback: get last available close price from history (1-day, 1-minute interval)
                hist = yfticker.history(period="1d", interval="1m")
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
            return float(price) if price is not None else None
        except Exception as e:
            logger.error(f"Error fetching current price for {ticker}: {e}")
            return None
