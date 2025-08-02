import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

class DataProvider:
    """Obtém dados históricos e em tempo real de ações usando yfinance."""
    def __init__(self, start_date=None, end_date=None, period='1y', interval='1d'):
        self.start_date = start_date
        self.end_date = end_date
        self.period = period
        self.interval = interval
        self.cache = {}

    def get_historical_data(self, ticker, refresh=False):
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
            logger.error(f"Erro ao obter dados para {ticker}: {e}")
            return None

        if isinstance(data, pd.DataFrame) and not data.empty:
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            # Garante sempre coluna 'Close'
            if 'Adj Close' in data.columns and 'Close' not in data.columns:
                data['Close'] = data['Adj Close']
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            if data.index.tz is not None:
                data.index = data.index.tz_convert(None)
            data.index.name = 'Date'
            self.cache[ticker] = data
            return data
        else:
            return None

    def get_current_price(self, ticker):
        ticker = ticker.upper()
        try:
            yfticker = yf.Ticker(ticker)
            info = yfticker.info
            price = info.get('regularMarketPrice') or info.get('currentPrice')
            if price is None:
                hist = yfticker.history(period="1d", interval="1m")
                if not hist.empty:
                    price = float(hist['Close'].iloc[-1])
            return float(price) if price is not None else None
        except Exception as e:
            logger.error(f"Erro ao obter preço atual para {ticker}: {e}")
            return None
