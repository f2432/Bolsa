
import yfinance as yf
import joblib
from ai.train_utils import train_direction_model, train_price_regressor
from config import settings

# Carregar configurações
tickers = settings.get("default_tickers", ["AAPL"])
period = settings.get("backtest_period", "1y")
interval = settings.get("interval", "1d")

# Usar apenas o primeiro ticker por simplicidade
ticker = tickers[0]
print(f"🔍 A obter dados de: {ticker}")
df = yf.download(ticker, period=period, interval=interval)

# Treinar modelo de classificação
print("⚙️ A treinar modelo de classificação...")
clf_model, clf_features = train_direction_model(df)
joblib.dump(clf_model, settings["model_classifier_path"])
print(f"✅ Modelo de classificação guardado em: {settings['model_classifier_path']}")

# Treinar modelo de regressão
print("⚙️ A treinar modelo de regressão...")
reg_model, reg_features = train_price_regressor(df)
joblib.dump(reg_model, settings["model_regressor_path"])
print(f"✅ Modelo de regressão guardado em: {settings['model_regressor_path']}")
