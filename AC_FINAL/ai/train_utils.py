
import pandas as pd
from sklearn.linear_model import LogisticRegression
from indicators.ta import compute_all_indicators

def prepare_features(data):
    """
    Prepara features completas a partir de um DataFrame de preços (OHLCV).
    Retorna DataFrame só com features técnicas.
    """
    df = pd.DataFrame(compute_all_indicators(data)).dropna()
    return df

def train_direction_model(data):
    """
    Treina um modelo simples de direção de preço (subir/descer no dia seguinte).
    Útil para experiências, tuning rápido ou benchmark.
    Retorna modelo treinado e lista dos nomes das features.
    """
    df = prepare_features(data)
    df['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    df = df.dropna()
    X = df.drop(columns=['target'])
    y = df['target']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, X.columns.tolist()


from sklearn.ensemble import RandomForestRegressor

def train_price_regressor(data):
    """
    Treina um modelo de regressão para prever o preço de fecho do dia seguinte.
    Retorna modelo treinado e lista dos nomes das features.
    """
    df = prepare_features(data)
    df['target'] = data['Close'].shift(-1)
    df = df.dropna()
    X = df.drop(columns=['target'])
    y = df['target']
    model = RandomForestRegressor()
    model.fit(X, y)
    return model, X.columns.tolist()
