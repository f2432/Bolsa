import pandas as pd
from sklearn.linear_model import LogisticRegression
from indicators.ta import compute_all_indicators

def prepare_features(data):
    """Prepara features completas a partir de um DataFrame de preços."""
    df = pd.DataFrame(compute_all_indicators(data)).dropna()
    return df

def train_direction_model(data):
    """Treina um modelo simples de direção de preço (subir/descer amanhã)."""
    df = prepare_features(data)
    df['target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    df = df.dropna()
    X = df.drop(columns=['target'])
    y = df['target']
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model, X.columns.tolist()
