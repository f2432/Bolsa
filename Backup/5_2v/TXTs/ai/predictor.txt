import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from indicators.ta import compute_all_indicators

class AIPredictor:
    """
    Classe para previsão de direção (e preço) via Machine Learning.
    Suporta regressão para previsão de preço e classificação multi-classe.
    """
    def __init__(self, model_type='logistic', n_ahead=1, multiclass=False, feature_set=None):
        self.model_type = model_type
        self.n_ahead = n_ahead
        self.multiclass = multiclass
        self.model = None
        self.reg_model = None
        self.scaler = None
        self.features = feature_set
        self.feature_names = []
        self.last_cv_score = None
        self.last_overfit_warning = None
        self.last_feature_importance = None

    def build_features(self, data):
        """Calcula e devolve DataFrame de features técnicas a partir de um DataFrame de OHLCV."""
        features = compute_all_indicators(data)
        df = pd.DataFrame(features).dropna()
        return df

    def _make_targets(self, close):
        n = self.n_ahead
        if self.multiclass:
            pct_change = close.pct_change(periods=n).shift(-n)
            y = pd.Series(np.where(pct_change > 0.005, 2,
                          np.where(pct_change < -0.005, 0, 1)),
                          index=close.index)
        else:
            y = (close.shift(-n) > close).astype(int)
        return y

    def train_on_data(self, data, model_type=None):
        print("[AIPredictor DEBUG] a treinar...")
        df = self.build_features(data)
        print("[AIPredictor DEBUG] shape das features após build_features:", df.shape)
        target = self._make_targets(data['Close'])
        future_close = data['Close'].shift(-self.n_ahead)

        # Alinha, remove NaNs, normaliza features
        X, y = df.align(target, join='inner', axis=0)
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        X = X.dropna()
        y = y.loc[X.index]
        future_close = future_close.loc[X.index].dropna()
        X = X.loc[future_close.index]
        y = y.loc[future_close.index]
        print("[AIPredictor DEBUG] Após dropna: X.shape=", X.shape, "y.shape=", y.shape)
        if y.isnull().any():
            print("[AIPredictor DEBUG] Ainda há NaN em y:", y[y.isnull()])
        if len(X) < 10:
            raise Exception("Dados insuficientes para treinar IA.")

        # Normalização
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        self.feature_names = X.columns.tolist()
        if model_type:
            self.model_type = model_type
        if self.model_type == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'mlp':
            model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000, random_state=42)
        else:
            model = LogisticRegression(max_iter=1000, random_state=42)

        # Validação cruzada (cross-validation)
        cv_scores = cross_val_score(model, X_scaled, y, cv=5)
        self.last_cv_score = f"CROSS-VALIDATION SCORE (Média 5 folds): {np.mean(cv_scores):.2%}"
        print("[AIPredictor DEBUG]", self.last_cv_score)

        # Validação de sobreajuste (train/test split)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)
        model.fit(X_train, y_train)
        score_train = model.score(X_train, y_train)
        score_test = model.score(X_test, y_test)
        if score_train - score_test > 0.2:
            self.last_overfit_warning = f"⚠️ Sinais de sobreajuste: Treino={score_train:.2f}, Teste={score_test:.2f}"
        else:
            self.last_overfit_warning = f"Treino={score_train:.2f}, Teste={score_test:.2f} (OK)"
        print("[AIPredictor DEBUG]", self.last_overfit_warning)
        self.model = model
        self.features = self.feature_names

        # Feature importance ou coeficientes
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
            self.last_feature_importance = sorted(zip(self.feature_names, imp), key=lambda x: abs(x[1]), reverse=True)
        elif hasattr(model, "coef_"):
            imp = model.coef_[0]
            self.last_feature_importance = sorted(zip(self.feature_names, imp), key=lambda x: abs(x[1]), reverse=True)
        else:
            self.last_feature_importance = None

        # Regressor para preço futuro (usa X_train apenas para consistência)
        reg = LinearRegression()
        reg.fit(X_train, future_close.iloc[:len(X_train)])
        self.reg_model = reg

    def predict_direction(self, data):
        if self.model is None or self.features is None or self.scaler is None:
            return None
        df = self.build_features(data).dropna()
        X = df[self.features]
        if X.empty:
            return None
        X_last = self.scaler.transform(X.iloc[[-1]])
        pred = self.model.predict(X_last)[0]
        return int(pred)

    def predict_proba(self, data):
        if self.model is None or self.features is None or self.scaler is None:
            return None
        df = self.build_features(data).dropna()
        X = df[self.features]
        if X.empty:
            return None
        X_last = self.scaler.transform(X.iloc[[-1]])
        try:
            proba = self.model.predict_proba(X_last)[0]
        except Exception:
            proba = None
        return proba

    def predict_price(self, data):
        if self.reg_model is None or self.features is None or self.scaler is None:
            return None
        df = self.build_features(data).dropna()
        X = df[self.features]
        if X.empty:
            return None
        X_last = self.scaler.transform(X.iloc[[-1]])
        price_pred = self.reg_model.predict(X_last)[0]
        return price_pred

    def get_last_features(self, data):
        df = self.build_features(data).dropna()
        X = df[self.features]
        if X.empty:
            return {}
        X_last = X.iloc[[-1]]
        return X_last.to_dict('records')[0]

    def save_model(self, path="modelo_ia.pkl"):
        joblib.dump((self.model, self.reg_model, self.scaler, self.features, self.n_ahead, self.model_type, self.multiclass), path)

    def load_model(self, path="modelo_ia.pkl"):
        (self.model, self.reg_model, self.scaler, self.features,
         self.n_ahead, self.model_type, self.multiclass) = joblib.load(path)

    # Métodos utilitários para mostrar scores e importâncias na interface
    def get_last_cv_score(self):
        return self.last_cv_score or ""

    def get_last_overfit_warning(self):
        return self.last_overfit_warning or ""

    def get_last_feature_importance(self, top_n=8):
        if not self.last_feature_importance:
            return "N/A"
        txt = "<br>".join([f"{name}: {imp:.3f}" for name, imp in self.last_feature_importance[:top_n]])
        return txt
