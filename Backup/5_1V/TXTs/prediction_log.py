import pandas as pd
import os

class PredictionLogger:
    def __init__(self, filename="prediction_log.csv"):
        self.filename = filename
        if os.path.isfile(self.filename):
            self.df = pd.read_csv(self.filename, parse_dates=["Data"])
        else:
            self.df = pd.DataFrame(columns=[
                "Data", "Ticker", "Modelo", "N_ahead", "MultiClass",
                "Direcao", "Prob_Queda", "Prob_Neutro", "Prob_Subida", "Preco_Real", "Preco_Previsto"
            ])

    def log(self, data, ticker, modelo, n_ahead, multiclass, direcao, proba, preco_real, preco_prev):
        new_entry = {
            "Data": data,
            "Ticker": ticker,
            "Modelo": modelo,
            "N_ahead": n_ahead,
            "MultiClass": multiclass,
            "Direcao": direcao,
            "Prob_Queda": proba[0] if len(proba) > 0 else None,
            "Prob_Neutro": proba[1] if len(proba) > 2 else None,
            "Prob_Subida": proba[-1] if len(proba) > 1 else None,
            "Preco_Real": preco_real,
            "Preco_Previsto": preco_prev
        }
        self.df = pd.concat([self.df, pd.DataFrame([new_entry])], ignore_index=True)
        self.df.to_csv(self.filename, index=False)

    def get_log(self, ticker=None):
        if ticker:
            return self.df[self.df["Ticker"] == ticker].copy()
        return self.df.copy()

