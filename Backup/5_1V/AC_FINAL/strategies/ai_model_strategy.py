
import pandas as pd
from strategies.base_strategy import BaseStrategy

class AIModelStrategy(BaseStrategy):
    """Strategy that uses a trained classifier to predict direction and act accordingly."""
    def __init__(self, predictor):
        super().__init__(name="AI Strategy")
        self.predictor = predictor

    def generate_signals(self, data):
        # Assume que predictor tem método predict(DataFrame) -> Series com valores: 1 (compra), -1 (venda), 0 (manter)
        if len(data) < 20 or 'Close' not in data.columns:
            return pd.Series(0, index=data.index)  # segurança

        try:
            signals = self.predictor.predict(data)
            signals.index = data.index  # garantir alinhamento
            return signals
        except Exception as e:
            print(f"Erro ao gerar sinais com AI: {e}")
            return pd.Series(0, index=data.index)
