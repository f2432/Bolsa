import os
import pandas as pd

class PortfolioManager:
    def __init__(self, filepath='portfolio.csv'):
        self.filepath = filepath
        self.positions = []
        self.load_from_file()  # Carrega ao iniciar

    def add_position(self, ticker, quantity, buy_price):
        ticker = ticker.upper()
        print(f"[DEBUG] Adding {quantity}x {ticker} @ {buy_price}")  # DEBUG
        position = next((p for p in self.positions if p['ticker'] == ticker), None)
        if position:
            total_cost = position['buy_price'] * position['quantity'] + buy_price * quantity
            total_qty = position['quantity'] + quantity
            if total_qty > 0:
                position['buy_price'] = total_cost / total_qty
            position['quantity'] = total_qty
        else:
            self.positions.append({'ticker': ticker, 'quantity': quantity, 'buy_price': buy_price})
        print(f"[DEBUG] Positions now: {self.positions}")  # DEBUG
        self.save_to_file()

    def remove_position(self, ticker):
        ticker = ticker.upper()
        self.positions = [p for p in self.positions if p['ticker'] != ticker]
        self.save_to_file()

    def save_to_file(self):
        try:
            print(f"[DEBUG] VOU GRAVAR! {self.positions}")
            df = pd.DataFrame(self.positions)
            df.to_csv(self.filepath, index=False)
            print(f"[DEBUG] Portfolio saved to {self.filepath}")  # DEBUG
        except Exception as e:
            print(f"[ERROR] Failed to save portfolio: {e}")

    def load_from_file(self):
        if os.path.isfile(self.filepath):
            try:
                df = pd.read_csv(self.filepath)
                self.positions = df.to_dict(orient='records')
                print(f"[DEBUG] Portfolio loaded from {self.filepath}")  # DEBUG
            except Exception as e:
                print(f"[ERROR] Failed to load portfolio: {e}")
                self.positions = []
        else:
            print(f"[DEBUG] No portfolio file found, starting empty.")  # DEBUG
            self.positions = []

    def calculate_metrics(self, data_provider):
        total_value = 0.0
        total_cost = 0.0
        detailed = []
        for pos in self.positions:
            ticker = pos['ticker']
            qty = pos['quantity']
            buy_price = pos['buy_price']
            current_price = data_provider.get_current_price(ticker)
            if current_price is None:
                current_price = 0.0
            position_value = qty * current_price
            position_cost = qty * buy_price
            profit = position_value - position_cost
            total_value += position_value
            total_cost += position_cost
            detailed.append({
                'ticker': ticker,
                'quantity': qty,
                'buy_price': buy_price,
                'current_price': current_price,
                'value': position_value,
                'profit': profit
            })
        total_profit = total_value - total_cost
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'total_profit': total_profit,
            'positions': detailed
        }
