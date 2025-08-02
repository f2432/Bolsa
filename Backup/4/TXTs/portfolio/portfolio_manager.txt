import csv
import os
from datetime import datetime

class PortfolioManager:
    def __init__(self, filename='portfolio.csv'):
        self.filename = filename
        self.positions = []
        self.load_portfolio()

    def load_portfolio(self):
        self.positions = []
        if not os.path.isfile(self.filename):
            print("[DEBUG] No portfolio file found, starting empty.")
            return
        with open(self.filename, newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('ticker') and row.get('quantity') and row.get('buy_price'):
                    # Suporte retrocompatível: se não houver buy_date, põe None
                    buy_date = row.get('buy_date')
                    if not buy_date or buy_date == "None":
                        buy_date = None
                    self.positions.append({
                        'ticker': row['ticker'],
                        'quantity': int(row['quantity']),
                        'buy_price': float(row['buy_price']),
                        'buy_date': buy_date
                    })
        print("[DEBUG] Portfolio loaded from portfolio.csv")

    def save_portfolio(self):
        with open(self.filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['ticker', 'quantity', 'buy_price', 'buy_date'])
            writer.writeheader()
            for pos in self.positions:
                writer.writerow(pos)
        print("[DEBUG] Portfolio saved to portfolio.csv")

    def add_position(self, ticker, quantity, buy_price, buy_date=None):
        found = False
        for pos in self.positions:
            if pos['ticker'] == ticker and pos.get('buy_date') == buy_date:
                pos['quantity'] += int(quantity)
                pos['buy_price'] = float(buy_price)  # atualiza preço de compra para o novo
                found = True
                break
        if not found:
            self.positions.append({
                'ticker': ticker,
                'quantity': int(quantity),
                'buy_price': float(buy_price),
                'buy_date': buy_date
            })
        self.save_portfolio()

    def calculate_metrics(self, data_provider):
        total_value = 0
        total_cost = 0
        total_profit = 0
        positions_metrics = []
        for pos in self.positions:
            ticker = pos['ticker']
            quantity = pos['quantity']
            buy_price = pos['buy_price']
            buy_date = pos.get('buy_date')
            current_price = data_provider.get_current_price(ticker)
            value = quantity * current_price if current_price else 0
            profit = (current_price - buy_price) * quantity if current_price else 0
            total_value += value
            total_cost += buy_price * quantity
            total_profit += profit
            positions_metrics.append({
                'ticker': ticker,
                'quantity': quantity,
                'buy_price': buy_price,
                'current_price': current_price,
                'value': value,
                'profit': profit,
                'buy_date': buy_date
            })
        return {
            'positions': positions_metrics,
            'total_value': total_value,
            'total_cost': total_cost,
            'total_profit': total_profit
        }
