
import matplotlib.pyplot as plt

def plot_signals_with_predictions(data, signals, predictions=None):
    '''
    Desenha gráfico com:
    - Preço de fecho
    - Pontos de compra/venda (sinais)
    - Linha com previsão (se existir)
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Preço de Fecho', color='blue')

    # Sinais
    buy_signals = signals[signals == 1].index
    sell_signals = signals[signals == -1].index
    ax.scatter(buy_signals, data.loc[buy_signals, 'Close'], label='Compra', marker='^', color='green')
    ax.scatter(sell_signals, data.loc[sell_signals, 'Close'], label='Venda', marker='v', color='red')

    # Previsões
    if predictions is not None:
        ax.plot(data.index, predictions, label='Previsão AI', linestyle='--', color='orange', alpha=0.7)

    ax.set_title("Sinais de Trading e Previsão AI")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço")
    ax.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
