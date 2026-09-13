# Bolsa

Aplicação desktop em Python e PyQt5 para estudar ações, acompanhar uma carteira experimental, calcular indicadores técnicos, testar estratégias e experimentar modelos de machine learning.

O projeto é uma ferramenta de apoio à análise. As ordens reais são decididas e executadas separadamente na XTB.

## Estado atual

A aplicação inclui:

- dados de mercado através de `yfinance`;
- indicadores técnicos e gráficos;
- estratégias SMA Crossover e RSI + MACD;
- backtests básicos e métricas de retorno, drawdown, Sharpe e win rate;
- carteira guardada em CSV;
- modelos Logistic Regression, Random Forest e MLP;
- registo de previsões e exploração de vários universos de ações.

O código compila, mas os resultados de machine learning e backtest ainda são experimentais. Existem problemas conhecidos de validação temporal, alinhamento do alvo de regressão, custos de transação e organização dos dados. Consulta `docs/metodo-de-trabalho.md` antes de usar resultados na análise de uma compra.

## Instalação

```bash
python -m venv .venv
```

No Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python main.py
```

## Organização

- `data/`: obtenção e preparação de dados
- `indicators/`: indicadores técnicos
- `strategies/`: geração de sinais
- `backtest/`: simulação e métricas
- `ai/`: experiências de machine learning
- `portfolio/`: leitura e cálculo da carteira
- `gui/`: interface PyQt5
- `docs/`: método de análise e decisões

## Prioridades

1. Retirar dados financeiros pessoais, binários, caches e duplicados do repositório e do histórico público.
2. Criar uma baseline de testes para carteira, sinais, backtest e modelos.
3. Corrigir a validação temporal e o alinhamento das previsões.
4. Separar configuração, dados gerados e código.
5. Só depois evoluir o ranking de ações e a análise fundamental.
