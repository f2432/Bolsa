# Instruções do projeto Bolsa

Antes de alterar código, lê `README.md` e `docs/metodo-de-trabalho.md`.

Esta aplicação apoia análise e aprendizagem. Não executa ordens e não transforma scores, previsões, indicadores técnicos ou backtests em recomendações automáticas de investimento.

Ao trabalhar com dados financeiros:

- distingue dados observados, cálculo, interpretação e decisão;
- não assumes que `portfolio.csv` representa a carteira atual sem reconciliação com a XTB;
- evita guardar no Git dados pessoais, extratos, posições reais, logs, caches, modelos e previsões geradas;
- indica fonte, instante de recolha, moeda, mercado, timezone e política de ajustamento de preços;
- em séries temporais, usa separação cronológica e ajusta transformações apenas no treino;
- em backtests, evita look-ahead bias, inclui custos, câmbio e slippage e compara com um benchmark;
- acrescenta ou atualiza testes para alterações de lógica financeira, matemática ou dados.

Faz mudanças pequenas e verificáveis. Antes de concluir, executa os testes e documenta limitações que possam alterar a interpretação dos resultados.
