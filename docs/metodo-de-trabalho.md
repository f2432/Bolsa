# Método de trabalho

## Objetivo

Usar a aplicação para reduzir o universo de empresas e organizar evidência. A decisão final deve resultar de uma tese verificável, valuation, risco e enquadramento na carteira. O indicador técnico serve sobretudo para escolher o momento e planear a execução.

## Fluxo para estudar uma ação

### 1. Ideia

Regista ticker, empresa, mercado, moeda e razão concreta para a empresa ter entrado no radar. Uma tendência ou notícia pode originar uma ideia, mas não é uma tese.

### 2. Filtro inicial

Confirma disponibilidade como ação real ou ação fracionada na XTB, liquidez, capitalização, moeda, histórico mínimo, setor e país. Exclui empresas que não compreendes ou cujos dados não são suficientemente fiáveis.

### 3. Qualidade do negócio

Analisa produto, clientes, fontes de receita, vantagem competitiva, concorrência, crescimento, margens, retorno sobre capital, free cash flow, dívida, diluição e qualidade da gestão. Usa relatórios e apresentações da própria empresa como fontes principais.

### 4. Valuation

Compara preço com resultados e cash flow normalizados, histórico da empresa e concorrentes. Usa pelo menos um cenário pessimista, base e otimista. Uma boa empresa pode ser uma má compra quando o preço exige um cenário demasiado perfeito.

### 5. Tese e invalidação

Escreve a tese em poucas frases, os catalisadores esperados, os riscos principais e factos observáveis que invalidariam a tese. Define horizonte de revisão, não apenas um preço-alvo.

### 6. Carteira e posição

Antes da ordem, mede exposição já existente ao setor, tema, país, moeda e fatores comuns. Define o valor máximo da posição e a perda que aceitas no cenário pessimista. Uma posição pequena não corrige uma tese fraca, mas limita o custo de aprendizagem.

### 7. Execução na XTB

Confirma que o instrumento é uma ação/ETF real e não um CFD. Regista moeda, custo cambial estimado, tipo de ordem, preço máximo aceitável e razão para comprar agora. Não uses o botão de compra para completar a análise que ficou por fazer.

### 8. Revisão

Revê após resultados, alteração material da tese ou na data definida. Distingue variação do preço de alteração do negócio. Avalia separadamente a qualidade da decisão e o resultado financeiro.

## Papel da aplicação

### Pode apoiar

- recolha normalizada de preços e fundamentais;
- filtros de liquidez, dimensão, crescimento, rentabilidade, dívida e valuation;
- comparação entre empresas;
- cálculo de exposição e concentração da carteira;
- indicadores técnicos e planeamento do ponto de entrada;
- registo da tese, decisão e revisão;
- validação histórica de sinais, com pressupostos explícitos.

### Ainda não deve decidir

- comprar ou vender com base numa probabilidade isolada;
- prever um preço futuro como se fosse um valor justo;
- escolher a melhor estratégia pelo maior Sharpe obtido no mesmo período;
- tratar dados do Yahoo Finance como fonte única para fundamentais;
- assumir que a carteira guardada no repositório corresponde à conta XTB atual.

## Ciclo de desenvolvimento

1. Escolher uma necessidade real de decisão.
2. Escrever o resultado esperado e um teste mínimo.
3. Implementar numa branch curta.
4. Validar com dados conhecidos e um caso adverso.
5. Rever o diff e as implicações financeiras do cálculo.
6. Integrar por Pull Request.
7. Registar no estado do Project apenas mudanças relevantes.

## Próxima versão recomendada

O próximo marco deve ser um diário de decisão ligado a uma watchlist e à carteira reconciliada. Cada candidato deve ter estado `ideia`, `em análise`, `aprovado`, `comprado`, `rejeitado` ou `em revisão`, sem qualquer compra automática.
