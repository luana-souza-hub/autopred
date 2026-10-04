# Interpretação dos critérios da Fase 4c (registrada antes da execução)

Registrada em 2026-10-03, **antes** de qualquer execução de explicabilidade no Tourism Monthly (`results/v2/tm/xai/` ainda não existe). Complementa, sem alterar, o arquivo `PREVISOES_tourism.md` (commit 1a08fa1), que continua sendo a referência das previsões e dos critérios.

## 1. H1: faixa do Spearman por série

A previsão diz "Spearman por série entre 0,65 e 0,85". A faixa é interpretada como **fechada**: a condição vale para um modelo quando a média, entre as 30 séries, do Spearman entre os vetores SHAP e LIME da mesma série (condição limpa) satisfaz **0,65 ≤ valor ≤ 0,85**.

## 2. Valores acima de 0,85

Um Spearman por série **acima de 0,85** conta como **fora da faixa** pelo critério literal, e portanto aquela condição da H1 não vale. Se isso acontecer, o texto do artigo deve explicar que o resultado indica concordância **maior** entre os métodos do que a prevista, e não menor; a direção do desvio deve ficar explícita junto do status.

## 3. Rótulos

Os rótulos de cada hipótese (**confirmou** quando todas as condições valem; **confirmou parcialmente** na faixa intermediária; **não confirmou** na faixa inferior) seguem **exatamente** as contagens de `PREVISOES_tourism.md`:

- H1: 6 condições; 6 = confirmou, 3 a 5 = parcial, 0 a 2 = não confirmou.
- H2: 10 condições; 10 = confirmou, 5 a 9 = parcial, 0 a 4 = não confirmou.
- H3: 6 condições; 6 = confirmou, 1 a 5 = parcial, 0 = não confirmou.

Esses rótulos **não serão reinterpretados** depois de ver os resultados. Qualquer leitura adicional (por exemplo, a direção de um desvio ou uma análise de sensibilidade) será apresentada à parte e rotulada como análise posterior, sem mudar o status.
