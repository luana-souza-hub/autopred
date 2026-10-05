# Ficha: 4.6 Análise de explicabilidade

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| PRT-02 | Semente do sorteio das 30 séries de explicabilidade | 43 (42 + 1) |  | scripts/run_xai_v2.py |
| PRT-10 | Características dos modelos de ML | 15: lag_1 a lag_12, roll_mean_3, roll_std_3, roll_mean_6; sem lag_season (= lag_12) e sem lag_24 no oficial |  | src/pipeline_v2.py (feature_names) |
| PRT-12 | Bootstrap dos IC | 2.000 reamostragens; semente 42; percentis 2,5 e 97,5; pareado por série; só séries com valor em todos os modelos |  | src/pipeline_v2.py (bootstrap_delta, bootstrap_ranks) |
| PRT-13 | Pontos explicados por série (XAI) | até 29 linhas de treino mais recentes + ponto de previsão (30); Car Parts 22 |  | src/pipeline_v2.py (pontos_explicados) |
| PRT-14 | Amostras do LIME | 200 (oficial); 1.000 e 5.000 só na sensibilidade |  | src/pipeline_v2.py; scripts/run_xai_v2.py |
| PRT-15 | SHAP do SVR | KernelExplainer; fundo de 50 linhas; nsamples = 2p + 2048 (p = 15); semente fixa; escala original |  | src/pipeline_v2.py (explicar_pontos) |
| PRT-16 | Ruído da análise de estabilidade (XAI) | σ = 30% do desvio padrão do treino, só no treino |  | src/pipeline_v2.py (treino_xai) |
| PRT-21 | Versões das bibliotecas | Python 3.11.0; numpy 2.4.4; pandas 3.0.2; scikit-learn 1.8.0; statsmodels 0.14.6; shap 0.51.0 |  | results/v2/*/*.meta.json |
| DEF-12 | Concordância SHAP × LIME | Pearson entre importâncias globais (cada vetor ÷ máximo); Spearman médio por série; condição limpa |  | results/v2/PREVISOES_tourism.md |
| DEF-13 | Estabilidade das explicações | Spearman por série entre limpo e ruído 30%; Jaccard top-3 e top-5; média entre séries |  | src/pipeline_v2.py (estabilidade) |
| DEF-14 | Normalização das importâncias | soma 1 em cada série antes da média entre séries |  | src/pipeline_v2.py (normalizar_por_serie) |
| BAS-11 | Car Parts na explicabilidade | 30 séries entre 2.505 elegíveis (169 descartadas por alvo constante); treino da última janela com 33 obs.; 21 linhas de treino; 22 pontos |  | results/v2/cp/xai/*.meta.json |

**Arquivos**

- `hiperparametros_v2.csv`
- `PREVISOES_tourism.md`
