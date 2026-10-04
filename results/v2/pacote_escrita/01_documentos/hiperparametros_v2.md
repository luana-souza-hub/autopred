# Hiperparâmetros efetivos do protocolo v2

Gerado por `tools/gerar_hiperparametros_v2.py` a partir de `src/pipeline_v2.py` (sha256 `bbe78767…`, o mesmo dos caches v2). Valores que o pipeline deixa no padrão da biblioteca foram lidos dos objetos instanciados (`get_params()`) e das assinaturas das classes, nas versões usadas: scikit-learn 1.8.0, statsmodels 0.14.6, shap 0.51.0, numpy 2.4.4, Python 3.11.0. A coluna Origem diz onde cada valor está definido.

Fallbacks ficam registrados no status da janela (FALLBACK_SNAIVE ou FALLBACK_MEAN) e o motivo no campo `info`; as tabelas de resultados trazem as versões com e sem fallback.

## Persistência (Baseline)

**Hiperparâmetros**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| regra | ŷ(t+k) = último valor do treino, k = 1..H | pipeline_v2.py:211 (model_persistence) |  |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| fallback | nenhum (sempre OK) | pipeline_v2.py:211 (model_persistence) |  |

## Sazonal ingênuo (Baseline)

**Hiperparâmetros**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| periodo_sazonal | 12 | pipeline_v2.py (SEASON) |  |
| regra | ŷ(t+k) = treino[n − 12 + ((k−1) mod 12)] | pipeline_v2.py:219 (_seasonal_naive) | repete o último ciclo de 12 meses do treino |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| treino_menor_que_periodo | n < 12: repete o último valor (persistência), status continua OK | pipeline_v2.py:219 (_seasonal_naive) | não ocorre no protocolo: MIN_TRAIN=24 e o truncamento mantém ao menos MIN_TRAIN |

## Média histórica (Baseline)

**Hiperparâmetros**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| regra | ŷ(t+k) = média do treino inteiro da janela | pipeline_v2.py:215 (model_mean) |  |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| fallback | nenhum (sempre OK) | pipeline_v2.py:215 (model_mean) |  |

## SES (Suavização exponencial)

**Hiperparâmetros**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| implementacao | statsmodels SimpleExpSmoothing | pipeline_v2.py:230 (model_ses) |  |
| initialization_method | estimated | pipeline_v2.py:230 (model_ses) |  |
| smoothing_level (alfa) | otimizado (optimized=True; o statsmodels minimiza a soma dos erros quadráticos) | pipeline_v2.py:230 (model_ses) |  |
| use_brute | True | padrão statsmodels 0.14.6 |  |
| remove_bias | False | padrão statsmodels 0.14.6 |  |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| excecao_no_ajuste | média do treino, status FALLBACK_MEAN | pipeline_v2.py:230 (model_ses) |  |

## Holt-Winters (Suavização exponencial)

**Hiperparâmetros**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| implementacao | statsmodels ExponentialSmoothing | pipeline_v2.py:241 (model_holt_winters) |  |
| trend | add | pipeline_v2.py:241 (model_holt_winters) |  |
| damped_trend | False | padrão statsmodels 0.14.6 |  |
| seasonal_periods | 12 | pipeline_v2.py:241 (model_holt_winters) |  |
| initialization_method | estimated | pipeline_v2.py:241 (model_holt_winters) |  |
| use_boxcox | False | padrão statsmodels 0.14.6 |  |
| alfa, beta, gama | otimizados (optimized=True; o statsmodels minimiza a soma dos erros quadráticos) | pipeline_v2.py:241 (model_holt_winters) |  |

**Regra de seleção**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| seasonal | ajusta 'add' e, se todo o treino > 0, também 'mul'; fica o de menor AIC | pipeline_v2.py:241 (model_holt_winters) | tipo escolhido gravado em info: {"sazonal": "add"\|"mul"}; seleção só com o treino da janela |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| treino_curto | n < 2 x 12 = 24: previsão do sazonal ingênuo, status FALLBACK_SNAIVE | pipeline_v2.py:241 (model_holt_winters) |  |
| nenhum_ajuste_valido | ambos os tipos lançam exceção: previsão do sazonal ingênuo, status FALLBACK_SNAIVE | pipeline_v2.py:241 (model_holt_winters) |  |

## SARIMA heurístico (ARIMA)

**Hiperparâmetros**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| implementacao | statsmodels SARIMAX | pipeline_v2.py:262 (_sarimax_fit) |  |
| order (p,d,q) | (1, 1, 1) | pipeline_v2.py (SARIMA_HEUR_ORDER) |  |
| seasonal_order (P,D,Q,s) | (1, 1, 0, 12) | pipeline_v2.py (SARIMA_HEUR_SEASONAL) | ordem fixa para todas as séries e janelas |
| enforce_stationarity | False | pipeline_v2.py:262 (_sarimax_fit) |  |
| enforce_invertibility | False | pipeline_v2.py:262 (_sarimax_fit) |  |
| trend | None | padrão statsmodels 0.14.6 | sem termo determinístico |
| otimizador (method) | lbfgs | padrão statsmodels 0.14.6 |  |
| maxiter | 50 | pipeline_v2.py (SARIMA_MAXITER) |  |
| limite_tempo_s | 60.0 | pipeline_v2.py (SARIMA_TIME_LIMIT_S) | medido depois do ajuste (não interrompe o ajuste) |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| treino_constante | desvio padrão do treino = 0: previsão do sazonal ingênuo, status FALLBACK_SNAIVE | pipeline_v2.py:282 (model_sarima_heuristic) | info: {"motivo": "treino_constante"} |
| excecao | exceção no ajuste: previsão do sazonal ingênuo, status FALLBACK_SNAIVE | pipeline_v2.py:282 (model_sarima_heuristic) | info: {"motivo": "excecao"} |
| tempo | ajuste > 60 s: previsão do sazonal ingênuo, status FALLBACK_SNAIVE | pipeline_v2.py:282 (model_sarima_heuristic) | info: {"motivo": "tempo"} |
| verossimilhanca_degenerada | nobs − loglikelihood_burn <= nº de parâmetros, ou llf não finito, ou llf == 0: previsão do sazonal ingênuo, status FALLBACK_SNAIVE | pipeline_v2.py:270 (verossimilhanca_degenerada) | info: {"motivo": "verossimilhanca_degenerada"}; típico de janelas de 24 obs com D=1 |

**Sinalização**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| previsao_explosiva | max\|previsão\| > 10 x max\|treino\|: status continua OK, info {"explosiva": true} | pipeline_v2.py:276 (_diagnostico_previsao) | reportado com e sem as explosivas |

## SARIMA automático (ARIMA)

**Hiperparâmetros**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| implementacao | statsmodels SARIMAX | pipeline_v2.py:262 (_sarimax_fit) |  |
| enforce_stationarity | False | pipeline_v2.py:262 (_sarimax_fit) |  |
| enforce_invertibility | False | pipeline_v2.py:262 (_sarimax_fit) |  |
| trend | None | padrão statsmodels 0.14.6 |  |
| otimizador (method) | lbfgs | padrão statsmodels 0.14.6 |  |
| maxiter | 50 | pipeline_v2.py (SARIMA_MAXITER) | em cada candidato e no ajuste final |

**Regra de seleção**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| grade_p | [0, 1, 2] | pipeline_v2.py (SARIMA_AUTO_GRID) |  |
| grade_d | [0, 1] | pipeline_v2.py (SARIMA_AUTO_GRID) |  |
| grade_q | [0, 1, 2] | pipeline_v2.py (SARIMA_AUTO_GRID) |  |
| grade_P | [0, 1] | pipeline_v2.py (SARIMA_AUTO_GRID) |  |
| grade_D | [0, 1] | pipeline_v2.py (SARIMA_AUTO_GRID) |  |
| grade_Q | [0, 1] | pipeline_v2.py (SARIMA_AUTO_GRID) |  |
| s | 12 | pipeline_v2.py (SEASON) |  |
| exclusao | p = q = P = Q = 0 (só diferenças) não é avaliado | pipeline_v2.py:298 (selecionar_ordem_sarima) |  |
| n_candidatos | 140 | pipeline_v2.py:298 (selecionar_ordem_sarima) | 144 combinações da grade menos as 4 excluídas |
| criterio | menor AIC entre os candidatos que convergem e não são degenerados | pipeline_v2.py:298 (selecionar_ordem_sarima) |  |
| dados_da_selecao | só o treino da janela (seleção refeita a cada janela; C4) | pipeline_v2.py:298 (selecionar_ordem_sarima) |  |
| ordem_de_busca | itertools.product na ordem p, d, q, P, D, Q | pipeline_v2.py:298 (selecionar_ordem_sarima) |  |
| limite_tempo_s | 300.0 | pipeline_v2.py (SARIMA_AUTO_TIME_LIMIT_S) | por janela; verificado antes de cada candidato; se estourar, usa o melhor até ali e grava busca_completa=false |
| ajuste_final | reajusta a ordem escolhida no mesmo treino e prevê H passos | pipeline_v2.py:322 (model_sarima_auto) |  |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| treino_constante | desvio padrão do treino = 0: previsão do sazonal ingênuo, status FALLBACK_SNAIVE | pipeline_v2.py:322 (model_sarima_auto) | sem campo motivo no info |
| sem_candidato_valido | nenhum candidato convergiu sem degeneração: previsão do sazonal ingênuo, status FALLBACK_SNAIVE | pipeline_v2.py:322 (model_sarima_auto) | info: {"motivo": "sem_candidato_valido"} |
| excecao | exceção no ajuste final: previsão do sazonal ingênuo, status FALLBACK_SNAIVE | pipeline_v2.py:322 (model_sarima_auto) | info: {"motivo": "excecao"} |

**Sinalização**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| previsao_explosiva | max\|previsão\| > 10 x max\|treino\|: status continua OK, info {"explosiva": true} | pipeline_v2.py:276 (_diagnostico_previsao) | reportado com e sem as explosivas |

## RF (ML)

**Hiperparâmetros**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| implementacao | sklearn.ensemble.RandomForestRegressor | pipeline_v2.py:337 (make_estimator) |  |
| bootstrap | True | padrão scikit-learn 1.8.0 |  |
| ccp_alpha | 0.0 | padrão scikit-learn 1.8.0 |  |
| criterion | squared_error | padrão scikit-learn 1.8.0 |  |
| max_depth | None | padrão scikit-learn 1.8.0 |  |
| max_features | 1.0 | padrão scikit-learn 1.8.0 |  |
| max_leaf_nodes | None | padrão scikit-learn 1.8.0 |  |
| max_samples | None | padrão scikit-learn 1.8.0 |  |
| min_impurity_decrease | 0.0 | padrão scikit-learn 1.8.0 |  |
| min_samples_leaf | 2 | pipeline_v2.py (HIPERPARAMETROS) |  |
| min_samples_split | 2 | padrão scikit-learn 1.8.0 |  |
| min_weight_fraction_leaf | 0.0 | padrão scikit-learn 1.8.0 |  |
| monotonic_cst | None | padrão scikit-learn 1.8.0 |  |
| n_estimators | 100 | pipeline_v2.py (HIPERPARAMETROS) |  |
| n_jobs | 1 | pipeline_v2.py (HIPERPARAMETROS) |  |
| oob_score | False | padrão scikit-learn 1.8.0 |  |
| warm_start | False | padrão scikit-learn 1.8.0 |  |

**Entrada**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| features | lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, lag_7, lag_8, lag_9, lag_10, lag_11, lag_12, roll_mean_3, roll_std_3, roll_mean_6 | pipeline_v2.py:185 (_linha_features) | lag_1..lag_12, médias móveis de 3 e 6, desvio padrão (ddof=1) de 3; sem lag_season (C5) |
| historico_minimo | 12 | pipeline_v2.py:181 (historico_minimo) | linhas de treino = n_treino − 12 (12 linhas na 1ª janela, com 24 obs) |
| estrategia | recursiva: cada passo usa as features de histórico + previsões anteriores | pipeline_v2.py:347 (ml_forecast) | mesma função monta treino e próximo passo (C1) |
| escalonamento | nenhum | pipeline_v2.py:347 (ml_forecast) |  |

**Sementes**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| random_state | 42 | pipeline_v2.py (HIPERPARAMETROS) |  |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| treino_insuficiente | menos de 2 linhas de features: média do treino, status FALLBACK_MEAN | pipeline_v2.py:347 (ml_forecast) | alvo constante não aciona fallback (árvores preveem a constante) |

## GB (ML)

**Hiperparâmetros**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| implementacao | sklearn.ensemble.GradientBoostingRegressor | pipeline_v2.py:337 (make_estimator) |  |
| alpha | 0.9 | padrão scikit-learn 1.8.0 |  |
| ccp_alpha | 0.0 | padrão scikit-learn 1.8.0 |  |
| criterion | friedman_mse | padrão scikit-learn 1.8.0 |  |
| init | None | padrão scikit-learn 1.8.0 |  |
| learning_rate | 0.1 | pipeline_v2.py (HIPERPARAMETROS) |  |
| loss | squared_error | padrão scikit-learn 1.8.0 |  |
| max_depth | 3 | pipeline_v2.py (HIPERPARAMETROS) |  |
| max_features | None | padrão scikit-learn 1.8.0 |  |
| max_leaf_nodes | None | padrão scikit-learn 1.8.0 |  |
| min_impurity_decrease | 0.0 | padrão scikit-learn 1.8.0 |  |
| min_samples_leaf | 1 | padrão scikit-learn 1.8.0 |  |
| min_samples_split | 2 | padrão scikit-learn 1.8.0 |  |
| min_weight_fraction_leaf | 0.0 | padrão scikit-learn 1.8.0 |  |
| n_estimators | 100 | pipeline_v2.py (HIPERPARAMETROS) |  |
| n_iter_no_change | None | padrão scikit-learn 1.8.0 |  |
| subsample | 0.8 | pipeline_v2.py (HIPERPARAMETROS) |  |
| tol | 0.0001 | padrão scikit-learn 1.8.0 |  |
| validation_fraction | 0.1 | padrão scikit-learn 1.8.0 |  |
| warm_start | False | padrão scikit-learn 1.8.0 |  |

**Entrada**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| features | lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, lag_7, lag_8, lag_9, lag_10, lag_11, lag_12, roll_mean_3, roll_std_3, roll_mean_6 | pipeline_v2.py:185 (_linha_features) | lag_1..lag_12, médias móveis de 3 e 6, desvio padrão (ddof=1) de 3; sem lag_season (C5) |
| historico_minimo | 12 | pipeline_v2.py:181 (historico_minimo) | linhas de treino = n_treino − 12 (12 linhas na 1ª janela, com 24 obs) |
| estrategia | recursiva: cada passo usa as features de histórico + previsões anteriores | pipeline_v2.py:347 (ml_forecast) | mesma função monta treino e próximo passo (C1) |
| escalonamento | nenhum | pipeline_v2.py:347 (ml_forecast) |  |

**Sementes**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| random_state | 42 | pipeline_v2.py (HIPERPARAMETROS) |  |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| treino_insuficiente | menos de 2 linhas de features: média do treino, status FALLBACK_MEAN | pipeline_v2.py:347 (ml_forecast) | alvo constante não aciona fallback (árvores preveem a constante) |

## SVR (ML)

**Hiperparâmetros**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| implementacao | sklearn.svm.SVR | pipeline_v2.py:337 (make_estimator) |  |
| C | 1.0 | pipeline_v2.py (HIPERPARAMETROS) |  |
| cache_size | 200 | padrão scikit-learn 1.8.0 |  |
| coef0 | 0.0 | padrão scikit-learn 1.8.0 |  |
| degree | 3 | padrão scikit-learn 1.8.0 |  |
| epsilon | 0.1 | pipeline_v2.py (HIPERPARAMETROS) |  |
| gamma | scale | pipeline_v2.py (HIPERPARAMETROS) |  |
| kernel | rbf | pipeline_v2.py (HIPERPARAMETROS) |  |
| max_iter | -1 | padrão scikit-learn 1.8.0 |  |
| shrinking | True | padrão scikit-learn 1.8.0 |  |
| tol | 0.001 | padrão scikit-learn 1.8.0 |  |

**Entrada**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| features | lag_1, lag_2, lag_3, lag_4, lag_5, lag_6, lag_7, lag_8, lag_9, lag_10, lag_11, lag_12, roll_mean_3, roll_std_3, roll_mean_6 | pipeline_v2.py:185 (_linha_features) | lag_1..lag_12, médias móveis de 3 e 6, desvio padrão (ddof=1) de 3; sem lag_season (C5) |
| historico_minimo | 12 | pipeline_v2.py:181 (historico_minimo) | linhas de treino = n_treino − 12 (12 linhas na 1ª janela, com 24 obs) |
| estrategia | recursiva: cada passo usa as features de histórico + previsões anteriores | pipeline_v2.py:347 (ml_forecast) | mesma função monta treino e próximo passo (C1) |
| escalonamento | StandardScaler em X e em y, ajustados só no treino da janela; previsão volta à escala original | pipeline_v2.py:347 (ml_forecast) |  |

**Sementes**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| random_state | não se aplica (SVR é determinístico) | padrão scikit-learn 1.8.0 |  |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| treino_insuficiente_ou_alvo_constante | menos de 2 linhas de features, ou desvio padrão do alvo = 0: média do treino, status FALLBACK_MEAN | pipeline_v2.py:347 (ml_forecast) |  |

## Regras comuns a todos os modelos

**Sementes**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| SEED | 42 | pipeline_v2.py (SEED) |  |
| sorteio_da_amostra | default_rng(42).choice sobre a ordem do arquivo | pipeline_v2.py:128 (sortear_indices) | M4 e Tourism: 200 séries; Car Parts: todas |
| perturbacao_cenarios | default_rng([42, serie_idx, origem, id_cenario]) | pipeline_v2.py:403 (rng_janela) | a mesma perturbação para todos os modelos na mesma série, janela e cenário |

**Fallback**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| sem_janela | n < MIN_TRAIN + H: nenhuma janela, status SEM_JANELA (métricas NaN) | pipeline_v2.py:440 (avaliar_serie) |  |
| previsao_invalida | forma diferente de (H,) ou valor não finito: previsão do sazonal ingênuo, status FALLBACK_SNAIVE | pipeline_v2.py:430 (_chamar) | info: {"motivo": "previsao_invalida"}; vale para qualquer modelo |
| erro | exceção não tratada pelo modelo: status ERRO, métricas NaN | pipeline_v2.py:440 (avaliar_serie) |  |

**Protocolo**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| janelas | 3 origens em linspace(24, n − H, 3); treino = [:o], teste = [o:o+H] | pipeline_v2.py:151 (make_rolling_windows) | H = 18 (M4, Car Parts) e 24 (Tourism), definido em scripts/run_v2.py |
| MIN_TRAIN | 24 | pipeline_v2.py (MIN_TRAIN) |  |

## Explicabilidade (RF, GB e SVR)

**Sementes**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| subconjunto_30_series | default_rng(43).choice | scripts/run_xai_v2.py |  |
| ruido_xai | default_rng([42, serie_idx, n_treino, 100]) | pipeline_v2.py:624 (treino_xai) |  |
| shap_svr | default_rng(42) para o fundo e np.random.seed(42) para o KernelExplainer | pipeline_v2.py:643 (explicar_pontos) |  |
| lime | random_state=42 | pipeline_v2.py:643 (explicar_pontos) |  |

**Configuração**

| Parâmetro | Valor | Origem | Observação |
|---|---|---|---|
| modelos_explicados | RF, GB e SVR com HIPERPARAMETROS e make_estimator (os mesmos da previsão) | pipeline_v2.py:570 (treinar_modelos_xai) |  |
| treino | série[:-H] (última janela); ruído σ = 30% do desvio do treino só no treino na condição ruido30 | pipeline_v2.py:624 (treino_xai) |  |
| minimo_linhas | 10 | pipeline_v2.py (MIN_LINHAS_XAI) | abaixo disso a série não é explicada |
| svr_alvo_constante | SVR não é treinado nem explicado (sem fallback) | pipeline_v2.py:570 (treinar_modelos_xai) |  |
| pontos_explicados | 29 linhas de treino mais recentes + ponto de previsão, iguais para SHAP e LIME | pipeline_v2.py:635 (pontos_explicados) |  |
| shap_arvores | TreeExplainer (exato) | pipeline_v2.py:643 (explicar_pontos) |  |
| shap_svr | KernelExplainer, fundo de 50 linhas sorteadas, nsamples = 2p + 2048, escala original do alvo | pipeline_v2.py:643 (explicar_pontos) |  |
| lime | LimeTabularExplainer modo regressão, num_samples=200, num_features = p, coeficientes por índice (as_map()[1]) | pipeline_v2.py:643 (explicar_pontos) |  |
| lime_discretize_continuous | True | padrão lime |  |
| lime_kernel_width | None | padrão lime | kernel_width None = 0,75 x sqrt(p) |
| lime_feature_selection | auto | padrão lime |  |
| lime_sample_around_instance | False | padrão lime |  |
