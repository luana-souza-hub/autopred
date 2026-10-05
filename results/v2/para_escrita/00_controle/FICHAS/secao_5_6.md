# Ficha: 5.6 Car Parts (exploratória)

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| BAS-03 | Car Parts: séries, comprimento e período | 2.674 séries; 51 obs.; jan/1998 a mar/2002 |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv; data/raw (.tsf) |
| BAS-04 | Car Parts: zeros | mediana 80,4%; 100,0% das séries com algum zero |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| BAS-06 | Força sazonal F_S (mediana) | M4 0,506; Car Parts 0,585; Tourism 0,838; Tourism 366 0,859 |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| BAS-07 | Séries com F_S acima do p95 do piso de ruído | M4 56,5%; Car Parts 18,0%; Tourism 98,5%; Tourism 366 99,2% |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| BAS-11 | Car Parts na explicabilidade | 30 séries entre 2.505 elegíveis (169 descartadas por alvo constante); treino da última janela com 33 obs.; 21 linhas de treino; 22 pontos |  | results/v2/cp/xai/*.meta.json |
| BAS-12 | Car Parts sem cenários adversos | decisão 3A: ruído e outliers proporcionais a σ sem sentido físico com ~80% de zeros |  | results/v2/HANDOFF.md |
| MOD-01 | Random Forest | 100 árvores; min_samples_leaf = 2; random_state = 42; demais no padrão do scikit-learn |  | results/v2/hiperparametros_v2.csv |
| DEF-11 | Métrica principal por base | M4: MAPE (MASE ao lado); Car Parts: MAE (MASE ao lado); Tourism: MASE na H4, MAPE e MAE em paralelo; entre bases: MAE ou MASE |  | results/v2/HANDOFF.md |
| CPL-01 | Car Parts: Persistência | MAE 0,333; posto 1º; sem fallback 0,333; MASE 0,568 (1º) | MAE [0,33; 0,39]; posto [1; 1] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-02 | Car Parts: SVR | MAE 0,459; posto 2º; sem fallback 0,507 (fallback 13,1%); MASE 0,682 (2º) | MAE [0,43; 0,49]; posto [2; 2] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-03 | Car Parts: Random Forest | MAE 0,500; posto 3º; sem fallback 0,500; MASE 0,785 (4º) | MAE [0,47; 0,53]; posto [3; 4] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-04 | Car Parts: SES | MAE 0,536; posto 4º; sem fallback 0,536; MASE 0,769 (3º) | MAE [0,51; 0,56]; posto [4; 6] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-05 | Car Parts: Gradient Boosting | MAE 0,556; posto 5º; sem fallback 0,556; MASE 0,874 (7º) | MAE [0,52; 0,58]; posto [5; 6] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-06 | Car Parts: Sazonal ingênuo | MAE 0,556; posto 5º; sem fallback 0,556; MASE 0,867 (6º) | MAE [0,50; 0,56]; posto [3; 6] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-07 | Car Parts: Média histórica | MAE 0,576; posto 7º; sem fallback 0,576; MASE 0,788 (5º) | MAE [0,56; 0,60]; posto [7; 8] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-08 | Car Parts: SARIMA heurístico | MAE 0,611; posto 8º; sem fallback 0,826 (fallback 67,1%); MASE 0,958 (8º) | MAE [0,56; 0,61]; posto [7; 8] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-09 | Car Parts: SARIMA automático | MAE 0,650; posto 9º; sem fallback 0,692 (fallback 6,0%); 0,630 sem explosivas; MASE 1,010 (9º) | MAE [0,61; 0,68]; posto [9; 9] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-10 | Car Parts: Holt-Winters | MAE 0,741; posto 10º; sem fallback 0,741; MASE 1,351 (10º) | MAE [0,72; 0,77]; posto [10; 10] | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-11 | Car Parts: cobertura do MAPE | indefinido em 15,8% das janelas; cobertura 84,2% das janelas e 89,0% das séries |  | results/v2/cp/limpo/tabela7_v2_cp.csv |
| CPL-12 | Car Parts: R² | indefinido em 15,8% das janelas; medianas de −0,13 a −1,18 |  | results/v2/cp/limpo/tabela7_v2_cp.csv |
| CPL-13 | Car Parts: faixa extremo de zeros (MAE) | Persistência 0,111; Holt-Winters 0,431 |  | results/v2/fase5/tabela7_antigo_v2_cp.csv |
| CPL-14 | Fig. 26: exemplos do RF | T2566: MAE 0,93, MAPE 36,8%; T2419: MAE 2,28, MAPE 70,8% |  | results/v2/fase5/figuras/fig26_v2_cp_exemplos_rf_dados.csv |
| CPX-01 | Car Parts: concordância SHAP × LIME (exploratória) | Pearson RF 0,911, GB 0,708, SVR 0,791; Spearman por série 0,842 / 0,792 / 0,658 | Spearman RF [0,783; 0,890]; GB [0,737; 0,843]; SVR [0,530; 0,764] | results/v2/cp/xai/xai_cp_concordancia_shap_lime.csv |
| CPX-02 | Car Parts: estabilidade SHAP (exploratória) | RF 0,173; GB 0,204; SVR 0,469 (29 séries no SVR); Jaccard top-3 de 0,220 a 0,376 | RF [0,061; 0,271]; GB [0,104; 0,296]; SVR [0,377; 0,558] | results/v2/cp/xai/xai_cp_estabilidade.csv |
| CPX-03 | Car Parts: lag_12 | posição global 7º a 9º; no top-3 de 13,3% a 23,3% das séries |  | results/v2/cp/xai/xai_cp_posicao_lag12.csv |
| CPX-04 | Car Parts: roll_mean_6 sob ruído | perde: RF SHAP −0,100; GB SHAP −0,060 | RF [−0,170; −0,041]; GB [−0,102; −0,019] | results/v2/cp/xai/xai_cp_redistribuicao_ruido.csv |
| FBK-02 | Fallback do SARIMA heurístico, linha de base | M4 33,3% (200 de 600); Car Parts 67,1% (5.380 de 8.022); Tourism 33,3% (200 de 600); Tourism 366 33,3% (366 de 1.098) |  | results/v2/sarima_limites_fallback_v2.csv; tabela3_v2_m4.csv; tabela7_v2_cp.csv |
| FBK-04 | Fallback do SARIMA automático, linha de base | M4 0%; Car Parts 6,0% (485 janelas, treino constante); Tourism 0%; Tourism 366 0% |  | results/v2/sarima_limites_fallback_v2.csv |
| FBK-05 | Fallback do SVR | M4 limpo 0,5%; M4 cenários 0% a 0,8%; Car Parts 13,1%; Tourism 0% |  | tabela3_v2_m4.csv; m4_adversos_status.csv; tabela7_v2_cp.csv; tabela_desempenho_v2_tm.csv |
| FBK-06 | Janelas explosivas do SARIMA automático, linha de base | M4 14; Car Parts 246; Tourism 8; Tourism 366 17 |  | results/v2/sarima_limites_fallback_v2.csv |
| FBK-07 | Janelas explosivas do SARIMA heurístico | linha de base: M4 1; Car Parts 60; Tourism 0; Tourism 366 0. Cenários oficiais do M4: 4 (ausências 10% 1, outliers 2% 1, outliers 5% 2); Tourism e Tourism 366: 1 cada (truncamento 25%) |  | results/v2/sarima_limites_fallback_v2.csv; m4_adversos_status.csv |
| CRZ-01 | Spearman dos rankings, MASE | M4 × Car Parts −0,200; Tourism × M4 +0,188; Tourism × Car Parts −0,770 | M4 × CP [−0,685; +0,248]; TM × M4 [−0,176; +0,636]; TM × CP [−0,867; −0,661] | notebooks/11a_desempenho_tourism_v2.ipynb (saída); results/v2/fase5/replicacao_h4_pares.csv |
| CRZ-02 | Spearman dos rankings, MAE | M4 × Car Parts +0,067; Tourism × M4 −0,091; Tourism × Car Parts −0,602 | M4 × CP [−0,588; +0,273]; TM × M4 [−0,297; +0,624]; TM × CP [−0,673; −0,248] | notebooks/11a_desempenho_tourism_v2.ipynb (saída); results/v2/fase5/replicacao_h4_pares.csv |
| CRZ-04 | Spearman M4 (MAPE) × Car Parts (MAE), referência com métricas diferentes | −0,480 | [−0,661; −0,146] | notebooks/10_car_parts_pipeline_v2.ipynb (saída) |
| CRZ-05 | Spearman M4 × Car Parts sem fallback de SVR e SARIMA | MAE +0,310; MASE −0,188 | MAE [−0,297; +0,455]; MASE [−0,625; +0,248] | notebooks/10_car_parts_pipeline_v2.ipynb (saída) |
| CRZ-06 | Queda do Holt-Winters M4 → Car Parts (mesma métrica) | MAE 3º → 10º; MASE 2º → 10º; ICs separados com e sem fallback | MAE [1; 7] → [10; 10]; MASE [1; 6] → [10; 10] | results/v2/cp/limpo/reversao_hierarquia_m4_cp_mesma_metrica.csv |
| CRZ-07 | Ordem das famílias por MASE nas três bases | M4: ML > Suav. > Baseline > ARIMA; Car Parts: Baseline > ML > Suav. > ARIMA; Tourism: Suav. > ARIMA > Baseline > ML. IC separado: Tourism × Car Parts em ARIMA e ML; Tourism × M4 nenhuma; M4 × Car Parts nenhuma (no MAE, Baseline) |  | results/v2/fase5/replicacao_h4_pares.csv |
| CUS-02 | Tempo de parede da linha de base (10 modelos) | M4 44,2 min (42,1 do SARIMA automático); Car Parts 166,1 min (158,0); Tourism 34,2 min (33,2) |  | results/v2/fase5/tabela6_antigo_v2.csv |
| HIP-07 | Car Parts (exploratória, critérios a posteriori) | H1 4 de 6; H2 6 de 10; H3 0 de 6 |  | results/v2/fase5/replicacao_H1_H4_resumo.csv |

**Arquivos**

- `fase5/tabela7_antigo_v2_cp.csv`
- `cp/limpo/tabela7_v2_cp.csv`
- `cp/limpo/cp_limpo_taxas_status.csv`
- `cp/limpo/cp_limpo_fallback_por_janela.csv`
- `cp/limpo/reversao_hierarquia_m4_cp_mesma_metrica.csv`
- `cp/limpo/reversao_familias_m4_cp_mesma_metrica.csv`
- `cp/xai/xai_cp_concordancia_shap_lime.csv`
- `cp/xai/xai_cp_estabilidade.csv`
- `cp/xai/xai_cp_posicao_lag12.csv`
- `cp/xai/xai_cp_redistribuicao_ruido.csv`
- `cp/xai/xai_cp_top10_global.csv`
- `fase5/figuras/fig23_v2_cp_mape.png`
- `fase5/figuras/fig23_v2_cp_mape_dados.csv`
- `fase5/figuras/fig24_v2_cp_mae.png`
- `fase5/figuras/fig24_v2_cp_mae_dados.csv`
- `fase5/figuras/fig25_v2_cp_4_metricas.png`
- `fase5/figuras/fig25_v2_cp_4_metricas_dados.csv`
- `fase5/figuras/fig26_v2_cp_exemplos_rf.png`
- `fase5/figuras/fig26_v2_cp_exemplos_rf_dados.csv`

**Dados ainda sem tabela:** L08, L04 (ver `00_controle/LACUNAS.md`)
