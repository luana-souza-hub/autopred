# Ficha: 5.2.4 Modelos de aprendizado de máquina e importância nativa

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| M4L-03 | M4 limpo: Gradient Boosting | MAPE 9,06%; posto MAPE 3º; MAE 290,35; R² −1,09; MASE 0,900; posto MASE 1º [1; 5] | MAPE [7,72; 10,91]; posto [1; 5] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4L-07 | M4 limpo: Random Forest | MAPE 10,13%; posto MAPE 7º; MAE 323,45; R² −1,05; MASE 0,970; posto MASE 5º [2; 7] | MAPE [7,89; 11,80]; posto [3; 7] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4L-08 | M4 limpo: SVR | MAPE 11,52%; posto MAPE 8º; MAE 415,89; R² −1,68; MASE 1,097; posto MASE 9º [7; 9] | MAPE [9,32; 14,74]; posto [7; 9] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4L-15 | M4: diferença GB × RF | 1,07 p.p. (MAPE) |  | results/v2/m4/limpo/tabela3_v2_m4.csv |
| M4L-12 | M4: famílias por MASE | ML 1º; Suavização exponencial 2º; Baseline 3º; ARIMA 4º | ML [1; 4]; Suav. [1; 3]; Baseline [1; 4]; ARIMA [2; 4] | results/v2/tm/limpo/familias_tres_bases_mase_tm.csv |
| M4L-18 | Fig. 12: importância nativa em M19247 | GB top-3 lag_1, lag_7, roll_mean_3; RF top-3 lag_1, roll_mean_3, lag_7; lag_12 fora do top-10 |  | results/v2/fase5/figuras/fig12_v2_m4_importancia_nativa_M19247_dados.csv |
| PRT-10 | Características dos modelos de ML | 15: lag_1 a lag_12, roll_mean_3, roll_std_3, roll_mean_6; sem lag_season (= lag_12) e sem lag_24 no oficial |  | src/pipeline_v2.py (feature_names) |
| MOD-01 | Random Forest | 100 árvores; min_samples_leaf = 2; random_state = 42; demais no padrão do scikit-learn |  | results/v2/hiperparametros_v2.csv |
| MOD-02 | Gradient Boosting | 100 estimadores; learning_rate = 0,1; max_depth = 3; subsample = 0,8; random_state = 42 |  | results/v2/hiperparametros_v2.csv |
| MOD-03 | SVR | kernel RBF; C = 1,0; epsilon = 0,1; gamma = scale; X e y padronizados só com o treino |  | results/v2/hiperparametros_v2.csv |

**Arquivos**

- `fase5/tabela3_antigo_v2_m4.csv`
- `fase5/figuras/fig12_v2_m4_importancia_nativa_M19247.png`
- `fase5/figuras/fig12_v2_m4_importancia_nativa_M19247_dados.csv`
