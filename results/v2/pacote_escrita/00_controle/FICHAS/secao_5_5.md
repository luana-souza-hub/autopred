# Ficha: 5.5 Síntese comparativa e implicações metodológicas

Fonte única dos números: `00_controle/FATOS_CONGELADOS.csv`. Caminhos relativos a `results/v2/` (cópias em `para_escrita/arquivos/`).

| id | fato | valor | IC 95% | origem |
|---|---|---|---|---|
| LIM-01 | Categorias de robustez (mapa síntese) | sobre a média do Δ% de cada tipo: Alta −2 < Δ% < 2; Moderada 2 ≤ Δ% < 10; Baixa Δ% ≥ 10; Melhora* Δ% ≤ −2 |  | tools/nb_builders/nb08v2.py:226 a 246 |
| LIM-02 | Critérios das recomendações | definidos depois do M4; os das recomendações 5 (restrição operacional) e 6 (explicabilidade) depois do Tourism |  | results/v2/fase5/tabela5_antigo_v2.csv |
| M4L-01 | M4 limpo: Holt-Winters | MAPE 8,36%; posto MAPE 1º; MAE 325,29; R² −1,22; MASE 0,918; posto MASE 2º [1; 6] | MAPE [6,19; 10,25]; posto [1; 3] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4L-04 | M4 limpo: SES | MAPE 9,68%; posto MAPE 4º; MAE 328,89; R² −1,06; MASE 0,938; posto MASE 3º [1; 7] | MAPE [8,29; 11,53]; posto [3; 7] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| M4A-02 | M4: Holt-Winters, ruído 30% | +17,6% (+1,47 p.p.) | [+2,5; +37,7] | results/v2/m4/adversos/m4_adversos_delta_ic95.csv |
| M4A-05 | M4: escore de robustez, ajustados | SES 3,0 < RF 3,6 < SVR 4,1 < GB 8,5 < HW 10,5 < SARIMA heur. 12,9 | sem IC | results/v2/m4/adversos/m4_adversos_hierarquia_robustez.csv |
| M4A-10 | M4: outliers, média dos dois níveis | RF 7,4% e SVR 8,2% (ICs sobrepostos) |  | results/v2/m4/adversos/m4_adversos_hierarquia_robustez.csv; m4/adversos/comparacao_tabela5_recomendacoes.csv |
| M4A-15 | M4: ruído 10% e 30%, todos os modelos (Δ%) | SES −3,8 / +1,1; RF +1,0 / +2,9; SVR −0,4 / +3,7; GB +1,6 / +11,9; HW −1,3 / +17,6; SARIMA heur. +2,6 / +17,8; Persistência +0,1 / +13,3; Sazonal ingênuo +4,5 / +15,8; Média −0,4 / −0,6 | ruído 30%: ver m4_adversos_delta_ic95.csv | results/v2/m4/adversos/m4_adversos_delta_pct.csv |
| TMA-02 | Tourism: escore de robustez, ajustados | SES 1,5 < RF 3,4 < SVR 6,1 < SARIMA heur. 6,2 < GB 6,3 < HW 13,7 | sem IC | results/v2/tm/adversos/tm_adversos_hierarquia_robustez.csv |
| TMA-06 | Tourism: outliers, média dos dois níveis (ML) | RF 4,3% < SVR 8,9% < GB 11,5% |  | results/v2/tm/adversos/tm_adversos_hierarquia_robustez.csv |
| TMA-09 | Spearman do escore de robustez M4 × Tourism | +0,829 (6 ajustados) | sem IC | notebooks/11b_cenarios_adversos_tourism_v2.ipynb (saída) |
| M4X-04 | M4: estabilidade SHAP (média; mediana) | RF 0,561; GB 0,436; SVR 0,616 (medianas 0,584 / 0,405 / 0,630) | RF [0,480; 0,632]; GB [0,372; 0,500]; SVR [0,528; 0,695] | results/v2/m4/xai/xai_m4_estabilidade.csv |
| TMX-03 | Tourism: estabilidade (média) | SHAP RF 0,679, GB 0,625, SVR 0,753; LIME 0,612 / 0,592 / 0,727 | SHAP RF [0,615; 0,739]; GB [0,551; 0,690]; SVR [0,691; 0,810]; LIME RF [0,548; 0,670]; GB [0,519; 0,662]; SVR [0,672; 0,778] | results/v2/tm/xai/xai_tm_estabilidade.csv |
| M4X-08 | M4: Q1, CV entre séries (reanálise, SHAP) | normalizado 1,211 / 1,015 / 0,700; escala original 1,702 / 1,493 / 1,167 (RF/GB/SVR) | normalizado RF [0,955; 1,298]; GB [0,849; 1,069]; SVR [0,621; 0,748] | results/v2/fase5/reanalise_cv_q1_m4.csv |
| FBK-03 | Fallback do SARIMA heurístico nos cenários do M4 | 33,3% em ruído, ausências e outliers; 52,2% no truncamento 50%; 65,8% no truncamento 25% |  | results/v2/m4/adversos/m4_adversos_status.csv |
| CUS-01 | Tempo de ajuste mediano por janela (M4) | SES 0,006 s; SVR 0,012 s; GB 0,069 s; SARIMA heur. 0,146 s; HW 0,306 s; RF 0,334 s; SARIMA automático 16,1 s |  | results/v2/fase5/custo_computacional_v2.csv |
| CUS-02 | Tempo de parede da linha de base (10 modelos) | M4 44,2 min (42,1 do SARIMA automático); Car Parts 166,1 min (158,0); Tourism 34,2 min (33,2) |  | results/v2/fase5/tabela6_antigo_v2.csv |
| CUS-03 | Ressalva dos tempos | dependem da máquina e do paralelismo; n_jobs do SARIMA automático no Car Parts e no Tourism não registrado |  | results/v2/HANDOFF.md |
| M4A-12 | SARIMA automático fora dos cenários | 42 min na linha de base do M4; 5 a 6 h estimadas para os 9 cenários |  | results/v2/logs/m4_limpo_sarima_auto.log; HANDOFF.md |
| M4L-02 | M4 limpo: SARIMA automático | MAPE 8,91%; posto MAPE 2º; MAE 344,72; R² −1,33; MASE 0,991; posto MASE 6º [2; 7] | MAPE [6,86; 11,74]; posto [1; 6] | results/v2/fase5/tabela3_antigo_v2_m4.csv |
| BAS-02 | M4: comprimento (mín. / mediana / máx.) | 66 / 198 / 918 |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |
| BAS-07 | Séries com F_S acima do p95 do piso de ruído | M4 56,5%; Car Parts 18,0%; Tourism 98,5%; Tourism 366 99,2% |  | results/v2/pacote_escrita/02_caracterizacao_bases/caracterizacao_bases.csv |

**Arquivos**

- `fase5/tabela4_antigo_v2.csv`
- `fase5/tabela5_antigo_v2.csv`
- `fase5/custo_computacional_v2.csv`
- `fase5/tabela6_antigo_v2.csv`

**Dados ainda sem tabela:** L11 (ver `00_controle/LACUNAS.md`)
